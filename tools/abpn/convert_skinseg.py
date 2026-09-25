#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_skinseg.py -- ModelScope `iic/cv_unet_skin-retouching`（Apache-2.0）的皮肤分割 TF 冻结图 `tf_graph.pb`
-> `models/skin_seg.onnx`（身体皮肤遮罩的语义门控，见 doc/analysis/abpn.md §7）。

需要 tensorflow-cpu、tf2onnx、onnx（onnxruntime 可选，用于自检）：
    pip install tensorflow-cpu tf2onnx onnx onnxruntime
一般通过 `python scripts/fetch_models.py --only skinseg` 调用；也可单独运行：
    python tools/abpn/convert_skinseg.py [--weights-dir models/abpn_weights] [--out models/skin_seg.onnx]
tf_graph.pb 缺失时自动从 modelscope.cn 下载（102 MB，校验 SHA256）。

转换后的约定：输入 `input_image:0` float32 [H, W, 3]，RGB，0..255（图内做 x/127.5-1），长边 800；
输出 `output_png:0` uint8 [H, W]，255 = 皮肤；画面中没有人时输出全 0。
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URL = ('https://www.modelscope.cn/api/v1/models/iic/cv_unet_skin-retouching/repo'
       '?Revision=master&FilePath=tf_graph.pb')
TF_GRAPH_SHA256 = '6fa1931e6f1baef129cbac43865197d501c39cf34b6aac477a381a0596cdda30'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, dst):
    tmp = dst + '.part'
    req = urllib.request.Request(url, headers={'User-Agent': 'portrait-retouch/0.3'})
    done, step = 0, 20 << 20
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, 'wb') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done // step != (done - len(chunk)) // step:
                print(f'[skinseg]   {done >> 20} MB', flush=True)
    os.replace(tmp, dst)


def version(*dists):
    for d in dists:
        try:
            return importlib.metadata.version(d)
        except importlib.metadata.PackageNotFoundError:
            continue
    return '?'


def self_test(path):
    """onnxruntime 自检：输出形状 / dtype 正确，纯灰图（无人）输出全 0。"""
    if importlib.util.find_spec('onnxruntime') is None:
        print('[skinseg] onnxruntime not installed; self-test skipped')
        return True
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    x = np.full((600, 800, 3), 128.0, np.float32)
    y = sess.run(None, {'input_image:0': x})[0]
    ok = y.shape == (600, 800) and y.dtype == np.uint8 and int(y.max()) == 0
    print(f'[skinseg] onnxruntime self-test: output {y.shape} {y.dtype}, max {int(y.max())} -> {"ok" if ok else "FAILED"}')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights-dir', default=os.path.join(ROOT, 'models', 'abpn_weights'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'skin_seg.onnx'))
    ap.add_argument('--opset', type=int, default=17)
    a = ap.parse_args()
    miss = [m for m in ('tensorflow', 'tf2onnx', 'onnx') if importlib.util.find_spec(m) is None]
    if miss:
        print(f'[skinseg] missing Python packages: {", ".join(miss)} '
              f'(pip install tensorflow-cpu tf2onnx onnx onnxruntime) -- interpreter: {sys.executable}')
        sys.exit(2)
    pb = os.path.join(a.weights_dir, 'tf_graph.pb')
    if not os.path.isfile(pb):
        os.makedirs(a.weights_dir, exist_ok=True)
        print('[skinseg] downloading tf_graph.pb (102 MB) from modelscope.cn ...', flush=True)
        download(URL, pb)
    digest = sha256(pb)
    if digest != TF_GRAPH_SHA256:
        print(f'[skinseg] tf_graph.pb SHA256 mismatch ({digest}); delete {pb} and retry')
        sys.exit(1)
    print('[skinseg] tf_graph.pb sha256 ok', flush=True)
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + '.tmp.onnx'
    cmd = [sys.executable, '-m', 'tf2onnx.convert', '--graphdef', pb, '--output', tmp,
           '--inputs', 'input_image:0', '--outputs', 'output_png:0', '--opset', str(a.opset)]
    print('[skinseg] ' + ' '.join(cmd), flush=True)
    if subprocess.run(cmd).returncode != 0 or not os.path.isfile(tmp):
        print('[skinseg] tf2onnx conversion failed')
        sys.exit(1)
    import onnx
    tf_v, conv_v = version('tensorflow-cpu', 'tensorflow', 'tensorflow-intel'), version('tf2onnx')
    m = onnx.load(tmp)
    inp, outp = m.graph.input[0], m.graph.output[0]
    for vi in (inp, outp):
        dims = vi.type.tensor_type.shape.dim
        for i, n in enumerate('HW'):
            if i < len(dims):
                dims[i].dim_param = n
    m.doc_string = ('Skin segmentation graph from ModelScope iic/cv_unet_skin-retouching (tf_graph.pb, Apache-2.0), '
                    f'converted with tf2onnx {conv_v} opset {a.opset}. Input: RGB HWC float32 0..255 (no batch dim), '
                    'long side resized to 800 px. Output: uint8 [H,W] skin mask, 255 = skin.')
    del m.metadata_props[:]
    for k, v in {
        'source': 'https://www.modelscope.cn/models/iic/cv_unet_skin-retouching (tf_graph.pb)',
        'license': 'Apache-2.0',
        'input_layout': 'HWC RGB float32 0..255, no batch dim',
        'input_resize': 'long side = 800 px (modelscope resize_on_long_side)',
        'output': 'uint8 [H,W] skin mask, 255 = skin; all zeros when no person is found',
        'converter': f'tf2onnx {conv_v}, tensorflow {tf_v}, opset {a.opset} (tools/abpn/convert_skinseg.py)',
    }.items():
        m.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(m)
    onnx.save(m, tmp)
    if not self_test(tmp):
        print(f'[skinseg] self-test failed; converted file left at {tmp}')
        sys.exit(1)
    os.replace(tmp, out)
    print(f'[skinseg] saved {out} ({os.path.getsize(out)} bytes, sha256={sha256(out)[:16]}...)')


if __name__ == '__main__':
    main()
