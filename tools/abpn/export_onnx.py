#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_onnx.py -- export the three ABPN skin-retouching nets to ONNX (opset 17)
and verify them with onnxruntime against PyTorch.

Weights come from ModelScope `iic/cv_unet_skin-retouching` (Apache-2.0):
  pytorch_model.pt      (generator UNet, 53 MB)
  joint_20210926.pth    (detection_net + inpainting_net, 227 MB)
Download URL pattern:
  https://www.modelscope.cn/api/v1/models/iic/cv_unet_skin-retouching/repo?Revision=master&FilePath=<file>
(`scripts/fetch_models.py --only abpn` downloads them and runs this script with --only blemish --out models).

  onnx/abpn_blend_unet.onnx      UNet(3,3)            image[1,3,H,W] -> blend_mg[1,3,H,W]   (H,W multiple of 16, nominal 512)
  onnx/abpn_blemish_detect.onnx  DetectionUNet(3,1)   image[1,3,H,W] -> logits[1,1,H,W]     (H,W multiple of 64, nominal 768)
  onnx/abpn_blemish_inpaint.onnx RetouchingNet(4,3)   image[1,3,H,W], mask[1,1,H,W] -> inpainted[1,3,H,W]
                                                                                            (H,W multiple of 64, nominal 576)

All image inputs are RGB, float32, normalised (x/255 - 0.5) * 2.  The inpaint
`image` input is the already-masked image (image * mask) and `mask` is 1 = keep,
0 = hole, exactly what SkinRetouchingPipeline.retouch_local passes.

Usage: python export_onnx.py [--weights-dir DIR] [--out DIR] [--only blend|detect|inpaint|blemish|all] [--fixed]
  --weights-dir  directory holding pytorch_model.pt / joint_20210926.pth (default: this script's directory)
  --out          output directory for the .onnx files (default: <weights-dir>/onnx)
  --only         which nets to export (default all; `blemish` = detect + inpaint, what the Rust engine needs)
  --fixed        export static shapes only
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
try:
    from abpn import DetectionUNet, RetouchingNet, UNet  # noqa: E402  (tools/abpn as a package)
except ImportError:  # running from example/modelscope-skin-retouching/
    sys.path.insert(0, HERE)
    from abpn_nets import DetectionUNet, RetouchingNet, UNet  # noqa: E402

WEIGHTS = HERE
OUT = os.path.join(HERE, 'onnx')
OPSET = 17
warnings.filterwarnings('ignore')


def load_nets(only='all'):
    generator = inpainting = detection = None
    if only in ('all', 'blend'):
        ck = torch.load(os.path.join(WEIGHTS, 'pytorch_model.pt'), map_location='cpu', weights_only=True)
        generator = UNet(3, 3, init_weights=False)
        generator.load_state_dict(ck['generator'], strict=True)
    if only == 'blend':
        generator.eval()
        return generator, None, None
    ck = torch.load(os.path.join(WEIGHTS, 'joint_20210926.pth'), map_location='cpu', weights_only=True)
    inpainting = RetouchingNet(in_channels=4, out_channels=3, init_weights=False)
    inpainting.load_state_dict(ck['inpainting_net'], strict=True)
    detection = DetectionUNet(n_channels=3, n_classes=1, init_weights=False)
    detection.load_state_dict(ck['detection_net'], strict=True)
    for m in (generator, inpainting, detection):
        if m is not None:
            m.eval()
    return generator, detection, inpainting


def try_export(model, args, path, input_names, output_names, dynamic):
    """Export with the TorchScript exporter (dynamo=False).  Returns 'dynamic' / 'fixed'."""
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {n: {2: 'H', 3: 'W'} for n in input_names + output_names}
    kw = dict(input_names=input_names, output_names=output_names, opset_version=OPSET,
              do_constant_folding=True, dynamo=False)
    if dynamic_axes:
        kw['dynamic_axes'] = dynamic_axes
    t0 = time.perf_counter()
    try:
        torch.onnx.export(model, args, path, **kw)
        print(f'  exported {"dynamic H/W" if dynamic else "fixed"} in {time.perf_counter() - t0:.1f}s')
        return 'dynamic' if dynamic else 'fixed'
    except Exception as e:  # noqa: BLE001
        print(f'  export ({"dynamic" if dynamic else "fixed"}) FAILED: {type(e).__name__}: {str(e)[:400]}')
        if dynamic:
            print('  -> falling back to fixed shapes')
            kw.pop('dynamic_axes', None)
            torch.onnx.export(model, args, path, **kw)
            return 'fixed'
        raise


def describe(path):
    m = onnx.load(path)
    onnx.checker.check_model(m)
    opsets = {o.domain or 'ai.onnx': o.version for o in m.opset_import}

    def fmt(vi):
        t = vi.type.tensor_type
        dims = [(d.dim_param or str(d.dim_value)) for d in t.shape.dim]
        return {'name': vi.name, 'dtype': onnx.TensorProto.DataType.Name(t.elem_type), 'shape': dims}

    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    info = {
        'file': os.path.basename(path),
        'size_bytes': os.path.getsize(path),
        'size_mb': round(os.path.getsize(path) / 2**20, 2),
        'opset': opsets,
        'ir_version': m.ir_version,
        'inputs': [fmt(v) for v in m.graph.input],
        'outputs': [fmt(v) for v in m.graph.output],
        'n_nodes': len(m.graph.node),
        'op_types': dict(sorted(ops.items())),
    }
    print('  ' + json.dumps({k: v for k, v in info.items() if k != 'op_types'}))
    print('  ops:', info['op_types'])
    return info


def verify(path, model, feeds_list):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    in_names = [i.name for i in sess.get_inputs()]
    out_name = sess.get_outputs()[0].name
    results = []
    for feeds in feeds_list:
        shape_desc = {k: list(v.shape) for k, v in feeds.items()}
        try:
            with torch.no_grad():
                ref = model(*[torch.from_numpy(feeds[n]) for n in in_names]).numpy()
            t0 = time.perf_counter()
            out = sess.run([out_name], feeds)[0]
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            out = sess.run([out_name], feeds)[0]
            dt2 = time.perf_counter() - t0
            diff = float(np.abs(out.astype(np.float64) - ref.astype(np.float64)).max())
            r = {'shapes': shape_desc, 'ok': bool(out.shape == ref.shape), 'max_abs_diff': diff,
                 'ort_ms_first': round(dt * 1000, 1), 'ort_ms_second': round(dt2 * 1000, 1)}
        except Exception as e:  # noqa: BLE001
            r = {'shapes': shape_desc, 'ok': False, 'error': f'{type(e).__name__}: {str(e)[:300]}'}
        print('  verify', json.dumps(r))
        results.append(r)
    return results


def real_roi(size):
    """a real face crop (bride) normalised to [-1,1], resized to size x size, NCHW float32."""
    import cv2
    p = os.path.join(WEIGHTS, 'out_abpn', 'face_bride_orig.png')
    if not os.path.exists(p):
        return np.random.uniform(-1, 1, (1, 3, size, size)).astype(np.float32)
    bgr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.resize(bgr[:, :, ::-1], (size, size), interpolation=cv2.INTER_AREA)
    x = (rgb.astype(np.float32) / 255.0 - 0.5) * 2
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def blob_mask(h, w, seed=0):
    rng = np.random.RandomState(seed)
    m = np.ones((h, w), np.float32)
    yy, xx = np.mgrid[:h, :w]
    for _ in range(6):
        cy, cx, r = rng.randint(0, h), rng.randint(0, w), rng.randint(6, 30)
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 0.0
    return m[None, None]


def main():
    global WEIGHTS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixed', action='store_true')
    ap.add_argument('--weights-dir', default=HERE)
    ap.add_argument('--out', default=None)
    ap.add_argument('--only', choices=['all', 'blend', 'detect', 'inpaint', 'blemish'], default='all')
    args = ap.parse_args()
    WEIGHTS = args.weights_dir
    OUT = args.out or os.path.join(WEIGHTS, 'onnx')
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    print('torch', torch.__version__, 'onnx', onnx.__version__, 'onnxruntime', ort.__version__)
    generator, detection, inpainting = load_nets(args.only)
    report = {}
    want = {'all': {'blend', 'detect', 'inpaint'}, 'blemish': {'detect', 'inpaint'}}.get(args.only, {args.only})

    # ------------------------------------------------------------ generator
    if 'blend' in want:
        print('\n[abpn_blend_unet] UNet(3,3) blend-layer generator')
        path = os.path.join(OUT, 'abpn_blend_unet.onnx')
        mode = try_export(generator, (torch.zeros(1, 3, 512, 512),), path, ['image'], ['blend_mg'], not args.fixed)
        info = describe(path)
        feeds = [{'image': real_roi(512)}, {'image': np.random.uniform(-1, 1, (1, 3, 512, 512)).astype(np.float32)}]
        if mode == 'dynamic':
            feeds += [{'image': real_roi(640)},
                      {'image': np.random.uniform(-1, 1, (1, 3, 384, 448)).astype(np.float32)},
                      {'image': np.random.uniform(-1, 1, (1, 3, 600, 600)).astype(np.float32)}]  # not a multiple of 16
        report['abpn_blend_unet'] = {'mode': mode, 'info': info, 'verify': verify(path, generator, feeds)}

    # ------------------------------------------------------------ detection
    if 'detect' in want:
        print('\n[abpn_blemish_detect] DetectionUNet(3,1) blemish segmentation (raw logits)')
        path = os.path.join(OUT, 'abpn_blemish_detect.onnx')
        mode = try_export(detection, (torch.zeros(1, 3, 768, 768),), path, ['image'], ['logits'], not args.fixed)
        info = describe(path)
        feeds = [{'image': real_roi(768)}, {'image': np.random.uniform(-1, 1, (1, 3, 768, 768)).astype(np.float32)}]
        if mode == 'dynamic':
            feeds += [{'image': real_roi(512)}, {'image': np.random.uniform(-1, 1, (1, 3, 1024, 768)).astype(np.float32)}]
        report['abpn_blemish_detect'] = {'mode': mode, 'info': info, 'verify': verify(path, detection, feeds)}

    if 'inpaint' not in want:
        finish(report)
        return

    # ------------------------------------------------------------ inpainting
    print('\n[abpn_blemish_inpaint] RetouchingNet(4,3) gated-conv inpainting (inputs: masked image, mask)')
    path = os.path.join(OUT, 'abpn_blemish_inpaint.onnx')
    S = 576
    mode = try_export(inpainting, (torch.zeros(1, 3, S, S), torch.ones(1, 1, S, S)), path,
                      ['image', 'mask'], ['inpainted'], not args.fixed)
    info = describe(path)

    def inp_feed(h, w, seed):
        img = real_roi(max(h, w))[:, :, :h, :w] if seed == 0 else np.random.uniform(-1, 1, (1, 3, h, w)).astype(np.float32)
        m = blob_mask(h, w, seed)
        return {'image': np.ascontiguousarray(img * m), 'mask': m.astype(np.float32)}
    feeds = [inp_feed(S, S, 0), inp_feed(S, S, 1)]
    if mode == 'dynamic':
        feeds += [inp_feed(640, 640, 2), inp_feed(512, 576, 3)]
    report['abpn_blemish_inpaint'] = {'mode': mode, 'info': info, 'verify': verify(path, inpainting, feeds)}
    finish(report)


def finish(report):
    with open(os.path.join(OUT, 'export_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('\nwrote', os.path.join(OUT, 'export_report.json'))
    for k, v in report.items():
        worst = max((r.get('max_abs_diff', float('inf')) for r in v['verify']), default=None)
        print(f'{k:22s} {v["mode"]:8s} {v["info"]["size_mb"]:8.2f} MB  worst max|diff| = {worst}')


if __name__ == '__main__':
    main()
