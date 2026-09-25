#!/usr/bin/env python
"""下载 / 校验 ONNX 模型到 models/（模型不入库）。

    python scripts/fetch_models.py [--only ultraface|2d106|facemesh|parsing|matting|genderage|abpn|skinseg|skinseg-lite] [--check] [--models-dir DIR]

来源与许可证：
- version-RFB-320.onnx  Ultra-Light-Fast-Generic-Face-Detector-1MB（MIT）
- 2d106det.onnx         insightface buffalo_l.zip（代码 MIT，模型仅限非商用；发布版本禁止携带）
- face_mesh.onnx        PINTO0309/facemesh_onnx_tensorrt release 1.0.0 `face_mesh_Nx3x192x192_post.onnx`
                        （MediaPipe Face Mesh，Apache-2.0）

- abpn_blemish_{detect,inpaint}.onnx  ModelScope iic/cv_unet_skin-retouching（ABPN，Apache-2.0）：
                        权重 joint_20210926.pth 下载到 models/abpn_weights/ 后用 tools/abpn/export_onnx.py 导出（需要 torch、onnx、onnxruntime；
                        只在 `--only abpn` 时执行）

- skin_seg.onnx          `--only skinseg`（推荐）：ModelScope iic/cv_unet_skin-retouching 的皮肤分割 TF 图 tf_graph.pb
                        （Apache-2.0）下载到 models/abpn_weights/ 后由 tools/abpn/convert_skinseg.py 用 tf2onnx 转换；
                        需要用装了 tensorflow-cpu + tf2onnx + onnx 的 Python 运行本脚本
- skin_seg_lite.onnx     `--only skinseg-lite`（备选，无需 TensorFlow）：Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP
                        的 DeepLabV3+（MIT，18 MB，精度较低）；引擎只在 skin_seg.onnx 不存在时使用

网络受限时可手动放置文件；`--check` 只检查文件（有固定 SHA256 的会校验）；`--models-dir` 指定其它目录。
"""
import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')

# SHA256 由首次下载时记录；为空表示不校验
ITEMS = {
    'ultraface': {
        'file': 'version-RFB-320.onnx',
        'url': 'https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/models/onnx/version-RFB-320.onnx',
        'sha256': '',
    },
    '2d106': {
        'file': '2d106det.onnx',
        'url': 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip',
        'zip_member': '2d106det.onnx',
        'sha256': '',
    },
    'facemesh': {
        'file': 'face_mesh.onnx',
        'url': 'https://github.com/PINTO0309/facemesh_onnx_tensorrt/releases/download/1.0.0/face_mesh_Nx3x192x192_post.onnx',
        'sha256': '',
    },
    # 奶油肌预设用到的可选模型
    'parsing': {
        'file': 'face_parsing_resnet18.onnx',
        'url': 'https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx',
        'sha256': '',
    },
    'matting': {
        'file': 'modnet_photographic.onnx',
        'url': 'https://github.com/yakhyo/modnet/releases/download/weights/modnet_photographic.onnx',
        'sha256': '',
    },
    'genderage': {
        'file': 'genderage.onnx',
        'url': 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip',
        'zip_member': 'genderage.onnx',
        'sha256': '',
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'portrait-retouch/0.1'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


ABPN_FILES = ['abpn_blemish_detect.onnx', 'abpn_blemish_inpaint.onnx']
ABPN_WEIGHTS = {
    'joint_20210926.pth': 'https://www.modelscope.cn/api/v1/models/iic/cv_unet_skin-retouching/repo?Revision=master&FilePath=joint_20210926.pth',
}


def fetch_abpn(check_only):
    """下载 ABPN 权重并导出瑕疵分割 / 修复两个 ONNX（tools/abpn/export_onnx.py）。"""
    present = [f for f in ABPN_FILES if os.path.isfile(os.path.join(MODELS, f))]
    if len(present) == len(ABPN_FILES):
        for f in ABPN_FILES:
            print(f'[abpn] {f}: present ({os.path.getsize(os.path.join(MODELS, f))} bytes)')
        return True
    if check_only:
        print('[abpn] MISSING: ' + ', '.join(f for f in ABPN_FILES if f not in present))
        return False
    wdir = os.path.join(MODELS, 'abpn_weights')
    os.makedirs(wdir, exist_ok=True)
    for name, url in ABPN_WEIGHTS.items():
        dst = os.path.join(wdir, name)
        if os.path.isfile(dst):
            continue
        print(f'[abpn] downloading {name} (~228 MB) ...')
        try:
            data = download(url)
        except Exception as e:  # noqa: BLE001
            print(f'[abpn] download failed: {e}\n       please download manually and put it at {dst}')
            return False
        with open(dst, 'wb') as f:
            f.write(data)
    script = os.path.join(ROOT, 'tools', 'abpn', 'export_onnx.py')
    cmd = [sys.executable, script, '--weights-dir', wdir, '--out', MODELS, '--only', 'blemish']
    print('[abpn] exporting: ' + ' '.join(cmd))
    import subprocess
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print('[abpn] export failed (torch / onnx / onnxruntime required: pip install torch onnx onnxruntime)')
        return False
    return all(os.path.isfile(os.path.join(MODELS, f)) for f in ABPN_FILES)


SKINSEG_FILE = 'skin_seg.onnx'
SKINSEG_LITE_FILE = 'skin_seg_lite.onnx'
SKINSEG_LITE_URL = ('https://raw.githubusercontent.com/Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP/main/'
                    '02.model/DeepLabV3Plus%28timm-mobilenetv3_large_100%29_1366_4.71M_0.8606/best_model_simplifier.onnx')
SKINSEG_LITE_SHA256 = 'a029ad09c8ee74e3cab17bd58c79c2e28d0bb45b03a0de8dd3614bebe6ef84b3'
CONVERT_DEPS = ('tensorflow', 'tf2onnx', 'onnx')


def missing_modules(names):
    import importlib.util
    return [n for n in names if importlib.util.find_spec(n) is None]


def fetch_skinseg(check_only):
    """推荐：ModelScope 皮肤分割 TF 图 tf_graph.pb -> tools/abpn/convert_skinseg.py（tf2onnx）-> models/skin_seg.onnx。
    转换需要 tensorflow-cpu、tf2onnx、onnx：用装了它们的 Python 运行本脚本。"""
    dst = os.path.join(MODELS, SKINSEG_FILE)
    if os.path.isfile(dst):
        print(f'[skinseg] {SKINSEG_FILE}: present ({os.path.getsize(dst)} bytes, sha256={sha256(dst)[:16]}...)')
        return True
    if check_only:
        print(f'[skinseg] {SKINSEG_FILE}: MISSING (optional, recommended: `--only skinseg`)')
        return False
    miss = missing_modules(CONVERT_DEPS)
    if miss:
        pip = os.path.join('.venv-convert', 'Scripts' if os.name == 'nt' else 'bin')
        print(f'[skinseg] this Python ({sys.executable}) lacks: {", ".join(miss)}. Set up a conversion environment:')
        print('            python -m venv .venv-convert')
        print(f'            {os.path.join(pip, "pip")} install tensorflow-cpu tf2onnx onnx onnxruntime')
        print(f'            {os.path.join(pip, "python")} scripts/fetch_models.py --only skinseg')
        print('          or download the lightweight model instead: python scripts/fetch_models.py --only skinseg-lite')
        return False
    wdir = os.path.join(MODELS, 'abpn_weights')
    script = os.path.join(ROOT, 'tools', 'abpn', 'convert_skinseg.py')
    cmd = [sys.executable, script, '--weights-dir', wdir, '--out', dst]
    print('[skinseg] converting: ' + ' '.join(cmd), flush=True)
    import subprocess
    r = subprocess.run(cmd)
    if r.returncode != 0 or not os.path.isfile(dst):
        print('[skinseg] conversion failed; `--only skinseg-lite` downloads the lightweight model instead')
        return False
    return True


def fetch_skinseg_lite(check_only):
    """备选：Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP 的 DeepLabV3+（MIT，18 MB）-> models/skin_seg_lite.onnx。
    引擎只在 skin_seg.onnx 不存在时使用它。"""
    dst = os.path.join(MODELS, SKINSEG_LITE_FILE)
    if os.path.isfile(dst):
        digest = sha256(dst)
        if digest != SKINSEG_LITE_SHA256:
            print(f'[skinseg-lite] {SKINSEG_LITE_FILE}: SHA256 mismatch ({digest})')
            return False
        print(f'[skinseg-lite] {SKINSEG_LITE_FILE}: present ({os.path.getsize(dst)} bytes, sha256 ok)')
        return True
    if check_only:
        print(f'[skinseg-lite] {SKINSEG_LITE_FILE}: MISSING (optional: `--only skinseg-lite`)')
        return False
    print(f'[skinseg-lite] downloading {SKINSEG_LITE_URL} ...', flush=True)
    try:
        data = download(SKINSEG_LITE_URL)
    except Exception as e:  # noqa: BLE001
        print(f'[skinseg-lite] download failed: {e}\n       please download manually and put it at {dst}')
        return False
    digest = hashlib.sha256(data).hexdigest()
    if digest != SKINSEG_LITE_SHA256:
        print(f'[skinseg-lite] SHA256 mismatch ({digest}); file not saved')
        return False
    with open(dst, 'wb') as f:
        f.write(data)
    print(f'[skinseg-lite] saved {dst} ({len(data)} bytes, sha256 ok)')
    return True


def main():
    global MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=list(ITEMS) + ['abpn', 'skinseg', 'skinseg-lite'])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--models-dir', default=MODELS, help='target directory (default: <repo>/models)')
    a = ap.parse_args()
    MODELS = os.path.abspath(a.models_dir)
    os.makedirs(MODELS, exist_ok=True)
    ok = True
    if a.only == 'abpn':
        sys.exit(0 if fetch_abpn(a.check) else 1)
    if a.only == 'skinseg':
        sys.exit(0 if fetch_skinseg(a.check) else 1)
    if a.only == 'skinseg-lite':
        sys.exit(0 if fetch_skinseg_lite(a.check) else 1)
    if a.check:
        fetch_abpn(True)
        fetch_skinseg(True)
        fetch_skinseg_lite(True)
    for key, it in ITEMS.items():
        if a.only and key != a.only:
            continue
        dst = os.path.join(MODELS, it['file'])
        if os.path.isfile(dst):
            digest = sha256(dst)
            if it['sha256'] and digest != it['sha256']:
                print(f'[{key}] {it["file"]}: SHA256 mismatch ({digest})')
                ok = False
            else:
                print(f'[{key}] {it["file"]}: present ({os.path.getsize(dst)} bytes, sha256={digest[:16]}...)')
            continue
        if a.check:
            print(f'[{key}] {it["file"]}: MISSING')
            ok = False
            continue
        print(f'[{key}] downloading {it["url"]} ...')
        try:
            data = download(it['url'])
        except Exception as e:  # noqa: BLE001
            print(f'[{key}] download failed: {e}\n       please download manually and put it at {dst}')
            ok = False
            continue
        if 'zip_member' in it:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                member = next(n for n in z.namelist() if n.endswith(it['zip_member']))
                data = z.read(member)
        with open(dst, 'wb') as f:
            f.write(data)
        print(f'[{key}] saved {dst} ({len(data)} bytes, sha256={sha256(dst)[:16]}...)')
    if not a.only and not a.check and not any(
            os.path.isfile(os.path.join(MODELS, f)) for f in (SKINSEG_FILE, SKINSEG_LITE_FILE)):
        print('[hint] no skin-segmentation model: light-colored clothes may be retouched as skin. Run '
              '`--only skinseg` (recommended, needs tensorflow-cpu + tf2onnx) or `--only skinseg-lite`.')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
