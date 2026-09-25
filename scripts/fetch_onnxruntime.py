#!/usr/bin/env python
"""获取 ONNX Runtime 动态库到 runtime/（`ort` 以 load-dynamic 模式加载）。

    python scripts/fetch_onnxruntime.py [--version 1.30.0] [--from-pip]

两种方式：
1. GitHub Release：microsoft/onnxruntime 的 onnxruntime-<os>-<arch>-<ver>.zip/tgz，解压出动态库；
2. `--from-pip`：从 pip 的 onnxruntime wheel 复制（适合 GitHub Release 不可达的网络环境，
   需要 `pip download onnxruntime` 可用）。

ort 2.0.0-rc.13 绑定 ONNX Runtime 1.28 API（最低 API 17，即 1.17+），1.20 ~ 1.30 均可加载。
运行时也可用环境变量 `ORT_DYLIB_PATH` 指向任意位置的动态库。
"""
import argparse
import glob
import io
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(ROOT, 'runtime')


def lib_names():
    s = platform.system()
    if s == 'Windows':
        return ['onnxruntime.dll', 'onnxruntime_providers_shared.dll']
    if s == 'Darwin':
        return ['libonnxruntime.dylib']
    return ['libonnxruntime.so']


def from_pip(version):
    with tempfile.TemporaryDirectory() as td:
        spec = f'onnxruntime=={version}' if version else 'onnxruntime'
        subprocess.check_call([sys.executable, '-m', 'pip', 'download', '--no-deps', '-d', td, spec])
        whl = glob.glob(os.path.join(td, '*.whl'))[0]
        with zipfile.ZipFile(whl) as z:
            for n in z.namelist():
                base = os.path.basename(n)
                if base in lib_names() or base.startswith('libonnxruntime.so') or base.startswith('libonnxruntime.') and base.endswith('.dylib'):
                    dst = os.path.join(RUNTIME, base)
                    with open(dst, 'wb') as f:
                        f.write(z.read(n))
                    print('saved', dst)


def from_github(version):
    s = platform.system()
    arch = platform.machine().lower()
    if s == 'Windows':
        name = f'onnxruntime-win-{"arm64" if "arm" in arch else "x64"}-{version}.zip'
    elif s == 'Darwin':
        name = f'onnxruntime-osx-{"arm64" if "arm" in arch else "x86_64"}-{version}.tgz'
    else:
        name = f'onnxruntime-linux-{"aarch64" if "aarch" in arch or "arm" in arch else "x64"}-{version}.tgz'
    url = f'https://github.com/microsoft/onnxruntime/releases/download/v{version}/{name}'
    print('downloading', url)
    req = urllib.request.Request(url, headers={'User-Agent': 'portrait-retouch/0.1'})
    data = urllib.request.urlopen(req, timeout=300).read()
    if name.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in z.namelist():
                base = os.path.basename(n)
                if base in lib_names():
                    with open(os.path.join(RUNTIME, base), 'wb') as f:
                        f.write(z.read(n))
                    print('saved', base)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as t:
            for m in t.getmembers():
                base = os.path.basename(m.name)
                if base.startswith('libonnxruntime') and m.isfile():
                    with open(os.path.join(RUNTIME, base), 'wb') as f:
                        shutil.copyfileobj(t.extractfile(m), f)
                    print('saved', base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='1.30.0')
    ap.add_argument('--from-pip', action='store_true')
    a = ap.parse_args()
    os.makedirs(RUNTIME, exist_ok=True)
    if a.from_pip:
        from_pip(a.version)
    else:
        try:
            from_github(a.version)
        except Exception as e:  # noqa: BLE001
            print(f'GitHub download failed ({e}); falling back to pip wheel')
            from_pip(a.version)


if __name__ == '__main__':
    main()
