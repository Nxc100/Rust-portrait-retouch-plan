#!/usr/bin/env python
"""数值对照脚本（方案 7.2 节）：用 numpy 独立实现磨皮方案 A 的全部公式，与 Rust 输出逐像素对比。

用法：
    python tools/oracle_numpy.py <input_image> <rust_output_image> [--smooth 0.5] [--brightness 1.1]
        [--saturation 1.1] [--no-log] [--tolerance 2]

Rust 侧应以 `retouch apply --no-faces --no-face-restrict`（以及相同的 smooth / brightness / saturation / --no-log，
不加美白 / 形变 / LUT）生成输出。输入图短边应 ≤ 720，避免工作副本缩放引入的重采样差异。

公式来源：BBGPUImageBeautifyFilter.m / GPUImageBilateralFilter.m / GPUImageSobelEdgeDetectionFilter.m /
GPUImageHSBFilter.m（RLUM/GLUM/BLUM = 0.3/0.59/0.11）。
"""
import argparse
import sys

import numpy as np
from PIL import Image

W9 = np.array([0.05, 0.09, 0.12, 0.15, 0.18, 0.15, 0.12, 0.09, 0.05], dtype=np.float32)


def shift(img, dx, dy):
    """边缘钳制的整数平移采样：返回 img[y+dy, x+dx]。"""
    h, w = img.shape[:2]
    ys = np.clip(np.arange(h) + dy, 0, h - 1)
    xs = np.clip(np.arange(w) + dx, 0, w - 1)
    return img[ys][:, xs]


def bilateral_pass(img, spacing, dnf, horizontal):
    c = img
    total = c * 0.18
    wsum = np.full(img.shape[:2], 0.18, dtype=np.float32)
    for i, wg in enumerate(W9):
        if i == 4:
            continue
        off = (i - 4) * spacing
        s = shift(img, off, 0) if horizontal else shift(img, 0, off)
        d = np.sqrt(((s - c) ** 2).sum(-1))
        w = wg * (1.0 - np.minimum(d * dnf, 1.0))
        wsum += w
        total += s * w[..., None]
    return total / wsum[..., None]


def bilateral(img, spacing=4, dnf=4.0):
    return bilateral_pass(bilateral_pass(img, spacing, dnf, True), spacing, dnf, False)


def sobel(img):
    lum = 0.2125 * img[..., 0] + 0.7154 * img[..., 1] + 0.0721 * img[..., 2]
    tl, t, tr = shift(lum, -1, -1), shift(lum, 0, -1), shift(lum, 1, -1)
    l, r = shift(lum, -1, 0), shift(lum, 1, 0)
    bl, b, br = shift(lum, -1, 1), shift(lum, 0, 1), shift(lum, 1, 1)
    hh = -tl - 2 * t - tr + bl + 2 * b + br
    vv = -bl - 2 * l - tl + br + 2 * r + tr
    return np.sqrt(hh * hh + vv * vv)


def combine_bb(origin, bilat, edge, degree, apply_log=True):
    r, g, b = origin[..., 0], origin[..., 1], origin[..., 2]
    mx, mn = origin.max(-1), origin.min(-1)
    skin = (edge < 0.2) & (r > 0.3725) & (g > 0.1568) & (b > 0.0784) & (r > b) \
        & ((mx - mn) > 0.0588) & (np.abs(r - g) > 0.0588)
    smooth = np.where(skin[..., None], bilat + (1 - degree) * (origin - bilat), origin)
    if apply_log:
        smooth = np.log(1 + 0.2 * smooth) / np.log(1.2)
    return smooth


def hsb(img, brightness, saturation):
    lum = 0.3 * img[..., 0] + 0.59 * img[..., 1] + 0.11 * img[..., 2]
    out = (lum[..., None] + (img - lum[..., None]) * saturation) * brightness
    return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('rust_output')
    ap.add_argument('--smooth', type=float, default=0.5)
    ap.add_argument('--brightness', type=float, default=1.1)
    ap.add_argument('--saturation', type=float, default=1.1)
    ap.add_argument('--no-log', action='store_true')
    ap.add_argument('--tolerance', type=float, default=2.0, help='允许的最大误差（/255）')
    ap.add_argument('--save-diff', default=None)
    a = ap.parse_args()

    src = np.asarray(Image.open(a.input).convert('RGB')).astype(np.float32) / 255.0
    rust = np.asarray(Image.open(a.rust_output).convert('RGB')).astype(np.float32)
    if min(src.shape[:2]) > 720:
        print('warning: short side > 720, Rust uses a resized work copy; differences expected', file=sys.stderr)

    bl = bilateral(src)
    ed = sobel(src)
    out = combine_bb(src, bl, ed, a.smooth, not a.no_log)
    out = hsb(out, a.brightness, a.saturation)
    ref = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.float32)
    ref = np.floor(ref)
    diff = np.abs(ref - rust)
    print(f'max |oracle - rust| = {diff.max():.1f}/255, mean = {diff.mean():.4f}, '
          f'pixels > {a.tolerance:.0f}: {(diff > a.tolerance).sum()} / {diff.size}')
    if a.save_diff:
        Image.fromarray(np.clip(diff * 32, 0, 255).astype(np.uint8)).save(a.save_diff)
    if diff.max() > a.tolerance:
        print('MISMATCH', file=sys.stderr)
        sys.exit(1)
    print('OK')


if __name__ == '__main__':
    main()
