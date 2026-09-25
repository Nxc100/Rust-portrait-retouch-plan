#!/usr/bin/env python
"""把美狐曲线滤镜包（filterFile.zip 解压目录）的"纯颜色部分"离线烘焙为 .cube（方案 4.5.3 节）。

    python tools/bake_lut.py <filterFile_dir> <out_dir> [--size 33] [--names cool,romance,...] [--verify photo.jpg]

`<filterFile_dir>` 为解压后包含 `cool/cool/fragment.glsl` 等子目录的目录（`python -m zipfile -e filterFile.zip .`
再逐个解压每款 zip）。脚本用 numpy 逐款直译 fragment.glsl 的颜色运算（GL_LINEAR + CLAMP_TO_EDGE 的曲线采样规则
与 shader 一致），在 N³ 网格上求值输出 `.cube`；空间部分另行导出：

- calm：`calm.cube`（基础曲线）+ `calm_overlay1.cube` / `calm_overlay2.cube` + `calm_mask.png` / `calm_mask1.png`，
  运行时：`--lut calm.cube --masked-lut calm_overlay1.cube:calm_mask.png:invert --masked-lut calm_overlay2.cube:calm_mask1.png:invert`
- blackwhite：`blackwhite.cube` + `blackwhite_vignette.png`（解析暗角烘焙成 1024×1024 灰度）；运行时用
  `--masked-lut blackwhite_dark.cube:blackwhite_vignette.png`
- brooklyn：只烘焙颜色部分，`map.png` 纹理叠加不烘焙。

`--verify photo.jpg` 会用 numpy 直译 shader 计算整图参考结果，与"烘焙 LUT 三线性查表"的结果比较，
报告最大误差（方案 8 节：误差 ≤ 2/255）。

合规：烘焙结果是美狐贴图的衍生物，只可用于个人学习，商用请自制 LUT。
"""
import argparse
import os
import struct
import sys

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- 曲线贴图与 GL 采样
def load_curve_tex(path):
    """返回 float32 [N, 256, 4]（0..1），行 0 为纹理 y=0（顶部）。"""
    if path.endswith('.bin'):
        b = open(path, 'rb').read()
        w, h = struct.unpack('<HH', b[:4])
        arr = np.frombuffer(b[4:4 + w * h * 4], dtype=np.uint8).reshape(h, w, 4)
    else:
        arr = np.asarray(Image.open(path).convert('RGBA'))
    return arr.astype(np.float32) / 255.0


def sample_curve(tex, x, y, ch):
    """texture2D(tex, vec2(x, y)).ch，GL_LINEAR + CLAMP_TO_EDGE，纹理 (256 × N)。"""
    n, w = tex.shape[0], tex.shape[1]
    fx = np.clip(np.asarray(x, dtype=np.float32) * w - 0.5, 0, w - 1)
    x0 = np.floor(fx).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    tx = fx - x0
    fy = np.clip(y * n - 0.5, 0, n - 1)
    y0 = int(np.floor(fy))
    y1 = min(y0 + 1, n - 1)
    ty = fy - y0
    c = tex[..., ch]
    a = c[y0, x0] + (c[y0, x1] - c[y0, x0]) * tx
    b = c[y1, x0] + (c[y1, x1] - c[y1, x0]) * tx
    return a + (b - a) * ty


# ---------------------------------------------------------------- 颜色空间工具（GLSL 直译）
def rgb2hsv(c):
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    e = 1e-10
    h = np.zeros_like(mx)
    m = (mx == r)
    h[m] = ((g - b) / (6.0 * d + e))[m]
    m2 = (mx == g) & ~m
    h[m2] = (1.0 / 3.0 + (b - r) / (6.0 * d + e))[m2]
    m3 = (mx == b) & ~m & ~m2
    h[m3] = (2.0 / 3.0 + (r - g) / (6.0 * d + e))[m3]
    h = np.mod(h, 1.0)
    s = d / (mx + e)
    return np.stack([h, s, mx], -1)


def hsv2rgb(c):
    h, s, v = c[..., 0], c[..., 1], c[..., 2]
    k = np.array([1.0, 2.0 / 3.0, 1.0 / 3.0])
    p = np.abs(np.mod(h[..., None] + k, 1.0) * 6.0 - 3.0)
    return v[..., None] * (1.0 + (np.clip(p - 1.0, 0, 1) - 1.0) * s[..., None])


def screen(a, b):
    return 1.0 - (1.0 - a) * (1.0 - b)


# ---------------------------------------------------------------- 各滤镜的纯颜色部分
def bake_cool(d, c):
    t = load_curve_tex(os.path.join(d, 'curve.bin'))
    r = sample_curve(t, c[..., 0], 0.0, 0)
    g = sample_curve(t, c[..., 1], 0.0, 1)
    b = sample_curve(t, c[..., 2], 0.0, 2)
    r = sample_curve(t, r, 0.0, 3) * 1.25 - 0.12549
    g = sample_curve(t, g, 0.0, 3) * 1.25 - 0.12549
    b = sample_curve(t, b, 0.0, 3) * 1.25 - 0.12549
    tc = np.stack([r, g, b], -1)
    return (c - tc) * 0.549 + tc


def bake_romance(d, c):
    t = load_curve_tex(os.path.join(d, 'curve.bin'))
    tc = screen(c, c)
    r = sample_curve(t, tc[..., 0], 0.0, 0)
    g = sample_curve(t, tc[..., 1], 0.0, 1)
    b = sample_curve(t, tc[..., 2], 0.0, 2)
    gray = (r + g + b) / 3.0
    sat = 1.15
    return np.stack([(1 - sat) * gray + sat * r, (1 - sat) * gray + sat * g, (1 - sat) * gray + sat * b], -1)


def bake_kevin(d, c):
    t = load_curve_tex(os.path.join(d, 'map.png'))
    return np.stack([sample_curve(t, c[..., i], 0.5, i) for i in range(3)], -1)


def bake_blackcat(d, c):
    t = load_curve_tex(os.path.join(d, 'curve.bin'))
    tc = np.stack([sample_curve(t, c[..., i], 0.0, i) for i in range(3)], -1)
    hsv = rgb2hsv(np.clip(tc, 0, 1))
    h, s = hsv[..., 0], hsv[..., 1] * 1.2
    sat_k = 0.3
    boost = np.zeros_like(h)
    boost[(h >= 0.0) & (h < 0.417)] = 1.0
    boost[(h > 0.958) & (h <= 1.0)] = 1.0
    m = (h >= 0.875) & (h <= 0.958)
    boost[m] = (np.abs(h - 0.875) / 0.0833)[m]
    m = (h >= 0.0417) & (h <= 0.125)
    boost[m] = (np.abs(h - 0.125) / 0.0833)[m]
    s = s + s * sat_k * boost
    tc = np.clip(hsv2rgb(np.stack([h, s, hsv[..., 2]], -1)), 0, 1)
    r = sample_curve(t, tc[..., 0], 1.0, 0)
    g = sample_curve(t, tc[..., 1], 1.0, 0)
    b = sample_curve(t, tc[..., 2], 1.0, 0)
    return np.stack([sample_curve(t, r, 1.0, 1), sample_curve(t, g, 1.0, 1), sample_curve(t, b, 1.0, 1)], -1)


def bake_emerald(d, c):
    t = load_curve_tex(os.path.join(d, 'curve.png'))
    tc = np.stack([sample_curve(t, c[..., i], 0.0, i) for i in range(3)], -1)
    hsl = rgb2hsv(np.clip(tc, 0, 1))  # shader 的 RGBtoHSL 实际上是 HSV 公式
    h, s = hsl[..., 0], np.clip(hsl[..., 1] * 1.5, 0, 1)
    dsat, dhue = 0.15, 0.08
    hn, sn = h.copy(), s.copy()
    m = (h >= 0.625) & (h <= 0.708)
    hn[m] = (h - h * dhue)[m]
    sn[m] = (s + s * dsat)[m]
    m = (h >= 0.542) & (h < 0.625)
    k = np.abs(h - 0.542) / 0.0833
    hn[m] = (h + h * dhue * k)[m]
    sn[m] = (s + s * dsat * k)[m]
    m = (h > 0.708) & (h <= 0.792)
    k = np.abs(h - 0.792) / 0.0833
    hn[m] = (h + h * dhue * k)[m]
    sn[m] = (s + s * dsat * k)[m]
    tc = np.clip(hsv2rgb(np.stack([hn, sn, hsl[..., 2]], -1)), 0, 1)
    r = sample_curve(t, tc[..., 0], 1.0, 0)
    g = sample_curve(t, tc[..., 1], 1.0, 0)
    b = sample_curve(t, tc[..., 2], 1.0, 0)
    return np.stack([sample_curve(t, r, 1.0, 1), sample_curve(t, g, 1.0, 1), sample_curve(t, b, 1.0, 1)], -1)


def calm_base(d, c):
    t = load_curve_tex(os.path.join(d, 'curve.bin'))
    lumw = np.array([0.2125, 0.7154, 0.0721])
    l = (c * lumw).sum(-1, keepdims=True)
    tc = l + (c - l) * 0.5
    r = sample_curve(t, tc[..., 0], 0.0, 0)
    r = sample_curve(t, r, 0.5, 0)
    g = sample_curve(t, tc[..., 1], 0.0, 1)
    g = sample_curve(t, g, 0.5, 1)
    b = sample_curve(t, tc[..., 2], 0.0, 2)
    b = sample_curve(t, b, 0.5, 2)
    b = sample_curve(t, b, 0.5, 1)
    return np.stack([r, g, b], -1), t


def bake_calm(d, c):
    base, _ = calm_base(d, c)
    return base


def bake_calm_overlay1(d, c):
    t = load_curve_tex(os.path.join(d, 'curve.bin'))
    return np.stack([sample_curve(t, c[..., i], 1.0, 0) for i in range(3)], -1)


def bake_calm_overlay2(d, c):
    t = load_curve_tex(os.path.join(d, 'curve.bin'))
    return np.stack([sample_curve(t, c[..., i], 1.0, 1) for i in range(3)], -1)


def bake_blackwhite(d, c):
    gray = (c * np.array([0.229, 0.587, 0.114])).sum(-1) * 1.33
    tc = np.stack([gray, gray, gray], -1)
    # calBrightnessContract(0, 16, 128)
    cv = 16.0 / 255.0
    cv = 1.0 / (1.0 - cv) - 1.0
    tc = tc + (tc - 128.0 / 255.0) * cv
    return tc


def bake_blackwhite_dark(d, c):
    """暗角 LUT：与 vignette mask 联用：out = mix(c, c - 0.5·0.5·(dist²/0.5)...) 用 mask 承载空间项，此处 LUT = c − 1。"""
    return np.clip(c - 1.0, 0, 1) * 0 + np.clip(c - 0.5, 0, 1)


def bake_brooklyn(d, c):
    t = load_curve_tex(os.path.join(d, 'curve.png'))
    tc = np.stack([sample_curve(t, c[..., i], 0.0, i) for i in range(3)], -1)
    gray = (tc * np.array([0.2125, 0.7154, 0.0721])).sum(-1, keepdims=True)
    tc = np.clip(0.88 * tc + 0.12 * gray, 0, 1)
    tc = np.clip(0.85 * (tc - 0.5) + 0.5, 0, 1)
    tc = np.clip(tc + 0.03, 0, 1)
    # hueAdjust(-0.0444) in YIQ
    yp = (tc * np.array([0.299, 0.587, 0.114])).sum(-1)
    i = (tc * np.array([0.595716, -0.274453, -0.321263])).sum(-1)
    q = (tc * np.array([0.211456, -0.522591, 0.31135])).sum(-1)
    hue = np.arctan2(q, i) + 0.0444
    chroma = np.sqrt(i * i + q * q)
    i, q = chroma * np.cos(hue), chroma * np.sin(hue)
    tc = np.stack([yp + 0.9563 * i + 0.6210 * q, yp - 0.2721 * i - 0.6474 * q, yp - 1.1070 * i + 1.7046 * q], -1)
    # multiplyBlend(bg=(0.5647,0.1961,0.0157, a=0.14), base=tc(a=1))
    bg = np.array([0.5647, 0.1961, 0.0157])
    a = 0.14
    tc = (1.0 - a) * tc + a * bg * tc  # a_out = 1
    tc = np.clip(tc, 0, 1)
    return np.stack([sample_curve(t, tc[..., i], 0.0, i) for i in range(3)], -1)


BAKERS = {
    'cool': [('cool', bake_cool)],
    'romance': [('romance', bake_romance)],
    'kevin': [('kevin', bake_kevin)],
    'blackcat': [('blackcat', bake_blackcat)],
    'emerald': [('emerald', bake_emerald)],
    'calm': [('calm', bake_calm), ('calm_overlay1', bake_calm_overlay1), ('calm_overlay2', bake_calm_overlay2)],
    'blackwhite': [('blackwhite', bake_blackwhite)],
    'brooklyn': [('brooklyn', bake_brooklyn)],
}


def identity_grid(n):
    b, g, r = np.mgrid[0:n, 0:n, 0:n]
    m = n - 1
    return np.stack([r / m, g / m, b / m], -1).reshape(-1, 3).astype(np.float32)


def write_cube(path, grid, n, title):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        for v in np.clip(grid, 0, 1):
            f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')


def trilinear(lut, n, c):
    m = n - 1
    f = np.clip(c, 0, 1) * m
    i0 = np.floor(f).astype(np.int32)
    i1 = np.minimum(i0 + 1, m)
    t = f - i0
    g = lut.reshape(n, n, n, 3)  # [b, g, r]
    def at(r, gg, b):
        return g[b, gg, r]
    c00 = at(i0[..., 0], i0[..., 1], i0[..., 2]) * (1 - t[..., 0:1]) + at(i1[..., 0], i0[..., 1], i0[..., 2]) * t[..., 0:1]
    c10 = at(i0[..., 0], i1[..., 1], i0[..., 2]) * (1 - t[..., 0:1]) + at(i1[..., 0], i1[..., 1], i0[..., 2]) * t[..., 0:1]
    c01 = at(i0[..., 0], i0[..., 1], i1[..., 2]) * (1 - t[..., 0:1]) + at(i1[..., 0], i0[..., 1], i1[..., 2]) * t[..., 0:1]
    c11 = at(i0[..., 0], i1[..., 1], i1[..., 2]) * (1 - t[..., 0:1]) + at(i1[..., 0], i1[..., 1], i1[..., 2]) * t[..., 0:1]
    c0 = c00 * (1 - t[..., 1:2]) + c10 * t[..., 1:2]
    c1 = c01 * (1 - t[..., 1:2]) + c11 * t[..., 1:2]
    return c0 * (1 - t[..., 2:3]) + c1 * t[..., 2:3]


def find_pack_dir(root, name):
    for cand in [os.path.join(root, name, name), os.path.join(root, name)]:
        if os.path.isfile(os.path.join(cand, 'fragment.glsl')):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('filter_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--size', type=int, default=33)
    ap.add_argument('--names', default=','.join(BAKERS))
    ap.add_argument('--verify', default=None, help='用一张照片验证烘焙误差')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    grid = identity_grid(a.size)
    photo = None
    if a.verify:
        photo = np.asarray(Image.open(a.verify).convert('RGB')).astype(np.float32) / 255.0
    for name in a.names.split(','):
        d = find_pack_dir(a.filter_dir, name)
        if d is None:
            print(f'[{name}] not found under {a.filter_dir}', file=sys.stderr)
            continue
        for out_name, fn in BAKERS[name]:
            lut = np.clip(fn(d, grid), 0, 1)
            path = os.path.join(a.out_dir, f'{out_name}.cube')
            write_cube(path, lut, a.size, out_name)
            msg = f'[{name}] wrote {path}'
            if photo is not None:
                ref = np.clip(fn(d, photo), 0, 1)
                via = trilinear(lut, a.size, photo)
                err = np.abs(ref - via) * 255
                # HSV 色相分段（blackcat / emerald）在分段边界不连续，网格插值在边界附近误差较大；
                # 用 65 网格或改为按色相分段的 1D 曲线可降低误差。
                msg += (f'  LUT vs direct: max {err.max():.2f}/255, mean {err.mean():.3f}/255, '
                        f'p99.9 {np.percentile(err, 99.9):.2f}/255')
            print(msg)
        if name == 'calm':
            for m in ['mask.png', 'mask1.png']:
                Image.open(os.path.join(d, m)).convert('L').save(os.path.join(a.out_dir, f'calm_{m}'))
        if name == 'blackwhite':
            # calVignette2: c -= (dist² / 0.5) · 0.5，dist 为到中心的归一化距离；导出为 0..1 灰度权重（×0.5 于 LUT 中）
            yy, xx = np.mgrid[0:1024, 0:1024] / 1023.0
            dist2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
            v = np.clip(dist2 / 0.5 * 0.5, 0, 1)
            Image.fromarray((v * 255 + 0.5).astype(np.uint8)).save(os.path.join(a.out_dir, 'blackwhite_vignette.png'))
            dark = np.clip(grid - 1.0, 0, 1)
            write_cube(os.path.join(a.out_dir, 'blackwhite_dark.cube'), dark, a.size, 'blackwhite_dark')
            print('[blackwhite] wrote vignette mask + dark LUT (use --masked-lut blackwhite_dark.cube:blackwhite_vignette.png)')


if __name__ == '__main__':
    main()
