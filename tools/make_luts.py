#!/usr/bin/env python
"""生成 luts/ 下的恒等查找图与几款参数化示例 LUT（零素材，版权属于本项目）。

    python tools/make_luts.py [--out luts]

输出：
- identity_512.png     恒等 512×512 查找图（方案 4.5.4 第 1 步，供设计师在 PS/LR 中调色）
- identity_33.cube     恒等 .cube
- whiten_soft.png      参数化美白查找图（提亮中间调、轻微降红饱和），可用于 `--whiten-lut`
- warm.cube / cool.cube / fade.cube / mono.cube   示例风格 LUT
- index.json           名称、文件、类型、默认强度
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def identity_512():
    y, x = np.mgrid[0:512, 0:512]
    r = (x % 64) / 63.0
    g = (y % 64) / 63.0
    b = ((y // 64) * 8 + x // 64) / 63.0
    return np.stack([r, g, b], -1).astype(np.float32)


def identity_grid(n):
    b, g, r = np.mgrid[0:n, 0:n, 0:n]  # r 变化最快
    m = n - 1
    return np.stack([r / m, g / m, b / m], -1).reshape(-1, 3).astype(np.float32)


def write_cube(path, grid, n, title):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        for v in np.clip(grid, 0, 1):
            f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')


def save_png(path, arr):
    Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8)).save(path)


def lum(c):
    return 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]


def whiten_soft(c):
    out = np.power(np.clip(c, 0, 1), 1 / 1.25)
    gb = 0.5 * (out[..., 1] + out[..., 2])
    out[..., 0] += (gb - out[..., 0]) * 0.06
    return out


def warm(c):
    out = c.copy()
    out[..., 0] = np.clip(c[..., 0] * 1.06 + 0.02, 0, 1)
    out[..., 2] = np.clip(c[..., 2] * 0.94, 0, 1)
    l = lum(out)[..., None]
    return np.clip(l + (out - l) * 1.08, 0, 1)


def cool(c):
    out = c.copy()
    out[..., 2] = np.clip(c[..., 2] * 1.06 + 0.02, 0, 1)
    out[..., 0] = np.clip(c[..., 0] * 0.95, 0, 1)
    return out


def fade(c):
    # 抬黑压白 + 轻微去饱和
    out = 0.06 + c * 0.9
    l = lum(out)[..., None]
    return np.clip(l + (out - l) * 0.85, 0, 1)


def mono(c):
    l = lum(c)
    curve = np.clip((l - 0.5) * 1.1 + 0.5, 0, 1)
    return np.stack([curve, curve, curve], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'luts'))
    ap.add_argument('--size', type=int, default=33)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ident = identity_512()
    save_png(os.path.join(a.out, 'identity_512.png'), ident)
    save_png(os.path.join(a.out, 'whiten_soft.png'), whiten_soft(ident))
    grid = identity_grid(a.size)
    write_cube(os.path.join(a.out, 'identity_33.cube'), grid, a.size, 'identity')
    styles = {'warm': warm, 'cool': cool, 'fade': fade, 'mono': mono}
    for name, fn in styles.items():
        write_cube(os.path.join(a.out, f'{name}.cube'), fn(grid), a.size, name)
    index = {
        'luts': [
            {'name': '原图', 'file': 'identity_512.png', 'kind': 'lookup512', 'default_intensity': 0.0},
            {'name': '暖调', 'file': 'warm.cube', 'kind': 'cube', 'default_intensity': 0.8},
            {'name': '冷调', 'file': 'cool.cube', 'kind': 'cube', 'default_intensity': 0.8},
            {'name': '褪色', 'file': 'fade.cube', 'kind': 'cube', 'default_intensity': 0.8},
            {'name': '黑白', 'file': 'mono.cube', 'kind': 'cube', 'default_intensity': 1.0},
        ],
        'whiten': [{'name': '柔和美白', 'file': 'whiten_soft.png', 'kind': 'lookup512'}],
    }
    with open(os.path.join(a.out, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print('written to', a.out)


if __name__ == '__main__':
    main()
