#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""整图与参考导出的色差：1/8 尺寸上逐像素 Lab ΔE（CIE76）的中位数与 90% 分位，对照"不处理"（原图对参考）。

    python tools/image_delta.py --orig test/原片 --ref "test/像素蛋糕“奶油肌”预设产物" --ours out/batch [--md out.md]

按文件名配对（不含扩展名、不区分大小写）。整图色差由背景主导：它衡量调色，脸部与身体的修图差别见 batch_compare.py。
本项目的输出做过形变（瘦脸等），逐像素比较在脸的轮廓附近会多出几个单位，缩到 1/8 后影响很小。
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from montage import imread  # noqa: E402

IMG_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp')
SCALE = 8


def index_dir(d):
    out = {}
    for p in glob.glob(os.path.join(d, '*')):
        stem, ext = os.path.splitext(os.path.basename(p))
        if os.path.isfile(p) and ext.lower() in IMG_EXT:
            out[stem.lower()] = p
    return out


def lab_small(p, size=None):
    im = imread(p)
    size = size or (im.shape[1] // SCALE, im.shape[0] // SCALE)
    im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_RGB2LAB), size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--orig', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--ours', required=True)
    ap.add_argument('--md', help='另存 Markdown 表格')
    args = ap.parse_args()
    orig, ref, ours = index_dir(args.orig), index_dir(args.ref), index_dir(args.ours)
    rows, base_all, ours_all = [], [], []
    for key in sorted(set(orig) & set(ref) & set(ours)):
        R, size = lab_small(ref[key])
        stats = []
        for p in (orig[key], ours[key]):
            X, _ = lab_small(p, size)
            d = np.linalg.norm(X - R, axis=-1)
            stats.append((float(np.median(d)), float(np.percentile(d, 90))))
        base_all.append(stats[0][0])
        ours_all.append(stats[1][0])
        rows.append((os.path.splitext(os.path.basename(orig[key]))[0], stats))
    lines = ['| 照片 | 不处理（中位 / p90） | 本程序（中位 / p90） |', '|---|---|---|']
    for name, ((b50, b90), (o50, o90)) in rows:
        lines.append(f'| {name} | {b50:.1f} / {b90:.1f} | {o50:.2f} / {o90:.2f} |')
    if rows:
        lines.append(f'| 平均 | {np.mean(base_all):.1f} | {np.mean(ours_all):.2f} |')
    text = '\n'.join(lines)
    print(text)
    if args.md:
        with open(args.md, 'w', encoding='utf-8') as fh:
            fh.write(text + '\n')


if __name__ == '__main__':
    main()
