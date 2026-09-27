#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脸部皮肤的平整度：亮度各带通的中位幅度相对原图之比，比较参考导出（如像素蛋糕）与若干本项目输出。

    python tools/skin_bands.py --orig test/原片 --landmarks out/batch_dark_interior/_landmarks \\
        --ref "test/像素蛋糕“婚纱-深色内景”预设产物" --ours 定稿=out/batch_dark_interior [--ours 名称=目录 ...]

带通 σ = 0.004 / 0.008 / 0.016 / 0.032 / 0.064 瞳距（下限 0.6 px），皮肤遮罩与 compare_to_reference.py 相同，
输出光流对齐到原图。用**中位**幅度而不是 RMS：像素蛋糕的清晰度、对比度放大少数强边缘（毛孔、皱纹、发丝），
RMS 由它们主导，看不出大多数皮肤像素被中性灰压平了（doc/analysis/wedding_dark_interior.md §3.2）。
逐脸输出，再按性别汇总平均；另给脸部低频（σ 0.1 瞳距）Lab 的标准差（色块 / 明暗不均）。
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_to_reference as ctr  # noqa: E402

IMG_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp')
BANDS = (0.004, 0.008, 0.016, 0.032, 0.064)
MIN_SCORE = 0.95


def find(d, stem):
    for p in glob.glob(os.path.join(d, '*')):
        s, ext = os.path.splitext(os.path.basename(p))
        if s.lower() == stem.lower() and ext.lower() in IMG_EXT:
            return p
    return None


def band_pass(L, sigmas):
    out, prev = [], L
    for s in sigmas:
        g = cv2.GaussianBlur(L, (0, 0), s)
        out.append(prev - g)
        prev = g
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--orig', required=True, help='原片目录')
    ap.add_argument('--landmarks', required=True, help='关键点 JSON 目录（retouch batch --landmarks-dir）')
    ap.add_argument('--ref', required=True, help='参考导出目录')
    ap.add_argument('--ours', action='append', default=[], metavar='名称=目录', help='本项目输出（可多次）')
    args = ap.parse_args()
    dirs = [('参考', args.ref)] + [tuple(o.split('=', 1)) for o in args.ours]
    ratios = {n: {'female': [], 'male': []} for n, _ in dirs}
    lows = {n: {'female': [], 'male': []} for n, _ in dirs}
    for lm in sorted(glob.glob(os.path.join(args.landmarks, '*.json'))):
        stem = os.path.splitext(os.path.basename(lm))[0]
        src = find(args.orig, stem)
        others = [find(d, stem) for _, d in dirs]
        if src is None or None in others:
            continue
        A = ctr.imread(src)
        faces = [f for f in json.load(open(lm, encoding='utf-8'))['faces'] if f.get('score', 1.0) >= MIN_SCORE]
        for k, f in enumerate(faces):
            gender = 'male' if f.get('gender', 'female').startswith('m') else 'female'
            (x1, y1, x2, y2), m = ctr.face_region(A, f)
            a = A[y1:y2, x1:x2]
            sig = [max(0.6, f['eye_distance'] * r) for r in BANDS]
            la = cv2.cvtColor(a.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
            base = band_pass(la[..., 0], sig)
            line = [f'{stem} f{k} {gender[0]}']
            for (name, _), p in zip(dirs, others):
                B = ctr.imread(p)[y1:y2, x1:x2]
                al, _ = ctr.flow_align(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(B, cv2.COLOR_RGB2GRAY), B)
                lb = cv2.cvtColor(al.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
                r = [np.median(np.abs(y[m])) / max(np.median(np.abs(x[m])), 1e-6)
                     for x, y in zip(base, band_pass(lb[..., 0], sig))]
                low = np.stack([cv2.GaussianBlur(lb[..., c], (0, 0), 0.1 * f['eye_distance']) for c in range(3)], -1)
                ratios[name][gender].append(r)
                lows[name][gender].append(low[m].std(0))
                line.append(f'{name} ' + ' '.join(f'{v:4.2f}' for v in r))
            print('  '.join(line))
    print('\n带通 σ（瞳距）: ' + ' / '.join(str(b) for b in BANDS))
    for gender, label in (('female', '女'), ('male', '男')):
        for name, _ in dirs:
            if not ratios[name][gender]:
                continue
            r, lo = np.mean(ratios[name][gender], 0), np.mean(lows[name][gender], 0)
            print(f'{label} {name}: 中位比 ' + ' '.join(f'{v:4.2f}' for v in r)
                  + f'   低频 std L/a/b {lo[0]:4.2f} {lo[1]:4.2f} {lo[2]:4.2f}   ({len(ratios[name][gender])} 张脸)')


if __name__ == '__main__':
    main()
