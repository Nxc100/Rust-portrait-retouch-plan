#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""脸部皮肤各尺度的明暗结构增益：每个亮度带通上 输出 ≈ α·底图 的最小二乘 α 与相关系数 r（皮肤遮罩内，光流对齐）。

    python tools/band_gain.py --orig test/原片 --landmarks <关键点目录> --base <底图目录> \\
        --ref <参考导出目录> [--ours 名称=目录 ...]

底图：有预设调色时为"只调色、不修图、不形变"的输出（见 doc/adding_a_preset.md 第 3 步），没有调色时就是原片目录。
带通 σ = 0.008 / 0.016 / 0.032 / 0.064 / 0.128 / 0.256 瞳距。α < 1 为压平，α > 1 为加强：参考在 0.064 以上的 α 大于 1
说明它有"立体"（五官与轮廓的明暗加强，`CreamParams::stereo`），细的几档 α 对应磨皮。r 低说明不是简单缩放
（有选择地修掉了一部分结构，如瑕疵、眼袋）。按性别汇总平均；逐脸行便于找离群的脸。
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
from skin_bands import find  # noqa: E402

BANDS = (0.008, 0.016, 0.032, 0.064, 0.128, 0.256)
MIN_SCORE = 0.95


def band_pass(L, sigmas):
    out, prev = [], L
    for s in sigmas:
        g = cv2.GaussianBlur(L, (0, 0), s)
        out.append(prev - g)
        prev = g
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--orig', required=True, help='原片目录（光流对齐的基准几何）')
    ap.add_argument('--landmarks', required=True, help='关键点 JSON 目录')
    ap.add_argument('--base', required=True, help='底图目录（只调色的输出；没有调色时同 --orig）')
    ap.add_argument('--ref', required=True, help='参考导出目录')
    ap.add_argument('--ours', action='append', default=[], metavar='名称=目录')
    args = ap.parse_args()
    dirs = [('参考', args.ref)] + [tuple(o.split('=', 1)) for o in args.ours]
    acc = {n: {'female': [], 'male': []} for n, _ in dirs}
    for lm in sorted(glob.glob(os.path.join(args.landmarks, '*.json'))):
        stem = os.path.splitext(os.path.basename(lm))[0]
        src, base = find(args.orig, stem), find(args.base, stem)
        others = [find(d, stem) for _, d in dirs]
        if src is None or base is None or None in others:
            continue
        A = ctr.imread(src)
        for k, f in enumerate(x for x in json.load(open(lm, encoding='utf-8'))['faces'] if x.get('score', 1.0) >= MIN_SCORE):
            gender = 'male' if f.get('gender', 'female').startswith('m') else 'female'
            (x1, y1, x2, y2), m = ctr.face_region(A, f)
            ag = cv2.cvtColor(A[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
            sig = [max(0.6, f['eye_distance'] * r) for r in BANDS]

            def lum(p):
                B = ctr.imread(p)[y1:y2, x1:x2]
                al, _ = ctr.flow_align(ag, cv2.cvtColor(B, cv2.COLOR_RGB2GRAY), B)
                return cv2.cvtColor(al.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]

            bb = band_pass(lum(base), sig)
            line = [f'{stem} f{k} {gender[0]}']
            for (name, _), p in zip(dirs, others):
                bx = band_pass(lum(p), sig)
                al_, rr = [], []
                for x, y in zip(bb, bx):
                    xv, yv = x[m], y[m]
                    al_.append(float((xv * yv).sum() / max((xv * xv).sum(), 1e-9)))
                    rr.append(float(np.corrcoef(xv, yv)[0, 1]))
                acc[name][gender].append((al_, rr))
                line.append(f'{name} α ' + ' '.join(f'{v:4.2f}' for v in al_))
            print('  '.join(line))
    print('\n带通 σ（瞳距）: ' + ' / '.join(str(b) for b in BANDS))
    for gender, label in (('female', '女'), ('male', '男')):
        for name, _ in dirs:
            rows = acc[name][gender]
            if rows:
                al_ = np.mean([r[0] for r in rows], 0)
                rr = np.mean([r[1] for r in rows], 0)
                print(f'{label} {name}: α ' + ' '.join(f'{v:4.2f}' for v in al_) + '   r ' + ' '.join(f'{v:4.2f}' for v in rr)
                      + f'   ({len(rows)} 张脸)')


if __name__ == '__main__':
    main()
