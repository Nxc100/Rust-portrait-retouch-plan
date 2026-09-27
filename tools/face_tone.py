#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐脸的皮肤颜色：脸部皮肤（compare_to_reference 的遮罩）与脸下方身体皮肤的 Lab 均值、脸 − 身体之差，
比较原图 / 底图 / 参考导出 / 本项目输出（光流对齐到原图）。

    python tools/face_tone.py --orig test/原片 --landmarks <关键点目录> --ref <参考导出目录> \\
        [--base <只调色目录>] [--ours 名称=目录 ...] [--only 照片名 ...]

用途：肤色是否偏黄 / 偏红 / 偏暗（逐脸，而不是全体平均）；脸和身体是否连贯（脸比身体白太多会像面具）。
身体皮肤 = 脸框下方 0.3～1.6 个脸高、左右各 1.2 个脸宽内、原图上符合肤色规则的像素（粗略，用于对比两边的差）。
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

MIN_SCORE = 0.95


def fmt(v):
    return '/'.join(f'{x:5.1f}' for x in v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--orig', required=True)
    ap.add_argument('--landmarks', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--base', default=None, help='只调色的输出目录（有预设调色时）')
    ap.add_argument('--ours', action='append', default=[], metavar='名称=目录')
    ap.add_argument('--only', nargs='*', default=[], help='只看这些照片')
    args = ap.parse_args()
    dirs = ([('底图', args.base)] if args.base else []) + [tuple(o.split('=', 1)) for o in args.ours] + [('参考', args.ref)]
    for lm in sorted(glob.glob(os.path.join(args.landmarks, '*.json'))):
        stem = os.path.splitext(os.path.basename(lm))[0]
        if args.only and stem not in args.only:
            continue
        src = find(args.orig, stem)
        paths = [find(d, stem) for _, d in dirs]
        if src is None or None in paths:
            continue
        A = ctr.imread(src)
        for k, f in enumerate(x for x in json.load(open(lm, encoding='utf-8'))['faces'] if x.get('score', 1.0) >= MIN_SCORE):
            (x1, y1, x2, y2), fm = ctr.face_region(A, f)
            bx1, by1, bx2, by2 = [int(v) for v in f['bbox']]
            w, h, cx = bx2 - bx1, by2 - by1, (bx1 + bx2) / 2
            X1, Y1 = max(0, int(cx - 1.2 * w)), min(A.shape[0], int(by2 + 0.3 * h))
            X2, Y2 = min(A.shape[1], int(cx + 1.2 * w)), min(A.shape[0], int(by2 + 1.6 * h))
            bm = ctr.skin_rule(A[Y1:Y2, X1:X2]).astype(bool) if Y2 > Y1 + 10 else np.zeros((1, 1), bool)
            ag = cv2.cvtColor(A[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
            print(f'{stem} f{k} {f.get("gender", "?")}')
            for name, p in [('原图', src)] + [(n, q) for (n, _), q in zip(dirs, paths)]:
                B = ctr.imread(p)
                al, _ = ctr.flow_align(ag, cv2.cvtColor(B[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY), B[y1:y2, x1:x2])
                lab = cv2.cvtColor(al.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[fm]
                face = lab.mean(0)
                line = f'  {name:6s} 脸 {fmt(face)} σL {lab[:, 0].std():4.1f}'
                if bm.sum() > 100:
                    body = cv2.cvtColor(B[Y1:Y2, X1:X2].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[bm].mean(0)
                    line += f'   身体 {fmt(body)}   脸−身体 ' + '/'.join(f'{x:+5.1f}' for x in face - body)
                print(line)


if __name__ == '__main__':
    main()
