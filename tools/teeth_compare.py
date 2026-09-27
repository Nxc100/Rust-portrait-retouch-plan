#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""牙齿颜色：用 `retouch apply --debug-dir` 导出的 teeth.png（与流水线同一份权重，> 0.6 的像素）做遮罩，
光流对齐后比较 原图 / 参考导出 / 本项目输出 的 Lab 均值，并拼出嘴部对比图。

    python tools/teeth_compare.py --debug <调试目录> --orig <原图> --ref <参考导出> --ours <本项目输出> [--out 嘴部.jpg]

没有 teeth.png 说明这张照片没有要美白的牙齿（没露齿、没有人脸解析或预设的牙齿美白为 0）——先确认照片里真的露齿。
先用 teeth_whiten > 0 的参数跑一遍 --debug-dir 导出 teeth.png（它不依赖美白强度的具体值，只要 > 0）。
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_to_reference as ctr  # noqa: E402
from montage import montage  # noqa: E402

MIN_SCORE = 0.95


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--debug', required=True, help='含 teeth.png 与 landmarks.json 的调试目录')
    ap.add_argument('--orig', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--ours', required=True)
    ap.add_argument('--out', default=None, help='嘴部对比图（每张露齿的脸一张，文件名加 _<脸序号>）')
    args = ap.parse_args()
    tp = os.path.join(args.debug, 'teeth.png')
    if not os.path.exists(tp):
        raise SystemExit('没有 teeth.png：这张照片没有要美白的牙齿')
    t = cv2.imdecode(np.fromfile(tp, np.uint8), 0).astype(np.float32) / 255
    faces = [f for f in json.load(open(os.path.join(args.debug, 'landmarks.json'), encoding='utf-8'))['faces']
             if f.get('score', 1.0) >= MIN_SCORE]
    A = ctr.imread(args.orig)
    for k, f in enumerate(faces):
        ml, mr = np.array(f['semantic']['mouth_l']), np.array(f['semantic']['mouth_r'])
        c, wd = (ml + mr) / 2, np.linalg.norm(mr - ml)
        x0, y0 = max(0, int(c[0] - 0.9 * wd)), max(0, int(c[1] - 0.6 * wd))
        x1, y1 = int(c[0] + 0.9 * wd), int(c[1] + 0.6 * wd)
        m = t[y0:y1, x0:x1] > 0.6
        if m.sum() < 50:
            continue
        a = A[y0:y1, x0:x1]
        row = [f'脸 {k} {f.get("gender", "?")} n={int(m.sum())}']
        for name, p in (('原图', None), ('本程序', args.ours), ('参考', args.ref)):
            if p is None:
                al = a
            else:
                B = ctr.imread(p)[y0:y1, x0:x1]
                al, _ = ctr.flow_align(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(B, cv2.COLOR_RGB2GRAY), B)
            lab = cv2.cvtColor(al.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[m]
            row.append(f'{name} ' + '/'.join(f'{v:5.1f}' for v in lab.mean(0)))
        print('  '.join(row))
        if args.out:
            stem, ext = os.path.splitext(args.out)
            montage([args.orig, args.ours, args.ref], ['原图', '本程序', '参考'], (x0, y0, x1, y1),
                    f'{stem}_{k}{ext}', 480, 92)


if __name__ == '__main__':
    main()
