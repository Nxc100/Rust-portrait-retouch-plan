#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重新生成文档引用的样张对比图（out/cream/、out/x04/、out/batch_compare/）。

    python tools/sample_montages.py [cream] [x04] [batch]

前提：已按 out/README.md 生成本项目输出（out/cream/ours_cream.jpg、out/x04/ours_cream.jpg、
out/x04/lite/ours_cream_lite.jpg、out/batch/*.jpg 与 out/batch/_landmarks/）；
轻量皮肤分割对照还需要 out/_work/beach_lite.jpg（`retouch apply ... --skinseg-model models/skin_seg_lite.onnx`）。
裁剪框为原图像素坐标，与各报告中的描述对应。
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from montage import imread, montage  # noqa: E402

BEACH, BEACH_PC, BEACH_OURS = 'test/1V3A2861.jpg', 'test/示例项目_导出/1V3A2861.jpg', 'out/cream/ours_cream.jpg'
BEACH_LITE = 'out/_work/beach_lite.jpg'
X04 = 'test/04_a_原图_X04.jpg'
X04_PC = 'test/示例项目_导出/04_a_原图_X04_(2).jpg'   # 重新导出的「奶油肌」（第一次导出带示例项目的调色）
X04_OURS, X04_LITE = 'out/x04/ours_cream.jpg', 'out/x04/lite/ours_cream_lite.jpg'


def three(orig, ref, ours):
    return [(orig, '原图'), (ref, '像素蛋糕'), (ours, '本程序')]


CREAM = [
    ('out/cream/compare_arm_scar.png', (3320, 2985, 3520, 3135), 352),
    ('out/cream/compare_bride_cheek.png', (3080, 2148, 3384, 2448), 302),
    ('out/cream/compare_bride_chest.png', (2752, 2748, 3452, 3300), 491),
    ('out/cream/compare_bride_eye.png', (2840, 1940, 3260, 2130), 420),
    ('out/cream/compare_bride_face.png', (2800, 1700, 3324, 2480), 314),
    ('out/cream/compare_bride_nose.png', (2936, 2064, 3260, 2332), 452),
    ('out/cream/compare_groom_cheek.png', (1648, 1500, 1952, 1800), 302),
    ('out/cream/compare_groom_face.png', (1352, 1148, 1964, 2008), 337),
    ('out/cream/compare_hands.png', (2300, 4548, 2904, 5048), 422),
    ('out/cream/compare_overview.png', (0, 0, 4355, 6533), 698),
]
X04_JOBS = [
    ('out/x04/compare_bride_chest.png', (1900, 1548, 2436, 2300), 321),
    ('out/x04/compare_bride_face.png', (2000, 1152, 2436, 1752), 400),
    ('out/x04/compare_bride_eye.png', (2130, 1350, 2400, 1480), 420),
    ('out/x04/compare_groom_face.png', (1552, 1052, 1960, 1700), 380),
    ('out/x04/compare_hands.png', (1748, 2052, 2084, 2452), 301),
    ('out/x04/compare_overview.png', (0, 0, 3648, 5472), 487),
    ('out/x04/compare_wall.png', (2560, 700, 2860, 900), 360),
]


def change_locations(orig, ours, out, thr, long_side=1400):
    """与原图差异 > thr/255 的连通块（红框），画在输出图的缩略上。"""
    a, b = imread(orig).astype(np.int16), imread(ours).astype(np.int16)
    d = (np.abs(a - b).max(2) > thr).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(d)
    s = long_side / max(a.shape[:2])
    vis = cv2.resize(imread(ours), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area >= 60:
            cv2.rectangle(vis, (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s)), (255, 0, 0), 2)
    cv2.imencode('.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))[1].tofile(out)
    print('saved', out, n - 1, 'components')


def batch_faces(orig_dir='test/原片', ref_dir='test/像素蛋糕“奶油肌”预设产物', ours_dir='out/batch',
                lm_dir='out/batch/_landmarks', out_dir='out/batch_compare'):
    """每张脸一张：原图 | 像素蛋糕 | 本程序（裁剪框取人脸框的 1.5 倍正方形）。"""
    os.makedirs(out_dir, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(orig_dir, '*'))):
        stem = os.path.splitext(os.path.basename(p))[0]
        refs = [q for q in glob.glob(os.path.join(ref_dir, '*')) if os.path.splitext(os.path.basename(q))[0].lower() == stem.lower()]
        lm = os.path.join(lm_dir, stem + '.json')
        ours = os.path.join(ours_dir, stem + '.jpg')
        if not refs or not os.path.exists(lm) or not os.path.exists(ours):
            continue
        for fi, f in enumerate(json.load(open(lm, encoding='utf-8'))['faces']):
            x1, y1, x2, y2 = f['bbox']
            cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.75
            box = (max(0, int(cx - s)), max(0, int(cy - s)), int(cx + s), int(cy + s))
            montage([p, refs[0], ours], ['原图', '像素蛋糕', '本程序'], box, os.path.join(out_dir, f'{stem}_f{fi}.png'), 520)


def main():
    which = set(sys.argv[1:]) or {'cream', 'x04', 'batch'}
    if 'cream' in which:
        for out, box, width in CREAM:
            items = three(BEACH, BEACH_PC, BEACH_OURS)
            montage([p for p, _ in items], [lab for _, lab in items], box, out, width)
        change_locations(BEACH, BEACH_OURS, 'out/cream/change_locations.png', 10)
    if 'x04' in which:
        for out, box, width in X04_JOBS:
            items = three(X04, X04_PC, X04_OURS)
            montage([p for p, _ in items], [lab for _, lab in items], box, out, width)
        montage([X04, X04_OURS, X04_LITE], ['原图', '完整皮肤分割', '轻量皮肤分割'], (1900, 1552, 2704, 2300),
                'out/x04/compare_chest_full_vs_lite.png', 442)
        if os.path.exists(BEACH_LITE):
            montage([BEACH, BEACH_OURS, BEACH_LITE], ['原图', '完整皮肤分割', '轻量皮肤分割'], (2752, 2748, 3704, 3648),
                    'out/x04/compare_beach_arm_full_vs_lite.png', 429)
        change_locations(X04, X04_OURS, 'out/x04/change_locations.png', 8)
    if 'batch' in which:
        batch_faces()


if __name__ == '__main__':
    main()
