#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 `retouch apply --debug-dir` 导出的各张脸皮肤遮罩（skin_face<k>.png）与身体遮罩（skin_body.png）按颜色叠到原图上，
并报告两两重叠的面积——用来查"一个人的遮罩盖到另一个人脸上"（两人脸挨得近时人脸解析的越界）与漏检的皮肤。

    python tools/mask_overlay.py --debug <调试目录> --image <原图> --out <叠加图.jpg> [--box x0,y0,x1,y1]

颜色：脸 0 红、脸 1 绿、脸 2 蓝、脸 3 黄；身体为青色描边。两张脸的遮罩重叠 = 同一片皮肤会按两个人的参数各处理一遍
（预设开了 `face_ownership` 时处理前会去掉落在别人脸轮廓内的部分，这里画的是去掉之前的原始遮罩）。
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from montage import imread  # noqa: E402

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


def read_mask(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--debug', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--box', default=None, help='只画这个框（原图坐标 x0,y0,x1,y1）')
    args = ap.parse_args()
    img = imread(args.image).astype(np.float32)
    faces = sorted(glob.glob(os.path.join(args.debug, 'skin_face*.png')))
    masks = [read_mask(p) for p in faces]
    vis = img.copy()
    for m, c in zip(masks, COLORS):
        vis = vis * (1 - 0.4 * m[..., None]) + np.array(c, np.float32) * 0.4 * m[..., None]
    body_p = os.path.join(args.debug, 'skin_body.png')
    if os.path.exists(body_p):
        edge = cv2.Canny((read_mask(body_p) > 0.5).astype(np.uint8) * 255, 50, 150) > 0
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        vis[edge] = (0, 255, 255)
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            n = int(((masks[i] > 0.5) & (masks[j] > 0.5)).sum())
            print(f'脸 {i} 与脸 {j} 的遮罩重叠 {n} 像素' + ('（越界：检查 face_ownership）' if n > 0 else ''))
    if args.box:
        x0, y0, x1, y1 = (int(v) for v in args.box.split(','))
        img, vis = img[y0:y1, x0:x1], vis[y0:y1, x0:x1]
    out = np.concatenate([img, vis], 1).clip(0, 255).astype(np.uint8)
    s = min(1.0, 2400 / out.shape[1])
    out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imencode('.jpg', cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
