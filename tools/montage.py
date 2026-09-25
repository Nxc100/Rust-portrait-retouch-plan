#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""并排裁剪对比图（原图 | 参考 | 本项目 …），每块上方写标题。out/*/compare_*.png 都由它生成。

    python tools/montage.py <out.png> <x0,y0,x1,y1> <tile_width> <img1>::<标题1> <img2>::<标题2> ...

tile_width 为每块缩放后的宽度（0 表示不缩放）；裁剪框用原图像素坐标。
"""
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def imread(p):
    return cv2.cvtColor(cv2.imdecode(np.fromfile(p, np.uint8), 1), cv2.COLOR_BGR2RGB)


def font(sz):
    for f in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/System/Library/Fonts/PingFang.ttc']:
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def montage(paths, labels, box, out, width=None):
    x0, y0, x1, y1 = box
    tiles = []
    for p in paths:
        im = imread(p)[y0:y1, x0:x1]
        if width:
            s = width / im.shape[1]
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
            im = cv2.resize(im, (width, int(round(im.shape[0] * s))), interpolation=interp)
        tiles.append(im)
    h = max(t.shape[0] for t in tiles)
    gap, head = 8, 44
    canvas = Image.new('RGB', (sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1), h + head), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    f = font(26)
    x = 0
    for t, lab in zip(tiles, labels):
        canvas.paste(Image.fromarray(t), (x, head))
        d.text((x + 8, 8), lab, fill=(0, 0, 0), font=f)
        x += t.shape[1] + gap
    canvas.save(out)
    print('saved', out, canvas.size)


if __name__ == '__main__':
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    out, box, width = sys.argv[1], [int(v) for v in sys.argv[2].split(',')], int(sys.argv[3])
    items = [a.rsplit('::', 1) for a in sys.argv[4:]]
    montage([i[0] for i in items], [i[1] if len(i) > 1 else '' for i in items], box, out, width or None)
