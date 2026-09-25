#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""皮肤纹理衰减与"局部频段能量"的关系：某张输出图（参考导出或本项目输出）相对原图，
在每张脸的皮肤区域内，按瞳距相对尺度分三个频段统计，并按原图的局部频段能量分箱给出能量比。

    python tools/texture_energy.py <orig> <output> <faces.json> [label]

频段（σ 以瞳距 ed 为单位）：fine 0–0.005、mid 0.005–0.035、low 0.035–0.10。
局部能量 = 频段值平方的高斯平滑（σ = 0.05 ed）再开方，单位为 Lab L。
能量图在各自像素网格上计算（不重采样图像内容，避免插值削弱细频段），再用 DIS 光流把输出图的平滑能量图对齐到原图。
该工具用于标定 skin/cream.rs 中"按局部能量自适应"的频段衰减（见 doc/analysis/cream_skin.md §1.3）。
"""
import json
import sys

import cv2
import numpy as np


def imread(p):
    return cv2.cvtColor(cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def skin_rule(rgb):
    r, g, b = rgb[..., 0] / 255.0, rgb[..., 1] / 255.0, rgb[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    return (r > 0.3725) & (g > 0.1568) & (b > 0.0784) & (r > b) & ((mx - mn) > 0.0588) & (np.abs(r - g) > 0.0588)


BANDS = [('fine', 0.0, 0.005), ('mid', 0.005, 0.035), ('low', 0.035, 0.10)]
EDGES = [0, 0.3, 0.6, 1.0, 1.5, 2.2, 3.2, 5.0, 99]


def band(L, lo, hi, ed):
    g1 = L if lo * ed < 0.3 else cv2.GaussianBlur(L, (0, 0), lo * ed)
    return g1 - cv2.GaussianBlur(L, (0, 0), hi * ed)


def face_mask(a, f, X1, Y1):
    ed = f['eye_distance']
    sem = f['semantic']
    poly = np.array(f['contour'] + f['forehead'], np.int32) - [X1, Y1]
    m = np.zeros(a.shape[:2], np.uint8)
    cv2.fillPoly(m, [poly], 1)
    k = max(9, int(round(0.17 * ed)) | 1)
    m = cv2.erode(m, np.ones((k, k), np.uint8)) & skin_rule(a).astype(np.uint8)
    for kk in ('pupil_l', 'pupil_r'):
        cv2.circle(m, (int(sem[kk][0] - X1), int(sem[kk][1] - Y1)), int(0.35 * ed), 0, -1)
    mc = (int((sem['mouth_l'][0] + sem['mouth_r'][0]) / 2 - X1), int((sem['mouth_l'][1] + sem['mouth_r'][1]) / 2 - Y1))
    cv2.ellipse(m, mc, (int(0.45 * ed), int(0.22 * ed)), 0, 0, 360, 0, -1)
    return m.astype(bool)


def measure_face(A, B, f):
    """一张脸：返回 {band: {'total': 比值, 'bins': [(e0, e1, 中位比值, 像素占比), ...]}}。"""
    x1, y1, x2, y2 = [int(v) for v in f['bbox']]
    w, h = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - 0.6 * w)), max(0, int(y1 - 0.7 * h))
    X2, Y2 = min(A.shape[1], int(x2 + 0.6 * w)), min(A.shape[0], int(y2 + 0.5 * h))
    a, b = A[Y1:Y2, X1:X2], B[Y1:Y2, X1:X2]
    ag, bg = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    fl = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(ag, bg, None)
    hh, ww = ag.shape
    gx, gy = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
    ed = f['eye_distance']
    m = face_mask(a, f, X1, Y1)
    la = cv2.cvtColor(a.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]
    lb = cv2.cvtColor(b.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]
    out = {}
    for name, lo, hi in BANDS:
        ba_, bb_ = band(la, lo, hi, ed), band(lb, lo, hi, ed)
        se = 0.05 * ed
        ea = np.sqrt(cv2.GaussianBlur(ba_ ** 2, (0, 0), se))
        eb = np.sqrt(cv2.GaussianBlur(bb_ ** 2, (0, 0), se))
        eb_al = cv2.remap(eb, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ratio = eb_al / np.maximum(ea, 0.02)
        tot = np.sqrt((eb_al[m] ** 2).mean()) / max(np.sqrt((ea[m] ** 2).mean()), 1e-6)
        cells = []
        for e0, e1 in zip(EDGES[:-1], EDGES[1:]):
            sel = m & (ea >= e0) & (ea < e1)
            if sel.sum() > 200:
                cells.append((e0, e1, float(np.median(ratio[sel])), float(sel.sum() / max(m.sum(), 1))))
        out[name] = {'total': float(tot), 'bins': cells}
    return out


def main():
    orig, other, faces_json = sys.argv[1:4]
    label = sys.argv[4] if len(sys.argv) > 4 else 'output'
    A = imread(orig)
    B = imread(other)
    faces = json.load(open(faces_json, encoding='utf-8'))['faces']
    for fi, f in enumerate(faces):
        print(f'== face {fi} ed {f["eye_distance"]:.1f} [{label}]  (能量区间:能量比(像素占比))')
        for name, r in measure_face(A, B, f).items():
            cells = ' '.join(f'{e0:.1f}-{e1:.1f}:{v:.2f}({100 * frac:.0f}%)' for e0, e1, v, frac in r['bins'])
            print(f'   {name:4s} total {r["total"]:.2f} | {cells}')


if __name__ == '__main__':
    main()
