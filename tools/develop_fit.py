#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""拟合"内嵌 Camera Raw 设置再应用一次"的逐像素颜色模型（src/color/develop.rs 的常数来源）。

像素蛋糕处理带 crs 设置（Adobe Camera Raw / Lightroom 的冲印参数，含 crs:AlreadyApplied=True）的 JPEG 时，
会把这些设置再应用一次；本脚本在参考导出图的非皮肤区域（σ2 平滑后的颜色，含金饰 / 刺绣 / 地毯纹样）上拟合模型常数，
并报告与 16³ LUT 上限的差距、按色相 / 亮度分箱的残差。

    python tools/develop_fit.py <orig.jpg> <reference.jpg> [--samples N]

模型（与 develop.rs 一致，Lab 空间）：
- 色调（L）：高光凸包 exp(−((x−c_h)/w_h)²)、黑色斜坡 1−smoothstep(0,e_k,x)、阴影凸包 exp(−((x−0.25)/0.18)²)、
  白色斜坡 smoothstep(0.6,1,x)；x = L/100；增益 × 滑块/100；色调变化按 k_tc·ΔL/L 同比缩放彩度；
- 白平衡：Δb = g_temp·(T−5500)/5500·x·10，Δa = g_tint·tint/100·x·10；暗部偏色挂在"黑色"滑块上；
- HSL：按 sRGB 的 HSV 色相分 8 桶（红 0、橙 30、黄 60、绿 120、浅绿 180、蓝 240、紫 270、洋红 300，Adobe 的定义），
  余弦窗（宽度系数 hw），色相偏移 g_hue·Σw·hue/100（度）、彩度 ×(1+g_sat·Σw·sat/100·bump(s))、
  bump(s) = (s/s_pk)^q·exp(q·(1 − s/s_pk))（在 HSV 饱和度 s_pk 处为 1：肤色降饱和最多，奶油色 / 金色较少）、
  亮度 +g_lum·Σw·lum/100·s^p_lum（s 为 HSV 饱和度，中性色不受影响）；
- 自然饱和度：C ×(1 + g_vib·vib/100·(1−s)·(1−vib_skin·w_橙))；饱和度：C ×(1 + sat/100)。
皮肤区域由 models/skin_seg.onnx 给出（膨胀后排除）。可选 --faces 把一张脸的皮肤也纳入拟合：
目标 = 参考导出图 − "只修图"的色调偏移（按原图 L 分箱，默认取海边样张新娘的实测值：该样张没有 crs 设置）。
"""
import argparse
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

CENTERS = np.array([0, 30, 60, 120, 180, 240, 270, 300], np.float64)
PN = ['g_high', 'c_high', 'w_high', 'g_black', 'e_black', 'k_tonechroma', 'g_temp', 'g_tint', 'g_blk_a', 'g_blk_b',
      'g_hue', 'g_sat', 'g_lum', 'p_lum', 'hsl_width', 'g_vib', 'vib_skin', 's_peak', 'q_sat']
P0 = np.array([5.0, 0.85, 0.15, 2.0, 0.5, 0.8, 0.5, 0.5, -7.0, -10.0, 25.0, 1.0, 5.0, 0.5, 0.9, 1.5, 0.6, 0.35, 1.0])
LB = [0, 0.5, 0.05, 0, 0.1, 0, 0, 0, -20, -20, 0, 0, 0, 0.2, 0.5, 0, 0, 0.1, 0]
UB = [20, 1.1, 0.5, 20, 0.9, 1.5, 3, 3, 20, 20, 60, 2.0, 30, 3, 2.0, 5, 1, 0.8, 6]
# 海边样张新娘（无 crs 设置）的"只修图"色调偏移：原图 L 分箱 → (dL, da, db)
RETOUCH_FEMALE = {35: (-0.1, -0.5, -0.1), 45: (1.0, -0.3, 0.2), 55: (1.5, -1.2, -0.2), 65: (1.3, -1.3, -0.7),
                  75: (1.3, -1.2, -0.8), 85: (0.5, -0.8, -1.1), 95: (-1.3, -0.1, 0.7)}


def imread(p):
    return cv2.cvtColor(cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def parse_crs(path):
    import re
    data = open(path, 'rb').read()
    i, j = data.find(b'<x:xmpmeta'), data.find(b'</x:xmpmeta>')
    if i < 0:
        return None
    x = data[i:j].decode('utf-8', 'replace')
    kv = dict(re.findall(r'crs:([A-Za-z0-9]+)="([^"]*)"', x))
    if kv.get('HasSettings') != 'True':
        return None
    f = lambda k, d=0.0: float(kv.get(k, d))
    names = ['Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta']
    return dict(temperature=f('Temperature', 5500), tint=f('Tint'), highlights=f('Highlights2012'), shadows=f('Shadows2012'),
                whites=f('Whites2012'), blacks=f('Blacks2012'), vibrance=f('Vibrance'), saturation=f('Saturation'),
                hue=[f('HueAdjustment' + n) for n in names], sat=[f('SaturationAdjustment' + n) for n in names],
                lum=[f('LuminanceAdjustment' + n) for n in names])


def hue_weights(h, width):
    c = CENTERS
    prev = np.roll(c, 1); prev[0] -= 360
    nxt = np.roll(c, -1); nxt[-1] += 360
    w = np.zeros(h.shape + (8,))
    for i in range(8):
        d = ((h - c[i] + 180) % 360) - 180
        span = np.where(d < 0, c[i] - prev[i], nxt[i] - c[i]) * width
        t = np.clip(np.abs(d) / span, 0, 1)
        w[:, i] = 0.5 * (1 + np.cos(np.pi * t))
    return w / np.maximum(w.sum(1, keepdims=True), 1e-6)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def model(lab, p, S, hsv):
    g_h, c_h, w_h, g_k, e_k, k_tc, g_temp, g_tint, g_ba, g_bb, g_hue, g_sat, g_lum, p_lum, hw, g_vib, vib_skin, s_pk, q_sat = p
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    x = L / 100
    C = np.hypot(a, b)
    h, s = hsv[:, 0], np.clip(hsv[:, 1], 0, 1)
    dL = (g_h * S['highlights'] * np.exp(-((x - c_h) / w_h) ** 2) + g_k * S['blacks'] * (1 - smoothstep(0, e_k, x))
          + g_h * S['shadows'] * np.exp(-((x - 0.25) / 0.18) ** 2) + g_h * S['whites'] * smoothstep(0.6, 1.0, x)) / 100
    tone_cf = 1 + k_tc * dL / np.maximum(L, 5)
    dark = 1 - smoothstep(0, e_k, x)
    a2 = a * tone_cf + g_tint * S['tint'] / 100 * x * 10 + g_ba * S['blacks'] / 100 * dark
    b2 = b * tone_cf + g_temp * (S['temperature'] - 5500) / 5500 * x * 10 + g_bb * S['blacks'] / 100 * dark
    w = hue_weights(h, hw)
    dh = g_hue * (w @ (np.array(S['hue']) / 100))
    r_ = np.maximum(s, 1e-4) / s_pk
    sat_f = 1 + g_sat * (w @ (np.array(S['sat']) / 100)) * np.power(r_, q_sat) * np.exp(q_sat * (1 - r_))
    dL = dL + g_lum * (w @ (np.array(S['lum']) / 100)) * np.power(s, p_lum)
    vib_f = 1 + g_vib * S['vibrance'] / 100 * (1 - s) * (1 - vib_skin * w[:, 1])
    satg_f = 1 + S['saturation'] / 100
    C2 = np.hypot(a2, b2) * sat_f * vib_f * satg_f
    H2 = np.arctan2(b2, a2) + np.radians(dh)
    return np.stack([L + dL, C2 * np.cos(H2), C2 * np.sin(H2)], 1)


def face_skin_samples(A, B, a, rng):
    """一张脸的皮肤低频颜色样本：原图 / 参考（光流对齐）/ HSV；目标减去"只修图"的色调偏移。"""
    f = json.load(open(a.faces, encoding='utf-8'))['faces'][a.face]
    ed = f['eye_distance']
    sem = f['semantic']
    x1, y1, x2, y2 = [int(v) for v in f['bbox']]
    X1, Y1, X2, Y2 = max(0, x1 - 60), max(0, y1 - 60), min(A.shape[1], x2 + 60), min(A.shape[0], y2 + 60)
    a_, b_ = A[Y1:Y2, X1:X2], B[Y1:Y2, X1:X2]
    fm = np.zeros(a_.shape[:2], np.uint8)
    cv2.fillPoly(fm, [np.array(f['contour'] + f['forehead'], np.int32) - [X1, Y1]], 1)
    k = max(9, int(round(0.25 * ed)) | 1)
    fm = cv2.erode(fm, np.ones((k, k), np.uint8))
    for kk in ('pupil_l', 'pupil_r'):
        cv2.circle(fm, (int(sem[kk][0] - X1), int(sem[kk][1] - Y1)), int(0.4 * ed), 0, -1)
    mc = (int((sem['mouth_l'][0] + sem['mouth_r'][0]) / 2 - X1), int((sem['mouth_l'][1] + sem['mouth_r'][1]) / 2 - Y1))
    cv2.ellipse(fm, mc, (int(0.5 * ed), int(0.3 * ed)), 0, 0, 360, 0, -1)
    fm = fm.astype(bool)
    ag, bg = cv2.cvtColor(a_, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b_, cv2.COLOR_RGB2GRAY)
    fl = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(ag, bg, None)
    gx, gy = np.meshgrid(np.arange(X2 - X1, dtype=np.float32), np.arange(Y2 - Y1, dtype=np.float32))
    bal = cv2.remap(b_, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sa = cv2.GaussianBlur(a_.astype(np.float32), (0, 0), 3)[fm] / 255
    sb = cv2.GaussianBlur(bal.astype(np.float32), (0, 0), 3)[fm] / 255
    pick = rng.choice(len(sa), min(4000, len(sa)), replace=False)
    to = lambda x, c: cv2.cvtColor(x[pick].reshape(-1, 1, 3), c).reshape(-1, 3).astype(np.float64)
    skinA, skinB, skinH = to(sa, cv2.COLOR_RGB2LAB), to(sb, cv2.COLOR_RGB2LAB), to(sa, cv2.COLOR_RGB2HSV)
    ks = np.array(sorted(RETOUCH_FEMALE))
    vs = np.array([RETOUCH_FEMALE[k] for k in ks])
    off = np.stack([np.interp(skinA[:, 0], ks, vs[:, c]) for c in range(3)], 1)
    skinB = skinB - off
    print(f'皮肤样本 {len(skinA)}：原图中位 Lab {np.median(skinA, 0).round(2)}，develop 目标中位 {np.median(skinB, 0).round(2)}')
    return skinA, skinB, skinH


def skin_mask(A, model_path):
    import onnxruntime as ort
    H, W = A.shape[:2]
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    s = 800.0 / max(H, W)
    size = (int(W * s), 800) if H > W else (800, int(H * s))
    small = cv2.resize(A, size, interpolation=cv2.INTER_LINEAR)
    sk = cv2.resize(sess.run(None, {'input_image:0': small.astype(np.float32)})[0], (W, H)) > 10
    return cv2.dilate(sk.astype(np.uint8), np.ones((81, 81), np.uint8)) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('orig'); ap.add_argument('ref')
    ap.add_argument('--samples', type=int, default=160000)
    ap.add_argument('--skin-model', default='models/skin_seg.onnx')
    ap.add_argument('--out', default=None, help='保存拟合常数的 JSON')
    ap.add_argument('--faces', default=None, help='retouch detect 输出的 faces.json；与 --face 一起把该脸皮肤纳入拟合')
    ap.add_argument('--face', type=int, default=1)
    ap.add_argument('--skin-weight', type=float, default=0.25, help='皮肤样本占总权重的比例')
    a = ap.parse_args()
    S = parse_crs(a.orig)
    if S is None:
        raise SystemExit('原图没有 crs:HasSettings="True" 的 Camera Raw 设置')
    print('crs settings:', json.dumps(S, ensure_ascii=False))
    A, B = imread(a.orig), imread(a.ref)
    nonskin = ~skin_mask(A, a.skin_model)
    rng = np.random.default_rng(1)
    idx = rng.choice(np.flatnonzero(nonskin.ravel()), min(a.samples, int(nonskin.sum())), replace=False)
    Ab = cv2.GaussianBlur(A.astype(np.float32), (0, 0), 2).reshape(-1, 3)[idx] / 255
    Bb = cv2.GaussianBlur(B.astype(np.float32), (0, 0), 2).reshape(-1, 3)[idx] / 255
    labA = cv2.cvtColor(Ab.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    labB = cv2.cvtColor(Bb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    hsvA = cv2.cvtColor(Ab.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float64)
    key = (np.clip(labA[:, 0] // 10, 0, 9) * 12 + (hsvA[:, 0] // 30)).astype(int)
    cnt = np.bincount(key, minlength=120).astype(float)
    wt = 1 / np.sqrt(np.maximum(cnt[key], 1)); wt /= wt.mean()
    skinA, skinB, skinH = face_skin_samples(A, B, a, rng) if a.faces else (None, None, None)
    half = len(Ab) // 2
    kk = (np.clip((Ab * 255) // 16, 0, 15).astype(int) * [256, 16, 1]).sum(1)
    c1 = np.bincount(kk[:half], minlength=4096).astype(float)
    lutd = np.stack([np.bincount(kk[:half], weights=(labB - labA)[:half, c], minlength=4096) / np.maximum(c1, 1) for c in range(3)], 1)
    ok = c1[kk[half:]] >= 3
    ceil = np.sqrt((((labA[half:][ok] + lutd[kk[half:][ok]]) - labB[half:][ok]) ** 2 * wt[half:][ok, None]).sum(1).mean())
    base = np.sqrt(((labA - labB) ** 2 * wt[:, None]).sum(1).mean())
    def resid(p):
        r1 = ((model(labA[:half], p, S, hsvA[:half]) - labB[:half]) * np.sqrt(wt[:half])[:, None]).ravel()
        if skinA is None:
            return r1
        ws = np.sqrt(a.skin_weight / (1 - a.skin_weight) * half / len(skinA))
        return np.concatenate([r1, ((model(skinA, p, S, skinH) - skinB) * ws).ravel()])
    r = least_squares(resid, P0, bounds=(LB, UB), max_nfev=300)
    pred = model(labA[half:], r.x, S, hsvA[half:])
    if skinA is not None:
        ps = model(skinA, r.x, S, skinH)
        print(f'皮肤：develop 目标中位 dLab {np.median(skinB - skinA, 0).round(2)}，模型 {np.median(ps - skinA, 0).round(2)}')
    fit = np.sqrt(((pred - labB[half:]) ** 2 * wt[half:, None]).sum(1).mean())
    print(f'加权 RMS ΔE：原样 {base:.3f}，16³ LUT 上限 {ceil:.3f}，参数模型 {fit:.3f}（留出集）')
    for n_, v in zip(PN, r.x):
        print(f'   {n_:14s} {v:9.4f}')
    if a.out:
        json.dump(dict(zip(PN, map(float, r.x))), open(a.out, 'w', encoding='utf-8'), indent=1)
    lA, lB = labA[half:], labB[half:]
    hueL = (np.degrees(np.arctan2(lA[:, 2], lA[:, 1])) + 360) % 360
    C = np.hypot(lA[:, 1], lA[:, 2])
    d0, d1 = lB - lA, lB - pred
    print('\n  分箱                 n      导出−原图 (dL,da,db)       导出−模型 (dL,da,db)')
    for lo in range(0, 360, 20):
        sel = (C >= 6) & (hueL >= lo) & (hueL < lo + 20)
        if sel.sum() > 300:
            print(f'  色相 {lo:3d}-{lo+20:<3d} {sel.sum():7d}  ({d0[sel,0].mean():+5.2f},{d0[sel,1].mean():+5.2f},{d0[sel,2].mean():+5.2f})   ({d1[sel,0].mean():+5.2f},{d1[sel,1].mean():+5.2f},{d1[sel,2].mean():+5.2f})')
    for lo in range(0, 100, 10):
        for nm, sel0 in [('中性', C < 6), ('有彩', C >= 6)]:
            sel = sel0 & (lA[:, 0] >= lo) & (lA[:, 0] < lo + 10)
            if sel.sum() > 300:
                print(f'  {nm} L {lo:2d}-{lo+10:<3d} {sel.sum():7d}  ({d0[sel,0].mean():+5.2f},{d0[sel,1].mean():+5.2f},{d0[sel,2].mean():+5.2f})   ({d1[sel,0].mean():+5.2f},{d1[sel,1].mean():+5.2f},{d1[sel,2].mean():+5.2f})')


if __name__ == '__main__':
    main()
