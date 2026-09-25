#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""拟合奶油肌"色调"环节的常数（src/skin/cream.rs 第 5 步），数据为参考导出图（像素蛋糕）相对原图的皮肤低频色变。

    python tools/skin_tone_fit.py <pairs.json> [--min-ed 60] [--samples 3000]

pairs.json：[{"orig": 原图, "ref": 参考导出, "faces": 关键点 JSON（retouch detect --json / batch --landmarks-dir）}, ...]
每张脸：皮肤遮罩（与 compare_to_reference.py 相同）内，把参考图光流对齐到原图，取 σ = 0.03 瞳距的低频 Lab，
随机采样像素，按性别分别做线性最小二乘：

    dL = lift · exp(−((L − 62)/26)²) − hl · t_hl + lift_per_b · b
    da = −pull_a · (a − target_a)
    db = −pull_b · (b − target_b) + y_br · t_br + y_hl · t_hl + y_sh · t_sh

t_hl：该脸 L 的 p90–p99.5 之间的 smoothstep（最高光）；t_br = smoothstep((L − 50)/35)（亮部）；
t_sh = clamp((55 − L)/12)（阴影）。与 cream.rs 的实现一一对应。
lift_per_b：像素蛋糕的提亮与皮肤的黄度成正比（在 RGB 里降黄会抬高亮度）；加入该项后逐脸 dL 残差减半。
输出各性别的常数、R² 与逐脸的残差均值（检验"向目标色拉"的模型是否对每张脸都成立）。
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_to_reference as ctr  # noqa: E402


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def face_samples(A, B, f, n, rng):
    (X1, Y1, X2, Y2), m = ctr.face_region(A, f)
    a, b = A[Y1:Y2, X1:X2], B[Y1:Y2, X1:X2]
    ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_al, _ = ctr.flow_align(ag, cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), b)
    s = max(0.03 * f['eye_distance'], 1.0)
    la = cv2.cvtColor(cv2.GaussianBlur(a.astype(np.float32) / 255, (0, 0), s), cv2.COLOR_RGB2LAB)
    lb = cv2.cvtColor(cv2.GaussianBlur(b_al.astype(np.float32) / 255, (0, 0), s), cv2.COLOR_RGB2LAB)
    idx = np.flatnonzero(m.ravel())
    if len(idx) < 500:
        return None
    L_face = la[..., 0].ravel()[idx]
    p90, p995 = np.percentile(L_face, 90), max(np.percentile(L_face, 99.5), np.percentile(L_face, 90) + 1)
    pick = rng.choice(idx, min(n, len(idx)), replace=False)
    x = la.reshape(-1, 3)[pick].astype(np.float64)
    d = (lb.reshape(-1, 3)[pick] - la.reshape(-1, 3)[pick]).astype(np.float64)
    L = x[:, 0]
    feats = {
        'bell': np.exp(-((L - 62) / 26) ** 2),
        't_hl': smoothstep((L - p90) / (p995 - p90)),
        't_br': smoothstep((L - 50) / 35),
        't_sh': np.clip((55 - L) / 12, 0, 1),
    }
    return x, d, feats


def lstsq(cols, y):
    X = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return coef, pred, r2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pairs')
    ap.add_argument('--min-ed', type=float, default=60)
    ap.add_argument('--samples', type=int, default=3000)
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    data = {'female': [], 'male': []}
    for pr in json.load(open(a.pairs, encoding='utf-8')):
        A, B = ctr.imread(pr['orig']), ctr.imread(pr['ref'])
        for fi, f in enumerate(json.load(open(pr['faces'], encoding='utf-8'))['faces']):
            g = pr.get('genders', {}).get(str(fi)) or f.get('gender')
            if g not in data or f['eye_distance'] < a.min_ed:
                continue
            s = face_samples(A, B, f, a.samples, rng)
            if s is not None:
                data[g].append((f'{os.path.basename(pr["orig"])}#{fi}', *s))
    result = {}
    for g, faces in data.items():
        if not faces:
            continue
        x = np.concatenate([v[1] for v in faces]); d = np.concatenate([v[2] for v in faces])
        F = {k: np.concatenate([v[3][k] for v in faces]) for k in faces[0][3]}
        one = np.ones(len(x))
        cL, pL, r2L = lstsq([F['bell'], -F['t_hl'], x[:, 2]], d[:, 0])
        ca, pa, r2a = lstsq([one, x[:, 1]], d[:, 1])
        cb, pb, r2b = lstsq([one, x[:, 2], F['t_br'], F['t_hl'], F['t_sh']], d[:, 2])
        pull_a, pull_b = -ca[1], -cb[1]
        res = dict(midtone_lift=cL[0], highlight_suppress=cL[1], lift_per_b=cL[2], pull_a=pull_a, target_a=ca[0] / pull_a,
                   pull_b=pull_b, target_b=cb[0] / pull_b, yellow_bright=cb[2], yellow_highlight=cb[3], yellow_shadow=cb[4])
        result[g] = {k: round(float(v), 3) for k, v in res.items()}
        print(f'== {g}: {len(faces)} 张脸, {len(x)} 个样本；R² dL {r2L:.2f} da {r2a:.2f} db {r2b:.2f}')
        for k, v in result[g].items():
            print(f'   {k:20s} {v:+.3f}')
        print('   逐脸：参考均值 (dL,da,db) → 模型残差均值')
        o = 0
        for name, xs, ds, _ in faces:
            n = len(xs)
            res_ = ds.mean(0) - np.array([pL[o:o + n].mean(), pa[o:o + n].mean(), pb[o:o + n].mean()])
            print(f'   {name:22s} ({ds[:, 0].mean():+.2f},{ds[:, 1].mean():+.2f},{ds[:, 2].mean():+.2f}) → '
                  f'({res_[0]:+.2f},{res_[1]:+.2f},{res_[2]:+.2f})')
            o += n
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
