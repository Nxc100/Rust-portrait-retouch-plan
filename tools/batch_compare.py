#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量对比：一批原片、参考导出（如像素蛋糕）与本项目输出，逐张、逐脸统计并汇总差距。

    python tools/batch_compare.py --orig test/原片 --ref "test/像素蛋糕“奶油肌”预设产物" \\
        --ours out/batch --landmarks out/batch/_landmarks [--md metrics.md] [--json metrics.json]

按文件名配对（不含扩展名、不区分大小写）；关键点来自 `retouch batch --landmarks-dir`（或 `retouch detect --json`）。
每张脸（参考 | 本项目，差值 = 本项目 − 参考）：
- 纹理：中频（0.005–0.035 瞳距）按局部能量分箱的能量比，对像素占比 ≥ 5% 的箱求加权差（正 = 本项目留的纹理更多）；
  细颗粒 / 低频的合计比（texture_energy.py）；
- 色调：皮肤区按原图 L 分箱（50–90）的 Lab 偏移之差、全脸平均偏移之差（compare_to_reference.py）；
- 色度：a/b 前四个带通比的平均差；低频色度统一斜率之差；
- 几何：下颌 jaw_l0/l1/l2 的 x 位移与下巴 y 位移之差（% 瞳距）。
另统计整张图"被改动"的面积占比（|Δ| > 6/255，σ2 平滑后）：参考与本项目都应只改人物皮肤。
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
import texture_energy as te  # noqa: E402

IMG_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp')


def index_dir(d):
    out = {}
    for p in glob.glob(os.path.join(d, '*')):
        stem, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() in IMG_EXT:
            out[stem.lower()] = p
    return out


def changed_area(A, B, thr=6.0):
    d = np.abs(A.astype(np.float32) - B.astype(np.float32)).max(2)
    return float((cv2.GaussianBlur(d, (0, 0), 2) > thr).mean() * 100)


def mid_bias(ref, ours, min_frac=0.05):
    """中频按能量分箱：对两边都有、占比 ≥ min_frac 的箱，求（本项目 − 参考）的加权平均。"""
    r = {(e0, e1): (v, fr) for e0, e1, v, fr in ref['mid']['bins']}
    o = {(e0, e1): (v, fr) for e0, e1, v, fr in ours['mid']['bins']}
    num = den = 0.0
    for k, (vr, fr) in r.items():
        if k in o and fr >= min_frac:
            num += (o[k][0] - vr) * fr
            den += fr
    return num / den if den else float('nan')


def tone_diff(ref, ours, lo=50, hi=90):
    keys = [k for k in ref['bins'] if k.isdigit() and lo <= int(k) < hi and k in ours['bins']]
    if not keys:
        return (float('nan'),) * 3
    d = np.array([np.subtract(ours['bins'][k], ref['bins'][k]) for k in keys])
    return tuple(float(v) for v in d.mean(0))


def face_metrics(A, B, C, f):
    (X1, Y1, X2, Y2), m = ctr.face_region(A, f)
    a = A[Y1:Y2, X1:X2]
    cr = ctr.measure(a, B[Y1:Y2, X1:X2], m, f, (X1, Y1))
    co = ctr.measure(a, C[Y1:Y2, X1:X2], m, f, (X1, Y1))
    tr = te.measure_face(A, B, f)
    to = te.measure_face(A, C, f)
    sh = lambda res, k, i: res['shifts'].get(k, (float('nan'),) * 2)[i]
    jaw = float(np.nanmean([sh(co, k, 0) - sh(cr, k, 0) for k in ('jaw_l0', 'jaw_l1', 'jaw_l2')]))
    return {
        'gender': f.get('gender'), 'eye_distance': f['eye_distance'], 'yaw': f.get('yaw_deg'),
        'mid_bias': mid_bias(tr, to),
        'fine': (tr['fine']['total'], to['fine']['total']),
        'low': (tr['low']['total'], to['low']['total']),
        'L_bands': (cr['L'], co['L']),
        'ab_band_diff': float(np.mean([np.mean(np.subtract(co[c][:4], cr[c][:4])) for c in ('a', 'b')])),
        'tone_diff_50_90': tone_diff(cr, co),
        'mean_dlab': (cr['mean'], co['mean']),
        'slopes': ((cr['slope_a'], cr['slope_b']), (co['slope_a'], co['slope_b'])),
        'jaw_x_diff': jaw,
        'chin_y_diff': sh(co, 'chin', 1) - sh(cr, 'chin', 1),
        'texture_ref': tr, 'texture_ours': to, 'compare_ref': cr, 'compare_ours': co,
    }


def fmt(v, n=2, sign=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:+.{n}f}' if sign else f'{v:.{n}f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--orig', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--ours', required=True)
    ap.add_argument('--landmarks', required=True)
    ap.add_argument('--md', default=None, help='把汇总表写成 Markdown')
    ap.add_argument('--json', default=None, help='把全部指标写成 JSON')
    a = ap.parse_args()
    orig, ref, ours = index_dir(a.orig), index_dir(a.ref), index_dir(a.ours)
    rows, results = [], {}
    for stem in sorted(orig):
        if stem not in ref or stem not in ours:
            print(f'skip {stem}: 缺少参考或本项目输出', file=sys.stderr)
            continue
        lm = [p for p in glob.glob(os.path.join(a.landmarks, '*.json')) if os.path.splitext(os.path.basename(p))[0].lower() == stem]
        if not lm:
            print(f'skip {stem}: 缺少关键点 JSON', file=sys.stderr)
            continue
        A, B, C = ctr.imread(orig[stem]), ctr.imread(ref[stem]), ctr.imread(ours[stem])
        if not (A.shape == B.shape == C.shape):
            print(f'skip {stem}: 尺寸不一致 {A.shape} {B.shape} {C.shape}', file=sys.stderr)
            continue
        faces = json.load(open(lm[0], encoding='utf-8'))['faces']
        name = os.path.basename(orig[stem])
        entry = {'changed_area_pct': (changed_area(A, B), changed_area(A, C)), 'faces': []}
        print(f'== {name}: {len(faces)} face(s), 改动面积 参考 {entry["changed_area_pct"][0]:.1f}% | 本项目 {entry["changed_area_pct"][1]:.1f}%', file=sys.stderr)
        for fi, f in enumerate(faces):
            try:
                mtr = face_metrics(A, B, C, f)
            except Exception as e:  # 极小 / 贴边的脸
                print(f'   face {fi}: 跳过（{e}）', file=sys.stderr)
                continue
            entry['faces'].append(mtr)
            rows.append((name, fi, mtr))
            print(f'   face {fi} {mtr["gender"]} ed {mtr["eye_distance"]:.0f}: 中频偏差 {fmt(mtr["mid_bias"])} '
                  f'细颗粒 {mtr["fine"][0]:.2f}|{mtr["fine"][1]:.2f} 低频 {mtr["low"][0]:.2f}|{mtr["low"][1]:.2f} '
                  f'色调差(50–90) {"/".join(fmt(v, 1) for v in mtr["tone_diff_50_90"])} 色度带差 {fmt(mtr["ab_band_diff"])} '
                  f'下颌 x 差 {fmt(mtr["jaw_x_diff"], 1)} 下巴 y 差 {fmt(mtr["chin_y_diff"], 1)}', file=sys.stderr)
        results[name] = entry
    if not rows:
        sys.exit('没有可对比的照片')
    # 汇总
    agg = lambda key, fn=lambda m: m: np.nanmean([fn(m[key]) for _, _, m in rows])
    summary = {
        'faces': len(rows),
        'mid_bias_mean': float(np.nanmean([m['mid_bias'] for _, _, m in rows])),
        'mid_bias_mae': float(np.nanmean([abs(m['mid_bias']) for _, _, m in rows])),
        'fine_diff_mean': float(np.nanmean([m['fine'][1] - m['fine'][0] for _, _, m in rows])),
        'low_diff_mean': float(np.nanmean([m['low'][1] - m['low'][0] for _, _, m in rows])),
        'tone_diff_mean': [float(v) for v in np.nanmean([m['tone_diff_50_90'] for _, _, m in rows], 0)],
        'tone_diff_mae': [float(v) for v in np.nanmean([np.abs(m['tone_diff_50_90']) for _, _, m in rows], 0)],
        'ab_band_diff_mean': float(np.nanmean([m['ab_band_diff'] for _, _, m in rows])),
        'jaw_x_diff_mean': float(np.nanmean([m['jaw_x_diff'] for _, _, m in rows])),
        'chin_y_diff_mean': float(np.nanmean([m['chin_y_diff'] for _, _, m in rows])),
    }
    del agg
    lines = ['| 照片 | 脸 | 性别 | 瞳距 | 中频偏差 | 细颗粒 参考/本项目 | 低频 参考/本项目 | 色调差 dL/da/db（L 50–90） | 色度带差 | 下颌 x 差 | 下巴 y 差 |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, fi, m in rows:
        lines.append(f'| {name} | {fi} | {m["gender"] or "?"} | {m["eye_distance"]:.0f} | {fmt(m["mid_bias"])} | '
                     f'{m["fine"][0]:.2f} / {m["fine"][1]:.2f} | {m["low"][0]:.2f} / {m["low"][1]:.2f} | '
                     f'{" / ".join(fmt(v, 1) for v in m["tone_diff_50_90"])} | {fmt(m["ab_band_diff"])} | '
                     f'{fmt(m["jaw_x_diff"], 1)} | {fmt(m["chin_y_diff"], 1)} |')
    s = summary
    lines.append(f'| **平均**（{s["faces"]} 张脸） | | | | {fmt(s["mid_bias_mean"])}（绝对 {s["mid_bias_mae"]:.2f}） | 差 {fmt(s["fine_diff_mean"])} | 差 {fmt(s["low_diff_mean"])} | '
                 f'{" / ".join(fmt(v, 1) for v in s["tone_diff_mean"])}（绝对 {" / ".join(f"{v:.1f}" for v in s["tone_diff_mae"])}） | {fmt(s["ab_band_diff_mean"])} | '
                 f'{fmt(s["jaw_x_diff_mean"], 1)} | {fmt(s["chin_y_diff_mean"], 1)} |')
    table = '\n'.join(lines)
    area = '\n'.join(f'- {n}: 改动面积 参考 {e["changed_area_pct"][0]:.1f}% ，本项目 {e["changed_area_pct"][1]:.1f}%' for n, e in results.items())
    print(table)
    print(area)
    if a.md:
        with open(a.md, 'w', encoding='utf-8') as fh:
            fh.write(table + '\n\n' + area + '\n')
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump({'summary': summary, 'images': results}, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
