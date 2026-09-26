#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""拟合奶油肌"身体色调"的常数（src/skin/cream.rs 第 5 步的身体分支），数据为参考导出图（像素蛋糕）相对原图的
身体皮肤低频色变。

    python tools/body_tone_fit.py --orig test/原片 --ref <参考导出目录> --debug <调试目录> [--ours <输出目录>] [--fit-shape]

--debug：每张照片一个子目录 `<照片名>/`，内含 `retouch apply --preset cream --debug-dir` 导出的 `skin_body.png` 与
`landmarks.json`（身体遮罩与人脸）。模型（L、b 为 σ = 0.03 瞳距当量的低频亮度与黄度，与 cream.rs 一致）：

    dL = lift · h(L)       h(L) = [0.8 + 0.2·S(20, 44, L)] · [1 − 0.9·S(64, 88, L)]（S 为 smoothstep）
    da = redness
    db = −pull_b · (b − target_b) · clamp((L − 30) / 30, 0, 1)

降黄是按比例向目标黄度拉近：偏黄的皮肤降得多，天光下偏蓝的皮肤（b 接近 0）反而略加暖（像素蛋糕的女性身体：
b −1.5 → +1.9，b 21 → −3.6）；同时报告常数模型 db = yellow · clamp(...) 的残差作对照。

身体像素按到各张脸中心的距离（以该脸的尺度 max(瞳距, 鼻梁顶–下巴 / 2) 为单位）软分配给各人，
权重 ∝ exp(−d / 0.5)，与 cream.rs 的 `owner_weight` / `body_tone_at` 相同；拟合时只用归属明确（权重 > 0.8）的像素，按性别分别求
lift / redness / yellow 的最小二乘解。--fit-shape 另在全部像素上网格搜索 h 的形状。
--ours 给出本项目的输出目录时，额外报告它相对参考图的逐张 / 逐性别偏差与 RMS。
--heatmap 输出身体皮肤上的亮度变化热图（每张照片一列：参考图 − 原图，给出 --ours 时下一行为输出 − 原图；
绿 = 0，黄 ≈ +3，红 ≈ +6，身体遮罩外为灰度原图）。
"""
import argparse
import glob
import itertools
import json
import os

import cv2
import numpy as np

SHAPE = dict(knee=64.0, top=88.0, keep=0.1, shadow=0.8)
SOFTNESS = 0.5
MIN_OWNER = 0.8


def imread(p):
    return cv2.cvtColor(cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def lab(p, step):
    return cv2.cvtColor(imread(p).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[::step, ::step]


def ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def lift_shape(L, knee, top, keep, shadow):
    return (shadow + (1 - shadow) * ss(20, 44, L)) * (1 - (1 - keep) * ss(knee, top, L))


def yellow_shape(L):
    return np.clip((L - 30) / 30, 0, 1)


def load_faces(path):
    d = json.load(open(path, encoding='utf-8'))
    faces = []
    for f in (d if isinstance(d, list) else d['faces']):
        s = f['semantic']
        scale = max(f['eye_distance'], float(np.hypot(*np.subtract(s['nose_bridge_top'], s['chin']))) / 2)
        x1, y1, x2, y2 = f['bbox']
        faces.append(dict(center=((x1 + x2) / 2, (y1 + y2) / 2), scale=scale, gender=(f.get('gender') or 'unknown')))
    return faces


def owner_weights(xs, ys, faces):
    """(N, 人数) 的归属权重（软最近人脸，与 cream.rs 一致）。"""
    d = np.stack([np.hypot(xs - f['center'][0], ys - f['center'][1]) / f['scale'] for f in faces], 1)
    w = np.exp(-(d - d.min(1, keepdims=True)) / SOFTNESS)
    return w / w.sum(1, keepdims=True)


def collect(orig, ref, dbg, ours=None):
    """一张照片的身体像素：低频 L 与各通道的低频变化、归属权重。"""
    faces = load_faces(os.path.join(dbg, 'landmarks.json'))
    ed = float(np.mean([f['scale'] for f in faces]))
    step = max(1, int(round(0.02 * ed)))
    A, R = lab(orig, step), lab(ref, step)
    O = lab(ours, step) if ours else None
    mask = cv2.imdecode(np.fromfile(os.path.join(dbg, 'skin_body.png'), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    M = cv2.erode((mask[::step, ::step] > 240).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    sigma = 0.03 * ed / step
    lo = lambda x: cv2.GaussianBlur(x, (0, 0), sigma)
    yy, xx = np.nonzero(M)
    out = dict(
        L=lo(A[..., 0])[yy, xx],
        b=lo(A[..., 2])[yy, xx],
        d=np.stack([lo(R[..., c] - A[..., c])[yy, xx] for c in range(3)], 1),
        w=owner_weights(xx * step, yy * step, faces),
        genders=[f['gender'] for f in faces],
    )
    if O is not None:
        out['ours'] = np.stack([lo(O[..., c] - A[..., c])[yy, xx] for c in range(3)], 1)
    return out


def by_gender(rows):
    """按归属明确的性别分组：{性别: [(照片, 下标)]}。"""
    groups = {}
    for name, r in rows.items():
        main = r['w'].argmax(1)
        clear = r['w'].max(1) > MIN_OWNER
        for i, g in enumerate(r['genders']):
            sel = np.flatnonzero(clear & (main == i))
            if len(sel):
                groups.setdefault(g, []).append((name, sel))
    return groups


def heatmap(jobs, out, long_side=700):
    """jobs: [(照片名, 原图, 参考图, 输出或 None, 身体遮罩)] → 亮度变化热图拼图。"""
    from PIL import Image, ImageDraw
    from montage import font

    def panel(orig, other, mask, label):
        A, B = imread(orig), imread(other)
        s = max(A.shape[:2]) / long_side
        size = (int(A.shape[1] / s), int(A.shape[0] / s))
        dl = lambda im: cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]
        d = cv2.resize(cv2.GaussianBlur(dl(B) - dl(A), (0, 0), s), size, interpolation=cv2.INTER_AREA)
        m = cv2.resize(cv2.imdecode(np.fromfile(mask, dtype=np.uint8), cv2.IMREAD_GRAYSCALE), size) > 128
        heat = cv2.applyColorMap((np.clip((d + 8) / 16, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        gray = (cv2.resize(A, size, interpolation=cv2.INTER_AREA).mean(2, keepdims=True) * 0.6).repeat(3, 2)
        im = Image.fromarray(np.where(m[..., None], heat, gray).astype(np.uint8))
        ImageDraw.Draw(im).text((10, 8), label, fill=(255, 255, 255), font=font(22))
        return im

    columns = []
    for name, orig, ref, ours, mask in jobs:
        col = [panel(orig, ref, mask, f'{name} 参考')]
        if ours:
            col.append(panel(orig, ours, mask, f'{name} 本程序'))
        columns.append(col)
    gap = 8
    width = sum(c[0].width for c in columns) + gap * (len(columns) - 1)
    height = sum(p.height for p in columns[0]) + gap * (len(columns[0]) - 1)
    canvas = Image.new('RGB', (width, height), 'white')
    x = 0
    for col in columns:
        y = 0
        for p in col:
            canvas.paste(p, (x, y))
            y += p.height + gap
        x += col[0].width + gap
    canvas.save(out)
    print(f'热图：{out} {canvas.size}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--orig', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--debug', required=True, help='retouch apply --debug-dir 的输出，每张照片一个子目录')
    ap.add_argument('--ours', default=None, help='本项目的输出目录（评估用）')
    ap.add_argument('--fit-shape', action='store_true', help='网格搜索提亮随亮度的形状')
    ap.add_argument('--heatmap', default=None, help='输出身体亮度变化热图（PNG）')
    args = ap.parse_args()

    rows, heat_jobs = {}, []
    for dbg in sorted(glob.glob(os.path.join(args.debug, '*', 'skin_body.png'))):
        name = os.path.basename(os.path.dirname(dbg))
        orig = glob.glob(os.path.join(args.orig, name + '.*'))
        ref = glob.glob(os.path.join(args.ref, name + '.*'))
        ours = glob.glob(os.path.join(args.ours, name + '.*')) if args.ours else [None]
        if not orig or not ref or not ours:
            print(f'跳过 {name}：缺少原图 / 参考图 / 输出')
            continue
        rows[name] = collect(orig[0], ref[0], os.path.dirname(dbg), ours[0])
        heat_jobs.append((name, orig[0], ref[0], ours[0], dbg))
        print(f'{name}: 身体像素 {len(rows[name]["L"])}，人物 {rows[name]["genders"]}')
    if not rows:
        raise SystemExit('没有可用的数据')
    if args.heatmap:
        heatmap(heat_jobs, args.heatmap)

    shape = dict(SHAPE)
    if args.fit_shape:
        L = np.concatenate([r['L'] for r in rows.values()])
        Y = np.concatenate([r['d'][:, 0] for r in rows.values()])
        best = None
        for knee, top, keep, sh in itertools.product([52, 56, 60, 64, 68], [80, 84, 88, 92, 96],
                                                     [0.0, 0.1, 0.2, 0.3], [0.6, 0.7, 0.8, 0.9, 1.0]):
            x = lift_shape(L, knee, top, keep, sh)
            k = (x * Y).sum() / (x * x).sum()
            err = ((k * x - Y) ** 2).mean()
            if best is None or err < best[0]:
                best = (err, dict(knee=knee, top=top, keep=keep, shadow=sh))
        shape = best[1]
        print(f'\n形状（全部像素）：{shape}，RMS {np.sqrt(best[0]):.2f}')

    print('\n按性别拟合（只用归属权重 > %.1f 的像素）' % MIN_OWNER)
    for g, parts in sorted(by_gender(rows).items()):
        L = np.concatenate([rows[n]['L'][s] for n, s in parts])
        b = np.concatenate([rows[n]['b'][s] for n, s in parts])
        d = np.concatenate([rows[n]['d'][s] for n, s in parts])
        h, y = lift_shape(L, **shape), yellow_shape(L)
        lift = (h * d[:, 0]).sum() / (h * h).sum()
        yellow = (y * d[:, 2]).sum() / max((y * y).sum(), 1e-9)
        # db = (−pull·b + pull·target)·y：对 [b·y, y] 的线性最小二乘
        (c_b, c_1), *_ = np.linalg.lstsq(np.stack([b * y, y], 1), d[:, 2], rcond=None)
        pull, target = -c_b, c_1 / -c_b if abs(c_b) > 1e-9 else 0.0
        red = d[:, 1].mean()
        rms = lambda e: np.sqrt((e ** 2).mean())
        print(f'  {g}: 像素 {len(L)}  lift {lift:.2f}（RMS {rms(lift * h - d[:, 0]):.2f}）'
              f'  redness {red:+.2f}（RMS {rms(red - d[:, 1]):.2f}）'
              f'  pull_b {pull:.3f} target_b {target:+.1f}（RMS {rms(-pull * (b - target) * y - d[:, 2]):.2f}；'
              f'常数 yellow {yellow:+.2f} 的 RMS {rms(yellow * y - d[:, 2]):.2f}）')
        for n, s in parts:
            e = lift * lift_shape(rows[n]['L'][s], **shape) - rows[n]['d'][s, 0]
            print(f'      {n}: 像素 {len(s):6d}  dL 残差 {e.mean():+.2f} / RMS {rms(e):.2f}')

    if args.ours:
        print('\n本项目输出相对参考图（低频变化之差：偏差 / RMS）')
        tot = {}
        for g, parts in sorted(by_gender(rows).items()):
            for n, s in parts:
                e = rows[n]['ours'][s] - rows[n]['d'][s]
                tot.setdefault(g, []).append(e)
                print(f'  {n:12s} {g:7s} 像素 {len(s):6d}  ' + '  '.join(
                    f'd{c} {e[:, i].mean():+5.2f}/{np.sqrt((e[:, i] ** 2).mean()):4.2f}' for i, c in enumerate('Lab')))
        for g, es in sorted(tot.items()):
            e = np.concatenate(es)
            print(f'  合计 {g:7s} ' + '  '.join(
                f'd{c} {e[:, i].mean():+5.2f}/{np.sqrt((e[:, i] ** 2).mean()):4.2f}' for i, c in enumerate('Lab')))


if __name__ == '__main__':
    main()
