#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""拟合预设调色（src/color/grade.rs）：自适应黑点 → 全局 3D LUT → 人物主体的亮度 / 色度调整。
数据为参考导出图（如像素蛋糕「婚纱-深色内景」）相对原图的颜色映射。

    python tools/grade_fit.py --orig test/原片 --ref <参考导出目录> --debug <调试目录> \\
        [--extra <原图>::<参考图>::<调试子目录> ...] [--size 33] [--cube luts/<名称>.cube] [--loo] [--search-black]

--debug：每张照片一个子目录 `<照片名>/`，内含 `retouch apply --debug-dir` 导出的 `matte.png`（人像 alpha）、
`skin_union.png`（皮肤）与 `landmarks.json`（人脸，用于排除脸型变形区）。
模型（颜色为 sRGB 0..1；与 grade.rs 的对应关系见各条）：
  1. 黑点：b = strength · max(0, P − target)，软黑点 x' = (x − b·s(x)) / (1 − b)，s = smoothstep(0, 2b, x)
     （逐通道，截到 0..1：亮部同硬黑点，暗部不截断）。P 为整图 min(R, G, B) 的 q 分位（8 位直方图，至少 1 个像素，
     与 grade.rs 相同）——没有真正暗部的朦胧 / 高调画面（阴天海滩、雾中山水）被压暗、加对比，有纯黑的画面不动。
     这里在缩小到长边 1000 px 的图上逐像素做（纹理已被平均掉，相当于底图）；grade.rs 在全分辨率上只对保边底图做、
     细节原样加回，二者在底图层面一致。--black-subject 为主体内使用黑点的比例（分析用；留一误差在 1 时最低，
     grade.rs 按 1 实现：人物与背景同样压暗，不写入预设）；
  2. 全局 3D LUT（N³，三线性）：在背景像素上做带平滑正则（三个方向的二阶差分）的加权最小二乘，各照片等权，
     离样本远的颜色由平滑外推、并弱约束到恒等；
  3. 主体：Lab 中 L' = L + s(L)、色度 × (1 + k)，s 为 L = 0, 20, …, 100 处取值的分段线性曲线，在人像 alpha > 0.9
     的非皮肤像素上（LUT 之后的残差）拟合。grade.rs 把"LUT + 主体调整"烘焙成第二张 LUT，按 alpha 与全局 LUT 的
     结果在 sRGB 上混合（alpha = 1 处与这里相同）。
区域：背景 = alpha < 0.05、主体 = alpha > 0.9，都离人像边缘 ≥ 1.5% 对角线（躯干、四肢的形变区），
都去掉皮肤（外扩）与人脸框的 1.6 倍（脸型变形区）。
--loo：逐张留一（其余照片拟合、这一张评估）报告背景与主体的 ΔE（CIE76）中位数；--search-black 网格搜索黑点参数。
"""
import argparse
import glob
import itertools
import json
import os
import sys

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

FIT_LONG_SIDE = 1000       # 拟合用的缩小尺寸（长边像素）
SAMPLES_PER_PHOTO = 60000  # 每张照片每个区域的样本数上限
SUBJECT_KNOTS = np.linspace(0.0, 100.0, 6)
IMG_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp')


def imread(p, flags=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), flags)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if flags == cv2.IMREAD_COLOR else img


def rgb_to_lab(rgb01):
    shp = rgb01.shape
    return cv2.cvtColor(rgb01.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(shp)


def lab_to_rgb(lab):
    shp = lab.shape
    return cv2.cvtColor(lab.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_Lab2RGB).reshape(shp)


# ---- 1. 自适应黑点 ----

def black_statistic(rgb_u8, q):
    """整图 min(R, G, B) 的 q 分位（0..1），按 8 位直方图取（与 grade.rs 相同）。"""
    hist = np.bincount(rgb_u8.min(axis=2).ravel(), minlength=256)
    cdf = np.cumsum(hist)
    return float(np.searchsorted(cdf, max(q / 100.0 * cdf[-1], 1.0))) / 255.0


def black_point(stat, strength, target):
    return strength * max(0.0, stat - target)


def apply_black(rgb01, b):
    """软黑点（与 grade.rs 的 soft_black 相同）。"""
    if b <= 0:
        return rgb01
    t = np.clip(rgb01 / (2.0 * b), 0.0, 1.0)
    return np.clip((rgb01 - b * t * t * (3.0 - 2.0 * t)) / (1.0 - b), 0.0, 1.0)


def region_black(p, cfg, region):
    """这张照片在某区域（背景 m ≈ 0、主体 m ≈ 1）上实际使用的黑点。"""
    b = black_point(p['stat_q'][cfg['q']], cfg['strength'], cfg['target'])
    return b * cfg['black_subject'] if region == 'subject' else b


# ---- 2. 平滑 3D LUT ----

def trilinear(x, n):
    """样本（K×3，0..1）的三线性权重：返回（K×8 的格点索引，K×8 的权重）；索引 = r + n·(g + n·b)。"""
    f = np.clip(x, 0.0, 1.0) * (n - 1)
    i0 = np.minimum(np.floor(f).astype(np.int64), n - 2)
    t = f - i0
    idx, wts = [], []
    for dr, dg, db in itertools.product((0, 1), repeat=3):
        idx.append((i0[:, 0] + dr) + n * ((i0[:, 1] + dg) + n * (i0[:, 2] + db)))
        wts.append(np.where(dr, t[:, 0], 1 - t[:, 0]) * np.where(dg, t[:, 1], 1 - t[:, 1])
                   * np.where(db, t[:, 2], 1 - t[:, 2]))
    return np.stack(idx, 1), np.stack(wts, 1)


def second_differences(n):
    """三个方向的二阶差分算子（稀疏），用于平滑正则。"""
    blocks = []
    grid = np.arange(n ** 3).reshape(n, n, n)  # [b, g, r]
    for axis in range(3):
        a = np.moveaxis(grid, axis, 0)
        c0, c1, c2 = a[:-2].ravel(), a[1:-1].ravel(), a[2:].ravel()
        rows = np.arange(len(c0))
        blocks.append(sp.csr_matrix((np.concatenate([np.ones_like(rows), -2 * np.ones_like(rows), np.ones_like(rows)]),
                                     (np.concatenate([rows, rows, rows]), np.concatenate([c0, c1, c2]))),
                                    shape=(len(c0), n ** 3)))
    return sp.vstack(blocks).tocsr()


def lut_identity(n):
    g = np.linspace(0.0, 1.0, n)
    b, gg, r = np.meshgrid(g, g, g, indexing='ij')
    return np.stack([r.ravel(), gg.ravel(), b.ravel()], 1)


def fit_lut(xs, ys, ws, n, smooth, prior_weight=1e-4):
    """加权最小二乘：Σ w·|A v − y|² + smooth·|D v|² + prior·|v − 恒等|²（逐通道），CG 求解。"""
    idx, wts = trilinear(xs, n)
    rows = np.repeat(np.arange(len(xs)), 8)
    A = sp.csr_matrix((wts.ravel(), (rows, idx.ravel())), shape=(len(xs), n ** 3))
    W = sp.diags(ws)
    D = second_differences(n)
    ident = lut_identity(n)
    M = (A.T @ W @ A + smooth * (D.T @ D) + prior_weight * sp.identity(n ** 3)).tocsr()
    diag = M.diagonal()
    pre = sp.diags(1.0 / np.maximum(diag, 1e-12))
    out = np.empty((n ** 3, 3))
    for c in range(3):
        rhs = A.T @ (ws * ys[:, c]) + prior_weight * ident[:, c]
        v, info = spla.cg(M, rhs, x0=ident[:, c], rtol=1e-7, maxiter=5000, M=pre)
        if info != 0:
            print(f'  警告：CG 未收敛（通道 {c}，info {info}）', file=sys.stderr)
        out[:, c] = v
    return np.clip(out, 0.0, 1.0)


def apply_lut(lut, x, n):
    idx, wts = trilinear(x, n)
    return (lut[idx] * wts[..., None]).sum(1)


# ---- 3. 主体 ----

def subject_basis(L):
    """分段线性曲线的基函数（K 个节点处的"帽子"函数）。"""
    k = SUBJECT_KNOTS
    L = np.clip(L, k[0], k[-1])
    out = np.zeros((len(L), len(k)))
    seg = np.clip(np.searchsorted(k, L, side='right') - 1, 0, len(k) - 2)
    t = (L - k[seg]) / (k[seg + 1] - k[seg])
    out[np.arange(len(L)), seg] = 1 - t
    out[np.arange(len(L)), seg + 1] = t
    return out


def fit_subject(pred_lab, true_lab, ws, smooth=0.05):
    """ΔL = s(L_pred)（带二阶差分平滑的最小二乘）与色度增益 k。"""
    B = subject_basis(pred_lab[:, 0])
    Dm = np.diff(np.eye(len(SUBJECT_KNOTS)), 2, axis=0)
    lhs = B.T @ (ws[:, None] * B) + smooth * ws.sum() * Dm.T @ Dm
    curve = np.linalg.solve(lhs, B.T @ (ws * (true_lab[:, 0] - pred_lab[:, 0])))
    cp = np.hypot(pred_lab[:, 1], pred_lab[:, 2])
    ct = np.hypot(true_lab[:, 1], true_lab[:, 2])
    k = float((ws * cp * (ct - cp)).sum() / max((ws * cp * cp).sum(), 1e-9))
    return curve, k


def apply_subject(lab, curve, k, m=1.0):
    out = lab.copy()
    out[:, 0] += m * (subject_basis(lab[:, 0]) @ curve)
    out[:, 1:] *= 1.0 + m * k
    return out


# ---- 数据 ----

def index_dir(d):
    return {os.path.splitext(os.path.basename(p))[0].lower(): p for p in glob.glob(os.path.join(d, '*'))
            if p.lower().endswith(IMG_EXT)}


def load_photo(stem, orig_path, ref_path, debug_dir, rng):
    O = imread(orig_path)
    R = imread(ref_path)
    if R.shape != O.shape:
        raise ValueError(f'{stem}: 尺寸不同 {O.shape} vs {R.shape}')
    h, w = O.shape[:2]
    s = FIT_LONG_SIDE / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    small = lambda a: cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    alpha = small(imread(os.path.join(debug_dir, 'matte.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255)
    skin = small(imread(os.path.join(debug_dir, 'skin_union.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255)
    diag = np.hypot(*size)
    edge = max(3, int(0.015 * diag)) | 1
    near_edge = cv2.dilate(cv2.Canny((alpha > 0.5).astype(np.uint8) * 255, 50, 150), np.ones((edge, edge), np.uint8)) > 0
    excl = cv2.dilate((skin > 0.2).astype(np.uint8), np.ones((edge, edge), np.uint8)) > 0
    lm = os.path.join(debug_dir, 'landmarks.json')
    if os.path.exists(lm):
        for f in json.load(open(lm, encoding='utf-8')).get('faces', []):
            x1, y1, x2, y2 = [v * s for v in f['bbox']]
            cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, 0.8 * (x2 - x1), 0.8 * (y2 - y1)
            excl[max(0, int(cy - hh)):int(cy + hh) + 1, max(0, int(cx - hw)):int(cx + hw) + 1] = True
    bg = (alpha < 0.05) & ~near_edge & ~excl
    subj = (alpha > 0.9) & ~near_edge & ~excl

    def pick(mask):
        ii = np.flatnonzero(mask.ravel())
        if len(ii) > SAMPLES_PER_PHOTO:
            ii = rng.choice(ii, SAMPLES_PER_PHOTO, replace=False)
        return ii

    Os, Rs = small(O).reshape(-1, 3) / 255.0, small(R).reshape(-1, 3) / 255.0
    ib, isub = pick(bg), pick(subj)
    return dict(stem=stem, stat_q={}, orig_u8=O, bg=(Os[ib], Rs[ib]), subject=(Os[isub], Rs[isub]))


def stats_for(photos, q):
    for p in photos:
        if q not in p['stat_q']:
            p['stat_q'][q] = black_statistic(p['orig_u8'], q)


# ---- 拟合与评估 ----

def fit_model(photos, cfg):
    xs, ys, ws = [], [], []
    for p in photos:
        o, r = p['bg']
        xs.append(apply_black(o, region_black(p, cfg, 'bg')))
        ys.append(r)
        ws.append(np.full(len(o), 1.0 / max(len(o), 1)))
    lut = fit_lut(np.concatenate(xs), np.concatenate(ys), np.concatenate(ws), cfg['size'], cfg['smooth'])
    pl, tl, sw = [], [], []
    for p in photos:
        o, r = p['subject']
        if len(o) < 200:
            continue
        pl.append(rgb_to_lab(apply_lut(lut, apply_black(o, region_black(p, cfg, 'subject')), cfg['size'])))
        tl.append(rgb_to_lab(r))
        sw.append(np.full(len(o), 1.0 / len(o)))
    curve, k = fit_subject(np.concatenate(pl), np.concatenate(tl), np.concatenate(sw))
    return lut, curve, k


def evaluate(p, model, cfg):
    lut, curve, k = model
    out = {}
    for name in ('bg', 'subject'):
        o, r = p[name]
        if len(o) < 200:
            out[name] = float('nan')
            continue
        pred = rgb_to_lab(apply_lut(lut, apply_black(o, region_black(p, cfg, name)), cfg['size']))
        if name == 'subject':
            pred = apply_subject(pred, curve, k)
        out[name] = float(np.median(np.linalg.norm(pred - rgb_to_lab(r), axis=1)))
    return out


def loo(photos, cfg):
    rows = []
    for i, p in enumerate(photos):
        model = fit_model([q for j, q in enumerate(photos) if j != i], cfg)
        rows.append((p['stem'], evaluate(p, model, cfg)))
    return rows


def write_cube(path, lut, n, title, comment):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for line in comment:
            f.write(f'# {line}\n')
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        for v in lut:
            f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--orig', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--debug', required=True, help='retouch apply --debug-dir 的输出，每张照片一个子目录')
    ap.add_argument('--extra', action='append', default=[], help='额外样本：<原图>::<参考图>::<调试子目录>')
    ap.add_argument('--size', type=int, default=33, help='LUT 边长')
    ap.add_argument('--smooth', type=float, default=1e-3, help='平滑正则权重')
    ap.add_argument('--black-q', type=float, default=2.0, help='黑点统计量：min(R,G,B) 的分位（%%）')
    ap.add_argument('--black-strength', type=float, default=0.6)
    ap.add_argument('--black-target', type=float, default=0.02)
    ap.add_argument('--black-subject', type=float, default=1.0, help='主体内使用黑点的比例（0..1，分析用）')
    ap.add_argument('--cube', default=None, help='写出 .cube')
    ap.add_argument('--title', default='grade')
    ap.add_argument('--loo', action='store_true', help='逐张留一评估')
    ap.add_argument('--search-black', action='store_true', help='按留一误差网格搜索黑点参数')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    refs = index_dir(a.ref)
    photos = []
    for o in sorted(glob.glob(os.path.join(a.orig, '*'))):
        stem = os.path.splitext(os.path.basename(o))[0]
        if stem.lower() in refs and os.path.isdir(os.path.join(a.debug, stem)):
            photos.append(load_photo(stem, o, refs[stem.lower()], os.path.join(a.debug, stem), rng))
    for spec in a.extra:
        o, r, d = spec.split('::')
        photos.append(load_photo(os.path.splitext(os.path.basename(o))[0], o, r, d, rng))
    print(f'{len(photos)} 张：' + ' '.join(p['stem'] for p in photos))

    cfg = dict(q=a.black_q, strength=a.black_strength, target=a.black_target, black_subject=a.black_subject,
               size=a.size, smooth=a.smooth)
    if a.search_black:
        best = None
        for q, s, t in itertools.product([0.5, 1.0, 2.0], [0.4, 0.6, 0.8], [0.02]):
            c = dict(cfg, q=q, strength=s, target=t, size=17)
            stats_for(photos, q)
            errs = [e['bg'] for _, e in loo(photos, c)]
            print(f'  黑点 q {q} 强度 {s} 目标 {t}: 背景 ΔE 平均 {np.mean(errs):.2f}  ' + ' '.join(f'{e:.2f}' for e in errs))
            if best is None or np.mean(errs) < best[0]:
                best = (np.mean(errs), q, s, t)
        cfg.update(q=best[1], strength=best[2], target=best[3])
        print(f'选定：q {cfg["q"]} 强度 {cfg["strength"]} 目标 {cfg["target"]}')
    stats_for(photos, cfg['q'])
    for p in photos:
        st = p['stat_q'][cfg['q']]
        print(f"  {p['stem']:9s} min(RGB) p{cfg['q']} = {st:.3f} → 黑点 {black_point(st, cfg['strength'], cfg['target']):.3f}")

    if a.loo:
        print('留一（背景 / 主体 ΔE 中位数）：')
        rows = loo(photos, cfg)
        for stem, e in rows:
            print(f'  {stem:9s} {e["bg"]:5.2f} / {e["subject"]:5.2f}')
        print(f'  平均      {np.nanmean([e["bg"] for _, e in rows]):5.2f} / {np.nanmean([e["subject"] for _, e in rows]):5.2f}')

    lut, curve, k = fit_model(photos, cfg)
    print('全部样本拟合（背景 / 主体 ΔE 中位数）：')
    for p in photos:
        e = evaluate(p, (lut, curve, k), cfg)
        print(f'  {p["stem"]:9s} {e["bg"]:5.2f} / {e["subject"]:5.2f}')
    params = {
        'black_point': {'percentile': cfg['q'], 'strength': cfg['strength'], 'target': cfg['target']},
        'subject': {'lightness': [round(float(v), 3) for v in curve], 'chroma': round(k, 4)},
    }
    print('调色参数（grade.rs / 预设）：')
    print(json.dumps(params, ensure_ascii=False))
    if a.cube:
        write_cube(a.cube, lut, cfg['size'], a.title, [
            f'{a.title}：tools/grade_fit.py 在 {len(photos)} 张参考导出上拟合的全局颜色映射（自适应黑点之后）',
            f'黑点 {json.dumps(params["black_point"], ensure_ascii=False)}；平滑 {cfg["smooth"]}',
        ])
        print('写出', a.cube)


if __name__ == '__main__':
    main()
