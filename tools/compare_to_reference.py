#!/usr/bin/env python
"""把我们的输出与参考导出（如像素蛋糕）在同一原图上做定量对比：
光流对齐后按皮肤区域统计 L/a/b 各频段能量比、Lab 偏移、关键点位移。

    python tools/compare_to_reference.py <orig> <reference> <ours> <faces.json>

也可作为模块使用（tools/batch_compare.py）：`face_region` 取脸部裁剪与皮肤遮罩，`measure` 返回指标字典。
"""
import sys, json
import cv2, numpy as np

SHIFT_KEYS = ['pupil_l', 'pupil_r', 'eye_outer_l', 'eye_outer_r', 'nose_tip', 'nose_wing_l', 'mouth_l',
              'jaw_l0', 'jaw_l1', 'jaw_l2', 'chin', 'jaw_r1']


def imread(p):
    return cv2.cvtColor(cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def skin_rule(rgb):
    r, g, b = rgb[..., 0] / 255.0, rgb[..., 1] / 255.0, rgb[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    return (r > 0.3725) & (g > 0.1568) & (b > 0.0784) & (r > b) & ((mx - mn) > 0.0588) & (np.abs(r - g) > 0.0588)


def flow_align(a_gray, b_gray, b_rgb):
    s = 0.5
    a2 = cv2.resize(a_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b2 = cv2.resize(b_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); dis.setFinestScale(1)
    flow = dis.calc(a2, b2, None) / s
    flow = cv2.resize(flow, (a_gray.shape[1], a_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    h, w = a_gray.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    al = cv2.remap(b_rgb, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return al, flow


def bands(L):
    out = []; prev = L
    for s in [1, 2, 4, 8, 16, 32]:
        g = cv2.GaussianBlur(L, (0, 0), s); out.append(prev - g); prev = g
    return out


def face_region(A, f):
    """脸部裁剪框（原图坐标）与去掉五官的皮肤遮罩（裁剪坐标）。"""
    x1, y1, x2, y2 = [int(v) for v in f['bbox']]; w, h = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - 0.6 * w)), max(0, int(y1 - 0.7 * h)); X2, Y2 = min(A.shape[1], int(x2 + 0.6 * w)), min(A.shape[0], int(y2 + 0.5 * h))
    a = A[Y1:Y2, X1:X2]
    poly = np.array(f['contour'] + f['forehead'], dtype=np.int32) - [X1, Y1]
    m = np.zeros(a.shape[:2], np.uint8); cv2.fillPoly(m, [poly], 1)
    sem = f['semantic']; ed = f['eye_distance']
    k = max(9, int(round(0.17 * ed)) | 1)      # 轮廓内缩 ≈ 0.17 瞳距（瞳距 240 px 时即原来的 41 px）
    m = cv2.erode(m, np.ones((k, k), np.uint8)) & skin_rule(a).astype(np.uint8)
    for kk, r in [('pupil_l', 0.32), ('pupil_r', 0.32), ('nostril_l', 0.1), ('nostril_r', 0.1)]:
        cv2.circle(m, (int(sem[kk][0] - X1), int(sem[kk][1] - Y1)), int(r * ed), 0, -1)
    cv2.ellipse(m, (int((sem['mouth_l'][0] + sem['mouth_r'][0]) / 2 - X1), int((sem['mouth_l'][1] + sem['mouth_r'][1]) / 2 - Y1)), (int(0.45 * ed), int(0.22 * ed)), 0, 0, 360, 0, -1)
    return (X1, Y1, X2, Y2), m.astype(bool)


def measure(a, b_raw, m, f, origin):
    """a：原图脸部裁剪；b_raw：输出图同一裁剪（未对齐）；m：皮肤遮罩；origin：裁剪左上角（原图坐标）。"""
    ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_al, flow = flow_align(ag, cv2.cvtColor(b_raw, cv2.COLOR_RGB2GRAY), b_raw)
    la = cv2.cvtColor(a.astype(np.float32) / 255, cv2.COLOR_RGB2LAB); lb = cv2.cvtColor(b_al.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    res = {}
    # 频段能量比：输出图不重采样（重采样本身会削弱最细的频段），而是把原图上的皮肤遮罩按光流搬到输出图的像素网格上
    h, w = m.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    m_out = cv2.remap(m.astype(np.uint8), gx - flow[..., 0], gy - flow[..., 1], cv2.INTER_NEAREST, borderValue=0).astype(bool)
    lr = cv2.cvtColor(b_raw.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    for ch, cname in [(0, 'L'), (1, 'a'), (2, 'b')]:
        xa = bands(la[..., ch]); xb = bands(lr[..., ch])
        res[cname] = [float(np.sqrt((y[m_out] ** 2).mean()) / max(np.sqrt((x[m] ** 2).mean()), 1e-6)) for x, y in zip(xa, xb)]
    d = (lb - la)[m]
    L0 = la[..., 0][m]
    bins = {}
    for lo in range(30, 95, 10):
        sel = (L0 >= lo) & (L0 < lo + 10)
        if sel.sum() > 300:
            bins[str(lo)] = (float(d[sel, 0].mean()), float(d[sel, 1].mean()), float(d[sel, 2].mean()))
    for q in [90, 98]:
        thr = np.percentile(L0, q); sel = L0 >= thr
        bins[f'p{q}'] = (float(d[sel, 0].mean()), float(d[sel, 1].mean()), float(d[sel, 2].mean()))
    res['bins'] = bins
    res['mean'] = (float(d[:, 0].mean()), float(d[:, 1].mean()), float(d[:, 2].mean()))
    slopes = []
    for ch in (1, 2):
        xa = cv2.GaussianBlur(la[..., ch], (0, 0), 12)[m]; xb = cv2.GaussianBlur(lb[..., ch], (0, 0), 12)[m]
        da = xa - xa.mean(); db = xb - xb.mean(); slopes.append(float((da * db).sum() / (da * da).sum()))
    res['slope_a'], res['slope_b'] = slopes
    ed = f['eye_distance']; sem = f['semantic']
    shifts = {}
    for k in SHIFT_KEYS:
        px, py = sem[k]
        yy, xx = int(py - origin[1]), int(px - origin[0])
        if 0 <= yy < flow.shape[0] and 0 <= xx < flow.shape[1]:
            v = flow[yy, xx]
            shifts[k] = (round(float(v[0] / ed * 100), 2), round(float(v[1] / ed * 100), 2))
    res['shifts'] = shifts
    return res


def print_result(name, res):
    print(f'--- {name}')
    print('   L bands  ', ' '.join(f'{v:.2f}' for v in res['L']))
    print('   a bands  ', ' '.join(f'{v:.2f}' for v in res['a']))
    print('   b bands  ', ' '.join(f'{v:.2f}' for v in res['b']))
    d = res['mean']
    print(f'   lowfreq chroma slope a {res["slope_a"]:.3f} b {res["slope_b"]:.3f}; mean dLab {d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f}')
    print('   dLab by L bin:', '  '.join(f'{k}:({v[0]:+.1f},{v[1]:+.1f},{v[2]:+.1f})' for k, v in res['bins'].items()))
    print('   landmark shift (%ed):', ' '.join(f'{k}=({v[0]:+.1f},{v[1]:+.1f})' for k, v in res['shifts'].items()))


def main():
    orig, ref, ours, faces_json = sys.argv[1:5]
    A = imread(orig); B = imread(ref); C = imread(ours)
    faces = json.load(open(faces_json, encoding='utf-8'))['faces']
    for fi, f in enumerate(faces):
        (X1, Y1, X2, Y2), mb = face_region(A, f)
        a = A[Y1:Y2, X1:X2]
        print(f'== face {fi} ed={f["eye_distance"]:.0f} gender={f.get("gender")}')
        for name, img in [('reference', B), ('ours', C)]:
            print_result(name, measure(a, img[Y1:Y2, X1:X2], mb, f, (X1, Y1)))


if __name__ == '__main__':
    main()
