#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重新生成文档引用的样张对比图（out/cream/、out/x04/、out/batch_compare/、out/body_tone/、out/neck/、
out/blotches/、out/batch_test/compare/）、完整对比图（out/对比图/：奶油肌两批、婚纱-深色内景两批）与最终对照组
（out/最终对照组/对比图/：两个预设并排，见 doc/test_report_final.md）。

    python tools/sample_montages.py [cream] [x04] [batch] [body] [neck] [blotches] [batch_test] [compare_all] [final]

前提：已按 out/README.md 生成本项目输出（out/cream/ours_cream.jpg、out/x04/ours_cream.jpg、
out/x04/lite/ours_cream_lite.jpg、out/batch/*.jpg 与 out/batch/_landmarks/，
out/batch_test/*.jpg 与 out/batch_test/_landmarks/；婚纱-深色内景为 out/batch_dark_interior/、
out/batch_test_dark_interior/，命令同上加 `--preset 婚纱-深色内景`）；
轻量皮肤分割对照还需要 out/_work/beach_lite.jpg（`retouch apply ... --skinseg-model models/skin_seg_lite.onnx`）；
身体色调修改前后的对比（body）还需要修改前（提交 5d21223）的批处理输出 out/_work/body_before/*.jpg：
    git worktree add ../prb-before 5d21223 && cd ../prb-before && cargo build --release
    target/release/retouch batch -i <原片…> -o <本仓库>/out/_work/body_before --preset cream
颈纹淡化修改前后的对比（neck）还需要关掉颈纹这一步的批处理输出 out/_work/neck_before/*.jpg：
    retouch batch -i test/原片 -o out/_work/neck_before --preset cream --no-report \
        --set cream_female.neck.strength=0 --set cream_male.neck.strength=0 --set cream_unknown.neck.strength=0
阴影色块修改前后的对比（blotches）还需要修改前（0.6.2）两批的输出，放在 out/_work/blotch_before/*.jpg。
裁剪框为原图像素坐标，与各报告中的描述对应。
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

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


# 修改前后对比：原图 | 修改前 | 修改后（out/batch）| 像素蛋糕
ORIG_DIR, REF_DIR, AFTER_DIR = 'test/原片', 'test/像素蛋糕“奶油肌”预设产物', 'out/batch'
# 第三批样张（doc/test_report_batch_test.md）：只有 1V3A2954 有像素蛋糕的导出
BATCH_TEST_DIR, BATCH_TEST_OUT = 'test/批量测试', 'out/batch_test'

# 完整对比图（out/对比图/）：每张照片一张全图，另有每张脸、脖子的局部；有像素蛋糕导出的照片多一列。
# （原片目录，本程序的批处理输出目录，像素蛋糕同名预设的导出目录，对比图目录）；关键点取批处理输出目录下的 _landmarks/
DARK_REF_DIR = 'test/像素蛋糕“婚纱-深色内景”预设产物'
COMPARE_SETS = [
    (ORIG_DIR, AFTER_DIR, REF_DIR, 'out/对比图/原片'),
    (BATCH_TEST_DIR, BATCH_TEST_OUT, REF_DIR, 'out/对比图/批量测试'),
    (ORIG_DIR, 'out/batch_dark_interior', DARK_REF_DIR, 'out/对比图/婚纱-深色内景/原片'),
    (BATCH_TEST_DIR, 'out/batch_test_dark_interior', DARK_REF_DIR, 'out/对比图/婚纱-深色内景/批量测试'),
]
# 最终对照组（out/最终对照组/，doc/test_report_final.md）：两个预设 × 两批照片的批处理输出放在
# out/最终对照组/<预设>/<批>/（含 _landmarks/），五列对比图放在 out/最终对照组/对比图/<批>/
FINAL_DIR = 'out/最终对照组'
FINAL_PRESETS = [('奶油肌', REF_DIR), ('婚纱-深色内景', DARK_REF_DIR)]
FINAL_SETS = [ORIG_DIR, BATCH_TEST_DIR]
FINAL_LONG_SIDE = 1600      # 五列全图里每块的长边
FINAL_CROP_WIDTH = 640      # 五列脸 / 脖子里每块的宽
COMPARE_LONG_SIDE = 2400    # 全图对比里每块的长边（像素）
COMPARE_CROP_WIDTH = 800    # 脸 / 脖子对比里每块的宽（像素）
COMPARE_MIN_SCORE = 0.95    # 人脸检测得分更低的是裙摆亮片、花束上的误检，不出局部图
COMPARE_QUALITY = 92

# 身体色调（doc/test_report_body_tone.md）
BODY_BEFORE_DIR, BODY_OUT_DIR = 'out/_work/body_before', 'out/body_tone'
BODY_JOBS = [
    ('IMG_5785', 'armpit', (1380, 3060, 1620, 3300), 360),          # 问题所在：新娘腋下褶纹
    ('IMG_5785', 'armpit_wide', (1000, 2900, 1600, 3500), 400),
    ('IMG_5785', 'bride', (760, 2400, 2250, 4250), 360),
    ('IMG_5785', 'hand_on_shoulder', (2100, 2550, 2800, 3150), 400),  # 新娘的手搭在新郎肩上：按人混合无接缝
    ('IMG_5785', 'groom_hands', (2250, 4300, 3250, 5050), 400),
    ('1V3A3101', 'bride', (1700, 1000, 3300, 2750), 380),
    ('1V3A3101', 'groom_hand', (4100, 2200, 4800, 2900), 360),
    ('1V3A2954', 'bride', (2100, 2900, 3650, 4500), 400),
    ('1V3A3127', 'bride', (1150, 1650, 3600, 3500), 420),
    ('1V3A2922', 'bride', (1900, 3400, 3200, 5400), 330),
    ('IMG_5770', 'bride', (3050, 1350, 3900, 2450), 360),             # 局限：像素蛋糕这张提亮较少
    ('1V3A3150', 'bride_arms', (1250, 3850, 3250, 4900), 420),        # 局限：新娘的脸没检测到
]

# 颈纹淡化（doc/test_report_neck.md）
NECK_BEFORE_DIR, NECK_OUT_DIR = 'out/_work/neck_before', 'out/neck'
NECK_JOBS = [
    ('1V3A2922', 'bride_neck', (2200, 3280, 2800, 3880), 360),        # 问题所在：新娘的颈纹与碎发
    ('1V3A2922', 'bride_neck_zoom', (2440, 3440, 2700, 3640), 390),   # 同上，1.5 倍放大
    ('1V3A2922', 'groom_neck', (1640, 2380, 2020, 2720), 360),
    ('1V3A2954', 'bride_necklace', (2560, 2900, 3120, 3400), 360),    # 项链、细吊坠链与项链的影子保留
    ('1V3A2954', 'groom_neck', (1470, 2640, 1800, 3000), 330),
    ('1V3A3101', 'groom_neck', (3560, 690, 3960, 1180), 330),
    ('1V3A3127', 'groom_stubble', (4050, 1400, 4450, 1700), 400),     # 胡茬保留、颈纹淡化
    ('IMG_5770', 'groom_tattoo', (4380, 540, 4620, 760), 330),        # 纹身保留（含笔画淡下去的尾迹）
    ('IMG_5785', 'groom_tattoo', (2880, 2360, 3180, 2720), 330),
    ('1V3A3150', 'groom_small_face', (3000, 3960, 3180, 4120), 360),  # 小脸：脖子的颗粒保留
]


def locate(stem):
    """照片的原图与本程序的输出：test/原片 → out/batch，test/批量测试 → out/batch_test；找不到时为 (None, None)。"""
    find = lambda d: next(iter(glob.glob(os.path.join(d, stem + '.*'))), None)
    for orig_dir, ours_dir in ((ORIG_DIR, AFTER_DIR), (BATCH_TEST_DIR, BATCH_TEST_OUT)):
        orig, ours = find(orig_dir), find(ours_dir)
        if orig and ours:
            return orig, ours
    return None, None


def before_after(jobs, before_dir, out_dir):
    """每项一张：原图 | 修改前（before_dir）| 修改后（本程序当前的输出）| 像素蛋糕（有导出时）。"""
    os.makedirs(out_dir, exist_ok=True)
    for stem, region, box, width in jobs:
        find = lambda d: next(iter(glob.glob(os.path.join(d, stem + '.*'))), None)
        orig, ours = locate(stem)
        paths, labels = [orig, find(before_dir), ours], ['原图', '修改前', '修改后']
        if not all(paths):
            print(f'跳过 {stem}/{region}：缺少输入 {paths}')
            continue
        ref = find(REF_DIR)
        if ref:
            paths, labels = paths + [ref], labels + ['像素蛋糕']
        montage(paths, labels, box, os.path.join(out_dir, f'compare_{stem}_{region}.png'), width)


def face_boxes(bbox, img_w, img_h):
    """人脸框 → （脸：1.5 倍正方形，脖子：脸下方）两个裁剪框，截在图内。"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(w, h) * 0.75
    clip = lambda b: (max(0, int(b[0])), max(0, int(b[1])), min(img_w, int(b[2])), min(img_h, int(b[3])))
    return clip((cx - s, cy - s, cx + s, cy + s)), clip((cx - 0.9 * w, y2 - 0.2 * h, cx + 0.9 * w, y2 + 1.0 * h))


def find_ref(ref_dir, stem):
    """参考导出目录里与 `stem` 同名（不区分大小写）的文件；没有时为 None。"""
    refs = [q for q in glob.glob(os.path.join(ref_dir, '*'))
            if os.path.isfile(q) and os.path.splitext(os.path.basename(q))[0].lower() == stem.lower()]
    return refs[0] if refs else None


def compare_all():
    """out/对比图/：每张照片 `<照片>_全图.jpg`，每张脸 `<照片>_脸<k>.jpg` / `<照片>_脖子<k>.jpg`；
    列为 原图 | 本程序 | 像素蛋糕（没有像素蛋糕导出时只有前两列）。"""
    for orig_dir, ours_dir, ref_dir, out_dir in COMPARE_SETS:
        os.makedirs(out_dir, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(orig_dir, '*'))):
            stem = os.path.splitext(os.path.basename(p))[0]
            ours = os.path.join(ours_dir, stem + '.jpg')
            if not os.path.exists(ours):
                print(f'跳过 {stem}：没有本程序的输出 {ours}')
                continue
            ref = find_ref(ref_dir, stem)
            paths, labels = ([p, ours, ref], ['原图', '本程序', '像素蛋糕']) if ref else ([p, ours], ['原图', '本程序'])
            with Image.open(ours) as im:
                w, h = im.size
            tile = COMPARE_LONG_SIDE if w >= h else round(COMPARE_LONG_SIDE * w / h)
            montage(paths, labels, (0, 0, w, h), os.path.join(out_dir, f'{stem}_全图.jpg'), tile, COMPARE_QUALITY)
            lm = os.path.join(ours_dir, '_landmarks', stem + '.json')
            if not os.path.exists(lm):
                print(f'{stem}：没有关键点 {lm}，只出全图')
                continue
            faces = json.load(open(lm, encoding='utf-8'))['faces']
            skipped = [i for i, f in enumerate(faces) if f.get('score', 1.0) < COMPARE_MIN_SCORE]
            if skipped:
                print(f'{stem}：检测得分 < {COMPARE_MIN_SCORE} 的 {len(skipped)} 个检测不出局部图（误检）')
            real = [f for f in faces if f.get('score', 1.0) >= COMPARE_MIN_SCORE]
            for k, f in enumerate(real, 1):
                face, neck = face_boxes(f['bbox'], w, h)
                for box, part in ((face, '脸'), (neck, '脖子')):
                    montage(paths, labels, box, os.path.join(out_dir, f'{stem}_{part}{k}.jpg'),
                            COMPARE_CROP_WIDTH, COMPARE_QUALITY)


def final_group():
    """out/最终对照组/对比图/<批>/：每张照片 `<照片>_全图.jpg`，每张脸 `<照片>_脸<k>.jpg` / `<照片>_脖子<k>.jpg`；
    列为 原图 | 奶油肌·本程序 | 奶油肌·像素蛋糕 | 深色内景·本程序 | 深色内景·像素蛋糕（没有像素蛋糕导出的列省去）。
    人脸取奶油肌那批的关键点（两个预设的人脸检测相同）。"""
    for orig_dir in FINAL_SETS:
        batch = os.path.basename(orig_dir)
        out_dir = os.path.join(FINAL_DIR, '对比图', batch)
        os.makedirs(out_dir, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(orig_dir, '*'))):
            stem = os.path.splitext(os.path.basename(p))[0]
            paths, labels = [p], ['原图']
            for preset, ref_dir in FINAL_PRESETS:
                ours = os.path.join(FINAL_DIR, preset, batch, stem + '.jpg')
                if not os.path.exists(ours):
                    continue
                short = '深色内景' if preset.startswith('婚纱') else preset
                paths.append(ours)
                labels.append(f'{short}·本程序')
                ref = find_ref(ref_dir, stem)
                if ref:
                    paths.append(ref)
                    labels.append(f'{short}·像素蛋糕')
            if len(paths) == 1:
                print(f'跳过 {stem}：没有本程序的输出')
                continue
            with Image.open(paths[1]) as im:
                w, h = im.size
            tile = FINAL_LONG_SIDE if w >= h else round(FINAL_LONG_SIDE * w / h)
            montage(paths, labels, (0, 0, w, h), os.path.join(out_dir, f'{stem}_全图.jpg'), tile, COMPARE_QUALITY)
            lm = os.path.join(FINAL_DIR, FINAL_PRESETS[0][0], batch, '_landmarks', stem + '.json')
            if not os.path.exists(lm):
                continue
            real = [f for f in json.load(open(lm, encoding='utf-8'))['faces'] if f.get('score', 1.0) >= COMPARE_MIN_SCORE]
            for k, f in enumerate(real, 1):
                face, neck = face_boxes(f['bbox'], w, h)
                for box, part in ((face, '脸'), (neck, '脖子')):
                    montage(paths, labels, box, os.path.join(out_dir, f'{stem}_{part}{k}.jpg'),
                            FINAL_CROP_WIDTH, COMPARE_QUALITY)


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
                lm_dir='out/batch/_landmarks', out_dir='out/batch_compare', necks=False):
    """每张脸一张：原图 | 像素蛋糕 | 本程序（裁剪框取人脸框的 1.5 倍正方形；没有参考导出时只有原图 | 本程序）。
    `necks` 时每张脸再加一张脖子（人脸框下方）的对比 `<照片>_f<脸>_neck.png`。"""
    os.makedirs(out_dir, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(orig_dir, '*'))):
        stem = os.path.splitext(os.path.basename(p))[0]
        refs = [q for q in glob.glob(os.path.join(ref_dir, '*')) if os.path.splitext(os.path.basename(q))[0].lower() == stem.lower()]
        lm = os.path.join(lm_dir, stem + '.json')
        ours = os.path.join(ours_dir, stem + '.jpg')
        if not os.path.exists(lm) or not os.path.exists(ours):
            continue
        paths, labels = ([p, refs[0], ours], ['原图', '像素蛋糕', '本程序']) if refs else ([p, ours], ['原图', '本程序'])
        for fi, f in enumerate(json.load(open(lm, encoding='utf-8'))['faces']):
            x1, y1, x2, y2 = f['bbox']
            cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.75
            box = (max(0, int(cx - s)), max(0, int(cy - s)), int(cx + s), int(cy + s))
            montage(paths, labels, box, os.path.join(out_dir, f'{stem}_f{fi}.png'), 520)
            if necks:
                w, h = x2 - x1, y2 - y1
                box = (max(0, int(cx - 0.9 * w)), max(0, int(y2 - 0.2 * h)), int(cx + 0.9 * w), int(y2 + 1.0 * h))
                montage(paths, labels, box, os.path.join(out_dir, f'{stem}_f{fi}_neck.png'), 520)


# 身体遮罩的阴影色块（doc/test_report_shadow_blotches.md）：颜色规则在有色光下漏掉的皮肤、语义模型的孤岛
BLOTCH_BEFORE_DIR, BLOTCH_OUT_DIR = 'out/_work/blotch_before', 'out/blotches'
BLOTCH_JOBS = [
    ('1V3A2889', 'bride_shoulder', (931, 1537, 2016, 2475), 420),      # 问题所在：朝天的肩膀（天光偏蓝）
    ('1V3A2957', 'bride_chest', (1731, 1695, 2952, 2736), 420),        # 锁骨与胸口的斑驳
    ('1V3A2837', 'bride_neck_arm', (2093, 1885, 2994, 2732), 420),     # 胸口的暗斑（高光）与抬起的手臂（语义孤岛）
    ('1V3A2958', 'bride_chest', (2362, 1911, 3487, 2981), 420),        # 胸口与右肩
    ('1V3A2958', 'bride_arm', (1450, 1900, 2450, 3300), 360),          # 左臂
    ('1V3A2957', 'bride_arm', (1000, 1900, 1900, 3900), 300),          # 右臂
    ('1V3A2922', 'bride_arm', (2250, 3300, 3200, 5400), 300),          # 整条手臂与手（有像素蛋糕对照）
    ('1V3A2954', 'bride_chest', (2560, 2900, 3150, 3420), 400),        # 胸口（有像素蛋糕对照）
    ('1V3A2954', 'roses', (2300, 3900, 3000, 4500), 400),              # 不变：玫瑰花束仍不当皮肤处理
    ('1V3A3127', 'scarf_cards', (1400, 1500, 3300, 2700), 420),        # 不变：红围巾、"囍"字卡片；染红的手肘找回
    ('1V3A3188', 'bow_tie', (1850, 1950, 2350, 2350), 400),            # 米色领结不再被当成皮肤处理
]


def main():
    which = set(sys.argv[1:]) or {'cream', 'x04', 'batch', 'body', 'neck', 'blotches', 'batch_test', 'compare_all',
                                  'final'}
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
    if 'body' in which:
        before_after(BODY_JOBS, BODY_BEFORE_DIR, BODY_OUT_DIR)
    if 'neck' in which:
        before_after(NECK_JOBS, NECK_BEFORE_DIR, NECK_OUT_DIR)
    if 'blotches' in which:
        before_after(BLOTCH_JOBS, BLOTCH_BEFORE_DIR, BLOTCH_OUT_DIR)
    if 'batch_test' in which:
        compare_dir = os.path.join(BATCH_TEST_OUT, 'compare')
        batch_faces(orig_dir=BATCH_TEST_DIR, ours_dir=BATCH_TEST_OUT, lm_dir=os.path.join(BATCH_TEST_OUT, '_landmarks'),
                    out_dir=compare_dir, necks=True)
        for p in sorted(glob.glob(os.path.join(BATCH_TEST_DIR, '*'))):
            stem = os.path.splitext(os.path.basename(p))[0]
            ours = os.path.join(BATCH_TEST_OUT, stem + '.jpg')
            if os.path.exists(ours):
                change_locations(p, ours, os.path.join(compare_dir, f'{stem}_changes.png'), 8)
    if 'final' in which:
        final_group()
    if 'compare_all' in which:
        compare_all()


if __name__ == '__main__':
    main()
