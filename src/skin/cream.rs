//! "奶油肌"模式：对像素蛋糕"奶油肌"预设导出图的逆向分析结果的实现（见 doc/analysis/cream_skin.md）。
//!
//! 逐张脸（以瞳距 ed 为尺度）在 Lab 空间内：
//! 1. 大尺度瑕疵修复（疤痕 / 痣：环核检测 + 调和插值 + 纹理移植，`heal.rs`）；
//! 2. 小瑕疵祛除（紧凑小斑检测 + 归一化卷积填充，`blemish.rs`）；
//! 3. 亮度三频段 + 能量自适应：`G_f = G(L; fine_sigma·ed)`、`G_m = G(L; mid_sigma·ed)`、
//!    `base = GF(L; detail_radius·ed, detail_eps)`（结构与光影，不动）；
//!    `fine = L − G_f`（颗粒）、`mid = G_f − G_m`（0.005–0.035 ed：毛孔簇、斑驳、细纹）、`low = G_m − base`；
//!    中频的局部能量 `E = √G(mid²; 0.05 ed)`（L 单位），衰减
//!    `α_mid(E) = detail_smooth·(1 − smoothstep(ln energy_lo, ln energy_hi, ln E))`（对数刻度：新郎的保留比例
//!    从 E≈0.7 一直缓慢升到 E≈11，线性刻度的 smoothstep 拟合不了）；
//!    `L' = L − α_fine·fine − α_mid(E)·mid − α_low·low`。
//!    依据：对像素蛋糕导出图按"局部能量分箱"统计（tools/texture_energy.py），两张样张、四张脸一致地表现为
//!    ——平滑皮肤处（E < 1–1.5）中频只剩 60–70%，纹理 / 五官边缘处（E > 3–5）几乎不动，最细颗粒只轻压 10–20%，
//!    更低频几乎不变；即"压平斑驳、保留毛孔与轮廓"。固定比例的衰减会在已经光滑的皮肤（浓妆）上过度磨皮；
//!    眼下区域的 α_mid / α_low 额外提高（细纹 / 卧蚕阴影）；
//! 4. 色度匀肤：`a' = a − α_c·(a − GF(a; guide L, r_c))`（b 同），并把低频色度向"随亮度变化的皮肤色度"
//!    （遮罩内 色度 ~ L 的线性回归）压缩 `unify`——高光彩度低是物理规律，不能当色斑拉平；
//! 5. 色调：中间调提亮（钟形，随 L 远离中心衰减）、最高光压制（去油光）；低频色度向目标肤色按比例拉近
//!    （偏黄偏红的皮肤降得多，白净的几乎不动），再加随亮度变化的 b 偏移（亮部 / 最高光 / 阴影）。
//!    常数由 tools/skin_tone_fit.py 在 10 张照片、16 张脸上拟合；
//! 6. 身体皮肤（颈胸臂手）用同样的三频段处理（参数独立）与更强的冷白偏移。
//!
//! 所有步骤只在各自遮罩的包围盒内计算并乘以羽化遮罩，遮罩外像素逐位不变。

use crate::buffer::{GrayF32, ImgF32};
use crate::color::lab::LabPlanes;
use crate::face::semantic::FaceKeyPoints;
use crate::geom::P;
use crate::skin::blemish::{detect_blemishes, detect_scale, inpaint_blobs, BlemishParams};
use crate::skin::guided::{fast_gaussian, guided_filter};
use crate::skin::heal::{detect_lesions, heal_lesions, HealParams};
use crate::skin::mask::fill_polygon;
use crate::skin::masks::{bbox_of, crop_gray, paste_gray, SkinMasks};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CreamParams {
    /// 中频（fine_sigma–mid_sigma：毛孔簇、斑驳、细纹）在平滑皮肤处的最大衰减比例 α_mid（0..1）
    pub detail_smooth: f32,
    /// 中频衰减随局部中频能量（Lab L 单位）从 energy_lo 到 energy_hi 平滑降到 0（对数刻度）
    pub energy_lo: f32,
    pub energy_hi: f32,
    /// 低频（mid_sigma–detail_radius：大块色斑）衰减比例 α_low（0..1）
    pub low_smooth: f32,
    /// 低频上限：导向滤波半径（× ed）
    pub detail_radius: f32,
    /// 低频导向滤波正则（L² 单位；越大越把中等对比的色块当作噪声压平，强边缘仍保留）
    pub detail_eps: f32,
    /// 细颗粒（< fine_sigma）衰减比例 α_fine（0..1）
    pub fine_smooth: f32,
    /// 细颗粒 / 中频分界的高斯 σ（× ed）
    pub fine_sigma: f32,
    /// 中频 / 低频分界的高斯 σ（× ed）
    pub mid_sigma: f32,
    /// 眼下区域额外的中频 / 低中频衰减（叠加到 α_mid、α_low）
    pub undereye_smooth: f32,
    /// 色度匀肤比例 α_c
    pub chroma_smooth: f32,
    /// 色度导向滤波半径（× ed）
    pub chroma_radius: f32,
    /// 低频色度向皮肤均值压缩的比例（肤色统一）
    pub chroma_unify: f32,
    /// 低频色度尺度（× ed）
    pub unify_sigma: f32,
    /// 中间调提亮量（L）：钟形 `lift · exp(−((L − center)/width)²)`
    pub midtone_lift: f32,
    pub lift_center_l: f32,
    pub lift_width_l: f32,
    /// 最高光压制量（L，作用于皮肤 L 的前 ~10%）
    pub highlight_suppress: f32,
    /// 随黄度提亮：`ΔL += lift_per_b · b`（低频 b）。像素蛋糕的提亮与皮肤黄度成正比（在 RGB 里降黄会抬高亮度）
    pub lift_per_b: f32,
    /// 肤色向目标色拉：低频色度 `a' = a − pull_a·(a − target_a)`（b 同）。像素蛋糕的肤色变化与原肤色成比例
    /// ——偏黄偏红的皮肤降得多，本就白净的皮肤几乎不动甚至略加暖（10 张照片 16 张脸，tools/skin_tone_fit.py）
    pub pull_a: f32,
    pub target_a: f32,
    pub pull_b: f32,
    pub target_b: f32,
    /// 额外的降红（a 偏移，叠加在目标色之上；默认 0）
    pub redness: f32,
    /// 额外的降黄（b 偏移，默认 0）；亮部随亮度渐增的偏移（L 50→85 smoothstep）；最高光（p90–p99.5）额外偏移；
    /// 阴影处额外偏移（暖阴影）
    pub yellow: f32,
    pub yellow_bright: f32,
    pub yellow_highlight: f32,
    pub yellow_shadow: f32,
    /// 眼部清晰：眼眶椭圆内的亮度局部对比增益 `L += k·(L − G(L; 0.04·瞳距))`（巩膜 / 高光更亮、
    /// 睫毛线与虹膜边缘更深）。像素蛋糕两张样张的眼区 L 标准差都变大（海边 +4–5%）
    pub eye_contrast: f32,
    /// 身体皮肤：中频衰减、低中频衰减、半径（× ed_mean）、细颗粒衰减、提亮、降红、降黄、色度匀肤
    pub body_smooth: f32,
    pub body_low_smooth: f32,
    pub body_radius: f32,
    pub body_fine_smooth: f32,
    pub body_lift: f32,
    pub body_redness: f32,
    pub body_yellow: f32,
    pub body_chroma_smooth: f32,
    /// 小瑕疵祛除
    pub blemish: BlemishParams,
    pub body_blemish: bool,
    /// 大尺度瑕疵修复（脸 / 身体）
    pub heal: HealParams,
    pub body_heal: HealParams,
}

impl Default for CreamParams {
    fn default() -> Self {
        Self {
            detail_smooth: 0.60,
            energy_lo: 1.2,
            energy_hi: 3.5,
            low_smooth: 0.03,
            detail_radius: 0.08,
            detail_eps: 60.0,
            fine_smooth: 0.26,
            fine_sigma: 0.005,
            mid_sigma: 0.035,
            undereye_smooth: 0.40,
            chroma_smooth: 0.33,
            chroma_radius: 0.14,
            chroma_unify: 0.15,
            unify_sigma: 0.05,
            midtone_lift: 0.405,
            lift_center_l: 62.0,
            lift_width_l: 26.0,
            highlight_suppress: 0.775,
            lift_per_b: 0.035,
            pull_a: 0.105,
            target_a: 3.67,
            pull_b: 0.149,
            target_b: 9.04,
            redness: 0.0,
            yellow: 0.0,
            yellow_bright: -0.5,
            yellow_highlight: 0.97,
            yellow_shadow: 0.05,
            eye_contrast: 0.30,
            body_smooth: 0.50,
            body_low_smooth: 0.25,
            body_radius: 0.05,
            body_fine_smooth: 0.25,
            body_lift: 2.4,
            body_redness: -0.5,
            body_yellow: -2.2,
            body_chroma_smooth: 0.35,
            blemish: BlemishParams::default(),
            body_blemish: true,
            heal: HealParams::default(),
            body_heal: HealParams::body_default(),
        }
    }
}

impl CreamParams {
    /// 男性默认（对应导出图中新郎的较轻处理：更小的细节衰减、几乎不提亮、保留胡茬、暖阴影；
    /// 但脸颊上的暗斑同样被祛除）。
    pub fn male_default() -> Self {
        Self {
            detail_smooth: 0.61,
            energy_lo: 0.7,
            energy_hi: 11.0,
            low_smooth: 0.03,
            detail_radius: 0.09,
            detail_eps: 80.0,
            fine_smooth: 0.05,
            undereye_smooth: 0.35,
            chroma_smooth: 0.27,
            chroma_unify: 0.2,
            midtone_lift: -0.086,
            lift_center_l: 62.0,
            lift_width_l: 26.0,
            highlight_suppress: 0.584,
            lift_per_b: 0.033,
            pull_a: 0.089,
            target_a: 8.35,
            pull_b: 0.090,
            target_b: 15.84,
            redness: 0.0,
            yellow: 0.0,
            yellow_bright: -1.07,
            yellow_highlight: 1.46,
            yellow_shadow: 0.67,
            eye_contrast: 0.25,
            blemish: BlemishParams {
                dark_contrast: 3.4,
                red_contrast: 3.6,
                max_radius: 0.022,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// 脸部几何（区域坐标）：用于眼下区域权重与五官 / 耳朵 / 发际线保护区。
#[derive(Clone)]
struct FaceGeom {
    pupil_l: P,
    pupil_r: P,
    nostril_l: P,
    nostril_r: P,
    nose_bottom: P,
    nose_wing_l: P,
    nose_wing_r: P,
    eye_outer_l: P,
    eye_outer_r: P,
    eye_inner_l: P,
    eye_inner_r: P,
    eye_top_l: P,
    eye_bot_l: P,
    eye_top_r: P,
    eye_bot_r: P,
    mouth_l: P,
    mouth_r: P,
    /// 脸轮廓 + 额头多边形（不含耳朵与头发）
    outline: Vec<P>,
    up: P,
    right: P,
    ed: f32,
}

impl FaceGeom {
    fn from_face(f: &FaceKeyPoints, off: P) -> Self {
        let mut outline: Vec<P> = f.contour.iter().map(|p| p.sub(off)).collect();
        outline.extend(f.forehead.iter().map(|p| p.sub(off)));
        Self {
            pupil_l: f.pupil_l.sub(off),
            pupil_r: f.pupil_r.sub(off),
            nostril_l: f.nostril_l.sub(off),
            nostril_r: f.nostril_r.sub(off),
            nose_bottom: f.nose_bottom.sub(off),
            nose_wing_l: f.nose_wing_l.sub(off),
            nose_wing_r: f.nose_wing_r.sub(off),
            eye_outer_l: f.eye_outer_l.sub(off),
            eye_outer_r: f.eye_outer_r.sub(off),
            eye_inner_l: f.eye_inner_l.sub(off),
            eye_inner_r: f.eye_inner_r.sub(off),
            eye_top_l: f.eye_top_l.sub(off),
            eye_bot_l: f.eye_bot_l.sub(off),
            eye_top_r: f.eye_top_r.sub(off),
            eye_bot_r: f.eye_bot_r.sub(off),
            mouth_l: f.mouth_l.sub(off),
            mouth_r: f.mouth_r.sub(off),
            outline,
            up: f.dir_up(),
            right: f.dir_right(),
            ed: f.eye_distance().max(8.0),
        }
    }
    fn shifted(&self, off: P) -> Self {
        Self {
            pupil_l: self.pupil_l.sub(off),
            pupil_r: self.pupil_r.sub(off),
            nostril_l: self.nostril_l.sub(off),
            nostril_r: self.nostril_r.sub(off),
            nose_bottom: self.nose_bottom.sub(off),
            nose_wing_l: self.nose_wing_l.sub(off),
            nose_wing_r: self.nose_wing_r.sub(off),
            eye_outer_l: self.eye_outer_l.sub(off),
            eye_outer_r: self.eye_outer_r.sub(off),
            eye_inner_l: self.eye_inner_l.sub(off),
            eye_inner_r: self.eye_inner_r.sub(off),
            eye_top_l: self.eye_top_l.sub(off),
            eye_bot_l: self.eye_bot_l.sub(off),
            eye_top_r: self.eye_top_r.sub(off),
            eye_bot_r: self.eye_bot_r.sub(off),
            mouth_l: self.mouth_l.sub(off),
            mouth_r: self.mouth_r.sub(off),
            outline: self.outline.iter().map(|p| p.sub(off)).collect(),
            up: self.up,
            right: self.right,
            ed: self.ed,
        }
    }
}

/// 软椭圆（沿 right / up 轴），值取 max。
fn draw_soft_ellipse(m: &mut GrayF32, c: P, right: P, up: P, rx: f32, ry: f32, v: f32) {
    let (w, h) = (m.w as i64, m.h as i64);
    let ext = rx.max(ry) * 1.2;
    let x0 = ((c.x - ext).floor() as i64).max(0);
    let y0 = ((c.y - ext).floor() as i64).max(0);
    let x1 = ((c.x + ext).ceil() as i64).min(w - 1);
    let y1 = ((c.y + ext).ceil() as i64).min(h - 1);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = P::new(x as f32 + 0.5, y as f32 + 0.5).sub(c);
            let dr = d.dot(right) / rx;
            let du = d.dot(up) / ry;
            let n = (dr * dr + du * du).sqrt();
            let t = ((1.15 - n) / 0.3).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t) * v;
            let i = (y * w + x) as usize;
            if t > m.data[i] {
                m.data[i] = t;
            }
        }
    }
}

fn punch_disc(m: &mut GrayF32, c: P, r: f32) {
    let (w, h) = (m.w as i64, m.h as i64);
    let x0 = ((c.x - r).floor() as i64).max(0);
    let y0 = ((c.y - r).floor() as i64).max(0);
    let x1 = ((c.x + r).ceil() as i64).min(w - 1);
    let y1 = ((c.y + r).ceil() as i64).min(h - 1);
    for y in y0..=y1 {
        for x in x0..=x1 {
            if P::new(x as f32 + 0.5, y as f32 + 0.5).dist(c) < r {
                m.data[(y * w + x) as usize] = 0.0;
            }
        }
    }
}

/// 一个待处理区域：遮罩（ROI 分辨率）、尺度、参数、可选脸部几何（ROI 坐标）。
struct Region<'a> {
    mask: &'a GrayF32,
    ed: f32,
    params: &'a CreamParams,
    is_face: bool,
    geom: Option<FaceGeom>,
    /// 语义皮肤概率（ROI 分辨率）；身体区域的瑕疵 / 疤痕检测只在语义上是皮肤的像素上进行
    skin_prob: Option<&'a GrayF32>,
}

fn crop_img(src: &ImgF32, x0: usize, y0: usize, cw: usize, ch: usize) -> ImgF32 {
    let mut out = ImgF32::new(cw, ch);
    out.data
        .par_chunks_mut(cw)
        .enumerate()
        .for_each(|(y, row)| {
            row.copy_from_slice(&src.data[(y0 + y) * src.w + x0..(y0 + y) * src.w + x0 + cw]);
        });
    out
}

/// 加权分位数（直方图近似）。
fn weighted_percentile(values: &[f32], weights: &[f32], q: f32) -> f32 {
    let mut hist = [0.0f64; 256];
    let mut total = 0.0f64;
    for (v, w) in values.iter().zip(weights) {
        if *w > 0.05 {
            let bin = ((v / 100.0).clamp(0.0, 1.0) * 255.0) as usize;
            hist[bin] += *w as f64;
            total += *w as f64;
        }
    }
    if total <= 0.0 {
        return 100.0;
    }
    let target = total * q as f64;
    let mut acc = 0.0;
    for (i, h) in hist.iter().enumerate() {
        acc += h;
        if acc >= target {
            return i as f32 / 255.0 * 100.0;
        }
    }
    100.0
}

fn subsample_for(rad: usize) -> usize {
    if rad >= 8 {
        4
    } else if rad >= 3 {
        2
    } else {
        1
    }
}

/// 纹理 / 轮廓处（高能量）中频也保留少量衰减：像素蛋糕在高能量箱的中频比约 0.94–0.97，不是 1。
const KEEP_MAX: f32 = 0.92;
/// 色调环节"向目标色拉"所用低频色度的尺度（× 瞳距当量），与 tools/skin_tone_fit.py 一致
const TONE_LOWPASS: f32 = 0.03;
/// 细颗粒 / 中频分界的最小高斯 σ（像素）
const FINE_SIGMA_FLOOR_PX: f32 = 0.5;
/// 身体区域强暗痕群否决的最低峰值（× 暗斑阈值），见 `BlemishParams::clutter_min_peak`
const BODY_CLUTTER_MIN_PEAK: f32 = 1.5;
/// 身体瑕疵检测的语义皮肤门控：概率在 LO..HI 之间线性过渡
const SEMANTIC_SKIN_LO: f32 = 0.45;
const SEMANTIC_SKIN_HI: f32 = 0.75;

/// 在区域遮罩的包围盒内处理三平面。
fn process_region(planes: &mut LabPlanes, r: &Region, timing: bool) {
    let p = r.params;
    let ed = r.ed.max(8.0);
    let tag = if r.is_face { "face" } else { "body" };
    let max_r = (p
        .chroma_radius
        .max(p.detail_radius)
        .max(p.unify_sigma * 3.0)
        * ed
        * 1.5) as usize;
    let Some((bx0, by0, bx1, by1)) = bbox_of(r.mask, 0.002, max_r.max(8)) else {
        return;
    };
    let (bw, bh) = (bx1 - bx0, by1 - by0);
    if bw < 4 || bh < 4 {
        return;
    }
    let m = crop_gray(r.mask, bx0, by0, bw, bh);
    let mut sub = LabPlanes {
        w: bw,
        h: bh,
        l: crop_gray(&planes.l, bx0, by0, bw, bh),
        a: crop_gray(&planes.a, bx0, by0, bw, bh),
        b: crop_gray(&planes.b, bx0, by0, bw, bh),
    };
    let geom = r
        .geom
        .as_ref()
        .map(|g| g.shifted(P::new(bx0 as f32, by0 as f32)));

    // 1. 大尺度瑕疵修复（脸部只在"轮廓多边形内缩 ∧ 遮罩"内进行，并扣掉眼周 / 鼻孔 / 嘴角保护区：
    //    耳朵、发际线、鼻孔、眼角、嘴角的暗结构不是瑕疵）
    let t = Instant::now();
    let hp = if r.is_face { &p.heal } else { &p.body_heal };
    let mut healed = 0;
    // 脸部瑕疵检测（大 / 小）的可处理区：轮廓多边形内缩 0.05 瞳距（耳朵、发际线的暗结构不是瑕疵）
    // ∧ 遮罩，再扣掉眼周 / 眉尾 / 鼻孔 / 鼻底 / 鼻翼 / 嘴角保护区
    let protect = match &geom {
        Some(g) if r.is_face => {
            let mut e = m.clone();
            let ed_g = g.ed;
            if g.outline.len() >= 3 {
                let mut poly = GrayF32::new(bw, bh);
                fill_polygon(&mut poly, &g.outline, 1.0);
                let mut poly = fast_gaussian(&poly, (0.04 * ed_g).max(1.0));
                poly.map_inplace(|v| if v > 0.9 { 1.0 } else { 0.0 });
                e.mul_inplace(&poly);
            }
            punch_disc(&mut e, g.pupil_l, 0.42 * ed_g);
            punch_disc(&mut e, g.pupil_r, 0.42 * ed_g);
            punch_disc(&mut e, g.eye_outer_l, 0.36 * ed_g);
            punch_disc(&mut e, g.eye_outer_r, 0.36 * ed_g);
            punch_disc(&mut e, g.nostril_l, 0.22 * ed_g);
            punch_disc(&mut e, g.nostril_r, 0.22 * ed_g);
            punch_disc(&mut e, g.nose_bottom, 0.22 * ed_g);
            punch_disc(&mut e, g.nose_wing_l, 0.22 * ed_g);
            punch_disc(&mut e, g.nose_wing_r, 0.22 * ed_g);
            punch_disc(&mut e, g.mouth_l, 0.18 * ed_g);
            punch_disc(&mut e, g.mouth_r, 0.18 * ed_g);
            e
        }
        _ => {
            // 身体：扣掉语义上不是皮肤的物体（纹身墨迹、首饰、衣褶）。纹身的粗笔画在检测分辨率上
            // 与大痣无异，连线状 / 暗痕群否决也拦不住；皮肤分割把墨迹判为非皮肤（IMG_5785：墨迹中位
            // 概率 0.36，周围皮肤 0.996），而 800 px 输入分辨率下几个像素的斑点仍是皮肤
            let mut e = m.clone();
            if let Some(sp) = r.skin_prob {
                e.data.par_iter_mut().zip(&sp.data).for_each(|(v, p)| {
                    *v *= ((*p - SEMANTIC_SKIN_LO) / (SEMANTIC_SKIN_HI - SEMANTIC_SKIN_LO))
                        .clamp(0.0, 1.0);
                });
            }
            e
        }
    };
    if hp.strength > 0.0 {
        let map = detect_lesions(&sub, &protect, ed, hp);
        healed = heal_lesions(&mut sub, &protect, &map, ed, hp);
    }
    if timing {
        eprintln!(
            "  [cream/{tag}] {bw}x{bh} heal: {healed} lesions, {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 2. 小瑕疵祛除
    let t = Instant::now();
    if p.blemish.strength > 0.0 && (r.is_face || p.body_blemish) {
        let bp = if r.is_face {
            p.blemish
        } else {
            // 身体区域大、瑕疵尺度大：用更粗的检测分辨率；粗分辨率下纹身细线的反差被抹淡，
            // 强暗痕群否决的门槛相应放低（身体上成片的是纹身 / 首饰，很少是痘印）
            BlemishParams {
                work_ed: p.blemish.work_ed * 0.6,
                clutter_min_peak: BODY_CLUTTER_MIN_PEAK,
                ..p.blemish
            }
        };
        let blob = detect_blemishes(&sub, &protect, ed, &bp);
        let sigma = (bp.fill_sigma * bp.max_radius * ed).max(2.0);
        inpaint_blobs(&mut sub, &blob, &m, sigma, detect_scale(ed, &bp));
    }
    if timing {
        eprintln!(
            "  [cream/{tag}] blemish: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 3. 亮度三频段 + 能量自适应：fine = L − G_f；mid = G_f − G_m；low = G_m − base；
    //    α_mid 随局部中频能量 E 平滑降为 0（平滑皮肤压斑驳，纹理 / 轮廓处不动）
    let t = Instant::now();
    let (alpha_mid, alpha_low, alpha_fine, rad_base, (e_lo, e_hi)) = if r.is_face {
        (
            p.detail_smooth,
            p.low_smooth,
            p.fine_smooth,
            p.detail_radius * ed,
            (p.energy_lo, p.energy_hi),
        )
    } else {
        (
            p.body_smooth,
            p.body_low_smooth,
            p.body_fine_smooth,
            p.body_radius * ed,
            (p.energy_lo * 1.2, p.energy_hi * 1.25),
        )
    };
    let guide = sub.l.clone();
    if alpha_mid > 0.0 || alpha_fine > 0.0 || alpha_low > 0.0 {
        let rad = rad_base.max(1.5) as usize;
        let base = guided_filter(&sub.l, &guide, rad, p.detail_eps, subsample_for(rad));
        // 细颗粒 / 中频分界至少 0.5 px（瞳距 < 100 px 的小脸才会触到下限）。像素蛋糕对小脸的像素级颗粒
        // 同样按中频压（10 张照片中瞳距 39–96 px 的 4 张脸：细颗粒只剩 0.73–0.89），下限 1 px 时本项目
        // 把这一档全算成细颗粒、只轻压（0.88–1.04）
        let sig_f = (p.fine_sigma * ed).max(FINE_SIGMA_FLOOR_PX);
        let g_f = fast_gaussian(&sub.l, sig_f);
        let g_m = fast_gaussian(&sub.l, (p.mid_sigma * ed).max(sig_f + 0.5));
        let mid_sq: Vec<f32> = g_f
            .data
            .par_iter()
            .zip(&g_m.data)
            .map(|(a, b)| (a - b) * (a - b))
            .collect();
        let energy = fast_gaussian(&GrayF32::from_vec(bw, bh, mid_sq), (0.05 * ed).max(1.0));
        // 眼下区域权重
        let mut boost: Option<GrayF32> = None;
        if let (Some(g), true) = (&geom, r.is_face && p.undereye_smooth > 0.0) {
            let mut ue = GrayF32::new(bw, bh);
            let down = g.up.mul(-1.0);
            for pupil in [g.pupil_l, g.pupil_r] {
                let c = pupil.add(down.mul(0.30 * g.ed));
                draw_soft_ellipse(&mut ue, c, g.right, g.up, 0.42 * g.ed, 0.22 * g.ed, 1.0);
            }
            boost = Some(ue);
        }
        let ue_gain = p.undereye_smooth;
        let ln_lo = e_lo.max(1e-3).ln();
        let ln_span = (e_hi.max(e_lo * 1.01).ln() - ln_lo).max(1e-3);
        sub.l.data.par_iter_mut().enumerate().for_each(|(i, l)| {
            let mk = m.data[i];
            if mk <= 0.0 {
                return;
            }
            let fine = *l - g_f.data[i];
            let mid = g_f.data[i] - g_m.data[i];
            let low = g_m.data[i] - base.data[i];
            let e = energy.data[i].max(1e-6).sqrt();
            let t = ((e.ln() - ln_lo) / ln_span).clamp(0.0, 1.0);
            let keep = KEEP_MAX * t * t * (3.0 - 2.0 * t);
            // 极强的边缘（E 远高于阈值）连细颗粒也不动
            let tf = ((e - 2.0 * e_hi) / (2.0 * e_hi)).clamp(0.0, 1.0);
            let b = boost.as_ref().map(|b| b.data[i] * ue_gain).unwrap_or(0.0);
            let am = (alpha_mid * (1.0 - keep) + b).min(0.9);
            let al = (alpha_low + b).min(0.9);
            let af = alpha_fine * (1.0 - tf);
            *l -= (am * mid + al * low + af * fine) * mk;
        });
    }
    if timing {
        eprintln!(
            "  [cream/{tag}] detail: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 4. 色度匀肤
    let t = Instant::now();
    let alpha_c = if r.is_face {
        p.chroma_smooth
    } else {
        p.body_chroma_smooth
    };
    if alpha_c > 0.0 {
        let rad = (p.chroma_radius * ed).max(2.0) as usize;
        for plane in [&mut sub.a, &mut sub.b] {
            let q = guided_filter(plane, &guide, rad, 40.0, 4);
            plane
                .data
                .par_iter_mut()
                .zip(&q.data)
                .zip(&m.data)
                .for_each(|((v, q), mk)| {
                    *v -= alpha_c * (*v - *q) * mk;
                });
        }
    }
    if r.is_face && p.chroma_unify > 0.0 {
        // 低频色度向"随亮度变化的皮肤色度"压缩：先在遮罩内做 色度 ~ L 的加权线性回归，只压缩偏离回归线的部分。
        // 高光处彩度低是反射光的物理规律，不是色斑；直接向均值压缩会给大面积高光加黄 / 加红
        // （X04 新娘 L 90+：b +1.3 → +0.1）
        let sigma = (p.unify_sigma * ed).max(2.0);
        let low_l = fast_gaussian(&sub.l, sigma);
        let (mut sw, mut sl, mut sll) = (0.0f64, 0.0f64, 0.0f64);
        for (l, w) in low_l.data.iter().zip(&m.data) {
            let (l, w) = (*l as f64, *w as f64);
            sw += w;
            sl += w * l;
            sll += w * l * l;
        }
        if sw >= 1.0 {
            let mean_l = sl / sw;
            let var_l = (sll / sw - mean_l * mean_l).max(1e-6);
            for plane in [&mut sub.a, &mut sub.b] {
                let low = fast_gaussian(plane, sigma);
                let (mut sv, mut slv) = (0.0f64, 0.0f64);
                for ((v, l), w) in low.data.iter().zip(&low_l.data).zip(&m.data) {
                    let (v, l, w) = (*v as f64, *l as f64, *w as f64);
                    sv += w * v;
                    slv += w * l * v;
                }
                let mean_v = sv / sw;
                let slope = ((slv / sw - mean_l * mean_v) / var_l) as f32;
                let (mean_l, mean_v) = (mean_l as f32, mean_v as f32);
                plane
                    .data
                    .par_iter_mut()
                    .zip(&low.data)
                    .zip(&low_l.data)
                    .zip(&m.data)
                    .for_each(|(((v, lo), ll), mk)| {
                        let target = mean_v + slope * (ll - mean_l);
                        *v -= p.chroma_unify * (*lo - target) * mk;
                    });
            }
        }
    }
    if timing {
        eprintln!(
            "  [cream/{tag}] chroma: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 5. 色调：亮度（中间调钟形提亮、最高光压制）；低频色度向目标色拉 + 随亮度变化的 b 偏移
    let t = Instant::now();
    if r.is_face {
        let p90 = weighted_percentile(&sub.l.data, &m.data, 0.90);
        let p995 = weighted_percentile(&sub.l.data, &m.data, 0.995).max(p90 + 1.0);
        let (lc, lw) = (p.lift_center_l, p.lift_width_l.max(1.0));
        let (lift, hl, red, yel, yel_hl, yel_sh) = (
            p.midtone_lift,
            p.highlight_suppress,
            p.redness,
            p.yellow,
            p.yellow_highlight,
            p.yellow_shadow,
        );
        let yel_br = p.yellow_bright;
        // 目标色拉只作用于低频色度，不额外压缩色度细节（那是第 4 步的事）
        let sigma = (TONE_LOWPASS * ed).max(1.0);
        let low_a = fast_gaussian(&sub.a, sigma);
        let low_b = fast_gaussian(&sub.b, sigma);
        sub.l
            .data
            .par_iter_mut()
            .zip(sub.a.data.par_iter_mut())
            .zip(sub.b.data.par_iter_mut())
            .zip(
                m.data
                    .par_iter()
                    .zip(low_a.data.par_iter().zip(&low_b.data)),
            )
            .for_each(|(((l, a), b), (mk, (la, lb)))| {
                if *mk <= 0.0 {
                    return;
                }
                let l0 = *l;
                let z = (l0 - lc) / lw;
                let t_lift = (-z * z).exp();
                let t_hl = ((l0 - p90) / (p995 - p90)).clamp(0.0, 1.0);
                let t_hl = t_hl * t_hl * (3.0 - 2.0 * t_hl);
                let t_sh = ((55.0 - l0) / 12.0).clamp(0.0, 1.0);
                let t_br = ((l0 - 50.0) / 35.0).clamp(0.0, 1.0);
                let t_br = t_br * t_br * (3.0 - 2.0 * t_br);
                *l += (lift * t_lift - hl * t_hl + p.lift_per_b * lb) * mk;
                *a += (red - p.pull_a * (la - p.target_a)) * mk;
                *b += (yel - p.pull_b * (lb - p.target_b)
                    + yel_br * t_br
                    + yel_hl * t_hl
                    + yel_sh * t_sh)
                    * mk;
            });
    } else {
        let (lift, red, yel) = (p.body_lift, p.body_redness, p.body_yellow);
        sub.l
            .data
            .par_iter_mut()
            .zip(sub.a.data.par_iter_mut())
            .zip(sub.b.data.par_iter_mut())
            .zip(&m.data)
            .for_each(|(((l, a), b), mk)| {
                if *mk <= 0.0 {
                    return;
                }
                let t = ((*l - 30.0) / 30.0).clamp(0.0, 1.0);
                *l += lift * t * mk;
                *a += red * mk;
                *b += yel * t * mk;
            });
    }
    if timing {
        eprintln!(
            "  [cream/{tag}] tone: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 6. 眼部清晰（眼眶椭圆：内外眼角连线为长轴、上下眼睑距离 + 睫毛余量为短轴，羽化）
    if let (Some(g), true) = (&geom, r.is_face && p.eye_contrast > 0.0) {
        let mut em = GrayF32::new(bw, bh);
        for (outer, inner, top, bot) in [
            (g.eye_outer_l, g.eye_inner_l, g.eye_top_l, g.eye_bot_l),
            (g.eye_outer_r, g.eye_inner_r, g.eye_top_r, g.eye_bot_r),
        ] {
            let axis = inner.sub(outer);
            let len = axis.len();
            if len < 2.0 {
                continue;
            }
            let right = axis.mul(1.0 / len);
            let up = P::new(right.y, -right.x);
            let up = if up.dot(g.up) < 0.0 { up.mul(-1.0) } else { up };
            let c = outer.add(inner).mul(0.5);
            // 短轴只多留睫毛的余量：再往下就是泪沟 / 眼袋的褶线，锐化它会让眼袋更明显（像素蛋糕反而把那里压平）
            let rx = 0.62 * len;
            let ry = 0.5 * top.dist(bot) + 0.04 * g.ed;
            draw_soft_ellipse(&mut em, c, right, up, rx, ry, 1.0);
        }
        let sigma = (0.04 * g.ed).max(1.0);
        let blur = fast_gaussian(&sub.l, sigma);
        let k = p.eye_contrast;
        sub.l
            .data
            .par_iter_mut()
            .zip(&blur.data)
            .zip(&em.data)
            .for_each(|((l, b), w)| {
                if *w > 0.0 {
                    *l = (*l + k * (*l - b) * w).clamp(0.0, 100.0);
                }
            });
    }
    paste_gray(&mut planes.l, &sub.l, bx0, by0);
    paste_gray(&mut planes.a, &sub.a, bx0, by0);
    paste_gray(&mut planes.b, &sub.b, bx0, by0);
}

/// 应用奶油肌。`faces[i]`（全图坐标）给出第 i 张脸的关键点（瞳距、眼下区域、五官保护区），
/// `face_params[i]` 给出其参数（可按性别不同）。`faces` 可少于遮罩数（缺省用平均瞳距、无保护区）。
pub fn cream_skin(
    orig: &ImgF32,
    masks: &SkinMasks,
    faces: &[FaceKeyPoints],
    face_params: &[&CreamParams],
    body_params: &CreamParams,
) -> ImgF32 {
    let timing = std::env::var("RETOUCH_TIMING").is_ok();
    let mut out = orig.clone();
    let Some((x0, y0, x1, y1)) = masks.roi else {
        return out;
    };
    let (cw, ch) = (x1 - x0, y1 - y0);
    if cw == 0 || ch == 0 {
        return out;
    }
    let t = Instant::now();
    let sub = crop_img(orig, x0, y0, cw, ch);
    let mut planes = LabPlanes::from_img(&sub);
    if timing {
        eprintln!(
            "  [cream] roi {cw}x{ch} to Lab: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    // 纹理 / 瑕疵的尺度用姿态稳健的脸部尺度（侧脸时瞳距偏小会把频段整体移到过细的尺度）；
    // 眼周等保护区的几何仍按真实瞳距（FaceGeom.ed）
    let face_eds: Vec<f32> = faces.iter().map(|f| f.scale_distance().max(8.0)).collect();
    let ed_mean = if face_eds.is_empty() {
        (orig.w.min(orig.h) as f32) * 0.06
    } else {
        face_eds.iter().sum::<f32>() / face_eds.len() as f32
    };
    let off = P::new(x0 as f32, y0 as f32);
    let skin_prob = masks
        .skin_prob
        .as_ref()
        .map(|sp| crop_gray(sp, x0, y0, cw, ch));
    for (i, fm) in masks.faces.iter().enumerate() {
        let m = crop_gray(fm, x0, y0, cw, ch);
        let params = face_params.get(i).copied().unwrap_or(body_params);
        process_region(
            &mut planes,
            &Region {
                mask: &m,
                ed: face_eds.get(i).copied().unwrap_or(ed_mean),
                params,
                is_face: true,
                geom: faces.get(i).map(|f| FaceGeom::from_face(f, off)),
                skin_prob: skin_prob.as_ref(),
            },
            timing,
        );
    }
    if let Some(b) = &masks.body {
        let m = crop_gray(b, x0, y0, cw, ch);
        process_region(
            &mut planes,
            &Region {
                mask: &m,
                ed: ed_mean,
                params: body_params,
                is_face: false,
                geom: None,
                skin_prob: skin_prob.as_ref(),
            },
            timing,
        );
    }
    let t = Instant::now();
    let union = crop_gray(&masks.union, x0, y0, cw, ch);
    let mut sub_out = sub;
    planes.blend_into(&mut sub_out, &union);
    out.data
        .par_chunks_mut(orig.w)
        .enumerate()
        .skip(y0)
        .take(ch)
        .for_each(|(y, row)| {
            let sy = y - y0;
            row[x0..x1].copy_from_slice(&sub_out.data[sy * cw..(sy + 1) * cw]);
        });
    if timing {
        eprintln!(
            "  [cream] blend back: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::GrayF32;

    #[test]
    fn outside_mask_is_untouched_and_inside_is_smoothed() {
        let (w, h) = (120, 100);
        let mut img = ImgF32::filled(w, h, [0.78, 0.60, 0.52]);
        for (i, p) in img.data.iter_mut().enumerate() {
            let n = ((i * 7919) % 11) as f32 / 11.0 * 0.04 - 0.02;
            p[0] += n;
            p[1] += n;
            p[2] += n;
        }
        let mut m = GrayF32::new(w, h);
        for y in 20..80 {
            for x in 20..100 {
                m.data[y * w + x] = 1.0;
            }
        }
        let masks = SkinMasks {
            faces: vec![m.clone()],
            body: None,
            union: m.clone(),
            roi: Some((10, 10, 110, 90)),
            skin_prob: None,
        };
        let p = CreamParams::default();
        let face = FaceKeyPoints {
            pupil_l: P::new(40.0, 40.0),
            pupil_r: P::new(80.0, 40.0),
            nose_bridge_top: P::new(60.0, 40.0),
            chin: P::new(60.0, 95.0),
            ..Default::default()
        };
        let out = cream_skin(&img, &masks, &[face], &[&p], &p);
        assert_eq!(out.data[5 * w + 5], img.data[5 * w + 5]);
        let var = |im: &ImgF32| {
            let vals: Vec<f32> = (30..70)
                .flat_map(|y| (30..90).map(move |x| (x, y)))
                .map(|(x, y)| im.get(x, y)[0])
                .collect();
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32
        };
        assert!(
            var(&out) < var(&img) * 0.8,
            "{} vs {}",
            var(&out),
            var(&img)
        );
    }
}
