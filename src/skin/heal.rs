//! 大尺度瑕疵（疤痕 / 痣 / 大块色斑）检测与"修复画笔"式填充。
//!
//! `blemish.rs` 只处理半径 ≤ 0.022 瞳距的紧凑小斑；手臂上的疤痕、较大的痣等（半径可达 0.15 瞳距、
//! 长宽比 2–4）需要本模块：
//!
//! 检测（在瞳距 ≈ `work_ed` 像素的降采样分辨率上，多尺度）：
//! 1. 零中心高斯环核估计"周围皮肤基准"：`ring = (G(2.5r) − c·G(0.8r)) / (1 − c)`，`c = (0.8/2.5)²`，
//!    该核在中心为 0、处处非负，因此基准不受瑕疵本身污染；
//! 2. 候选：中心比环暗（L）或比环红（a）超过阈值，或"能量判据"——中心带通 RMS 明显高于环
//!    （疤痕常是"暗弧 + 亮脊"，带符号的均值差互相抵消，但局部纹理能量是周围皮肤的 2–3 倍）；
//! 3. 剔除：距遮罩边缘不足 ~2r、环内带通 RMS 过大（周围不是平滑皮肤：五官、指缝、发丝）、
//!    大尺度梯度大（边缘）、环太暗（深阴影）；
//! 4. 滞后阈值：在半阈值图上取连通域，要求含有超过阈值的像素，且面积在 [π(0.35r)², π(1.8r)²]、
//!    主轴伸长率（PCA）≤ max_aspect、填充率 ≥ min_fill——发丝阴影 / 皱褶在半阈值下连成长条而被整体剔除；
//!    连通域边界贴着"剔除区"的比例 ≤ max_cut_fraction——被遮罩 / 边缘规则切断的指缝、鼻孔、发际线段
//!    不是四周被平滑皮肤包围的孤立瑕疵；各尺度取并集。
//!
//! 填充（全分辨率，逐个瑕疵）：
//! 1. 瑕疵核心向外扩 `margin·r` 为替换区 Ω，再向外 0.3r 为羽化带；
//! 2. 调和插值：在 Ω 内解 Laplace 方程（Dirichlet 边界 = 周围原像素），多分辨率 Gauss-Seidel，
//!    得到与周围明暗 / 色度连续的低频（L、a、b 三平面）；
//! 3. 纹理移植（healing brush）：在瑕疵周围 2–4 倍半径的 16 个方向上寻找一块干净皮肤
//!    （全部在可处理遮罩内、不含其他瑕疵、均值最接近环、高频能量接近环），
//!    把它的高通分量 `L − G(L; texture_sigma·ed)` 加到 Ω 的 L 上，
//!    使填充区拥有与周围一致的皮肤颗粒而不是一片模糊。
//!
//! 参考：FabSoften（CVPRW 2020）的瑕疵检测思路、Pérez 等 Poisson image editing（SIGGRAPH 2003）
//! 的调和 / 梯度域插值、Photoshop 修复画笔"纹理来自源、明暗来自目标"的原则，
//! 以及像素蛋糕"皮肤瑕疵祛除"的产物特征（疤痕被完全填平且保留皮肤颗粒）。

use crate::buffer::GrayF32;
use crate::color::lab::LabPlanes;
use crate::skin::blemish::{down, gradient_mag};
use crate::skin::guided::fast_gaussian;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct HealParams {
    /// 0 关闭；>1 更激进（阈值除以 strength）
    pub strength: f32,
    /// 最小 / 最大瑕疵半径（× 瞳距）
    pub min_radius: f32,
    pub max_radius: f32,
    /// 中心比环暗的阈值（L）
    pub dark_contrast: f32,
    /// 中心比环红的阈值（a）
    pub red_contrast: f32,
    /// 环内带通残差 RMS 上限（周围必须是平滑皮肤：五官 / 指缝 / 发丝处很大）
    pub ring_rms_max: f32,
    /// 能量判据：中心带通 RMS ≥ energy_abs 且 ≥ energy_ratio × 环 RMS（捕捉"暗 + 亮"互相抵消的疤痕）；
    /// 只在环非常平滑（环 RMS ≤ energy_ring_max）时启用；energy_abs ≤ 0 关闭
    pub energy_abs: f32,
    pub energy_ratio: f32,
    pub energy_ring_max: f32,
    /// 连通域边界上与"被剔除像素"（遮罩边缘 / 边缘 / 结构区）相邻的比例上限：
    /// 真瑕疵四周都是有效的平滑皮肤；被遮罩切断的指缝、鼻孔、发际线段则大量贴着剔除区
    pub max_cut_fraction: f32,
    /// 大尺度梯度阈值（L / 像素，检测分辨率）
    pub edge_reject: f32,
    /// 环 L 低于此值视为深阴影
    pub min_surround_l: f32,
    /// 连通域最大长宽比、最小填充率
    pub max_aspect: f32,
    pub min_fill: f32,
    /// 检测分辨率对应的瞳距（像素）
    pub work_ed: f32,
    /// 纹理移植的高通尺度（× 瞳距）与增益
    pub texture_sigma: f32,
    pub texture_gain: f32,
    /// 替换区向外扩展量（× 瑕疵半径）
    pub margin: f32,
}

impl Default for HealParams {
    /// 脸部默认：只处理明显的大痣 / 斑（半径 0.02–0.05 瞳距），阈值偏高。
    fn default() -> Self {
        Self {
            strength: 1.0,
            min_radius: 0.02,
            max_radius: 0.05,
            dark_contrast: 5.0,
            red_contrast: 4.5,
            ring_rms_max: 4.0,
            energy_abs: 0.0,
            energy_ratio: 2.2,
            energy_ring_max: 2.0,
            max_cut_fraction: 0.10,
            edge_reject: 0.6,
            min_surround_l: 40.0,
            max_aspect: 4.0,
            min_fill: 0.3,
            work_ed: 120.0,
            texture_sigma: 0.025,
            texture_gain: 0.9,
            margin: 0.35,
        }
    }
}

impl HealParams {
    /// 身体默认：颈胸臂手上的疤痕 / 痣，半径可达 0.16 瞳距。
    pub fn body_default() -> Self {
        Self {
            min_radius: 0.025,
            max_radius: 0.16,
            dark_contrast: 3.5,
            red_contrast: 3.5,
            ring_rms_max: 4.5,
            energy_abs: 3.0,
            energy_ratio: 2.0,
            energy_ring_max: 2.5,
            ..Default::default()
        }
    }
}

/// 一处待修复瑕疵（区域坐标、全分辨率）。
#[derive(Clone, Debug)]
pub struct Lesion {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
    /// w×h，1 = 瑕疵核心
    pub mask: Vec<u8>,
    /// 等效半径（像素）
    pub r: f32,
}

/// 检测结果：瑕疵列表 + 检测分辨率上的并集图（供纹理源搜索时避开其他瑕疵）。
pub struct LesionMap {
    pub lesions: Vec<Lesion>,
    pub blob_s: GrayF32,
    pub scale: f32,
}

fn zip2(a: &GrayF32, b: &GrayF32, f: impl Fn(f32, f32) -> f32 + Sync) -> GrayF32 {
    let data: Vec<f32> = a
        .data
        .par_iter()
        .zip(&b.data)
        .map(|(x, y)| f(*x, *y))
        .collect();
    GrayF32::from_vec(a.w, a.h, data)
}

/// 零中心高斯环核均值。
fn ring_mean(src: &GrayF32, r: f32) -> GrayF32 {
    let c = (0.8f32 / 2.5).powi(2);
    let big = fast_gaussian(src, 2.5 * r);
    let small = fast_gaussian(src, 0.8 * r);
    zip2(&big, &small, move |b, s| (b - c * s) / (1.0 - c))
}

pub(crate) struct Component {
    pub(crate) pixels: Vec<usize>,
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) x1: usize,
    pub(crate) y1: usize,
}

/// 4 连通域（`limit` 为单个连通域的扫描上限，超过即视为大区域并整体丢弃）。
pub(crate) fn components(hit: &[u8], w: usize, h: usize, limit: usize) -> Vec<Component> {
    let mut label = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..w * h {
        if hit[start] == 0 || label[start] {
            continue;
        }
        stack.clear();
        stack.push(start);
        label[start] = true;
        let mut comp = Component {
            pixels: Vec::new(),
            x0: w,
            y0: h,
            x1: 0,
            y1: 0,
        };
        let mut too_big = false;
        while let Some(i) = stack.pop() {
            comp.pixels.push(i);
            let (x, y) = (i % w, i / w);
            comp.x0 = comp.x0.min(x);
            comp.x1 = comp.x1.max(x);
            comp.y0 = comp.y0.min(y);
            comp.y1 = comp.y1.max(y);
            let nb = [
                (x > 0).then(|| i - 1),
                (x + 1 < w).then(|| i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < h).then(|| i + w),
            ];
            for j in nb.into_iter().flatten() {
                if hit[j] != 0 && !label[j] {
                    label[j] = true;
                    stack.push(j);
                }
            }
            if comp.pixels.len() > limit {
                too_big = true;
            }
        }
        if !too_big {
            out.push(comp);
        }
    }
    out
}

/// 主轴伸长率（PCA）：√(λ₁/λ₂)，单像素为 1。
pub(crate) fn elongation(pixels: &[usize], w: usize) -> f32 {
    let n = pixels.len() as f64;
    if n < 2.0 {
        return 1.0;
    }
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    for &i in pixels {
        sx += (i % w) as f64;
        sy += (i / w) as f64;
    }
    let (mx, my) = (sx / n, sy / n);
    let (mut cxx, mut cyy, mut cxy) = (0.0f64, 0.0f64, 0.0f64);
    for &i in pixels {
        let dx = (i % w) as f64 - mx;
        let dy = (i / w) as f64 - my;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    // 加上单像素自身的方差 1/12，避免一行像素的 λ₂ = 0
    let (cxx, cyy) = (cxx / n + 1.0 / 12.0, cyy / n + 1.0 / 12.0);
    let cxy = cxy / n;
    let tr = 0.5 * (cxx + cyy);
    let d = ((0.5 * (cxx - cyy)).powi(2) + cxy * cxy).sqrt();
    let (l1, l2) = (tr + d, (tr - d).max(1e-6));
    (l1 / l2).sqrt() as f32
}

/// 检测大尺度瑕疵。`eligible` 为可处理皮肤遮罩（>0.5 有效）。返回区域坐标下的瑕疵列表。
pub fn detect_lesions(
    planes: &LabPlanes,
    eligible: &GrayF32,
    ed: f32,
    p: &HealParams,
) -> LesionMap {
    let (w, h) = (planes.w, planes.h);
    let s = (p.work_ed / ed.max(1.0)).clamp(0.08, 1.0);
    let empty = |sw, sh| LesionMap {
        lesions: Vec::new(),
        blob_s: GrayF32::new(sw, sh),
        scale: s,
    };
    if p.strength <= 0.0 || w < 8 || h < 8 {
        return empty(1, 1);
    }
    let l = down(&planes.l, s);
    let a = down(&planes.a, s);
    let el = down(eligible, s);
    let (sw, sh) = (l.w, l.h);
    let ed_s = ed * s;
    let k = 1.0 / p.strength.max(0.05);
    let r_min = (p.min_radius * ed_s).max(1.5);
    let r_max = (p.max_radius * ed_s).max(r_min + 0.5);
    let n_scales = (((r_max / r_min).ln() / 1.5f32.ln()).ceil() as usize + 1).clamp(2, 6);
    let mut blob = vec![0u8; sw * sh];
    for si in 0..n_scales {
        let t = si as f32 / (n_scales - 1) as f32;
        let r = r_min * (r_max / r_min).powf(t);
        let inner_l = fast_gaussian(&l, r * 0.6);
        let inner_a = fast_gaussian(&a, r * 0.6);
        let ring_l = ring_mean(&l, r);
        let ring_a = ring_mean(&a, r);
        let hp = zip2(&l, &fast_gaussian(&l, r), |x, y| x - y);
        let hp2 = GrayF32::from_vec(sw, sh, hp.data.par_iter().map(|v| v * v).collect());
        let e_in = fast_gaussian(&hp2, r * 0.6);
        let e_ring = ring_mean(&hp2, r);
        let grad = gradient_mag(&fast_gaussian(&l, r * 2.0));
        let el_er = fast_gaussian(&el, r * 1.2);
        let dark_t = p.dark_contrast * k;
        let red_t = p.red_contrast * k;
        let edge_t = p.edge_reject * k;
        let e_abs = p.energy_abs * k;
        let e_ratio = p.energy_ratio.max(1.0);
        let rms_t = p.ring_rms_max;
        // 响应：1 = 阈值；rejected = 被规则剔除（而非响应低）
        let (resp, rejected): (Vec<f32>, Vec<bool>) = (0..sw * sh)
            .into_par_iter()
            .map(|i| {
                let er = e_ring.data[i].max(0.0).sqrt();
                if el_er.data[i] < 0.97
                    || grad.data[i] > edge_t
                    || ring_l.data[i] < p.min_surround_l
                    || er > rms_t
                {
                    return (0.0, true);
                }
                let dark = (ring_l.data[i] - inner_l.data[i]) / dark_t;
                let red = (inner_a.data[i] - ring_a.data[i]) / red_t;
                let ei = e_in.data[i].max(0.0).sqrt();
                let energy = if e_abs > 0.0 && ei >= e_abs && er <= p.energy_ring_max {
                    ei / (er.max(0.3) * e_ratio)
                } else {
                    0.0
                };
                (dark.max(red).max(energy), false)
            })
            .unzip();
        let low: Vec<u8> = resp.iter().map(|v| (*v >= 0.5) as u8).collect();
        let min_area = (std::f32::consts::PI * (0.35 * r).powi(2)).max(3.0) as usize;
        let max_area = (std::f32::consts::PI * (1.8 * r).powi(2)) as usize;
        for c in components(&low, sw, sh, max_area * 2) {
            let area = c.pixels.len();
            let peak = c.pixels.iter().map(|&i| resp[i]).fold(0.0f32, f32::max);
            let bw = (c.x1 - c.x0 + 1) as f32;
            let bh = (c.y1 - c.y0 + 1) as f32;
            let fill = area as f32 / (bw * bh);
            // 边界接触剔除区的比例
            let (mut cut, mut boundary) = (0usize, 0usize);
            for &i in &c.pixels {
                let (x, y) = (i % sw, i / sw);
                let nb = [
                    (x > 0).then(|| i - 1),
                    (x + 1 < sw).then(|| i + 1),
                    (y > 0).then(|| i - sw),
                    (y + 1 < sh).then(|| i + sw),
                ];
                for j in nb.into_iter().flatten() {
                    if low[j] == 0 {
                        boundary += 1;
                        if rejected[j] {
                            cut += 1;
                        }
                    }
                }
            }
            let cut_frac = cut as f32 / boundary.max(1) as f32;
            if peak >= 1.0
                && area >= min_area
                && area <= max_area
                && elongation(&c.pixels, sw) <= p.max_aspect
                && fill >= p.min_fill
                && cut_frac <= p.max_cut_fraction
            {
                for &i in &c.pixels {
                    blob[i] = 1;
                }
            }
        }
    }
    // 各尺度并集的连通域 → 全分辨率瑕疵核心
    let max_final = (std::f32::consts::PI * (2.2 * r_max).powi(2)) as usize;
    let mut lesions = Vec::new();
    for c in components(&blob, sw, sh, max_final) {
        let area = c.pixels.len();
        let r_s = (area as f32 / std::f32::consts::PI).sqrt();
        if r_s < 0.5 * r_min {
            continue;
        }
        // 连通域局部图（带 1 像素边）
        let (ox, oy) = (c.x0.saturating_sub(1), c.y0.saturating_sub(1));
        let (cw, ch) = (c.x1 + 2 - ox, c.y1 + 2 - oy);
        let mut cimg = GrayF32::new(cw.min(sw - ox), ch.min(sh - oy));
        for &i in &c.pixels {
            let (x, y) = (i % sw - ox, i / sw - oy);
            if x < cimg.w && y < cimg.h {
                cimg.data[y * cimg.w + x] = 1.0;
            }
        }
        let fx0 = ((c.x0 as f32) / s).floor().max(0.0) as usize;
        let fy0 = ((c.y0 as f32) / s).floor().max(0.0) as usize;
        let fx1 = (((c.x1 + 1) as f32) / s).ceil() as usize;
        let fy1 = (((c.y1 + 1) as f32) / s).ceil() as usize;
        let (fx1, fy1) = (fx1.min(w), fy1.min(h));
        if fx1 <= fx0 || fy1 <= fy0 {
            continue;
        }
        let (lw, lh) = (fx1 - fx0, fy1 - fy0);
        let mut mask = vec![0u8; lw * lh];
        let mut count = 0usize;
        for y in 0..lh {
            for x in 0..lw {
                let sx = ((fx0 + x) as f32 + 0.5) * s - ox as f32;
                let sy = ((fy0 + y) as f32 + 0.5) * s - oy as f32;
                if cimg.sample_bilinear(sx, sy) >= 0.5 {
                    mask[y * lw + x] = 1;
                    count += 1;
                }
            }
        }
        if count == 0 {
            continue;
        }
        lesions.push(Lesion {
            x0: fx0,
            y0: fy0,
            w: lw,
            h: lh,
            mask,
            r: (count as f32 / std::f32::consts::PI).sqrt(),
        });
    }
    let blob_s = GrayF32::from_vec(sw, sh, blob.iter().map(|v| *v as f32).collect());
    LesionMap {
        lesions,
        blob_s,
        scale: s,
    }
}

/// 3-4 倒角距离变换：到 `inside`（非零）像素的近似欧氏距离。
fn chamfer_distance(inside: &[u8], w: usize, h: usize) -> Vec<f32> {
    let big = (w + h) as f32 * 3.0;
    let mut d: Vec<f32> = inside
        .iter()
        .map(|v| if *v != 0 { 0.0 } else { big })
        .collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let mut v = d[i];
            if x > 0 {
                v = v.min(d[i - 1] + 3.0);
            }
            if y > 0 {
                v = v.min(d[i - w] + 3.0);
                if x > 0 {
                    v = v.min(d[i - w - 1] + 4.0);
                }
                if x + 1 < w {
                    v = v.min(d[i - w + 1] + 4.0);
                }
            }
            d[i] = v;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            let mut v = d[i];
            if x + 1 < w {
                v = v.min(d[i + 1] + 3.0);
            }
            if y + 1 < h {
                v = v.min(d[i + w] + 3.0);
                if x > 0 {
                    v = v.min(d[i + w - 1] + 4.0);
                }
                if x + 1 < w {
                    v = v.min(d[i + w + 1] + 4.0);
                }
            }
            d[i] = v;
        }
    }
    d.iter_mut().for_each(|v| *v /= 3.0);
    d
}

fn gauss_seidel(vals: &mut [f32], inside: &[bool], w: usize, h: usize, iters: usize) {
    for _ in 0..iters {
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                if inside[i] {
                    vals[i] = 0.25 * (vals[i - 1] + vals[i + 1] + vals[i - w] + vals[i + w]);
                }
            }
        }
    }
}

/// 调和插值：在 `inside` 内解 Laplace 方程，边界值为 `inside` 外的原值。多分辨率 Gauss-Seidel。
/// 假定裁剪块最外一圈像素都不在 `inside` 内。
pub fn solve_harmonic(vals: &mut [f32], inside: &[bool], w: usize, h: usize) {
    if w >= 24 && h >= 24 {
        let (cw, ch) = (w / 2, h / 2);
        let mut cv = vec![0.0f32; cw * ch];
        let mut ci = vec![false; cw * ch];
        for y in 0..ch {
            for x in 0..cw {
                let idx = [
                    (2 * y) * w + 2 * x,
                    (2 * y) * w + 2 * x + 1,
                    (2 * y + 1) * w + 2 * x,
                    (2 * y + 1) * w + 2 * x + 1,
                ];
                let n_in = idx.iter().filter(|&&i| inside[i]).count();
                let ins = n_in == 4;
                let (mut sum, mut n) = (0.0f32, 0usize);
                for &i in &idx {
                    if ins || !inside[i] {
                        sum += vals[i];
                        n += 1;
                    }
                }
                ci[y * cw + x] = ins;
                cv[y * cw + x] = sum / n.max(1) as f32;
            }
        }
        // 保证粗层最外圈不在内部
        for x in 0..cw {
            ci[x] = false;
            ci[(ch - 1) * cw + x] = false;
        }
        for y in 0..ch {
            ci[y * cw] = false;
            ci[y * cw + cw - 1] = false;
        }
        solve_harmonic(&mut cv, &ci, cw, ch);
        let coarse = GrayF32::from_vec(cw, ch, cv);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if inside[i] {
                    vals[i] =
                        coarse.sample_bilinear((x as f32 + 0.5) * 0.5, (y as f32 + 0.5) * 0.5);
                }
            }
        }
        gauss_seidel(vals, inside, w, h, 60);
    } else {
        // 初值：边界均值
        let (mut sum, mut n) = (0.0f32, 0usize);
        for i in 0..w * h {
            if !inside[i] {
                sum += vals[i];
                n += 1;
            }
        }
        let mean = sum / n.max(1) as f32;
        for i in 0..w * h {
            if inside[i] {
                vals[i] = mean;
            }
        }
        gauss_seidel(vals, inside, w, h, 400);
    }
}

struct Stats {
    mean_l: f32,
    mean_a: f32,
    std_hp: f32,
}

fn stats(
    l: &GrayF32,
    a: &GrayF32,
    l_blur: &GrayF32,
    idx: impl Iterator<Item = usize>,
) -> Option<Stats> {
    let (mut sl, mut sa, mut sh, mut sh2, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
    for i in idx {
        let hp = (l.data[i] - l_blur.data[i]) as f64;
        sl += l.data[i] as f64;
        sa += a.data[i] as f64;
        sh += hp;
        sh2 += hp * hp;
        n += 1;
    }
    if n == 0 {
        return None;
    }
    let nf = n as f64;
    let mh = sh / nf;
    Some(Stats {
        mean_l: (sl / nf) as f32,
        mean_a: (sa / nf) as f32,
        std_hp: ((sh2 / nf - mh * mh).max(0.0)).sqrt() as f32,
    })
}

/// 修复所有瑕疵（就地修改 `planes`）。返回实际修复的数量。
pub fn heal_lesions(
    planes: &mut LabPlanes,
    eligible: &GrayF32,
    map: &LesionMap,
    ed: f32,
    p: &HealParams,
) -> usize {
    let (w, h) = (planes.w, planes.h);
    if map.lesions.is_empty() {
        return 0;
    }
    let sigma_t = (p.texture_sigma * ed).max(1.5);
    let l_blur = fast_gaussian(&planes.l, sigma_t);
    let s = map.scale;
    let is_lesion = |gx: usize, gy: usize| -> bool {
        map.blob_s
            .sample_bilinear((gx as f32 + 0.5) * s, (gy as f32 + 0.5) * s)
            >= 0.25
    };
    let mut healed = 0usize;
    for les in &map.lesions {
        let r = les.r.max(1.5);
        let m = (p.margin * r).max(2.0);
        let band = (0.3 * r).max(2.0);
        let pad = (m + band).ceil() as usize + 3;
        let cx0 = les.x0.saturating_sub(pad);
        let cy0 = les.y0.saturating_sub(pad);
        let cx1 = (les.x0 + les.w + pad).min(w);
        let cy1 = (les.y0 + les.h + pad).min(h);
        if cx1 <= cx0 + 4 || cy1 <= cy0 + 4 {
            continue;
        }
        let (cw, ch) = (cx1 - cx0, cy1 - cy0);
        let mut core = vec![0u8; cw * ch];
        for y in 0..les.h {
            for x in 0..les.w {
                if les.mask[y * les.w + x] != 0 {
                    let (gx, gy) = (les.x0 + x, les.y0 + y);
                    if gx >= cx0 && gx < cx1 && gy >= cy0 && gy < cy1 {
                        core[(gy - cy0) * cw + (gx - cx0)] = 1;
                    }
                }
            }
        }
        let dist = chamfer_distance(&core, cw, ch);
        let inside: Vec<bool> = dist.iter().map(|d| *d <= m + band).collect();
        // 替换区不能碰到裁剪块边缘（否则说明瑕疵贴近区域边缘，放弃）
        let touches = (0..cw).any(|x| inside[x] || inside[(ch - 1) * cw + x])
            || (0..ch).any(|y| inside[y * cw] || inside[y * cw + cw - 1]);
        if touches {
            continue;
        }
        let weight: Vec<f32> = dist
            .iter()
            .map(|d| ((m + band - *d) / band).clamp(0.0, 1.0))
            .collect();
        let crop = |plane: &GrayF32| -> Vec<f32> {
            let mut v = vec![0.0f32; cw * ch];
            for y in 0..ch {
                v[y * cw..(y + 1) * cw]
                    .copy_from_slice(&plane.data[(cy0 + y) * w + cx0..(cy0 + y) * w + cx1]);
            }
            v
        };
        // 纹理源搜索（在原始 L 上进行，先于任何修改）
        let omega: Vec<(usize, usize)> = (0..cw * ch)
            .filter(|&i| inside[i])
            .map(|i| (i % cw, i / cw))
            .collect();
        let ring_idx: Vec<usize> = (0..cw * ch)
            .filter(|&i| !inside[i] && dist[i] <= m + band + 1.5 * r)
            .map(|i| (cy0 + i / cw) * w + cx0 + i % cw)
            .collect();
        let ring = stats(&planes.l, &planes.a, &l_blur, ring_idx.into_iter());
        let mut best: Option<(f32, i64, i64)> = None;
        if let Some(ring) = &ring {
            if p.texture_gain > 0.0 {
                for k in 0..16 {
                    let th = k as f32 * std::f32::consts::TAU / 16.0;
                    for rr in [2.2f32, 3.0, 4.0] {
                        let rad = rr * (r + m);
                        let dx = (rad * th.cos()).round() as i64;
                        let dy = (rad * th.sin()).round() as i64;
                        let mut ok = true;
                        let mut idx = Vec::with_capacity(omega.len());
                        for &(x, y) in &omega {
                            let gx = cx0 as i64 + x as i64 + dx;
                            let gy = cy0 as i64 + y as i64 + dy;
                            if gx < 0 || gy < 0 || gx >= w as i64 || gy >= h as i64 {
                                ok = false;
                                break;
                            }
                            let (gx, gy) = (gx as usize, gy as usize);
                            if eligible.data[gy * w + gx] < 0.9 || is_lesion(gx, gy) {
                                ok = false;
                                break;
                            }
                            idx.push(gy * w + gx);
                        }
                        if !ok {
                            continue;
                        }
                        let Some(st) = stats(&planes.l, &planes.a, &l_blur, idx.into_iter()) else {
                            continue;
                        };
                        let score = (st.mean_l - ring.mean_l).abs()
                            + 0.7 * (st.mean_a - ring.mean_a).abs()
                            + 1.5 * (st.std_hp - 1.3 * ring.std_hp).max(0.0)
                            + 0.5 * (0.5 * ring.std_hp - st.std_hp).max(0.0);
                        if best.map(|b| score < b.0).unwrap_or(true) {
                            best = Some((score, dx, dy));
                        }
                    }
                }
            }
        }
        // 调和插值 L / a / b
        let mut planes_c = [crop(&planes.l), crop(&planes.a), crop(&planes.b)];
        for v in planes_c.iter_mut() {
            solve_harmonic(v, &inside, cw, ch);
        }
        // 纹理移植（仅 L）
        if let Some((_, dx, dy)) = best {
            for &(x, y) in &omega {
                let gx = (cx0 as i64 + x as i64 + dx) as usize;
                let gy = (cy0 as i64 + y as i64 + dy) as usize;
                let gi = gy * w + gx;
                planes_c[0][y * cw + x] += p.texture_gain * (planes.l.data[gi] - l_blur.data[gi]);
            }
        }
        for (pi, plane) in [&mut planes.l, &mut planes.a, &mut planes.b]
            .into_iter()
            .enumerate()
        {
            for &(x, y) in &omega {
                let i = y * cw + x;
                let gi = (cy0 + y) * w + cx0 + x;
                let wgt = weight[i];
                plane.data[gi] += (planes_c[pi][i] - plane.data[gi]) * wgt;
            }
        }
        healed += 1;
    }
    healed
}

/// 检测 + 修复的便捷封装。返回修复数量。
pub fn heal(planes: &mut LabPlanes, eligible: &GrayF32, ed: f32, p: &HealParams) -> usize {
    if p.strength <= 0.0 {
        return 0;
    }
    let map = detect_lesions(planes, eligible, ed, p);
    heal_lesions(planes, eligible, &map, ed, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::ImgF32;

    fn synth() -> (ImgF32, usize, usize) {
        let (w, h) = (400, 300);
        let mut img = ImgF32::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // 平滑光照梯度 + 细颗粒
                let base = 0.70 + 0.10 * (x as f32 / w as f32) - 0.05 * (y as f32 / h as f32);
                let grain = (((x * 31 + y * 17) % 7) as f32 / 7.0 - 0.5) * 0.02;
                let mut px = [base + grain, base * 0.80 + grain, base * 0.70 + grain];
                // 椭圆疤痕 40×16（暗 + 偏红）
                let (dx, dy) = ((x as f32 - 150.0) / 20.0, (y as f32 - 120.0) / 8.0);
                if dx * dx + dy * dy < 1.0 {
                    px = [base * 0.78, base * 0.55, base * 0.50];
                }
                // 长阴影带（不应被处理）
                if (300..306).contains(&x) && (40..260).contains(&y) {
                    px = [base * 0.80, base * 0.64, base * 0.56];
                }
                img.data[y * w + x] = px;
            }
        }
        (img, w, h)
    }

    #[test]
    fn detects_elongated_scar_not_shadow_band_and_heals() {
        let (img, w, h) = synth();
        let mut planes = LabPlanes::from_img(&img);
        let eligible = GrayF32::filled(w, h, 1.0);
        let p = HealParams {
            work_ed: 200.0,
            ..HealParams::body_default()
        };
        let map = detect_lesions(&planes, &eligible, 200.0, &p);
        assert!(!map.lesions.is_empty(), "scar not detected");
        let covers_scar = map.lesions.iter().any(|l| {
            let (x, y) = (150usize, 120usize);
            x >= l.x0
                && x < l.x0 + l.w
                && y >= l.y0
                && y < l.y0 + l.h
                && l.mask[(y - l.y0) * l.w + (x - l.x0)] != 0
        });
        assert!(covers_scar, "lesion mask does not cover the scar");
        let covers_band = map
            .lesions
            .iter()
            .any(|l| l.x0 + l.w > 295 && l.x0 < 311 && l.y0 < 200 && l.y0 + l.h > 100);
        assert!(!covers_band, "shadow band wrongly healed");
        let before = planes.l.get(150, 120);
        let ring = planes.l.get(150, 140);
        let n = heal_lesions(&mut planes, &eligible, &map, 200.0, &p);
        assert!(n >= 1);
        let after = planes.l.get(150, 120);
        assert!(
            (after - ring).abs() < 2.5 && after > before + 5.0,
            "not healed: before {before} after {after} ring {ring}"
        );
        // 阴影带保持
        assert!((planes.l.get(303, 150) - LabPlanes::from_img(&img).l.get(303, 150)).abs() < 1e-3);
    }

    #[test]
    fn harmonic_fill_reproduces_linear_ramp() {
        let (w, h) = (40, 30);
        let mut vals: Vec<f32> = (0..w * h)
            .map(|i| (i % w) as f32 * 2.0 + (i / w) as f32 * 0.5)
            .collect();
        let truth = vals.clone();
        let inside: Vec<bool> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                (8..32).contains(&x) && (6..24).contains(&y)
            })
            .collect();
        for i in 0..w * h {
            if inside[i] {
                vals[i] = 0.0;
            }
        }
        solve_harmonic(&mut vals, &inside, w, h);
        let err = vals
            .iter()
            .zip(&truth)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 0.2, "max err {err}");
    }
}
