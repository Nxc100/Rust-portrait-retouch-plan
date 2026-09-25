//! 瑕疵（痘痘 / 痣 / 斑点）检测与填充。
//!
//! 检测（在"瞳距 ≈ work_ed 像素"的降采样分辨率上进行）：
//! 1. 多尺度 DoG：暗斑 = 外环均值 − 中心均值 > 阈值；红斑 = 中心 a − 外环 a > 阈值；
//! 2. 大尺度亮度梯度大的位置（边缘 / 皱纹一侧）不算；深阴影（外环 L 过低）不算；
//! 3. 逐峰值分割（响应局部极大为种子、向外生长到 45% 峰值、半径 ≤ 2.5r），每个峰各自成斑，
//!    再按面积 ≤ 6πr²、主轴伸长率 ≤ 2.2、填充率 ≥ 0.4 过滤：排除法令纹、阴影带、胡茬区域等长条 / 大面积响应，
//!    而密集的斑簇仍能逐个祛除；
//! 4. 线状结构否决：从峰值出发在 max(0.3·峰值, 阈值) 以上连通生长，能延伸到 4r 之外的峰属于一条暗线
//!    （纹身笔画、发丝、项链、眉尾）而不是孤立的斑，整条线都不处理——纹身的转角 / 笔画端点单看是紧凑的暗块，
//!    只靠形状规则会被一段段"祛除"（IMG_5785 颈部纹身）。探测用**不按可处理区裁剪**的原始响应：
//!    笔画只有一小段落在遮罩内时，遮罩内看它仍是孤立小块，延伸部分在遮罩外；
//! 5. 强暗痕群否决：高反差的候选（响应 ≥ `clutter_min_peak`，脸上 3 ≈ 9 L）若在 2.5r–6r 的环内还有其他
//!    强暗痕（≥ 0.5·峰值的像素占环面积 ≥ 6%），视为一组笔画 / 发丝 / 首饰的一部分——纹身细线常断成小段，
//!    连通性判据连不起来。孤立的痣照样祛除；脸上低反差的痘印 / 斑簇不受影响。
//!
//! 填充：归一化卷积（以非瑕疵皮肤像素为权重的高斯加权均值），在 L / a / b 三平面替换瑕疵像素。
//! 参考 FabSoften（CVPRW 2020）与像素蛋糕"皮肤瑕疵祛除"的思路，无需额外模型。

use crate::buffer::GrayF32;
use crate::color::lab::LabPlanes;
use crate::skin::guided::fast_gaussian;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BlemishParams {
    /// 0 关闭；1 为默认阈值；>1 更激进（阈值除以 strength）
    pub strength: f32,
    /// 最小 / 最大瑕疵半径（× 瞳距）
    pub min_radius: f32,
    pub max_radius: f32,
    /// 暗斑对比阈值（L 单位）
    pub dark_contrast: f32,
    /// 红斑对比阈值（a 单位）
    pub red_contrast: f32,
    /// 大尺度梯度阈值（L / 像素，在检测分辨率上），超过则视为边缘不处理
    pub edge_reject: f32,
    /// 外环 L 低于此值视为深阴影，不处理
    pub min_surround_l: f32,
    /// 填充核 σ（× 最大瑕疵半径）
    pub fill_sigma: f32,
    /// 检测分辨率对应的瞳距（像素）
    pub work_ed: f32,
    /// 连通域最大面积（× π r²）、最大长宽比、最小填充率
    pub max_area_factor: f32,
    pub max_aspect: f32,
    pub min_fill: f32,
    /// 滞后阈值：半阈值连通域（整条结构）的最大伸长率 / 最小填充率，超出则其中所有峰都不算瑕疵
    /// （鼻翼沟、法令纹、阴影带在半阈值下是长条，而斑簇仍是团块）
    pub max_aspect_low: f32,
    pub min_fill_low: f32,
    /// 滞后阈值：半阈值连通域的最大面积（× 单个斑的最大面积）。一整片相连的暗区（鼻侧阴影、眉尾、颊部暗块）
    /// 会被多个峰切成小块逐个通过，故按整体面积剔除；孤立斑在半阈值下的面积只有上限的 ~0.3 倍，
    /// 三两个相邻的小斑仍能通过（测得眉尾 / 鼻翼沟 / 耳廓处的环带通 RMS 与真斑重叠，不能用作判据）
    pub max_area_low_factor: f32,
    /// 强暗痕群否决适用的最低峰值响应（× 暗斑阈值）：只有这么强的候选才检查周围是否还有其他强暗痕。
    /// 脸上 3（约 9 L，痘印 / 斑簇不受影响）；身体上的细纹身线在粗检测分辨率下反差被抹淡，用更低的值
    pub clutter_min_peak: f32,
}

impl Default for BlemishParams {
    fn default() -> Self {
        Self {
            strength: 1.0,
            min_radius: 0.008,
            max_radius: 0.022,
            dark_contrast: 3.0,
            red_contrast: 3.2,
            edge_reject: 0.9,
            min_surround_l: 42.0,
            fill_sigma: 2.0,
            work_ed: 160.0,
            max_area_factor: 6.0,
            max_aspect: 2.2,
            min_fill: 0.4,
            max_aspect_low: 3.0,
            min_fill_low: 0.22,
            max_area_low_factor: 1.2,
            clutter_min_peak: 3.0,
        }
    }
}

pub(crate) fn gradient_mag(src: &GrayF32) -> GrayF32 {
    let (w, h) = (src.w, src.h);
    let mut out = GrayF32::new(w, h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let gx = src.get_clamped(x as isize + 1, y as isize)
                - src.get_clamped(x as isize - 1, y as isize);
            let gy = src.get_clamped(x as isize, y as isize + 1)
                - src.get_clamped(x as isize, y as isize - 1);
            *v = (gx * gx + gy * gy).sqrt() * 0.5;
        }
    });
    out
}

/// 检测尺度（≤ 1）。
pub fn detect_scale(ed: f32, p: &BlemishParams) -> f32 {
    (p.work_ed / ed.max(1.0)).clamp(0.12, 1.0)
}

pub(crate) fn down(src: &GrayF32, s: f32) -> GrayF32 {
    if s >= 0.999 {
        return src.clone();
    }
    src.resize(
        ((src.w as f32 * s).round() as usize).max(1),
        ((src.h as f32 * s).round() as usize).max(1),
    )
}

/// 逐峰值分割：
/// 1. 滞后阈值——半阈值（0.5）连通域是"整条结构"：伸长率 > `max_aspect_low` 或填充率 < `min_fill_low`
///    的（鼻翼沟、法令纹、阴影带、发丝）整体剔除，其中的峰不再参与；斑簇仍是团块，保留；
/// 2. 以响应图（1 = 阈值）的局部极大为种子，向外生长到 ≥ max(0.45·峰值, 0.5) 且距种子 ≤ 2.5r 的像素，
///    每个峰各自成斑（密集斑簇不会连成一片而被面积规则整体剔除），再按面积 / 伸长率 / 填充率过滤。返回 0/1 图。
///
/// `resp` 为按可处理区 / 边缘 / 阴影裁剪后的响应，`raw` 为未裁剪的原始响应（线状结构探测用）。
fn segment_peaks(
    resp: &[f32],
    raw: &[f32],
    w: usize,
    h: usize,
    r: f32,
    p: &BlemishParams,
) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let max_area = (p.max_area_factor * std::f32::consts::PI * r * r).max(4.0) as usize;
    let low: Vec<u8> = resp.iter().map(|v| (*v >= 0.5) as u8).collect();
    let max_area_low = (p.max_area_low_factor.max(1.0) * max_area as f32) as usize;
    let mut allowed = vec![false; w * h];
    for c in crate::skin::heal::components(&low, w, h, max_area_low) {
        let area = c.pixels.len();
        let bw = (c.x1 - c.x0 + 1) as f32;
        let bh = (c.y1 - c.y0 + 1) as f32;
        let fill = area as f32 / (bw * bh);
        if area <= max_area_low
            && crate::skin::heal::elongation(&c.pixels, w) <= p.max_aspect_low
            && fill >= p.min_fill_low
        {
            for &i in &c.pixels {
                allowed[i] = true;
            }
        }
    }
    let win = (r.round() as isize).max(1);
    let mut peaks: Vec<(f32, usize)> = (0..w * h)
        .into_par_iter()
        .filter_map(|i| {
            let v = resp[i];
            if v < 1.0 {
                return None;
            }
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            for dy in -win..=win {
                for dx in -win..=win {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if resp[j] > v || (resp[j] == v && j < i) {
                        return None;
                    }
                }
            }
            Some((v, i))
        })
        .collect();
    peaks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut assigned = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    let mut comp: Vec<usize> = Vec::new();
    let reach = 2.5 * r;
    let mut line = LineProbe::new(w, h, (LINE_REACH * r).max(3.0));
    for (pv, pi) in peaks {
        if assigned[pi] || !allowed[pi] {
            continue;
        }
        if line.is_line(raw, pi, (LINE_THRESHOLD * pv).max(1.0)) {
            // 整条线上的像素都标记为已处理，同一笔画上的其他峰不再重复探测
            for &j in line.visited() {
                assigned[j] = true;
            }
            continue;
        }
        if pv >= p.clutter_min_peak
            && clutter_fraction(raw, w, h, pi, r, 0.5 * pv) >= CLUTTER_FRACTION
        {
            continue;
        }
        let grow_t = (0.45 * pv).max(0.5);
        let (px, py) = ((pi % w) as f32, (pi / w) as f32);
        stack.clear();
        comp.clear();
        stack.push(pi);
        assigned[pi] = true;
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
        while let Some(i) = stack.pop() {
            comp.push(i);
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
            let nb = [
                (x > 0).then(|| i - 1),
                (x + 1 < w).then(|| i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < h).then(|| i + w),
            ];
            for j in nb.into_iter().flatten() {
                if assigned[j] || !allowed[j] || resp[j] < grow_t {
                    continue;
                }
                let (jx, jy) = ((j % w) as f32, (j / w) as f32);
                if (jx - px).powi(2) + (jy - py).powi(2) > reach * reach {
                    continue;
                }
                assigned[j] = true;
                stack.push(j);
            }
        }
        let area = comp.len();
        let bw = (x1 - x0 + 1) as f32;
        let bh = (y1 - y0 + 1) as f32;
        let fill = area as f32 / (bw * bh);
        if area <= max_area
            && crate::skin::heal::elongation(&comp, w) <= p.max_aspect
            && fill >= p.min_fill
        {
            for &i in &comp {
                out[i] = 1;
            }
        }
    }
    out
}

/// 线状结构否决的生长阈值（× 峰值）与判定距离（× 当前尺度半径 r）。
const LINE_THRESHOLD: f32 = 0.3;
const LINE_REACH: f32 = 4.0;
/// 强暗痕群否决：环的内外半径（× r）、环内强暗像素（≥ 0.5·峰值）的占比阈值
/// （适用的最低峰值见 `BlemishParams::clutter_min_peak`）。
const CLUTTER_INNER: f32 = 2.5;
const CLUTTER_OUTER: f32 = 6.0;
const CLUTTER_FRACTION: f32 = 0.06;

/// 峰值周围环带（`CLUTTER_INNER·r`–`CLUTTER_OUTER·r`）内响应 ≥ `threshold` 的像素占比（越界像素不计入分母）。
fn clutter_fraction(raw: &[f32], w: usize, h: usize, seed: usize, r: f32, threshold: f32) -> f32 {
    let (sx, sy) = ((seed % w) as isize, (seed / w) as isize);
    let (ri, ro) = (CLUTTER_INNER * r, CLUTTER_OUTER * r);
    let ext = ro.ceil() as isize;
    let (mut hit, mut total) = (0usize, 0usize);
    for dy in -ext..=ext {
        let y = sy + dy;
        if y < 0 || y >= h as isize {
            continue;
        }
        for dx in -ext..=ext {
            let x = sx + dx;
            if x < 0 || x >= w as isize {
                continue;
            }
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if d < ri || d > ro {
                continue;
            }
            total += 1;
            if raw[y as usize * w + x as usize] >= threshold {
                hit += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hit as f32 / total as f32
    }
}

/// 线状结构探测：从峰值像素出发，在响应 ≥ 阈值的像素上做连通生长（只在以峰为中心、半径 `reach` 的范围内），
/// 若有连通像素落在 `reach` 之外（即生长到了边界），说明这个峰是一条更长的暗结构的一部分。
struct LineProbe {
    w: usize,
    h: usize,
    reach: f32,
    /// 访问标记（代数计数，避免每次清空整张图）
    stamp: Vec<u32>,
    generation: u32,
    visited: Vec<usize>,
    stack: Vec<usize>,
}

impl LineProbe {
    fn new(w: usize, h: usize, reach: f32) -> Self {
        Self {
            w,
            h,
            reach,
            stamp: vec![0; w * h],
            generation: 0,
            visited: Vec::new(),
            stack: Vec::new(),
        }
    }

    /// 最近一次探测访问到的像素（`reach` 范围内）。
    fn visited(&self) -> &[usize] {
        &self.visited
    }

    fn is_line(&mut self, resp: &[f32], seed: usize, threshold: f32) -> bool {
        self.generation += 1;
        let g = self.generation;
        let (w, h) = (self.w, self.h);
        let (sx, sy) = ((seed % w) as f32, (seed / w) as f32);
        let r2 = self.reach * self.reach;
        self.visited.clear();
        self.stack.clear();
        self.stack.push(seed);
        self.stamp[seed] = g;
        let mut escaped = false;
        while let Some(i) = self.stack.pop() {
            self.visited.push(i);
            let (x, y) = (i % w, i / w);
            let nb = [
                (x > 0).then(|| i - 1),
                (x + 1 < w).then(|| i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < h).then(|| i + w),
            ];
            for j in nb.into_iter().flatten() {
                if self.stamp[j] == g || resp[j] < threshold {
                    continue;
                }
                self.stamp[j] = g;
                let (jx, jy) = ((j % w) as f32, (j / w) as f32);
                if (jx - sx).powi(2) + (jy - sy).powi(2) > r2 {
                    escaped = true; // 连通到了探测范围之外
                    continue;
                }
                self.stack.push(j);
            }
        }
        escaped
    }
}

/// 检测瑕疵，返回 0..1 软遮罩（全分辨率）。`eligible` 为可处理皮肤（>0.5 视为有效）。
pub fn detect_blemishes(
    planes: &LabPlanes,
    eligible: &GrayF32,
    ed: f32,
    p: &BlemishParams,
) -> GrayF32 {
    let (w, h) = (planes.w, planes.h);
    if p.strength <= 0.0 {
        return GrayF32::new(w, h);
    }
    let s = detect_scale(ed, p);
    let l = down(&planes.l, s);
    let a = down(&planes.a, s);
    let el = down(eligible, s);
    let (sw, sh) = (l.w, l.h);
    let ed_s = ed * s;
    let k = 1.0 / p.strength.max(0.05);
    let r_min = (p.min_radius * ed_s).max(1.0);
    let r_max = (p.max_radius * ed_s).max(r_min + 0.5);
    let scales = [r_min, (r_min * r_max).sqrt(), r_max];
    let mut blob = GrayF32::new(sw, sh);
    for &r in &scales {
        let inner_l = fast_gaussian(&l, r * 0.6);
        let outer_l = fast_gaussian(&l, r * 2.2);
        let inner_a = fast_gaussian(&a, r * 0.6);
        let outer_a = fast_gaussian(&a, r * 2.2);
        let grad = gradient_mag(&fast_gaussian(&l, r * 1.5));
        // 可处理区域向内收缩 2r，避免五官 / 发际边缘
        let el_er = fast_gaussian(&el, r * 1.0);
        let dark_t = p.dark_contrast * k;
        let red_t = p.red_contrast * k;
        let edge_t = p.edge_reject * k;
        let raw: Vec<f32> = (0..sw * sh)
            .into_par_iter()
            .map(|i| {
                let dark = (outer_l.data[i] - inner_l.data[i]) / dark_t;
                let red = (inner_a.data[i] - outer_a.data[i]) / red_t;
                dark.max(red)
            })
            .collect();
        let resp: Vec<f32> = (0..sw * sh)
            .into_par_iter()
            .map(|i| {
                if el_er.data[i] < 0.9
                    || grad.data[i] > edge_t
                    || outer_l.data[i] < p.min_surround_l
                {
                    0.0
                } else {
                    raw[i]
                }
            })
            .collect();
        let kept = segment_peaks(&resp, &raw, sw, sh, r, p);
        // 按尺度稍作膨胀
        let kf: Vec<f32> = kept.iter().map(|v| *v as f32).collect();
        let dil = fast_gaussian(&GrayF32::from_vec(sw, sh, kf), r * 1.0);
        blob.data.par_iter_mut().zip(&dil.data).for_each(|(b, v)| {
            if *v > 0.10 {
                *b = 1.0;
            }
        });
    }
    // 软边：向外羽化约一个瑕疵半径，避免填充边界形成"光环"
    let blob = fast_gaussian(&blob, (r_max * 0.8).max(1.0));
    let mut full = if s < 0.999 { blob.resize(w, h) } else { blob };
    full.data
        .par_iter_mut()
        .zip(&eligible.data)
        .for_each(|(b, e)| {
            *b = if *e < 0.5 {
                0.0
            } else {
                (*b * 1.3).clamp(0.0, 0.92)
            };
        });
    full
}

/// 归一化卷积填充：瑕疵像素 = 周围非瑕疵皮肤像素的高斯加权均值（填充值在降采样分辨率上计算）。
pub fn inpaint_blobs(
    planes: &mut LabPlanes,
    blob: &GrayF32,
    eligible: &GrayF32,
    sigma: f32,
    scale: f32,
) {
    let (w, h) = (planes.w, planes.h);
    if !blob.data.iter().any(|v| *v > 0.01) {
        return;
    }
    let s = scale.clamp(0.12, 1.0);
    let blob_s = down(blob, s);
    let el_s = down(eligible, s);
    let weight: Vec<f32> = blob_s
        .data
        .par_iter()
        .zip(&el_s.data)
        .map(|(b, e)| if *b > 0.05 || *e < 0.5 { 0.0 } else { 1.0 })
        .collect();
    let weight = GrayF32::from_vec(blob_s.w, blob_s.h, weight);
    let den = fast_gaussian(&weight, (sigma * s).max(1.0));
    for plane in [&mut planes.l, &mut planes.a, &mut planes.b] {
        let ps = down(plane, s);
        let wx: Vec<f32> = ps
            .data
            .par_iter()
            .zip(&weight.data)
            .map(|(v, m)| v * m)
            .collect();
        let num = fast_gaussian(&GrayF32::from_vec(ps.w, ps.h, wx), (sigma * s).max(1.0));
        let fill_s: Vec<f32> = num
            .data
            .par_iter()
            .zip(&den.data)
            .zip(&ps.data)
            .map(|((n, d), v)| if *d > 1e-4 { n / d } else { *v })
            .collect();
        let fill = GrayF32::from_vec(ps.w, ps.h, fill_s);
        let fill = if s < 0.999 { fill.resize(w, h) } else { fill };
        plane.data.par_iter_mut().enumerate().for_each(|(i, v)| {
            let b = blob.data[i];
            if b > 0.0 {
                *v += (fill.data[i] - *v) * b;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::ImgF32;

    #[test]
    fn detects_and_fills_dark_spot_but_not_shadow_band() {
        let (w, h) = (240, 160);
        let mut img = ImgF32::filled(w, h, [0.80, 0.62, 0.55]);
        // 圆形暗斑（半径 4）
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 - 80.0).powi(2) + (y as f32 - 60.0).powi(2)).sqrt();
                if d < 4.0 {
                    img.data[y * w + x] = [0.55, 0.40, 0.35];
                }
                // 长条阴影带（法令纹）
                if (170..176).contains(&x) && (20..140).contains(&y) {
                    img.data[y * w + x] = [0.60, 0.44, 0.38];
                }
            }
        }
        let mut planes = LabPlanes::from_img(&img);
        let eligible = GrayF32::filled(w, h, 1.0);
        let p = BlemishParams {
            work_ed: 200.0,
            ..Default::default()
        };
        let blob = detect_blemishes(&planes, &eligible, 200.0, &p);
        assert!(
            blob.get(80, 60) > 0.3,
            "spot not detected: {}",
            blob.get(80, 60)
        );
        assert!(
            blob.get(173, 80) < 0.05,
            "shadow band wrongly detected: {}",
            blob.get(173, 80)
        );
        let before = planes.l.get(80, 60);
        inpaint_blobs(&mut planes, &blob, &eligible, 10.0, detect_scale(200.0, &p));
        let after = planes.l.get(80, 60);
        assert!(after > before + 4.0, "not filled: {before} -> {after}");
    }

    /// 纹身式笔画：墨迹 L 比皮肤低约 30、宽 3 px，笔画断成一段段（真实纹身的细线在检测分辨率上就是这样）；
    /// 另有一个孤立的痣（同样高反差）与一个低反差的痘印。
    fn tattoo_scene(w: usize, h: usize) -> ImgF32 {
        let mut img = ImgF32::filled(w, h, [0.80, 0.62, 0.55]);
        let ink = [0.33, 0.28, 0.30];
        let mut stroke = |x0: f32, y0: f32, x1: f32, y1: f32, dashed: bool| {
            let len = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
            let n = (len * 2.0) as usize;
            for k in 0..=n {
                let t = k as f32 / n as f32;
                if dashed && ((t * len) as usize % 12) >= 8 {
                    continue; // 4 px 的断口
                }
                let (cx, cy) = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (x, y) = ((cx as i32 + dx) as usize, (cy as i32 + dy) as usize);
                        img.data[y * w + x] = ink;
                    }
                }
            }
        };
        let pts = [
            (60.0, 60.0),
            (120.0, 50.0),
            (140.0, 110.0),
            (95.0, 150.0),
            (50.0, 115.0),
        ];
        for k in 0..5 {
            let (a, b) = (pts[k], pts[(k + 1) % 5]);
            stroke(a.0, a.1, b.0, b.1, k % 2 == 1);
        }
        // 五边形里的细三角（断续）
        stroke(100.0, 80.0, 125.0, 95.0, true);
        stroke(125.0, 95.0, 105.0, 120.0, true);
        stroke(105.0, 120.0, 100.0, 80.0, true);
        // 钩
        stroke(170.0, 40.0, 190.0, 90.0, false);
        stroke(190.0, 90.0, 175.0, 120.0, true);
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 - 250.0).powi(2) + (y as f32 - 170.0).powi(2)).sqrt();
                if d < 4.0 {
                    img.data[y * w + x] = ink; // 孤立的痣
                }
                let d = ((x as f32 - 250.0).powi(2) + (y as f32 - 60.0).powi(2)).sqrt();
                if d < 4.0 {
                    img.data[y * w + x] = [0.70, 0.52, 0.46]; // 低反差痘印
                }
            }
        }
        img
    }

    #[test]
    fn keeps_tattoo_like_strokes_but_removes_isolated_spots() {
        let (w, h) = (300, 220);
        let planes = LabPlanes::from_img(&tattoo_scene(w, h));
        let p = BlemishParams {
            work_ed: 200.0,
            ..Default::default()
        };
        // 全图可处理，以及只有纹身的一角落在可处理区里（纹身跨过脸部遮罩的边界）
        let partial = GrayF32::from_vec(
            w,
            h,
            (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    let corner = (100..160).contains(&x) && (70..135).contains(&y);
                    let hook_end = (165..200).contains(&x) && (95..135).contains(&y);
                    let spots = (225..275).contains(&x)
                        && ((35..85).contains(&y) || (145..195).contains(&y));
                    if corner || hook_end || spots {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        );
        for eligible in [GrayF32::filled(w, h, 1.0), partial] {
            let blob = detect_blemishes(&planes, &eligible, 200.0, &p);
            assert!(
                blob.get(250, 170) > 0.3,
                "isolated mole missed: {}",
                blob.get(250, 170)
            );
            assert!(
                blob.get(250, 60) > 0.3,
                "faint spot missed: {}",
                blob.get(250, 60)
            );
            let on_strokes = (0..h)
                .flat_map(|y| (0..210).map(move |x| (x, y)))
                .filter(|&(x, y)| blob.get(x, y) > 0.3)
                .count();
            assert_eq!(on_strokes, 0, "tattoo strokes treated as blemishes");
        }
    }
}
