//! 颈纹淡化（奶油肌）：抹平脖子上的颈纹（连同横过脖子的细碎发），保留项链、纹身、胡茬与脖子 / 下颌的轮廓。
//!
//! 依据（doc/test_report_neck.md）：像素蛋糕「奶油肌」在脖子上把颈纹所在的中等局部能量（L 起伏 3–8）压到
//! 三到五成，暗纹与两侧的亮脊一起被压平；改动图上还有一根根锐利的亮线，恰好贴着每一条颈纹的线芯与碎发——
//! 细线被直接填平。项链、纹身、胡茬、轮廓原样保留。手臂、胸口等其余身体皮肤没有这一步。男女都做。
//!
//! 每张脸的脖子单独处理，尺度用该脸的瞳距当量：
//! 1. **区域**：人脸解析的脖子类（羽化）× 身体遮罩 × 语义皮肤门控 × 异物门控。异物有两种判据：色度与脖子上
//!    皮肤色度之比低（项链、胡茬、黑发只有皮肤的 0.4–0.55，颈纹是颜色深一些的皮肤，≈ 1），或者是比两侧皮肤
//!    暗得多的细结构（灰度闭运算的黑顶帽：颈纹只有 3–8 L，纹身墨迹、项链的影子、碎发 12–30 L；纹身墨迹的色度
//!    接近皮肤的七成，只靠色度认不全）。异物之间的窄缝（闭运算 0.08 瞳距当量）连成一片——纹身笔画淡下去的
//!    尾迹是皮肤色、和颈纹一样深，只能从它夹在浓笔画之间认出；异物周围一圈（膨胀 0.03 瞳距当量）也不处理——
//!    项链投在皮肤上的影子、描边是皮肤色的细暗线，抹掉后项链像是贴上去的。区域在处理前的颜色上确定（匀肤之后
//!    异物的色差会变小）。
//! 2. **平滑**：`Ls = G(L; σf)` 去掉细颗粒（毛孔纹理不动）；`Ls` 里的异物像素用周围皮肤的归一化平均填上，得到
//!    只有皮肤的 `Lk`（否则导向滤波会把项链的暗色拉进旁边的皮肤，形成光晕）；`S = GF(Lk; r, eps)` 为导向滤波：
//!    窗口内起伏的方差远小于 eps 的被抹平（颈纹），远大于 eps 的保留（轮廓、下颌阴影、硬阴影的边缘）。只在局部
//!    中频能量 E 落在能量窗内的地方叠加 `strength·(S − Lk)`：更低的是本来就平滑的皮肤，更高的是胡茬、发丝等
//!    纹理繁杂处。区域在解析图覆盖范围的边缘渐隐（脖子可能延伸到范围之外，不能留下硬边）。
//! 3. **细线填平**：平滑只作用在去掉细颗粒的 `Ls` 上，颈纹、碎发锐利的线芯留在细颗粒里。方向线段闭运算量出
//!    细长暗线的深度（`morph::dark_line_depth`：比线段短的暗点——胡茬、毛孔、痣——为 0）；原图上线深 3–6 L
//!    以上的像素是"线"（皮沟等细纹理更浅；深过 12–20 L 的是项链的影子、深色发丝、纹身，连同周围一圈不算），
//!    把它平滑之后仍比两侧暗的部分按 `line_fill` 填平。细颗粒在填平时原样保留。

use crate::buffer::GrayF32;
use crate::color::lab::LabPlanes;
use crate::face::parsing::{ParseMap, NECK_CLASSES};
use crate::skin::gaussian::gaussian_blur_gray;
use crate::skin::guided::{fast_gaussian, guided_filter, masked_gaussian};
use crate::skin::masks::{bbox_of, crop_gray, paste_gray};
use crate::skin::morph::{close, dark_line_depth, dilate};
use crate::skin::smoothstep;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// 颈纹淡化参数（`CreamParams::neck`）。
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NeckParams {
    /// 强度 0..1（0 关闭整步）：能量窗内用保边平滑的结果替换（去掉细颗粒后的）亮度的比例；也乘在细线填平上
    pub strength: f32,
    /// 保边平滑（导向滤波）的半径（× 瞳距当量）
    pub radius: f32,
    /// 导向滤波的正则（L²）：窗口内起伏的方差远小于它的被抹平（颈纹），远大于它的保留（轮廓、下颌阴影）
    pub eps: f32,
    /// 能量窗（局部中频能量，L）：`energy_lo / 2 → energy_lo` 渐入，`energy_hi → 2·energy_hi` 渐出
    pub energy_lo: f32,
    pub energy_hi: f32,
    /// 细线填平 0..1（再乘 `strength`）：原图上是细长暗线（颈纹的线芯、碎发）的像素，平滑之后仍比两侧暗的部分
    /// 按此比例填平（不会比两侧更亮）；比线段短的暗点（胡茬、毛孔、痣）不动
    pub line_fill: f32,
}

impl Default for NeckParams {
    fn default() -> Self {
        Self {
            strength: 1.0,
            radius: 0.05,
            eps: 80.0,
            energy_lo: 0.8,
            energy_hi: 6.0,
            line_fill: 1.0,
        }
    }
}

/// 区域坐标下的矩形。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
}

/// 一张脸的脖子：矩形与其中的处理权重、皮肤色置信度（0..1，异物为 0）、原图上是细长暗线的程度
/// （0..1；不做细线填平时为 None）。
pub struct NeckZone {
    pub rect: Rect,
    pub weight: GrayF32,
    pub skin: GrayF32,
    pub line: Option<GrayF32>,
}

/// 脖子区域的羽化（× 瞳距当量）
const FEATHER: f32 = 0.04;
/// 皮肤色度参考：脖子类内的归一化高斯（× 瞳距当量）
const CHROMA_REF_SIGMA: f32 = 0.08;
/// 异物判据一：色度 / 皮肤色度低于此区间（平滑过渡）
const SKIN_CHROMA_RATIO: (f32, f32) = (0.55, 0.8);
/// 异物判据二：暗结构的深度（黑顶帽，L）高于此区间（平滑过渡）
const FOREIGN_DEPTH: (f32, f32) = (10.0, 16.0);
/// 黑顶帽闭运算的窗口半宽（× 瞳距当量）：填得平纹身笔画、项链影子，填不平下颌的大片阴影
const DEPTH_WINDOW: f32 = 0.05;
/// 量深度前的去噪（像素）：压住噪点与毛孔；更细的去噪会把深一些的颈纹也量成异物
const DEPTH_DENOISE_SIGMA: f32 = 1.4;
/// 异物之间窄于约两倍此宽度（× 瞳距当量）的缝也算异物：纹身笔画之间的淡墨迹、并排的链条之间
const FOREIGN_GAP: f32 = 0.08;
/// 异物周围不处理的宽度（× 瞳距当量）：项链的影子与描边、纹身与碎发的边缘
const FOREIGN_GUARD: f32 = 0.03;
/// 局部中频能量的平滑尺度（× 瞳距当量，与三频段磨皮相同）
const ENERGY_SIGMA: f32 = 0.05;
/// 脖子区域外扩的边距（× 瞳距当量）：羽化尾部与滤波窗口
const MARGIN: f32 = 0.15;
/// 解析图覆盖范围边缘的渐隐宽度（× 瞳距当量）
const BORDER_FADE: f32 = 0.1;
/// 可见的脖子面积（× 瞳距当量²）小于此值时不处理（被衣物 / 围巾遮住）
const MIN_AREA: f32 = 0.02;
/// 细颗粒分界的下限（像素）：小脸上的毛孔、颗粒只有一两个像素，分界再细就会被当作颈纹抹掉。比三频段磨皮的
/// 下限（0.5 像素）高：那里中频只是部分平滑，这里是整段换成平滑结果（瞳距当量 96 的脖子纹理会只剩一半）
const GRAIN_SIGMA_MIN: f32 = 1.0;
/// 细线：方向线段的半长（× 瞳距当量）；比线段短的暗点不算线
const LINE_HALF: f32 = 0.04;
/// 细线：线段的方向数（细线与线段的夹角不超过 11°）
const LINE_DIRECTIONS: usize = 8;
/// 细线：原图上的线深（L）在此区间渐入；更浅的是皮沟等细纹理
const LINE_DEPTH: (f32, f32) = (3.0, 6.0);
/// 细线：原图上的线深（L）在此区间渐出；更深的是项链的影子、深色发丝、纹身笔画，与异物一样保留
const LINE_DEPTH_MAX: (f32, f32) = (12.0, 20.0);
/// 细线：量线深前的去噪（像素，精确高斯）
const LINE_DENOISE_SIGMA: f32 = 0.7;

/// 解析图覆盖范围（原图坐标，含边界）与区域（原图中左上角 `origin`、尺寸 `size`）的交，区域坐标。
pub fn parse_rect(parse: &ParseMap, origin: (usize, usize), size: (usize, usize)) -> Option<Rect> {
    let (bx0, by0, bx1, by1) = parse.bbox;
    let clip = |v: i32, o: usize, n: usize| (v.max(0) as usize).saturating_sub(o).min(n);
    let (x0, y0) = (clip(bx0, origin.0, size.0), clip(by0, origin.1, size.1));
    let (x1, y1) = (
        clip(bx1.saturating_add(1), origin.0, size.0),
        clip(by1.saturating_add(1), origin.1, size.1),
    );
    (x1 > x0 && y1 > y0).then_some(Rect {
        x0,
        y0,
        w: x1 - x0,
        h: y1 - y0,
    })
}

/// 构建一张脸的脖子区域。`origin` 为区域左上角在原图中的坐标，`rect` 通常来自 [`parse_rect`]；
/// `allowed` 为 `rect` 内允许处理的权重（身体遮罩 × 语义皮肤门控）；`planes` 为区域的 Lab（处理前）；
/// `p` 决定是否要量细线。看不到脖子（被遮挡、太小）时返回 None。
pub fn neck_zone(
    parse: &ParseMap,
    origin: (usize, usize),
    rect: Rect,
    planes: &LabPlanes,
    allowed: &GrayF32,
    ed: f32,
    p: &NeckParams,
) -> Option<NeckZone> {
    assert_eq!((allowed.w, allowed.h), (rect.w, rect.h));
    let ed = ed.max(8.0);
    let neck = parse.mask_in(
        origin.0 + rect.x0,
        origin.1 + rect.y0,
        rect.w,
        rect.h,
        &NECK_CLASSES,
    );
    let visible: f32 = neck
        .data
        .par_iter()
        .zip(&allowed.data)
        .map(|(n, a)| n * a)
        .sum();
    if visible < MIN_AREA * ed * ed {
        return None;
    }
    // 收紧到脖子的包围盒（外扩羽化与滤波的边距）
    let (bx0, by0, bx1, by1) = bbox_of(&neck, 0.5, (MARGIN * ed).ceil() as usize)?;
    let sub = Rect {
        x0: rect.x0 + bx0,
        y0: rect.y0 + by0,
        w: bx1 - bx0,
        h: by1 - by0,
    };
    let neck = crop_gray(&neck, bx0, by0, sub.w, sub.h);
    let allowed = crop_gray(allowed, bx0, by0, sub.w, sub.h);
    let mut weight = fast_gaussian(&neck, (FEATHER * ed).max(1.0));
    let fade = (BORDER_FADE * ed).max(1.0);
    // 渐隐只在解析图覆盖范围自己的边缘（全图坐标，含边界）：图像边缘之外没有脖子，不必渐隐
    let (px0, py0, px1, py1) = parse.bbox;
    let (gx0, gy0) = ((origin.0 + sub.x0) as i64, (origin.1 + sub.y0) as i64);
    weight
        .data
        .par_iter_mut()
        .zip(&allowed.data)
        .enumerate()
        .for_each(|(i, (w, a))| {
            let (x, y) = (gx0 + (i % sub.w) as i64, gy0 + (i / sub.w) as i64);
            let edge = (x - px0 as i64)
                .min(px1 as i64 - x)
                .min(y - py0 as i64)
                .min(py1 as i64 - y)
                .max(0) as f32;
            *w = w.clamp(0.0, 1.0) * a * smoothstep(0.0, fade, edge);
        });
    let skin = skin_gate(planes, sub, &neck, ed);
    weight.mul_inplace(&skin);
    let line = (p.line_fill > 0.0).then(|| line_map(planes, sub, ed));
    Some(NeckZone {
        rect: sub,
        weight,
        skin,
        line,
    })
}

/// 原图上是细线的程度（0..1）：线深在 [`LINE_DEPTH`] 渐入；太深的线（[`LINE_DEPTH_MAX`]）连同周围一圈
/// （[`FOREIGN_GUARD`]）不算——它们的边缘在去噪后变浅，单看像素会被当成普通细线。
fn line_map(planes: &LabPlanes, r: Rect, ed: f32) -> GrayF32 {
    let depth = line_depth(&planes.l, r, None, ed);
    let ((lo, hi), (deep_lo, deep_hi)) = (LINE_DEPTH, LINE_DEPTH_MAX);
    let mut too_deep = depth.clone();
    too_deep.map_inplace(|v| smoothstep(deep_lo, deep_hi, v));
    let too_deep = dilate(&too_deep, (FOREIGN_GUARD * ed).round().max(1.0) as usize);
    let line = depth
        .data
        .par_iter()
        .zip(&too_deep.data)
        .map(|(d, t)| smoothstep(lo, hi, *d) * (1.0 - t))
        .collect();
    GrayF32::from_vec(r.w, r.h, line)
}

/// 矩形 `r` 内细长暗线的深度（L）：去噪后按 [`LINE_HALF`] / [`LINE_DIRECTIONS`] 量（见 [`dark_line_depth`]）。
/// 在区域 `l` 上四周外扩一个线段半长（与去噪半径）量，再裁回 `r`：线段在 `r` 的边缘不会因出界而把台阶、
/// 阴影量成线。`inner` 给出时代替 `l` 在 `r` 内的内容（平滑后的亮度）。
fn line_depth(l: &GrayF32, r: Rect, inner: Option<&GrayF32>, ed: f32) -> GrayF32 {
    let half = (LINE_HALF * ed).round().max(2.0) as usize;
    let pad = half + (3.0 * LINE_DENOISE_SIGMA).ceil() as usize;
    let (px0, py0) = (r.x0.saturating_sub(pad), r.y0.saturating_sub(pad));
    let (px1, py1) = ((r.x0 + r.w + pad).min(l.w), (r.y0 + r.h + pad).min(l.h));
    let mut padded = crop_gray(l, px0, py0, px1 - px0, py1 - py0);
    if let Some(inner) = inner {
        paste_gray(&mut padded, inner, r.x0 - px0, r.y0 - py0);
    }
    let depth = dark_line_depth(
        &gaussian_blur_gray(&padded, LINE_DENOISE_SIGMA),
        half,
        LINE_DIRECTIONS,
    );
    crop_gray(&depth, r.x0 - px0, r.y0 - py0, r.w, r.h)
}

/// 皮肤门控（1 = 皮肤，0 = 异物及其周围）：色度相对皮肤低（[`SKIN_CHROMA_RATIO`]）或暗结构深
/// （[`FOREIGN_DEPTH`]）的像素是异物；闭合 [`FOREIGN_GAP`] 的窄缝、膨胀 [`FOREIGN_GUARD`] 后羽化。
fn skin_gate(planes: &LabPlanes, r: Rect, neck: &GrayF32, ed: f32) -> GrayF32 {
    let a = crop_gray(&planes.a, r.x0, r.y0, r.w, r.h);
    let b = crop_gray(&planes.b, r.x0, r.y0, r.w, r.h);
    let chroma: Vec<f32> = a
        .data
        .par_iter()
        .zip(&b.data)
        .map(|(a, b)| a.hypot(*b))
        .collect();
    let chroma = GrayF32::from_vec(r.w, r.h, chroma);
    let skin = masked_gaussian(&chroma, neck, CHROMA_REF_SIGMA * ed);
    let local = fast_gaussian(&chroma, 1.0);
    // 闭运算把窄于窗口的暗结构填成两侧皮肤的亮度，与原亮度之差即其深度（台阶、大片阴影为 0）
    let l = fast_gaussian(
        &crop_gray(&planes.l, r.x0, r.y0, r.w, r.h),
        DEPTH_DENOISE_SIGMA,
    );
    let filled = close(&l, (DEPTH_WINDOW * ed).round().max(1.0) as usize);
    let (c_lo, c_hi) = SKIN_CHROMA_RATIO;
    let (d_lo, d_hi) = FOREIGN_DEPTH;
    let foreign: Vec<f32> = local
        .data
        .par_iter()
        .zip(&skin.data)
        .zip(l.data.par_iter().zip(&filled.data))
        .map(|((c, s), (v, f))| {
            let greyish = 1.0 - smoothstep(c_lo, c_hi, c / s.max(1.0));
            let deep = smoothstep(d_lo, d_hi, f - v);
            greyish.max(deep)
        })
        .collect();
    let foreign = GrayF32::from_vec(r.w, r.h, foreign);
    let gap = (FOREIGN_GAP * ed).round().max(1.0) as usize;
    let guard = (FOREIGN_GUARD * ed).round().max(1.0);
    let foreign = dilate(&close(&foreign, gap), guard as usize);
    let mut gate = fast_gaussian(&foreign, 0.5 * guard);
    gate.map_inplace(|f| (1.0 - f).clamp(0.0, 1.0));
    gate
}

/// 颈纹淡化：原地修改区域的亮度平面 `l`。`fine_sigma` / `mid_sigma` 为像素单位的细颗粒 / 中频分界，
/// 与三频段磨皮一致（细颗粒不动，能量按中频计算；细颗粒分界不低于 1 像素，见 `GRAIN_SIGMA_MIN`）。
pub fn soften_neck(
    l: &mut GrayF32,
    zone: &NeckZone,
    ed: f32,
    p: &NeckParams,
    fine_sigma: f32,
    mid_sigma: f32,
) {
    if p.strength <= 0.0 {
        return;
    }
    let ed = ed.max(8.0);
    let fine_sigma = fine_sigma.max(GRAIN_SIGMA_MIN);
    let Rect { x0, y0, w, h } = zone.rect;
    let mut sub = crop_gray(l, x0, y0, w, h);
    let ls = fast_gaussian(&sub, fine_sigma);
    let radius = (p.radius * ed).round().max(1.0) as usize;
    // 异物（项链、纹身、胡茬）用周围皮肤的平均填上，平滑与能量都只看皮肤
    let fill = masked_gaussian(&ls, &zone.skin, radius as f32);
    let skin_only: Vec<f32> = ls
        .data
        .par_iter()
        .zip(&fill.data)
        .zip(&zone.skin.data)
        .map(|((v, f), s)| v * s + f * (1.0 - s))
        .collect();
    let skin_only = GrayF32::from_vec(w, h, skin_only);
    let gm = fast_gaussian(&skin_only, mid_sigma.max(fine_sigma + 0.5));
    let mid_sq: Vec<f32> = skin_only
        .data
        .par_iter()
        .zip(&gm.data)
        .map(|(a, b)| (a - b) * (a - b))
        .collect();
    let energy = fast_gaussian(
        &GrayF32::from_vec(w, h, mid_sq),
        (ENERGY_SIGMA * ed).max(1.0),
    );
    let smooth = guided_filter(&skin_only, &skin_only, radius, p.eps.max(1e-3), 1);
    let ln2 = std::f32::consts::LN_2;
    let ln_lo = p.energy_lo.max(1e-3).ln();
    let ln_hi = p.energy_hi.max(p.energy_lo * 1.01).max(1e-3).ln();
    sub.data.par_iter_mut().enumerate().for_each(|(i, v)| {
        let weight = zone.weight.data[i];
        if weight <= 0.0 {
            return;
        }
        let ln_e = energy.data[i].max(1e-12).sqrt().ln();
        let window =
            smoothstep(ln_lo - ln2, ln_lo, ln_e) * (1.0 - smoothstep(ln_hi, ln_hi + ln2, ln_e));
        // 平滑量在"只有皮肤"的图上取：异物处两者都是周围皮肤的平均，差为 0——即使门控没有完全归零，
        // 也不会把纹身、项链当作起伏抹掉
        *v += p.strength * weight * window * (smooth.data[i] - skin_only.data[i]);
    });
    let line_fill = (p.strength * p.line_fill).clamp(0.0, 1.0);
    if let (Some(line), true) = (&zone.line, line_fill > 0.0) {
        // 平滑之后线芯仍比两侧暗多少：原图上判定为细线的像素按比例填平（量线深前的去噪只用于测量，
        // 加回的是平滑的量，细颗粒不受影响）
        let depth = line_depth(l, zone.rect, Some(&sub), ed);
        sub.data
            .par_iter_mut()
            .zip(zone.weight.data.par_iter().zip(&line.data))
            .zip(&depth.data)
            .for_each(|((v, (weight, line)), d)| *v += line_fill * weight * line * d);
    }
    paste_gray(l, &sub, x0, y0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::face::parsing::{CLS_BACKGROUND, CLS_NECK};
    use crate::geom::Affine;

    const W: usize = 200;
    const H: usize = 160;
    const ED: f32 = 100.0;
    const FINE: f32 = 1.0;
    const MID: f32 = 3.5;
    /// 皮肤的 L / a / b
    const SKIN: [f32; 3] = [70.0, 12.0, 14.0];
    const WRINKLE_ROWS: [usize; 3] = [50, 70, 90];

    /// 覆盖原图 [0, 200)²、每个类别像素对应 4×4 原图像素的解析图；脖子约为 x ∈ [18, 178)、y ∈ [18, 138)。
    fn parse() -> ParseMap {
        let size = 50;
        let classes = (0..size * size)
            .map(|i| {
                let (u, v) = (i % size, i / size);
                if (5..45).contains(&u) && (5..35).contains(&v) {
                    CLS_NECK
                } else {
                    CLS_BACKGROUND
                }
            })
            .collect();
        ParseMap {
            size,
            classes,
            affine: Affine {
                a: 0.25,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 0.25,
                f: 0.0,
            },
            bbox: (0, 0, 199, 199),
        }
    }

    /// `h` 行的皮肤平面：`dark(x, y)` 为各像素变暗的量，`grey(x, y)` 为真的像素色度降到皮肤的 15%。
    fn scene(
        h: usize,
        dark: impl Fn(usize, usize) -> f32,
        grey: impl Fn(usize, usize) -> bool,
    ) -> LabPlanes {
        let mut planes = LabPlanes {
            w: W,
            h,
            l: GrayF32::new(W, h),
            a: GrayF32::new(W, h),
            b: GrayF32::new(W, h),
        };
        for y in 0..h {
            for x in 0..W {
                let i = y * W + x;
                let k = if grey(x, y) { 0.15 } else { 1.0 };
                planes.l.data[i] = SKIN[0] - dark(x, y);
                planes.a.data[i] = SKIN[1] * k;
                planes.b.data[i] = SKIN[2] * k;
            }
        }
        planes
    }

    /// 颈纹：[`WRINKLE_ROWS`] 处 σ = 1.5 px、深 `depth` 的横向暗线。
    fn wrinkles(depth: f32) -> impl Fn(usize, usize) -> f32 {
        move |_, y| {
            WRINKLE_ROWS
                .iter()
                .map(|&c| depth * (-((y as f32 - c as f32).powi(2)) / 4.5).exp())
                .sum()
        }
    }

    fn zone_with(planes: &LabPlanes, p: &NeckParams) -> Option<NeckZone> {
        let pm = parse();
        let rect = parse_rect(&pm, (0, 0), (planes.w, planes.h))?;
        let allowed = GrayF32::from_vec(rect.w, rect.h, vec![1.0; rect.w * rect.h]);
        neck_zone(&pm, (0, 0), rect, planes, &allowed, ED, p)
    }

    fn zone_of(planes: &LabPlanes) -> Option<NeckZone> {
        zone_with(planes, &NeckParams::default())
    }

    fn soften_with(planes: &LabPlanes, p: &NeckParams) -> GrayF32 {
        let zone = zone_with(planes, p).expect("neck visible");
        let mut l = planes.l.clone();
        soften_neck(&mut l, &zone, ED, p, FINE, MID);
        l
    }

    fn soften(planes: &LabPlanes, strength: f32) -> GrayF32 {
        let p = NeckParams {
            strength,
            ..NeckParams::default()
        };
        soften_with(planes, &p)
    }

    /// 暗线的深度：两侧 ±8 px 的平均减线心。
    fn line_depth(l: &GrayF32, x: usize, y: usize) -> f32 {
        0.5 * (l.get(x, y - 8) + l.get(x, y + 8)) - l.get(x, y)
    }

    fn max_change(
        a: &GrayF32,
        b: &GrayF32,
        xs: std::ops::Range<usize>,
        ys: std::ops::Range<usize>,
    ) -> f32 {
        ys.flat_map(|y| xs.clone().map(move |x| (x, y)))
            .map(|(x, y)| (a.get(x, y) - b.get(x, y)).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn parse_rect_is_the_parse_crop_within_the_region() {
        let mut pm = parse();
        pm.bbox = (-5, 10, 60, 300);
        let r = parse_rect(&pm, (20, 0), (100, 100)).unwrap();
        assert_eq!(
            r,
            Rect {
                x0: 0,
                y0: 10,
                w: 41,
                h: 90
            }
        );
        pm.bbox = (200, 200, 250, 250);
        assert_eq!(parse_rect(&pm, (0, 0), (100, 100)), None);
    }

    #[test]
    fn a_hidden_neck_has_no_zone() {
        let planes = scene(H, |_, _| 0.0, |_, _| false);
        let pm = parse();
        let rect = parse_rect(&pm, (0, 0), (W, H)).unwrap();
        let hidden = GrayF32::new(rect.w, rect.h);
        assert!(neck_zone(
            &pm,
            (0, 0),
            rect,
            &planes,
            &hidden,
            ED,
            &NeckParams::default()
        )
        .is_none());
        assert!(zone_of(&planes).is_some());
    }

    #[test]
    fn moderate_lines_are_flattened_and_nothing_else_changes() {
        let planes = scene(H, wrinkles(5.0), |_, _| false);
        let l = soften(&planes, 1.0);
        // 孤立的细暗线被摊平到一半左右（真实的颈纹之间有亮脊，两者一起被拉平）
        for y in WRINKLE_ROWS {
            let (before, after) = (line_depth(&planes.l, 100, y), line_depth(&l, 100, y));
            assert!(
                before > 4.5 && after < 0.6 * before,
                "row {y}: {before} → {after}"
            );
        }
        // 离脖子（x ≥ 18）一个羽化尾部以外不变
        assert!(max_change(&planes.l, &l, 0..4, 0..H) < 1e-6);
    }

    #[test]
    fn strength_scales_the_smoothing_linearly() {
        // 不填细线时，改动与强度成正比；强度 0 不动
        let planes = scene(H, wrinkles(5.0), |_, _| false);
        let at = |strength| {
            let p = NeckParams {
                strength,
                line_fill: 0.0,
                ..NeckParams::default()
            };
            soften_with(&planes, &p)
        };
        let (full, half) = (at(1.0), at(0.5));
        for i in 0..planes.l.data.len() {
            let (o, f, h) = (planes.l.data[i], full.data[i], half.data[i]);
            assert!((h - o - 0.5 * (f - o)).abs() < 1e-3, "pixel {i}");
        }
        assert_eq!(soften(&planes, 0.0).data, planes.l.data);
        assert!(zone_with(
            &planes,
            &NeckParams {
                line_fill: 0.0,
                ..NeckParams::default()
            }
        )
        .unwrap()
        .line
        .is_none());
    }

    #[test]
    fn thin_lines_are_filled_but_dots_are_kept() {
        // 宽 2、暗 8 的长线（碎发、颈纹的线芯）与一排 2×2、同样暗的点（胡茬、毛孔）
        let dark = |x: usize, y: usize| {
            let line = (60..62).contains(&y) && (20..180).contains(&x);
            let dot = (100..102).contains(&y) && x % 12 < 2 && (24..176).contains(&x);
            if line || dot {
                8.0
            } else {
                0.0
            }
        };
        let planes = scene(H, dark, |_, _| false);
        let depth = |l: &GrayF32, x: usize, y: usize| {
            0.5 * (l.get(x, y - 4) + l.get(x, y + 4)) - l.get(x, y)
        };
        let (plain, filled) = (
            soften_with(
                &planes,
                &NeckParams {
                    line_fill: 0.0,
                    ..NeckParams::default()
                },
            ),
            soften(&planes, 1.0),
        );
        for x in [60, 100, 140] {
            let (before, smoothed, after) = (
                depth(&planes.l, x, 60),
                depth(&plain, x, 60),
                depth(&filled, x, 60),
            );
            assert!(
                after < 0.2 * before && after < 0.5 * smoothed,
                "line at {x}: {before} → {smoothed} → {after}"
            );
            // 填到与两侧齐平为止，不会更亮
            assert!(after > -0.3, "line at {x} overshoots: {after}");
        }
        for x in [48, 96, 144] {
            let (smoothed, after) = (depth(&plain, x, 100), depth(&filled, x, 100));
            assert!(
                (after - smoothed).abs() < 0.05,
                "dot at {x}: {smoothed} → {after}"
            );
        }
    }

    #[test]
    fn fine_grain_passes_through() {
        // 同一组颈纹加 / 不加 ±1 L 的棋盘格细颗粒：两个结果之差仍是那层细颗粒
        let lines = wrinkles(5.0);
        let grain = |x: usize, y: usize| if (x + y).is_multiple_of(2) { 1.0 } else { -1.0 };
        let plain = soften(&scene(H, &lines, |_, _| false), 1.0);
        let grainy = soften(
            &scene(H, |x, y| lines(x, y) - grain(x, y), |_, _| false),
            1.0,
        );
        for y in 20..140 {
            for x in 20..180 {
                let kept = grainy.get(x, y) - plain.get(x, y);
                assert!((kept - grain(x, y)).abs() < 0.1, "({x},{y}): {kept}");
            }
        }
    }

    #[test]
    fn foreign_objects_and_their_surroundings_are_kept() {
        // 灰色细带（项链、纹身墨迹：暗 25 L、色度低）与皮肤色的深暗带（项链的影子、粗碎发：暗 20 L）
        let lines = wrinkles(5.0);
        let dark = |x: usize, y: usize| match y {
            108..=113 => 25.0,
            124..=128 => 20.0,
            _ => lines(x, y),
        };
        let planes = scene(H, dark, |_, y| (108..=113).contains(&y));
        let l = soften(&planes, 1.0);
        let guard = (FOREIGN_GUARD * ED).round() as usize;
        for band in [108..114, 124..129] {
            // 异物本身不动，保护宽度内只剩羽化边上的微小变化
            assert!(
                max_change(&planes.l, &l, 0..W, band.clone()) < 0.01,
                "{band:?}"
            );
            let around = band.start - guard..band.end + guard;
            assert!(
                max_change(&planes.l, &l, 0..W, around.clone()) < 0.3,
                "{around:?}"
            );
        }
        // 颈纹照样淡化
        assert!(line_depth(&l, 100, 70) < 0.6 * line_depth(&planes.l, 100, 70));
    }

    #[test]
    fn gaps_between_foreign_strokes_are_kept() {
        // 两条浓笔画（暗 25 L，色度低）之间一条淡而皮肤色的尾迹（暗 8 L，单看与颈纹无异）
        let dark = |x: usize, y: usize| match (x, y) {
            (80..=85, 60..=100) | (96..=101, 60..=100) => 25.0,
            (90 | 91, 60..=100) => 8.0,
            _ => 0.0,
        };
        let grey = |x: usize, _| (80..=85).contains(&x) || (96..=101).contains(&x);
        let planes = scene(H, dark, |x, y| (60..=100).contains(&y) && grey(x, y));
        let l = soften(&planes, 1.0);
        assert!(max_change(&planes.l, &l, 86..96, 60..101) < 1e-3);
    }

    #[test]
    fn sharp_edges_stay_sharp() {
        // 暗 40 L 的锐利台阶（硬阴影）：导向滤波在方差远大于 eps 处保边——跨边缘的跳变几乎不变，
        // 只有两侧半径内的"肩"被磨圆一点；离边缘一个半径以外不动
        let planes = scene(H, |x, _| if x >= 100 { 40.0 } else { 0.0 }, |_, _| false);
        let l = soften(&planes, 1.0);
        for y in 30..130 {
            let jump = l.get(99, y) - l.get(100, y);
            assert!(jump > 0.9 * 40.0, "row {y}: {jump}");
        }
        let far = |x: usize| !(88..112).contains(&x);
        for y in 30..130 {
            for x in (20..180).filter(|&x| far(x)) {
                assert!((l.get(x, y) - planes.l.get(x, y)).abs() < 0.05, "({x},{y})");
            }
        }
    }

    #[test]
    fn the_weight_fades_out_where_the_parse_crop_ends() {
        // 解析图只覆盖到 y = 99：脖子在覆盖范围的下边缘被截断，权重在边缘归零、向内渐入
        let planes = scene(H, |_, _| 0.0, |_, _| false);
        let mut pm = parse();
        pm.bbox = (0, 0, 199, 99);
        let rect = parse_rect(&pm, (0, 0), (W, H)).unwrap();
        let allowed = GrayF32::from_vec(rect.w, rect.h, vec![1.0; rect.w * rect.h]);
        let p = NeckParams::default();
        let zone = neck_zone(&pm, (0, 0), rect, &planes, &allowed, ED, &p).unwrap();
        let at = |x: usize, y: usize| zone.weight.get(x - zone.rect.x0, y - zone.rect.y0);
        assert_eq!(zone.rect.y0 + zone.rect.h, 100);
        assert!(at(100, 99) < 1e-6);
        assert!(at(100, 70) > 0.99);
        assert!(at(100, 95) < at(100, 92) && at(100, 92) < at(100, 88));
    }

    #[test]
    fn no_fade_where_only_the_image_ends() {
        // 区域（图像）只到 y = 100，解析图还往下覆盖：脖子是被画面截断的，处理到图像边缘为止
        let planes = scene(100, |_, _| 0.0, |_, _| false);
        let zone = zone_of(&planes).unwrap();
        let at = |x: usize, y: usize| zone.weight.get(x - zone.rect.x0, y - zone.rect.y0);
        assert_eq!(zone.rect.y0 + zone.rect.h, 100);
        assert!(at(100, 99) > 0.99, "{}", at(100, 99));
    }

    #[test]
    fn shading_steps_at_the_zone_edge_are_not_lines() {
        // 暗 10 L 的竖直台阶一直延伸到区域外：区域边缘处线段出界，不补边距时会把台阶量成细线
        let planes = scene(H, |x, _| if x >= 100 { 10.0 } else { 0.0 }, |_, _| false);
        let zone = zone_of(&planes).unwrap();
        let line = zone.line.as_ref().unwrap();
        let r = zone.rect;
        assert!(r.y0 + r.h < H, "the region continues below the zone");
        for y in [r.y0, r.y0 + r.h - 1] {
            for x in 96..104 {
                let v = line.get(x - r.x0, y - r.y0);
                assert!(v < 1e-3, "({x},{y}): {v}");
            }
        }
    }
}
