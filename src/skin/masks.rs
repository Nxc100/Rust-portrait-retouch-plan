//! 皮肤遮罩装配：人脸解析（BiSeNet）+ 人像抠图（MODNet）+ 颜色规则 → 脸部 / 身体羽化遮罩。
//!
//! 无解析模型时退化为"轮廓多边形 ∧ 颜色规则"，无抠图模型时身体遮罩退化为下巴以下的颈部梯形 ∧ 颜色规则。
//! 身体皮肤用严格肤色规则（YCbCr + 饱和度）并做形态学清理，避免亮片、红花、礁石被判为皮肤；
//! 有语义皮肤分割（`face::skinseg`）时再与其概率图取交集——米色缎面婚纱、粉色团扇、金饰的颜色都落在
//! 肤色规则内，只有语义模型能把它们排除。两者各有盲区（颜色阈值在有色光下漏掉皮肤，语义模型偶尔留下孤岛），
//! 有抠图与语义模型时按颜色连续性修正：遮罩的边界只落在真正的颜色边缘上（`skin::continuity`）。
//! 身体遮罩在半分辨率上计算后上采样。

use crate::buffer::{GrayF32, ImgF32};
use crate::color::lab::rgb_to_lab;
use crate::face::parsing::{EXCLUDE_CLASSES, FACE_SKIN_CLASSES, NECK_CLASSES};
use crate::face::semantic::FaceKeyPoints;
use crate::geom::P;
use crate::skin::continuity::reconcile_body_mask;
use crate::skin::guided::{fast_gaussian, guided_filter};
use crate::skin::mask::{fill_polygon, is_skin_rgb, is_skin_strict};
use rayon::prelude::*;
use std::sync::Arc;

/// 装配结果（全分辨率）。
#[derive(Clone, Debug)]
pub struct SkinMasks {
    /// 每张脸的脸部皮肤遮罩（与 faces 顺序一致）
    pub faces: Vec<GrayF32>,
    /// 身体皮肤（脖颈、胸、手臂、手），已扣除脸部
    pub body: Option<GrayF32>,
    /// 所有遮罩的并集
    pub union: GrayF32,
    /// 并集的包围盒 (x0, y0, x1, y1)，右下为开区间；None 表示空
    pub roi: Option<(usize, usize, usize, usize)>,
    /// 语义皮肤概率（全分辨率，0..1；有皮肤分割模型且处理身体时才有）。身体遮罩为了连贯会把小块非皮肤
    /// （纹身、项链、衣褶）包进来，瑕疵 / 疤痕检测再用它排除这些物体
    pub skin_prob: Option<Arc<GrayF32>>,
}

/// 亮度平面（Lab 的 L，0..100）。
pub fn luminance_plane(img: &ImgF32) -> GrayF32 {
    let data: Vec<f32> = img.data.par_iter().map(|p| rgb_to_lab(*p)[0]).collect();
    GrayF32::from_vec(img.w, img.h, data)
}

pub fn crop_gray(src: &GrayF32, x0: usize, y0: usize, cw: usize, ch: usize) -> GrayF32 {
    let mut out = GrayF32::new(cw, ch);
    out.data
        .par_chunks_mut(cw)
        .enumerate()
        .for_each(|(y, row)| {
            row.copy_from_slice(&src.data[(y0 + y) * src.w + x0..(y0 + y) * src.w + x0 + cw]);
        });
    out
}

pub fn paste_gray(dst: &mut GrayF32, src: &GrayF32, x0: usize, y0: usize) {
    let (cw, ch) = (src.w, src.h);
    let dw = dst.w;
    dst.data
        .par_chunks_mut(dw)
        .enumerate()
        .skip(y0)
        .take(ch)
        .for_each(|(y, row)| {
            let sy = y - y0;
            row[x0..x0 + cw].copy_from_slice(&src.data[sy * cw..(sy + 1) * cw]);
        });
}

/// 只在包围盒内做高斯模糊（其余区域保持）。
fn blur_in_bbox(m: &GrayF32, bbox: (usize, usize, usize, usize), sigma: f32) -> GrayF32 {
    let (x0, y0, x1, y1) = bbox;
    let (cw, ch) = (x1 - x0, y1 - y0);
    if cw < 2 || ch < 2 {
        return m.clone();
    }
    let sub = fast_gaussian(&crop_gray(m, x0, y0, cw, ch), sigma);
    let mut out = m.clone();
    paste_gray(&mut out, &sub, x0, y0);
    out
}

/// 在子窗口内用亮度引导羽化二值遮罩（导向滤波），避免全图计算。
fn feather_guided(
    mask: &GrayF32,
    orig: &ImgF32,
    bbox: (usize, usize, usize, usize),
    radius: usize,
) -> GrayF32 {
    let (x0, y0, x1, y1) = bbox;
    let (cw, ch) = (x1 - x0, y1 - y0);
    if cw < 4 || ch < 4 {
        return mask.clone();
    }
    let sub_m = crop_gray(mask, x0, y0, cw, ch);
    let mut sub_l = GrayF32::new(cw, ch);
    sub_l
        .data
        .par_chunks_mut(cw)
        .enumerate()
        .for_each(|(y, lrow)| {
            let src_p = &orig.data[(y0 + y) * orig.w + x0..(y0 + y) * orig.w + x1];
            for (l, p) in lrow.iter_mut().zip(src_p) {
                *l = rgb_to_lab(*p)[0];
            }
        });
    let mut sub = guided_filter(&sub_m, &sub_l, radius.max(2), 30.0, 4);
    sub.map_inplace(|v| v.clamp(0.0, 1.0));
    let mut out = mask.clone();
    paste_gray(&mut out, &sub, x0, y0);
    out
}

pub fn bbox_of(m: &GrayF32, thr: f32, margin: usize) -> Option<(usize, usize, usize, usize)> {
    let (w, h) = (m.w, m.h);
    let rows: Vec<bool> = m
        .data
        .par_chunks(w)
        .map(|r| r.iter().any(|v| *v > thr))
        .collect();
    let y0 = rows.iter().position(|b| *b)?;
    let y1 = rows.iter().rposition(|b| *b)? + 1;
    let (x0, x1) = (y0..y1)
        .into_par_iter()
        .map(|y| {
            let row = &m.data[y * w..(y + 1) * w];
            let a = row.iter().position(|v| *v > thr).unwrap_or(w);
            let b = row
                .iter()
                .rposition(|v| *v > thr)
                .map(|b| b + 1)
                .unwrap_or(0);
            (a, b)
        })
        .reduce(|| (w, 0), |p, q| (p.0.min(q.0), p.1.max(q.1)));
    if x0 >= x1 {
        return None;
    }
    Some((
        x0.saturating_sub(margin),
        y0.saturating_sub(margin),
        (x1 + margin).min(w),
        (y1 + margin).min(h),
    ))
}

/// 颈部梯形（无抠图模型时的身体退化方案）：下巴以下、宽度 1.2 倍脸宽、高度 2.6 倍瞳距。
fn neck_polygon(f: &FaceKeyPoints) -> Vec<P> {
    let up = f.dir_up();
    let right = f.dir_right();
    let ed = f.eye_distance();
    let jl = f.jaw_l[2];
    let jr = f.jaw_r[2];
    let down = up.mul(-1.0);
    vec![
        jl.sub(right.mul(0.3 * ed)),
        jr.add(right.mul(0.3 * ed)),
        jr.add(right.mul(1.1 * ed)).add(down.mul(2.6 * ed)),
        jl.sub(right.mul(1.1 * ed)).add(down.mul(2.6 * ed)),
    ]
}

/// 形态学清理（开运算）：`sigma` 控制去除的斑点尺度。
fn morph_clean(m: &GrayF32, sigma: f32) -> GrayF32 {
    let mut e = fast_gaussian(m, sigma);
    e.map_inplace(|v| if v > 0.72 { 1.0 } else { 0.0 });
    let mut d = fast_gaussian(&e, sigma);
    d.map_inplace(|v| if v > 0.2 { 1.0 } else { 0.0 });
    d
}

/// 装配皮肤遮罩。`matte` 为可选的人像 alpha（全分辨率）。
pub fn build_skin_masks(
    orig: &ImgF32,
    faces: &[FaceKeyPoints],
    matte: Option<&GrayF32>,
    skin_prob: Option<&Arc<GrayF32>>,
) -> SkinMasks {
    let (w, h) = (orig.w, orig.h);
    let timing = std::env::var("RETOUCH_TIMING").is_ok();
    let t0 = std::time::Instant::now();
    let mut face_masks = Vec::with_capacity(faces.len());
    let mut exclude = GrayF32::new(w, h);
    let mut neck = GrayF32::new(w, h);
    let mut ed_sum = 0.0;
    for f in faces {
        let ed = f.eye_distance().max(8.0);
        ed_sum += ed;
        let mut m = match &f.parse {
            Some(pm) => {
                let (bx0, by0, bx1, by1) = pm.bbox;
                let bb = (
                    bx0.max(0) as usize,
                    by0.max(0) as usize,
                    (bx1.max(0) as usize).min(w),
                    (by1.max(0) as usize).min(h),
                );
                let mut ex = pm.mask_of(w, h, &EXCLUDE_CLASSES);
                ex = blur_in_bbox(&ex, bb, 0.01 * ed);
                ex.map_inplace(|v| if v > 0.15 { 1.0 } else { 0.0 });
                exclude.max_inplace(&ex);
                neck.max_inplace(&pm.mask_of(w, h, &NECK_CLASSES));
                let mut sk = pm.mask_of(w, h, &FACE_SKIN_CLASSES);
                sk.data
                    .par_iter_mut()
                    .zip(&ex.data)
                    .for_each(|(a, e)| *a *= 1.0 - e);
                sk
            }
            None => {
                let mut poly: Vec<P> = f.contour.clone();
                poly.extend(f.forehead.iter().copied());
                let mut sk = GrayF32::new(w, h);
                fill_polygon(&mut sk, &poly, 1.0);
                let rule: Vec<f32> = orig
                    .data
                    .par_iter()
                    .map(|p| if is_skin_rgb(*p) { 1.0 } else { 0.0 })
                    .collect();
                sk.mul_inplace(&fast_gaussian(&GrayF32::from_vec(w, h, rule), 2.0));
                for (c, r) in [(f.pupil_l, 0.30 * ed), (f.pupil_r, 0.30 * ed)] {
                    punch_circle(&mut sk, c, r);
                }
                punch_circle(&mut sk, f.mouth_l.mid(f.mouth_r), 0.42 * ed);
                sk
            }
        };
        let bbox = bbox_of(&m, 0.5, (0.1 * ed) as usize).unwrap_or((0, 0, w, h));
        m = feather_guided(&m, orig, bbox, (0.02 * ed).max(2.0) as usize);
        face_masks.push(m);
    }
    if timing {
        eprintln!(
            "  [masks] faces: {:.0} ms",
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    let ed_mean = if faces.is_empty() {
        (w.min(h) as f32) * 0.06
    } else {
        ed_sum / faces.len() as f32
    };
    let mut face_union = GrayF32::new(w, h);
    for m in &face_masks {
        face_union.max_inplace(m);
    }
    // 身体（半分辨率计算）
    let t1 = std::time::Instant::now();
    let body = if faces.is_empty() && matte.is_none() {
        None
    } else {
        let s = 0.5f32;
        let (hw, hh) = (
            ((w as f32 * s) as usize).max(1),
            ((h as f32 * s) as usize).max(1),
        );
        let small = orig.resize(hw, hh);
        let strict: Vec<f32> = small
            .data
            .par_iter()
            .map(|p| if is_skin_strict(*p) { 1.0 } else { 0.0 })
            .collect();
        let strict = GrayF32::from_vec(hw, hh, strict);
        let person = matte.map(|alpha| alpha.resize(hw, hh));
        let mut b = match &person {
            Some(a_s) => {
                let data: Vec<f32> = a_s
                    .data
                    .par_iter()
                    .zip(&strict.data)
                    .map(|(a, r)| if *a > 0.5 { *r } else { 0.0 })
                    .collect();
                GrayF32::from_vec(hw, hh, data)
            }
            None => {
                let mut b = GrayF32::new(hw, hh);
                for f in faces {
                    let poly: Vec<P> = neck_polygon(f).into_iter().map(|p| p.mul(s)).collect();
                    fill_polygon(&mut b, &poly, 1.0);
                }
                b.mul_inplace(&strict);
                b
            }
        };
        // 语义皮肤门控：概率图（分辨率较低）先轻微膨胀再软阈值，边界仍由颜色规则 / 抠图决定
        if let Some(sp) = skin_prob.map(|sp| sp.as_ref()) {
            let sp_h = sp.resize(hw, hh);
            let g = fast_gaussian(&sp_h, (0.008 * hw.min(hh) as f32).max(1.0));
            b.data
                .par_iter_mut()
                .zip(&g.data)
                .for_each(|(v, p)| *v *= ((*p - 0.15) / 0.25).clamp(0.0, 1.0));
            // 颜色连续性：找回有色光下被颜色规则漏掉的皮肤，去掉语义模型在同色区域里留下的孤岛
            if let Some(person) = &person {
                reconcile_body_mask(&mut b, &small, person, &g, &strict, ed_mean * s);
            }
        }
        b = morph_clean(&b, (0.02 * ed_mean * s).max(1.5));
        b.max_inplace(&neck.resize(hw, hh));
        let fu_s = face_union.resize(hw, hh);
        let ex_s = exclude.resize(hw, hh);
        b.data
            .par_iter_mut()
            .zip(&fu_s.data)
            .zip(&ex_s.data)
            .for_each(|((v, fu), ex)| {
                *v *= (1.0 - fu) * (1.0 - ex);
            });
        let mut b = fast_gaussian(&b, (0.015 * ed_mean * s).max(1.0));
        b.map_inplace(|v| v.clamp(0.0, 1.0));
        let mut full = b.resize(w, h);
        // 上采样后再次扣除全分辨率脸部，保证脸 / 身体不重叠
        full.data
            .par_iter_mut()
            .zip(&face_union.data)
            .for_each(|(v, fu)| *v *= 1.0 - fu);
        Some(full)
    };
    if timing {
        eprintln!("  [masks] body: {:.0} ms", t1.elapsed().as_secs_f64() * 1e3);
    }
    let mut union = face_union.clone();
    if let Some(b) = &body {
        union.max_inplace(b);
    }
    let roi = bbox_of(&union, 0.01, (0.15 * ed_mean) as usize);
    SkinMasks {
        faces: face_masks,
        body,
        union,
        roi,
        skin_prob: skin_prob.cloned(),
    }
}

fn punch_circle(m: &mut GrayF32, c: P, r: f32) {
    let (w, h) = (m.w as i64, m.h as i64);
    let (x0, y0) = (
        ((c.x - r).floor() as i64).max(0),
        ((c.y - r).floor() as i64).max(0),
    );
    let (x1, y1) = (
        ((c.x + r).ceil() as i64).min(w - 1),
        ((c.y + r).ceil() as i64).min(h - 1),
    );
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = P::new(x as f32 + 0.5, y as f32 + 0.5).dist(c);
            if d < r {
                m.data[(y * w + x) as usize] = 0.0;
            }
        }
    }
}
