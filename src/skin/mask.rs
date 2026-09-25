//! 皮肤遮罩：颜色肤色规则（原 shader）+ 人脸多边形羽化遮罩。

use crate::buffer::{GrayF32, ImgF32};
use crate::face::semantic::FaceKeyPoints;
use crate::geom::P;
use crate::skin::gaussian::gaussian_blur_gray;
use rayon::prelude::*;

/// BBGPUImageBeautifyFilter 组合 shader 中的肤色规则（不含边缘条件）。
#[inline]
pub fn is_skin_rgb(p: [f32; 3]) -> bool {
    let (r, g, b) = (p[0], p[1], p[2]);
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    r > 0.3725 && g > 0.1568 && b > 0.0784 && r > b && (mx - mn) > 0.0588 && (r - g).abs() > 0.0588
}

/// 更严格的肤色规则（身体皮肤用）：YCbCr 椭圆区 + 饱和度 / 亮度限制，排除红花、亮片、深色衣物与礁石。
#[inline]
pub fn is_skin_strict(p: [f32; 3]) -> bool {
    let (r, g, b) = (p[0] * 255.0, p[1] * 255.0, p[2] * 255.0);
    let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
    let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let s = (mx - mn) / mx.max(1.0);
    (135.0..=173.0).contains(&cr)
        && (77.0..=127.0).contains(&cb)
        && r > g
        && g > b
        && (0.10..=0.62).contains(&s)
        && mx > 0.30 * 255.0
}

/// 扫描线奇偶规则多边形填充（顶点为像素中心制坐标）。
pub fn fill_polygon(mask: &mut GrayF32, poly: &[P], value: f32) {
    if poly.len() < 3 {
        return;
    }
    let (w, h) = (mask.w, mask.h);
    let n = poly.len();
    let ymin = poly
        .iter()
        .map(|p| p.y)
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as usize;
    let ymax = (poly.iter().map(|p| p.y).fold(f32::MIN, f32::max).ceil() as usize)
        .min(h.saturating_sub(1));
    if ymin > ymax {
        return;
    }
    let rows = &mut mask.data[ymin * w..(ymax + 1) * w];
    rows.par_chunks_mut(w).enumerate().for_each(|(dy, row)| {
        let yc = (ymin + dy) as f32 + 0.5;
        let mut xs: Vec<f32> = Vec::with_capacity(8);
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            if (a.y <= yc && b.y > yc) || (b.y <= yc && a.y > yc) {
                let t = (yc - a.y) / (b.y - a.y);
                xs.push(a.x + (b.x - a.x) * t);
            }
        }
        xs.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
        for pair in xs.chunks(2) {
            if pair.len() < 2 {
                break;
            }
            let x0 = (pair[0] - 0.5).ceil().max(0.0) as usize;
            let x1 = (pair[1] - 0.5).floor().min(w as f32 - 1.0);
            if x1 < 0.0 {
                continue;
            }
            let x1 = x1 as usize;
            for v in row.iter_mut().take(x1 + 1).skip(x0) {
                *v = value;
            }
        }
    });
}

/// 人脸区域多边形（轮廓 + 额头补点），按 `scale` 缩放到工作分辨率。
pub fn face_polygon(f: &FaceKeyPoints, scale: f32) -> Vec<P> {
    let mut poly: Vec<P> = f.contour.iter().map(|p| p.mul(scale)).collect();
    poly.extend(f.forehead.iter().map(|p| p.mul(scale)));
    poly
}

/// 在给定分辨率上生成所有人脸的羽化遮罩（并集）。`scale` = 该分辨率 / 原图分辨率。
/// 羽化 σ = 0.08 × 瞳距（工作分辨率下）。
pub fn face_polygon_mask(w: usize, h: usize, faces: &[FaceKeyPoints], scale: f32) -> GrayF32 {
    let mut acc = GrayF32::new(w, h);
    for f in faces {
        let poly = face_polygon(f, scale);
        let mut m = GrayF32::new(w, h);
        fill_polygon(&mut m, &poly, 1.0);
        let sigma = (0.08 * f.eye_distance() * scale).max(0.5);
        let m = gaussian_blur_gray(&m, sigma);
        acc.max_inplace(&m);
    }
    acc
}

/// 全分辨率人脸遮罩：在短边 `work_short` 的工作分辨率上生成后上采样。
pub fn face_mask_fullres(w: usize, h: usize, faces: &[FaceKeyPoints], work_short: f32) -> GrayF32 {
    let scale = crate::buffer::work_scale(w, h, work_short);
    if scale >= 1.0 {
        return face_polygon_mask(w, h, faces, 1.0);
    }
    let ww = ((w as f32 * scale).round() as usize).max(1);
    let wh = ((h as f32 * scale).round() as usize).max(1);
    face_polygon_mask(ww, wh, faces, scale).resize(w, h)
}

/// 颜色肤色遮罩（在工作副本上计算，二值规则 + 轻微羽化，σ = 1.5 px）。
pub fn skin_color_mask(work: &ImgF32) -> GrayF32 {
    let data: Vec<f32> = work
        .data
        .par_iter()
        .map(|p| if is_skin_rgb(*p) { 1.0 } else { 0.0 })
        .collect();
    let m = GrayF32::from_vec(work.w, work.h, data);
    gaussian_blur_gray(&m, 1.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_fill_square() {
        let mut m = GrayF32::new(10, 10);
        let poly = [
            P::new(2.0, 2.0),
            P::new(8.0, 2.0),
            P::new(8.0, 8.0),
            P::new(2.0, 8.0),
        ];
        fill_polygon(&mut m, &poly, 1.0);
        // 像素中心 (2.5..7.5) 在内部
        assert_eq!(m.get(2, 2), 1.0);
        assert_eq!(m.get(7, 7), 1.0);
        assert_eq!(m.get(1, 5), 0.0);
        assert_eq!(m.get(8, 5), 0.0);
        let sum: f32 = m.data.iter().sum();
        assert_eq!(sum, 36.0);
    }

    #[test]
    fn skin_rule_matches_shader() {
        assert!(is_skin_rgb([0.8, 0.6, 0.5]));
        assert!(!is_skin_rgb([0.5, 0.5, 0.5]));
        assert!(!is_skin_rgb([0.3, 0.6, 0.8]));
    }
}
