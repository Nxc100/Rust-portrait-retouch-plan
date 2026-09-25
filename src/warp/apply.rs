//! 并行应用形变（反向映射，仅处理包围盒内像素），保纹理重采样。
//!
//! 任何插值核在分数像素位置都会把相邻像素的颗粒平均掉：双线性在半像素处相当于 [0.5, 0.5]，
//! 噪声状的毛孔 / 皮肤颗粒幅度降到约 0.5；Catmull-Rom 也有约 15% 的平均损失（X04、海边样张实测）。
//! 因此把图像拆成两层：
//! - 基底 `base = G(src; σ = 1 px)`：平滑，用 Catmull-Rom 双三次采样，几乎无损；
//! - 细节 `detail = src − base`：像素级颗粒。在颗粒弱的平滑区（皮肤）用最近邻采样——只是"搬动"像素，
//!   幅度不变；在强边缘处（睫毛、眉、唇线）用双三次，避免最近邻的锯齿。两者按细节的局部能量平滑过渡。
//!
//! 输出 = 双三次(base)(q) + 混合细节(q)，q 为反向映射得到的源位置。

use crate::buffer::{GrayF32, ImgF32};
use crate::geom::P;
use crate::skin::guided::fast_gaussian;
use crate::warp::face_warp::FaceWarp;
use rayon::prelude::*;

/// 整数抽头的可分离高斯模糊（边缘钳制），用于小 σ（基底 / 细节分离）。
fn blur_rgb(src: &ImgF32, sigma: f32) -> ImgF32 {
    let r = (3.0 * sigma).ceil().max(1.0) as isize;
    let mut k: Vec<f32> = (-r..=r)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    let (w, h) = (src.w, src.h);
    let mut tmp = ImgF32::new(w, h);
    tmp.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut acc = [0.0f32; 3];
            for (j, kv) in k.iter().enumerate() {
                let p = src.get_clamped(x as isize + j as isize - r, y as isize);
                acc[0] += p[0] * kv;
                acc[1] += p[1] * kv;
                acc[2] += p[2] * kv;
            }
            *o = acc;
        }
    });
    let mut out = ImgF32::new(w, h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut acc = [0.0f32; 3];
            for (j, kv) in k.iter().enumerate() {
                let p = tmp.get_clamped(x as isize, y as isize + j as isize - r);
                acc[0] += p[0] * kv;
                acc[1] += p[1] * kv;
                acc[2] += p[2] * kv;
            }
            *o = acc;
        }
    });
    out
}

/// 细节局部 RMS 低于 `DETAIL_LO` 的像素用最近邻细节，高于 `DETAIL_HI` 用双三次，中间平滑过渡（0..1 RGB 单位）。
const DETAIL_LO: f32 = 0.012;
const DETAIL_HI: f32 = 0.035;

pub fn apply_warps(src: &ImgF32, warps: &[FaceWarp]) -> ImgF32 {
    let warps: Vec<&FaceWarp> = warps.iter().filter(|w| !w.is_identity()).collect();
    if warps.is_empty() {
        return src.clone();
    }
    let mut out = src.clone();
    let (mut x0, mut y0, mut x1, mut y1) = (src.w as f32, src.h as f32, 0.0f32, 0.0f32);
    for w in &warps {
        x0 = x0.min(w.bbox.0.x);
        y0 = y0.min(w.bbox.0.y);
        x1 = x1.max(w.bbox.1.x);
        y1 = y1.max(w.bbox.1.y);
    }
    let xs = x0.floor().max(0.0) as usize;
    let ys = y0.floor().max(0.0) as usize;
    let xe = ((x1.ceil().max(0.0) as usize) + 1).min(src.w);
    let ye = ((y1.ceil().max(0.0) as usize) + 1).min(src.h);
    if xs >= xe || ys >= ye {
        return out;
    }
    // 源区域：包围盒外扩一圈（采样位置可能落在盒外几个像素）
    let pad = 8usize;
    let (sx0, sy0) = (xs.saturating_sub(pad), ys.saturating_sub(pad));
    let (sx1, sy1) = ((xe + pad).min(src.w), (ye + pad).min(src.h));
    let (cw, ch) = (sx1 - sx0, sy1 - sy0);
    let mut crop = ImgF32::new(cw, ch);
    crop.data
        .par_chunks_mut(cw)
        .enumerate()
        .for_each(|(y, row)| {
            row.copy_from_slice(&src.data[(sy0 + y) * src.w + sx0..(sy0 + y) * src.w + sx1]);
        });
    let base = blur_rgb(&crop, 1.0);
    let mut detail = crop.clone();
    detail
        .data
        .par_iter_mut()
        .zip(&base.data)
        .for_each(|(d, b)| {
            d[0] -= b[0];
            d[1] -= b[1];
            d[2] -= b[2];
        });
    // 细节局部能量（三通道平均），决定最近邻 / 双三次的混合
    let energy: Vec<f32> = detail
        .data
        .par_iter()
        .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / 3.0)
        .collect();
    let energy = fast_gaussian(&GrayF32::from_vec(cw, ch, energy), 2.0);
    let off = P::new(sx0 as f32, sy0 as f32);
    let width = src.w;
    out.data[ys * width..ye * width]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(dy, row)| {
            let y = ys + dy;
            for (x, out_px) in row.iter_mut().enumerate().take(xe).skip(xs) {
                let p = P::new(x as f32 + 0.5, y as f32 + 0.5);
                let mut q = p;
                for w in &warps {
                    let (lo, hi) = w.bbox;
                    if q.x < lo.x || q.y < lo.y || q.x > hi.x || q.y > hi.y {
                        continue;
                    }
                    q = w.map(q);
                }
                if q == p {
                    continue;
                }
                let ql = q.sub(off);
                let (nx, ny) = ((ql.x - 0.5).round() as isize, (ql.y - 0.5).round() as isize);
                let e = energy.get_clamped(nx, ny).max(0.0).sqrt();
                let t = ((e - DETAIL_LO) / (DETAIL_HI - DETAIL_LO)).clamp(0.0, 1.0);
                let beta = 1.0 - t * t * (3.0 - 2.0 * t);
                let v = if beta <= 1e-3 {
                    // 强边缘：整体双三次（base + detail = 原图，线性）
                    crop.sample_bicubic_unclamped(ql.x, ql.y)
                } else {
                    let b = base.sample_bicubic_unclamped(ql.x, ql.y);
                    let dn = detail.get_clamped(nx, ny);
                    if beta >= 0.999 {
                        [b[0] + dn[0], b[1] + dn[1], b[2] + dn[2]]
                    } else {
                        let dc = detail.sample_bicubic_unclamped(ql.x, ql.y);
                        [
                            b[0] + dc[0] + beta * (dn[0] - dc[0]),
                            b[1] + dc[1] + beta * (dn[1] - dc[1]),
                            b[2] + dc[2] + beta * (dn[2] - dc[2]),
                        ]
                    }
                };
                *out_px = [
                    v[0].clamp(0.0, 1.0),
                    v[1].clamp(0.0, 1.0),
                    v[2].clamp(0.0, 1.0),
                ];
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::warp::face_warp::{FaceWarp, WarpParams};

    #[test]
    fn identity_warp_is_bitwise_identity() {
        let mut img = ImgF32::new(64, 64);
        for (i, p) in img.data.iter_mut().enumerate() {
            *p = [(i % 64) as f32 / 63.0, (i / 64) as f32 / 63.0, 0.5];
        }
        let f = crate::warp::face_warp::tests::synthetic_face();
        let w = FaceWarp::from_face(&f, &WarpParams::default());
        let out = apply_warps(&img, &[w]);
        assert_eq!(out.data, img.data);
    }

    #[test]
    fn warp_keeps_fine_grain_amplitude() {
        // 平坦灰 + 细颗粒噪声；形变后颗粒的 RMS 应基本保持（双线性会降到 ~0.6）
        // synthetic_face：瞳距 100，脸中心 (300, 300)，下巴约在 y = 510
        let (w, h) = (600, 700);
        let mut img = ImgF32::filled(w, h, [0.6, 0.5, 0.45]);
        let mut s = 12345u32;
        for p in img.data.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = ((s >> 8) as f32 / 16777216.0 - 0.5) * 0.02;
            p[0] += n;
            p[1] += n;
            p[2] += n;
        }
        let f = crate::warp::face_warp::tests::synthetic_face();
        let wp = WarpParams {
            thin_face: 1.0,
            chin_lift: 1.0,
            ..Default::default()
        };
        let warp = FaceWarp::from_face(&f, &wp);
        let out = apply_warps(&img, std::slice::from_ref(&warp));
        let rms = |im: &ImgF32| {
            let base = blur_rgb(im, 1.0);
            let mut acc = 0.0f64;
            let mut n = 0usize;
            for y in 380..520 {
                for x in 180..420 {
                    let i = y * w + x;
                    let d = (im.data[i][0] - base.data[i][0]) as f64;
                    acc += d * d;
                    n += 1;
                }
            }
            (acc / n as f64).sqrt()
        };
        let (r0, r1) = (rms(&img), rms(&out));
        assert!(r1 > 0.9 * r0, "grain lost: {r0} -> {r1}");
        // 确实发生了形变（测量区域内大部分像素改变）
        let changed = (380..520)
            .flat_map(|y| (180..420).map(move |x| y * w + x))
            .filter(|&i| (out.data[i][0] - img.data[i][0]).abs() > 1e-4)
            .count();
        assert!(
            changed > 140 * 240 / 2,
            "warp did not move the region: {changed}"
        );
    }
}
