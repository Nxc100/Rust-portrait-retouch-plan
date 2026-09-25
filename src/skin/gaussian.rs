//! 可分离卷积：高斯模糊、gpupixel 风格带间距的盒式模糊、3×3 小核。

use crate::buffer::{GrayF32, ImgF32};
use rayon::prelude::*;

/// 归一化一维高斯核，半径 = ceil(3σ)。返回 (offsets, weights)。
pub fn gaussian_taps(sigma: f32) -> (Vec<f32>, Vec<f32>) {
    if sigma <= 0.0 {
        return (vec![0.0], vec![1.0]);
    }
    let r = (3.0 * sigma).ceil() as i32;
    let mut offs = Vec::with_capacity((2 * r + 1) as usize);
    let mut ws = Vec::with_capacity((2 * r + 1) as usize);
    let mut sum = 0.0;
    for i in -r..=r {
        let w = (-(i * i) as f32 / (2.0 * sigma * sigma)).exp();
        offs.push(i as f32);
        ws.push(w);
        sum += w;
    }
    for w in &mut ws {
        *w /= sum;
    }
    (offs, ws)
}

/// gpupixel `BoxBlurFilter`（radius r、texelSpacing s）：2r+1 个 tap，偏移 k·s，权重 1/(2r+1)。
pub fn box_taps(radius: usize, spacing: f32) -> (Vec<f32>, Vec<f32>) {
    let n = 2 * radius + 1;
    let w = 1.0 / n as f32;
    let offs: Vec<f32> = (0..n)
        .map(|i| (i as f32 - radius as f32) * spacing)
        .collect();
    (offs, vec![w; n])
}

fn all_integral(offs: &[f32]) -> bool {
    offs.iter().all(|o| (o - o.round()).abs() < 1e-6)
}

/// 可分离卷积（先水平后垂直），边缘钳制；非整数偏移用双线性采样。
pub fn conv_sep_rgb(src: &ImgF32, offs: &[f32], ws: &[f32]) -> ImgF32 {
    let h = conv_pass_rgb(src, offs, ws, true);
    conv_pass_rgb(&h, offs, ws, false)
}

pub fn conv_pass_rgb(src: &ImgF32, offs: &[f32], ws: &[f32], horizontal: bool) -> ImgF32 {
    let mut out = ImgF32::new(src.w, src.h);
    let integral = all_integral(offs);
    let ioffs: Vec<isize> = offs.iter().map(|o| o.round() as isize).collect();
    out.data
        .par_chunks_mut(src.w)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, out_px) in row.iter_mut().enumerate() {
                let mut acc = [0.0f32; 3];
                if integral {
                    for (o, w) in ioffs.iter().zip(ws) {
                        let s = if horizontal {
                            src.get_clamped(x as isize + o, y as isize)
                        } else {
                            src.get_clamped(x as isize, y as isize + o)
                        };
                        acc[0] += s[0] * w;
                        acc[1] += s[1] * w;
                        acc[2] += s[2] * w;
                    }
                } else {
                    for (o, w) in offs.iter().zip(ws) {
                        let s = if horizontal {
                            src.sample_bilinear(x as f32 + 0.5 + o, y as f32 + 0.5)
                        } else {
                            src.sample_bilinear(x as f32 + 0.5, y as f32 + 0.5 + o)
                        };
                        acc[0] += s[0] * w;
                        acc[1] += s[1] * w;
                        acc[2] += s[2] * w;
                    }
                }
                *out_px = acc;
            }
        });
    out
}

pub fn conv_sep_gray(src: &GrayF32, offs: &[f32], ws: &[f32]) -> GrayF32 {
    let h = conv_pass_gray(src, offs, ws, true);
    conv_pass_gray(&h, offs, ws, false)
}

pub fn conv_pass_gray(src: &GrayF32, offs: &[f32], ws: &[f32], horizontal: bool) -> GrayF32 {
    let mut out = GrayF32::new(src.w, src.h);
    let integral = all_integral(offs);
    let ioffs: Vec<isize> = offs.iter().map(|o| o.round() as isize).collect();
    out.data
        .par_chunks_mut(src.w)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, out_px) in row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                if integral {
                    for (o, w) in ioffs.iter().zip(ws) {
                        let s = if horizontal {
                            src.get_clamped(x as isize + o, y as isize)
                        } else {
                            src.get_clamped(x as isize, y as isize + o)
                        };
                        acc += s * w;
                    }
                } else {
                    for (o, w) in offs.iter().zip(ws) {
                        let s = if horizontal {
                            src.sample_bilinear(x as f32 + 0.5 + o, y as f32 + 0.5)
                        } else {
                            src.sample_bilinear(x as f32 + 0.5, y as f32 + 0.5 + o)
                        };
                        acc += s * w;
                    }
                }
                *out_px = acc;
            }
        });
    out
}

pub fn gaussian_blur_rgb(src: &ImgF32, sigma: f32) -> ImgF32 {
    let (o, w) = gaussian_taps(sigma);
    conv_sep_rgb(src, &o, &w)
}

pub fn gaussian_blur_gray(src: &GrayF32, sigma: f32) -> GrayF32 {
    let (o, w) = gaussian_taps(sigma);
    conv_sep_gray(src, &o, &w)
}

/// gpupixel BoxBlur（可分离，先水平后垂直）。
pub fn box_blur_rgb(src: &ImgF32, radius: usize, spacing: f32) -> ImgF32 {
    let (o, w) = box_taps(radius, spacing);
    conv_sep_rgb(src, &o, &w)
}

/// gpupixel 锐化用的 3×3 低通：中心 1/4，四邻 1/8，四角 1/16（在 1 像素偏移处采样）。
pub fn blur3x3_rgb(src: &ImgF32, offset: f32) -> ImgF32 {
    // 二项核 [1/4, 1/2, 1/4] 的可分离形式恰好等于该 3×3 核
    let offs = [-offset, 0.0, offset];
    let ws = [0.25, 0.5, 0.25];
    conv_sep_rgb(src, &offs, &ws)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_preserves_constant_and_sums_to_one() {
        let (_, w) = gaussian_taps(2.5);
        let s: f32 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        let img = ImgF32::filled(20, 10, [0.3, 0.6, 0.9]);
        let b = gaussian_blur_rgb(&img, 3.0);
        assert!(b.max_abs_diff(&img) < 1e-5);
        let g = GrayF32::filled(20, 10, 0.7);
        let bg = gaussian_blur_gray(&g, 2.0);
        assert!(bg.data.iter().all(|v| (v - 0.7).abs() < 1e-5));
    }

    #[test]
    fn box_taps_layout() {
        let (o, w) = box_taps(4, 4.0);
        assert_eq!(o.len(), 9);
        assert!((o[0] + 16.0).abs() < 1e-6 && (o[8] - 16.0).abs() < 1e-6);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
