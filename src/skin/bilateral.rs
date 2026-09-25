//! `GPUImageBilateralFilter` 等价实现：可分离两遍（先水平后垂直），每遍 9 tap。
//!
//! 每个 tap 的实际权重 = `W9[i] * (1 - min(‖sample − center‖ * dnf, 1))`，
//! 输出 = Σ(sample·w) / Σw，中心 tap 权重固定 0.18（`‖·‖` 为 RGB 欧氏距离）。

use crate::buffer::ImgF32;
use rayon::prelude::*;

pub const W9: [f32; 9] = [0.05, 0.09, 0.12, 0.15, 0.18, 0.15, 0.12, 0.09, 0.05];

/// 默认参数（BBGPUImageBeautifyFilter：dnf = 4.0；GPUImage 默认 texelSpacingMultiplier = 4.0）。
pub const DEFAULT_SPACING: f32 = 4.0;
pub const DEFAULT_DNF: f32 = 4.0;

#[inline]
fn accumulate(c: [f32; 3], s: [f32; 3], wg: f32, dnf: f32, sum: &mut [f32; 3], wsum: &mut f32) {
    let d = ((s[0] - c[0]).powi(2) + (s[1] - c[1]).powi(2) + (s[2] - c[2]).powi(2)).sqrt();
    let w = wg * (1.0 - (d * dnf).min(1.0));
    *wsum += w;
    sum[0] += s[0] * w;
    sum[1] += s[1] * w;
    sum[2] += s[2] * w;
}

fn bilateral_pass(src: &ImgF32, spacing: f32, dnf: f32, horizontal: bool) -> ImgF32 {
    let mut out = ImgF32::new(src.w, src.h);
    let integral = (spacing - spacing.round()).abs() < 1e-6;
    let ispacing = spacing.round() as isize;
    out.data
        .par_chunks_mut(src.w)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, out_px) in row.iter_mut().enumerate() {
                let c = src.get(x, y);
                let mut sum = [c[0] * 0.18, c[1] * 0.18, c[2] * 0.18];
                let mut wsum = 0.18f32;
                for (i, &wg) in W9.iter().enumerate() {
                    if i == 4 {
                        continue;
                    }
                    let s = if integral {
                        let off = (i as isize - 4) * ispacing;
                        if horizontal {
                            src.get_clamped(x as isize + off, y as isize)
                        } else {
                            src.get_clamped(x as isize, y as isize + off)
                        }
                    } else {
                        let off = (i as f32 - 4.0) * spacing;
                        if horizontal {
                            src.sample_bilinear(x as f32 + 0.5 + off, y as f32 + 0.5)
                        } else {
                            src.sample_bilinear(x as f32 + 0.5, y as f32 + 0.5 + off)
                        }
                    };
                    accumulate(c, s, wg, dnf, &mut sum, &mut wsum);
                }
                *out_px = [sum[0] / wsum, sum[1] / wsum, sum[2] / wsum];
            }
        });
    out
}

/// GPUImageBilateralFilter 等价实现：先水平后垂直。
pub fn bilateral_gpuimage(src: &ImgF32, spacing: f32, dnf: f32) -> ImgF32 {
    let h = bilateral_pass(src, spacing, dnf, true);
    bilateral_pass(&h, spacing, dnf, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_image_unchanged() {
        let img = ImgF32::filled(64, 48, [0.6, 0.4, 0.3]);
        let b = bilateral_gpuimage(&img, 4.0, 4.0);
        assert!(b.max_abs_diff(&img) < 1e-6);
    }

    #[test]
    fn strong_edge_preserved_by_range_weight() {
        // 左半黑右半白：dnf=4 时相差 1.0 的样本权重为 0，边缘不被抹平
        let mut img = ImgF32::new(64, 8);
        for y in 0..8 {
            for x in 0..64 {
                img.data[y * 64 + x] = if x < 32 { [0.0; 3] } else { [1.0; 3] };
            }
        }
        let b = bilateral_gpuimage(&img, 4.0, 4.0);
        assert!(b.max_abs_diff(&img) < 1e-6);
    }
}
