//! 磨皮方案 C：pixpark/gpupixel `BeautyFaceFilter` 的均值 / 方差自适应磨皮。
//!
//! 原 shader（beauty_face_unit_filter.cc）：
//! ```text
//! mean = BoxBlur(src, radius 4, spacing 4)
//! var  = min(((src − mean) · 7.07)², 1)
//! p    = clamp((min(r, mean.r − 0.1) − 0.2) · 4, 0, 1)        // 肤色概率（红通道）
//! kMin = clamp((1 − meanVar / (meanVar + 0.1)) · p · blurAlpha, 0, 1)
//! res  = mix(src, mean, kMin)
//! out  = res + sharpen · (src − blur3x3(src)) · 2
//! ```
//! mean / var 在短边 720 的工作副本上计算后上采样；锐化项在全分辨率进行。

use crate::buffer::{GrayF32, ImgF32};
use crate::skin::gaussian::{blur3x3_rgb, box_blur_rgb};
use rayon::prelude::*;

pub const DEFAULT_RADIUS: usize = 4;
pub const DEFAULT_SPACING: f32 = 4.0;
pub const DEFAULT_DELTA: f32 = 7.07;
pub const THETA: f32 = 0.1;

#[derive(Clone, Debug)]
pub struct GpuPixelPrecomp {
    pub mean: ImgF32,
    pub var: ImgF32,
}

pub fn precompute_gpupixel(orig: &ImgF32, work_short: f32) -> GpuPixelPrecomp {
    let (work, scale) = orig.work_copy(work_short);
    let mean_w = box_blur_rgb(&work, DEFAULT_RADIUS, DEFAULT_SPACING);
    let var_w = work.zip_map(&mean_w, |s, m| {
        let mut v = [0.0f32; 3];
        for c in 0..3 {
            let d = (s[c] - m[c]) * DEFAULT_DELTA;
            v[c] = (d * d).min(1.0);
        }
        v
    });
    if scale < 1.0 {
        GpuPixelPrecomp {
            mean: mean_w.resize(orig.w, orig.h),
            var: var_w.resize(orig.w, orig.h),
        }
    } else {
        GpuPixelPrecomp {
            mean: mean_w,
            var: var_w,
        }
    }
}

pub fn smooth_gpupixel(
    orig: &ImgF32,
    pre: &GpuPixelPrecomp,
    face_mask: Option<&GrayF32>,
    blur_alpha: f32,
    sharpen: f32,
) -> ImgF32 {
    let hp = if sharpen > 0.0 {
        Some(blur3x3_rgb(orig, 1.0))
    } else {
        None
    };
    let mut out = ImgF32::new(orig.w, orig.h);
    out.data.par_iter_mut().enumerate().for_each(|(i, px)| {
        let s = orig.data[i];
        let m = pre.mean.data[i];
        let v = pre.var.data[i];
        let p = ((s[0].min(m[0] - 0.1) - 0.2) * 4.0).clamp(0.0, 1.0);
        let mean_var = (v[0] + v[1] + v[2]) / 3.0;
        let mut k = (1.0 - mean_var / (mean_var + THETA)) * p * blur_alpha;
        k = k.clamp(0.0, 1.0);
        if let Some(fm) = face_mask {
            k *= fm.data[i];
        }
        let mut t = [0.0f32; 3];
        for c in 0..3 {
            t[c] = s[c] + (m[c] - s[c]) * k;
        }
        if let Some(b) = &hp {
            let bl = b.data[i];
            for c in 0..3 {
                t[c] = (t[c] + sharpen * (s[c] - bl[c]) * 2.0).clamp(0.0, 1.0);
            }
        }
        *px = t;
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_zero_no_sharpen_is_identity() {
        let mut img = ImgF32::new(40, 30);
        for (i, p) in img.data.iter_mut().enumerate() {
            *p = [0.7 + 0.1 * ((i % 5) as f32 / 5.0), 0.5, 0.4];
        }
        let pre = precompute_gpupixel(&img, 720.0);
        let out = smooth_gpupixel(&img, &pre, None, 0.0, 0.0);
        assert!(out.max_abs_diff(&img) < 1e-6);
    }
}
