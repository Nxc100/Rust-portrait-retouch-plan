//! 磨皮方案 B：高反差保留（YUCIHighPassSkinSmoothing 流程）。
//!
//! 原流程（Core Image）：
//! 1. mask 生成：`CIExposureAdjust(-1 EV)` → GreenBlue overlay（退化为 `2·G·B`）
//!    → HighPass（`x − gaussian(x, radius) + 0.5`）→ HardLight 自混合 ×3 → 色阶 `(x − 75/255) · 255/89`
//! 2. toned = RGB 复合曲线（控制点 (0,0) (120/255,146/255) (1,1)）按 amount 混合
//! 3. `CIBlendWithMask(inputImage=src, background=toned, mask)`：mask 白处保留原图，黑处取 toned
//! 4. `CISharpenLuminance(sharpness = 0.6 × amount)`
//!
//! 这里 mask 在短边 1000 的工作副本上计算后上采样，曲线与混合在全分辨率逐像素进行。

use crate::buffer::{GrayF32, ImgF32};
use crate::color::blend::hardlight;
use crate::color::curve::Curve256;
use crate::skin::gaussian::{gaussian_blur_gray, gaussian_blur_rgb};
use rayon::prelude::*;

/// YUCI 默认参数。
pub const DEFAULT_RADIUS: f32 = 8.0;
pub const DEFAULT_AMOUNT: f32 = 0.75;
pub const DEFAULT_SHARPNESS: f32 = 0.6;

/// YUCI 默认肤色曲线控制点。
pub fn default_skin_curve() -> Curve256 {
    Curve256::from_control_points(&[(0.0, 0.0), (120.0 / 255.0, 146.0 / 255.0), (1.0, 1.0)])
}

#[derive(Clone, Debug)]
pub struct FreqSepPrecomp {
    /// 全分辨率高反差遮罩（1 = 保留原图，0 = 取曲线提亮结果）
    pub mask: GrayF32,
}

/// 生成高反差遮罩。`radius` 以短边 `work_short` 像素为单位。
pub fn precompute_freqsep(orig: &ImgF32, work_short: f32, radius: f32) -> FreqSepPrecomp {
    let (work, scale) = orig.work_copy(work_short);
    // 曝光 -1EV 后 2·G·B = 0.5·g·b
    let gb: Vec<f32> = work.data.par_iter().map(|p| 0.5 * p[1] * p[2]).collect();
    let gb = GrayF32::from_vec(work.w, work.h, gb);
    let r = if scale < 1.0 {
        radius
    } else {
        radius * (orig.w.min(orig.h) as f32 / work_short).max(1.0)
    };
    let blur = gaussian_blur_gray(&gb, r);
    let mut mask = GrayF32::new(work.w, work.h);
    mask.data.par_iter_mut().enumerate().for_each(|(i, m)| {
        let hp = (gb.data[i] - blur.data[i] + 0.5).clamp(0.0, 1.0);
        let mut v = hp;
        for _ in 0..3 {
            v = hardlight(v, v);
        }
        let k = 255.0 / (164.0 - 75.0);
        *m = ((v - 75.0 / 255.0) * k).clamp(0.0, 1.0);
    });
    let mask = if scale < 1.0 {
        mask.resize(orig.w, orig.h)
    } else {
        mask
    };
    FreqSepPrecomp { mask }
}

/// 应用方案 B。`amount` 对应 YUCI inputAmount（曲线强度），`sharpness` 为 inputSharpnessFactor。
pub fn smooth_freqsep(
    orig: &ImgF32,
    pre: &FreqSepPrecomp,
    face_mask: Option<&GrayF32>,
    amount: f32,
    sharpness: f32,
    curve: &Curve256,
) -> ImgF32 {
    let mut out = ImgF32::new(orig.w, orig.h);
    out.data.par_iter_mut().enumerate().for_each(|(i, px)| {
        let s = orig.data[i];
        let m = pre.mask.data[i];
        let mut t = [0.0f32; 3];
        for c in 0..3 {
            let toned = s[c] + (curve.eval(s[c]) - s[c]) * amount;
            // mask 白 → 原图；黑 → toned
            t[c] = toned + (s[c] - toned) * m;
        }
        *px = t;
    });
    let sharp = sharpness * amount;
    if sharp > 0.0 {
        let blur = gaussian_blur_rgb(&out, 1.0);
        out.data.par_iter_mut().zip(&blur.data).for_each(|(p, b)| {
            for c in 0..3 {
                p[c] = (p[c] + (p[c] - b[c]) * sharp).clamp(0.0, 1.0);
            }
        });
    }
    if let Some(fm) = face_mask {
        out.data.par_iter_mut().enumerate().for_each(|(i, p)| {
            let m = fm.data[i];
            let s = orig.data[i];
            for c in 0..3 {
                p[c] = s[c] + (p[c] - s[c]) * m;
            }
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amount_zero_is_identity() {
        let mut img = ImgF32::new(40, 30);
        for (i, p) in img.data.iter_mut().enumerate() {
            *p = [0.7 + 0.1 * ((i % 7) as f32 / 7.0), 0.55, 0.45];
        }
        let pre = precompute_freqsep(&img, 1000.0, 8.0);
        let out = smooth_freqsep(&img, &pre, None, 0.0, 0.6, &default_skin_curve());
        assert!(out.max_abs_diff(&img) < 1e-6);
    }
}
