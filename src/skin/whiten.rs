//! 美白：参数化曲线（零素材）或 512 查找图（美狐 `white.png` / gpupixel 风格），均乘以皮肤遮罩。

use crate::buffer::{GrayF32, ImgF32};
use crate::color::lookup512::lookup_px;
use rayon::prelude::*;

/// 参数化美白：`c' = pow(c, 1 / (1 + 0.35·whiten))`，并轻微降低红色饱和（更接近"通透"而非"发黄"）。
#[inline]
pub fn whiten_px_curve(p: [f32; 3], whiten: f32) -> [f32; 3] {
    let gamma = 1.0 / (1.0 + 0.35 * whiten);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        out[c] = p[c].clamp(0.0, 1.0).powf(gamma);
    }
    // 降红：把 r 向 (g+b)/2 靠拢 8%·whiten
    let gb = 0.5 * (out[1] + out[2]);
    out[0] += (gb - out[0]) * 0.08 * whiten;
    out
}

/// 曲线模式美白。`mask` 为皮肤遮罩（None = 全图）。
pub fn apply_curve(img: &mut ImgF32, mask: Option<&GrayF32>, whiten: f32) {
    if whiten <= 0.0 {
        return;
    }
    img.data.par_iter_mut().enumerate().for_each(|(i, p)| {
        let m = mask.map(|m| m.data[i]).unwrap_or(1.0);
        if m <= 0.0 {
            return;
        }
        let n = whiten_px_curve(*p, whiten);
        for c in 0..3 {
            p[c] += (n[c] - p[c]) * m;
        }
    });
}

/// 查找图模式美白：`intensity = whiten`，再按皮肤遮罩混合。
pub fn apply_lookup(img: &mut ImgF32, lut: &ImgF32, mask: Option<&GrayF32>, whiten: f32) {
    if whiten <= 0.0 {
        return;
    }
    img.data.par_iter_mut().enumerate().for_each(|(i, p)| {
        let m = mask.map(|m| m.data[i]).unwrap_or(1.0);
        if m <= 0.0 {
            return;
        }
        let n = lookup_px(lut, *p);
        for c in 0..3 {
            p[c] += (n[c] - p[c]) * whiten * m;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_brightens_and_zero_is_identity() {
        let p = [0.5, 0.4, 0.35];
        let q = whiten_px_curve(p, 1.0);
        assert!(q[1] > p[1] && q[2] > p[2]);
        let z = whiten_px_curve(p, 0.0);
        assert!((0..3).all(|c| (z[c] - p[c]).abs() < 1e-6));
    }
}
