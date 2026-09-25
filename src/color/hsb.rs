//! `GPUImageHSBFilter` 的 adjustBrightness + adjustSaturation 等价实现。
//!
//! 注意：GPUImage（以及美狐仓库内的副本）实际编译使用的亮度权重是 PDF 规范值
//! `RLUM = 0.3, GLUM = 0.59, BLUM = 0.11`（graficaobscura 的 0.3086/0.6094/0.0820 在源码中被注释掉）。
//! 方案文档 2.2 节写的是被注释掉的那组，本实现以源码为准。
//!
//! 亮度矩阵是各向同性缩放，与饱和矩阵可交换，因此结果与两次 adjust 的先后顺序无关：
//! `c' = clamp(brightness * (lum + (c - lum) * saturation), 0, 1)`。

use crate::buffer::ImgF32;
use rayon::prelude::*;

pub const RLUM: f32 = 0.3;
pub const GLUM: f32 = 0.59;
pub const BLUM: f32 = 0.11;

#[inline]
pub fn hsb_px(p: [f32; 3], brightness: f32, saturation: f32) -> [f32; 3] {
    let lum = RLUM * p[0] + GLUM * p[1] + BLUM * p[2];
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let v = lum + (p[c] - lum) * saturation;
        out[c] = (v * brightness).clamp(0.0, 1.0);
    }
    out
}

pub fn hsb_brightness_saturation(img: &mut ImgF32, brightness: f32, saturation: f32) {
    if (brightness - 1.0).abs() < 1e-6 && (saturation - 1.0).abs() < 1e-6 {
        return;
    }
    img.data
        .par_iter_mut()
        .for_each(|p| *p = hsb_px(*p, brightness, saturation));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsb_identity_and_gray_invariance() {
        let p = [0.2, 0.5, 0.7];
        let q = hsb_px(p, 1.0, 1.0);
        assert!((0..3).all(|c| (p[c] - q[c]).abs() < 1e-6));
        // 灰色在任意饱和度下不变，亮度 1.1 时按比例放大
        let g = hsb_px([0.4, 0.4, 0.4], 1.1, 1.7);
        assert!((0..3).all(|c| (g[c] - 0.44).abs() < 1e-5));
    }
}
