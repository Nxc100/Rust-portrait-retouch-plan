//! `GPUImageSobelEdgeDetectionFilter`：灰度（0.2125, 0.7154, 0.0721）+ 3×3 Sobel，输出梯度模长。

use crate::buffer::{GrayF32, ImgF32};
use rayon::prelude::*;

pub const LUMA: [f32; 3] = [0.2125, 0.7154, 0.0721];

/// 返回与 src 同尺寸的边缘强度图（edgeStrength = 1.0，步长 1 像素，边缘钳制）。
pub fn sobel_edge(src: &ImgF32) -> GrayF32 {
    let (w, h) = (src.w, src.h);
    let lum: Vec<f32> = src
        .data
        .par_iter()
        .map(|p| LUMA[0] * p[0] + LUMA[1] * p[1] + LUMA[2] * p[2])
        .collect();
    let at = |x: isize, y: isize| -> f32 {
        let xi = x.clamp(0, w as isize - 1) as usize;
        let yi = y.clamp(0, h as isize - 1) as usize;
        lum[yi * w + xi]
    };
    let mut out = GrayF32::new(w, h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let y = y as isize;
        for (x, out_px) in row.iter_mut().enumerate() {
            let xi = x as isize;
            let (tl, t, tr) = (at(xi - 1, y - 1), at(xi, y - 1), at(xi + 1, y - 1));
            let (l, r) = (at(xi - 1, y), at(xi + 1, y));
            let (bl, b, br) = (at(xi - 1, y + 1), at(xi, y + 1), at(xi + 1, y + 1));
            let hh = -tl - 2.0 * t - tr + bl + 2.0 * b + br;
            let vv = -bl - 2.0 * l - tl + br + 2.0 * r + tr;
            *out_px = (hh * hh + vv * vv).sqrt();
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_is_zero_and_step_is_strong() {
        let flat = ImgF32::filled(16, 16, [0.5; 3]);
        assert!(sobel_edge(&flat).data.iter().all(|v| v.abs() < 1e-6));
        let mut step = ImgF32::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                step.data[y * 16 + x] = if x < 8 { [0.0; 3] } else { [1.0; 3] };
            }
        }
        let e = sobel_edge(&step);
        assert!(e.get(7, 8) > 0.2 && e.get(8, 8) > 0.2 && e.get(2, 8) < 1e-6);
    }
}
