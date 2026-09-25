//! sRGB（D65）↔ CIELAB 转换与平面缓冲。奶油肌模式在 Lab 空间分别处理亮度纹理与色度。

use crate::buffer::{GrayF32, ImgF32};
use rayon::prelude::*;

#[inline]
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
pub(crate) fn f_lab(t: f32) -> f32 {
    if t > 0.008_856 {
        t.cbrt()
    } else {
        7.787 * t + 16.0 / 116.0
    }
}

#[inline]
pub(crate) fn f_lab_inv(t: f32) -> f32 {
    let t3 = t * t * t;
    if t3 > 0.008_856 {
        t3
    } else {
        (t - 16.0 / 116.0) / 7.787
    }
}

const XN: f32 = 0.950_47;
const ZN: f32 = 1.088_83;

/// sRGB 0..1 → Lab（L 0..100，a/b 约 −128..127）。
#[inline]
pub fn rgb_to_lab(p: [f32; 3]) -> [f32; 3] {
    let r = srgb_to_linear(p[0].clamp(0.0, 1.0));
    let g = srgb_to_linear(p[1].clamp(0.0, 1.0));
    let b = srgb_to_linear(p[2].clamp(0.0, 1.0));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / XN;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / ZN;
    let (fx, fy, fz) = (f_lab(x), f_lab(y), f_lab(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// Lab → sRGB 0..1（钳制）。
#[inline]
pub fn lab_to_rgb(c: [f32; 3]) -> [f32; 3] {
    let fy = (c[0] + 16.0) / 116.0;
    let fx = fy + c[1] / 500.0;
    let fz = fy - c[2] / 200.0;
    let x = f_lab_inv(fx) * XN;
    let y = f_lab_inv(fy);
    let z = f_lab_inv(fz) * ZN;
    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;
    [
        linear_to_srgb(r.clamp(0.0, 1.0)),
        linear_to_srgb(g.clamp(0.0, 1.0)),
        linear_to_srgb(b.clamp(0.0, 1.0)),
    ]
}

/// Lab 三平面。
#[derive(Clone, Debug)]
pub struct LabPlanes {
    pub w: usize,
    pub h: usize,
    pub l: GrayF32,
    pub a: GrayF32,
    pub b: GrayF32,
}

impl LabPlanes {
    pub fn from_img(img: &ImgF32) -> Self {
        let n = img.w * img.h;
        let mut l = vec![0.0f32; n];
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        let w = img.w.max(1);
        l.par_chunks_mut(w)
            .zip(a.par_chunks_mut(w))
            .zip(b.par_chunks_mut(w))
            .zip(img.data.par_chunks(w))
            .for_each(|(((lr, ar), br), src)| {
                for (i, p) in src.iter().enumerate() {
                    let c = rgb_to_lab(*p);
                    lr[i] = c[0];
                    ar[i] = c[1];
                    br[i] = c[2];
                }
            });
        Self {
            w: img.w,
            h: img.h,
            l: GrayF32::from_vec(img.w, img.h, l),
            a: GrayF32::from_vec(img.w, img.h, a),
            b: GrayF32::from_vec(img.w, img.h, b),
        }
    }

    pub fn to_img(&self) -> ImgF32 {
        let mut out = ImgF32::new(self.w, self.h);
        let w = self.w.max(1);
        out.data
            .par_chunks_mut(w)
            .zip(self.l.data.par_chunks(w))
            .zip(self.a.data.par_chunks(w))
            .zip(self.b.data.par_chunks(w))
            .for_each(|(((dst, l), a), b)| {
                for i in 0..dst.len() {
                    dst[i] = lab_to_rgb([l[i], a[i], b[i]]);
                }
            });
        out
    }

    /// 把 Lab 结果按遮罩混回 `img`（遮罩为 0 的像素保持原值，避免往返误差）。
    pub fn blend_into(&self, img: &mut ImgF32, mask: &GrayF32) {
        assert_eq!((img.w, img.h), (self.w, self.h));
        let w = self.w.max(1);
        img.data
            .par_chunks_mut(w)
            .zip(self.l.data.par_chunks(w))
            .zip(self.a.data.par_chunks(w))
            .zip(self.b.data.par_chunks(w))
            .zip(mask.data.par_chunks(w))
            .for_each(|((((dst, l), a), b), m)| {
                for i in 0..dst.len() {
                    let t = m[i];
                    if t <= 0.0005 {
                        continue;
                    }
                    let n = lab_to_rgb([l[i], a[i], b[i]]);
                    let o = dst[i];
                    dst[i] = [
                        o[0] + (n[0] - o[0]) * t,
                        o[1] + (n[1] - o[1]) * t,
                        o[2] + (n[2] - o[2]) * t,
                    ];
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_roundtrip() {
        for &p in &[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.8, 0.6, 0.5],
            [0.2, 0.3, 0.7],
            [0.5, 0.5, 0.5],
        ] {
            let q = lab_to_rgb(rgb_to_lab(p));
            for c in 0..3 {
                assert!((q[c] - p[c]).abs() < 2e-4, "{p:?} -> {q:?}");
            }
        }
        let white = rgb_to_lab([1.0, 1.0, 1.0]);
        assert!((white[0] - 100.0).abs() < 0.05 && white[1].abs() < 0.1 && white[2].abs() < 0.1);
    }
}
