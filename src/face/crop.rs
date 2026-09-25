//! 仿射裁剪（等价 `cv2.warpAffine(img, M, (size, size))`，双线性，边界填 0，像素索引制）。

use crate::geom::{Affine, P};

/// 把原图按 `m`（原图索引坐标 → 裁剪索引坐标）裁剪为 `size×size` 的 RGB f32 图（0..255），行优先 HWC。
pub fn warp_affine_rgb8(img: &image::RgbImage, m: &Affine, size: usize) -> Vec<[f32; 3]> {
    let inv = m.inverse();
    let (w, h) = (img.width() as i64, img.height() as i64);
    let raw = img.as_raw();
    let mut out = vec![[0.0f32; 3]; size * size];
    let px = |x: i64, y: i64| -> [f32; 3] {
        if x < 0 || y < 0 || x >= w || y >= h {
            [0.0; 3]
        } else {
            let i = ((y * w + x) * 3) as usize;
            [raw[i] as f32, raw[i + 1] as f32, raw[i + 2] as f32]
        }
    };
    for v in 0..size {
        for u in 0..size {
            let s = inv.apply(P::new(u as f32, v as f32));
            let fx = s.x.floor();
            let fy = s.y.floor();
            let tx = s.x - fx;
            let ty = s.y - fy;
            let (x0, y0) = (fx as i64, fy as i64);
            let p00 = px(x0, y0);
            let p10 = px(x0 + 1, y0);
            let p01 = px(x0, y0 + 1);
            let p11 = px(x0 + 1, y0 + 1);
            let mut o = [0.0f32; 3];
            for c in 0..3 {
                let a = p00[c] + (p10[c] - p00[c]) * tx;
                let b = p01[c] + (p11[c] - p01[c]) * tx;
                o[c] = a + (b - a) * ty;
            }
            out[v * size + u] = o;
        }
    }
    out
}

/// HWC (0..255) → NCHW f32 张量，按 `scale` 与 `bias` 归一化：`v * scale + bias`。
pub fn to_nchw(hwc: &[[f32; 3]], size: usize, scale: f32, bias: f32) -> ndarray::Array4<f32> {
    let mut arr = ndarray::Array4::<f32>::zeros((1, 3, size, size));
    for y in 0..size {
        for x in 0..size {
            let p = hwc[y * size + x];
            for (c, v) in p.iter().enumerate() {
                arr[[0, c, y, x]] = v * scale + bias;
            }
        }
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_crop_copies_pixels() {
        let mut img = image::RgbImage::new(8, 8);
        for (i, p) in img.pixels_mut().enumerate() {
            *p = image::Rgb([i as u8, (i * 3) as u8, 7]);
        }
        let out = warp_affine_rgb8(&img, &Affine::IDENTITY, 8);
        for (i, px) in out.iter().enumerate() {
            assert_eq!(px[0], i as f32);
        }
    }
}
