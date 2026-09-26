//! 盒式均值、转置与快速导向滤波（He, Sun, Tang. Guided Image Filtering, TPAMI'12；
//! Fast Guided Filter, 2015 的下采样系数版本）。
//!
//! 导向滤波是奶油肌模式的核心：以亮度为引导，对亮度 / 色度做保边平滑，
//! 在皮肤内压平毛孔与色斑而不越过五官、发际线等强边缘（无双边滤波的"梯度反转"光晕）。

use crate::buffer::GrayF32;
use rayon::prelude::*;

/// 转置。
pub fn transpose(src: &GrayF32) -> GrayF32 {
    let (w, h) = (src.w, src.h);
    let mut out = GrayF32::new(h, w);
    out.data.par_chunks_mut(h).enumerate().for_each(|(x, row)| {
        for (y, v) in row.iter_mut().enumerate() {
            *v = src.data[y * w + x];
        }
    });
    out
}

/// 逐行滑动窗口均值（窗口 2r+1，边缘按实际覆盖像素数归一化）。
fn box_pass_rows(src: &GrayF32, r: usize) -> GrayF32 {
    let (w, h) = (src.w, src.h);
    let mut out = GrayF32::new(w, h);
    if w == 0 || h == 0 {
        return out;
    }
    out.data
        .par_chunks_mut(w)
        .zip(src.data.par_chunks(w))
        .for_each(|(o, s)| {
            let mut prefix = vec![0.0f64; w + 1];
            for i in 0..w {
                prefix[i + 1] = prefix[i] + s[i] as f64;
            }
            for (x, v) in o.iter_mut().enumerate() {
                let lo = x.saturating_sub(r);
                let hi = (x + r).min(w - 1);
                *v = ((prefix[hi + 1] - prefix[lo]) / (hi - lo + 1) as f64) as f32;
            }
        });
    out
}

/// 边缘钳制的盒式均值（窗口 2r+1），分离实现。
pub fn box_filter(src: &GrayF32, r: usize) -> GrayF32 {
    box_filter_n(src, r, 1)
}

/// 连续 `n` 次盒式均值（只做两次转置）。三次盒式 ≈ 高斯。
pub fn box_filter_n(src: &GrayF32, r: usize, n: usize) -> GrayF32 {
    if r == 0 || n == 0 {
        return src.clone();
    }
    let mut h = box_pass_rows(src, r);
    for _ in 1..n {
        h = box_pass_rows(&h, r);
    }
    let mut t = transpose(&h);
    for _ in 0..n {
        t = box_pass_rows(&t, r);
    }
    transpose(&t)
}

/// 三次盒式模糊近似高斯（O(1)/像素，与 σ 无关；适合遮罩、低频与大 σ 用途）。
pub fn fast_gaussian(src: &GrayF32, sigma: f32) -> GrayF32 {
    if sigma <= 0.8 {
        return crate::skin::gaussian::gaussian_blur_gray(src, sigma);
    }
    // 三个宽 W=2r+1 的盒式方差和 = 3(W²−1)/12 = σ²  →  r = (√(4σ²+1) − 1)/2
    let r = (((4.0 * sigma * sigma + 1.0).sqrt() - 1.0) * 0.5)
        .round()
        .max(1.0) as usize;
    box_filter_n(src, r, 3)
}

/// 遮罩内的高斯低通（归一化卷积 `G(v·m) / G(m)`）：遮罩外的背景、衣物不参与平均，
/// 皮肤边缘处的低频值不会被拉亮或拉暗。遮罩权重过小处保留原值。
pub fn masked_gaussian(src: &GrayF32, mask: &GrayF32, sigma: f32) -> GrayF32 {
    let weighted: Vec<f32> = src
        .data
        .par_iter()
        .zip(&mask.data)
        .map(|(v, m)| v * m)
        .collect();
    let num = fast_gaussian(&GrayF32::from_vec(src.w, src.h, weighted), sigma);
    let den = fast_gaussian(mask, sigma);
    let data = num
        .data
        .par_iter()
        .zip(&den.data)
        .zip(&src.data)
        .map(|((n, d), v)| if *d > 1e-3 { n / d } else { *v })
        .collect();
    GrayF32::from_vec(src.w, src.h, data)
}

/// 导向滤波系数计算的下采样倍数（按全分辨率半径选择：半径越大越可以粗算，质量几乎不变）。
pub fn subsample_for(radius: usize) -> usize {
    if radius >= 8 {
        4
    } else if radius >= 3 {
        2
    } else {
        1
    }
}

fn zip_map(a: &GrayF32, b: &GrayF32, f: impl Fn(f32, f32) -> f32 + Sync) -> GrayF32 {
    let data: Vec<f32> = a
        .data
        .par_iter()
        .zip(&b.data)
        .map(|(x, y)| f(*x, *y))
        .collect();
    GrayF32::from_vec(a.w, a.h, data)
}

/// 快速导向滤波。`guide` 为引导图（一般为亮度 L），`p` 为待滤波图（与 guide 同尺寸），
/// `radius` 为全分辨率像素半径，`eps` 为正则项（量纲同 guide²），
/// `subsample` ≥ 1 为系数计算的下采样倍数（速度约提升 subsample² 倍，质量几乎不变）。
pub fn guided_filter(
    p: &GrayF32,
    guide: &GrayF32,
    radius: usize,
    eps: f32,
    subsample: usize,
) -> GrayF32 {
    assert_eq!((p.w, p.h), (guide.w, guide.h));
    let s = subsample.max(1);
    let (w, h) = (p.w, p.h);
    let (sw, sh) = ((w / s).max(1), (h / s).max(1));
    let (i_s, p_s) = if s > 1 {
        (guide.resize(sw, sh), p.resize(sw, sh))
    } else {
        (guide.clone(), p.clone())
    };
    let r_s = (radius / s).max(1);
    let mean_i = box_filter(&i_s, r_s);
    let mean_p = box_filter(&p_s, r_s);
    let corr_i = box_filter(&zip_map(&i_s, &i_s, |x, y| x * y), r_s);
    let corr_ip = box_filter(&zip_map(&i_s, &p_s, |x, y| x * y), r_s);
    let n = sw * sh;
    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    a.par_iter_mut()
        .zip(b.par_iter_mut())
        .enumerate()
        .for_each(|(k, (av, bv))| {
            let mi = mean_i.data[k];
            let mp = mean_p.data[k];
            let var = corr_i.data[k] - mi * mi;
            let cov = corr_ip.data[k] - mi * mp;
            let ak = cov / (var + eps);
            *av = ak;
            *bv = mp - ak * mi;
        });
    let mean_a = box_filter(&GrayF32::from_vec(sw, sh, a), r_s);
    let mean_b = box_filter(&GrayF32::from_vec(sw, sh, b), r_s);
    let (ma, mb) = if s > 1 {
        (mean_a.resize(w, h), mean_b.resize(w, h))
    } else {
        (mean_a, mean_b)
    };
    let data: Vec<f32> = ma
        .data
        .par_iter()
        .zip(&mb.data)
        .zip(&guide.data)
        .map(|((a, b), g)| a * g + b)
        .collect();
    GrayF32::from_vec(w, h, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_filter_constant_and_mean() {
        let g = GrayF32::filled(20, 12, 0.3);
        let b = box_filter(&g, 3);
        assert!(b.data.iter().all(|v| (v - 0.3).abs() < 1e-6));
        let mut g = GrayF32::new(5, 1);
        g.data = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let b = box_filter(&g, 1);
        assert!((b.data[2] - 1.0 / 3.0).abs() < 1e-6 && (b.data[0]).abs() < 1e-6);
    }

    #[test]
    fn masked_lowpass_ignores_pixels_outside_the_mask() {
        let (w, h) = (60, 20);
        let mut src = GrayF32::new(w, h);
        let mut mask = GrayF32::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let inside = x < 30;
                src.data[y * w + x] = if inside { 50.0 } else { 95.0 };
                mask.data[y * w + x] = if inside { 1.0 } else { 0.0 };
            }
        }
        let low = masked_gaussian(&src, &mask, 4.0);
        // 皮肤边缘处的低频值不被旁边的亮背景拉高
        assert!((low.data[10 * w + 29] - 50.0).abs() < 1e-3);
        assert!((low.data[10 * w + 5] - 50.0).abs() < 1e-3);
    }

    #[test]
    fn guided_filter_preserves_step_and_smooths_noise() {
        let (w, h) = (64, 32);
        let mut g = GrayF32::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let base = if x < 32 { 20.0 } else { 60.0 };
                let noise = (((x * 7 + y * 13) % 5) as f32 - 2.0) * 1.0;
                g.data[y * w + x] = base + noise;
            }
        }
        let q = guided_filter(&g, &g, 4, 4.0, 1);
        // 边缘两侧均值保持
        let left: f32 = (0..h).map(|y| q.get(10, y)).sum::<f32>() / h as f32;
        let right: f32 = (0..h).map(|y| q.get(54, y)).sum::<f32>() / h as f32;
        assert!((left - 20.0).abs() < 1.0 && (right - 60.0).abs() < 1.0);
        // 噪声被压低
        let var_in: f32 = (0..h).map(|y| (g.get(10, y) - 20.0).powi(2)).sum::<f32>() / h as f32;
        let var_out: f32 = (0..h).map(|y| (q.get(10, y) - left).powi(2)).sum::<f32>() / h as f32;
        assert!(var_out < var_in * 0.5, "{var_in} -> {var_out}");
        // 边缘不模糊：边缘两侧相邻像素差仍接近 40
        let edge = q.get(32, 16) - q.get(31, 16);
        assert!(edge > 25.0, "edge {edge}");
    }
}
