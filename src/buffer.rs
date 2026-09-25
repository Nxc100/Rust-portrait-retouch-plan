//! f32 sRGB 图像缓冲、灰度缓冲、双线性采样与缩放。
//!
//! 约定：颜色在 sRGB 伽马空间以 f32（0..1）运算，不做线性化（与 GPUImage 一致）；
//! 坐标为像素中心制；采样等价于 `GL_LINEAR + CLAMP_TO_EDGE`。

use rayon::prelude::*;

/// 三通道 f32 图像（行优先，sRGB 0..1）。
#[derive(Clone, Debug)]
pub struct ImgF32 {
    pub w: usize,
    pub h: usize,
    pub data: Vec<[f32; 3]>,
}

/// 单通道 f32 图像（边缘图、遮罩）。
#[derive(Clone, Debug)]
pub struct GrayF32 {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

#[inline]
pub fn q8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// 计算把短边缩到 `short_side` 的缩放系数（只缩小不放大）。
pub fn work_scale(w: usize, h: usize, short_side: f32) -> f32 {
    let short = w.min(h).max(1) as f32;
    (short_side / short).min(1.0)
}

/// 分离式三角滤波重采样的权重表：对每个目标索引给出 (起始源索引, 权重列表)。
/// 缩小时滤波支撑按比例放大（等价 `image::imageops::FilterType::Triangle`），
/// 放大时退化为双线性插值。
fn triangle_weights(src_len: usize, dst_len: usize) -> Vec<(usize, Vec<f32>)> {
    let ratio = src_len as f32 / dst_len as f32;
    let support = ratio.max(1.0);
    let inv = 1.0 / support;
    (0..dst_len)
        .map(|d| {
            let center = (d as f32 + 0.5) * ratio;
            let left = (center - support).floor().max(0.0) as usize;
            let left = left.min(src_len - 1);
            let right = ((center + support).ceil() as usize)
                .min(src_len)
                .max(left + 1);
            let mut ws = Vec::with_capacity(right - left);
            let mut sum = 0.0f32;
            for i in left..right {
                let x = ((i as f32 + 0.5) - center).abs() * inv;
                let wgt = (1.0 - x).max(0.0);
                ws.push(wgt);
                sum += wgt;
            }
            if sum > 1e-12 {
                for wgt in &mut ws {
                    *wgt /= sum;
                }
            } else {
                let n = ws.len();
                for (i, wgt) in ws.iter_mut().enumerate() {
                    *wgt = if i == n / 2 { 1.0 } else { 0.0 };
                }
            }
            (left, ws)
        })
        .collect()
}

impl ImgF32 {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![[0.0; 3]; w * h],
        }
    }

    pub fn filled(w: usize, h: usize, px: [f32; 3]) -> Self {
        Self {
            w,
            h,
            data: vec![px; w * h],
        }
    }

    pub fn from_rgb8(img: &image::RgbImage) -> Self {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let raw = img.as_raw();
        let mut data = vec![[0.0f32; 3]; w * h];
        if w == 0 || h == 0 {
            return Self { w, h, data };
        }
        data.par_chunks_mut(w)
            .zip(raw.par_chunks(w * 3))
            .for_each(|(row, src)| {
                for (dst, s) in row.iter_mut().zip(src.chunks_exact(3)) {
                    *dst = [
                        s[0] as f32 / 255.0,
                        s[1] as f32 / 255.0,
                        s[2] as f32 / 255.0,
                    ];
                }
            });
        Self { w, h, data }
    }

    pub fn to_rgb8(&self) -> image::RgbImage {
        let mut raw = vec![0u8; self.w * self.h * 3];
        if self.w > 0 && self.h > 0 {
            raw.par_chunks_mut(self.w * 3)
                .zip(self.data.par_chunks(self.w))
                .for_each(|(dst, src)| {
                    for (d, s) in dst.chunks_exact_mut(3).zip(src) {
                        d[0] = q8(s[0]);
                        d[1] = q8(s[1]);
                        d[2] = q8(s[2]);
                    }
                });
        }
        image::RgbImage::from_raw(self.w as u32, self.h as u32, raw).expect("size mismatch")
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [f32; 3] {
        self.data[y * self.w + x]
    }

    /// 带边缘钳制的整数坐标读取。
    #[inline]
    pub fn get_clamped(&self, x: isize, y: isize) -> [f32; 3] {
        let xi = x.clamp(0, self.w as isize - 1) as usize;
        let yi = y.clamp(0, self.h as isize - 1) as usize;
        self.data[yi * self.w + xi]
    }

    /// 等价 GL_LINEAR + CLAMP_TO_EDGE；(x, y) 为像素中心制坐标。
    #[inline]
    pub fn sample_bilinear(&self, x: f32, y: f32) -> [f32; 3] {
        let fx = (x - 0.5).clamp(0.0, (self.w - 1) as f32);
        let fy = (y - 0.5).clamp(0.0, (self.h - 1) as f32);
        let x0 = fx as usize;
        let y0 = fy as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let (p00, p10, p01, p11) = (
            self.get(x0, y0),
            self.get(x1, y0),
            self.get(x0, y1),
            self.get(x1, y1),
        );
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let a = p00[c] + (p10[c] - p00[c]) * tx;
            let b = p01[c] + (p11[c] - p01[c]) * tx;
            out[c] = a + (b - a) * ty;
        }
        out
    }

    /// Catmull-Rom 双三次采样（像素中心制坐标，边缘钳制，结果钳制到 [0, 1]）。
    #[inline]
    pub fn sample_bicubic(&self, x: f32, y: f32) -> [f32; 3] {
        let o = self.sample_bicubic_unclamped(x, y);
        [
            o[0].clamp(0.0, 1.0),
            o[1].clamp(0.0, 1.0),
            o[2].clamp(0.0, 1.0),
        ]
    }

    /// Catmull-Rom 双三次采样，不钳制结果（用于带符号的细节层）。
    #[inline]
    pub fn sample_bicubic_unclamped(&self, x: f32, y: f32) -> [f32; 3] {
        #[inline]
        fn kern(t: f32) -> [f32; 4] {
            // a = −0.5
            let t2 = t * t;
            let t3 = t2 * t;
            [
                -0.5 * t3 + t2 - 0.5 * t,
                1.5 * t3 - 2.5 * t2 + 1.0,
                -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                0.5 * t3 - 0.5 * t2,
            ]
        }
        let fx = x - 0.5;
        let fy = y - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let wx = kern(fx - x0);
        let wy = kern(fy - y0);
        let (xi, yi) = (x0 as isize, y0 as isize);
        let mut out = [0.0f32; 3];
        if xi >= 1 && yi >= 1 && xi + 2 < self.w as isize && yi + 2 < self.h as isize {
            // 4×4 抽头全在图内：直接按行索引
            let (bx, by) = ((xi - 1) as usize, (yi - 1) as usize);
            for (j, wyj) in wy.iter().enumerate() {
                let row = &self.data[(by + j) * self.w + bx..(by + j) * self.w + bx + 4];
                let mut acc = [0.0f32; 3];
                for (p, wxi) in row.iter().zip(wx.iter()) {
                    acc[0] += p[0] * wxi;
                    acc[1] += p[1] * wxi;
                    acc[2] += p[2] * wxi;
                }
                out[0] += acc[0] * wyj;
                out[1] += acc[1] * wyj;
                out[2] += acc[2] * wyj;
            }
            return out;
        }
        for (j, wyj) in wy.iter().enumerate() {
            let mut row = [0.0f32; 3];
            for (i, wxi) in wx.iter().enumerate() {
                let p = self.get_clamped(xi - 1 + i as isize, yi - 1 + j as isize);
                row[0] += p[0] * wxi;
                row[1] += p[1] * wxi;
                row[2] += p[2] * wxi;
            }
            out[0] += row[0] * wyj;
            out[1] += row[1] * wyj;
            out[2] += row[2] * wyj;
        }
        out
    }

    /// 分离式三角滤波缩放（缩小时抗混叠，放大时为双线性）。
    pub fn resize(&self, nw: usize, nh: usize) -> ImgF32 {
        if nw == self.w && nh == self.h {
            return self.clone();
        }
        let wx = triangle_weights(self.w, nw);
        let wy = triangle_weights(self.h, nh);
        let mut tmp = vec![[0.0f32; 3]; nw * self.h];
        tmp.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
            let src = &self.data[y * self.w..(y + 1) * self.w];
            for (x, (left, ws)) in wx.iter().enumerate() {
                let mut acc = [0.0f32; 3];
                for (k, wgt) in ws.iter().enumerate() {
                    let s = src[left + k];
                    acc[0] += s[0] * wgt;
                    acc[1] += s[1] * wgt;
                    acc[2] += s[2] * wgt;
                }
                row[x] = acc;
            }
        });
        let mut out = ImgF32::new(nw, nh);
        out.data
            .par_chunks_mut(nw)
            .enumerate()
            .for_each(|(y, row)| {
                let (left, ws) = &wy[y];
                for x in 0..nw {
                    let mut acc = [0.0f32; 3];
                    for (k, wgt) in ws.iter().enumerate() {
                        let s = tmp[(left + k) * nw + x];
                        acc[0] += s[0] * wgt;
                        acc[1] += s[1] * wgt;
                        acc[2] += s[2] * wgt;
                    }
                    row[x] = acc;
                }
            });
        out
    }

    /// 缩放到短边 `short_side`（只缩小）；返回工作副本与缩放系数（1.0 表示未缩放，返回克隆）。
    pub fn work_copy(&self, short_side: f32) -> (ImgF32, f32) {
        let scale = work_scale(self.w, self.h, short_side);
        if scale >= 1.0 {
            (self.clone(), 1.0)
        } else {
            let nw = ((self.w as f32 * scale).round() as usize).max(1);
            let nh = ((self.h as f32 * scale).round() as usize).max(1);
            (self.resize(nw, nh), scale)
        }
    }

    /// 逐像素并行映射。
    pub fn map_inplace<F: Fn([f32; 3]) -> [f32; 3] + Sync>(&mut self, f: F) {
        self.data.par_iter_mut().for_each(|p| *p = f(*p));
    }

    /// 与另一幅同尺寸图像逐像素并行组合。
    pub fn zip_map<F: Fn([f32; 3], [f32; 3]) -> [f32; 3] + Sync>(
        &self,
        o: &ImgF32,
        f: F,
    ) -> ImgF32 {
        assert_eq!((self.w, self.h), (o.w, o.h));
        let data: Vec<[f32; 3]> = self
            .data
            .par_iter()
            .zip(&o.data)
            .map(|(a, b)| f(*a, *b))
            .collect();
        ImgF32 {
            w: self.w,
            h: self.h,
            data,
        }
    }

    /// 最大绝对差（用于测试）。
    pub fn max_abs_diff(&self, o: &ImgF32) -> f32 {
        self.data
            .iter()
            .zip(&o.data)
            .map(|(a, b)| (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0f32, f32::max))
            .fold(0.0f32, f32::max)
    }
}

impl GrayF32 {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![0.0; w * h],
        }
    }
    pub fn filled(w: usize, h: usize, v: f32) -> Self {
        Self {
            w,
            h,
            data: vec![v; w * h],
        }
    }
    pub fn from_vec(w: usize, h: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), w * h);
        Self { w, h, data }
    }
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }
    #[inline]
    pub fn get_clamped(&self, x: isize, y: isize) -> f32 {
        let xi = x.clamp(0, self.w as isize - 1) as usize;
        let yi = y.clamp(0, self.h as isize - 1) as usize;
        self.data[yi * self.w + xi]
    }
    #[inline]
    pub fn sample_bilinear(&self, x: f32, y: f32) -> f32 {
        let fx = (x - 0.5).clamp(0.0, (self.w - 1) as f32);
        let fy = (y - 0.5).clamp(0.0, (self.h - 1) as f32);
        let x0 = fx as usize;
        let y0 = fy as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let a = self.get(x0, y0) + (self.get(x1, y0) - self.get(x0, y0)) * tx;
        let b = self.get(x0, y1) + (self.get(x1, y1) - self.get(x0, y1)) * tx;
        a + (b - a) * ty
    }
    /// 分离式三角滤波缩放。
    pub fn resize(&self, nw: usize, nh: usize) -> GrayF32 {
        if nw == self.w && nh == self.h {
            return self.clone();
        }
        let wx = triangle_weights(self.w, nw);
        let wy = triangle_weights(self.h, nh);
        let mut tmp = vec![0.0f32; nw * self.h];
        tmp.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
            let src = &self.data[y * self.w..(y + 1) * self.w];
            for (x, (left, ws)) in wx.iter().enumerate() {
                let mut acc = 0.0f32;
                for (k, wgt) in ws.iter().enumerate() {
                    acc += src[left + k] * wgt;
                }
                row[x] = acc;
            }
        });
        let mut out = GrayF32::new(nw, nh);
        out.data
            .par_chunks_mut(nw)
            .enumerate()
            .for_each(|(y, row)| {
                let (left, ws) = &wy[y];
                for x in 0..nw {
                    let mut acc = 0.0f32;
                    for (k, wgt) in ws.iter().enumerate() {
                        acc += tmp[(left + k) * nw + x] * wgt;
                    }
                    row[x] = acc;
                }
            });
        out
    }
    pub fn map_inplace<F: Fn(f32) -> f32 + Sync>(&mut self, f: F) {
        self.data.par_iter_mut().for_each(|v| *v = f(*v));
    }
    /// 逐像素取最大（多人脸遮罩并集）。
    pub fn max_inplace(&mut self, o: &GrayF32) {
        assert_eq!((self.w, self.h), (o.w, o.h));
        self.data
            .par_iter_mut()
            .zip(&o.data)
            .for_each(|(a, b)| *a = a.max(*b));
    }
    /// 逐像素相乘。
    pub fn mul_inplace(&mut self, o: &GrayF32) {
        assert_eq!((self.w, self.h), (o.w, o.h));
        self.data
            .par_iter_mut()
            .zip(&o.data)
            .for_each(|(a, b)| *a *= *b);
    }
    pub fn to_luma8(&self) -> image::GrayImage {
        let raw: Vec<u8> = self.data.iter().map(|v| q8(*v)).collect();
        image::GrayImage::from_raw(self.w as u32, self.h as u32, raw).expect("size mismatch")
    }
    pub fn from_luma8(img: &image::GrayImage) -> Self {
        let data = img.as_raw().iter().map(|v| *v as f32 / 255.0).collect();
        Self {
            w: img.width() as usize,
            h: img.height() as usize,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb8_roundtrip_exact() {
        let mut img = image::RgbImage::new(7, 5);
        for (i, p) in img.pixels_mut().enumerate() {
            *p = image::Rgb([
                (i * 7 % 256) as u8,
                (i * 13 % 256) as u8,
                (i * 29 % 256) as u8,
            ]);
        }
        let f = ImgF32::from_rgb8(&img);
        let back = f.to_rgb8();
        assert_eq!(img.as_raw(), back.as_raw());
    }

    #[test]
    fn resize_constant_preserved() {
        let img = ImgF32::filled(40, 30, [0.25, 0.5, 0.75]);
        let small = img.resize(13, 7);
        let big = small.resize(80, 60);
        for p in &big.data {
            assert!(
                (p[0] - 0.25).abs() < 1e-5
                    && (p[1] - 0.5).abs() < 1e-5
                    && (p[2] - 0.75).abs() < 1e-5
            );
        }
    }

    #[test]
    fn bilinear_at_centers_is_exact() {
        let mut img = ImgF32::new(4, 3);
        for (i, p) in img.data.iter_mut().enumerate() {
            *p = [i as f32 / 12.0, 0.0, 1.0];
        }
        for y in 0..3 {
            for x in 0..4 {
                let s = img.sample_bilinear(x as f32 + 0.5, y as f32 + 0.5);
                assert!((s[0] - img.get(x, y)[0]).abs() < 1e-6);
            }
        }
        let s = img.sample_bilinear(-10.0, -10.0);
        assert!((s[0] - img.get(0, 0)[0]).abs() < 1e-6);
    }

    #[test]
    fn bicubic_exact_at_centers_and_on_linear_ramps() {
        let mut img = ImgF32::new(9, 7);
        for (i, p) in img.data.iter_mut().enumerate() {
            let (x, y) = ((i % 9) as f32, (i / 9) as f32);
            *p = [0.05 + 0.08 * x, 0.1 + 0.1 * y, 0.5];
        }
        for y in 0..7 {
            for x in 0..9 {
                let s = img.sample_bicubic(x as f32 + 0.5, y as f32 + 0.5);
                let g = img.get(x, y);
                assert!((s[0] - g[0]).abs() < 1e-5 && (s[1] - g[1]).abs() < 1e-5);
            }
        }
        // Catmull-Rom 精确再现线性斜坡（远离边缘）
        let s = img.sample_bicubic(4.25, 3.75);
        assert!((s[0] - (0.05 + 0.08 * 3.75)).abs() < 1e-5, "{}", s[0]);
        assert!((s[1] - (0.1 + 0.1 * 3.25)).abs() < 1e-5, "{}", s[1]);
        // 常数图恒等、边缘钳制
        let c = ImgF32::filled(5, 5, [0.3, 0.6, 0.9]);
        let s = c.sample_bicubic(-3.0, 7.2);
        assert!((s[0] - 0.3).abs() < 1e-6 && (s[2] - 0.9).abs() < 1e-6);
    }
}
