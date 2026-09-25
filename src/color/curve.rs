//! 一维颜色曲线（256 表），控制点 → 自然三次样条（与 GPUImage / YUCIRGBToneCurve 算法一致）。

use crate::buffer::ImgF32;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Curve256 {
    pub table: [f32; 256],
}

impl Default for Curve256 {
    fn default() -> Self {
        Self::identity()
    }
}

/// 自然三次样条的二阶导（GPUImageToneCurveFilter `secondDerivative` 的等价实现）。
fn second_derivatives(pts: &[(f64, f64)]) -> Vec<f64> {
    let n = pts.len();
    if n < 3 {
        return vec![0.0; n];
    }
    // 三对角系统：h_{i-1} M_{i-1} + 2 (h_{i-1}+h_i) M_i + h_i M_{i+1} = 6 (d_i - d_{i-1})
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut r = vec![0.0; n];
    b[0] = 1.0;
    b[n - 1] = 1.0;
    for i in 1..n - 1 {
        let h0 = pts[i].0 - pts[i - 1].0;
        let h1 = pts[i + 1].0 - pts[i].0;
        a[i] = h0;
        b[i] = 2.0 * (h0 + h1);
        c[i] = h1;
        r[i] = 6.0 * ((pts[i + 1].1 - pts[i].1) / h1 - (pts[i].1 - pts[i - 1].1) / h0);
    }
    // Thomas 算法
    for i in 1..n {
        let m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        r[i] -= m * r[i - 1];
    }
    let mut x = vec![0.0; n];
    x[n - 1] = r[n - 1] / b[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = (r[i] - c[i] * x[i + 1]) / b[i];
    }
    x
}

impl Curve256 {
    pub fn identity() -> Self {
        let mut table = [0.0f32; 256];
        for (i, v) in table.iter_mut().enumerate() {
            *v = i as f32 / 255.0;
        }
        Self { table }
    }

    /// 由 0..1 域内的控制点构造。与 GPUImage 一致：按 x 排序、转到 0..255 域、自然三次样条，
    /// 首点之前取 0、末点之后取 255，结果钳制到 0..255。
    pub fn from_control_points(points: &[(f32, f32)]) -> Self {
        let mut pts: Vec<(f64, f64)> = points
            .iter()
            .map(|&(x, y)| ((x as f64) * 255.0, (y as f64) * 255.0))
            .collect();
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
        let mut table = [0.0f32; 256];
        if pts.is_empty() {
            return Self::identity();
        }
        if pts.len() == 1 {
            // 单个控制点无法定义曲线，退化为恒等
            return Self::identity();
        }
        let m = second_derivatives(&pts);
        let first_x = pts[0].0;
        let last_x = pts[pts.len() - 1].0;
        for (i, v) in table.iter_mut().enumerate() {
            let x = i as f64;
            let y = if x < first_x.floor() {
                0.0
            } else if x > last_x {
                255.0
            } else {
                // 找到区段
                let mut k = 0;
                while k + 2 < pts.len() && x > pts[k + 1].0 {
                    k += 1;
                }
                let (x0, y0) = pts[k];
                let (x1, y1) = pts[k + 1];
                let h = (x1 - x0).max(1e-9);
                let a = (x1 - x) / h;
                let b = (x - x0) / h;
                a * y0
                    + b * y1
                    + ((a * a * a - a) * m[k] + (b * b * b - b) * m[k + 1]) * h * h / 6.0
            };
            *v = (y.clamp(0.0, 255.0) / 255.0) as f32;
        }
        Self { table }
    }

    /// 线性插值查表。
    #[inline]
    pub fn eval(&self, v: f32) -> f32 {
        let f = v.clamp(0.0, 1.0) * 255.0;
        let i = f as usize;
        let j = (i + 1).min(255);
        let t = f - i as f32;
        self.table[i] + (self.table[j] - self.table[i]) * t
    }

    /// 对 RGB 三通道同时应用并按 intensity 混合。
    pub fn apply_rgb(&self, img: &mut ImgF32, intensity: f32) {
        if intensity <= 0.0 {
            return;
        }
        img.data.par_iter_mut().for_each(|p| {
            for v in p.iter_mut() {
                let n = self.eval(*v);
                *v += (n - *v) * intensity;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_control_points_give_identity() {
        let c = Curve256::from_control_points(&[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]);
        for i in 0..256 {
            assert!((c.table[i] - i as f32 / 255.0).abs() < 1e-4);
        }
    }

    #[test]
    fn yuci_curve_brightens_midtones_and_passes_control_points() {
        let c = Curve256::from_control_points(&[
            (0.0, 0.0),
            (120.0 / 255.0, 146.0 / 255.0),
            (1.0, 1.0),
        ]);
        assert!((c.table[120] - 146.0 / 255.0).abs() < 1e-3);
        assert!(c.table[0].abs() < 1e-6 && (c.table[255] - 1.0).abs() < 1e-6);
        assert!(c.table[60] > 60.0 / 255.0);
    }
}
