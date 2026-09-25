//! .cube 3D LUT 解析、三线性查表与导出。

use crate::buffer::ImgF32;
use rayon::prelude::*;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Lut3D {
    pub n: usize,
    /// index = r + n * (g + n * b)，r 变化最快（.cube 规范）。
    pub data: Vec<[f32; 3]>,
    pub domain_min: [f32; 3],
    pub domain_max: [f32; 3],
    pub title: Option<String>,
}

impl Lut3D {
    pub fn parse_cube(text: &str) -> Result<Self, String> {
        let mut n = 0usize;
        let mut data = Vec::new();
        let mut domain_min = [0.0f32; 3];
        let mut domain_max = [1.0f32; 3];
        let mut title = None;
        for line in text.lines() {
            let l = line.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            if let Some(rest) = l.strip_prefix("TITLE") {
                title = Some(rest.trim().trim_matches('"').to_string());
                continue;
            }
            if let Some(rest) = l.strip_prefix("DOMAIN_MIN") {
                let v: Vec<f32> = rest
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if v.len() == 3 {
                    domain_min = [v[0], v[1], v[2]];
                }
                continue;
            }
            if let Some(rest) = l.strip_prefix("DOMAIN_MAX") {
                let v: Vec<f32> = rest
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if v.len() == 3 {
                    domain_max = [v[0], v[1], v[2]];
                }
                continue;
            }
            if let Some(rest) = l.strip_prefix("LUT_3D_SIZE") {
                n = rest
                    .trim()
                    .parse()
                    .map_err(|_| "bad LUT_3D_SIZE".to_string())?;
                continue;
            }
            if l.starts_with("LUT_1D_SIZE") {
                return Err("1D LUT not supported".into());
            }
            if l.starts_with(|c: char| c.is_ascii_alphabetic()) {
                // 其他未知关键字（如 LUT_3D_INPUT_RANGE）忽略
                continue;
            }
            let v: Vec<f32> = l
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if v.len() == 3 {
                data.push([v[0], v[1], v[2]]);
            }
        }
        if n == 0 || data.len() != n * n * n {
            return Err(format!(
                "size mismatch: LUT_3D_SIZE={n}, entries={}",
                data.len()
            ));
        }
        Ok(Self {
            n,
            data,
            domain_min,
            domain_max,
            title,
        })
    }

    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())?;
        Self::parse_cube(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.as_ref().display()))
    }

    /// 恒等 LUT。
    pub fn identity(n: usize) -> Self {
        let m = (n - 1) as f32;
        let mut data = Vec::with_capacity(n * n * n);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    data.push([r as f32 / m, g as f32 / m, b as f32 / m]);
                }
            }
        }
        Self {
            n,
            data,
            domain_min: [0.0; 3],
            domain_max: [1.0; 3],
            title: None,
        }
    }

    /// 用任意颜色映射函数烘焙 LUT。
    pub fn bake<F: Fn([f32; 3]) -> [f32; 3]>(n: usize, f: F) -> Self {
        let mut lut = Self::identity(n);
        for v in lut.data.iter_mut() {
            *v = f(*v);
        }
        lut
    }

    /// 序列化为 .cube 文本。
    pub fn to_cube_string(&self) -> String {
        let mut s = String::new();
        if let Some(t) = &self.title {
            s.push_str(&format!("TITLE \"{t}\"\n"));
        }
        s.push_str(&format!("LUT_3D_SIZE {}\n", self.n));
        s.push_str(&format!(
            "DOMAIN_MIN {} {} {}\nDOMAIN_MAX {} {} {}\n",
            self.domain_min[0],
            self.domain_min[1],
            self.domain_min[2],
            self.domain_max[0],
            self.domain_max[1],
            self.domain_max[2]
        ));
        for v in &self.data {
            s.push_str(&format!("{:.6} {:.6} {:.6}\n", v[0], v[1], v[2]));
        }
        s
    }

    #[inline]
    fn at(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        self.data[r + self.n * (g + self.n * b)]
    }

    /// 三线性查表。
    #[inline]
    pub fn lookup(&self, c: [f32; 3]) -> [f32; 3] {
        let m = (self.n - 1) as f32;
        let mut f = [0.0f32; 3];
        for i in 0..3 {
            let range = (self.domain_max[i] - self.domain_min[i]).max(1e-6);
            f[i] = ((c[i] - self.domain_min[i]) / range).clamp(0.0, 1.0) * m;
        }
        let i0 = [f[0] as usize, f[1] as usize, f[2] as usize];
        let i1 = [
            (i0[0] + 1).min(self.n - 1),
            (i0[1] + 1).min(self.n - 1),
            (i0[2] + 1).min(self.n - 1),
        ];
        let t = [
            f[0] - i0[0] as f32,
            f[1] - i0[1] as f32,
            f[2] - i0[2] as f32,
        ];
        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        };
        let c00 = lerp(
            self.at(i0[0], i0[1], i0[2]),
            self.at(i1[0], i0[1], i0[2]),
            t[0],
        );
        let c10 = lerp(
            self.at(i0[0], i1[1], i0[2]),
            self.at(i1[0], i1[1], i0[2]),
            t[0],
        );
        let c01 = lerp(
            self.at(i0[0], i0[1], i1[2]),
            self.at(i1[0], i0[1], i1[2]),
            t[0],
        );
        let c11 = lerp(
            self.at(i0[0], i1[1], i1[2]),
            self.at(i1[0], i1[1], i1[2]),
            t[0],
        );
        let c0 = lerp(c00, c10, t[1]);
        let c1 = lerp(c01, c11, t[1]);
        lerp(c0, c1, t[2])
    }
}

/// 对整幅图应用 3D LUT 并按 intensity 混合。
pub fn apply_lut3d(img: &mut ImgF32, lut: &Lut3D, intensity: f32) {
    if intensity <= 0.0 {
        return;
    }
    img.data.par_iter_mut().for_each(|p| {
        let n = lut.lookup(*p);
        for c in 0..3 {
            p[c] += (n[c] - p[c]) * intensity;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_cube_roundtrip() {
        let lut = Lut3D::identity(33);
        let text = lut.to_cube_string();
        let parsed = Lut3D::parse_cube(&text).unwrap();
        assert_eq!(parsed.n, 33);
        let mut img = ImgF32::new(32, 32);
        for (i, p) in img.data.iter_mut().enumerate() {
            let x = (i % 32) as f32 / 31.0;
            let y = (i / 32) as f32 / 31.0;
            *p = [x, y, (x * 0.3 + y * 0.7) % 1.0];
        }
        let orig = img.clone();
        apply_lut3d(&mut img, &parsed, 1.0);
        assert!(img.max_abs_diff(&orig) <= 1.0 / 255.0);
    }

    #[test]
    fn rejects_bad_size() {
        assert!(Lut3D::parse_cube("LUT_3D_SIZE 2\n0 0 0\n").is_err());
        assert!(Lut3D::parse_cube("LUT_1D_SIZE 2\n").is_err());
    }
}
