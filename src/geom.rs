//! 二维点 / 向量与 2×3 仿射变换。

use serde::{Deserialize, Serialize};

/// 像素坐标点（像素中心制：像素 (x, y) 的中心为 (x + 0.5, y + 0.5)）。
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct P {
    pub x: f32,
    pub y: f32,
}

#[allow(clippy::should_implement_trait)]
impl P {
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    #[inline]
    pub fn sub(self, o: P) -> P {
        P::new(self.x - o.x, self.y - o.y)
    }
    #[inline]
    pub fn add(self, o: P) -> P {
        P::new(self.x + o.x, self.y + o.y)
    }
    #[inline]
    pub fn mul(self, k: f32) -> P {
        P::new(self.x * k, self.y * k)
    }
    #[inline]
    pub fn dot(self, o: P) -> f32 {
        self.x * o.x + self.y * o.y
    }
    #[inline]
    pub fn len(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    #[inline]
    pub fn dist(self, o: P) -> f32 {
        self.sub(o).len()
    }
    #[inline]
    pub fn norm(self) -> P {
        let l = self.len();
        if l > 1e-6 {
            self.mul(1.0 / l)
        } else {
            P::new(0.0, 0.0)
        }
    }
    #[inline]
    pub fn lerp(self, o: P, t: f32) -> P {
        P::new(self.x + (o.x - self.x) * t, self.y + (o.y - self.y) * t)
    }
    #[inline]
    pub fn mid(self, o: P) -> P {
        self.lerp(o, 0.5)
    }
    /// 旋转 90°：(x, y) -> (-y, x)。
    #[inline]
    pub fn perp(self) -> P {
        P::new(-self.y, self.x)
    }
}

/// 平均点。
pub fn centroid(pts: &[P]) -> P {
    if pts.is_empty() {
        return P::default();
    }
    let mut s = P::default();
    for p in pts {
        s = s.add(*p);
    }
    s.mul(1.0 / pts.len() as f32)
}

/// 2×3 仿射：`x' = a x + b y + c; y' = d x + e y + f`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
}

impl Affine {
    pub const IDENTITY: Affine = Affine {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 0.0,
        e: 1.0,
        f: 0.0,
    };

    /// 以 `center` 为中心、缩放 `scale`、旋转 `rot`（弧度），并把中心平移到
    /// `(out_size/2, out_size/2)` 的变换（与 insightface `face_align.transform`
    /// 及 MediaPipe 的 ROI 裁剪一致）。
    pub fn crop(center: P, scale: f32, rot: f32, out_size: f32) -> Affine {
        let (s, c) = rot.sin_cos();
        let a = scale * c;
        let b = -scale * s;
        let d = scale * s;
        let e = scale * c;
        let half = out_size * 0.5;
        Affine {
            a,
            b,
            c: half - (a * center.x + b * center.y),
            d,
            e,
            f: half - (d * center.x + e * center.y),
        }
    }

    #[inline]
    pub fn apply(&self, p: P) -> P {
        P::new(
            self.a * p.x + self.b * p.y + self.c,
            self.d * p.x + self.e * p.y + self.f,
        )
    }

    pub fn inverse(&self) -> Affine {
        let det = self.a * self.e - self.b * self.d;
        let inv = if det.abs() < 1e-12 { 0.0 } else { 1.0 / det };
        let ia = self.e * inv;
        let ib = -self.b * inv;
        let id = -self.d * inv;
        let ie = self.a * inv;
        Affine {
            a: ia,
            b: ib,
            c: -(ia * self.c + ib * self.f),
            d: id,
            e: ie,
            f: -(id * self.c + ie * self.f),
        }
    }

    /// 变换中的各向同性缩放系数。
    pub fn scale(&self) -> f32 {
        (self.a * self.a + self.d * self.d).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_roundtrip() {
        let m = Affine::crop(P::new(100.0, 50.0), 0.5, 0.3, 192.0);
        let inv = m.inverse();
        let p = P::new(123.0, 45.0);
        let q = inv.apply(m.apply(p));
        assert!((q.x - p.x).abs() < 1e-3 && (q.y - p.y).abs() < 1e-3);
        let c = m.apply(P::new(100.0, 50.0));
        assert!((c.x - 96.0).abs() < 1e-4 && (c.y - 96.0).abs() < 1e-4);
    }
}
