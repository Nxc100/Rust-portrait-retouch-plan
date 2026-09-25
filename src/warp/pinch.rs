//! 形变原语（像素空间，反向映射：输入为输出像素位置，返回应从原图采样的位置）。

use crate::geom::P;

/// 美狐 `warpPositionToUse1`：以 a 为圆心、半径 radius，把采样点沿 a→b 方向反向偏移 delta。
///
/// 原 shader 在归一化坐标中用 aspect_ratio 修正 y 后计算距离；像素空间 x/y 同尺度，修正项消失。
/// 原实现中位移方向 `normalize(B − A)` 在未修正的纹理坐标中计算，存在轻微各向异性（属原实现缺陷），
/// 此处采用各向同性的像素空间方向。
#[inline]
pub fn pinch(cur: P, a: P, b: P, radius: f32, delta: f32) -> P {
    let r = cur.dist(a);
    if r >= radius {
        return cur;
    }
    let dir = b.sub(a).norm();
    let d2 = radius * radius - r * r;
    let alpha = d2 / (d2 + (r - delta) * (r - delta));
    let alpha = alpha * alpha;
    cur.sub(dir.mul(alpha * delta))
}

/// 美狐 `adjust_eye`：以 center 为圆心的径向幂次缩放，`w = (d / radius)^k`。
/// 原实现用 `(d + 0.01) / radius` 规避 pow(0)（0.01 为归一化常数），像素空间用 1 px 下限。
#[inline]
pub fn enlarge(cur: P, center: P, radius: f32, k: f32) -> P {
    let d = cur.dist(center);
    if d >= radius {
        return cur;
    }
    let w = (d.max(1.0) / radius).powf(k);
    center.add(cur.sub(center).mul(w))
}

/// gpupixel `curveWarp`：origin 附近的采样点沿 origin→target 方向按线性衰减偏移
/// `offset = (target − origin) · delta · clamp(1 − d / radius, 0, 1)`，radius = |target − origin|。
#[inline]
pub fn curve_warp(cur: P, origin: P, target: P, delta: f32) -> P {
    let radius = target.dist(origin);
    if radius < 1e-3 {
        return cur;
    }
    let ratio = (1.0 - cur.dist(origin) / radius).clamp(0.0, 1.0);
    if ratio <= 0.0 {
        return cur;
    }
    cur.sub(target.sub(origin).mul(delta * ratio))
}

/// gpupixel `enlargeEye`：`w = clamp(1 − (1 − (d/r)²) · delta, 0, 1)`，`cur = origin + (cur − origin) · w`。
#[inline]
pub fn enlarge_gpupixel(cur: P, origin: P, radius: f32, delta: f32) -> P {
    let d = cur.dist(origin);
    if d >= radius {
        return cur;
    }
    let w = d / radius;
    let w = (1.0 - (1.0 - w * w) * delta).clamp(0.0, 1.0);
    origin.add(cur.sub(origin).mul(w))
}

/// 整体收窄：以 `center` 为中心、沿单位向量 `axis` 压缩。输出像素到中轴的有符号距离 d，
/// 采样点向外偏移 `d · k · w(r)`，`w = 1 − smoothstep(0, radius, r)`。
#[inline]
pub fn narrow(cur: P, center: P, axis: P, k: f32, radius: f32) -> P {
    let v = cur.sub(center);
    let r = v.len();
    if r >= radius || radius <= 0.0 {
        return cur;
    }
    let t = r / radius;
    let w = 1.0 - t * t * (3.0 - 2.0 * t);
    let d = v.dot(axis);
    cur.add(axis.mul(d * k * w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinch_delta_zero_is_identity() {
        let cur = P::new(10.0, 12.0);
        let out = pinch(cur, P::new(8.0, 8.0), P::new(50.0, 8.0), 30.0, 0.0);
        assert_eq!(out, cur);
    }

    #[test]
    fn pinch_outside_radius_unchanged_and_inside_moves_away_from_b() {
        let a = P::new(0.0, 0.0);
        let b = P::new(100.0, 0.0);
        assert_eq!(pinch(P::new(50.0, 0.0), a, b, 30.0, 5.0), P::new(50.0, 0.0));
        let p = pinch(P::new(1.0, 0.0), a, b, 30.0, 5.0);
        assert!(p.x < 1.0 && p.x > -5.0);
    }

    #[test]
    fn enlarge_k_zero_is_identity() {
        let cur = P::new(10.0, 12.0);
        assert_eq!(enlarge(cur, P::new(8.0, 8.0), 30.0, 0.0), cur);
        let c = P::new(0.0, 0.0);
        let q = enlarge(P::new(10.0, 0.0), c, 40.0, 0.24);
        assert!(q.x < 10.0 && q.x > 0.0);
    }

    #[test]
    fn gpupixel_primitives_identity_at_zero() {
        let cur = P::new(3.0, 4.0);
        assert_eq!(
            curve_warp(cur, P::new(0.0, 0.0), P::new(10.0, 0.0), 0.0),
            cur
        );
        assert_eq!(enlarge_gpupixel(cur, P::new(0.0, 0.0), 10.0, 0.0), cur);
    }
}
