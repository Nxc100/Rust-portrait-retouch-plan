//! Photoshop 风格混合模式（标量，0..1）。

#[inline]
pub fn multiply(a: f32, b: f32) -> f32 {
    a * b
}

#[inline]
pub fn screen(a: f32, b: f32) -> f32 {
    1.0 - (1.0 - a) * (1.0 - b)
}

/// overlay(base, blend)：base < 0.5 时 multiply，否则 screen。
#[inline]
pub fn overlay(base: f32, blend: f32) -> f32 {
    if base < 0.5 {
        2.0 * base * blend
    } else {
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    }
}

/// hardlight(base, blend) = overlay(blend, base)。
#[inline]
pub fn hardlight(base: f32, blend: f32) -> f32 {
    overlay(blend, base)
}

/// Pegtop 形式的柔光。
#[inline]
pub fn softlight(base: f32, blend: f32) -> f32 {
    (1.0 - 2.0 * blend) * base * base + 2.0 * blend * base
}

#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
pub fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_identities() {
        assert!((overlay(0.5, 0.5) - 0.5).abs() < 1e-6);
        assert!((hardlight(0.25, 0.5) - 0.25).abs() < 1e-6);
        assert!((screen(0.0, 0.3) - 0.3).abs() < 1e-6);
        assert!((multiply(1.0, 0.3) - 0.3).abs() < 1e-6);
    }
}
