//! 附录 B：UI 滑块（0–100）到内部参数的映射。

use crate::pipeline::RetouchParams;
use serde::{Deserialize, Serialize};

/// UI 滑块值（0–100）。
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct UiSliders {
    /// 磨皮，默认 50
    pub smooth: u8,
    /// 亮度，默认 50（⇔ brightness 1.1）
    pub brightness: u8,
    /// 饱和度，默认 50（⇔ saturation 1.1）
    pub saturation: u8,
    /// 美白，默认 0
    pub whiten: u8,
    pub thin_face: u8,
    pub big_eye: u8,
    pub thin_nose: u8,
    /// 滤镜强度，默认 80
    pub style_intensity: u8,
}

impl Default for UiSliders {
    fn default() -> Self {
        Self {
            smooth: 50,
            brightness: 50,
            saturation: 50,
            whiten: 0,
            thin_face: 0,
            big_eye: 0,
            thin_nose: 0,
            style_intensity: 80,
        }
    }
}

impl UiSliders {
    /// 把滑块值写入参数（保留 `base` 中的模式 / LUT 等非滑块字段）。
    pub fn apply_to(&self, base: &RetouchParams) -> RetouchParams {
        let f = |v: u8| (v.min(100) as f32) / 100.0;
        let mut p = base.clone();
        p.smooth = f(self.smooth);
        p.brightness = 1.0 + 0.2 * f(self.brightness);
        p.saturation = 1.0 + 0.2 * f(self.saturation);
        p.whiten = f(self.whiten);
        p.thin_face = f(self.thin_face);
        p.big_eye = f(self.big_eye);
        p.thin_nose = f(self.thin_nose);
        p.style_intensity = f(self.style_intensity);
        p
    }

    /// 从参数反推滑块值。
    pub fn from_params(p: &RetouchParams) -> Self {
        let g = |v: f32| (v.clamp(0.0, 1.0) * 100.0).round() as u8;
        Self {
            smooth: g(p.smooth),
            brightness: g((p.brightness - 1.0) / 0.2),
            saturation: g((p.saturation - 1.0) / 0.2),
            whiten: g(p.whiten),
            thin_face: g(p.thin_face),
            big_eye: g(p.big_eye),
            thin_nose: g(p.thin_nose),
            style_intensity: g(p.style_intensity),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_meihu() {
        let p = UiSliders::default().apply_to(&RetouchParams::default());
        assert!((p.smooth - 0.5).abs() < 1e-6);
        assert!((p.brightness - 1.1).abs() < 1e-6);
        assert!((p.saturation - 1.1).abs() < 1e-6);
        assert!((p.style_intensity - 0.8).abs() < 1e-6);
        let back = UiSliders::from_params(&p);
        assert_eq!(back.brightness, 50);
        assert_eq!(back.smooth, 50);
    }
}
