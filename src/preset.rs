//! 预设文件（JSON）：把一整套参数（磨皮模式、奶油肌参数、按性别的美型强度、LUT 路径）打包为一键模板。

use crate::face::attribute::Gender;
use crate::pipeline::{RetouchParams, SmoothMode, StyleFilter, WhitenMode};
use crate::skin::cream::CreamParams;
use crate::warp::face_warp::{ReshapeStyle, WarpCoefficients, WarpParams};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// 按性别区分的美型参数。
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct GenderWarp {
    pub female: WarpParams,
    pub male: WarpParams,
    /// 未知性别（无属性模型）时使用
    pub unknown: WarpParams,
}

impl GenderWarp {
    pub fn pick(&self, g: Option<Gender>) -> WarpParams {
        match g {
            Some(Gender::Female) => self.female,
            Some(Gender::Male) => self.male,
            None => self.unknown,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Preset {
    pub name: String,
    pub description: String,
    pub smooth_mode: SmoothMode,
    pub smooth: f32,
    pub restrict_to_face: bool,
    pub brightness: f32,
    pub saturation: f32,
    pub apply_log_curve: bool,
    pub whiten: f32,
    /// 512 查找图路径（相对预设文件）
    pub whiten_lut: Option<String>,
    /// 奶油肌参数（女 / 男 / 未知）
    pub cream_female: CreamParams,
    pub cream_male: CreamParams,
    pub cream_unknown: CreamParams,
    /// AI 瑕疵祛除强度（0 关闭）
    pub ai_blemish: f32,
    /// 身体皮肤参数取自哪一套（默认 female）
    pub reshape: GenderWarp,
    pub reshape_style: ReshapeStyle,
    pub warp_coeffs: WarpCoefficients,
    pub attenuate_yaw: bool,
    /// 风格 LUT（512 PNG 或 .cube，相对预设文件）
    pub style: Option<String>,
    pub style_intensity: f32,
    /// 再应用输入文件内嵌的 Camera Raw 冲印设置（XMP crs；见 `color::develop`，默认关闭）
    pub embedded_develop: bool,
}

impl Default for Preset {
    fn default() -> Self {
        Self {
            name: "default".into(),
            description: String::new(),
            smooth_mode: SmoothMode::Faithful,
            smooth: 0.5,
            restrict_to_face: true,
            brightness: 1.1,
            saturation: 1.1,
            apply_log_curve: true,
            whiten: 0.0,
            whiten_lut: None,
            cream_female: CreamParams::default(),
            cream_male: CreamParams::male_default(),
            cream_unknown: CreamParams::default(),
            ai_blemish: 1.0,
            reshape: GenderWarp::default(),
            reshape_style: ReshapeStyle::Meihu,
            warp_coeffs: WarpCoefficients::default(),
            attenuate_yaw: true,
            style: None,
            style_intensity: 0.8,
            embedded_develop: false,
        }
    }
}

impl Preset {
    /// 内置预设的规范名（[`Preset::load_named`] 可识别）。
    pub fn builtin_names() -> &'static [&'static str] {
        &["cream_skin"]
    }

    /// 按名字取预设：内置名（`cream` / `cream_skin` / `creamskin` / `奶油肌`，不区分大小写）或 JSON 文件路径。
    pub fn load_named(name: &str) -> anyhow::Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "cream" | "cream_skin" | "creamskin" | "奶油肌" => Ok(Self::cream_skin()),
            _ => Self::load(Path::new(name)),
        }
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut p: Preset =
            serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let base = path.parent().unwrap_or(Path::new("."));
        if let Some(s) = &p.style {
            p.style = Some(resolve(base, s));
        }
        if let Some(s) = &p.whiten_lut {
            p.whiten_lut = Some(resolve(base, s));
        }
        Ok(p)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    /// 应用 JSON 覆盖（`{"cream_female": {"detail_smooth": 0.5}}` 形式，递归合并）。
    pub fn merge_json(&mut self, patch: &serde_json::Value) -> anyhow::Result<()> {
        let mut base = serde_json::to_value(&*self)?;
        merge(&mut base, patch);
        *self = serde_json::from_value(base)?;
        Ok(())
    }

    /// 转为运行参数（不含按性别 / 逐脸的部分，这些在 `RetouchParams::per_face_*` 中）。
    pub fn to_params(&self) -> anyhow::Result<RetouchParams> {
        let mut p = RetouchParams {
            smooth: self.smooth,
            smooth_mode: self.smooth_mode,
            restrict_to_face: self.restrict_to_face,
            brightness: self.brightness,
            saturation: self.saturation,
            apply_log_curve: self.apply_log_curve,
            whiten: self.whiten,
            reshape_style: self.reshape_style,
            warp_coeffs: self.warp_coeffs,
            attenuate_yaw: self.attenuate_yaw,
            style_intensity: self.style_intensity,
            embedded_develop: self.embedded_develop,
            cream: self.cream_unknown.clone(),
            cream_female: Some(self.cream_female.clone()),
            cream_male: Some(self.cream_male.clone()),
            ai_blemish: self.ai_blemish,
            gender_warp: Some(self.reshape.clone()),
            thin_face: self.reshape.unknown.thin_face,
            big_eye: self.reshape.unknown.big_eye,
            thin_nose: self.reshape.unknown.thin_nose,
            chin_lift: self.reshape.unknown.chin_lift,
            face_narrow: self.reshape.unknown.face_narrow,
            ..Default::default()
        };
        if let Some(path) = &self.whiten_lut {
            p.whiten_mode =
                WhitenMode::Lookup512(Arc::new(crate::color::lookup512::load_lookup512(path)?));
        }
        if let Some(path) = &self.style {
            let ext = Path::new(path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            p.style = Some(if ext == "cube" {
                StyleFilter::Cube(Arc::new(crate::color::lut3d::Lut3D::load(path)?))
            } else {
                StyleFilter::Lookup512(Arc::new(crate::color::lookup512::load_lookup512(path)?))
            });
        }
        Ok(p)
    }

    /// 内置"奶油肌"预设（像素蛋糕同名预设的逆向拟合）。
    pub fn cream_skin() -> Self {
        Self {
            name: "奶油肌".into(),
            description: "对标像素蛋糕「奶油肌」：保纹理磨皮 + 匀肤 + 奶油色调 + 祛瑕疵 + 轻度瘦脸缩下巴；背景与头发、五官逐位不变".into(),
            smooth_mode: SmoothMode::Cream,
            smooth: 1.0,
            restrict_to_face: true,
            brightness: 1.0,
            saturation: 1.0,
            apply_log_curve: false,
            whiten: 0.0,
            whiten_lut: None,
            cream_female: CreamParams::default(),
            cream_male: CreamParams::male_default(),
            cream_unknown: CreamParams { detail_smooth: 0.18, ..CreamParams::default() },
            ai_blemish: 1.0,
            reshape: GenderWarp {
                female: WarpParams { thin_face: 0.15, big_eye: 0.0, thin_nose: 0.25, chin_lift: 0.90, face_narrow: 0.030 },
                male: WarpParams { thin_face: 0.16, big_eye: 0.0, thin_nose: 0.05, chin_lift: 0.12, face_narrow: 0.007 },
                unknown: WarpParams { thin_face: 0.14, big_eye: 0.0, thin_nose: 0.15, chin_lift: 0.50, face_narrow: 0.018 },
            },
            reshape_style: ReshapeStyle::Meihu,
            warp_coeffs: WarpCoefficients { thin_face_profile: Some([0.25, 0.25, 0.22]), ..Default::default() },
            attenuate_yaw: true,
            style: None,
            style_intensity: 0.0,
            embedded_develop: false,
        }
    }
}

fn resolve(base: &Path, s: &str) -> String {
    let p = Path::new(s);
    if p.is_absolute() || p.exists() {
        s.to_string()
    } else {
        base.join(p).to_string_lossy().to_string()
    }
}

fn merge(base: &mut serde_json::Value, patch: &serde_json::Value) {
    match (base, patch) {
        (serde_json::Value::Object(b), serde_json::Value::Object(p)) => {
            for (k, v) in p {
                match b.get_mut(k) {
                    Some(bv) if bv.is_object() && v.is_object() => merge(bv, v),
                    _ => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (b, p) => *b = p.clone(),
    }
}

/// 把 `a.b.c=1.5` 形式的覆盖转为 JSON。
pub fn kv_to_json(spec: &str) -> anyhow::Result<serde_json::Value> {
    let (k, v) = spec
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("expected key=value, got {spec}"))?;
    let val: serde_json::Value =
        serde_json::from_str(v).unwrap_or_else(|_| serde_json::Value::String(v.to_string()));
    let mut out = val;
    for part in k.split('.').rev() {
        out = serde_json::json!({ part: out });
    }
    Ok(out)
}

#[allow(dead_code)]
fn _preset_path(p: &Preset) -> PathBuf {
    PathBuf::from(&p.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_roundtrip_and_merge() {
        let p = Preset::cream_skin();
        let text = serde_json::to_string(&p).unwrap();
        let q: Preset = serde_json::from_str(&text).unwrap();
        assert_eq!(q.smooth_mode, SmoothMode::Cream);
        let mut r = q.clone();
        r.merge_json(&kv_to_json("cream_female.detail_smooth=0.9").unwrap())
            .unwrap();
        assert!((r.cream_female.detail_smooth - 0.9).abs() < 1e-6);
        assert!((r.cream_female.chroma_smooth - q.cream_female.chroma_smooth).abs() < 1e-6);
        let params = r.to_params().unwrap();
        assert_eq!(params.smooth_mode, SmoothMode::Cream);
        assert!(params.gender_warp.is_some());
    }
}
