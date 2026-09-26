//! 用户可调参数 → [`RetouchParams`]：命令行（`retouch apply / batch / bench`）与图形界面共用的唯一组装逻辑。
//!
//! - 选了预设（内置名或 JSON 文件）：以预设为准，再叠加 `sets`（`key.sub=value` 覆盖）与开关
//!   （`body_skin`、`ai_blemish`），手动滑块不生效；
//! - 没选预设：由手动参数（磨皮模式 / 强度、提亮饱和、美白、形变……）组装；
//! - 两种情况都叠加：内嵌冲印开关、美白查找图、形变系数 JSON、风格 LUT、遮罩 LUT。
//!
//! 字段默认值即命令行各选项的默认值；序列化为 JSON（snake_case）后可直接由界面传入。

use crate::buffer::GrayF32;
use crate::color::lookup512::load_lookup512;
use crate::color::lut3d::Lut3D;
use crate::pipeline::{MaskedLutOp, RetouchParams, SmoothMode, StyleFilter, WhitenMode};
use crate::preset::{kv_to_json, Preset};
use crate::skin::smooth_freqsep;
use crate::warp::face_warp::{ReshapeStyle, WarpCoefficients};
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// 用户可调参数（含义与 `retouch apply --help` 一致）。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ParamOptions {
    // ---- 预设 ----
    /// 预设：内置名（见 [`Preset::builtin_names`]）或 JSON 文件路径；None 表示用手动参数
    pub preset: Option<String>,
    /// 覆盖预设中的字段：`key.sub=value`（如 `cream_female.detail_smooth=0.5`）
    pub sets: Vec<String>,
    // ---- 手动参数（无预设时生效）----
    /// 磨皮强度 0..1
    pub smooth: f32,
    pub smooth_mode: SmoothMode,
    /// 只在人脸多边形内磨皮 / 美白
    pub restrict_to_face: bool,
    /// HSB 亮度 / 饱和度（1.0 = 不变）
    pub brightness: f32,
    pub saturation: f32,
    /// log 提亮曲线
    pub apply_log_curve: bool,
    /// 美白 0..1
    pub whiten: f32,
    pub thin_face: f32,
    pub big_eye: f32,
    pub thin_nose: f32,
    /// 缩下巴 / 提升下半脸 0..1
    pub chin_lift: f32,
    /// 整体收窄比例（如 0.03）
    pub face_narrow: f32,
    pub reshape_style: ReshapeStyle,
    /// 侧脸时衰减瘦脸 / 瘦鼻
    pub attenuate_yaw: bool,
    /// 方案 B 高反差半径（相对短边 1000 px）/ 锐化系数；方案 C 锐化系数
    pub freqsep_radius: f32,
    pub freqsep_sharpness: f32,
    pub gpupixel_sharpen: f32,
    // ---- 两种情况都生效 ----
    /// 奶油肌：处理身体皮肤（颈胸臂手）
    pub body_skin: bool,
    /// 奶油肌：AI 瑕疵祛除
    pub ai_blemish: bool,
    /// 再应用输入 JPEG 内嵌的 Camera Raw 设置：None 跟随预设（无预设时关闭），Some 强制开 / 关
    pub embedded_develop: Option<bool>,
    /// 美白查找图（512×512 PNG），不指定则用参数化曲线
    pub whiten_lut: Option<PathBuf>,
    /// 形变系数 JSON（字段见 `WarpCoefficients`）
    pub warp_coeffs: Option<PathBuf>,
    /// 风格 LUT：512×512 PNG 或 .cube
    pub style_lut: Option<PathBuf>,
    /// 风格 LUT 强度 0..1（有预设且没指定 `style_lut` 时沿用预设的强度）
    pub style_intensity: f32,
    /// 遮罩 LUT：`<cube>:<mask.png>[:invert][:strength]`
    pub masked_luts: Vec<String>,
}

impl Default for ParamOptions {
    fn default() -> Self {
        Self {
            preset: None,
            sets: Vec::new(),
            smooth: 0.5,
            smooth_mode: SmoothMode::Faithful,
            restrict_to_face: true,
            brightness: 1.1,
            saturation: 1.1,
            apply_log_curve: true,
            whiten: 0.0,
            thin_face: 0.0,
            big_eye: 0.0,
            thin_nose: 0.0,
            chin_lift: 0.0,
            face_narrow: 0.0,
            reshape_style: ReshapeStyle::Meihu,
            attenuate_yaw: true,
            freqsep_radius: smooth_freqsep::DEFAULT_RADIUS,
            freqsep_sharpness: smooth_freqsep::DEFAULT_SHARPNESS,
            gpupixel_sharpen: 0.0,
            body_skin: true,
            ai_blemish: true,
            embedded_develop: None,
            whiten_lut: None,
            warp_coeffs: None,
            style_lut: None,
            style_intensity: 0.8,
            masked_luts: Vec::new(),
        }
    }
}

impl ParamOptions {
    /// 组装修图参数（会读取所引用的预设 / LUT / 系数文件）。
    pub fn build(&self) -> anyhow::Result<RetouchParams> {
        let mut p = match &self.preset {
            Some(name) => {
                let mut preset = Preset::load_named(name)?;
                for s in &self.sets {
                    preset.merge_json(&kv_to_json(s)?)?;
                }
                let mut p = preset.to_params()?;
                if !self.body_skin {
                    p.body_skin = false;
                }
                if !self.ai_blemish {
                    p.ai_blemish = 0.0;
                }
                p
            }
            None => RetouchParams {
                smooth: self.smooth,
                smooth_mode: self.smooth_mode,
                restrict_to_face: self.restrict_to_face,
                body_skin: self.body_skin,
                ai_blemish: if self.ai_blemish { 1.0 } else { 0.0 },
                brightness: self.brightness,
                saturation: self.saturation,
                apply_log_curve: self.apply_log_curve,
                whiten: self.whiten,
                thin_face: self.thin_face,
                big_eye: self.big_eye,
                thin_nose: self.thin_nose,
                chin_lift: self.chin_lift,
                face_narrow: self.face_narrow,
                reshape_style: self.reshape_style,
                attenuate_yaw: self.attenuate_yaw,
                style_intensity: self.style_intensity,
                freqsep_radius: self.freqsep_radius,
                freqsep_sharpness: self.freqsep_sharpness,
                gpupixel_sharpen: self.gpupixel_sharpen,
                ..Default::default()
            },
        };
        if let Some(on) = self.embedded_develop {
            p.embedded_develop = on;
        }
        if let Some(path) = &self.whiten_lut {
            p.whiten_mode = WhitenMode::Lookup512(Arc::new(load_lookup512(path)?));
        }
        if let Some(path) = &self.warp_coeffs {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read {}", path.display()))?;
            p.warp_coeffs = serde_json::from_str::<WarpCoefficients>(&text)
                .with_context(|| format!("parse {}", path.display()))?;
        }
        if let Some(path) = &self.style_lut {
            p.style = Some(load_style(path)?);
            if self.preset.is_some() {
                p.style_intensity = self.style_intensity;
            }
        }
        for spec in &self.masked_luts {
            p.masked_ops.push(parse_masked_lut(spec)?);
        }
        Ok(p)
    }
}

/// 读取风格 LUT：`.cube` 为 3D LUT，其他扩展名按 512×512 查找图读取。
pub fn load_style(path: &Path) -> anyhow::Result<StyleFilter> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "cube" {
        Ok(StyleFilter::Cube(Arc::new(Lut3D::load(path)?)))
    } else {
        Ok(StyleFilter::Lookup512(Arc::new(load_lookup512(path)?)))
    }
}

/// 解析遮罩 LUT：`<cube>:<mask.png>[:invert][:strength]`。
pub fn parse_masked_lut(spec: &str) -> anyhow::Result<MaskedLutOp> {
    let parts: Vec<&str> = spec.split(':').collect();
    anyhow::ensure!(
        parts.len() >= 2,
        "masked-lut spec must be <cube>:<mask.png>[:invert][:strength]"
    );
    let lut = Lut3D::load(parts[0])?;
    let mask_img = image::open(parts[1])
        .with_context(|| format!("open {}", parts[1]))?
        .to_luma8();
    let mut invert = false;
    let mut strength = 1.0;
    for p in &parts[2..] {
        if *p == "invert" {
            invert = true;
        } else if let Ok(v) = p.parse::<f32>() {
            strength = v;
        }
    }
    Ok(MaskedLutOp {
        lut: Arc::new(lut),
        mask: Arc::new(GrayF32::from_luma8(&mask_img)),
        invert,
        strength,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_defaults_match_cli_defaults() {
        let p = ParamOptions::default().build().unwrap();
        assert_eq!(p.smooth_mode, SmoothMode::Faithful);
        assert!((p.smooth - 0.5).abs() < 1e-6);
        assert!((p.brightness - 1.1).abs() < 1e-6 && (p.saturation - 1.1).abs() < 1e-6);
        assert!(p.apply_log_curve && p.restrict_to_face && p.attenuate_yaw && p.body_skin);
        assert!((p.ai_blemish - 1.0).abs() < 1e-6);
        assert!(!p.embedded_develop);
        assert!(p.style.is_none() && p.masked_ops.is_empty());
        assert!((p.style_intensity - 0.8).abs() < 1e-6);
    }

    #[test]
    fn preset_with_overrides_and_switches() {
        let opts = ParamOptions {
            preset: Some("奶油肌".into()),
            sets: vec!["cream_female.detail_smooth=0.33".into()],
            body_skin: false,
            ai_blemish: false,
            // 手动参数在有预设时不生效
            smooth: 0.1,
            thin_face: 0.9,
            ..Default::default()
        };
        let p = opts.build().unwrap();
        assert_eq!(p.smooth_mode, SmoothMode::Cream);
        assert!(!p.body_skin);
        assert_eq!(p.ai_blemish, 0.0);
        let female = p.cream_female.as_ref().unwrap();
        assert!((female.detail_smooth - 0.33).abs() < 1e-6);
        assert!((p.smooth - Preset::cream_skin().smooth).abs() < 1e-6);
        // 内嵌冲印：None 跟随预设（关），Some 强制
        assert!(!p.embedded_develop);
        let forced = ParamOptions {
            embedded_develop: Some(true),
            ..opts
        };
        assert!(forced.build().unwrap().embedded_develop);
    }

    #[test]
    fn bad_inputs_are_reported() {
        let missing = ParamOptions {
            style_lut: Some(PathBuf::from("no/such/file.cube")),
            ..Default::default()
        };
        assert!(missing.build().is_err());
        assert!(parse_masked_lut("only_one_part").is_err());
        let bad_set = ParamOptions {
            preset: Some("cream".into()),
            sets: vec!["no_equals_sign".into()],
            ..Default::default()
        };
        assert!(bad_set.build().is_err());
        let bad_preset = ParamOptions {
            preset: Some("no/such/preset.json".into()),
            ..Default::default()
        };
        assert!(bad_preset.build().is_err());
    }

    #[test]
    fn json_roundtrip_with_partial_fields() {
        // 界面只传部分字段，其余取默认
        let o: ParamOptions = serde_json::from_str(
            r#"{"smooth_mode":"freqsep","smooth":0.7,"embedded_develop":false}"#,
        )
        .unwrap();
        assert_eq!(o.smooth_mode, SmoothMode::FreqSep);
        assert!((o.smooth - 0.7).abs() < 1e-6);
        assert_eq!(o.embedded_develop, Some(false));
        assert!((o.brightness - 1.1).abs() < 1e-6);
        let back: ParamOptions = serde_json::from_str(&serde_json::to_string(&o).unwrap()).unwrap();
        assert_eq!(back, o);
    }
}
