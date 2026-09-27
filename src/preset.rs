//! 预设文件（JSON）：把一整套参数（磨皮模式、奶油肌参数、按性别的美型强度、预设调色、LUT 路径）打包为一键模板。

use crate::color::grade::{
    BlackPoint, Grade, GradeParams, SubjectTone, BUILTIN_PREFIX, WEDDING_DARK_INTERIOR,
};
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
    /// 预设调色（自适应黑点 + 全局 LUT + 人物主体调整，见 `color::grade`；LUT 为 `builtin:<名称>` 或相对预设文件的路径）
    pub grade: Option<GradeParams>,
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
            grade: None,
            style: None,
            style_intensity: 0.8,
            embedded_develop: false,
        }
    }
}

impl Preset {
    /// 内置预设的规范名（[`Preset::load_named`] 可识别）。
    pub fn builtin_names() -> &'static [&'static str] {
        &["cream_skin", "wedding_dark_interior"]
    }

    /// 按名字取预设：内置名（不区分大小写）或 JSON 文件路径。内置名：
    /// - 奶油肌：`cream` / `cream_skin` / `creamskin` / `奶油肌`；
    /// - 婚纱-深色内景：`wedding_dark_interior` / `wedding_dark` / `dark_interior` / `婚纱-深色内景` / `婚纱深色内景`。
    pub fn load_named(name: &str) -> anyhow::Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "cream" | "cream_skin" | "creamskin" | "奶油肌" => Ok(Self::cream_skin()),
            "wedding_dark_interior"
            | "wedding_dark"
            | "dark_interior"
            | "婚纱-深色内景"
            | "婚纱深色内景" => Ok(Self::wedding_dark_interior()),
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
        if let Some(g) = &mut p.grade {
            // 空的 lut 原样保留，由 `to_params` 报出明确的错误
            if !g.lut.trim().is_empty() && !g.lut.starts_with(BUILTIN_PREFIX) {
                g.lut = resolve(base, &g.lut);
            }
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
        if let Some(g) = &self.grade {
            p.grade = Some(Arc::new(Grade::load(g)?));
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
            cream_female: cream_preset_params(CreamParams::default()),
            cream_male: cream_preset_params(CreamParams::male_default()),
            cream_unknown: cream_preset_params(CreamParams { detail_smooth: 0.18, ..CreamParams::default() }),
            ai_blemish: 1.0,
            reshape: GenderWarp {
                female: WarpParams { thin_face: 0.15, big_eye: 0.0, thin_nose: 0.25, chin_lift: 0.90, face_narrow: 0.030 },
                male: WarpParams { thin_face: 0.16, big_eye: 0.0, thin_nose: 0.05, chin_lift: 0.12, face_narrow: 0.007 },
                unknown: WarpParams { thin_face: 0.14, big_eye: 0.0, thin_nose: 0.15, chin_lift: 0.50, face_narrow: 0.018 },
            },
            reshape_style: ReshapeStyle::Meihu,
            warp_coeffs: WarpCoefficients { thin_face_profile: Some([0.25, 0.25, 0.22]), ..Default::default() },
            attenuate_yaw: true,
            grade: None,
            style: None,
            style_intensity: 0.0,
            embedded_develop: false,
        }
    }

    /// 内置"婚纱-深色内景"预设（像素蛋糕同名预设的逆向拟合）：修图用奶油肌的算子，另加整体调色——
    /// 自适应黑点 + 全局 LUT（整体压暗、高光压低、蓝紫加深、红橙黄提亮）+ 人物主体提亮
    /// （`tools/grade_fit.py`，8 张参考导出，doc/analysis/wedding_dark_interior.md）。
    pub fn wedding_dark_interior() -> Self {
        let female = wedding_dark_female();
        Self {
            name: "婚纱-深色内景".into(),
            description: "对标像素蛋糕「婚纱-深色内景」：整体压暗、蓝紫加深、人物提亮的调色 + 保纹理磨皮 + 祛瑕疵 + 瘦脸".into(),
            cream_unknown: female.clone(),
            cream_female: female,
            cream_male: wedding_dark_male(),
            // 像素蛋糕这个预设的瘦脸更重、小脸更轻、眼睛放大（参数：瘦脸 0.25、小脸 0.6、眼睛整体大小 0.65）：
            // 女性下颌中下段比奶油肌多收约 1.5% 瞳距，眼角少收一半；下巴提得少（光流位移，doc/analysis/wedding_dark_interior.md）
            reshape: GenderWarp {
                female: WarpParams { thin_face: 0.32, big_eye: 0.10, thin_nose: 0.30, chin_lift: 0.62, face_narrow: 0.008 },
                male: WarpParams { thin_face: 0.16, big_eye: 0.0, thin_nose: 0.05, chin_lift: 0.12, face_narrow: 0.007 },
                unknown: WarpParams { thin_face: 0.24, big_eye: 0.05, thin_nose: 0.15, chin_lift: 0.40, face_narrow: 0.008 },
            },
            // 瘦脸的三对锚点：下颌中、下段加重
            warp_coeffs: WarpCoefficients { thin_face_profile: Some([0.18, 0.34, 0.36]), ..Default::default() },
            grade: Some(GradeParams {
                lut: format!("{BUILTIN_PREFIX}{WEDDING_DARK_INTERIOR}"),
                strength: 1.0,
                black_point: BlackPoint { percentile: 2.0, strength: 0.6, target: 0.02 },
                subject: SubjectTone { lightness: [1.45, 5.19, 6.53, 6.53, 5.67, 4.44], chroma: 0.097 },
            }),
            ..Self::cream_skin()
        }
    }
}

/// 「奶油肌」预设在奶油肌算子默认参数之上另开的两项（doc/test_report_final.md §4）：牙齿美白——像素蛋糕奶油肌
/// "牙齿美白 0.4"，按 1V3A3101、1V3A2954 两位露齿新娘的导出拟合为 0.3（红度、黄度与参考一致，亮度少约 3）；
/// 脸部遮罩按人归属——贴脸时对方的解析遮罩越界，那片会按两个人的参数各处理一遍（IMG_5785 新郎的细颗粒
/// 0.88 → 0.93，参考 0.94）。
fn cream_preset_params(base: CreamParams) -> CreamParams {
    CreamParams {
        teeth_whiten: 0.3,
        face_ownership: true,
        ..base
    }
}

/// 「婚纱-深色内景」的女性修图参数：奶油肌的算子，肤色常数在调色之后的颜色上拟合（调色先于修图）——
/// 脸 `tools/skin_tone_fit.py`、身体 `tools/body_tone_fit.py`，原图换成只调色不修图的输出、参考为像素蛋糕的导出。
/// 像素蛋糕在这个预设里把肤色拉得比奶油肌狠得多（黄度按 0.67 拉向 13），调色压暗后皮肤仍回到目标色附近。
/// 磨皮：该预设以"中性灰平整 0.8"为主、"质感保留 0.7"——压平 0.035–0.08 瞳距的斑驳起伏（`low_smooth`），
/// 毛孔一级的纹理多留（`detail_smooth` 比奶油肌轻）；"中性灰立体 0.5"把五官与轮廓的明暗加强（`stereo`）。
/// 按脸部皮肤各带通的中位幅度与大尺度结构的增益，在 7 张照片、13 张脸上与参考导出对齐
/// （doc/test_report_final.md §3）。另开身体色调外溢、牙齿美白与脸部遮罩的按人归属。
fn wedding_dark_female() -> CreamParams {
    CreamParams {
        detail_smooth: 0.33,
        fine_smooth: 0.15,
        low_smooth: 0.50,
        stereo: 0.22,
        body_tone_spill: 1.0,
        teeth_whiten: 1.0,
        face_ownership: true,
        midtone_lift: 6.04,
        highlight_suppress: 1.24,
        lift_per_b: -0.20,
        pull_a: 0.227,
        target_a: 4.79,
        pull_b: 0.671,
        target_b: 13.22,
        yellow_bright: -4.05,
        yellow_highlight: -1.80,
        yellow_shadow: -2.16,
        body_lift: 6.84,
        body_redness: -2.16,
        body_yellow: 0.0,
        body_pull_b: 0.666,
        body_target_b: 10.1,
        ..CreamParams::default()
    }
}

/// 「婚纱-深色内景」的男性修图参数（拟合方法同 [`wedding_dark_female`]；男性的细颗粒压得少、立体加强得多）。
fn wedding_dark_male() -> CreamParams {
    CreamParams {
        detail_smooth: 0.33,
        fine_smooth: 0.10,
        low_smooth: 0.50,
        stereo: 0.32,
        body_tone_spill: 1.0,
        teeth_whiten: 1.0,
        face_ownership: true,
        midtone_lift: 6.85,
        highlight_suppress: 1.21,
        lift_per_b: -0.243,
        pull_a: 0.252,
        target_a: 6.79,
        pull_b: 0.463,
        target_b: 18.27,
        yellow_bright: -7.91,
        yellow_highlight: 1.02,
        yellow_shadow: -4.35,
        body_lift: 3.54,
        body_redness: -2.58,
        body_yellow: 0.0,
        body_pull_b: 0.383,
        body_target_b: 7.3,
        ..CreamParams::male_default()
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

    #[test]
    fn builtin_presets_and_aliases() {
        for name in Preset::builtin_names() {
            assert!(Preset::load_named(name).is_ok(), "{name}");
        }
        for alias in [
            "wedding_dark_interior",
            "Wedding_Dark",
            "dark_interior",
            "婚纱-深色内景",
            "婚纱深色内景",
        ] {
            assert_eq!(
                Preset::load_named(alias).unwrap().name,
                "婚纱-深色内景",
                "{alias}"
            );
        }
        assert_eq!(Preset::load_named("奶油肌").unwrap().name, "奶油肌");
    }

    #[test]
    fn only_the_dark_interior_preset_grades() {
        // 奶油肌不带预设调色（输出与加入调色之前逐位相同）；婚纱-深色内景带内置 LUT 的调色
        assert!(Preset::cream_skin().grade.is_none());
        assert!(Preset::cream_skin().to_params().unwrap().grade.is_none());
        let dark = Preset::wedding_dark_interior();
        assert_eq!(dark.smooth_mode, SmoothMode::Cream);
        assert!(dark.grade.as_ref().unwrap().lut.starts_with(BUILTIN_PREFIX));
        assert!(dark.to_params().unwrap().grade.is_some());
    }

    #[test]
    fn dark_interior_survives_a_json_roundtrip() {
        // `retouch preset -o` 导出、再用 --preset 加载：内置 LUT 的引用原样保留
        let dir = std::env::temp_dir().join(format!("preset_dark_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dark.json");
        let p = Preset::wedding_dark_interior();
        p.save(&path).unwrap();
        let q = Preset::load(&path).unwrap();
        assert_eq!(q.grade, p.grade);
        assert_eq!(q.cream_female.pull_b, p.cream_female.pull_b);
        assert!(q.to_params().unwrap().grade.is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn grade_lut_paths_resolve_relative_to_the_preset_file() {
        let dir = std::env::temp_dir().join(format!("preset_grade_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("look.cube"),
            crate::color::lut3d::Lut3D::identity(2).to_cube_string(),
        )
        .unwrap();
        let path = dir.join("custom.json");
        std::fs::write(&path, r#"{"grade": {"lut": "look.cube"}}"#).unwrap();
        let p = Preset::load(&path).unwrap();
        let g = p.grade.as_ref().unwrap();
        assert!(Path::new(&g.lut).is_file(), "{}", g.lut);
        assert!(p.to_params().unwrap().grade.is_some());
        std::fs::remove_dir_all(&dir).ok();
    }
}
