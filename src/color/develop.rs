//! 内嵌 Camera Raw 冲印设置（XMP `crs:`）的读取与近似再应用（可选功能，默认关闭）。
//!
//! Lightroom / ACR 导出的 JPEG 会在 XMP 里记录冲印参数，并标注 `crs:AlreadyApplied="True"`
//! （像素已包含这些调整）。像素蛋糕的一个示例项目导出 X04 时把这些设置又应用了一次：
//! 色温 4550、高光 −44、黑色 −14、纹理 +7、清晰度 +7、自然饱和度 +13、橙色色相 −6 / 饱和度 −20 / 明亮度 +9、
//! 红色明亮度 −20，导出图上背景墙偏红、奶油色礼服与金饰变暗变淡、肤色变淡偏粉、中等反差纹理增强约 7%，
//! 与这些参数逐项吻合。但重新导出的 X04 以及 5 张同样带 crs 设置的批量样张，非皮肤区都与原图一致——
//! 这是那个项目的设置，**不是「奶油肌」预设的行为**，所以预设默认不开启（`--embedded-develop` 开启）。
//! 详见 `doc/test_report_x04.md` §6 与 `doc/test_report_batch.md`。
//!
//! 这里用参数模型近似这一步（不是 Adobe 的算法）：
//! - 颜色模型（色调、暗部偏色、HSL、自然饱和度 / 饱和度），常数由 `tools/develop_fit.py`
//!   在 X04 非皮肤区域 + 新娘皮肤上拟合；模型只依赖颜色，烘焙成 65³ 3D LUT，作用在保边基底上
//!   （细节原样加回，对应 ACR 高光 / 阴影的局部色调映射）；
//! - 纹理 / 清晰度：亮度的带通增强（σ ≈ 1…20 px @ 5472 长边），只作用于中等反差细节，强边缘不动。
//!
//! 已拟合：高光 / 阴影 / 白色 / 黑色、HSL 色相与饱和度、自然饱和度、饱和度、纹理 + 清晰度、色温（效果很弱：
//! 像素蛋糕似乎把记录的白平衡当作基准，只留下约 1/10 的偏移）。按常见定义实现但**未经样张验证**：曝光、对比度。
//! HSL 明亮度在 X04 上没有可测的效果（红 −20 与橙 +9 在肤色上相互抵消），不模拟；其余设置（去朦胧、校准、
//! 色调曲线、分离色调 / 颜色分级、暗角、颗粒、局部调整等）也不模拟，读取时列入 [`DevelopSettings::ignored`]。

use crate::buffer::{GrayF32, ImgF32};
use crate::color::lab::{f_lab, f_lab_inv, lab_to_rgb, linear_to_srgb, rgb_to_lab, srgb_to_linear};
use crate::color::lut3d::Lut3D;
use crate::skin::guided::{fast_gaussian, guided_filter};
use rayon::prelude::*;
use std::collections::HashMap;

/// HSL 八个色相通道的名称（XMP 属性后缀）与 HSV 色相中心（度，Adobe 的划分）。
pub const HSL_NAMES: [&str; 8] = [
    "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta",
];
const HSL_CENTERS: [f32; 8] = [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0];

/// 从 XMP 读到的冲印参数（滑块原始值）。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DevelopSettings {
    /// `crs:AlreadyApplied="True"`：像素已包含这些调整
    pub already_applied: bool,
    /// 色温：开尔文（raw 来源）或 −100..100 的相对值（JPEG 来源）；None 表示未记录
    pub temperature: Option<f32>,
    pub tint: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub texture: f32,
    pub clarity: f32,
    pub vibrance: f32,
    pub saturation: f32,
    /// HSL：按 [`HSL_NAMES`] 顺序
    pub hsl_hue: [f32; 8],
    pub hsl_sat: [f32; 8],
    pub hsl_lum: [f32; 8],
    /// 读到但不模拟的非默认设置（`名称=值`）
    pub ignored: Vec<String>,
}

// ---- 拟合常数（tools/develop_fit.py，X04：非皮肤 60k 样本 + 新娘皮肤，皮肤权重 0.35）----
const G_HIGH: f32 = 2.5194;
const C_HIGH: f32 = 1.0155;
const W_HIGH: f32 = 0.3700;
const G_BLACK: f32 = 9.0672;
const E_BLACK: f32 = 0.9;
const K_TONE_CHROMA: f32 = 0.5242;
const G_TEMP: f32 = 0.4855;
const G_BLACK_A: f32 = -14.329;
const G_BLACK_B: f32 = -15.461;
const G_HUE: f32 = 33.444;
const G_SAT: f32 = 0.4646;
/// HSL 饱和度调整随 HSV 饱和度的权重：`bump(s) = (s/S_PEAK)^Q·exp(Q·(1 − s/S_PEAK))`，在 s = S_PEAK 处为 1。
/// X04 上像素蛋糕把肤色（s ≈ 0.3）降饱和最多，同属"橙色"的低饱和背景墙反而更饱和、金饰几乎不变
const S_PEAK: f32 = 0.3164;
const Q_SAT: f32 = 6.0;
const HSL_WIDTH: f32 = 0.5890;
const G_VIB: f32 = 0.6347;
const VIB_SKIN: f32 = 1.0;
/// 对比度 ±100 时中间调两侧的最大 L 变化约 ±7.5（常见 S 曲线，未经样张验证）
const K_CONTRAST: f32 = 40.0;
/// 纹理 + 清晰度的带通增益：X04（+7 / +7）实测中等反差细节约 +7%
const K_LOCAL_CONTRAST: f32 = 0.0055;
/// 带通尺度按图像长边相对 X04（5472 px）缩放
const REF_LONG_SIDE: f32 = 5472.0;
/// 颜色模型的保边基底：导向滤波半径（px @ 5472 长边）与方差门限（局部标准差 ≈ 0.03 ≈ 3 L 以下视为纹理）
const BASE_RADIUS: f32 = 16.0;
const BASE_EPS: f32 = 0.03 * 0.03;

impl DevelopSettings {
    /// 解析文件字节（JPEG 走 APP1 段，其他格式直接搜索 XMP 包）。
    /// 只有 `crs:HasSettings="True"` 时返回 Some。
    pub fn from_file_bytes(bytes: &[u8]) -> Option<Self> {
        Self::from_xmp(&extract_xmp(bytes)?)
    }

    pub fn from_path(path: &std::path::Path) -> std::io::Result<Option<Self>> {
        Ok(Self::from_file_bytes(&std::fs::read(path)?))
    }

    /// 解析 XMP 文本中的 `crs:` 属性（属性形式 `crs:Name="v"` 与元素形式 `<crs:Name>v</crs:Name>`）。
    pub fn from_xmp(xmp: &str) -> Option<Self> {
        let kv = crs_properties(xmp);
        if kv.get("HasSettings").map(|v| v.as_str()) != Some("True") {
            return None;
        }
        let num = |k: &str| kv.get(k).and_then(|v| v.trim().parse::<f32>().ok());
        let f = |k: &str| num(k).unwrap_or(0.0);
        let f2 = |k: &str, old: &str| num(k).or_else(|| num(old)).unwrap_or(0.0);
        let mut s = Self {
            already_applied: kv.get("AlreadyApplied").map(|v| v.as_str()) == Some("True"),
            temperature: num("Temperature"),
            tint: f("Tint"),
            exposure: f2("Exposure2012", "Exposure"),
            contrast: f2("Contrast2012", "Contrast"),
            highlights: f("Highlights2012"),
            shadows: f("Shadows2012"),
            whites: f("Whites2012"),
            blacks: f("Blacks2012"),
            texture: f("Texture"),
            clarity: f2("Clarity2012", "Clarity"),
            vibrance: f("Vibrance"),
            saturation: f("Saturation"),
            ..Default::default()
        };
        for (i, n) in HSL_NAMES.iter().enumerate() {
            s.hsl_hue[i] = f(&format!("HueAdjustment{n}"));
            s.hsl_sat[i] = f(&format!("SaturationAdjustment{n}"));
            s.hsl_lum[i] = f(&format!("LuminanceAdjustment{n}"));
        }
        const IGNORED_NUMERIC: [&str; 24] = [
            "Dehaze",
            "GrainAmount",
            "PostCropVignetteAmount",
            "VignetteAmount",
            "LuminanceSmoothing",
            "ParametricShadows",
            "ParametricDarks",
            "ParametricLights",
            "ParametricHighlights",
            "SplitToningShadowSaturation",
            "SplitToningHighlightSaturation",
            "ColorGradeShadowSat",
            "ColorGradeMidtoneSat",
            "ColorGradeHighlightSat",
            "ColorGradeGlobalSat",
            "ShadowTint",
            "RedHue",
            "RedSaturation",
            "GreenHue",
            "GreenSaturation",
            "BlueHue",
            "BlueSaturation",
            "DefringePurpleAmount",
            "DefringeGreenAmount",
        ];
        for k in IGNORED_NUMERIC {
            if let Some(v) = num(k) {
                if v != 0.0 {
                    s.ignored.push(format!("{k}={}", kv[k]));
                }
            }
        }
        for (i, n) in HSL_NAMES.iter().enumerate() {
            if s.hsl_lum[i] != 0.0 {
                s.ignored
                    .push(format!("LuminanceAdjustment{n}={}", s.hsl_lum[i]));
            }
        }
        if kv.get("ConvertToGrayscale").map(|v| v.as_str()) == Some("True") {
            s.ignored.push("ConvertToGrayscale=True".into());
        }
        if let Some(v) = kv.get("ToneCurveName2012") {
            if v != "Linear" {
                s.ignored.push(format!("ToneCurveName2012={v}"));
            }
        }
        if xmp.contains("<crs:GradientBasedCorrections")
            || xmp.contains("<crs:PaintBasedCorrections")
            || xmp.contains("<crs:MaskGroupBasedCorrections")
        {
            s.ignored.push("local corrections".into());
        }
        Some(s)
    }

    /// 相对色温换算为模型使用的开尔文值（JPEG 来源的 −100..100 按 ±2750 K 近似）。
    fn temperature_kelvin(&self) -> f32 {
        match self.temperature {
            Some(t) if t.abs() <= 100.0 => 5500.0 + 27.5 * t,
            Some(t) => t,
            None => 5500.0,
        }
    }

    fn has_color(&self) -> bool {
        self.exposure != 0.0
            || self.contrast != 0.0
            || self.highlights != 0.0
            || self.shadows != 0.0
            || self.whites != 0.0
            || self.blacks != 0.0
            || self.vibrance != 0.0
            || self.saturation != 0.0
            || (self.temperature_kelvin() - 5500.0).abs() > 1.0
            || self.hsl_hue.iter().chain(&self.hsl_sat).any(|v| *v != 0.0)
    }

    fn local_contrast_gain(&self) -> f32 {
        (K_LOCAL_CONTRAST * (self.texture + self.clarity)).clamp(-0.5, 0.5)
    }

    /// 是否不产生任何变化。
    pub fn is_noop(&self) -> bool {
        !self.has_color() && self.local_contrast_gain() == 0.0
    }

    /// 一行摘要（CLI 输出用），只列非零项。
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(t) = self.temperature {
            parts.push(format!("temp {t}"));
        }
        for (name, v) in [
            ("tint", self.tint),
            ("exposure", self.exposure),
            ("contrast", self.contrast),
            ("highlights", self.highlights),
            ("shadows", self.shadows),
            ("whites", self.whites),
            ("blacks", self.blacks),
            ("texture", self.texture),
            ("clarity", self.clarity),
            ("vibrance", self.vibrance),
            ("saturation", self.saturation),
        ] {
            if v != 0.0 {
                parts.push(format!("{name} {v:+}"));
            }
        }
        for (i, n) in HSL_NAMES.iter().enumerate() {
            let (h, s, l) = (self.hsl_hue[i], self.hsl_sat[i], self.hsl_lum[i]);
            if h != 0.0 || s != 0.0 || l != 0.0 {
                parts.push(format!("{} h{h:+}/s{s:+}/l{l:+}", n.to_ascii_lowercase()));
            }
        }
        let mut out = parts.join(", ");
        if self.already_applied {
            out.push_str(" (AlreadyApplied)");
        }
        if !self.ignored.is_empty() {
            out.push_str(&format!("; not emulated: {}", self.ignored.join(", ")));
        }
        out
    }

    /// 逐像素颜色模型（sRGB 0..1 → sRGB 0..1）。
    pub fn map_color(&self, p: [f32; 3]) -> [f32; 3] {
        let (h, s) = rgb_to_hue_sat(p);
        let mut rgb = p;
        if self.exposure != 0.0 {
            let g = 2f32.powf(self.exposure);
            for c in rgb.iter_mut() {
                *c = linear_to_srgb((srgb_to_linear(*c) * g).min(1.0));
            }
        }
        let [l, a, b] = rgb_to_lab(rgb);
        let x = l / 100.0;
        let dark = 1.0 - smoothstep(0.0, E_BLACK, x);
        let dl = (G_HIGH * self.highlights * (-((x - C_HIGH) / W_HIGH).powi(2)).exp()
            + G_BLACK * self.blacks * dark
            + G_HIGH * self.shadows * (-((x - 0.25) / 0.18).powi(2)).exp()
            + G_HIGH * self.whites * smoothstep(0.6, 1.0, x))
            / 100.0
            + self.contrast / 100.0 * K_CONTRAST * (x - 0.5) * 4.0 * x * (1.0 - x);
        let tone_cf = 1.0 + K_TONE_CHROMA * dl / l.max(5.0);
        let a2 = a * tone_cf + G_BLACK_A * self.blacks / 100.0 * dark;
        let b2 = b * tone_cf
            + G_TEMP * (self.temperature_kelvin() - 5500.0) / 5500.0 * x * 10.0
            + G_BLACK_B * self.blacks / 100.0 * dark;
        let w = hue_weights(h, HSL_WIDTH);
        let dot = |v: &[f32; 8]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>() / 100.0;
        let dh = G_HUE * dot(&self.hsl_hue);
        let r = s.max(1e-4) / S_PEAK;
        let sat_f = 1.0 + G_SAT * dot(&self.hsl_sat) * r.powf(Q_SAT) * (Q_SAT * (1.0 - r)).exp();
        let vib_f = 1.0 + G_VIB * self.vibrance / 100.0 * (1.0 - s) * (1.0 - VIB_SKIN * w[1]);
        let satg = 1.0 + self.saturation / 100.0;
        let c2 = a2.hypot(b2) * (sat_f * vib_f * satg).max(0.0);
        let h2 = b2.atan2(a2) + dh.to_radians();
        lab_to_rgb([l + dl, c2 * h2.cos(), c2 * h2.sin()])
    }

    /// 把颜色模型烘焙为 3D LUT。
    pub fn bake_lut(&self, n: usize) -> Lut3D {
        let mut lut = Lut3D::identity(n);
        lut.data
            .par_iter_mut()
            .for_each(|v| *v = self.map_color(*v));
        lut
    }

    /// 应用到整幅图：颜色模型（LUT）→ 纹理 / 清晰度带通增强。
    ///
    /// 颜色模型作用在保边基底上、细节原样加回（`out = img + LUT(base) − base`）：ACR 的高光 / 阴影
    /// 是局部色调映射，不压缩纹理——X04 上像素蛋糕在高光压暗处的细节带通比仍 ≥ 1，逐像素套曲线
    /// 则会把 L 55–75 的细纹理压掉约 5%。
    pub fn apply(&self, img: &mut ImgF32) {
        let s = (img.w.max(img.h) as f32 / REF_LONG_SIDE).max(0.2);
        if self.has_color() {
            let lut = self.bake_lut(65);
            let r = ((BASE_RADIUS * s).round() as usize).max(1);
            let base = edge_aware_base(img, r, BASE_EPS);
            img.data.par_iter_mut().zip(&base.data).for_each(|(p, b)| {
                let m = lut.lookup(*b);
                for c in 0..3 {
                    p[c] = (p[c] + m[c] - b[c]).clamp(0.0, 1.0);
                }
            });
        }
        let k = self.local_contrast_gain();
        if k != 0.0 {
            let t = self.texture.abs();
            let c = self.clarity.abs();
            let share = if t + c > 0.0 { c / (t + c) } else { 0.5 };
            local_contrast(img, k, 4.0 + 32.0 * share, s);
        }
    }

    /// 8 位图像版本（CLI：解码后、人脸检测前调用）。
    pub fn apply_rgb8(&self, img: &mut image::RgbImage) {
        if self.is_noop() {
            return;
        }
        let mut f = ImgF32::from_rgb8(img);
        self.apply(&mut f);
        *img = f.to_rgb8();
    }
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// HSV 色相（度）与饱和度。
#[inline]
fn rgb_to_hue_sat(p: [f32; 3]) -> (f32, f32) {
    let [r, g, b] = [
        p[0].clamp(0.0, 1.0),
        p[1].clamp(0.0, 1.0),
        p[2].clamp(0.0, 1.0),
    ];
    let mx = r.max(g).max(b);
    let d = mx - r.min(g).min(b);
    let s = if mx > 0.0 { d / mx } else { 0.0 };
    let h = if d <= 1e-7 {
        0.0
    } else if mx == r {
        (60.0 * (g - b) / d).rem_euclid(360.0)
    } else if mx == g {
        120.0 + 60.0 * (b - r) / d
    } else {
        240.0 + 60.0 * (r - g) / d
    };
    (h, s)
}

/// 八个色相通道的余弦窗权重（归一化；窗宽 = 到相邻中心的距离 × width）。
fn hue_weights(h: f32, width: f32) -> [f32; 8] {
    let mut w = [0.0f32; 8];
    let mut sum = 0.0;
    for i in 0..8 {
        let c = HSL_CENTERS[i];
        let prev = if i == 0 {
            HSL_CENTERS[7] - 360.0
        } else {
            HSL_CENTERS[i - 1]
        };
        let next = if i == 7 {
            HSL_CENTERS[0] + 360.0
        } else {
            HSL_CENTERS[i + 1]
        };
        let d = (h - c + 180.0).rem_euclid(360.0) - 180.0;
        let span = if d < 0.0 { c - prev } else { next - c } * width;
        let t = (d.abs() / span).min(1.0);
        w[i] = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
        sum += w[i];
    }
    let n = sum.max(1e-6);
    w.iter_mut().for_each(|v| *v /= n);
    w
}

/// 平面提取（逐像素函数）。
fn plane(img: &ImgF32, f: impl Fn(&[f32; 3]) -> f32 + Sync + Send) -> GrayF32 {
    GrayF32::from_vec(img.w, img.h, img.data.par_iter().map(f).collect())
}

#[inline]
fn luma(p: &[f32; 3]) -> f32 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}

/// 线性光亮度 Y（D65，0..1）。
#[inline]
fn linear_y(p: &[f32; 3]) -> f32 {
    0.2126 * srgb_to_linear(p[0].clamp(0.0, 1.0))
        + 0.7152 * srgb_to_linear(p[1].clamp(0.0, 1.0))
        + 0.0722 * srgb_to_linear(p[2].clamp(0.0, 1.0))
}

#[inline]
fn lstar(y: f32) -> f32 {
    116.0 * f_lab(y.max(0.0)) - 16.0
}

/// 保边基底：以亮度为引导逐通道导向滤波（半径 r，eps 为局部方差门限）。
fn edge_aware_base(img: &ImgF32, r: usize, eps: f32) -> ImgF32 {
    let guide = plane(img, luma);
    let sub = (r / 4).clamp(1, 4);
    let ch: Vec<GrayF32> = (0..3)
        .map(|c| guided_filter(&plane(img, move |p| p[c]), &guide, r, eps, sub))
        .collect();
    let mut out = ImgF32::new(img.w, img.h);
    out.data
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, o)| *o = [ch[0].data[i], ch[1].data[i], ch[2].data[i]]);
    out
}

/// 亮度带通增强：`L* += k·w(E)·(G(σa) − G(σb))`，σa = 1、σb = `sigma_b`（px @ 5472 长边，× s）；
/// E 为 1..4 px 带通的局部 RMS（L* 单位），只增强中等反差（E ≈ 1..3），平坦区与强边缘不动。
/// 新亮度通过线性光等比缩放 RGB 实现：L* 精确改变 dL，色度坐标不变（等量加到 sRGB 三通道
/// 会让饱和色的 L* 变化偏小）。
fn local_contrast(img: &mut ImgF32, k: f32, sigma_b: f32, s: f32) {
    let l = plane(img, |p| lstar(linear_y(p)));
    let ga = fast_gaussian(&l, s);
    let g4 = fast_gaussian(&l, 4.0 * s);
    let gb = fast_gaussian(&l, sigma_b * s);
    let band: Vec<f32> = ga
        .data
        .par_iter()
        .zip(&g4.data)
        .map(|(a, b)| (a - b) * (a - b))
        .collect();
    let energy = fast_gaussian(&GrayF32::from_vec(img.w, img.h, band), 8.0 * s);
    img.data
        .par_iter_mut()
        .zip(&l.data)
        .zip(&ga.data)
        .zip(&gb.data)
        .zip(&energy.data)
        .for_each(|((((p, lv), a), b), e)| {
            let e = e.max(0.0).sqrt();
            let w = smoothstep(0.3, 1.0, e) * (1.0 - smoothstep(3.0, 6.0, e));
            let dl = k * w * (a - b);
            if dl == 0.0 {
                return;
            }
            let y0 = f_lab_inv((lv + 16.0) / 116.0);
            if y0 <= 1e-5 {
                return;
            }
            let ratio = f_lab_inv((lv + dl + 16.0) / 116.0).max(0.0) / y0;
            for c in p.iter_mut() {
                *c = linear_to_srgb((srgb_to_linear(c.clamp(0.0, 1.0)) * ratio).min(1.0));
            }
        });
}

/// 取出 XMP 包文本：JPEG 读 APP1（标准 XMP + 扩展 XMP 按偏移拼接），其他格式直接搜索 `<x:xmpmeta`。
pub fn extract_xmp(bytes: &[u8]) -> Option<String> {
    if bytes.len() >= 4 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
        const STD: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
        const EXT: &[u8] = b"http://ns.adobe.com/xmp/extension/\0";
        let mut std_xmp: Option<&[u8]> = None;
        let mut ext: Vec<(&[u8], u32, &[u8])> = Vec::new();
        let mut i = 2usize;
        while i + 4 <= bytes.len() {
            if bytes[i] != 0xFF {
                break;
            }
            let marker = bytes[i + 1];
            if marker == 0xFF {
                i += 1;
                continue;
            }
            if marker == 0x01 || (0xD0..=0xD8).contains(&marker) {
                i += 2;
                continue;
            }
            if marker == 0xDA || marker == 0xD9 {
                break;
            }
            let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            if len < 2 || i + 2 + len > bytes.len() {
                break;
            }
            let payload = &bytes[i + 4..i + 2 + len];
            if marker == 0xE1 {
                if let Some(x) = payload.strip_prefix(STD) {
                    std_xmp.get_or_insert(x);
                } else if let Some(x) = payload.strip_prefix(EXT) {
                    if x.len() >= 40 {
                        let off = u32::from_be_bytes([x[36], x[37], x[38], x[39]]);
                        ext.push((&x[..32], off, &x[40..]));
                    }
                }
            }
            i += 2 + len;
        }
        let mut text = String::from_utf8_lossy(std_xmp?).into_owned();
        if let Some(&(guid, _, _)) = ext.first() {
            let mut parts: Vec<_> = ext.iter().filter(|e| e.0 == guid).collect();
            parts.sort_by_key(|e| e.1);
            let joined: Vec<u8> = parts.iter().flat_map(|e| e.2.iter().copied()).collect();
            text.push('\n');
            text.push_str(&String::from_utf8_lossy(&joined));
        }
        return Some(text);
    }
    let start = find(bytes, b"<x:xmpmeta")?;
    let end = find(&bytes[start..], b"</x:xmpmeta>").map_or(bytes.len(), |e| start + e + 12);
    Some(String::from_utf8_lossy(&bytes[start..end]).into_owned())
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// `crs:` 属性与简单元素 → 名称 / 值（先出现者优先）；`crs:Look` 结构体内部（配置文件自带参数）跳过。
fn crs_properties(xmp: &str) -> HashMap<String, String> {
    let mut text = xmp.to_string();
    while let Some(i) = text.find("<crs:Look>").or_else(|| text.find("<crs:Look ")) {
        match text[i..].find("</crs:Look>") {
            Some(j) => text.replace_range(i..i + j + "</crs:Look>".len(), ""),
            None => break,
        }
    }
    let b = text.as_bytes();
    let mut m = HashMap::new();
    let mut pos = 0usize;
    while let Some(k) = text[pos..].find("crs:") {
        let at = pos + k;
        let start = at + 4;
        let name_len = b[start..]
            .iter()
            .take_while(|c| c.is_ascii_alphanumeric())
            .count();
        let end = start + name_len;
        pos = end.max(at + 1);
        if name_len == 0 {
            continue;
        }
        let name = &text[start..end];
        let prev = if at > 0 { b[at - 1] } else { b' ' };
        let value = if prev == b'<' {
            if b.get(end) != Some(&b'>') {
                continue;
            }
            text[end + 1..]
                .find('<')
                .map(|e| text[end + 1..end + 1 + e].trim())
        } else if b.get(end) == Some(&b'=') {
            match b.get(end + 1) {
                Some(&q) if q == b'"' || q == b'\'' => text[end + 2..]
                    .find(q as char)
                    .map(|e| &text[end + 2..end + 2 + e]),
                _ => None,
            }
        } else {
            None
        };
        if let Some(v) = value.filter(|v| !v.is_empty()) {
            m.entry(name.to_string()).or_insert_with(|| v.to_string());
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    const XMP_X04: &str = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    crs:Version="15.0" crs:ProcessVersion="11.0" crs:WhiteBalance="Custom" crs:Temperature="4550" crs:Tint="+1"
    crs:Exposure2012="0.00" crs:Contrast2012="0" crs:Highlights2012="-44" crs:Shadows2012="+3" crs:Whites2012="0"
    crs:Blacks2012="-14" crs:Texture="+7" crs:Clarity2012="+7" crs:Dehaze="0" crs:Vibrance="+13" crs:Saturation="0"
    crs:HueAdjustmentOrange="-6" crs:SaturationAdjustmentOrange="-20" crs:LuminanceAdjustmentOrange="+9"
    crs:LuminanceAdjustmentRed="-20" crs:BlueSaturation="+25" crs:ToneCurveName2012="Linear"
    crs:HasSettings="True" crs:AlreadyApplied="True">
    <crs:Look><rdf:Description crs:Name="Adobe Color" crs:Amount="1"><crs:Parameters><rdf:Description
      crs:Highlights2012="+99" crs:ConvertToGrayscale="False"/></crs:Parameters></rdf:Description></crs:Look>
    <crs:Sharpness>40</crs:Sharpness>
    </rdf:Description></rdf:RDF></x:xmpmeta>"#;

    fn jpeg_with_app1(payloads: &[Vec<u8>]) -> Vec<u8> {
        let mut out = vec![0xFF, 0xD8];
        for p in payloads {
            out.extend_from_slice(&[0xFF, 0xE1]);
            out.extend_from_slice(&((p.len() + 2) as u16).to_be_bytes());
            out.extend_from_slice(p);
        }
        out.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x02, 0x12, 0x34, 0xFF, 0xD9]);
        out
    }

    #[test]
    fn parses_crs_settings_from_jpeg_app1() {
        let mut p = b"http://ns.adobe.com/xap/1.0/\0".to_vec();
        p.extend_from_slice(XMP_X04.as_bytes());
        let s = DevelopSettings::from_file_bytes(&jpeg_with_app1(&[p])).unwrap();
        assert!(s.already_applied);
        assert_eq!(s.temperature, Some(4550.0));
        assert_eq!(
            (s.tint, s.highlights, s.shadows, s.blacks),
            (1.0, -44.0, 3.0, -14.0)
        );
        assert_eq!((s.texture, s.clarity, s.vibrance), (7.0, 7.0, 13.0));
        assert_eq!(s.hsl_hue[1], -6.0);
        assert_eq!(s.hsl_sat[1], -20.0);
        assert_eq!(s.hsl_lum[1], 9.0);
        assert_eq!(s.hsl_lum[0], -20.0);
        // crs:Look 内部的参数不算；未模拟的非零设置被列出
        assert_eq!(
            s.ignored,
            vec![
                "BlueSaturation=+25".to_string(),
                "LuminanceAdjustmentRed=-20".to_string(),
                "LuminanceAdjustmentOrange=9".to_string()
            ]
        );
        assert!(!s.is_noop());
    }

    #[test]
    fn extended_xmp_and_element_form_and_plain_scan() {
        let std = b"http://ns.adobe.com/xap/1.0/\0<x:xmpmeta><rdf:Description crs:HasSettings=\"True\"/></x:xmpmeta>".to_vec();
        let body = b"<x:xmpmeta><crs:Exposure2012>+0.50</crs:Exposure2012><crs:Vibrance>+10</crs:Vibrance></x:xmpmeta>";
        let (a, b) = body.split_at(30);
        let ext = |off: usize, chunk: &[u8]| {
            let mut v = b"http://ns.adobe.com/xmp/extension/\0".to_vec();
            v.extend_from_slice(&[b'A'; 32]);
            v.extend_from_slice(&(body.len() as u32).to_be_bytes());
            v.extend_from_slice(&(off as u32).to_be_bytes());
            v.extend_from_slice(chunk);
            v
        };
        // 扩展段乱序也能按偏移拼回
        let jpg = jpeg_with_app1(&[std, ext(30, b), ext(0, a)]);
        let s = DevelopSettings::from_file_bytes(&jpg).unwrap();
        assert_eq!((s.exposure, s.vibrance), (0.5, 10.0));
        // 非 JPEG：直接搜索 XMP 包
        let mut tiff = b"II*\0garbage".to_vec();
        tiff.extend_from_slice(XMP_X04.as_bytes());
        assert_eq!(
            DevelopSettings::from_file_bytes(&tiff).unwrap().highlights,
            -44.0
        );
        // 没有 HasSettings=True：不返回设置
        let none = XMP_X04.replace("crs:HasSettings=\"True\"", "crs:HasSettings=\"False\"");
        assert!(DevelopSettings::from_xmp(&none).is_none());
        assert!(DevelopSettings::from_file_bytes(&[0xFF, 0xD8, 0xFF, 0xD9]).is_none());
    }

    #[test]
    fn neutral_settings_are_identity() {
        let s = DevelopSettings {
            temperature: Some(5500.0),
            ..Default::default()
        };
        assert!(s.is_noop());
        for &p in &[
            [0.0, 0.0, 0.0],
            [0.2, 0.5, 0.9],
            [0.95, 0.7, 0.6],
            [1.0, 1.0, 1.0],
        ] {
            let q = s.map_color(p);
            for c in 0..3 {
                assert!((q[c] - p[c]).abs() < 2e-3, "{p:?} -> {q:?}");
            }
        }
    }

    #[test]
    fn x04_model_directions() {
        let s = DevelopSettings::from_xmp(XMP_X04).unwrap();
        let lab = |p| rgb_to_lab(p);
        // 奶油色高光：变暗
        let cream = [0.93, 0.89, 0.80];
        assert!(lab(s.map_color(cream))[0] < lab(cream)[0] - 0.8);
        // 肤色：彩度降低（橙色饱和度 −20）
        let skin = [0.85, 0.66, 0.56];
        let (c0, c1) = (lab(skin), lab(s.map_color(skin)));
        assert!(c1[1].hypot(c1[2]) < c0[1].hypot(c0[2]) - 1.0);
        // 中灰：亮度几乎不动（色调调整集中在高光 / 最暗部）
        let grey = [0.4, 0.4, 0.4];
        assert!((lab(s.map_color(grey))[0] - lab(grey)[0]).abs() < 1.0);
    }

    #[test]
    fn texture_boosts_mid_contrast_detail_only() {
        // 左半：中等反差细纹（±0.03）；右半：强边缘（0.05 / 0.95 条纹）
        let (w, h) = (256, 128);
        let mut img = ImgF32::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 {
                    0.5 + 0.03 * if (x / 3 + y / 3) % 2 == 0 { 1.0 } else { -1.0 }
                } else if (x / 8) % 2 == 0 {
                    0.05
                } else {
                    0.95
                };
                img.data[y * w + x] = [v, v, v];
            }
        }
        let orig = img.clone();
        let s = DevelopSettings {
            texture: 7.0,
            clarity: 7.0,
            ..Default::default()
        };
        s.apply(&mut img);
        let amp = |im: &ImgF32, x0: usize, x1: usize| {
            let mut acc = 0.0f64;
            let mut n = 0.0;
            for y in 32..96 {
                for x in x0..x1 {
                    let d = im.data[y * w + x][0] as f64 - 0.5;
                    acc += d * d;
                    n += 1.0;
                }
            }
            (acc / n).sqrt()
        };
        let gain = amp(&img, 40, 88) / amp(&orig, 40, 88);
        assert!(gain > 1.03 && gain < 1.12, "mid-contrast gain {gain}");
        let edge = amp(&img, 160, 224) / amp(&orig, 160, 224);
        assert!((edge - 1.0).abs() < 0.01, "strong edges changed: {edge}");
    }
}
