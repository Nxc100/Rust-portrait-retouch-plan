//! 预设调色：自适应黑点 → 全局 3D LUT → 人物主体的亮度 / 色度调整。
//!
//! 对应像素蛋糕"AI 风格"一类预设的整体调色——「婚纱-深色内景」的调色就是一组 AI 风格：整体压暗、
//! 高光压低、蓝紫变深、红橙黄提亮，另对人物主体单独提亮。参考导出图上的分析（doc/analysis/wedding_dark_interior.md）：
//! 同一张照片内调色是干净的逐像素颜色映射（ΔE ≈ 1、没有暗角与局部调整），但各张之间不是同一张 LUT——
//! 没有真正暗部的朦胧 / 高调画面（阴天海滩、雾中山水）被压得更暗，人物比同色的背景亮。于是分三步，
//! 常数都由 `tools/grade_fit.py` 在参考导出图上拟合：
//! 1. **自适应黑点**：`b = strength · max(0, P − target)`，P 为整图 min(R, G, B) 的分位（8 位直方图），
//!    软黑点 `x' = (x − b·s(x)) / (1 − b)`，s = smoothstep(0, 2b, x)（sRGB，逐通道）：亮部与硬黑点相同，
//!    暗部不截断（头发、深色衣物保留层次）。只作用在保边底图上、细节原样加回——像素蛋糕的这一步只移动局部明暗，
//!    不放大纹理（1V3A3150 黑点 0.31，逐像素拉伸会把脸上的纹理放大 1.44 倍，参考图上不变）。有纯黑的画面 b ≈ 0；
//! 2. **全局 3D LUT**（sRGB → sRGB，三线性）；
//! 3. **主体**：Lab 中 `L += s(L)`（分段线性，L = 0, 20, …, 100 处取值）、色度 × (1 + chroma)，
//!    在人像 alpha 内按 alpha 混合。与第 2 步一起预先烘焙成第二张 LUT，逐像素只查表。
//!
//! 预设以 [`GradeParams`] 描述（JSON）；LUT 写 `builtin:<名称>` 时取编译进程序的内置 LUT，否则为 .cube 路径。

use crate::buffer::{GrayF32, ImgF32};
use crate::color::lab::{lab_to_rgb, rgb_to_lab};
use crate::color::lut3d::Lut3D;
use crate::geom::P;
use crate::skin::guided::{edge_aware_base, fast_gaussian};
use crate::skin::morph::{components, dilate, erode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};

/// 黑点的保边底图：导向滤波半径（px @ 长边 5472，按图像长边缩放）与方差门限，与 `color::develop` 相同。
const BASE_RADIUS: f32 = 16.0;
const BASE_EPS: f32 = 0.03 * 0.03;
const REF_LONG_SIDE: f32 = 5472.0;

/// 人物主体权重的清理：开运算的半径（× 脸部尺度，切断发丝一样细的连接）、计算分辨率（脸部尺度约
/// SUBJECT_WORK_SCALE 像素）、没有人脸的连通块至少占画面的比例（没检测到脸的人仍算主体）
const SUBJECT_OPEN: f32 = 0.05;
const SUBJECT_WORK_SCALE: f32 = 16.0;
const SUBJECT_MIN_AREA: f32 = 0.01;

/// 预设里引用内置 LUT 的前缀：`builtin:<名称>`。
pub const BUILTIN_PREFIX: &str = "builtin:";

/// 「婚纱-深色内景」的全局 LUT（`tools/grade_fit.py` 拟合）。
pub const WEDDING_DARK_INTERIOR: &str = "wedding_dark_interior";

/// 主体亮度曲线的节点（L）。
pub const SUBJECT_KNOTS: [f32; 6] = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0];

/// 自适应黑点。
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct BlackPoint {
    /// 统计量：整图 min(R, G, B) 的分位（%）
    pub percentile: f32,
    /// 黑点 = strength · max(0, 统计量 − target)；0 关闭
    pub strength: f32,
    pub target: f32,
}

impl Default for BlackPoint {
    fn default() -> Self {
        Self {
            percentile: 2.0,
            strength: 0.0,
            target: 0.02,
        }
    }
}

impl BlackPoint {
    /// 这张图的黑点（sRGB 0..1）。
    pub fn of(&self, img: &ImgF32) -> f32 {
        if self.strength <= 0.0 {
            return 0.0;
        }
        let stat = min_channel_percentile(img, self.percentile);
        (self.strength * (stat - self.target).max(0.0)).min(0.9)
    }
}

/// 人物主体（人像 alpha 内）的调整。
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SubjectTone {
    /// [`SUBJECT_KNOTS`] 处的亮度增量（L），其间线性插值
    pub lightness: [f32; 6],
    /// 色度增益：Lab 色度 × (1 + chroma)
    pub chroma: f32,
}

impl SubjectTone {
    pub fn is_noop(&self) -> bool {
        self.chroma == 0.0 && self.lightness.iter().all(|v| *v == 0.0)
    }

    /// 亮度 `l` 处的增量。
    pub fn lightness_at(&self, l: f32) -> f32 {
        let k = &SUBJECT_KNOTS;
        let l = l.clamp(k[0], k[k.len() - 1]);
        let i = k
            .iter()
            .rposition(|v| *v <= l)
            .unwrap_or(0)
            .min(k.len() - 2);
        let t = (l - k[i]) / (k[i + 1] - k[i]);
        self.lightness[i] + (self.lightness[i + 1] - self.lightness[i]) * t
    }

    /// sRGB 0..1 → sRGB 0..1。
    pub fn map(&self, p: [f32; 3]) -> [f32; 3] {
        let [l, a, b] = rgb_to_lab(p);
        let g = 1.0 + self.chroma;
        lab_to_rgb([l + self.lightness_at(l), a * g, b * g])
    }
}

/// 预设里的调色参数。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GradeParams {
    /// 全局 3D LUT：`builtin:<名称>`（内置）或 .cube 路径（相对预设文件）
    pub lut: String,
    /// 与调色前的混合强度（0..1）
    pub strength: f32,
    pub black_point: BlackPoint,
    pub subject: SubjectTone,
}

impl Default for GradeParams {
    fn default() -> Self {
        Self {
            lut: String::new(),
            strength: 1.0,
            black_point: BlackPoint::default(),
            subject: SubjectTone::default(),
        }
    }
}

/// 可直接应用的调色：LUT 已加载，主体调整已烘焙成第二张 LUT。
#[derive(Clone, Debug)]
pub struct Grade {
    base: Arc<Lut3D>,
    /// LUT 之后再做主体调整的合成 LUT（主体调整为空时没有）
    subject: Option<Lut3D>,
    strength: f32,
    black_point: BlackPoint,
}

impl Grade {
    /// 按预设参数加载（读取所引用的 LUT）。
    pub fn load(p: &GradeParams) -> anyhow::Result<Self> {
        Ok(Self::new(load_lut(&p.lut)?, p))
    }

    pub fn new(base: Arc<Lut3D>, p: &GradeParams) -> Self {
        let subject = (!p.subject.is_noop()).then(|| {
            let mut s = (*base).clone();
            s.data.par_iter_mut().for_each(|v| *v = p.subject.map(*v));
            s
        });
        Self {
            base,
            subject,
            strength: p.strength.clamp(0.0, 1.0),
            black_point: p.black_point,
        }
    }

    /// 这张图的黑点（由调色前的整图统计）。
    pub fn black_point_of(&self, img: &ImgF32) -> f32 {
        self.black_point.of(img)
    }

    /// 是否有人物主体的调整（需要人像 alpha）。
    pub fn adjusts_subject(&self) -> bool {
        self.subject.is_some()
    }

    /// 原地应用。`black` 为 [`Grade::black_point_of`] 的结果（截到 0..0.9）；`matte` 为人像 alpha（任意尺寸，
    /// 按比例取样；为 None 或空图时不做主体调整）。
    pub fn apply(&self, img: &mut ImgF32, black: f32, matte: Option<&GrayF32>) {
        if self.strength <= 0.0 {
            return;
        }
        let (w, h) = (img.w, img.h);
        let black = black.clamp(0.0, 0.9);
        let matte = matte.filter(|m| m.w > 0 && m.h > 0);
        let subject = self.subject.as_ref().zip(matte);
        let base = (black > 0.0).then(|| {
            let s = (w.max(h) as f32 / REF_LONG_SIDE).max(0.2);
            edge_aware_base(img, ((BASE_RADIUS * s).round() as usize).max(1), BASE_EPS)
        });
        img.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, p) in row.iter_mut().enumerate() {
                let c = match &base {
                    Some(base) => {
                        let b = base.data[y * w + x];
                        [0, 1, 2].map(|i| (soft_black(b[i], black) + p[i] - b[i]).clamp(0.0, 1.0))
                    }
                    None => p.map(|v| v.clamp(0.0, 1.0)),
                };
                let mut out = self.base.lookup(c);
                if let Some((lut, m)) = subject {
                    let a = alpha_at(m, x, y, w, h);
                    if a > 0.0 {
                        let s = lut.lookup(c);
                        for i in 0..3 {
                            out[i] += (s[i] - out[i]) * a;
                        }
                    }
                }
                for i in 0..3 {
                    p[i] += (out[i] - p[i]) * self.strength;
                }
            }
        });
    }
}

/// 软黑点：`(x − b·s(x)) / (1 − b)`，s = smoothstep(0, 2b, x)，结果截到 0..1。
#[inline]
fn soft_black(x: f32, b: f32) -> f32 {
    if b <= 0.0 {
        return x.clamp(0.0, 1.0);
    }
    let t = (x / (2.0 * b)).clamp(0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    ((x - b * s) / (1.0 - b)).clamp(0.0, 1.0)
}

#[inline]
fn alpha_at(m: &GrayF32, x: usize, y: usize, w: usize, h: usize) -> f32 {
    let a = if m.w == w && m.h == h {
        m.get(x, y)
    } else {
        m.sample_bilinear(
            (x as f32 + 0.5) * m.w as f32 / w as f32,
            (y as f32 + 0.5) * m.h as f32 / h as f32,
        )
    };
    a.clamp(0.0, 1.0)
}

/// 整图 min(R, G, B) 的分位（0..1）：8 位直方图上累计计数首次达到 `percentile`%（至少 1 个像素）的一格
/// （与 grade_fit.py 相同）。
pub fn min_channel_percentile(img: &ImgF32, percentile: f32) -> f32 {
    let hist = img
        .data
        .par_chunks(1 << 16)
        .map(|chunk| {
            let mut h = [0u64; 256];
            for p in chunk {
                let m = p[0].min(p[1]).min(p[2]).clamp(0.0, 1.0);
                h[(m * 255.0).round() as usize] += 1;
            }
            h
        })
        .reduce(
            || [0u64; 256],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        );
    let total: u64 = hist.iter().sum();
    let goal = (f64::from(percentile.clamp(0.0, 100.0)) / 100.0 * total as f64).max(1.0);
    let mut acc = 0u64;
    for (i, c) in hist.iter().enumerate() {
        acc += c;
        if acc as f64 >= goal {
            return i as f32 / 255.0;
        }
    }
    1.0
}

/// 人物主体的权重：人像 alpha 去掉与人不相连的孤立块。MODNet 偶尔在背景里给出一块圆形误检（1V3A3127 新娘
/// 颈旁的红色背景），主体调整会把它提亮成一块圆斑。在缩小的二值图上做开运算切断发丝一样细的连接，只保留含有
/// 人脸中心（`seeds`，原图坐标）或面积足够大（没检测到脸的人）的连通块，再乘回原来的软 alpha。
/// `scale` 为脸部尺度（像素，`FaceKeyPoints::scale_distance`）。
pub fn subject_weights(matte: &GrayF32, seeds: &[P], scale: f32) -> GrayF32 {
    let (w, h) = (matte.w, matte.h);
    if w == 0 || h == 0 {
        return matte.clone();
    }
    let k = (scale / SUBJECT_WORK_SCALE).floor().max(1.0);
    let (sw, sh) = (
        ((w as f32 / k).round() as usize).max(1),
        ((h as f32 / k).round() as usize).max(1),
    );
    let small = matte.resize(sw, sh);
    let binary = GrayF32::from_vec(
        sw,
        sh,
        small
            .data
            .iter()
            .map(|v| f32::from(u8::from(*v > 0.5)))
            .collect(),
    );
    let r = ((SUBJECT_OPEN * scale / k).round() as usize).max(1);
    let opened = dilate(&erode(&binary, r), r);
    let hit: Vec<u8> = opened.data.iter().map(|v| u8::from(*v > 0.5)).collect();
    let seeds: Vec<usize> = seeds
        .iter()
        .map(|p| {
            let x = ((p.x / w as f32) * sw as f32).clamp(0.0, (sw - 1) as f32) as usize;
            let y = ((p.y / h as f32) * sh as f32).clamp(0.0, (sh - 1) as f32) as usize;
            y * sw + x
        })
        .collect();
    let min_area = (SUBJECT_MIN_AREA * (sw * sh) as f32) as usize;
    let mut keep = GrayF32::new(sw, sh);
    for c in components(&hit, sw, sh, usize::MAX) {
        if c.pixels.len() >= min_area || c.pixels.iter().any(|i| seeds.contains(i)) {
            c.pixels.iter().for_each(|&i| keep.data[i] = 1.0);
        }
    }
    // 开运算削掉的边缘补回来（只在原二值图内），再羽化成与 alpha 相乘的软权重
    let mut keep = dilate(&keep, r + 1);
    keep.data
        .iter_mut()
        .zip(&binary.data)
        .for_each(|(v, b)| *v *= b);
    let keep = fast_gaussian(&keep, 1.0).resize(w, h);
    let data = matte
        .data
        .par_iter()
        .zip(&keep.data)
        .map(|(a, m)| a * m.clamp(0.0, 1.0))
        .collect();
    GrayF32::from_vec(w, h, data)
}

/// 按预设里的写法取 LUT：`builtin:<名称>` 为编译进程序的内置 LUT，其余为 .cube 路径。
pub fn load_lut(spec: &str) -> anyhow::Result<Arc<Lut3D>> {
    if spec.trim().is_empty() {
        anyhow::bail!("grade.lut is empty: use builtin:<name> or a .cube path");
    }
    match spec.strip_prefix(BUILTIN_PREFIX) {
        Some(name) => builtin_lut(name)
            .ok_or_else(|| anyhow::anyhow!("grade.lut: no builtin LUT named {name:?}")),
        None => Lut3D::load(spec)
            .map(Arc::new)
            .map_err(|e| anyhow::anyhow!("grade.lut {spec}: {e}")),
    }
}

/// 内置 LUT（首次使用时解析，之后共享）。
pub fn builtin_lut(name: &str) -> Option<Arc<Lut3D>> {
    static WEDDING: OnceLock<Arc<Lut3D>> = OnceLock::new();
    match name {
        WEDDING_DARK_INTERIOR => Some(
            WEDDING
                .get_or_init(|| {
                    let text = include_str!("../../luts/wedding_dark_interior.cube");
                    Arc::new(Lut3D::parse_cube(text).expect("builtin LUT wedding_dark_interior"))
                })
                .clone(),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(w: usize, h: usize, c: [f32; 3]) -> ImgF32 {
        let mut img = ImgF32::new(w, h);
        img.data.iter_mut().for_each(|p| *p = c);
        img
    }

    fn identity_grade(subject: SubjectTone, black: BlackPoint) -> Grade {
        let p = GradeParams {
            lut: String::new(),
            strength: 1.0,
            black_point: black,
            subject,
        };
        Grade::new(Arc::new(Lut3D::identity(17)), &p)
    }

    #[test]
    fn black_point_rises_only_without_real_blacks() {
        let bp = BlackPoint {
            percentile: 2.0,
            strength: 0.6,
            target: 0.02,
        };
        // 灰雾一样的画面：min(RGB) 全是 0.6 → 黑点 0.6·(0.6 − 0.02)
        let hazy = filled(40, 30, [0.7, 0.65, 0.6]);
        let expect = 0.6 * ((0.6f32 * 255.0).round() / 255.0 - 0.02);
        assert!((bp.of(&hazy) - expect).abs() < 1e-6, "{}", bp.of(&hazy));
        // 有 5% 的纯黑：2% 分位落在黑里，不动
        let mut contrasty = hazy.clone();
        contrasty.data[..60].iter_mut().for_each(|p| *p = [0.0; 3]);
        assert_eq!(bp.of(&contrasty), 0.0);
        assert_eq!(
            BlackPoint::default().of(&hazy),
            0.0,
            "strength 0 disables it"
        );
    }

    #[test]
    fn identity_grade_keeps_the_image() {
        let mut img = ImgF32::new(16, 8);
        for (i, p) in img.data.iter_mut().enumerate() {
            let x = i as f32 / 128.0;
            *p = [x, (x * 3.0) % 1.0, 1.0 - x];
        }
        let orig = img.clone();
        let g = identity_grade(SubjectTone::default(), BlackPoint::default());
        g.apply(&mut img, 0.0, None);
        assert!(img.max_abs_diff(&orig) < 1e-5);
    }

    #[test]
    fn black_point_stretches_the_range_without_crushing_shadows() {
        // b = 0.2：2b 以上与硬黑点相同，以下逐渐减弱——0.2 → 0.125（硬黑点为 0），0.1 仍大于 0
        let mut img = filled(4, 4, [0.5, 0.2, 0.9]);
        identity_grade(SubjectTone::default(), BlackPoint::default()).apply(&mut img, 0.2, None);
        let p = img.data[0];
        assert!(
            (p[0] - 0.375).abs() < 1e-4
                && (p[1] - 0.125).abs() < 1e-4
                && (p[2] - 0.875).abs() < 1e-4,
            "{p:?}"
        );
        assert!(soft_black(0.1, 0.2) > 0.05 && soft_black(0.0, 0.2) == 0.0);
        assert!(
            (1..100).all(
                |i| soft_black(i as f32 / 100.0, 0.2) > soft_black((i - 1) as f32 / 100.0, 0.2)
            ),
            "monotonic"
        );
    }

    #[test]
    fn black_point_moves_the_base_but_keeps_the_texture() {
        // 0.5 附近 ±0.01 的细纹理：黑点 0.3 逐像素拉伸会放大 1/(1 − 0.3) ≈ 1.43 倍，作用在底图上则基本不变
        let mut img = ImgF32::new(64, 64);
        for (i, p) in img.data.iter_mut().enumerate() {
            let v = 0.5
                + if (i % 64 + i / 64) % 2 == 0 {
                    0.01
                } else {
                    -0.01
                };
            *p = [v; 3];
        }
        let amplitude = |im: &ImgF32| (im.data[64 * 32 + 32][0] - im.data[64 * 32 + 33][0]).abs();
        let before = amplitude(&img);
        identity_grade(SubjectTone::default(), BlackPoint::default()).apply(&mut img, 0.3, None);
        let after = amplitude(&img);
        assert!(
            (after / before - 1.0).abs() < 0.15,
            "texture {before} → {after}"
        );
        let mean = img.data.iter().map(|p| p[0]).sum::<f32>() / img.data.len() as f32;
        assert!(
            (mean - soft_black(0.5, 0.3)).abs() < 0.01,
            "the base still moves: {mean}"
        );
    }

    #[test]
    fn subject_adjustment_follows_the_matte() {
        let subject = SubjectTone {
            lightness: [10.0; 6],
            chroma: 0.0,
        };
        let g = identity_grade(subject, BlackPoint::default());
        let gray = [0.4, 0.4, 0.4];
        let mut img = filled(3, 1, gray);
        let matte = GrayF32::from_vec(3, 1, vec![0.0, 0.5, 1.0]);
        g.apply(&mut img, 0.0, Some(&matte));
        let l = |p: [f32; 3]| rgb_to_lab(p)[0];
        let l0 = l(gray);
        let d: Vec<f32> = img.data.iter().map(|p| l(*p) - l0).collect();
        assert!(d[0].abs() < 1e-3, "{d:?}");
        assert!((d[2] - 10.0).abs() < 0.5, "{d:?}");
        assert!(d[1] > 4.0 && d[1] < 6.0, "{d:?}");
        // 没有人像 alpha 时不做主体调整
        let mut plain = filled(3, 1, gray);
        g.apply(&mut plain, 0.0, None);
        assert!(plain.data.iter().all(|p| (l(*p) - l0).abs() < 1e-3));
    }

    #[test]
    fn subject_curve_interpolates_between_knots() {
        let s = SubjectTone {
            lightness: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            chroma: 0.0,
        };
        assert!((s.lightness_at(50.0) - 5.0).abs() < 1e-6);
        assert_eq!(s.lightness_at(-5.0), 0.0);
        assert_eq!(s.lightness_at(120.0), 10.0);
    }

    #[test]
    fn builtin_lut_darkens_and_deepens_blue() {
        let lut = load_lut("builtin:wedding_dark_interior").unwrap();
        let gray = lut.lookup([0.6, 0.6, 0.6]);
        assert!(
            rgb_to_lab(gray)[0] < rgb_to_lab([0.6; 3])[0] - 5.0,
            "{gray:?}"
        );
        let sky = [0.45, 0.65, 0.9];
        assert!(
            rgb_to_lab(lut.lookup(sky))[0] < rgb_to_lab(sky)[0],
            "the blues get darker"
        );
        assert!(load_lut("builtin:no_such_lut").is_err());
    }

    #[test]
    fn params_roundtrip_and_default_to_off() {
        let p = GradeParams {
            lut: "builtin:wedding_dark_interior".into(),
            strength: 0.8,
            black_point: BlackPoint {
                percentile: 2.0,
                strength: 0.6,
                target: 0.02,
            },
            subject: SubjectTone {
                lightness: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                chroma: 0.1,
            },
        };
        let back: GradeParams = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(back, p);
        let partial: GradeParams = serde_json::from_str(r#"{"lut": "x.cube"}"#).unwrap();
        assert_eq!(partial.strength, 1.0);
        assert_eq!(partial.black_point.strength, 0.0);
        assert!(partial.subject.is_noop());
    }

    /// 128×128 的人像 alpha：左边一个人（含"脸"），右下一个孤立圆斑，右上一个用 1 像素细线挂在人身上的圆斑
    /// （圆斑各占画面约 0.7%，小于"没有人脸也保留"的面积）。
    fn matte_with_blobs() -> GrayF32 {
        const N: usize = 128;
        let mut m = GrayF32::new(N, N);
        let disk = |m: &mut GrayF32, cx: f32, cy: f32, r: f32| {
            for y in 0..N {
                for x in 0..N {
                    if (x as f32 - cx).hypot(y as f32 - cy) <= r {
                        m.data[y * N + x] = 1.0;
                    }
                }
            }
        };
        for y in 16..120 {
            for x in 4..48 {
                m.data[y * N + x] = 1.0;
            }
        }
        disk(&mut m, 96.0, 96.0, 6.0);
        disk(&mut m, 96.0, 30.0, 6.0);
        for x in 48..90 {
            m.data[30 * N + x] = 1.0;
        }
        m
    }

    #[test]
    fn subject_weights_drop_blobs_detached_from_people() {
        let m = matte_with_blobs();
        let s = subject_weights(&m, &[P::new(26.0, 30.0)], 16.0);
        assert!(s.get(26, 80) > 0.99, "the person stays");
        assert!(s.get(96, 96) < 0.01, "isolated blob dropped");
        assert!(
            s.get(96, 30) < 0.01,
            "blob hanging on a hair-thin bridge dropped"
        );
        // 没有人脸时：面积够大的块（没检测到脸的人）仍保留，小块去掉
        let s = subject_weights(&m, &[], 16.0);
        assert!(s.get(26, 80) > 0.99 && s.get(96, 96) < 0.01);
    }

    #[test]
    fn lut_errors_name_the_grade_field() {
        // `"grade": {}` 或 `--set grade.strength=…` 会得到空的 lut：报错要指明是 grade.lut
        for spec in ["", "  ", "no/such/look.cube", "builtin:nothing"] {
            let e = load_lut(spec).unwrap_err().to_string();
            assert!(e.contains("grade.lut"), "{spec:?}: {e}");
        }
    }

    #[test]
    fn percentile_zero_is_the_darkest_pixel_and_black_is_bounded() {
        // 分位 0 落在第一个非空的格（不是空的第 0 格）
        let img = filled(8, 8, [0.7, 0.65, 0.6]);
        assert_eq!(
            min_channel_percentile(&img, 0.0),
            (0.6f32 * 255.0).round() / 255.0
        );
        // 越界的黑点被截到 0.9，不产生 NaN
        let mut img = filled(8, 8, [0.95, 0.5, 0.1]);
        identity_grade(SubjectTone::default(), BlackPoint::default()).apply(&mut img, 1.5, None);
        assert!(img.data.iter().flatten().all(|v| v.is_finite()));
    }
}
