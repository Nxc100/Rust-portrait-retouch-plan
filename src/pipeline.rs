//! 流水线编排：皮肤遮罩 → 磨皮（A/B/C/奶油肌）→ 提亮饱和 → 美白 → 形变 → 风格 LUT → 遮罩 LUT。
//! 预设调色（`color::grade`）在形变之前：奶油肌在修图之前作用于底图（像素蛋糕先调色、后修图），
//! 其余磨皮模式在磨皮、美白之后。
//!
//! 顺序与美狐 Demo 一致（颜色 → 形变 → 风格），关键点只在原图上检测一次。
//! 所有半径 / 位移以瞳距为尺度，所有模糊在固定短边的工作副本上进行，预览与导出观感一致。

use crate::buffer::{GrayF32, ImgF32};
use crate::color::curve::Curve256;
use crate::color::grade::{subject_weights, Grade};
use crate::color::lookup512::apply_lookup512;
use crate::color::lut3d::{apply_lut3d, Lut3D};
use crate::face::attribute::Gender;
use crate::face::semantic::FaceKeyPoints;
use crate::geom::P;
use crate::preset::GenderWarp;
use crate::skin::ai_blemish::AiPatches;
use crate::skin::cream::CreamParams;
use crate::skin::masks::SkinMasks;
use crate::skin::{self, FaithfulPrecomp, FreqSepPrecomp, GpuPixelPrecomp};
use crate::sync::lock;
use crate::warp::face_warp::{FaceWarp, ReshapeStyle, WarpCoefficients, WarpParams};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::{Arc, Mutex, OnceLock};

/// 磨皮模式。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SmoothMode {
    /// 方案 A：忠实还原 BBGPUImageBeautifyFilter（"直播磨皮"）
    #[default]
    Faithful,
    /// 方案 B：频率分离 / 高反差保留（"自然磨皮"，静态修图推荐）
    FreqSep,
    /// 方案 C：gpupixel 均值 / 方差自适应磨皮
    GpuPixel,
    /// 奶油肌：导向滤波保纹理磨皮 + 匀肤 + 奶油色调 + 祛瑕疵（对标像素蛋糕同名预设）
    Cream,
}

/// 美白模式。
#[derive(Clone, Debug, Default)]
pub enum WhitenMode {
    /// 参数化曲线（零素材）
    #[default]
    Curve,
    /// 512 查找图（自制 white.png）
    Lookup512(Arc<ImgF32>),
}

/// 风格滤镜。
#[derive(Clone, Debug)]
pub enum StyleFilter {
    Lookup512(Arc<ImgF32>),
    Cube(Arc<Lut3D>),
}

/// 带空间遮罩的 LUT（曲线滤镜包中的暗角 / 渐变部分）：
/// `out = mix(cur, lut(cur), m(x, y) · strength)`，`invert` 时 m = 1 − mask。
#[derive(Clone, Debug)]
pub struct MaskedLutOp {
    pub lut: Arc<Lut3D>,
    /// 任意尺寸灰度遮罩，运行时拉伸到全图
    pub mask: Arc<GrayF32>,
    pub invert: bool,
    pub strength: f32,
}

#[derive(Clone, Debug)]
pub struct RetouchParams {
    // ---- 磨皮 ----
    /// 0..1，对应 smoothDegree（A）/ amount（B）/ blurAlpha（C）/ 整体强度（奶油肌），默认 0.5
    pub smooth: f32,
    pub smooth_mode: SmoothMode,
    /// 仅在人脸多边形内磨皮 / 美白（改进项，默认 true）
    pub restrict_to_face: bool,
    /// 方案 B：高反差半径（相对短边 1000 px），默认 8
    pub freqsep_radius: f32,
    /// 方案 B：锐化系数（× amount），默认 0.6
    pub freqsep_sharpness: f32,
    /// 方案 C：锐化系数，默认 0
    pub gpupixel_sharpen: f32,
    /// 奶油肌参数（默认 / 女 / 男；有性别信息时按性别选择）
    pub cream: CreamParams,
    pub cream_female: Option<CreamParams>,
    pub cream_male: Option<CreamParams>,
    /// 奶油肌是否处理身体皮肤（颈胸臂手）
    pub body_skin: bool,
    /// 奶油肌：AI 瑕疵祛除强度（0 关闭；需要 models/abpn_blemish_*.onnx，缺失时自动跳过）
    pub ai_blemish: f32,
    // ---- 提亮 / 饱和（HSB，作用于全图）----
    /// 默认 1.1
    pub brightness: f32,
    /// 默认 1.1
    pub saturation: f32,
    /// 还原 BB 的 log 提亮，默认 true（会改变背景，可关闭）
    pub apply_log_curve: bool,
    // ---- 美白 ----
    /// 0..1，默认 0
    pub whiten: f32,
    pub whiten_mode: WhitenMode,
    // ---- 形变 ----
    pub thin_face: f32,
    pub big_eye: f32,
    pub thin_nose: f32,
    pub chin_lift: f32,
    pub face_narrow: f32,
    /// 按性别区分的美型参数（有则优先于上面的全局值）
    pub gender_warp: Option<GenderWarp>,
    /// 逐脸覆盖（按传入 faces 的顺序；None 表示不覆盖）
    pub per_face_warp: Vec<Option<WarpParams>>,
    pub reshape_style: ReshapeStyle,
    pub warp_coeffs: WarpCoefficients,
    /// 侧脸时衰减瘦脸 / 瘦鼻（默认 true）
    pub attenuate_yaw: bool,
    // ---- 预设调色（见 `color::grade`）：奶油肌在修图之前作用于底图，其余模式在磨皮之后；
    // 人物主体的调整需要人像 alpha，没有时只做全局部分 ----
    pub grade: Option<Arc<Grade>>,
    // ---- 风格 ----
    pub style: Option<StyleFilter>,
    /// 0..1，默认 0.8
    pub style_intensity: f32,
    pub masked_ops: Vec<MaskedLutOp>,
    // ---- 输入文件 ----
    /// 再应用输入文件 XMP 里的 Camera Raw 冲印设置（见 `color::develop`；默认关闭）。
    /// 修图管线本身不读文件：由调用方（CLI）读取设置并在人脸检测前作用于图像。
    pub embedded_develop: bool,
}

impl Default for RetouchParams {
    fn default() -> Self {
        Self {
            smooth: 0.5,
            smooth_mode: SmoothMode::Faithful,
            restrict_to_face: true,
            freqsep_radius: skin::smooth_freqsep::DEFAULT_RADIUS,
            freqsep_sharpness: skin::smooth_freqsep::DEFAULT_SHARPNESS,
            gpupixel_sharpen: 0.0,
            cream: CreamParams::default(),
            cream_female: None,
            cream_male: None,
            body_skin: true,
            ai_blemish: 1.0,
            brightness: 1.1,
            saturation: 1.1,
            apply_log_curve: true,
            whiten: 0.0,
            whiten_mode: WhitenMode::Curve,
            thin_face: 0.0,
            big_eye: 0.0,
            thin_nose: 0.0,
            chin_lift: 0.0,
            face_narrow: 0.0,
            gender_warp: None,
            per_face_warp: Vec::new(),
            reshape_style: ReshapeStyle::Meihu,
            warp_coeffs: WarpCoefficients::default(),
            attenuate_yaw: true,
            grade: None,
            style: None,
            style_intensity: 0.8,
            masked_ops: Vec::new(),
            embedded_develop: false,
        }
    }
}

impl RetouchParams {
    /// 完全不改变图像的参数（用于恒等测试）。
    pub fn identity() -> Self {
        Self {
            smooth: 0.0,
            brightness: 1.0,
            saturation: 1.0,
            apply_log_curve: false,
            style_intensity: 0.0,
            ..Default::default()
        }
    }

    /// 全局美型参数。
    pub fn global_warp(&self) -> WarpParams {
        WarpParams {
            thin_face: self.thin_face,
            big_eye: self.big_eye,
            thin_nose: self.thin_nose,
            chin_lift: self.chin_lift,
            face_narrow: self.face_narrow,
        }
    }

    /// 第 `index` 张脸（按传入顺序）实际使用的美型参数。
    pub fn warp_for(&self, index: usize, gender: Option<Gender>) -> WarpParams {
        if let Some(Some(w)) = self.per_face_warp.get(index) {
            return *w;
        }
        if let Some(gw) = &self.gender_warp {
            return gw.pick(gender);
        }
        self.global_warp()
    }

    /// 第 `index` 张脸的奶油肌参数。
    pub fn cream_for(&self, gender: Option<Gender>) -> &CreamParams {
        match gender {
            Some(Gender::Female) => self.cream_female.as_ref().unwrap_or(&self.cream),
            Some(Gender::Male) => self.cream_male.as_ref().unwrap_or(&self.cream),
            None => &self.cream,
        }
    }

    pub fn has_warp(&self) -> bool {
        !self.global_warp().is_zero()
            || self
                .gender_warp
                .as_ref()
                .is_some_and(|g| !g.female.is_zero() || !g.male.is_zero() || !g.unknown.is_zero())
            || self
                .per_face_warp
                .iter()
                .any(|w| w.is_some_and(|w| !w.is_zero()))
    }
}

/// 只依赖原图的中间结果（惰性计算、可缓存；滑块变化时复用）。
pub struct Precomp {
    pub orig: ImgF32,
    faithful: OnceLock<FaithfulPrecomp>,
    gpupixel: OnceLock<GpuPixelPrecomp>,
    freqsep: Mutex<Option<(f32, Arc<FreqSepPrecomp>)>>,
    face_mask: Mutex<Option<(u64, Arc<GrayF32>)>>,
    skin_masks: Mutex<Option<(u64, Arc<SkinMasks>)>>,
    matte: Mutex<Option<Arc<GrayF32>>>,
    skin_prob: Mutex<Option<Arc<GrayF32>>>,
    ai_patches: Mutex<Option<(u64, Arc<AiPatches>)>>,
    skin_curve: OnceLock<Curve256>,
}

impl Precomp {
    pub fn new(orig: ImgF32) -> Self {
        Self {
            orig,
            faithful: OnceLock::new(),
            gpupixel: OnceLock::new(),
            freqsep: Mutex::new(None),
            face_mask: Mutex::new(None),
            skin_masks: Mutex::new(None),
            matte: Mutex::new(None),
            skin_prob: Mutex::new(None),
            ai_patches: Mutex::new(None),
            skin_curve: OnceLock::new(),
        }
    }
    pub fn from_rgb8(img: &image::RgbImage) -> Self {
        Self::new(ImgF32::from_rgb8(img))
    }
    pub fn faithful(&self) -> &FaithfulPrecomp {
        self.faithful
            .get_or_init(|| skin::precompute_faithful(&self.orig, skin::WORK_SHORT_SIDE_A))
    }
    pub fn gpupixel(&self) -> &GpuPixelPrecomp {
        self.gpupixel
            .get_or_init(|| skin::precompute_gpupixel(&self.orig, skin::WORK_SHORT_SIDE_A))
    }
    pub fn freqsep(&self, radius: f32) -> Arc<FreqSepPrecomp> {
        let mut g = lock(&self.freqsep);
        if let Some((r, p)) = g.as_ref() {
            if (*r - radius).abs() < 1e-6 {
                return p.clone();
            }
        }
        let p = Arc::new(skin::precompute_freqsep(
            &self.orig,
            skin::WORK_SHORT_SIDE_B,
            radius,
        ));
        *g = Some((radius, p.clone()));
        p
    }
    pub fn skin_curve(&self) -> &Curve256 {
        self.skin_curve
            .get_or_init(skin::smooth_freqsep::default_skin_curve)
    }
    /// 人脸多边形羽化遮罩（按人脸集合缓存）。
    pub fn face_mask(&self, faces: &[FaceKeyPoints]) -> Arc<GrayF32> {
        let key = faces_key(faces);
        let mut g = lock(&self.face_mask);
        if let Some((k, m)) = g.as_ref() {
            if *k == key {
                return m.clone();
            }
        }
        let m = Arc::new(skin::face_mask_fullres(
            self.orig.w,
            self.orig.h,
            faces,
            skin::WORK_SHORT_SIDE_A,
        ));
        *g = Some((key, m.clone()));
        m
    }
    /// 设置 / 读取人像 alpha（由 Engine 用 MODNet 计算；无模型时为 None）。
    pub fn set_matte(&self, matte: Option<Arc<GrayF32>>) {
        *lock(&self.matte) = matte;
    }
    pub fn matte(&self) -> Option<Arc<GrayF32>> {
        lock(&self.matte).clone()
    }
    /// 设置 / 读取语义皮肤概率图（由 Engine 用皮肤分割模型计算；无模型时为 None）。
    pub fn set_skin_prob(&self, p: Option<Arc<GrayF32>>) {
        *lock(&self.skin_prob) = p;
    }
    pub fn skin_prob(&self) -> Option<Arc<GrayF32>> {
        lock(&self.skin_prob).clone()
    }
    /// 设置 / 读取 AI 瑕疵补丁（由 Engine 计算；按人脸集合键缓存）。
    pub fn set_ai_patches(&self, key: u64, patches: Arc<AiPatches>) {
        *lock(&self.ai_patches) = Some((key, patches));
    }
    pub fn ai_patches(&self, key: u64) -> Option<Arc<AiPatches>> {
        let g = lock(&self.ai_patches);
        match g.as_ref() {
            Some((k, p)) if *k == key => Some(p.clone()),
            _ => None,
        }
    }
    /// 奶油肌用的皮肤遮罩（解析 + 抠图 + 颜色规则；按人脸集合与是否含身体缓存）。
    pub fn skin_masks(&self, faces: &[FaceKeyPoints], body: bool) -> Arc<SkinMasks> {
        let matte = if body { self.matte() } else { None };
        let skin = if body { self.skin_prob() } else { None };
        let mut key = faces_key(faces);
        key ^= if body { 0x9e37_79b9_7f4a_7c15 } else { 0 };
        key ^= if matte.is_some() {
            0x5851_f42d_4c95_7f2d
        } else {
            0
        };
        key ^= if skin.is_some() {
            0x2545_f491_4f6c_dd1d
        } else {
            0
        };
        let mut g = lock(&self.skin_masks);
        if let Some((k, m)) = g.as_ref() {
            if *k == key {
                return m.clone();
            }
        }
        let m = Arc::new(skin::masks::build_skin_masks(
            &self.orig,
            faces,
            matte.as_ref(),
            skin.as_ref(),
        ));
        *g = Some((key, m.clone()));
        m
    }
    /// 是否已计算过某模式的中间结果（用于测试 / 基准）。
    pub fn has_faithful(&self) -> bool {
        self.faithful.get().is_some()
    }
}

pub fn faces_key(faces: &[FaceKeyPoints]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut feed = |v: f32| {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    feed(faces.len() as f32);
    for f in faces {
        for p in f.contour.iter().chain(f.forehead.iter()) {
            feed(p.x);
            feed(p.y);
        }
        feed(f.pupil_l.x);
        feed(f.pupil_r.x);
        feed(if f.parse.is_some() { 1.0 } else { 0.0 });
    }
    h
}

/// 多人脸策略：按面积升序（小脸先应用）；任意两脸 IoU > 0.3 时只处理最大脸。
/// 返回 (原始索引, 人脸)。
pub fn select_faces_indexed(faces: &[FaceKeyPoints]) -> Vec<(usize, FaceKeyPoints)> {
    let mut v: Vec<(usize, FaceKeyPoints)> = faces.iter().cloned().enumerate().collect();
    v.sort_by(|a, b| {
        a.1.bbox
            .area()
            .partial_cmp(&b.1.bbox.area())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let overlap = v
        .iter()
        .enumerate()
        .any(|(i, a)| v.iter().skip(i + 1).any(|b| a.1.bbox.iou(&b.1.bbox) > 0.3));
    if overlap {
        if let Some(largest) = v.last().cloned() {
            return vec![largest];
        }
    }
    v
}

/// 多人脸策略（只返回人脸）。
pub fn select_faces(faces: &[FaceKeyPoints]) -> Vec<FaceKeyPoints> {
    select_faces_indexed(faces)
        .into_iter()
        .map(|(_, f)| f)
        .collect()
}

/// 皮肤遮罩（颜色规则 × 人脸多边形），用于美白。
fn skin_mask_for_whiten(pre: &Precomp, face_mask: Option<&Arc<GrayF32>>) -> GrayF32 {
    let (work, scale) = pre.orig.work_copy(skin::WORK_SHORT_SIDE_A);
    let mut m = skin::skin_color_mask(&work);
    if scale < 1.0 {
        m = m.resize(pre.orig.w, pre.orig.h);
    }
    if let Some(fm) = face_mask {
        m.mul_inplace(fm);
    }
    m
}

/// 应用带遮罩的 LUT。
pub fn apply_masked_lut(img: &mut ImgF32, op: &MaskedLutOp) {
    if op.strength <= 0.0 {
        return;
    }
    let (w, h) = (img.w as f32, img.h as f32);
    let mask = &op.mask;
    let sx = mask.w as f32 / w;
    let sy = mask.h as f32 / h;
    let width = img.w;
    img.data
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, p) in row.iter_mut().enumerate() {
                let mut m = mask.sample_bilinear((x as f32 + 0.5) * sx, (y as f32 + 0.5) * sy);
                if op.invert {
                    m = 1.0 - m;
                }
                let t = (m * op.strength).clamp(0.0, 1.0);
                if t <= 0.0 {
                    continue;
                }
                let n = op.lut.lookup(*p);
                for c in 0..3 {
                    p[c] += (n[c] - p[c]) * t;
                }
            }
        });
}

/// 奶油肌的底图：`pre` 带着这组人脸的 AI 瑕疵补丁且 `ai_blemish > 0` 时是修补后的原图，否则就是原图。
fn cream_base<'a>(pre: &'a Precomp, faces: &[FaceKeyPoints], p: &RetouchParams) -> Cow<'a, ImgF32> {
    if p.ai_blemish > 0.0 {
        if let Some(pt) = pre.ai_patches(faces_key(faces)) {
            if !pt.is_empty() {
                return Cow::Owned(pt.apply_to(&pre.orig, p.ai_blemish));
            }
        }
    }
    Cow::Borrowed(&pre.orig)
}

/// 奶油肌修图的输入：底图（[`cream_base`]）再做预设调色。像素蛋糕先调色、后修图——肤色统一作用在调色后的
/// 颜色上，调色把整张图压暗、加对比时，皮肤仍被拉回目标色（doc/analysis/wedding_dark_interior.md）。
fn cream_input<'a>(
    pre: &'a Precomp,
    faces: &[FaceKeyPoints],
    p: &RetouchParams,
) -> Cow<'a, ImgF32> {
    apply_grade(pre, p, faces, cream_base(pre, faces, p))
}

/// 预设调色（见 `color::grade`）：黑点由原图统计；人物主体用人像 alpha（原图几何，所以要在形变之前），
/// 去掉与人脸不相连的误检块（[`subject_weights`]）。没有调色时原样返回。
fn apply_grade<'a>(
    pre: &Precomp,
    p: &RetouchParams,
    faces: &[FaceKeyPoints],
    img: Cow<'a, ImgF32>,
) -> Cow<'a, ImgF32> {
    let Some(g) = &p.grade else {
        return img;
    };
    let t = std::time::Instant::now();
    let black = g.black_point_of(&pre.orig);
    let subject = pre.matte().filter(|_| g.adjusts_subject()).map(|m| {
        let seeds: Vec<P> = faces.iter().map(|f| f.bbox.center()).collect();
        let scale = if faces.is_empty() {
            pre.orig.w.min(pre.orig.h) as f32 * 0.06
        } else {
            faces.iter().map(|f| f.scale_distance()).sum::<f32>() / faces.len() as f32
        };
        subject_weights(&m, &seeds, scale)
    });
    let mut out = img.into_owned();
    g.apply(&mut out, black, subject.as_ref());
    if std::env::var("RETOUCH_TIMING").is_ok() {
        eprintln!(
            "  [grade] black {black:.3}, {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    Cow::Owned(out)
}

/// 奶油肌颈纹淡化的处理权重（全图，0..1；调试 / 可视化用）：与 [`retouch_with`] 相同的人脸选择、参数与底图。
/// 皮肤遮罩与 AI 补丁用 `pre` 当前带的（见 `Engine::neck_weights`）。
pub fn cream_neck_weights(pre: &Precomp, faces: &[FaceKeyPoints], p: &RetouchParams) -> GrayF32 {
    let faces_sel: Vec<FaceKeyPoints> = select_faces_indexed(faces)
        .into_iter()
        .map(|(_, f)| f.clone())
        .collect();
    if faces_sel.is_empty() {
        return GrayF32::new(pre.orig.w, pre.orig.h);
    }
    let masks = pre.skin_masks(&faces_sel, p.body_skin);
    let params: Vec<&CreamParams> = faces_sel.iter().map(|f| p.cream_for(f.gender)).collect();
    let body_params = p.cream_female.as_ref().unwrap_or(&p.cream);
    let base = cream_input(pre, faces, p);
    skin::cream::neck_weights(&base, &masks, &faces_sel, &params, body_params)
}

/// 奶油肌牙齿美白的逐脸权重（`base` 为调色后、修图前的图，`strength` 为总强度）。
fn cream_teeth(
    base: &ImgF32,
    faces_sel: &[FaceKeyPoints],
    p: &RetouchParams,
    strength: f32,
) -> Vec<skin::teeth::TeethWeights> {
    faces_sel
        .iter()
        .filter_map(|f| {
            skin::teeth::teeth_weights(base, f, p.cream_for(f.gender).teeth_whiten * strength)
        })
        .collect()
}

/// 奶油肌牙齿美白的处理权重（全图，0..1；调试 / 可视化用）：与 [`retouch_with`] 相同的人脸选择、参数与底图。
pub fn cream_teeth_weights(pre: &Precomp, faces: &[FaceKeyPoints], p: &RetouchParams) -> GrayF32 {
    let mut all = GrayF32::new(pre.orig.w, pre.orig.h);
    let strength = p.smooth.clamp(0.0, 1.0);
    let faces_sel: Vec<FaceKeyPoints> = select_faces_indexed(faces)
        .into_iter()
        .map(|(_, f)| f.clone())
        .collect();
    if strength > 0.0 && !faces_sel.is_empty() {
        let base = cream_input(pre, faces, p);
        for tw in cream_teeth(&base, &faces_sel, p, strength) {
            tw.max_into(&mut all);
        }
    }
    all
}

/// 完整流水线（f32 输出）。
pub fn retouch_with(pre: &Precomp, faces: &[FaceKeyPoints], p: &RetouchParams) -> ImgF32 {
    let orig = &pre.orig;
    let selected = select_faces_indexed(faces);
    let faces_sel: Vec<FaceKeyPoints> = selected.iter().map(|(_, f)| f.clone()).collect();

    // 1. 皮肤遮罩（可选）
    let face_mask = if p.restrict_to_face && !faces_sel.is_empty() {
        Some(pre.face_mask(&faces_sel))
    } else {
        None
    };
    let fm = face_mask.as_deref();

    // 2. 磨皮 + 提亮 / 饱和
    let mut img = match p.smooth_mode {
        SmoothMode::Faithful => skin::smooth_faithful(
            orig,
            pre.faithful(),
            fm,
            p.smooth.clamp(0.0, 1.0),
            p.apply_log_curve,
            p.brightness,
            p.saturation,
        ),
        SmoothMode::FreqSep => {
            let fp = pre.freqsep(p.freqsep_radius);
            let mut out = skin::smooth_freqsep(
                orig,
                &fp,
                fm,
                p.smooth.clamp(0.0, 1.0),
                p.freqsep_sharpness,
                pre.skin_curve(),
            );
            post_tone(&mut out, p);
            out
        }
        SmoothMode::GpuPixel => {
            let mut out = skin::smooth_gpupixel(
                orig,
                pre.gpupixel(),
                fm,
                p.smooth.clamp(0.0, 1.0),
                p.gpupixel_sharpen,
            );
            post_tone(&mut out, p);
            out
        }
        SmoothMode::Cream => {
            let strength = p.smooth.clamp(0.0, 1.0);
            let mut out = if strength > 0.0 && !faces_sel.is_empty() {
                let masks = pre.skin_masks(&faces_sel, p.body_skin);
                let params: Vec<&CreamParams> =
                    faces_sel.iter().map(|f| p.cream_for(f.gender)).collect();
                let body_params = p.cream_female.as_ref().unwrap_or(&p.cream);
                // AI 瑕疵祛除（先于经典流程；补丁只含被修改的像素）→ 预设调色 → 修图
                let base = cream_input(pre, faces, p);
                let mut o =
                    skin::cream::cream_skin(&base, &masks, &faces_sel, &params, body_params);
                if strength < 1.0 {
                    let plain = apply_grade(pre, p, faces, Cow::Borrowed(orig));
                    o = plain.zip_map(&o, |a, b| {
                        [
                            a[0] + (b[0] - a[0]) * strength,
                            a[1] + (b[1] - a[1]) * strength,
                            a[2] + (b[2] - a[2]) * strength,
                        ]
                    });
                }
                // 牙齿美白（按性别的奶油肌参数，0 时不处理）：门限取调色后、修图前的颜色
                for tw in cream_teeth(&base, &faces_sel, p, strength) {
                    skin::teeth::whiten_teeth(&mut o, &tw);
                }
                o
            } else {
                apply_grade(pre, p, faces, Cow::Borrowed(orig)).into_owned()
            };
            post_tone(&mut out, p);
            out
        }
    };

    // 3. 美白
    if p.whiten > 0.0 {
        let mask = skin_mask_for_whiten(pre, face_mask.as_ref());
        match &p.whiten_mode {
            WhitenMode::Curve => {
                skin::whiten::apply_curve(&mut img, Some(&mask), p.whiten.clamp(0.0, 1.0))
            }
            WhitenMode::Lookup512(lut) => {
                skin::whiten::apply_lookup(&mut img, lut, Some(&mask), p.whiten.clamp(0.0, 1.0))
            }
        }
    }

    // 4. 预设调色：奶油肌已在修图之前做过；其余磨皮模式的预计算都取自原图，只能在磨皮之后做
    if p.smooth_mode != SmoothMode::Cream {
        img = apply_grade(pre, p, faces, Cow::Owned(img)).into_owned();
    }

    // 5. 形变
    if !selected.is_empty() && p.has_warp() {
        let warps: Vec<FaceWarp> = selected
            .iter()
            .map(|(idx, f)| {
                let wp = p.warp_for(*idx, f.gender);
                FaceWarp::from_face_with(f, &wp, &p.warp_coeffs, p.reshape_style, p.attenuate_yaw)
            })
            .collect();
        let t = std::time::Instant::now();
        img = crate::warp::apply_warps(&img, &warps);
        if std::env::var("RETOUCH_TIMING").is_ok() {
            eprintln!("  [warp] {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        }
    }

    // 6. 风格
    match (&p.style, p.style_intensity) {
        (Some(StyleFilter::Lookup512(l)), k) if k > 0.0 => apply_lookup512(&mut img, l, k.min(1.0)),
        (Some(StyleFilter::Cube(l)), k) if k > 0.0 => apply_lut3d(&mut img, l, k.min(1.0)),
        _ => {}
    }
    for op in &p.masked_ops {
        apply_masked_lut(&mut img, op);
    }
    img
}

/// 方案 B / C / 奶油肌之后的 log 提亮 + HSB（与方案 A 的后半段一致）。
fn post_tone(img: &mut ImgF32, p: &RetouchParams) {
    if p.apply_log_curve {
        let ln12 = 1.2f32.ln();
        img.map_inplace(|px| {
            let mut o = px;
            for v in o.iter_mut() {
                *v = (1.0 + 0.2 * *v).ln() / ln12;
            }
            o
        });
    }
    crate::color::hsb::hsb_brightness_saturation(img, p.brightness, p.saturation);
}

/// 便捷函数：从 RgbImage 直接修图（不缓存中间结果，无抠图模型）。
pub fn retouch_impl(
    input: &image::RgbImage,
    faces: &[FaceKeyPoints],
    p: &RetouchParams,
) -> image::RgbImage {
    let pre = Precomp::from_rgb8(input);
    retouch_with(&pre, faces, p).to_rgb8()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: usize, h: usize) -> image::RgbImage {
        let mut img = image::RgbImage::new(w as u32, h as u32);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([
                (x * 255 / w as u32) as u8,
                (y * 255 / h as u32) as u8,
                ((x + y) % 256) as u8,
            ]);
        }
        img
    }

    #[test]
    fn identity_params_no_faces_is_bitwise_identity() {
        let img = gradient(80, 60);
        let out = retouch_impl(&img, &[], &RetouchParams::identity());
        assert_eq!(img.as_raw(), out.as_raw());
    }

    #[test]
    fn identity_params_with_face_but_zero_warp_is_identity() {
        let img = gradient(600, 700);
        let f = crate::warp::face_warp::tests::synthetic_face();
        let out = retouch_impl(&img, &[f], &RetouchParams::identity());
        assert_eq!(img.as_raw(), out.as_raw());
    }

    #[test]
    fn cream_mode_leaves_background_untouched() {
        let mut img = gradient(600, 700);
        // 让脸区域是肤色
        for (x, y, p) in img.enumerate_pixels_mut() {
            if (200..400).contains(&x) && (180..520).contains(&y) {
                *p = image::Rgb([200, 150, 125]);
            }
        }
        let f = crate::warp::face_warp::tests::synthetic_face();
        let p = RetouchParams {
            smooth_mode: SmoothMode::Cream,
            smooth: 1.0,
            brightness: 1.0,
            saturation: 1.0,
            apply_log_curve: false,
            body_skin: false,
            ..Default::default()
        };
        let out = retouch_impl(&img, &[f], &p);
        assert_eq!(img.get_pixel(5, 5), out.get_pixel(5, 5));
        assert_eq!(img.get_pixel(590, 690), out.get_pixel(590, 690));
        // 脸内像素被提亮
        let a = img.get_pixel(300, 350)[1] as i32;
        let b = out.get_pixel(300, 350)[1] as i32;
        assert!(b >= a, "{a} -> {b}");
    }

    #[test]
    fn overlapping_faces_keep_largest() {
        let mut a = crate::warp::face_warp::tests::synthetic_face();
        let mut b = a.clone();
        a.bbox = crate::face::semantic::FaceBox {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
            score: 1.0,
        };
        b.bbox = crate::face::semantic::FaceBox {
            x1: 10.0,
            y1: 10.0,
            x2: 150.0,
            y2: 150.0,
            score: 1.0,
        };
        let sel = select_faces(&[a, b]);
        assert_eq!(sel.len(), 1);
        assert!((sel[0].bbox.x2 - 150.0).abs() < 1e-6);
    }

    #[test]
    fn per_face_and_gender_warp_selection() {
        let mut p = RetouchParams {
            thin_face: 0.3,
            ..Default::default()
        };
        assert!((p.warp_for(0, None).thin_face - 0.3).abs() < 1e-6);
        p.gender_warp = Some(GenderWarp {
            female: WarpParams {
                thin_face: 0.5,
                ..Default::default()
            },
            male: WarpParams {
                thin_face: 0.1,
                ..Default::default()
            },
            unknown: WarpParams {
                thin_face: 0.2,
                ..Default::default()
            },
        });
        assert!((p.warp_for(0, Some(Gender::Male)).thin_face - 0.1).abs() < 1e-6);
        assert!((p.warp_for(0, None).thin_face - 0.2).abs() < 1e-6);
        p.per_face_warp = vec![
            None,
            Some(WarpParams {
                thin_face: 0.9,
                ..Default::default()
            }),
        ];
        assert!((p.warp_for(1, Some(Gender::Female)).thin_face - 0.9).abs() < 1e-6);
        assert!(p.has_warp());
    }
}
