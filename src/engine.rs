//! `Engine`：持有 ONNX Session 与中间结果缓存的公开入口。
//!
//! 设计要点：
//! - 检测与修图分离：拖滑块时只调 `retouch`，关键点复用；
//! - `retouch` 为纯函数（不修改输入），内部按图像内容哈希缓存只依赖原图的中间结果
//!   （双边 / 边缘 / 均值方差 / 高反差遮罩 / 人脸遮罩 / 人像 alpha），改 `smooth` 时只重跑组合之后的步骤；
//! - 可选模型（存在即加载）：人脸解析（BiSeNet）、人像抠图（MODNet）、性别年龄（insightface，非商用）。

use crate::face::attribute::GenderAge;
use crate::face::detector::UltraFace;
use crate::face::lm_2d106::Lm2d106;
use crate::face::lm_facemesh::FaceMesh;
use crate::face::matting::PersonMatting;
use crate::face::parsing::FaceParser;
use crate::face::semantic::{FaceBox, FaceKeyPoints, LandmarkModel};
use crate::face::skinseg::SkinSeg;
use crate::pipeline::{retouch_with, Precomp, RetouchParams, SmoothMode};
use crate::skin::ai_blemish::{AiBlemish, AiPatches};
use crate::skin::masks::SkinMasks;
use crate::sync::lock;
use crate::{GrayF32, ImgF32};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// 关键点模型选择。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LandmarkKind {
    /// InsightFace 2d106det（开发期，模型非商用）
    #[default]
    Lm2d106,
    /// MediaPipe Face Mesh 468 点（Apache-2.0，发布期）
    FaceMesh,
}

impl LandmarkKind {
    pub fn default_file(&self) -> &'static str {
        match self {
            LandmarkKind::Lm2d106 => "2d106det.onnx",
            LandmarkKind::FaceMesh => "face_mesh.onnx",
        }
    }
}

impl std::str::FromStr for LandmarkKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "2d106" | "2d106det" | "lm2d106" | "insightface" => Ok(LandmarkKind::Lm2d106),
            "facemesh" | "face_mesh" | "mediapipe" | "mesh" => Ok(LandmarkKind::FaceMesh),
            _ => Err(format!(
                "unknown landmark model: {s} (expected 2d106 | facemesh)"
            )),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineConfig {
    /// 模型目录（默认 `models/`）
    pub models_dir: PathBuf,
    pub landmark: LandmarkKind,
    /// 显式指定检测 / 关键点模型文件（覆盖 models_dir 下的默认文件名）
    pub detector_model: Option<PathBuf>,
    pub landmark_model: Option<PathBuf>,
    /// 可选模型：None 时在 models_dir 下按默认文件名查找，不存在则不启用
    pub parsing_model: Option<PathBuf>,
    pub matting_model: Option<PathBuf>,
    pub attribute_model: Option<PathBuf>,
    /// AI 瑕疵祛除（ABPN 分割 + 修复；两个文件都存在才启用）
    pub ai_blemish_detect_model: Option<PathBuf>,
    pub ai_blemish_inpaint_model: Option<PathBuf>,
    /// 语义皮肤分割（身体皮肤遮罩的门控）；None 时依次查找 models_dir 下的
    /// `skin_seg.onnx`（ModelScope，推荐）与 `skin_seg_lite.onnx`（轻量备选）
    pub skin_seg_model: Option<PathBuf>,
    pub enable_parsing: bool,
    pub enable_matting: bool,
    pub enable_attribute: bool,
    pub enable_ai_blemish: bool,
    pub enable_skin_seg: bool,
    /// ONNX Runtime 动态库路径（None 时自动查找，见 `face::ort_util`）
    pub ort_dylib: Option<PathBuf>,
    /// ORT 线程数
    pub threads: usize,
    /// 是否缓存中间结果
    pub cache_precomp: bool,
    /// UltraFace 分数阈值 / NMS IoU
    pub score_threshold: f32,
    pub iou_threshold: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            models_dir: PathBuf::from("models"),
            landmark: LandmarkKind::Lm2d106,
            detector_model: None,
            landmark_model: None,
            parsing_model: None,
            matting_model: None,
            attribute_model: None,
            ai_blemish_detect_model: None,
            ai_blemish_inpaint_model: None,
            skin_seg_model: None,
            enable_parsing: true,
            enable_matting: true,
            enable_attribute: true,
            enable_ai_blemish: true,
            enable_skin_seg: true,
            ort_dylib: None,
            threads: 4,
            cache_precomp: true,
            score_threshold: 0.7,
            iou_threshold: 0.3,
        }
    }
}

pub const PARSING_MODEL_FILE: &str = "face_parsing_resnet18.onnx";
pub const MATTING_MODEL_FILE: &str = "modnet_photographic.onnx";
pub const ATTRIBUTE_MODEL_FILE: &str = "genderage.onnx";
pub const AI_BLEMISH_DETECT_FILE: &str = "abpn_blemish_detect.onnx";
pub const AI_BLEMISH_INPAINT_FILE: &str = "abpn_blemish_inpaint.onnx";
pub const SKIN_SEG_MODEL_FILE: &str = "skin_seg.onnx";
pub const SKIN_SEG_LITE_MODEL_FILE: &str = "skin_seg_lite.onnx";

pub struct Engine {
    cfg: EngineConfig,
    detector: Mutex<UltraFace>,
    landmark: Mutex<Box<dyn LandmarkModel>>,
    landmark_name: &'static str,
    parser: Option<Mutex<FaceParser>>,
    matting: Option<Mutex<PersonMatting>>,
    attribute: Option<Mutex<GenderAge>>,
    ai_blemish: Option<Mutex<AiBlemish>>,
    skin_seg: Option<Mutex<SkinSeg>>,
    skin_seg_name: Option<String>,
    cache: Mutex<Option<(u64, Arc<Precomp>)>>,
    matte_cache: Mutex<Option<(u64, Arc<GrayF32>)>>,
    skin_cache: Mutex<Option<(u64, Arc<GrayF32>)>>,
    ort_path: Option<PathBuf>,
}

impl Engine {
    pub fn new(cfg: EngineConfig) -> anyhow::Result<Self> {
        let ort_path = crate::face::ort_util::init_ort(cfg.ort_dylib.as_deref())?;
        let det_path = cfg
            .detector_model
            .clone()
            .unwrap_or_else(|| cfg.models_dir.join("version-RFB-320.onnx"));
        let lm_path = cfg
            .landmark_model
            .clone()
            .unwrap_or_else(|| cfg.models_dir.join(cfg.landmark.default_file()));
        let mut detector = UltraFace::new(&det_path, cfg.threads)?;
        detector.score_threshold = cfg.score_threshold;
        detector.iou_threshold = cfg.iou_threshold;
        let landmark: Box<dyn LandmarkModel> = match cfg.landmark {
            LandmarkKind::Lm2d106 => Box::new(Lm2d106::new(&lm_path, cfg.threads)?),
            LandmarkKind::FaceMesh => Box::new(FaceMesh::new(&lm_path, cfg.threads)?),
        };
        let landmark_name = landmark.name();
        let optional =
            |enabled: bool, explicit: &Option<PathBuf>, default: &str| -> Option<PathBuf> {
                if !enabled {
                    return None;
                }
                match explicit {
                    Some(p) => Some(p.clone()),
                    None => {
                        let p = cfg.models_dir.join(default);
                        if p.is_file() {
                            Some(p)
                        } else {
                            None
                        }
                    }
                }
            };
        let parser = match optional(cfg.enable_parsing, &cfg.parsing_model, PARSING_MODEL_FILE) {
            Some(p) => Some(Mutex::new(FaceParser::new(&p, cfg.threads)?)),
            None => None,
        };
        let matting = match optional(cfg.enable_matting, &cfg.matting_model, MATTING_MODEL_FILE) {
            Some(p) => Some(Mutex::new(PersonMatting::new(&p, cfg.threads)?)),
            None => None,
        };
        let attribute = match optional(
            cfg.enable_attribute,
            &cfg.attribute_model,
            ATTRIBUTE_MODEL_FILE,
        ) {
            Some(p) => Some(Mutex::new(GenderAge::new(&p, cfg.threads)?)),
            None => None,
        };
        let ai_blemish = match (
            optional(
                cfg.enable_ai_blemish,
                &cfg.ai_blemish_detect_model,
                AI_BLEMISH_DETECT_FILE,
            ),
            optional(
                cfg.enable_ai_blemish,
                &cfg.ai_blemish_inpaint_model,
                AI_BLEMISH_INPAINT_FILE,
            ),
        ) {
            (Some(d), Some(i)) => Some(Mutex::new(AiBlemish::new(&d, &i, cfg.threads.max(4))?)),
            _ => None,
        };
        // 语义皮肤分割：显式路径 > skin_seg.onnx（ModelScope，推荐）> skin_seg_lite.onnx（轻量备选）
        let skin_seg_path = optional(
            cfg.enable_skin_seg,
            &cfg.skin_seg_model,
            SKIN_SEG_MODEL_FILE,
        )
        .or_else(|| {
            optional(
                cfg.enable_skin_seg && cfg.skin_seg_model.is_none(),
                &None::<PathBuf>,
                SKIN_SEG_LITE_MODEL_FILE,
            )
        });
        let (skin_seg, skin_seg_name) = match skin_seg_path {
            Some(p) => (
                Some(Mutex::new(SkinSeg::new(&p, cfg.threads)?)),
                p.file_name().map(|s| s.to_string_lossy().to_string()),
            ),
            None => (None, None),
        };
        Ok(Self {
            cfg,
            detector: Mutex::new(detector),
            landmark: Mutex::new(landmark),
            landmark_name,
            parser,
            matting,
            attribute,
            ai_blemish,
            skin_seg,
            skin_seg_name,
            cache: Mutex::new(None),
            matte_cache: Mutex::new(None),
            skin_cache: Mutex::new(None),
            ort_path,
        })
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }
    /// 已加载模型的简短标签：关键点模型名，以及 `+parsing` / `+matting` / `+genderage` / `+ai-blemish` /
    /// `+skinseg(<文件>)`（命令行横幅、界面状态栏共用）。
    pub fn model_tags(&self) -> Vec<String> {
        let mut tags = vec![self.landmark_name().to_string()];
        for (on, tag) in [
            (self.has_parsing(), "+parsing"),
            (self.has_matting(), "+matting"),
            (self.has_attribute(), "+genderage"),
            (self.has_ai_blemish(), "+ai-blemish"),
        ] {
            if on {
                tags.push(tag.to_string());
            }
        }
        if let Some(n) = self.skin_seg_name() {
            tags.push(format!("+skinseg({n})"));
        }
        tags
    }

    pub fn landmark_name(&self) -> &'static str {
        self.landmark_name
    }
    pub fn has_parsing(&self) -> bool {
        self.parser.is_some()
    }
    pub fn has_matting(&self) -> bool {
        self.matting.is_some()
    }
    pub fn has_ai_blemish(&self) -> bool {
        self.ai_blemish.is_some()
    }
    pub fn has_skin_seg(&self) -> bool {
        self.skin_seg.is_some()
    }
    /// 已加载的语义皮肤分割模型文件名（None 表示未启用）。
    pub fn skin_seg_name(&self) -> Option<&str> {
        self.skin_seg_name.as_deref()
    }

    /// 语义皮肤概率图（与原图同尺寸），按图像缓存；无模型返回 None。
    pub fn skin_prob(&self, img: &image::RgbImage) -> anyhow::Result<Option<Arc<GrayF32>>> {
        let Some(m) = &self.skin_seg else {
            return Ok(None);
        };
        let key = image_key(img);
        {
            let g = lock(&self.skin_cache);
            if let Some((k, p)) = g.as_ref() {
                if *k == key {
                    return Ok(Some(p.clone()));
                }
            }
        }
        let t = std::time::Instant::now();
        let p = Arc::new(lock(m).skin_prob(img)?);
        if std::env::var("RETOUCH_TIMING").is_ok() {
            eprintln!("  [skin-seg] {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        *lock(&self.skin_cache) = Some((key, p.clone()));
        Ok(Some(p))
    }

    /// AI 瑕疵补丁（ABPN 分割 + 修复，逐脸 1.5× ROI），按图像 + 人脸集合缓存在 Precomp 中；无模型返回 None。
    pub fn ai_patches(
        &self,
        pre: &Precomp,
        faces: &[FaceKeyPoints],
    ) -> anyhow::Result<Option<Arc<AiPatches>>> {
        let Some(m) = &self.ai_blemish else {
            return Ok(None);
        };
        let key = crate::pipeline::faces_key(faces);
        if let Some(p) = pre.ai_patches(key) {
            return Ok(Some(p));
        }
        let timing = std::env::var("RETOUCH_TIMING").is_ok();
        let t = std::time::Instant::now();
        let boxes: Vec<FaceBox> = faces.iter().map(|f| f.bbox).collect();
        let patches = Arc::new(lock(m).clean_faces(&pre.orig, &boxes)?);
        if timing {
            for (i, p) in patches.patches.iter().enumerate() {
                let s = &p.stats;
                eprintln!(
                    "  [ai-blemish] face {i}: roi {:?} holes {} windows {}/{} detect {:.0} ms inpaint {:.0} ms",
                    s.roi, s.holes, s.windows_run, s.windows_total, s.ms_detect, s.ms_inpaint
                );
            }
            eprintln!(
                "  [ai-blemish] total {:.0} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        pre.set_ai_patches(key, patches.clone());
        Ok(Some(patches))
    }
    pub fn has_attribute(&self) -> bool {
        self.attribute.is_some()
    }
    /// 实际加载的 ONNX Runtime 动态库路径（None = 系统默认搜索）。
    pub fn ort_library_path(&self) -> Option<&PathBuf> {
        self.ort_path.as_ref()
    }

    /// 仅检测人脸框。
    pub fn detect_boxes(&self, img: &image::RgbImage) -> anyhow::Result<Vec<FaceBox>> {
        lock(&self.detector).detect(img)
    }

    /// 对给定人脸框提取语义关键点（含可选的解析 / 性别年龄）。
    pub fn landmarks_for_box(
        &self,
        img: &image::RgbImage,
        face_box: &FaceBox,
    ) -> anyhow::Result<FaceKeyPoints> {
        let mut f = lock(&self.landmark).detect(img, face_box)?;
        if let Some(p) = &self.parser {
            f.parse = Some(Arc::new(lock(p).parse(img, face_box)?));
        }
        if let Some(a) = &self.attribute {
            let (g, age, _conf) = lock(a).predict(img, face_box)?;
            f.gender = Some(g);
            f.age = Some(age);
        }
        Ok(f)
    }

    /// 在原图上检测所有人脸并返回语义关键点（像素坐标），按面积降序。
    pub fn detect_faces(&self, img: &image::RgbImage) -> anyhow::Result<Vec<FaceKeyPoints>> {
        let boxes = self.detect_boxes(img)?;
        let mut out = Vec::with_capacity(boxes.len());
        for b in &boxes {
            out.push(self.landmarks_for_box(img, b)?);
        }
        Ok(out)
    }

    /// 人像 alpha（MODNet），按图像缓存；无模型返回 None。
    pub fn person_matte(&self, img: &image::RgbImage) -> anyhow::Result<Option<Arc<GrayF32>>> {
        let Some(m) = &self.matting else {
            return Ok(None);
        };
        let key = image_key(img);
        {
            let g = lock(&self.matte_cache);
            if let Some((k, m)) = g.as_ref() {
                if *k == key {
                    return Ok(Some(m.clone()));
                }
            }
        }
        let matte = Arc::new(lock(m).matte(img)?);
        *lock(&self.matte_cache) = Some((key, matte.clone()));
        Ok(Some(matte))
    }

    /// 取得（或复用）该图像的中间结果。
    pub fn precomp(&self, img: &image::RgbImage) -> Arc<Precomp> {
        if !self.cfg.cache_precomp {
            return Arc::new(Precomp::from_rgb8(img));
        }
        let key = image_key(img);
        let mut g = lock(&self.cache);
        if let Some((k, p)) = g.as_ref() {
            if *k == key {
                return p.clone();
            }
        }
        let p = Arc::new(Precomp::from_rgb8(img));
        *g = Some((key, p.clone()));
        p
    }

    pub fn clear_cache(&self) {
        *lock(&self.cache) = None;
        *lock(&self.matte_cache) = None;
        *lock(&self.skin_cache) = None;
    }

    /// 奶油肌用的皮肤遮罩（调试 / 导出用）。
    pub fn skin_masks(
        &self,
        img: &image::RgbImage,
        faces: &[FaceKeyPoints],
        body: bool,
    ) -> anyhow::Result<Arc<SkinMasks>> {
        let pre = self.precomp(img);
        if body {
            pre.set_matte(self.person_matte(img)?);
            pre.set_skin_prob(self.skin_prob(img)?);
        }
        Ok(pre.skin_masks(faces, body))
    }

    /// 完整修图；`faces` 为 `detect_faces` 的结果（可缓存，滑块调整时不必重检）。
    pub fn retouch(
        &self,
        img: &image::RgbImage,
        faces: &[FaceKeyPoints],
        p: &RetouchParams,
    ) -> image::RgbImage {
        self.retouch_f32(img, faces, p).to_rgb8()
    }

    pub fn retouch_f32(
        &self,
        img: &image::RgbImage,
        faces: &[FaceKeyPoints],
        p: &RetouchParams,
    ) -> ImgF32 {
        let pre = self.precomp(img);
        if p.smooth_mode == SmoothMode::Cream && p.body_skin && pre.matte().is_none() {
            match self.person_matte(img) {
                Ok(m) => pre.set_matte(m),
                Err(e) => eprintln!("warning: person matting failed: {e}"),
            }
        }
        if p.smooth_mode == SmoothMode::Cream && p.body_skin && pre.skin_prob().is_none() {
            match self.skin_prob(img) {
                Ok(m) => pre.set_skin_prob(m),
                Err(e) => eprintln!("warning: skin segmentation failed: {e}"),
            }
        }
        if p.smooth_mode == SmoothMode::Cream && p.ai_blemish > 0.0 && !faces.is_empty() {
            if let Err(e) = self.ai_patches(&pre, faces) {
                eprintln!("warning: AI blemish removal failed: {e}");
            }
        }
        retouch_with(&pre, faces, p)
    }
}

/// 图像内容哈希：尺寸 + 步进采样（每 997 个字节取一个）。
pub fn image_key(img: &image::RgbImage) -> u64 {
    let raw = img.as_raw();
    let mut h: u64 = 0xcbf29ce484222325;
    let mut feed = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    };
    for b in img
        .width()
        .to_le_bytes()
        .iter()
        .chain(img.height().to_le_bytes().iter())
    {
        feed(*b);
    }
    let mut i = 0;
    while i < raw.len() {
        feed(raw[i]);
        i += 997;
    }
    feed(raw.len() as u8);
    h
}
