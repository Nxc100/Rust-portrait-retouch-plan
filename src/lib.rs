//! # portrait-retouch
//!
//! 纯 Rust（CPU + rayon）人像修图：磨皮 / 美白 / 提亮饱和 / 瘦脸 / 大眼 / 瘦鼻 / LUT 风格滤镜。
//!
//! 算法来源（详见 `doc/rust_portrait_retouch_plan.md` 与 `doc/analysis/`）：
//! - 磨皮方案 A（`SmoothMode::Faithful`）：忠实还原美狐 Demo 使用的 `BBGPUImageBeautifyFilter`
//!   （GPUImage 双边 + Sobel + 肤色规则组合 + log 提亮 + HSB）。
//! - 磨皮方案 B（`SmoothMode::FreqSep`）：YUCIHighPassSkinSmoothing 的高反差保留磨皮。
//! - 磨皮方案 C（`SmoothMode::GpuPixel`）：pixpark/gpupixel 的均值/方差自适应磨皮。
//! - 形变：美狐 `GLImageFaceChangeFilter` 的 warpPositionToUse1 / adjust_eye / newNarrowNose_2
//!   在像素空间的等价移植；另提供 gpupixel 风格的 curveWarp / enlargeEye。
//! - 风格：GPUImage 512×512 查找图、.cube 3D LUT、带空间遮罩的 LUT（曲线滤镜包的暗角部分）。
//! - 人脸：UltraFace RFB-320 检测；InsightFace 2d106det（开发期）或 MediaPipe Face Mesh（发布期）关键点，
//!   统一映射为语义结构体 [`FaceKeyPoints`]，上层不依赖具体索引。
//!
//! 典型用法：
//! ```no_run
//! use portrait_retouch::{Engine, EngineConfig, RetouchParams};
//! let engine = Engine::new(EngineConfig::default()).unwrap();
//! let img = image::open("photo.jpg").unwrap().to_rgb8();
//! let faces = engine.detect_faces(&img).unwrap();
//! let params = RetouchParams { smooth: 0.5, thin_face: 0.5, big_eye: 0.3, ..Default::default() };
//! let out = engine.retouch(&img, &faces, &params);
//! out.save("out.jpg").unwrap();
//! ```

pub mod batch;
pub mod buffer;
pub mod color;
pub mod debug;
pub mod engine;
pub mod face;
pub mod geom;
pub mod photo;
pub mod pipeline;
pub mod preset;
pub mod skin;
mod sync;
pub mod ui_map;
pub mod warp;

pub use buffer::{GrayF32, ImgF32};
pub use color::curve::Curve256;
pub use color::lut3d::Lut3D;
pub use engine::{Engine, EngineConfig, LandmarkKind};
pub use face::attribute::Gender;
pub use face::semantic::{FaceBox, FaceKeyPoints, LandmarkModel};
pub use geom::P;
pub use photo::{Photo, PhotoMetadata};
pub use pipeline::{
    retouch_impl, MaskedLutOp, Precomp, RetouchParams, SmoothMode, StyleFilter, WhitenMode,
};
pub use preset::{GenderWarp, Preset};
pub use skin::ai_blemish::{AiBlemish, AiPatches};
pub use skin::cream::CreamParams;
pub use warp::face_warp::{ReshapeStyle, WarpCoefficients, WarpParams};
