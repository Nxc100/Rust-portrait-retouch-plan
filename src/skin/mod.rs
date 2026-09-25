//! 磨皮 / 美白 / 皮肤遮罩。

pub mod ai_blemish;
pub mod bilateral;
pub mod blemish;
pub mod cream;
pub mod edge;
pub mod gaussian;
pub mod guided;
pub mod heal;
pub mod mask;
pub mod masks;
pub mod smooth_faithful;
pub mod smooth_freqsep;
pub mod smooth_gpupixel;
pub mod whiten;

pub use bilateral::bilateral_gpuimage;
pub use edge::sobel_edge;
pub use mask::{face_mask_fullres, is_skin_rgb, skin_color_mask};
pub use smooth_faithful::{combine_bb, precompute_faithful, smooth_faithful, FaithfulPrecomp};
pub use smooth_freqsep::{precompute_freqsep, smooth_freqsep, FreqSepPrecomp};
pub use smooth_gpupixel::{precompute_gpupixel, smooth_gpupixel, GpuPixelPrecomp};

/// 方案 A / C 的工作副本短边（美狐 Demo 为 720×1280 视频流）。
pub const WORK_SHORT_SIDE_A: f32 = 720.0;
/// 方案 B 的工作副本短边（YUCI 默认 radius=8 相对 ~1000 px 短边）。
pub const WORK_SHORT_SIDE_B: f32 = 1000.0;
