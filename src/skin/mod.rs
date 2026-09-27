//! 磨皮 / 美白 / 皮肤遮罩。

pub mod ai_blemish;
pub mod bilateral;
pub mod blemish;
pub mod continuity;
pub mod cream;
pub mod edge;
pub mod gaussian;
pub mod guided;
pub mod heal;
pub mod mask;
pub mod masks;
pub mod morph;
pub mod neck;
pub mod smooth_faithful;
pub mod smooth_freqsep;
pub mod smooth_gpupixel;
pub mod spill;
pub mod teeth;
pub mod whiten;

pub use bilateral::bilateral_gpuimage;
pub use edge::sobel_edge;
pub use mask::{face_mask_fullres, is_skin_rgb, skin_color_mask};
pub use smooth_faithful::{combine_bb, precompute_faithful, smooth_faithful, FaithfulPrecomp};
pub use smooth_freqsep::{precompute_freqsep, smooth_freqsep, FreqSepPrecomp};
pub use smooth_gpupixel::{precompute_gpupixel, smooth_gpupixel, GpuPixelPrecomp};

/// 平滑阶跃：`x` 从 `e0` 到 `e1` 时由 0 平滑过渡到 1（区间外钳制）。
pub(crate) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 方案 A / C 的工作副本短边（美狐 Demo 为 720×1280 视频流）。
pub const WORK_SHORT_SIDE_A: f32 = 720.0;
/// 方案 B 的工作副本短边（YUCI 默认 radius=8 相对 ~1000 px 短边）。
pub const WORK_SHORT_SIDE_B: f32 = 1000.0;
