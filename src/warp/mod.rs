//! 形变：瘦脸 / 大眼 / 瘦鼻（反向映射 + 双线性采样）。

pub mod apply;
pub mod face_warp;
pub mod pinch;

pub use apply::apply_warps;
pub use face_warp::{FaceWarp, ReshapeStyle, Step, WarpCoefficients, WarpParams};
pub use pinch::{curve_warp, enlarge, enlarge_gpupixel, pinch};
