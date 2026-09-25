//! 人脸检测与关键点：UltraFace 检测 + 2d106det / Face Mesh 关键点 → 语义结构体。

pub mod attribute;
pub mod crop;
pub mod detector;
pub mod lm_2d106;
pub mod lm_facemesh;
pub mod matting;
pub mod ort_util;
pub mod parsing;
pub mod semantic;
pub mod skinseg;

pub use attribute::{Gender, GenderAge};
pub use matting::PersonMatting;
pub use parsing::{FaceParser, ParseMap};

pub use detector::UltraFace;
pub use lm_2d106::Lm2d106;
pub use lm_facemesh::FaceMesh;
pub use semantic::{FaceBox, FaceKeyPoints, LandmarkModel};
