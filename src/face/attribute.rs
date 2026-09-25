//! InsightFace `genderage.onnx`（buffalo_l，模型仅限非商用）：性别 / 年龄，用于预设按性别选择美型强度。
//!
//! 与 `model_zoo/attribute.py` 一致：裁剪 scale = 96 / (max(w,h)·1.5)，RGB 0..255；输出 `[1,3]`：
//! `gender = argmax(pred[0..2])`（0 女 1 男），`age = pred[2] × 100`。

use super::crop::{to_nchw, warp_affine_rgb8};
use super::semantic::FaceBox;
use crate::geom::{Affine, P};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const INPUT_SIZE: usize = 96;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gender {
    Female,
    Male,
}

pub struct GenderAge {
    session: ort::session::Session,
    input_name: String,
}

impl GenderAge {
    pub fn new(model: &Path, threads: usize) -> anyhow::Result<Self> {
        let session = super::ort_util::make_session(model, threads)?;
        let input_name = session
            .inputs()
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "data".into());
        Ok(Self {
            session,
            input_name,
        })
    }

    /// 返回 (性别, 年龄, 性别置信度 0.5..1)。
    pub fn predict(
        &mut self,
        img: &image::RgbImage,
        face_box: &FaceBox,
    ) -> anyhow::Result<(Gender, f32, f32)> {
        let side = face_box.width().max(face_box.height()).max(8.0) * 1.5;
        let c = face_box.center();
        let scale = INPUT_SIZE as f32 / side;
        let m = Affine::crop(P::new(c.x - 0.5, c.y - 0.5), scale, 0.0, INPUT_SIZE as f32);
        let hwc = warp_affine_rgb8(img, &m, INPUT_SIZE);
        let input = to_nchw(&hwc, INPUT_SIZE, 1.0, 0.0);
        let tensor = ort::value::Tensor::from_array(input)?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])?;
        let (_shape, pred) = outputs[0].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            pred.len() >= 3,
            "unexpected genderage output length {}",
            pred.len()
        );
        let (f, mm) = (pred[0], pred[1]);
        let gender = if mm > f { Gender::Male } else { Gender::Female };
        let conf = 1.0 / (1.0 + (-(mm - f).abs()).exp());
        Ok((gender, pred[2] * 100.0, conf))
    }
}
