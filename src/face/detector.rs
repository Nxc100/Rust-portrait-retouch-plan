//! UltraFace `version-RFB-320.onnx` 人脸检测（MIT）。
//!
//! 输入 `1×3×240×320` RGB，`(x − 127) / 128`；输出 `scores 1×4420×2`、`boxes 1×4420×4`（归一化）。
//! 后处理：阈值 → NMS（IoU 0.3）→ 还原到原图像素 → 按面积降序。

use super::semantic::FaceBox;
use std::path::Path;

pub struct UltraFace {
    session: ort::session::Session,
    input_name: String,
    pub score_threshold: f32,
    pub iou_threshold: f32,
}

pub const INPUT_W: u32 = 320;
pub const INPUT_H: u32 = 240;

impl UltraFace {
    pub fn new(model: &Path, threads: usize) -> anyhow::Result<Self> {
        let session = super::ort_util::make_session(model, threads)?;
        let input_name = session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .find(|n| n == "input")
            .or_else(|| session.inputs().first().map(|o| o.name().to_string()))
            .ok_or_else(|| anyhow::anyhow!("model has no inputs"))?;
        Ok(Self {
            session,
            input_name,
            score_threshold: 0.7,
            iou_threshold: 0.3,
        })
    }

    pub fn detect(&mut self, img: &image::RgbImage) -> anyhow::Result<Vec<FaceBox>> {
        let resized =
            image::imageops::resize(img, INPUT_W, INPUT_H, image::imageops::FilterType::Triangle);
        let mut input = ndarray::Array4::<f32>::zeros((1, 3, INPUT_H as usize, INPUT_W as usize));
        for (x, y, p) in resized.enumerate_pixels() {
            for c in 0..3 {
                input[[0, c, y as usize, x as usize]] = (p[c] as f32 - 127.0) / 128.0;
            }
        }
        let tensor = ort::value::Tensor::from_array(input)?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])?;
        let (sshape, scores) = outputs["scores"].try_extract_tensor::<f32>()?;
        let (_bshape, boxes) = outputs["boxes"].try_extract_tensor::<f32>()?;
        let n = sshape[1] as usize;
        let (w, h) = (img.width() as f32, img.height() as f32);
        let mut cands = Vec::new();
        for i in 0..n {
            let s = scores[i * 2 + 1];
            if s < self.score_threshold {
                continue;
            }
            let b = &boxes[i * 4..i * 4 + 4];
            cands.push(FaceBox {
                x1: (b[0] * w).clamp(0.0, w),
                y1: (b[1] * h).clamp(0.0, h),
                x2: (b[2] * w).clamp(0.0, w),
                y2: (b[3] * h).clamp(0.0, h),
                score: s,
            });
        }
        let mut kept = nms(cands, self.iou_threshold);
        kept.sort_by(|a, b| {
            b.area()
                .partial_cmp(&a.area())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(kept)
    }
}

/// 硬 NMS：按分数降序，抑制 IoU 高于阈值的候选。
pub fn nms(mut cands: Vec<FaceBox>, iou_threshold: f32) -> Vec<FaceBox> {
    cands.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<FaceBox> = Vec::new();
    for c in cands {
        if kept.iter().all(|k| k.iou(&c) <= iou_threshold) {
            kept.push(c);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nms_suppresses_overlaps() {
        let a = FaceBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            score: 0.9,
        };
        let b = FaceBox {
            x1: 1.0,
            y1: 1.0,
            x2: 11.0,
            y2: 11.0,
            score: 0.8,
        };
        let c = FaceBox {
            x1: 50.0,
            y1: 50.0,
            x2: 60.0,
            y2: 60.0,
            score: 0.95,
        };
        let k = nms(vec![a, b, c], 0.3);
        assert_eq!(k.len(), 2);
        assert!((k[0].score - 0.95).abs() < 1e-6);
    }
}
