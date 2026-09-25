//! InsightFace `2d106det.onnx`（开发期，模型仅限非商用）。
//!
//! 前后处理与官方 `model_zoo/landmark.py` 一致：
//! - 裁剪：center = 框中心，scale = 192 / (max(w,h)·1.5)，无旋转，`warpAffine` 到 192×192；
//! - 输入 RGB，像素值 0..255（模型图内含 Sub/Mul，mean = 0，std = 1）；
//! - 输出 `fc1[212]` → (106, 2)，`pred = (pred + 1) · 96` → 逆仿射回原图。
//!
//! 索引语义（P2 阶段用 `retouch detect` 可视化并与 Face Mesh 逐点对照投票后确定，见 doc/analysis）：
//! - 轮廓 33 点从左太阳穴到右太阳穴的顺序为 [`CONTOUR_ORDER`]（0 = 下巴尖，1 = 左太阳穴，17 = 右太阳穴）；
//! - 左眼 33..42（35 外眼角、39 内眼角、40 上睑、33 下睑、34/38 眼中心），右眼 87..96（93 外、89 内、94 上、87 下、88/92 中心）；
//! - 鼻 72..86（72 鼻梁顶、73/74 鼻梁、86 鼻尖上方、80 鼻底中、78/79/85/84 鼻底、77/83 鼻翼、76/82 鼻侧、75/81 鼻梁侧）；
//! - 嘴 52..71（52 左角、61 右角）；左眉 43..51；右眉 97..105。

use super::crop::{to_nchw, warp_affine_rgb8};
use super::semantic::{FaceBox, FaceKeyPoints, LandmarkModel};
use crate::geom::{Affine, P};
use std::path::Path;

pub const INPUT_SIZE: usize = 192;
pub const NUM_POINTS: usize = 106;

/// Face++ 轮廓索引 k（0 左太阳穴 … 16 下巴 … 32 右太阳穴）→ 2d106det 索引。
pub const CONTOUR_ORDER: [usize; 33] = [
    1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18, 32, 31,
    30, 29, 28, 27, 26, 25, 17,
];

pub struct Lm2d106 {
    session: ort::session::Session,
    input_name: String,
}

impl Lm2d106 {
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

    /// 返回 106 个原图坐标点（像素中心制）。
    pub fn raw_points(
        &mut self,
        img: &image::RgbImage,
        face_box: &FaceBox,
    ) -> anyhow::Result<Vec<P>> {
        let w = face_box.width();
        let h = face_box.height();
        let center = face_box.center();
        let scale = INPUT_SIZE as f32 / (w.max(h).max(1.0) * 1.5);
        // 索引制：中心坐标先减 0.5
        let m = Affine::crop(
            P::new(center.x - 0.5, center.y - 0.5),
            scale,
            0.0,
            INPUT_SIZE as f32,
        );
        let hwc = warp_affine_rgb8(img, &m, INPUT_SIZE);
        let input = to_nchw(&hwc, INPUT_SIZE, 1.0, 0.0);
        let tensor = ort::value::Tensor::from_array(input)?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])?;
        let (_shape, pred) = outputs[0].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            pred.len() >= NUM_POINTS * 2,
            "unexpected output length {}",
            pred.len()
        );
        let inv = m.inverse();
        let half = (INPUT_SIZE / 2) as f32;
        let pts = (0..NUM_POINTS)
            .map(|i| {
                let x = (pred[i * 2] + 1.0) * half;
                let y = (pred[i * 2 + 1] + 1.0) * half;
                let p = inv.apply(P::new(x, y));
                P::new(p.x + 0.5, p.y + 0.5)
            })
            .collect();
        Ok(pts)
    }

    /// 由 106 点构造语义结构体。
    pub fn to_semantic(pts: &[P], face_box: &FaceBox) -> FaceKeyPoints {
        let p = |i: usize| pts[i];
        let contour: Vec<P> = CONTOUR_ORDER.iter().map(|&i| pts[i]).collect();
        let pupil_l = p(34).mid(p(38));
        let pupil_r = p(88).mid(p(92));
        let chin = p(0);
        let nose_bridge_top = p(72);
        let nose_tip = p(86).mid(p(80));
        let eye_outer_l = p(35);
        let eye_outer_r = p(93);
        let dir_up = nose_bridge_top.sub(chin).norm();
        // 眉毛最高点（沿 dir_up 投影最大）
        let brow_top = (43..52)
            .chain(97..106)
            .map(|i| pts[i])
            .max_by(|a, b| {
                a.dot(dir_up)
                    .partial_cmp(&b.dot(dir_up))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(nose_bridge_top);
        let face_h = brow_top.sub(chin).dot(dir_up).abs();
        let forehead =
            FaceKeyPoints::synth_forehead(contour[0], contour[32], brow_top, dir_up, face_h, 15);
        let yaw_deg = FaceKeyPoints::estimate_yaw_deg(eye_outer_l, eye_outer_r, nose_tip);
        FaceKeyPoints {
            bbox: *face_box,
            pupil_l,
            pupil_r,
            eye_outer_l,
            eye_inner_l: p(39),
            eye_inner_r: p(89),
            eye_outer_r,
            eye_top_l: p(40),
            eye_bot_l: p(33),
            eye_top_r: p(94),
            eye_bot_r: p(87),
            nose_bridge_top,
            nose_tip,
            nose_bottom: p(80),
            nostril_l: p(79),
            nostril_r: p(85),
            nose_wing_l: p(77),
            nose_wing_r: p(83),
            chin,
            jaw_l: [contour[4], contour[9], contour[13]],
            jaw_r: [contour[28], contour[23], contour[19]],
            mouth_l: p(52),
            mouth_r: p(61),
            contour,
            forehead,
            yaw_deg,
            model: "2d106det".into(),
            raw: pts.to_vec(),
            score: face_box.score,
            gender: None,
            age: None,
            parse: None,
        }
    }
}

impl LandmarkModel for Lm2d106 {
    fn name(&self) -> &'static str {
        "2d106det"
    }
    fn detect(
        &mut self,
        img: &image::RgbImage,
        face_box: &FaceBox,
    ) -> anyhow::Result<FaceKeyPoints> {
        let pts = self.raw_points(img, face_box)?;
        Ok(Self::to_semantic(&pts, face_box))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contour_order_is_a_permutation_of_0_32() {
        let mut v = CONTOUR_ORDER.to_vec();
        v.sort_unstable();
        assert_eq!(v, (0..33).collect::<Vec<_>>());
        assert_eq!(CONTOUR_ORDER[16], 0);
        assert_eq!(CONTOUR_ORDER[4], 12);
        assert_eq!(CONTOUR_ORDER[28], 28);
    }
}
