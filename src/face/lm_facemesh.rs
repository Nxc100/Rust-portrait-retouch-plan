//! MediaPipe Face Mesh（Apache-2.0，发布期），ONNX 来自 PINTO0309/facemesh_onnx_tensorrt
//! （`face_mesh_Nx3x192x192_post.onnx`，468 点，含后处理）。
//!
//! 模型 I/O：
//! - `input [N,3,192,192]` RGB 0..1；`crop_x1/crop_y1/crop_width/crop_height [N,1]` int32；
//! - `score [N,1]`（logit），`final_landmarks [N,468,3]` int32 =
//!   `int(lm · crop_wh / 192 + 0.5 + crop_xy)`。
//!   我们传 crop = (0, 0, 192·K, 192·K)，K = 256，再除以 K 得到 1/256 像素精度的 192 空间坐标。
//!
//! ROI 与 MediaPipe 一致：以框为中心、1.5× 放大的正方形；第一遍不旋转，第二遍按两眼连线
//! （33 → 263）旋转对齐后重新推理（MediaPipe `face_landmark_landmarks_to_roi` 的做法）。

use super::crop::{to_nchw, warp_affine_rgb8};
use super::semantic::{FaceBox, FaceKeyPoints, LandmarkModel};
use crate::geom::{centroid, Affine, P};
use std::path::Path;

pub const INPUT_SIZE: usize = 192;
pub const NUM_POINTS: usize = 468;
const K: f32 = 256.0;

/// 左眼（图像左，MediaPipe "RIGHT_EYE"）轮廓 16 点
pub const LEFT_EYE_RING: [usize; 16] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
];
/// 右眼（图像右，MediaPipe "LEFT_EYE"）轮廓 16 点
pub const RIGHT_EYE_RING: [usize; 16] = [
    263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
];
/// 轮廓：左太阳穴 → 下巴 → 右太阳穴
pub const CONTOUR: [usize; 21] = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361,
    323, 454,
];
/// 额头：右太阳穴上方 → 头顶 → 左太阳穴上方
pub const FOREHEAD: [usize; 15] = [
    356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127,
];

pub struct FaceMesh {
    session: ort::session::Session,
    names: [String; 5],
    /// 是否做第二遍旋转对齐推理
    pub refine_rotation: bool,
}

impl FaceMesh {
    pub fn new(model: &Path, threads: usize) -> anyhow::Result<Self> {
        let session = super::ort_util::make_session(model, threads)?;
        let mut names: [String; 5] = [
            "input".into(),
            "crop_x1".into(),
            "crop_y1".into(),
            "crop_width".into(),
            "crop_height".into(),
        ];
        let actual: Vec<String> = session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if actual.len() == 5 {
            for (i, n) in actual.into_iter().enumerate() {
                names[i] = n;
            }
        }
        Ok(Self {
            session,
            names,
            refine_rotation: true,
        })
    }

    fn run_crop(
        &mut self,
        img: &image::RgbImage,
        center: P,
        side: f32,
        rot: f32,
    ) -> anyhow::Result<(Vec<P>, f32)> {
        let scale = INPUT_SIZE as f32 / side.max(1.0);
        let m = Affine::crop(
            P::new(center.x - 0.5, center.y - 0.5),
            scale,
            rot,
            INPUT_SIZE as f32,
        );
        let hwc = warp_affine_rgb8(img, &m, INPUT_SIZE);
        let input = to_nchw(&hwc, INPUT_SIZE, 1.0 / 255.0, 0.0);
        let t_in = ort::value::Tensor::from_array(input)?;
        let zero = ndarray::Array2::<i32>::zeros((1, 1));
        let wh = ndarray::Array2::<i32>::from_elem((1, 1), (INPUT_SIZE as f32 * K) as i32);
        let outputs = self.session.run(ort::inputs![
            self.names[0].as_str() => t_in,
            self.names[1].as_str() => ort::value::Tensor::from_array(zero.clone())?,
            self.names[2].as_str() => ort::value::Tensor::from_array(zero)?,
            self.names[3].as_str() => ort::value::Tensor::from_array(wh.clone())?,
            self.names[4].as_str() => ort::value::Tensor::from_array(wh)?,
        ])?;
        let (_s, score) = outputs["score"].try_extract_tensor::<f32>()?;
        let (_l, lm) = outputs["final_landmarks"].try_extract_tensor::<i32>()?;
        anyhow::ensure!(
            lm.len() >= NUM_POINTS * 3,
            "unexpected landmark length {}",
            lm.len()
        );
        let inv = m.inverse();
        let pts = (0..NUM_POINTS)
            .map(|i| {
                let x = lm[i * 3] as f32 / K;
                let y = lm[i * 3 + 1] as f32 / K;
                let p = inv.apply(P::new(x, y));
                P::new(p.x + 0.5, p.y + 0.5)
            })
            .collect();
        let logit = score.first().copied().unwrap_or(0.0);
        let prob = 1.0 / (1.0 + (-logit).exp());
        Ok((pts, prob))
    }

    /// 返回 468 个原图坐标点与置信度。
    pub fn raw_points(
        &mut self,
        img: &image::RgbImage,
        face_box: &FaceBox,
    ) -> anyhow::Result<(Vec<P>, f32)> {
        let side = face_box.width().max(face_box.height()) * 1.5;
        let (pts, score) = self.run_crop(img, face_box.center(), side, 0.0)?;
        if !self.refine_rotation {
            return Ok((pts, score));
        }
        // 第二遍：按关键点重新计算 ROI（旋转对齐 + 1.5× 正方形）
        let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in &pts {
            minx = minx.min(p.x);
            miny = miny.min(p.y);
            maxx = maxx.max(p.x);
            maxy = maxy.max(p.y);
        }
        let center = P::new(0.5 * (minx + maxx), 0.5 * (miny + maxy));
        let side2 = (maxx - minx).max(maxy - miny) * 1.5;
        let eye = pts[263].sub(pts[33]);
        let angle = eye.y.atan2(eye.x); // 眼线相对水平的角度
        let (pts2, score2) = self.run_crop(img, center, side2, -angle)?;
        Ok((pts2, score2))
    }

    pub fn to_semantic(pts: &[P], face_box: &FaceBox, score: f32) -> FaceKeyPoints {
        let p = |i: usize| pts[i];
        let left_ring: Vec<P> = LEFT_EYE_RING.iter().map(|&i| pts[i]).collect();
        let right_ring: Vec<P> = RIGHT_EYE_RING.iter().map(|&i| pts[i]).collect();
        let contour: Vec<P> = CONTOUR.iter().map(|&i| pts[i]).collect();
        let forehead: Vec<P> = FOREHEAD.iter().map(|&i| pts[i]).collect();
        let eye_outer_l = p(33);
        let eye_outer_r = p(263);
        let nose_tip = p(1);
        FaceKeyPoints {
            bbox: *face_box,
            pupil_l: centroid(&left_ring),
            pupil_r: centroid(&right_ring),
            eye_outer_l,
            eye_inner_l: p(133),
            eye_inner_r: p(362),
            eye_outer_r,
            eye_top_l: p(159),
            eye_bot_l: p(145),
            eye_top_r: p(386),
            eye_bot_r: p(374),
            nose_bridge_top: p(168),
            nose_tip,
            nose_bottom: p(2),
            nostril_l: p(98),
            nostril_r: p(327),
            nose_wing_l: p(129),
            nose_wing_r: p(358),
            chin: p(152),
            jaw_l: [p(132), p(172), p(150)],
            jaw_r: [p(361), p(397), p(379)],
            mouth_l: p(61),
            mouth_r: p(291),
            contour,
            forehead,
            yaw_deg: FaceKeyPoints::estimate_yaw_deg(eye_outer_l, eye_outer_r, nose_tip),
            model: "facemesh".into(),
            raw: pts.to_vec(),
            score,
            gender: None,
            age: None,
            parse: None,
        }
    }
}

impl LandmarkModel for FaceMesh {
    fn name(&self) -> &'static str {
        "facemesh"
    }
    fn detect(
        &mut self,
        img: &image::RgbImage,
        face_box: &FaceBox,
    ) -> anyhow::Result<FaceKeyPoints> {
        let (pts, score) = self.raw_points(img, face_box)?;
        Ok(Self::to_semantic(&pts, face_box, score))
    }
}
