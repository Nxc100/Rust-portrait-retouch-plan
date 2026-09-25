//! 语义关键点结构体：Rust 实现不依赖具体模型索引，各模型自行提供映射。

use crate::face::attribute::Gender;
use crate::face::parsing::ParseMap;
use crate::geom::{centroid, P};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// 人脸框（像素坐标）。
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct FaceBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl FaceBox {
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
    pub fn center(&self) -> P {
        P::new(0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))
    }
    pub fn iou(&self, o: &FaceBox) -> f32 {
        let ix1 = self.x1.max(o.x1);
        let iy1 = self.y1.max(o.y1);
        let ix2 = self.x2.min(o.x2);
        let iy2 = self.y2.min(o.y2);
        let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
        inter / (self.area() + o.area() - inter + 1e-5)
    }
    pub fn scaled(&self, s: f32) -> FaceBox {
        FaceBox {
            x1: self.x1 * s,
            y1: self.y1 * s,
            x2: self.x2 * s,
            y2: self.y2 * s,
            score: self.score,
        }
    }
}

/// 语义关键点（像素中心制坐标）。"左 / 右"以图像左右为准（非受试者左右）。
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FaceKeyPoints {
    pub bbox: FaceBox,
    /// 图像左侧眼的瞳孔中心（Face++ 74）
    pub pupil_l: P,
    /// 图像右侧（77）
    pub pupil_r: P,
    /// 左眼外眼角（52）
    pub eye_outer_l: P,
    /// 左眼内眼角（55）
    pub eye_inner_l: P,
    /// 右眼内眼角（58）
    pub eye_inner_r: P,
    /// 右眼外眼角（61）
    pub eye_outer_r: P,
    pub eye_top_l: P,
    pub eye_bot_l: P,
    pub eye_top_r: P,
    pub eye_bot_r: P,
    /// 鼻梁顶点（43）
    pub nose_bridge_top: P,
    /// 鼻尖（46）
    pub nose_tip: P,
    /// 鼻底中点（49，鼻小柱）
    pub nose_bottom: P,
    /// 左 / 右鼻孔（48 / 50）
    pub nostril_l: P,
    pub nostril_r: P,
    /// 左 / 右鼻翼（82 / 83）
    pub nose_wing_l: P,
    pub nose_wing_r: P,
    /// 下巴尖（16）
    pub chin: P,
    /// 左侧 [颧下 4, 中下颌 9, 下颌角 13]
    pub jaw_l: [P; 3],
    /// 右侧 [28, 23, 19]
    pub jaw_r: [P; 3],
    pub mouth_l: P,
    pub mouth_r: P,
    /// 完整轮廓：左太阳穴 → 下巴 → 右太阳穴
    pub contour: Vec<P>,
    /// 额头补点：右太阳穴 → 头顶 → 左太阳穴（与 contour 首尾相接成闭合多边形）
    pub forehead: Vec<P>,
    /// 粗略偏航角（度，正值为脸朝图像右方）
    pub yaw_deg: f32,
    /// 关键点模型名
    pub model: String,
    /// 模型原始点（调试 / 可视化用）
    pub raw: Vec<P>,
    /// 关键点置信度（模型给出时）
    pub score: f32,
    /// 性别 / 年龄（有属性模型时）
    pub gender: Option<Gender>,
    pub age: Option<f32>,
    /// 人脸解析结果（有解析模型时；不序列化）
    #[serde(skip)]
    pub parse: Option<Arc<ParseMap>>,
}

impl FaceKeyPoints {
    /// 瞳距（像素）。
    pub fn eye_distance(&self) -> f32 {
        self.pupil_l.dist(self.pupil_r)
    }
    /// 姿态稳健的脸部尺度（瞳距当量）：正脸时 = 瞳距；侧脸时瞳距按 cos(yaw) 缩短，
    /// 改用鼻梁顶—下巴距离 / 2 兜底（正脸样张上该距离约为瞳距的 1.86–1.95 倍）。
    /// 例：X04 新郎 yaw 62°，瞳距 70.7 px，鼻梁—下巴 224.5 px → 112 px。
    pub fn scale_distance(&self) -> f32 {
        self.eye_distance()
            .max(self.nose_bridge_top.dist(self.chin) / 2.0)
    }
    /// 图像右方向（左瞳 → 右瞳）。
    pub fn dir_right(&self) -> P {
        self.pupil_r.sub(self.pupil_l).norm()
    }
    /// 上方向（下巴 → 鼻梁顶点）。
    pub fn dir_up(&self) -> P {
        self.nose_bridge_top.sub(self.chin).norm()
    }
    pub fn left_eye_width(&self) -> f32 {
        self.eye_outer_l.dist(self.eye_inner_l)
    }
    pub fn right_eye_width(&self) -> f32 {
        self.eye_outer_r.dist(self.eye_inner_r)
    }

    /// 所有坐标乘以 `s`（预览缩略图用）。
    pub fn scaled(&self, s: f32) -> FaceKeyPoints {
        let m = |p: P| p.mul(s);
        FaceKeyPoints {
            bbox: self.bbox.scaled(s),
            pupil_l: m(self.pupil_l),
            pupil_r: m(self.pupil_r),
            eye_outer_l: m(self.eye_outer_l),
            eye_inner_l: m(self.eye_inner_l),
            eye_inner_r: m(self.eye_inner_r),
            eye_outer_r: m(self.eye_outer_r),
            eye_top_l: m(self.eye_top_l),
            eye_bot_l: m(self.eye_bot_l),
            eye_top_r: m(self.eye_top_r),
            eye_bot_r: m(self.eye_bot_r),
            nose_bridge_top: m(self.nose_bridge_top),
            nose_tip: m(self.nose_tip),
            nose_bottom: m(self.nose_bottom),
            nostril_l: m(self.nostril_l),
            nostril_r: m(self.nostril_r),
            nose_wing_l: m(self.nose_wing_l),
            nose_wing_r: m(self.nose_wing_r),
            chin: m(self.chin),
            jaw_l: [m(self.jaw_l[0]), m(self.jaw_l[1]), m(self.jaw_l[2])],
            jaw_r: [m(self.jaw_r[0]), m(self.jaw_r[1]), m(self.jaw_r[2])],
            mouth_l: m(self.mouth_l),
            mouth_r: m(self.mouth_r),
            contour: self.contour.iter().map(|p| m(*p)).collect(),
            forehead: self.forehead.iter().map(|p| m(*p)).collect(),
            yaw_deg: self.yaw_deg,
            model: self.model.clone(),
            raw: self.raw.iter().map(|p| m(*p)).collect(),
            score: self.score,
            gender: self.gender,
            age: self.age,
            parse: self.parse.as_ref().map(|p| Arc::new(p.scaled(s))),
        }
    }

    /// 由鼻尖相对两外眼角的位置估计偏航角（粗略，用于侧脸时衰减形变）。
    pub fn estimate_yaw_deg(eye_outer_l: P, eye_outer_r: P, nose_tip: P) -> f32 {
        let axis = eye_outer_r.sub(eye_outer_l);
        let len = axis.len();
        if len < 1e-3 {
            return 0.0;
        }
        let r = nose_tip.sub(eye_outer_l).dot(axis) / (len * len);
        ((r - 0.5) * 2.0).clamp(-1.0, 1.0).asin().to_degrees()
    }

    /// 合成额头补点：从右太阳穴到左太阳穴的半椭圆弧，最高点在眉毛以上 `0.6 × 脸高`。
    /// `brow_top` 为眉毛最高点，`face_h` 为下巴到眉毛的距离。
    pub fn synth_forehead(
        temple_l: P,
        temple_r: P,
        brow_top: P,
        dir_up: P,
        face_h: f32,
        n: usize,
    ) -> Vec<P> {
        let mut out = Vec::with_capacity(n);
        let base_mid = temple_l.mid(temple_r);
        let brow_h = brow_top.sub(base_mid).dot(dir_up).max(0.0);
        let peak = brow_h + 0.6 * face_h;
        for i in 0..n {
            let t = (i as f32 + 0.5) / n as f32;
            let base = temple_r.lerp(temple_l, t);
            let lift = (std::f32::consts::PI * t).sin() * peak;
            out.push(base.add(dir_up.mul(lift)));
        }
        out
    }

    /// 附录 A 的映射正确性检查，返回违反的规则列表（空表示通过）。
    pub fn sanity_check(&self) -> Vec<String> {
        let mut v = Vec::new();
        if self.pupil_r.x <= self.pupil_l.x {
            v.push("pupil_r.x > pupil_l.x".into());
        }
        if !(self.eye_outer_l.x < self.eye_inner_l.x
            && self.eye_inner_l.x < self.eye_inner_r.x
            && self.eye_inner_r.x < self.eye_outer_r.x)
        {
            v.push("eye_outer_l.x < eye_inner_l.x < eye_inner_r.x < eye_outer_r.x".into());
        }
        if !(self.chin.y > self.nose_tip.y && self.nose_tip.y > self.nose_bridge_top.y) {
            v.push("chin.y > nose_tip.y > nose_bridge_top.y".into());
        }
        let ratio = self.left_eye_width() / self.right_eye_width().max(1e-6);
        if self.yaw_deg.abs() < 15.0 && !(0.8 < ratio && ratio < 1.25) {
            v.push(format!(
                "eye width ratio {ratio:.2} outside (0.8, 1.25) on frontal face"
            ));
        }
        for i in 0..3 {
            if !(self.jaw_l[i].x < self.chin.x && self.chin.x < self.jaw_r[i].x) {
                v.push(format!("jaw_l[{i}].x < chin.x < jaw_r[{i}].x"));
            }
        }
        if !(self.jaw_l[0].y < self.jaw_l[1].y && self.jaw_l[1].y < self.jaw_l[2].y) {
            v.push("jaw_l[0].y < jaw_l[1].y < jaw_l[2].y".into());
        }
        let r2 = self.eye_distance() / self.left_eye_width().max(1e-6);
        if !(1.5 < r2 && r2 < 3.0) {
            v.push(format!(
                "pupil distance / left eye width = {r2:.2} outside (1.5, 3.0)"
            ));
        }
        v
    }

    /// 所有语义点（用于可视化）。
    pub fn semantic_points(&self) -> Vec<(&'static str, P)> {
        let mut v = vec![
            ("pupil_l", self.pupil_l),
            ("pupil_r", self.pupil_r),
            ("eye_outer_l", self.eye_outer_l),
            ("eye_inner_l", self.eye_inner_l),
            ("eye_inner_r", self.eye_inner_r),
            ("eye_outer_r", self.eye_outer_r),
            ("nose_bridge_top", self.nose_bridge_top),
            ("nose_tip", self.nose_tip),
            ("nose_bottom", self.nose_bottom),
            ("nostril_l", self.nostril_l),
            ("nostril_r", self.nostril_r),
            ("nose_wing_l", self.nose_wing_l),
            ("nose_wing_r", self.nose_wing_r),
            ("chin", self.chin),
            ("mouth_l", self.mouth_l),
            ("mouth_r", self.mouth_r),
        ];
        for i in 0..3 {
            v.push((["jaw_l0", "jaw_l1", "jaw_l2"][i], self.jaw_l[i]));
            v.push((["jaw_r0", "jaw_r1", "jaw_r2"][i], self.jaw_r[i]));
        }
        v
    }

    /// 眼轮廓均值（用于无虹膜点的模型估计瞳孔中心）。
    pub fn eye_center_from_ring(ring: &[P]) -> P {
        centroid(ring)
    }
}

/// 关键点模型统一接口。
pub trait LandmarkModel: Send {
    fn name(&self) -> &'static str;
    /// 在原图上按人脸框检测语义关键点（像素坐标）。
    fn detect(
        &mut self,
        img: &image::RgbImage,
        face_box: &FaceBox,
    ) -> anyhow::Result<FaceKeyPoints>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaw_estimate_symmetry() {
        let l = P::new(0.0, 0.0);
        let r = P::new(100.0, 0.0);
        assert!(FaceKeyPoints::estimate_yaw_deg(l, r, P::new(50.0, 40.0)).abs() < 1e-4);
        assert!(FaceKeyPoints::estimate_yaw_deg(l, r, P::new(75.0, 40.0)) > 20.0);
        assert!(FaceKeyPoints::estimate_yaw_deg(l, r, P::new(25.0, 40.0)) < -20.0);
    }

    #[test]
    fn forehead_arc_is_above_temples() {
        let up = P::new(0.0, -1.0);
        let arc = FaceKeyPoints::synth_forehead(
            P::new(0.0, 100.0),
            P::new(100.0, 100.0),
            P::new(50.0, 80.0),
            up,
            100.0,
            9,
        );
        assert_eq!(arc.len(), 9);
        assert!(arc[4].y < 80.0 - 50.0);
        assert!(arc[0].x > arc[8].x);
    }
}
