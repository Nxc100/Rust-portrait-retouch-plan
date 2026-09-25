//! 由语义关键点生成形变步骤序列（瘦脸 / 大眼 / 瘦鼻 / 缩下巴 / 整体收窄），并提供反向映射。

use crate::face::semantic::FaceKeyPoints;
use crate::geom::P;
use crate::warp::pinch::{curve_warp, enlarge, enlarge_gpupixel, narrow, pinch};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub enum Step {
    /// warpPositionToUse1：以 a 为圆心、半径 radius，把采样点沿 a→b 方向反向偏移 delta
    Pinch { a: P, b: P, radius: f32, delta: f32 },
    /// adjust_eye：以 center 为圆心的径向幂次缩放
    Enlarge { center: P, radius: f32, k: f32 },
    /// gpupixel curveWarp
    CurveWarp { origin: P, target: P, delta: f32 },
    /// gpupixel enlargeEye
    EnlargeGp { origin: P, radius: f32, delta: f32 },
    /// 以 center 为中心、沿 axis 方向的整体收窄（水平压缩，随距离平滑衰减）
    Narrow {
        center: P,
        axis: P,
        k: f32,
        radius: f32,
    },
}

/// 形变风格。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReshapeStyle {
    /// 美狐 GLImageFaceChangeFilter（默认）
    #[default]
    Meihu,
    /// pixpark/gpupixel FaceReshapeFilter
    GpuPixel,
}

/// 各系数以瞳距 `ed` 为尺度，可从 JSON 加载以在样张上微调。
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WarpCoefficients {
    /// 瘦脸锚点相对轮廓点的外推量（× ed）：[0.13, 0.33, 0.33]
    pub thin_face_offsets: [f32; 3],
    /// 瘦脸影响半径（× ed）：0.4 × 2ed = 0.8
    pub thin_face_radius: f32,
    /// 瘦脸位移系数：delta = profile_i × intensity × thin_face_delta × 2ed
    pub thin_face_delta: f32,
    /// 三对锚点的位移权重；None 时用美狐的 [sin(π/25), sin(2π/25), sin(2π/25)]
    pub thin_face_profile: Option<[f32; 3]>,
    /// 大眼幂次系数：k = intensity × 0.24
    pub big_eye_k: f32,
    /// 大眼半径（× 左眼宽）
    pub big_eye_radius: f32,
    /// 瘦鼻半径（× ed）：原 0.16 归一化常数按 720×1280、瞳距≈0.12 宽换算 ≈ 1.3
    pub thin_nose_radius: f32,
    /// 瘦鼻位移（× ed × intensity）
    pub thin_nose_delta: f32,
    /// 瘦鼻鼻孔锚点沿 dir_up 上移量（× ed）
    pub thin_nose_lift: f32,
    /// 缩下巴：以下巴为圆心向鼻梁方向推挤，半径（× ed）与位移（× ed × intensity）
    pub chin_lift_radius: f32,
    pub chin_lift_delta: f32,
    /// 整体收窄的衰减半径（× ed），中心为鼻尖
    pub face_narrow_radius: f32,
    /// gpupixel 风格：瘦脸 delta 上限、大眼 delta 上限（原属性范围 [0, 0.15]）
    pub gp_thin_face_max: f32,
    pub gp_big_eye_max: f32,
    /// 侧脸衰减：|yaw| ≤ soft 不衰减，≥ hard 衰减到 min_scale
    pub yaw_soft_deg: f32,
    pub yaw_hard_deg: f32,
    pub yaw_min_scale: f32,
}

impl Default for WarpCoefficients {
    fn default() -> Self {
        Self {
            thin_face_offsets: [0.13, 0.33, 0.33],
            thin_face_radius: 0.8,
            thin_face_delta: 0.15,
            thin_face_profile: None,
            big_eye_k: 0.24,
            big_eye_radius: 1.0,
            thin_nose_radius: 1.3,
            thin_nose_delta: 0.04,
            thin_nose_lift: 1.3,
            chin_lift_radius: 2.0,
            chin_lift_delta: 0.045,
            face_narrow_radius: 3.0,
            gp_thin_face_max: 0.15,
            gp_big_eye_max: 0.15,
            yaw_soft_deg: 30.0,
            yaw_hard_deg: 45.0,
            yaw_min_scale: 0.3,
        }
    }
}

impl WarpCoefficients {
    /// 侧脸衰减系数。
    pub fn yaw_attenuation(&self, yaw_deg: f32) -> f32 {
        let y = yaw_deg.abs();
        if y <= self.yaw_soft_deg {
            1.0
        } else if y >= self.yaw_hard_deg {
            self.yaw_min_scale
        } else {
            let t = (y - self.yaw_soft_deg) / (self.yaw_hard_deg - self.yaw_soft_deg).max(1e-6);
            1.0 + (self.yaw_min_scale - 1.0) * t
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WarpParams {
    pub thin_face: f32,
    pub big_eye: f32,
    pub thin_nose: f32,
    /// 缩下巴 / 提升下半脸（0..1）
    pub chin_lift: f32,
    /// 整体收窄比例（0..~0.05；0.031 ≈ 瞳孔向中轴移动 1.1% 瞳距）
    pub face_narrow: f32,
}

impl WarpParams {
    pub fn is_zero(&self) -> bool {
        self.thin_face <= 0.0
            && self.big_eye <= 0.0
            && self.thin_nose <= 0.0
            && self.chin_lift <= 0.0
            && self.face_narrow <= 0.0
    }
}

#[derive(Clone, Debug)]
pub struct FaceWarp {
    pub steps: Vec<Step>,
    /// 所有影响圆的包围盒（min, max），用于跳过无关像素
    pub bbox: (P, P),
}

impl FaceWarp {
    pub fn from_face(f: &FaceKeyPoints, p: &WarpParams) -> Self {
        Self::from_face_with(
            f,
            p,
            &WarpCoefficients::default(),
            ReshapeStyle::Meihu,
            true,
        )
    }

    /// `attenuate_yaw`：是否按 yaw_deg 衰减瘦脸 / 瘦鼻 / 缩下巴 / 收窄。
    pub fn from_face_with(
        f: &FaceKeyPoints,
        p: &WarpParams,
        c: &WarpCoefficients,
        style: ReshapeStyle,
        attenuate_yaw: bool,
    ) -> Self {
        let ed = f.eye_distance().max(1.0);
        let dir_right = f.dir_right();
        let dir_up = f.dir_up();
        let att = if attenuate_yaw {
            c.yaw_attenuation(f.yaw_deg)
        } else {
            1.0
        };
        let thin_face = (p.thin_face * att).clamp(0.0, 1.0);
        let thin_nose = (p.thin_nose * att).clamp(0.0, 1.0);
        let chin_lift = (p.chin_lift * att).clamp(0.0, 1.0);
        let face_narrow = (p.face_narrow * att).clamp(0.0, 0.2);
        let big_eye = p.big_eye.clamp(0.0, 1.0);
        let mut steps = Vec::new();

        match style {
            ReshapeStyle::Meihu => {
                // ---- 瘦脸：与 adjust_thinFace 逐项对应 ----
                if thin_face > 0.0 {
                    let x = std::f32::consts::PI / 25.0;
                    let scale = 2.0 * ed;
                    let radius = c.thin_face_radius * ed;
                    let profile =
                        c.thin_face_profile
                            .unwrap_or([x.sin(), (2.0 * x).sin(), (2.0 * x).sin()]);
                    for (i, s) in profile.iter().enumerate() {
                        let delta = s * thin_face * c.thin_face_delta * scale;
                        let l = f.jaw_l[i].sub(dir_right.mul(ed * c.thin_face_offsets[i]));
                        let r = f.jaw_r[i].add(dir_right.mul(ed * c.thin_face_offsets[i]));
                        steps.push(Step::Pinch {
                            a: l,
                            b: r,
                            radius,
                            delta,
                        });
                        steps.push(Step::Pinch {
                            a: r,
                            b: l,
                            radius,
                            delta,
                        });
                    }
                }
                // ---- 大眼：与 adjust_eye 逐项对应（原实现只用左眼宽度）----
                if big_eye > 0.0 {
                    let radius = f.left_eye_width() * c.big_eye_radius;
                    let k = big_eye * c.big_eye_k;
                    steps.push(Step::Enlarge {
                        center: f.pupil_l,
                        radius,
                        k,
                    });
                    steps.push(Step::Enlarge {
                        center: f.pupil_r,
                        radius,
                        k,
                    });
                }
            }
            ReshapeStyle::GpuPixel => {
                if thin_face > 0.0 {
                    let delta = thin_face * c.gp_thin_face_max;
                    let bridge = |t: f32| f.nose_bridge_top.lerp(f.nose_tip, t);
                    let n44 = bridge(1.0 / 3.0);
                    let n45 = bridge(2.0 / 3.0);
                    let n46 = f.nose_tip;
                    let n49 = f.nose_bottom;
                    let cl = |k: usize| f.contour[k.min(f.contour.len() - 1)];
                    let m = f.contour.len() as f32 / 33.0;
                    let ci = |k: usize| cl((k as f32 * m).round() as usize);
                    let pairs = [
                        (ci(3), n44),
                        (ci(29), n44),
                        (ci(7), n45),
                        (ci(25), n45),
                        (ci(10), n46),
                        (ci(22), n46),
                        (ci(14), n49),
                        (ci(18), n49),
                        (f.chin, n49),
                    ];
                    for (o, t) in pairs {
                        steps.push(Step::CurveWarp {
                            origin: o,
                            target: t,
                            delta,
                        });
                    }
                }
                if big_eye > 0.0 {
                    let delta = big_eye * c.gp_big_eye_max;
                    let rl = f.pupil_l.dist(f.eye_top_l) * 5.0;
                    let rr = f.pupil_r.dist(f.eye_top_r) * 5.0;
                    steps.push(Step::EnlargeGp {
                        origin: f.pupil_l,
                        radius: rl.max(1.0),
                        delta,
                    });
                    steps.push(Step::EnlargeGp {
                        origin: f.pupil_r,
                        radius: rr.max(1.0),
                        delta,
                    });
                }
            }
        }
        // ---- 瘦鼻：newNarrowNose_2 的 ed 尺度等效换算（两种风格共用）----
        if thin_nose > 0.0 {
            let radius = c.thin_nose_radius * ed;
            let delta = c.thin_nose_delta * ed * thin_nose;
            let lift = dir_up.mul(c.thin_nose_lift * ed);
            let a = [f.nostril_l.add(lift), f.nose_wing_l];
            let b = [f.nostril_r.add(lift), f.nose_wing_r];
            for i in 0..2 {
                steps.push(Step::Pinch {
                    a: a[i],
                    b: b[i],
                    radius,
                    delta,
                });
                steps.push(Step::Pinch {
                    a: b[i],
                    b: a[i],
                    radius,
                    delta,
                });
            }
        }
        // ---- 缩下巴：以下巴为圆心、向鼻梁方向推挤，大半径覆盖嘴与鼻底 ----
        if chin_lift > 0.0 {
            steps.push(Step::Pinch {
                a: f.chin,
                b: f.nose_bridge_top,
                radius: c.chin_lift_radius * ed,
                delta: c.chin_lift_delta * ed * chin_lift,
            });
        }
        // ---- 整体收窄：以鼻尖为中心沿 dir_right 压缩 ----
        if face_narrow > 0.0 {
            steps.push(Step::Narrow {
                center: f.nose_tip,
                axis: dir_right,
                k: face_narrow,
                radius: c.face_narrow_radius * ed,
            });
        }
        let bbox = compute_bbox(&steps);
        Self { steps, bbox }
    }

    pub fn is_identity(&self) -> bool {
        self.steps.is_empty()
    }

    /// 反向映射：输出像素位置 → 应从原图采样的位置。
    #[inline]
    pub fn map(&self, mut pos: P) -> P {
        for s in &self.steps {
            pos = match *s {
                Step::Pinch {
                    a,
                    b,
                    radius,
                    delta,
                } => pinch(pos, a, b, radius, delta),
                Step::Enlarge { center, radius, k } => enlarge(pos, center, radius, k),
                Step::CurveWarp {
                    origin,
                    target,
                    delta,
                } => curve_warp(pos, origin, target, delta),
                Step::EnlargeGp {
                    origin,
                    radius,
                    delta,
                } => enlarge_gpupixel(pos, origin, radius, delta),
                Step::Narrow {
                    center,
                    axis,
                    k,
                    radius,
                } => narrow(pos, center, axis, k, radius),
            };
        }
        pos
    }
}

fn compute_bbox(steps: &[Step]) -> (P, P) {
    let mut mn = P::new(f32::MAX, f32::MAX);
    let mut mx = P::new(f32::MIN, f32::MIN);
    for s in steps {
        let (c, r) = match *s {
            Step::Pinch {
                a, radius, delta, ..
            } => (a, radius + delta.abs()),
            Step::Enlarge { center, radius, .. } => (center, radius),
            Step::CurveWarp { origin, target, .. } => (origin, target.dist(origin)),
            Step::EnlargeGp { origin, radius, .. } => (origin, radius),
            Step::Narrow { center, radius, .. } => (center, radius),
        };
        mn = P::new(mn.x.min(c.x - r), mn.y.min(c.y - r));
        mx = P::new(mx.x.max(c.x + r), mx.y.max(c.y + r));
    }
    (mn, mx)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::face::semantic::FaceBox;

    pub(crate) fn synthetic_face() -> FaceKeyPoints {
        // 正脸：瞳距 100，脸宽 ~260，图像 600×700
        let cx = 300.0;
        let eye_y = 300.0;
        let p = |x: f32, y: f32| P::new(x, y);
        let contour: Vec<P> = (0..33)
            .map(|i| {
                let t = i as f32 / 32.0; // 0 左太阳穴 → 1 右太阳穴，经过下巴
                let ang = std::f32::consts::PI * (1.0 - t); // π → 0
                p(cx + 140.0 * ang.cos(), eye_y + 20.0 + 190.0 * ang.sin())
            })
            .collect();
        let forehead: Vec<P> = (0..9)
            .map(|i| p(cx + 140.0 - 35.0 * i as f32, eye_y - 150.0))
            .collect();
        FaceKeyPoints {
            bbox: FaceBox {
                x1: 140.0,
                y1: 150.0,
                x2: 460.0,
                y2: 530.0,
                score: 1.0,
            },
            pupil_l: p(cx - 50.0, eye_y),
            pupil_r: p(cx + 50.0, eye_y),
            eye_outer_l: p(cx - 75.0, eye_y),
            eye_inner_l: p(cx - 28.0, eye_y),
            eye_inner_r: p(cx + 28.0, eye_y),
            eye_outer_r: p(cx + 75.0, eye_y),
            eye_top_l: p(cx - 50.0, eye_y - 12.0),
            eye_bot_l: p(cx - 50.0, eye_y + 12.0),
            eye_top_r: p(cx + 50.0, eye_y - 12.0),
            eye_bot_r: p(cx + 50.0, eye_y + 12.0),
            nose_bridge_top: p(cx, eye_y),
            nose_tip: p(cx, eye_y + 90.0),
            nose_bottom: p(cx, eye_y + 105.0),
            nostril_l: p(cx - 18.0, eye_y + 100.0),
            nostril_r: p(cx + 18.0, eye_y + 100.0),
            nose_wing_l: p(cx - 30.0, eye_y + 90.0),
            nose_wing_r: p(cx + 30.0, eye_y + 90.0),
            chin: contour[16],
            jaw_l: [contour[4], contour[9], contour[13]],
            jaw_r: [contour[28], contour[23], contour[19]],
            mouth_l: p(cx - 35.0, eye_y + 150.0),
            mouth_r: p(cx + 35.0, eye_y + 150.0),
            contour,
            forehead,
            yaw_deg: 0.0,
            model: "synthetic".into(),
            raw: vec![],
            score: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn zero_params_produce_no_steps() {
        let f = synthetic_face();
        let w = FaceWarp::from_face(&f, &WarpParams::default());
        assert!(w.is_identity());
        let q = w.map(P::new(300.0, 300.0));
        assert_eq!(q, P::new(300.0, 300.0));
    }

    #[test]
    fn thin_face_moves_samples_outward_near_jaw() {
        let f = synthetic_face();
        let w = FaceWarp::from_face(
            &f,
            &WarpParams {
                thin_face: 1.0,
                ..Default::default()
            },
        );
        assert_eq!(w.steps.len(), 6);
        let a = f.jaw_l[1];
        let q = w.map(a);
        assert!(q.x < a.x, "expected outward sampling, got {q:?} for {a:?}");
    }

    #[test]
    fn big_eye_shrinks_sampling_radius() {
        let f = synthetic_face();
        let w = FaceWarp::from_face(
            &f,
            &WarpParams {
                big_eye: 1.0,
                ..Default::default()
            },
        );
        let c = f.pupil_l;
        let p = P::new(c.x + 10.0, c.y);
        let q = w.map(p);
        assert!(q.x < p.x && q.x > c.x);
    }

    #[test]
    fn chin_lift_samples_below_and_narrow_samples_outward() {
        let f = synthetic_face();
        let w = FaceWarp::from_face(
            &f,
            &WarpParams {
                chin_lift: 1.0,
                ..Default::default()
            },
        );
        let q = w.map(f.chin);
        assert!(
            q.y > f.chin.y + 3.0,
            "chin should sample from below: {q:?} vs {:?}",
            f.chin
        );
        let w2 = FaceWarp::from_face(
            &f,
            &WarpParams {
                face_narrow: 0.03,
                ..Default::default()
            },
        );
        let p = P::new(f.nose_tip.x + 100.0, f.nose_tip.y);
        let q2 = w2.map(p);
        assert!(q2.x > p.x + 1.0 && q2.x < p.x + 3.5, "{q2:?}");
    }

    #[test]
    fn sanity_check_passes_on_synthetic_face() {
        let f = synthetic_face();
        assert!(f.sanity_check().is_empty(), "{:?}", f.sanity_check());
    }

    #[test]
    fn yaw_attenuation_curve() {
        let c = WarpCoefficients::default();
        assert_eq!(c.yaw_attenuation(10.0), 1.0);
        assert!((c.yaw_attenuation(37.5) - 0.65).abs() < 1e-5);
        assert_eq!(c.yaw_attenuation(80.0), 0.3);
    }
}
