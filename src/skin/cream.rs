//! "奶油肌"模式：对像素蛋糕"奶油肌"预设导出图的逆向分析结果的实现（见 doc/analysis/cream_skin.md）。
//!
//! 逐张脸（以瞳距 ed 为尺度）在 Lab 空间内：
//! 1. 大尺度瑕疵修复（疤痕 / 痣：环核检测 + 调和插值 + 纹理移植，`heal.rs`）；
//! 2. 小瑕疵祛除（紧凑小斑检测 + 归一化卷积填充，`blemish.rs`）；
//! 3. 亮度三频段 + 能量自适应：`G_f = G(L; fine_sigma·ed)`、`G_m = G(L; mid_sigma·ed)`、
//!    `base = GF(L; detail_radius·ed, detail_eps)`（结构与光影，不动）；
//!    `fine = L − G_f`（颗粒）、`mid = G_f − G_m`（0.005–0.035 ed：毛孔簇、斑驳、细纹）、`low = G_m − base`；
//!    中频的局部能量 `E = √G(mid²; 0.05 ed)`（L 单位），衰减
//!    `α_mid(E) = detail_smooth·(1 − smoothstep(ln energy_lo, ln energy_hi, ln E))`（对数刻度：新郎的保留比例
//!    从 E≈0.7 一直缓慢升到 E≈11，线性刻度的 smoothstep 拟合不了）；
//!    `L' = L − α_fine·fine − α_mid(E)·mid − α_low·low`。
//!    依据：对像素蛋糕导出图按"局部能量分箱"统计（tools/texture_energy.py），两张样张、四张脸一致地表现为
//!    ——平滑皮肤处（E < 1–1.5）中频只剩 60–70%，纹理 / 五官边缘处（E > 3–5）几乎不动，最细颗粒只轻压 10–20%，
//!    更低频几乎不变；即"压平斑驳、保留毛孔与轮廓"。固定比例的衰减会在已经光滑的皮肤（浓妆）上过度磨皮；
//!    眼下区域的 α_mid / α_low 额外提高（细纹 / 卧蚕阴影）；
//! 4. 色度匀肤：`a' = a − α_c·(a − GF(a; guide L, r_c))`（b 同），并把低频色度向"随亮度变化的皮肤色度"
//!    （遮罩内 色度 ~ L 的线性回归）压缩 `unify`——高光彩度低是物理规律，不能当色斑拉平；
//! 5. 色调：中间调提亮（钟形，随 L 远离中心衰减）、最高光压制（去油光）；低频色度向目标肤色按比例拉近
//!    （偏黄偏红的皮肤降得多，白净的几乎不动），再加随亮度变化的 b 偏移（亮部 / 最高光 / 阴影）。
//!    常数由 tools/skin_tone_fit.py 在 10 张照片、16 张脸上拟合；之后可选"立体"：脸内 0.06–0.25 瞳距的明暗结构
//!    （遮罩内归一化低通之差）按 `stereo` 加强（像素蛋糕的"中性灰立体"；奶油肌不开）；
//! 6. 身体皮肤（颈胸臂手）用同样的三频段处理（参数独立）；色调为随低频亮度变化的提亮（阴影到中间调是平台、
//!    高光渐弱）与冷白偏移。色调只看低频亮度，不改变局部对比：按像素亮度提亮会把腋下、肘弯等暗褶纹加深。
//!    色调参数按人：身体像素按到各张脸的距离软分配，混合各人（男女不同）的身体色调参数。
//! 7. 颈纹淡化：每张脸的脖子（人脸解析的脖子类）上对去掉细颗粒的亮度做保边平滑，只处理局部能量中等、
//!    不是异物的结构，再把颈纹的线芯与横过脖子的碎发（细长暗线）填平——项链、纹身、胡茬与轮廓不动（`neck.rs`）。
//!    两张脸的解析范围可能盖到同一段脖子（贴脸、亲吻），脖子像素按与身体色调相同的归属规则分给各人，不重复处理。
//!
//! 所有步骤只在各自遮罩的包围盒内计算并乘以羽化遮罩，遮罩外像素逐位不变。

use crate::buffer::{GrayF32, ImgF32};
use crate::color::lab::LabPlanes;
use crate::face::semantic::FaceKeyPoints;
use crate::geom::P;
use crate::skin::blemish::{detect_blemishes, detect_scale, inpaint_blobs, BlemishParams};
use crate::skin::guided::{fast_gaussian, guided_filter, masked_gaussian, subsample_for};
use crate::skin::heal::{detect_lesions, heal_lesions, HealParams};
use crate::skin::mask::fill_polygon;
use crate::skin::masks::{bbox_of, crop_gray, paste_gray, SkinMasks};
use crate::skin::neck::{self, NeckParams, NeckZone};
use crate::skin::smoothstep;
use crate::skin::spill;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CreamParams {
    /// 中频（fine_sigma–mid_sigma：毛孔簇、斑驳、细纹）在平滑皮肤处的最大衰减比例 α_mid（0..1）
    pub detail_smooth: f32,
    /// 中频衰减随局部中频能量（Lab L 单位）从 energy_lo 到 energy_hi 平滑降到 0（对数刻度）
    pub energy_lo: f32,
    pub energy_hi: f32,
    /// 低频（mid_sigma–detail_radius：大块色斑）衰减比例 α_low（0..1）
    pub low_smooth: f32,
    /// 低频上限：导向滤波半径（× ed）
    pub detail_radius: f32,
    /// 低频导向滤波正则（L² 单位；越大越把中等对比的色块当作噪声压平，强边缘仍保留）
    pub detail_eps: f32,
    /// 细颗粒（< fine_sigma）衰减比例 α_fine（0..1）
    pub fine_smooth: f32,
    /// 细颗粒 / 中频分界的高斯 σ（× ed）
    pub fine_sigma: f32,
    /// 中频 / 低频分界的高斯 σ（× ed）
    pub mid_sigma: f32,
    /// 眼下区域额外的中频 / 低中频衰减（叠加到 α_mid、α_low）
    pub undereye_smooth: f32,
    /// 色度匀肤比例 α_c
    pub chroma_smooth: f32,
    /// 色度导向滤波半径（× ed）
    pub chroma_radius: f32,
    /// 低频色度向皮肤均值压缩的比例（肤色统一）
    pub chroma_unify: f32,
    /// 低频色度尺度（× ed）
    pub unify_sigma: f32,
    /// 中间调提亮量（L）：钟形 `lift · exp(−((L − center)/width)²)`
    pub midtone_lift: f32,
    pub lift_center_l: f32,
    pub lift_width_l: f32,
    /// 最高光压制量（L，作用于皮肤 L 的前 ~10%）
    pub highlight_suppress: f32,
    /// 随黄度提亮：`ΔL += lift_per_b · b`（低频 b）。像素蛋糕的提亮与皮肤黄度成正比（在 RGB 里降黄会抬高亮度）
    pub lift_per_b: f32,
    /// 肤色向目标色拉：低频色度 `a' = a − pull_a·(a − target_a)`（b 同）。像素蛋糕的肤色变化与原肤色成比例
    /// ——偏黄偏红的皮肤降得多，本就白净的皮肤几乎不动甚至略加暖（10 张照片 16 张脸，tools/skin_tone_fit.py）
    pub pull_a: f32,
    pub target_a: f32,
    pub pull_b: f32,
    pub target_b: f32,
    /// 额外的降红（a 偏移，叠加在目标色之上；默认 0）
    pub redness: f32,
    /// 额外的降黄（b 偏移，默认 0）；亮部随亮度渐增的偏移（L 50→85 smoothstep）；最高光（p90–p99.5）额外偏移；
    /// 阴影处额外偏移（暖阴影）
    pub yellow: f32,
    pub yellow_bright: f32,
    pub yellow_highlight: f32,
    pub yellow_shadow: f32,
    /// 眼部清晰：眼眶椭圆内的亮度局部对比增益 `L += k·(L − G(L; 0.04·瞳距))`（巩膜 / 高光更亮、
    /// 睫毛线与虹膜边缘更深）。像素蛋糕两张样张的眼区 L 标准差都变大（海边 +4–5%）
    pub eye_contrast: f32,
    /// 立体（像素蛋糕"中性灰立体"）：脸内大尺度明暗结构（`STEREO_FINE`–`STEREO_COARSE` 瞳距，只在脸部遮罩内取样）
    /// 的增益（0 关闭）。色调按像素亮度提亮 / 压高光会压平脸的明暗，像素蛋糕反而把五官与轮廓的明暗加强
    pub stereo: f32,
    /// 身体皮肤：中频衰减、低中频衰减、半径（× ed_mean）、细颗粒衰减、提亮、降红、降黄、色度匀肤。
    /// 提亮 `body_lift` 是阴影到中间调的平台值（L），随低频亮度的形状见 `body_lift_weight`；
    /// 降黄是低频黄度 b 按比例 `body_pull_b` 向 `body_target_b` 拉近（偏黄的降得多，天光下偏蓝的皮肤略加暖），
    /// 另加偏移 `body_yellow`，都随低频亮度从 L 30 到 60 线性加满。提亮 / 降红 / 降黄按人取值：身体像素按到
    /// 各张脸的距离混合各人的参数（见 `body_tone_at`），其余身体参数取自调用方给的 `body_params`
    pub body_smooth: f32,
    pub body_low_smooth: f32,
    pub body_radius: f32,
    pub body_fine_smooth: f32,
    pub body_lift: f32,
    pub body_redness: f32,
    pub body_yellow: f32,
    /// 预设文件里没有这一项（0.6.3 之前导出）时为 0：保持文件里原来的常数降黄 `body_yellow`
    #[serde(default)]
    pub body_pull_b: f32,
    pub body_target_b: f32,
    pub body_chroma_smooth: f32,
    /// 身体色调外溢到语义遮罩漏掉的相邻皮肤（0..1，见 `skin::spill`；0 关闭）。只补提亮 / 降红 / 降黄，
    /// 色调很强的预设（婚纱-深色内景）才需要——漏掉的皮肤与处理过的皮肤差得多，会显成色块
    pub body_tone_spill: f32,
    /// 牙齿美白（0..1，见 `skin::teeth`；0 关闭，需要人脸解析）
    pub teeth_whiten: f32,
    /// 脸部遮罩按人归属（取自 `body_params`）：多人时去掉每张脸遮罩里落在别人脸轮廓内的部分（脸轮廓为关键点的下颌线 + 额头点围成的多边形）。
    /// 两人脸挨得近时，一个人的解析裁剪框会把框里另一个人的脸也标成皮肤（IMG_5785 新郎的遮罩盖住新娘的右颊与
    /// 右眼睫毛、新娘的盖住新郎的左半边脸），不加限制时那片再按对方的参数处理一遍——色调很强的预设（婚纱-深色
    /// 内景）里显成边界分明的色块、睫毛被提亮成金铜色。算子默认关（不改变不用预设时的输出），两个内置预设都开
    pub face_ownership: bool,
    /// 小瑕疵祛除
    pub blemish: BlemishParams,
    pub body_blemish: bool,
    /// 大尺度瑕疵修复（脸 / 身体）
    pub heal: HealParams,
    pub body_heal: HealParams,
    /// 颈纹淡化（脖子区域，见 `skin::neck`）
    pub neck: NeckParams,
}

impl Default for CreamParams {
    fn default() -> Self {
        Self {
            detail_smooth: 0.60,
            energy_lo: 1.2,
            energy_hi: 3.5,
            low_smooth: 0.03,
            detail_radius: 0.08,
            detail_eps: 60.0,
            fine_smooth: 0.26,
            fine_sigma: 0.005,
            mid_sigma: 0.035,
            undereye_smooth: 0.40,
            chroma_smooth: 0.33,
            chroma_radius: 0.14,
            chroma_unify: 0.15,
            unify_sigma: 0.05,
            midtone_lift: 0.405,
            lift_center_l: 62.0,
            lift_width_l: 26.0,
            highlight_suppress: 0.775,
            lift_per_b: 0.035,
            pull_a: 0.105,
            target_a: 3.67,
            pull_b: 0.149,
            target_b: 9.04,
            redness: 0.0,
            yellow: 0.0,
            yellow_bright: -0.5,
            yellow_highlight: 0.97,
            yellow_shadow: 0.05,
            eye_contrast: 0.30,
            stereo: 0.0,
            body_smooth: 0.50,
            body_low_smooth: 0.25,
            body_radius: 0.05,
            body_fine_smooth: 0.25,
            body_lift: 4.6,
            body_redness: -0.85,
            // 像素蛋糕的女性身体：黄度按 0.19 的比例向 b 1.3 拉近（b −1.5 → +1.9，b 21 → −3.6；
            // tools/body_tone_fit.py，7 张照片，残差 RMS 1.51 → 0.78）
            body_yellow: 0.0,
            body_pull_b: 0.19,
            body_target_b: 1.3,
            body_chroma_smooth: 0.35,
            body_tone_spill: 0.0,
            teeth_whiten: 0.0,
            face_ownership: false,
            blemish: BlemishParams::default(),
            body_blemish: true,
            heal: HealParams::default(),
            body_heal: HealParams::body_default(),
            neck: NeckParams::default(),
        }
    }
}

impl CreamParams {
    /// 男性默认（对应导出图中新郎的较轻处理：更小的细节衰减、几乎不提亮、保留胡茬、暖阴影；
    /// 但脸颊上的暗斑同样被祛除）。
    pub fn male_default() -> Self {
        Self {
            detail_smooth: 0.61,
            energy_lo: 0.7,
            energy_hi: 11.0,
            low_smooth: 0.03,
            detail_radius: 0.09,
            detail_eps: 80.0,
            fine_smooth: 0.05,
            undereye_smooth: 0.35,
            chroma_smooth: 0.27,
            chroma_unify: 0.2,
            midtone_lift: -0.086,
            lift_center_l: 62.0,
            lift_width_l: 26.0,
            highlight_suppress: 0.584,
            lift_per_b: 0.033,
            pull_a: 0.089,
            target_a: 8.35,
            pull_b: 0.090,
            target_b: 15.84,
            redness: 0.0,
            yellow: 0.0,
            yellow_bright: -1.07,
            yellow_highlight: 1.46,
            yellow_shadow: 0.67,
            eye_contrast: 0.25,
            // 像素蛋糕对新郎的脖子、手只轻度提亮与降黄（tools/body_tone_fit.py：lift 2.6、a −0.6、b −1.1）；
            // 男性身体的降黄量与黄度几乎无关（比例拟合 pull 0.04，残差与常数相同），用常数偏移
            body_lift: 2.6,
            body_redness: -0.6,
            body_yellow: -1.1,
            body_pull_b: 0.0,
            blemish: BlemishParams {
                dark_contrast: 3.4,
                red_contrast: 3.6,
                max_radius: 0.022,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// 脸部几何（区域坐标）：用于眼下区域权重与五官 / 耳朵 / 发际线保护区。
#[derive(Clone)]
struct FaceGeom {
    pupil_l: P,
    pupil_r: P,
    nostril_l: P,
    nostril_r: P,
    nose_bottom: P,
    nose_wing_l: P,
    nose_wing_r: P,
    eye_outer_l: P,
    eye_outer_r: P,
    eye_inner_l: P,
    eye_inner_r: P,
    eye_top_l: P,
    eye_bot_l: P,
    eye_top_r: P,
    eye_bot_r: P,
    mouth_l: P,
    mouth_r: P,
    /// 脸轮廓 + 额头多边形（不含耳朵与头发）
    outline: Vec<P>,
    up: P,
    right: P,
    ed: f32,
}

impl FaceGeom {
    fn from_face(f: &FaceKeyPoints, off: P) -> Self {
        let mut outline: Vec<P> = f.contour.iter().map(|p| p.sub(off)).collect();
        outline.extend(f.forehead.iter().map(|p| p.sub(off)));
        Self {
            pupil_l: f.pupil_l.sub(off),
            pupil_r: f.pupil_r.sub(off),
            nostril_l: f.nostril_l.sub(off),
            nostril_r: f.nostril_r.sub(off),
            nose_bottom: f.nose_bottom.sub(off),
            nose_wing_l: f.nose_wing_l.sub(off),
            nose_wing_r: f.nose_wing_r.sub(off),
            eye_outer_l: f.eye_outer_l.sub(off),
            eye_outer_r: f.eye_outer_r.sub(off),
            eye_inner_l: f.eye_inner_l.sub(off),
            eye_inner_r: f.eye_inner_r.sub(off),
            eye_top_l: f.eye_top_l.sub(off),
            eye_bot_l: f.eye_bot_l.sub(off),
            eye_top_r: f.eye_top_r.sub(off),
            eye_bot_r: f.eye_bot_r.sub(off),
            mouth_l: f.mouth_l.sub(off),
            mouth_r: f.mouth_r.sub(off),
            outline,
            up: f.dir_up(),
            right: f.dir_right(),
            ed: f.eye_distance().max(8.0),
        }
    }
    fn shifted(&self, off: P) -> Self {
        Self {
            pupil_l: self.pupil_l.sub(off),
            pupil_r: self.pupil_r.sub(off),
            nostril_l: self.nostril_l.sub(off),
            nostril_r: self.nostril_r.sub(off),
            nose_bottom: self.nose_bottom.sub(off),
            nose_wing_l: self.nose_wing_l.sub(off),
            nose_wing_r: self.nose_wing_r.sub(off),
            eye_outer_l: self.eye_outer_l.sub(off),
            eye_outer_r: self.eye_outer_r.sub(off),
            eye_inner_l: self.eye_inner_l.sub(off),
            eye_inner_r: self.eye_inner_r.sub(off),
            eye_top_l: self.eye_top_l.sub(off),
            eye_bot_l: self.eye_bot_l.sub(off),
            eye_top_r: self.eye_top_r.sub(off),
            eye_bot_r: self.eye_bot_r.sub(off),
            mouth_l: self.mouth_l.sub(off),
            mouth_r: self.mouth_r.sub(off),
            outline: self.outline.iter().map(|p| p.sub(off)).collect(),
            up: self.up,
            right: self.right,
            ed: self.ed,
        }
    }
}

/// 软椭圆（沿 right / up 轴），值取 max。
fn draw_soft_ellipse(m: &mut GrayF32, c: P, right: P, up: P, rx: f32, ry: f32, v: f32) {
    let (w, h) = (m.w as i64, m.h as i64);
    let ext = rx.max(ry) * 1.2;
    let x0 = ((c.x - ext).floor() as i64).max(0);
    let y0 = ((c.y - ext).floor() as i64).max(0);
    let x1 = ((c.x + ext).ceil() as i64).min(w - 1);
    let y1 = ((c.y + ext).ceil() as i64).min(h - 1);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = P::new(x as f32 + 0.5, y as f32 + 0.5).sub(c);
            let dr = d.dot(right) / rx;
            let du = d.dot(up) / ry;
            let n = (dr * dr + du * du).sqrt();
            let t = ((1.15 - n) / 0.3).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t) * v;
            let i = (y * w + x) as usize;
            if t > m.data[i] {
                m.data[i] = t;
            }
        }
    }
}

fn punch_disc(m: &mut GrayF32, c: P, r: f32) {
    let (w, h) = (m.w as i64, m.h as i64);
    let x0 = ((c.x - r).floor() as i64).max(0);
    let y0 = ((c.y - r).floor() as i64).max(0);
    let x1 = ((c.x + r).ceil() as i64).min(w - 1);
    let y1 = ((c.y + r).ceil() as i64).min(h - 1);
    for y in y0..=y1 {
        for x in x0..=x1 {
            if P::new(x as f32 + 0.5, y as f32 + 0.5).dist(c) < r {
                m.data[(y * w + x) as usize] = 0.0;
            }
        }
    }
}

/// 一个待处理区域：遮罩（ROI 分辨率）、尺度、参数、可选脸部几何（ROI 坐标）。
struct Region<'a> {
    mask: &'a GrayF32,
    ed: f32,
    params: &'a CreamParams,
    is_face: bool,
    geom: Option<FaceGeom>,
    /// 语义皮肤概率（ROI 分辨率）；身体区域的瑕疵 / 疤痕检测只在语义上是皮肤的像素上进行
    skin_prob: Option<&'a GrayF32>,
    /// 身体色调的各人来源（ROI 坐标）；为空时用 `params` 的身体色调
    body_tones: &'a [BodyTone],
    /// 身体色调的遮罩（ROI 坐标，≥ `mask`：外溢到漏掉的相邻皮肤，见 `skin::spill`）；None 时即 `mask`
    tone_mask: Option<&'a GrayF32>,
}

/// 身体色调：提亮、降红、降黄偏移、降黄的拉力与"拉力 × 目标黄度"。黄度改变量 `yellow + pull_target − pull_b·b`
/// 对各项是线性的，按人混合时直接对各项加权即可。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Tone {
    lift: f32,
    redness: f32,
    yellow: f32,
    pull_b: f32,
    pull_target: f32,
}

impl Tone {
    fn of(p: &CreamParams) -> Self {
        Self {
            lift: p.body_lift,
            redness: p.body_redness,
            yellow: p.body_yellow,
            pull_b: p.body_pull_b,
            pull_target: p.body_pull_b * p.body_target_b,
        }
    }

    /// 低频黄度为 `b` 时 b 的改变量（未乘亮度权重）。
    fn yellow_at(&self, b: f32) -> f32 {
        self.yellow + self.pull_target - self.pull_b * b
    }

    /// `self + w · o`（逐项）。
    fn add_scaled(self, o: &Tone, w: f32) -> Self {
        Self {
            lift: self.lift + w * o.lift,
            redness: self.redness + w * o.redness,
            yellow: self.yellow + w * o.yellow,
            pull_b: self.pull_b + w * o.pull_b,
            pull_target: self.pull_target + w * o.pull_target,
        }
    }
}

/// 一个人的身体色调：以其脸中心与尺度定位，身体像素按距离软分配（见 [`body_tone_at`]）。
#[derive(Clone, Copy, Debug)]
struct BodyTone {
    center: P,
    /// 脸的尺度（`FaceKeyPoints::scale_distance`）
    scale: f32,
    tone: Tone,
}

impl BodyTone {
    fn of(params: &CreamParams, center: P, scale: f32) -> Self {
        Self {
            center,
            scale: scale.max(1.0),
            tone: Tone::of(params),
        }
    }
}

/// 身体像素的归属：到各脸中心的距离以该脸尺度为单位，权重 ∝ exp(−(d − d_min) / 软度)。
/// 软度 0.5：自己身上（距离差 ≥ 1.5 个脸尺度）的权重 > 95%，两人之间几个脸尺度内平滑过渡，不出现接缝
const BODY_OWNER_SOFTNESS: f32 = 0.5;

/// 归属权重（未归一化）：`d` 为到某人脸中心的距离、`d_min` 为到最近那张脸的距离，都以脸尺度为单位。
fn owner_weight(d: f32, d_min: f32) -> f32 {
    (-(d - d_min) / BODY_OWNER_SOFTNESS).exp()
}

/// `pos` 属于第 `k` 个人的份额（0..1，各人之和为 1）：`anchors` 为各人的（中心，尺度），归属规则同 [`body_tone_at`]。
fn owner_share(anchors: &[(P, f32)], pos: P, k: usize) -> f32 {
    let dist = |&(c, s): &(P, f32)| pos.dist(c) / s;
    let d_min = anchors.iter().map(dist).fold(f32::INFINITY, f32::min);
    let weight = |a: &(P, f32)| owner_weight(dist(a), d_min);
    weight(&anchors[k]) / anchors.iter().map(weight).sum::<f32>()
}

/// 某个身体像素的色调：各人参数按归属权重混合；没有来源时用 `fallback`。逐像素调用，
/// 权重现算两遍（求和、混合）而不存起来，省掉每个像素一次的内存分配。
fn body_tone_at(tones: &[BodyTone], pos: P, fallback: Tone) -> Tone {
    let dist = |t: &BodyTone| pos.dist(t.center) / t.scale;
    let d_min = tones.iter().map(dist).fold(f32::INFINITY, f32::min);
    if !d_min.is_finite() {
        return fallback;
    }
    let weight = |t: &BodyTone| owner_weight(dist(t), d_min);
    let sw: f32 = tones.iter().map(weight).sum();
    tones.iter().fold(Tone::default(), |acc, t| {
        acc.add_scaled(&t.tone, weight(t) / sw)
    })
}

/// 各人的身体色调都相同（只有一个人、或同性别）时的常数色调，省掉逐像素的归属计算；没有来源时为 `fallback`。
fn uniform_body_tone(tones: &[BodyTone], fallback: Tone) -> Option<Tone> {
    match tones.split_first() {
        None => Some(fallback),
        Some((first, rest)) => rest
            .iter()
            .all(|t| t.tone == first.tone)
            .then_some(first.tone),
    }
}

fn crop_img(src: &ImgF32, x0: usize, y0: usize, cw: usize, ch: usize) -> ImgF32 {
    let mut out = ImgF32::new(cw, ch);
    out.data
        .par_chunks_mut(cw)
        .enumerate()
        .for_each(|(y, row)| {
            row.copy_from_slice(&src.data[(y0 + y) * src.w + x0..(y0 + y) * src.w + x0 + cw]);
        });
    out
}

/// 加权分位数（直方图近似）。
fn weighted_percentile(values: &[f32], weights: &[f32], q: f32) -> f32 {
    let mut hist = [0.0f64; 256];
    let mut total = 0.0f64;
    for (v, w) in values.iter().zip(weights) {
        if *w > 0.05 {
            let bin = ((v / 100.0).clamp(0.0, 1.0) * 255.0) as usize;
            hist[bin] += *w as f64;
            total += *w as f64;
        }
    }
    if total <= 0.0 {
        return 100.0;
    }
    let target = total * q as f64;
    let mut acc = 0.0;
    for (i, h) in hist.iter().enumerate() {
        acc += h;
        if acc >= target {
            return i as f32 / 255.0 * 100.0;
        }
    }
    100.0
}

/// 纹理 / 轮廓处（高能量）中频也保留少量衰减：像素蛋糕在高能量箱的中频比约 0.94–0.97，不是 1。
const KEEP_MAX: f32 = 0.92;
/// 色调环节"向目标色拉"所用低频色度的尺度（× 瞳距当量），与 tools/skin_tone_fit.py 一致
const TONE_LOWPASS: f32 = 0.03;
/// 细颗粒 / 中频分界的最小高斯 σ（像素）
const FINE_SIGMA_FLOOR_PX: f32 = 0.5;
/// 身体区域强暗痕群否决的最低峰值（× 暗斑阈值），见 `BlemishParams::clutter_min_peak`
const BODY_CLUTTER_MIN_PEAK: f32 = 1.5;
/// 身体瑕疵检测的语义皮肤门控：概率在 LO..HI 之间线性过渡
const SEMANTIC_SKIN_LO: f32 = 0.45;
const SEMANTIC_SKIN_HI: f32 = 0.75;

/// 语义皮肤门控：皮肤分割概率 → 0..1（纹身墨迹、首饰、衣褶为 0）。
fn semantic_skin_gate(p: f32) -> f32 {
    ((p - SEMANTIC_SKIN_LO) / (SEMANTIC_SKIN_HI - SEMANTIC_SKIN_LO)).clamp(0.0, 1.0)
}
/// 身体提亮随低频亮度的形状（tools/body_tone_fit.py 在 7 张照片的身体皮肤上拟合像素蛋糕「奶油肌」）：
/// 阴影到中间调是平台（L 20 以下减到平台的 `SHADOW_KEEP`，20→44 回到平台），高光 L 64→88 平滑降到平台的
/// `HIGHLIGHT_KEEP`。像素蛋糕对暗部（腋下、手臂背光面）提得和中间调一样多，对高光（肩头、手背）提得少
const BODY_LIFT_SHADOW_L: (f32, f32) = (20.0, 44.0);
const BODY_LIFT_SHADOW_KEEP: f32 = 0.8;
const BODY_LIFT_FADE_L: (f32, f32) = (64.0, 88.0);
const BODY_LIFT_HIGHLIGHT_KEEP: f32 = 0.1;
/// 身体降黄随低频亮度加满的区间（L）
const BODY_YELLOW_L: (f32, f32) = (30.0, 60.0);

/// 身体提亮的权重（0..1），`l` 为低频亮度（Lab L）。
fn body_lift_weight(l: f32) -> f32 {
    let (s0, s1) = BODY_LIFT_SHADOW_L;
    let (f0, f1) = BODY_LIFT_FADE_L;
    let shadow = BODY_LIFT_SHADOW_KEEP + (1.0 - BODY_LIFT_SHADOW_KEEP) * smoothstep(s0, s1, l);
    shadow * (1.0 - (1.0 - BODY_LIFT_HIGHLIGHT_KEEP) * smoothstep(f0, f1, l))
}

/// 身体降黄的权重（0..1），`l` 为低频亮度。
fn body_yellow_weight(l: f32) -> f32 {
    let (y0, y1) = BODY_YELLOW_L;
    ((l - y0) / (y1 - y0)).clamp(0.0, 1.0)
}

/// 在区域遮罩的包围盒内处理三平面。
fn process_region(planes: &mut LabPlanes, r: &Region, timing: bool) {
    let p = r.params;
    let ed = r.ed.max(8.0);
    let tag = if r.is_face { "face" } else { "body" };
    let max_r = (p
        .chroma_radius
        .max(p.detail_radius)
        .max(p.unify_sigma * 3.0)
        * ed
        * 1.5) as usize;
    let Some((bx0, by0, bx1, by1)) = bbox_of(r.tone_mask.unwrap_or(r.mask), 0.002, max_r.max(8))
    else {
        return;
    };
    let (bw, bh) = (bx1 - bx0, by1 - by0);
    if bw < 4 || bh < 4 {
        return;
    }
    let m = crop_gray(r.mask, bx0, by0, bw, bh);
    let tone_m = r.tone_mask.map(|t| crop_gray(t, bx0, by0, bw, bh));
    let mut sub = LabPlanes {
        w: bw,
        h: bh,
        l: crop_gray(&planes.l, bx0, by0, bw, bh),
        a: crop_gray(&planes.a, bx0, by0, bw, bh),
        b: crop_gray(&planes.b, bx0, by0, bw, bh),
    };
    let geom = r
        .geom
        .as_ref()
        .map(|g| g.shifted(P::new(bx0 as f32, by0 as f32)));

    // 1. 大尺度瑕疵修复（脸部只在"轮廓多边形内缩 ∧ 遮罩"内进行，并扣掉眼周 / 鼻孔 / 嘴角保护区：
    //    耳朵、发际线、鼻孔、眼角、嘴角的暗结构不是瑕疵）
    let t = Instant::now();
    let hp = if r.is_face { &p.heal } else { &p.body_heal };
    let mut healed = 0;
    // 脸部瑕疵检测（大 / 小）的可处理区：轮廓多边形内缩 0.05 瞳距（耳朵、发际线的暗结构不是瑕疵）
    // ∧ 遮罩，再扣掉眼周 / 眉尾 / 鼻孔 / 鼻底 / 鼻翼 / 嘴角保护区
    let protect = match &geom {
        Some(g) if r.is_face => {
            let mut e = m.clone();
            let ed_g = g.ed;
            if g.outline.len() >= 3 {
                let mut poly = GrayF32::new(bw, bh);
                fill_polygon(&mut poly, &g.outline, 1.0);
                let mut poly = fast_gaussian(&poly, (0.04 * ed_g).max(1.0));
                poly.map_inplace(|v| if v > 0.9 { 1.0 } else { 0.0 });
                e.mul_inplace(&poly);
            }
            punch_disc(&mut e, g.pupil_l, 0.42 * ed_g);
            punch_disc(&mut e, g.pupil_r, 0.42 * ed_g);
            punch_disc(&mut e, g.eye_outer_l, 0.36 * ed_g);
            punch_disc(&mut e, g.eye_outer_r, 0.36 * ed_g);
            punch_disc(&mut e, g.nostril_l, 0.22 * ed_g);
            punch_disc(&mut e, g.nostril_r, 0.22 * ed_g);
            punch_disc(&mut e, g.nose_bottom, 0.22 * ed_g);
            punch_disc(&mut e, g.nose_wing_l, 0.22 * ed_g);
            punch_disc(&mut e, g.nose_wing_r, 0.22 * ed_g);
            punch_disc(&mut e, g.mouth_l, 0.18 * ed_g);
            punch_disc(&mut e, g.mouth_r, 0.18 * ed_g);
            e
        }
        _ => {
            // 身体：扣掉语义上不是皮肤的物体（纹身墨迹、首饰、衣褶）。纹身的粗笔画在检测分辨率上
            // 与大痣无异，连线状 / 暗痕群否决也拦不住；皮肤分割把墨迹判为非皮肤（IMG_5785：墨迹中位
            // 概率 0.36，周围皮肤 0.996），而 800 px 输入分辨率下几个像素的斑点仍是皮肤
            let mut e = m.clone();
            if let Some(sp) = r.skin_prob {
                e.data
                    .par_iter_mut()
                    .zip(&sp.data)
                    .for_each(|(v, p)| *v *= semantic_skin_gate(*p));
            }
            e
        }
    };
    if hp.strength > 0.0 {
        let map = detect_lesions(&sub, &protect, ed, hp);
        healed = heal_lesions(&mut sub, &protect, &map, ed, hp);
    }
    if timing {
        eprintln!(
            "  [cream/{tag}] {bw}x{bh} heal: {healed} lesions, {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 2. 小瑕疵祛除
    let t = Instant::now();
    if p.blemish.strength > 0.0 && (r.is_face || p.body_blemish) {
        let bp = if r.is_face {
            p.blemish
        } else {
            // 身体区域大、瑕疵尺度大：用更粗的检测分辨率；粗分辨率下纹身细线的反差被抹淡，
            // 强暗痕群否决的门槛相应放低（身体上成片的是纹身 / 首饰，很少是痘印）
            BlemishParams {
                work_ed: p.blemish.work_ed * 0.6,
                clutter_min_peak: BODY_CLUTTER_MIN_PEAK,
                ..p.blemish
            }
        };
        let blob = detect_blemishes(&sub, &protect, ed, &bp);
        let sigma = (bp.fill_sigma * bp.max_radius * ed).max(2.0);
        inpaint_blobs(&mut sub, &blob, &m, sigma, detect_scale(ed, &bp));
    }
    if timing {
        eprintln!(
            "  [cream/{tag}] blemish: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 3. 亮度三频段 + 能量自适应：fine = L − G_f；mid = G_f − G_m；low = G_m − base；
    //    α_mid 随局部中频能量 E 平滑降为 0（平滑皮肤压斑驳，纹理 / 轮廓处不动）
    let t = Instant::now();
    let (alpha_mid, alpha_low, alpha_fine, rad_base, (e_lo, e_hi)) = if r.is_face {
        (
            p.detail_smooth,
            p.low_smooth,
            p.fine_smooth,
            p.detail_radius * ed,
            (p.energy_lo, p.energy_hi),
        )
    } else {
        (
            p.body_smooth,
            p.body_low_smooth,
            p.body_fine_smooth,
            p.body_radius * ed,
            (p.energy_lo * 1.2, p.energy_hi * 1.25),
        )
    };
    let guide = sub.l.clone();
    if alpha_mid > 0.0 || alpha_fine > 0.0 || alpha_low > 0.0 {
        let rad = rad_base.max(1.5) as usize;
        let base = guided_filter(&sub.l, &guide, rad, p.detail_eps, subsample_for(rad));
        // 细颗粒 / 中频分界至少 0.5 px（瞳距 < 100 px 的小脸才会触到下限）。像素蛋糕对小脸的像素级颗粒
        // 同样按中频压（10 张照片中瞳距 39–96 px 的 4 张脸：细颗粒只剩 0.73–0.89），下限 1 px 时本项目
        // 把这一档全算成细颗粒、只轻压（0.88–1.04）
        let sig_f = (p.fine_sigma * ed).max(FINE_SIGMA_FLOOR_PX);
        let g_f = fast_gaussian(&sub.l, sig_f);
        let g_m = fast_gaussian(&sub.l, (p.mid_sigma * ed).max(sig_f + 0.5));
        let mid_sq: Vec<f32> = g_f
            .data
            .par_iter()
            .zip(&g_m.data)
            .map(|(a, b)| (a - b) * (a - b))
            .collect();
        let energy = fast_gaussian(&GrayF32::from_vec(bw, bh, mid_sq), (0.05 * ed).max(1.0));
        // 眼下区域权重
        let mut boost: Option<GrayF32> = None;
        if let (Some(g), true) = (&geom, r.is_face && p.undereye_smooth > 0.0) {
            let mut ue = GrayF32::new(bw, bh);
            let down = g.up.mul(-1.0);
            for pupil in [g.pupil_l, g.pupil_r] {
                let c = pupil.add(down.mul(0.30 * g.ed));
                draw_soft_ellipse(&mut ue, c, g.right, g.up, 0.42 * g.ed, 0.22 * g.ed, 1.0);
            }
            boost = Some(ue);
        }
        let ue_gain = p.undereye_smooth;
        let ln_lo = e_lo.max(1e-3).ln();
        let ln_span = (e_hi.max(e_lo * 1.01).ln() - ln_lo).max(1e-3);
        sub.l.data.par_iter_mut().enumerate().for_each(|(i, l)| {
            let mk = m.data[i];
            if mk <= 0.0 {
                return;
            }
            let fine = *l - g_f.data[i];
            let mid = g_f.data[i] - g_m.data[i];
            let low = g_m.data[i] - base.data[i];
            let e = energy.data[i].max(1e-6).sqrt();
            let t = ((e.ln() - ln_lo) / ln_span).clamp(0.0, 1.0);
            let keep = KEEP_MAX * t * t * (3.0 - 2.0 * t);
            // 极强的边缘（E 远高于阈值）连细颗粒也不动
            let tf = ((e - 2.0 * e_hi) / (2.0 * e_hi)).clamp(0.0, 1.0);
            let b = boost.as_ref().map(|b| b.data[i] * ue_gain).unwrap_or(0.0);
            let am = (alpha_mid * (1.0 - keep) + b).min(0.9);
            let al = (alpha_low + b).min(0.9);
            let af = alpha_fine * (1.0 - tf);
            *l -= (am * mid + al * low + af * fine) * mk;
        });
    }
    if timing {
        eprintln!(
            "  [cream/{tag}] detail: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 4. 色度匀肤
    let t = Instant::now();
    let alpha_c = if r.is_face {
        p.chroma_smooth
    } else {
        p.body_chroma_smooth
    };
    if alpha_c > 0.0 {
        let rad = (p.chroma_radius * ed).max(2.0) as usize;
        for plane in [&mut sub.a, &mut sub.b] {
            let q = guided_filter(plane, &guide, rad, 40.0, 4);
            plane
                .data
                .par_iter_mut()
                .zip(&q.data)
                .zip(&m.data)
                .for_each(|((v, q), mk)| {
                    *v -= alpha_c * (*v - *q) * mk;
                });
        }
    }
    if r.is_face && p.chroma_unify > 0.0 {
        // 低频色度向"随亮度变化的皮肤色度"压缩：先在遮罩内做 色度 ~ L 的加权线性回归，只压缩偏离回归线的部分。
        // 高光处彩度低是反射光的物理规律，不是色斑；直接向均值压缩会给大面积高光加黄 / 加红
        // （X04 新娘 L 90+：b +1.3 → +0.1）
        let sigma = (p.unify_sigma * ed).max(2.0);
        let low_l = fast_gaussian(&sub.l, sigma);
        let (mut sw, mut sl, mut sll) = (0.0f64, 0.0f64, 0.0f64);
        for (l, w) in low_l.data.iter().zip(&m.data) {
            let (l, w) = (*l as f64, *w as f64);
            sw += w;
            sl += w * l;
            sll += w * l * l;
        }
        if sw >= 1.0 {
            let mean_l = sl / sw;
            let var_l = (sll / sw - mean_l * mean_l).max(1e-6);
            for plane in [&mut sub.a, &mut sub.b] {
                let low = fast_gaussian(plane, sigma);
                let (mut sv, mut slv) = (0.0f64, 0.0f64);
                for ((v, l), w) in low.data.iter().zip(&low_l.data).zip(&m.data) {
                    let (v, l, w) = (*v as f64, *l as f64, *w as f64);
                    sv += w * v;
                    slv += w * l * v;
                }
                let mean_v = sv / sw;
                let slope = ((slv / sw - mean_l * mean_v) / var_l) as f32;
                let (mean_l, mean_v) = (mean_l as f32, mean_v as f32);
                plane
                    .data
                    .par_iter_mut()
                    .zip(&low.data)
                    .zip(&low_l.data)
                    .zip(&m.data)
                    .for_each(|(((v, lo), ll), mk)| {
                        let target = mean_v + slope * (ll - mean_l);
                        *v -= p.chroma_unify * (*lo - target) * mk;
                    });
            }
        }
    }
    if timing {
        eprintln!(
            "  [cream/{tag}] chroma: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 5. 色调：亮度（中间调钟形提亮、最高光压制）；低频色度向目标色拉 + 随亮度变化的 b 偏移
    let t = Instant::now();
    if r.is_face {
        let p90 = weighted_percentile(&sub.l.data, &m.data, 0.90);
        let p995 = weighted_percentile(&sub.l.data, &m.data, 0.995).max(p90 + 1.0);
        let (lc, lw) = (p.lift_center_l, p.lift_width_l.max(1.0));
        let (lift, hl, red, yel, yel_hl, yel_sh) = (
            p.midtone_lift,
            p.highlight_suppress,
            p.redness,
            p.yellow,
            p.yellow_highlight,
            p.yellow_shadow,
        );
        let yel_br = p.yellow_bright;
        // 目标色拉只作用于低频色度，不额外压缩色度细节（那是第 4 步的事）
        let sigma = (TONE_LOWPASS * ed).max(1.0);
        let low_a = fast_gaussian(&sub.a, sigma);
        let low_b = fast_gaussian(&sub.b, sigma);
        sub.l
            .data
            .par_iter_mut()
            .zip(sub.a.data.par_iter_mut())
            .zip(sub.b.data.par_iter_mut())
            .zip(
                m.data
                    .par_iter()
                    .zip(low_a.data.par_iter().zip(&low_b.data)),
            )
            .for_each(|(((l, a), b), (mk, (la, lb)))| {
                if *mk <= 0.0 {
                    return;
                }
                let l0 = *l;
                let z = (l0 - lc) / lw;
                let t_lift = (-z * z).exp();
                let t_hl = ((l0 - p90) / (p995 - p90)).clamp(0.0, 1.0);
                let t_hl = t_hl * t_hl * (3.0 - 2.0 * t_hl);
                let t_sh = ((55.0 - l0) / 12.0).clamp(0.0, 1.0);
                let t_br = ((l0 - 50.0) / 35.0).clamp(0.0, 1.0);
                let t_br = t_br * t_br * (3.0 - 2.0 * t_br);
                *l += (lift * t_lift - hl * t_hl + p.lift_per_b * lb) * mk;
                *a += (red - p.pull_a * (la - p.target_a)) * mk;
                *b += (yel - p.pull_b * (lb - p.target_b)
                    + yel_br * t_br
                    + yel_hl * t_hl
                    + yel_sh * t_sh)
                    * mk;
            });
    } else {
        // 权重只看低频亮度：同一块皮肤上的暗褶纹与周围提得一样多，局部对比不变
        // （按像素亮度提亮时，L 30–60 的斜率 1 + lift/30 会把腋下、肘弯的褶纹加深约 8%）
        // 黄度同样只看遮罩内的低频黄度：降黄随皮肤本身的黄度而定，不随局部起伏
        let fallback = Tone::of(p);
        let uniform = uniform_body_tone(r.body_tones, fallback);
        let origin = P::new(bx0 as f32, by0 as f32);
        let sigma = (TONE_LOWPASS * ed).max(1.0);
        let m = tone_m.as_ref().unwrap_or(&m);
        let low_l = masked_gaussian(&sub.l, m, sigma);
        let low_b = masked_gaussian(&sub.b, m, sigma);
        sub.l
            .data
            .par_iter_mut()
            .zip(sub.a.data.par_iter_mut())
            .zip(sub.b.data.par_iter_mut())
            .zip(
                m.data
                    .par_iter()
                    .zip(low_l.data.par_iter().zip(&low_b.data)),
            )
            .enumerate()
            .for_each(|(i, (((l, a), b), (mk, (ll, lb))))| {
                if *mk <= 0.0 {
                    return;
                }
                let tone = uniform.unwrap_or_else(|| {
                    let pos = origin.add(P::new((i % bw) as f32, (i / bw) as f32));
                    body_tone_at(r.body_tones, pos, fallback)
                });
                *l += tone.lift * body_lift_weight(*ll) * mk;
                *a += tone.redness * mk;
                *b += tone.yellow_at(*lb) * body_yellow_weight(*ll) * mk;
            });
    }
    if timing {
        eprintln!(
            "  [cream/{tag}] tone: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 5b. 立体：脸内五官与轮廓尺度的明暗结构按比例加强（只在遮罩内取样：眼、眉、嘴、头发与背景不参与）
    if r.is_face && p.stereo > 0.0 {
        let t = Instant::now();
        let fine = masked_gaussian(&sub.l, &m, (STEREO_FINE * ed).max(1.0));
        let coarse = masked_gaussian(&sub.l, &m, (STEREO_COARSE * ed).max(1.0));
        sub.l
            .data
            .par_iter_mut()
            .zip(fine.data.par_iter().zip(&coarse.data))
            .zip(&m.data)
            .for_each(|((l, (f, c)), mk)| {
                if *mk > 0.0 {
                    *l = (*l + p.stereo * (f - c) * mk).clamp(0.0, 100.0);
                }
            });
        if timing {
            eprintln!(
                "  [cream/{tag}] stereo: {:.0} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    // 6. 眼部清晰（眼眶椭圆：内外眼角连线为长轴、上下眼睑距离 + 睫毛余量为短轴，羽化）
    if let (Some(g), true) = (&geom, r.is_face && p.eye_contrast > 0.0) {
        let mut em = GrayF32::new(bw, bh);
        for (outer, inner, top, bot) in [
            (g.eye_outer_l, g.eye_inner_l, g.eye_top_l, g.eye_bot_l),
            (g.eye_outer_r, g.eye_inner_r, g.eye_top_r, g.eye_bot_r),
        ] {
            let axis = inner.sub(outer);
            let len = axis.len();
            if len < 2.0 {
                continue;
            }
            let right = axis.mul(1.0 / len);
            let up = P::new(right.y, -right.x);
            let up = if up.dot(g.up) < 0.0 { up.mul(-1.0) } else { up };
            let c = outer.add(inner).mul(0.5);
            // 短轴只多留睫毛的余量：再往下就是泪沟 / 眼袋的褶线，锐化它会让眼袋更明显（像素蛋糕反而把那里压平）
            let rx = 0.62 * len;
            let ry = 0.5 * top.dist(bot) + 0.04 * g.ed;
            draw_soft_ellipse(&mut em, c, right, up, rx, ry, 1.0);
        }
        let sigma = (0.04 * g.ed).max(1.0);
        let blur = fast_gaussian(&sub.l, sigma);
        let k = p.eye_contrast;
        sub.l
            .data
            .par_iter_mut()
            .zip(&blur.data)
            .zip(&em.data)
            .for_each(|((l, b), w)| {
                if *w > 0.0 {
                    *l = (*l + k * (*l - b) * w).clamp(0.0, 100.0);
                }
            });
    }
    paste_gray(&mut planes.l, &sub.l, bx0, by0);
    paste_gray(&mut planes.a, &sub.a, bx0, by0);
    paste_gray(&mut planes.b, &sub.b, bx0, by0);
}

/// 各张脸的脖子区域（需要人脸解析；颈纹强度为 0 的脸跳过）。`planes` 为处理前的区域 Lab，
/// `body` / `skin_prob` 为区域坐标的身体遮罩与语义皮肤概率，`people[i]` 为第 i 张脸的参数与尺度。
/// 脖子像素按归属份额（[`owner_share`]，以各张脸的中心与尺度计）分给各人：贴脸、亲吻时一张脸的解析范围会盖到
/// 另一个人的脖子，按份额分摊后每个像素合计只处理一次，并用它主人的参数与尺度。
fn neck_zones(
    planes: &LabPlanes,
    origin: (usize, usize),
    body: &GrayF32,
    skin_prob: Option<&GrayF32>,
    faces: &[FaceKeyPoints],
    people: &[(&CreamParams, f32)],
) -> Vec<(usize, NeckZone)> {
    let off = P::new(origin.0 as f32, origin.1 as f32);
    let anchors: Vec<(P, f32)> = faces
        .iter()
        .zip(people)
        .map(|(f, &(_, ed))| (face_center(f).sub(off), ed.max(1.0)))
        .collect();
    faces
        .iter()
        .zip(people)
        .enumerate()
        .filter_map(|(i, (f, &(p, ed)))| {
            if p.neck.strength <= 0.0 {
                return None;
            }
            let parse = f.parse.as_ref()?;
            let rect = neck::parse_rect(parse, origin, (planes.w, planes.h))?;
            let mut allowed = crop_gray(body, rect.x0, rect.y0, rect.w, rect.h);
            let sp = skin_prob.map(|sp| crop_gray(sp, rect.x0, rect.y0, rect.w, rect.h));
            let shared = anchors.len() > 1;
            allowed.data.par_iter_mut().enumerate().for_each(|(j, a)| {
                if let Some(sp) = &sp {
                    *a *= semantic_skin_gate(sp.data[j]);
                }
                if shared && *a > 0.0 {
                    let pos = P::new((rect.x0 + j % rect.w) as f32, (rect.y0 + j / rect.w) as f32);
                    *a *= owner_share(&anchors, pos, i);
                }
            });
            neck::neck_zone(parse, origin, rect, planes, &allowed, ed, &p.neck).map(|z| (i, z))
        })
        .collect()
}

/// 立体的结构层：脸部遮罩内归一化低通 σ（× 瞳距）之差——细于 STEREO_FINE 的是纹理与斑驳（磨皮管），粗于
/// STEREO_COARSE 的是整张脸的受光（色调管），中间是鼻梁、颧骨、眼窝、下颌这些五官与轮廓的明暗
const STEREO_FINE: f32 = 0.06;
const STEREO_COARSE: f32 = 0.25;

/// 脸轮廓（关键点的下颌线 + 额头点围成的多边形）的羽化 σ（× 瞳距）与栅格化的目标尺度（缩到瞳距约 OUTLINE_ED 像素）
const OUTLINE_FEATHER: f32 = 0.04;
const OUTLINE_ED: f32 = 16.0;

/// 各张脸的轮廓区域（区域坐标，低分辨率），用来判断脸部遮罩的越界：一个人的解析遮罩落在别人脸轮廓内、又不在
/// 自己脸轮廓内的部分属于别人。轮廓都不覆盖的地方（耳朵、发际、遮罩的羽化边）不受影响。
struct FaceOutlines {
    small: Vec<GrayF32>,
    size: (usize, usize),
}

impl FaceOutlines {
    /// `off` 为区域左上角（全图坐标），`size` 为区域大小，`eds` 为各脸的瞳距当量。
    fn new(faces: &[FaceKeyPoints], off: P, size: (usize, usize), eds: &[f32]) -> Self {
        let ed_min = eds.iter().copied().fold(f32::INFINITY, f32::min).max(1.0);
        let k = (ed_min / OUTLINE_ED).floor().max(1.0);
        let (sw, sh) = (
            ((size.0 as f32 / k).ceil() as usize).max(1),
            ((size.1 as f32 / k).ceil() as usize).max(1),
        );
        let small = faces
            .iter()
            .zip(eds)
            .map(|(f, &ed)| {
                let poly: Vec<P> = f
                    .contour
                    .iter()
                    .chain(&f.forehead)
                    .map(|p| p.sub(off).mul(1.0 / k))
                    .collect();
                let mut m = GrayF32::new(sw, sh);
                fill_polygon(&mut m, &poly, 1.0);
                fast_gaussian(&m, (OUTLINE_FEATHER * ed / k).max(0.7))
            })
            .collect();
        Self { small, size }
    }

    /// 第 `k` 张脸的遮罩要去掉的份额（0..1，区域分辨率）：别人轮廓内 ×（1 − 自己轮廓内）。
    fn foreign(&self, k: usize) -> GrayF32 {
        let own = &self.small[k];
        let data = (0..own.data.len())
            .map(|i| {
                let other = self
                    .small
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != k)
                    .map(|(_, m)| m.data[i])
                    .fold(0.0f32, f32::max);
                other * (1.0 - own.data[i])
            })
            .collect();
        GrayF32::from_vec(own.w, own.h, data).resize(self.size.0, self.size.1)
    }
}

/// 人脸框的中心（全图坐标）。
fn face_center(f: &FaceKeyPoints) -> P {
    P::new((f.bbox.x1 + f.bbox.x2) * 0.5, (f.bbox.y1 + f.bbox.y2) * 0.5)
}

/// 各张脸的纹理尺度：姿态稳健的 `scale_distance`（侧脸时瞳距偏小会把频段整体移到过细的尺度）。
fn face_scales(faces: &[FaceKeyPoints]) -> Vec<f32> {
    faces.iter().map(|f| f.scale_distance().max(8.0)).collect()
}

/// 第 i 张脸的参数与尺度（参数缺省时用身体参数）。
fn people_of<'a>(
    face_eds: &[f32],
    face_params: &[&'a CreamParams],
    body_params: &'a CreamParams,
) -> Vec<(&'a CreamParams, f32)> {
    face_eds
        .iter()
        .enumerate()
        .map(|(i, &ed)| (face_params.get(i).copied().unwrap_or(body_params), ed))
        .collect()
}

/// 颈纹淡化的处理权重（全图坐标，0..1；调试 / 可视化用，与 [`cream_skin`] 的区域一致）。
pub fn neck_weights(
    orig: &ImgF32,
    masks: &SkinMasks,
    faces: &[FaceKeyPoints],
    face_params: &[&CreamParams],
    body_params: &CreamParams,
) -> GrayF32 {
    let mut out = GrayF32::new(orig.w, orig.h);
    let (Some((x0, y0, x1, y1)), Some(body)) = (masks.roi, masks.body.as_ref()) else {
        return out;
    };
    let (cw, ch) = (x1 - x0, y1 - y0);
    if cw == 0 || ch == 0 {
        return out;
    }
    let planes = LabPlanes::from_img(&crop_img(orig, x0, y0, cw, ch));
    let body = crop_gray(body, x0, y0, cw, ch);
    let skin_prob = masks
        .skin_prob
        .as_ref()
        .map(|sp| crop_gray(sp, x0, y0, cw, ch));
    let face_eds = face_scales(faces);
    let people = people_of(&face_eds, face_params, body_params);
    for (_, z) in neck_zones(&planes, (x0, y0), &body, skin_prob.as_ref(), faces, &people) {
        let (zx, zy) = (x0 + z.rect.x0, y0 + z.rect.y0);
        for y in 0..z.rect.h {
            for x in 0..z.rect.w {
                let o = &mut out.data[(zy + y) * orig.w + zx + x];
                *o = o.max(z.weight.data[y * z.rect.w + x]);
            }
        }
    }
    out
}

/// 应用奶油肌。`faces[i]`（全图坐标）给出第 i 张脸的关键点（瞳距、眼下区域、五官保护区），
/// `face_params[i]` 给出其参数（可按性别不同）。`faces` 可少于遮罩数（缺省用平均瞳距、无保护区）。
/// 身体区域用 `body_params` 处理，其中色调（提亮 / 降红 / 降黄）按像素归属混合 `face_params` 各人的身体色调；
/// 每张脸的脖子再按该脸的参数做颈纹淡化（需要人脸解析与身体遮罩）。
pub fn cream_skin(
    orig: &ImgF32,
    masks: &SkinMasks,
    faces: &[FaceKeyPoints],
    face_params: &[&CreamParams],
    body_params: &CreamParams,
) -> ImgF32 {
    let timing = std::env::var("RETOUCH_TIMING").is_ok();
    let mut out = orig.clone();
    let Some((x0, y0, x1, y1)) = masks.roi else {
        return out;
    };
    let (cw, ch) = (x1 - x0, y1 - y0);
    if cw == 0 || ch == 0 {
        return out;
    }
    let t = Instant::now();
    let sub = crop_img(orig, x0, y0, cw, ch);
    let mut planes = LabPlanes::from_img(&sub);
    if timing {
        eprintln!(
            "  [cream] roi {cw}x{ch} to Lab: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    // 纹理 / 瑕疵的尺度用姿态稳健的脸部尺度；眼周等保护区的几何仍按真实瞳距（FaceGeom.ed）
    let face_eds = face_scales(faces);
    let ed_mean = if face_eds.is_empty() {
        (orig.w.min(orig.h) as f32) * 0.06
    } else {
        face_eds.iter().sum::<f32>() / face_eds.len() as f32
    };
    let off = P::new(x0 as f32, y0 as f32);
    let skin_prob = masks
        .skin_prob
        .as_ref()
        .map(|sp| crop_gray(sp, x0, y0, cw, ch));
    let body_mask = masks.body.as_ref().map(|b| crop_gray(b, x0, y0, cw, ch));
    let people = people_of(&face_eds, face_params, body_params);
    // 颈纹：脖子区域（异物门控、细线）在处理前的颜色上确定，身体处理完后再平滑、填平
    let t = Instant::now();
    let necks = match &body_mask {
        Some(b) => neck_zones(&planes, (x0, y0), b, skin_prob.as_ref(), faces, &people),
        None => Vec::new(),
    };
    let neck_zones_ms = t.elapsed().as_secs_f64() * 1e3;
    // 身体色调外溢：在处理前的颜色上确定（见 `skin::spill`）
    let spill = body_mask
        .as_ref()
        .zip(masks.person.as_ref())
        .filter(|_| body_params.body_tone_spill > 0.0)
        .map(|(body, person)| {
            let mut faces_roi = GrayF32::new(cw, ch);
            for fm in &masks.faces {
                faces_roi.max_inplace(&crop_gray(fm, x0, y0, cw, ch));
            }
            let person_roi = crop_gray(person, x0, y0, cw, ch);
            let open = [x0 == 0, y0 == 0, x1 == orig.w, y1 == orig.h];
            spill::tone_spill(&planes, body, &faces_roi, &person_roi, ed_mean, open)
        });
    let strength = body_params.body_tone_spill;
    let body_tone_mask = body_mask
        .as_ref()
        .zip(spill.as_ref())
        .map(|(b, sp)| spill::with_spill(b, sp, strength));
    let outlines = (body_params.face_ownership && faces.len() > 1)
        .then(|| FaceOutlines::new(faces, off, (cw, ch), &face_eds));
    for (i, fm) in masks.faces.iter().enumerate() {
        let mut m = crop_gray(fm, x0, y0, cw, ch);
        if let Some(o) = outlines.as_ref().filter(|_| i < faces.len()) {
            let foreign = o.foreign(i);
            m.data
                .par_iter_mut()
                .zip(&foreign.data)
                .for_each(|(v, f)| *v *= 1.0 - f);
        }
        let params = face_params.get(i).copied().unwrap_or(body_params);
        process_region(
            &mut planes,
            &Region {
                mask: &m,
                ed: face_eds.get(i).copied().unwrap_or(ed_mean),
                params,
                is_face: true,
                geom: faces.get(i).map(|f| FaceGeom::from_face(f, off)),
                skin_prob: skin_prob.as_ref(),
                body_tones: &[],
                tone_mask: None,
            },
            timing,
        );
    }
    if let Some(m) = &body_mask {
        let body_tones: Vec<BodyTone> = faces
            .iter()
            .zip(&people)
            .map(|(f, &(params, scale))| BodyTone::of(params, face_center(f).sub(off), scale))
            .collect();
        process_region(
            &mut planes,
            &Region {
                mask: m,
                ed: ed_mean,
                params: body_params,
                is_face: false,
                geom: None,
                skin_prob: skin_prob.as_ref(),
                body_tones: &body_tones,
                tone_mask: body_tone_mask.as_ref(),
            },
            timing,
        );
    }
    let t = Instant::now();
    for (i, zone) in &necks {
        let (p, ed) = people[*i];
        let fine = (p.fine_sigma * ed).max(FINE_SIGMA_FLOOR_PX);
        neck::soften_neck(&mut planes.l, zone, ed, &p.neck, fine, p.mid_sigma * ed);
    }
    if timing && !necks.is_empty() {
        eprintln!(
            "  [cream/neck] {} zone(s): zones {:.0} ms, smoothing {:.0} ms",
            necks.len(),
            neck_zones_ms,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let t = Instant::now();
    let mut union = crop_gray(&masks.union, x0, y0, cw, ch);
    if let Some(sp) = &spill {
        union = spill::with_spill(&union, sp, strength);
    }
    let mut sub_out = sub;
    planes.blend_into(&mut sub_out, &union);
    out.data
        .par_chunks_mut(orig.w)
        .enumerate()
        .skip(y0)
        .take(ch)
        .for_each(|(y, row)| {
            let sy = y - y0;
            row[x0..x1].copy_from_slice(&sub_out.data[sy * cw..(sy + 1) * cw]);
        });
    if timing {
        eprintln!(
            "  [cream] blend back: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::GrayF32;

    #[test]
    fn outside_mask_is_untouched_and_inside_is_smoothed() {
        let (w, h) = (120, 100);
        let mut img = ImgF32::filled(w, h, [0.78, 0.60, 0.52]);
        for (i, p) in img.data.iter_mut().enumerate() {
            let n = ((i * 7919) % 11) as f32 / 11.0 * 0.04 - 0.02;
            p[0] += n;
            p[1] += n;
            p[2] += n;
        }
        let mut m = GrayF32::new(w, h);
        for y in 20..80 {
            for x in 20..100 {
                m.data[y * w + x] = 1.0;
            }
        }
        let masks = SkinMasks {
            faces: vec![m.clone()],
            body: None,
            union: m.clone(),
            roi: Some((10, 10, 110, 90)),
            skin_prob: None,
            person: None,
        };
        let p = CreamParams::default();
        let face = FaceKeyPoints {
            pupil_l: P::new(40.0, 40.0),
            pupil_r: P::new(80.0, 40.0),
            nose_bridge_top: P::new(60.0, 40.0),
            chin: P::new(60.0, 95.0),
            ..Default::default()
        };
        let out = cream_skin(&img, &masks, &[face], &[&p], &p);
        assert_eq!(out.data[5 * w + 5], img.data[5 * w + 5]);
        let var = |im: &ImgF32| {
            let vals: Vec<f32> = (30..70)
                .flat_map(|y| (30..90).map(move |x| (x, y)))
                .map(|(x, y)| im.get(x, y)[0])
                .collect();
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32
        };
        assert!(
            var(&out) < var(&img) * 0.8,
            "{} vs {}",
            var(&out),
            var(&img)
        );
    }

    #[test]
    fn stereo_deepens_facial_shading_only_inside_the_mask() {
        // 脸内一个柔和的亮包（颧骨 / 鼻梁一级，σ = 0.15 瞳距）：开立体后中心更亮、周围更暗，遮罩外逐位不变
        let (w, h) = (120, 100);
        let mut img = ImgF32::filled(w, h, [0.70, 0.55, 0.48]);
        for y in 0..h {
            for x in 0..w {
                let r2 = (x as f32 - 60.0).powi(2) + (y as f32 - 50.0).powi(2);
                let bump = 0.08 * (-r2 / (2.0 * 6.0 * 6.0)).exp();
                for c in &mut img.data[y * w + x] {
                    *c += bump;
                }
            }
        }
        let mut m = GrayF32::new(w, h);
        for y in 15..85 {
            for x in 15..105 {
                m.data[y * w + x] = 1.0;
            }
        }
        let masks = SkinMasks {
            faces: vec![m.clone()],
            body: None,
            union: m.clone(),
            roi: Some((5, 5, 115, 95)),
            skin_prob: None,
            person: None,
        };
        let face = FaceKeyPoints {
            pupil_l: P::new(40.0, 40.0),
            pupil_r: P::new(80.0, 40.0),
            nose_bridge_top: P::new(60.0, 40.0),
            chin: P::new(60.0, 95.0),
            ..Default::default()
        };
        let flat = CreamParams::default();
        let stereo = CreamParams {
            stereo: 0.4,
            ..CreamParams::default()
        };
        let a = cream_skin(&img, &masks, std::slice::from_ref(&face), &[&flat], &flat);
        let b = cream_skin(
            &img,
            &masks,
            std::slice::from_ref(&face),
            &[&stereo],
            &stereo,
        );
        let l = |im: &ImgF32, x: usize, y: usize| crate::color::lab::rgb_to_lab(im.get(x, y))[0];
        assert!(l(&b, 60, 50) > l(&a, 60, 50) + 0.3, "centre brighter");
        assert!(l(&b, 60, 64) < l(&a, 60, 64), "surroundings darker");
        assert_eq!(b.get(2, 2), img.get(2, 2), "outside the mask");
    }

    #[test]
    fn body_lift_is_flat_in_shadows_and_fades_in_highlights() {
        // 阴影与中间调提得一样多（腋下、背光面不比周围暗），高光渐弱
        assert!((body_lift_weight(50.0) - 1.0).abs() < 1e-6);
        assert!((body_lift_weight(62.0) - 1.0).abs() < 1e-6);
        assert!(body_lift_weight(40.0) > 0.95);
        assert!(body_lift_weight(10.0) >= BODY_LIFT_SHADOW_KEEP - 1e-6);
        assert!((body_lift_weight(95.0) - BODY_LIFT_HIGHLIGHT_KEEP).abs() < 1e-6);
        let mut prev = f32::INFINITY;
        for l in (44..=100).step_by(4) {
            let w = body_lift_weight(l as f32);
            assert!(w <= prev + 1e-6, "non-increasing above the shadows");
            prev = w;
        }
    }

    #[test]
    fn body_tone_does_not_deepen_dark_folds() {
        // 暗部皮肤上的细褶纹（2 px 暗 / 2 px 亮的条纹）：只做身体色调时整体提亮，但褶纹的明暗差不变
        let (w, h) = (400, 300);
        let mut img = ImgF32::new(w, h);
        for (i, p) in img.data.iter_mut().enumerate() {
            let v = if (i % w / 2) % 2 == 0 { 0.30 } else { 0.40 };
            *p = [v * 1.10, v, v * 0.90];
        }
        let mut body = GrayF32::new(w, h);
        for y in 40..260 {
            for x in 40..360 {
                body.data[y * w + x] = 1.0;
            }
        }
        let masks = SkinMasks {
            faces: vec![],
            body: Some(body.clone()),
            union: body,
            roi: Some((0, 0, w, h)),
            skin_prob: None,
            person: None,
        };
        let mut p = CreamParams {
            body_smooth: 0.0,
            body_low_smooth: 0.0,
            body_fine_smooth: 0.0,
            body_chroma_smooth: 0.0,
            body_blemish: false,
            ..CreamParams::default()
        };
        p.body_heal.strength = 0.0;
        // 一张脸只用来给出尺度（瞳距 200 → 低频 σ 6 px，远大于条纹周期）与身体色调的归属
        let face = FaceKeyPoints {
            pupil_l: P::new(100.0, 20.0),
            pupil_r: P::new(300.0, 20.0),
            nose_bridge_top: P::new(200.0, 20.0),
            chin: P::new(200.0, 60.0),
            ..Default::default()
        };
        let out = cream_skin(&img, &masks, &[face], &[&p], &p);
        let stripes = |im: &ImgF32| {
            let lab = LabPlanes::from_img(im);
            let (mut dark, mut bright, mut n) = (0.0f32, 0.0f32, 0.0f32);
            for y in 100..200 {
                for x in (100..300).step_by(4) {
                    dark += lab.l.data[y * w + x];
                    bright += lab.l.data[y * w + x + 2];
                    n += 1.0;
                }
            }
            (dark / n, bright / n)
        };
        let (d0, b0) = stripes(&img);
        let (d1, b1) = stripes(&out);
        assert!(
            d1 - d0 > 0.9 * p.body_lift,
            "dark folds lifted like their surroundings: {d0} → {d1}"
        );
        let (c0, c1) = (b0 - d0, b1 - d1);
        assert!(
            (c1 - c0).abs() < 0.02 * c0,
            "fold contrast unchanged: {c0} → {c1}"
        );
    }

    #[test]
    fn body_tone_follows_the_nearest_person() {
        let her = BodyTone::of(&CreamParams::default(), P::new(0.0, 0.0), 100.0);
        let him = BodyTone::of(&CreamParams::male_default(), P::new(1000.0, 0.0), 100.0);
        let fallback = Tone {
            lift: 1.0,
            ..Tone::default()
        };
        assert_eq!(body_tone_at(&[], P::new(5.0, 5.0), fallback), fallback);
        let near_her = body_tone_at(&[her, him], P::new(150.0, 200.0), fallback);
        assert!((near_her.lift - her.tone.lift).abs() < 0.01, "{near_her:?}");
        let near_him = body_tone_at(&[her, him], P::new(900.0, 300.0), fallback);
        assert!((near_him.lift - him.tone.lift).abs() < 0.01, "{near_him:?}");
        // 两人中间平滑过渡：各项取平均，黄度改变量也是两人的平均
        let mid = body_tone_at(&[her, him], P::new(500.0, 0.0), fallback);
        assert!(
            (mid.lift - 0.5 * (her.tone.lift + him.tone.lift)).abs() < 1e-3,
            "{mid:?}"
        );
        for b in [0.0, 10.0, 20.0] {
            let avg = 0.5 * (her.tone.yellow_at(b) + him.tone.yellow_at(b));
            assert!((mid.yellow_at(b) - avg).abs() < 1e-3, "b {b}: {mid:?}");
        }
        // 参数相同（或没有来源）时是常数，不必逐像素计算
        assert_eq!(uniform_body_tone(&[], fallback), Some(fallback));
        assert_eq!(uniform_body_tone(&[her, her], fallback), Some(her.tone));
        assert_eq!(uniform_body_tone(&[her, him], fallback), None);
    }

    #[test]
    fn body_yellow_pulls_towards_the_target() {
        // 女性：偏黄的皮肤降黄，偏蓝的（天光下）略加暖，目标黄度处只剩偏移；男性：常数偏移
        let her = Tone::of(&CreamParams::default());
        let p = CreamParams::default();
        assert!((her.yellow_at(p.body_target_b) - p.body_yellow).abs() < 1e-5);
        assert!(her.yellow_at(20.0) < -3.0 && her.yellow_at(-2.0) > 0.5);
        let him = Tone::of(&CreamParams::male_default());
        assert_eq!(him.yellow_at(0.0), him.yellow_at(20.0));
    }

    #[test]
    fn presets_without_the_body_pull_keep_their_constant_yellow() {
        // 0.6.3 之前导出的预设没有 body_pull_b：只用文件里的常数降黄，不再叠加默认的比例降黄
        let old: CreamParams = serde_json::from_str(r#"{"body_yellow": -2.2}"#).unwrap();
        assert_eq!(old.body_pull_b, 0.0);
        assert_eq!(Tone::of(&old).yellow_at(15.0), -2.2);
        // 新导出的预设写出了这一项，读回不变
        let p = CreamParams::default();
        let back: CreamParams = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(back.body_pull_b, p.body_pull_b);
    }

    #[test]
    fn face_outlines_remove_only_the_part_inside_another_face() {
        // 两张方脸（轮廓 20..60、70..110），第 0 张脸的解析遮罩越界盖到第 1 张脸上：越界处份额 ≈ 1，
        // 自己脸上与两张脸都不覆盖的地方为 0
        let square = |x0: f32| FaceKeyPoints {
            contour: vec![
                P::new(x0, 20.0),
                P::new(x0 + 40.0, 20.0),
                P::new(x0 + 40.0, 60.0),
                P::new(x0, 60.0),
            ],
            ..Default::default()
        };
        let faces = [square(20.0), square(70.0)];
        let o = FaceOutlines::new(&faces, P::new(0.0, 0.0), (130, 80), &[40.0, 40.0]);
        let f0 = o.foreign(0);
        assert!(f0.get(90, 40) > 0.95, "{}", f0.get(90, 40));
        assert!(f0.get(40, 40) < 0.01, "{}", f0.get(40, 40));
        assert!(f0.get(40, 75) < 0.01 && f0.get(125, 40) < 0.01);
        let f1 = o.foreign(1);
        assert!(f1.get(40, 40) > 0.95 && f1.get(90, 40) < 0.01);
    }

    #[test]
    fn owner_share_splits_between_people_and_sums_to_one() {
        let anchors = [(P::new(0.0, 0.0), 100.0), (P::new(1000.0, 0.0), 100.0)];
        let at = |x: f32, y: f32| {
            let pos = P::new(x, y);
            (owner_share(&anchors, pos, 0), owner_share(&anchors, pos, 1))
        };
        // 自己的脖子（脸下方）几乎全归自己；两人正中各半；份额之和为 1
        let (a, b) = at(80.0, 250.0);
        assert!(a > 0.999 && b < 0.001, "{a} {b}");
        let (a, b) = at(500.0, 300.0);
        assert!((a - 0.5).abs() < 1e-5 && (b - 0.5).abs() < 1e-5, "{a} {b}");
        for x in [0.0, 300.0, 480.0, 520.0, 900.0] {
            let (a, b) = at(x, 120.0);
            assert!((a + b - 1.0).abs() < 1e-5, "{x}: {a} + {b}");
        }
        // 只有一个人时全归他
        assert_eq!(owner_share(&anchors[..1], P::new(700.0, 0.0), 0), 1.0);
    }
}
