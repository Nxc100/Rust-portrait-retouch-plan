//! 牙齿美白：人脸解析的口腔内部里，亮而不红的像素（牙齿——牙龈、舌头、嘴唇偏红，口腔深处偏暗）在 Lab 中
//! 把黄度、红度拉向暖白，亮度向白提一截。
//!
//! 像素蛋糕「婚纱-深色内景」（牙齿美白 0.3 + 牙齿祛瑕疵 0.75）对调色后的牙齿（1V3A3101、1V3A2954 两位露齿的
//! 新娘，光流对齐后逐像素比较）：黄度 b 27.6 → 10.8、红度 a 18.2 → 10.1（本来就不黄的 b 7.9 不动），亮度
//! +3 ~ +9、越暗提得越多（约为 0.15 ×（100 − L））；亮度起伏不变，没有额外的匀化
//! （doc/test_report_wedding_dark_interior.md）。像素蛋糕奶油肌也美白牙齿（"牙齿美白 0.4"），本项目奶油肌取 0.3
//! （doc/test_report_final.md §4）。
//! 没有人脸解析（`FaceKeyPoints::parse`）时不处理：只靠颜色分不开牙齿与唇边、皮肤。

use crate::buffer::{GrayF32, ImgF32};
use crate::color::lab::{lab_to_rgb, rgb_to_lab};
use crate::face::parsing::CLS_MOUTH;
use crate::face::semantic::FaceKeyPoints;
use crate::skin::guided::fast_gaussian;
use crate::skin::smoothstep;
use rayon::prelude::*;

/// 拉向的目标：红度、黄度（Lab；只往下拉，本来就不红 / 不黄的不动）
const TARGET_A: f32 = 10.0;
const TARGET_B: f32 = 10.0;
/// 亮度向 100 拉的比例
const LIFT_L: f32 = 0.18;
/// 牙齿的颜色门限：亮度 L_IN→L_FULL 渐入（口腔深处的暗部不动）；红度比「牙齿参考红度」高 A_KEEP→A_OUT 时渐出
/// （牙龈、舌头、嘴唇）。参考红度取口腔里最亮的 REF_TOP 那部分像素（一定是牙齿）的中位数：场景偏色（暖色
/// 内景、红裙映在牙齿上）时门限跟着移，靠近牙龈、被唇色映红的牙齿也不漏掉。
const L_IN: f32 = 35.0;
const L_FULL: f32 = 50.0;
const A_KEEP: f32 = 8.0;
const A_OUT: f32 = 16.0;
const REF_TOP: f32 = 0.25;
/// 牙齿边缘：紧挨牙齿、被牙龈与唇色映红的一圈（相对红度 A_OUT→A_RIM_OUT 渐出，嘴唇更红、不动）按牙齿权重
/// 羽化（σ = RIM × 嘴宽）再乘 2 的值处理：贴着牙齿接近全权重，约 3σ 外归零——像素蛋糕对这一圈也减红减黄
/// 约一半，不然白牙镶一道橙边
const A_RIM_OUT: f32 = 24.0;
const RIM: f32 = 0.03;
/// 口腔里够亮（L ≥ L_FULL）的像素少于 MIN_TEETH × 嘴宽² 时视为没露齿
const MIN_TEETH: f32 = 0.01;
/// 处理范围：嘴角连线中点起，横向 ± HALF_W × 嘴宽、纵向 ± HALF_H × 嘴宽
const HALF_W: f32 = 0.8;
const HALF_H: f32 = 0.6;
/// 口腔遮罩的羽化（× 嘴宽）
const FEATHER: f32 = 0.03;

/// 一张脸的牙齿权重：矩形 (`x0`, `y0`) 起、`weights` 大小的区域内，每个像素的美白拉力（已乘强度）。
#[derive(Clone, Debug)]
pub struct TeethWeights {
    pub x0: usize,
    pub y0: usize,
    pub weights: GrayF32,
}

impl TeethWeights {
    /// 写进全图大小的权重图（取最大值，多张脸叠加用）。
    pub fn max_into(&self, all: &mut GrayF32) {
        let w = self.weights.w;
        for (dy, row) in self.weights.data.chunks(w).enumerate() {
            let o = (self.y0 + dy) * all.w + self.x0;
            for (d, v) in all.data[o..o + w].iter_mut().zip(row) {
                *d = d.max(*v);
            }
        }
    }
}

/// 按权重做牙齿美白（原地）。权重通常在修图前的图上算（[`teeth_weights`]），作用在修图后的同几何图像上。
pub fn whiten_teeth(img: &mut ImgF32, tw: &TeethWeights) {
    let (w, iw) = (tw.weights.w, img.w);
    img.data
        .par_chunks_mut(iw)
        .skip(tw.y0)
        .take(tw.weights.h)
        .enumerate()
        .for_each(|(dy, row)| {
            for (dx, p) in row[tw.x0..tw.x0 + w].iter_mut().enumerate() {
                let t = tw.weights.data[dy * w + dx];
                if t > 0.0 {
                    *p = whiten(*p, t);
                }
            }
        });
}

/// 牙齿权重（含强度）：人脸解析的口腔区域 × 颜色门限（用 `img` 的颜色），再加上牙齿边缘那一圈。没有解析图、
/// 强度为 0、嘴太小或没露齿时为 None。
pub fn teeth_weights(img: &ImgF32, face: &FaceKeyPoints, strength: f32) -> Option<TeethWeights> {
    let parse = face.parse.as_ref()?;
    let strength = strength.clamp(0.0, 1.0);
    let width = face.mouth_l.dist(face.mouth_r);
    if strength <= 0.0 || width < 4.0 {
        return None;
    }
    let c = face.mouth_l.add(face.mouth_r).mul(0.5);
    let clamp_x = |v: f32| v.clamp(0.0, img.w as f32) as usize;
    let clamp_y = |v: f32| v.clamp(0.0, img.h as f32) as usize;
    let (x0, x1) = (clamp_x(c.x - HALF_W * width), clamp_x(c.x + HALF_W * width));
    let (y0, y1) = (clamp_y(c.y - HALF_H * width), clamp_y(c.y + HALF_H * width));
    if x1 <= x0 + 1 || y1 <= y0 + 1 {
        return None;
    }
    let (w, h) = (x1 - x0, y1 - y0);
    let mouth = fast_gaussian(
        &parse.mask_in(x0, y0, w, h, &[CLS_MOUTH]),
        (FEATHER * width).max(0.7),
    );
    // 矩形内每个像素的（口腔遮罩, L, a）
    let px: Vec<(f32, f32, f32)> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let [l, a, _] = rgb_to_lab(img.data[(y0 + i / w) * img.w + x0 + i % w]);
            (mouth.data[i], l, a)
        })
        .collect();
    let a_ref = reference_redness(&px, MIN_TEETH * width * width)?;
    let core = GrayF32::from_vec(
        w,
        h,
        px.par_iter()
            .map(|&(m, l, a)| m * gate(l, a - a_ref, A_OUT))
            .collect(),
    );
    let spread = fast_gaussian(&core, (RIM * width).max(0.7));
    let data = px
        .par_iter()
        .zip(core.data.par_iter().zip(&spread.data))
        .map(|(&(_, l, a), (&c, &s))| {
            strength * c.max((2.0 * s).min(1.0) * gate(l, a - a_ref, A_RIM_OUT))
        })
        .collect();
    Some(TeethWeights {
        x0,
        y0,
        weights: GrayF32::from_vec(w, h, data),
    })
}

/// 牙齿参考红度：口腔核心（遮罩 ≥ 0.5）里够亮的像素按亮度取最亮的 REF_TOP，红度的中位数。够亮的像素少于
/// `min_px` 时为 None（没露齿）。
fn reference_redness(px: &[(f32, f32, f32)], min_px: f32) -> Option<f32> {
    let mut bright: Vec<(f32, f32)> = px
        .iter()
        .filter(|&&(m, l, _)| m >= 0.5 && l >= L_FULL)
        .map(|&(_, l, a)| (l, a))
        .collect();
    if (bright.len() as f32) < min_px.max(1.0) {
        return None;
    }
    bright.sort_unstable_by(|p, q| q.0.total_cmp(&p.0));
    let top = ((bright.len() as f32 * REF_TOP).ceil() as usize).max(1);
    let mut a: Vec<f32> = bright[..top].iter().map(|p| p.1).collect();
    let mid = a.len() / 2;
    Some(*a.select_nth_unstable_by(mid, f32::total_cmp).1)
}

/// 颜色门限：够亮（不是口腔深处）且不比牙齿红太多（不是牙龈、舌头、嘴唇）。`da` 为相对参考红度的红度，
/// 在 A_KEEP→`a_out` 渐出。
#[inline]
fn gate(l: f32, da: f32, a_out: f32) -> f32 {
    smoothstep(L_IN, L_FULL, l) * (1.0 - smoothstep(A_KEEP, a_out, da))
}

/// 权重 `t` 下的美白：红度、黄度拉向目标（只往下拉），亮度向 100 拉 `t × LIFT_L`。
#[inline]
fn whiten(p: [f32; 3], t: f32) -> [f32; 3] {
    let [l, a, b] = rgb_to_lab(p);
    let l2 = l + t * LIFT_L * (100.0 - l).max(0.0);
    let a2 = a - t * (a - TARGET_A).max(0.0);
    let b2 = b - t * (b - TARGET_B).max(0.0);
    lab_to_rgb([l2, a2, b2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yellow_teeth_turn_warm_white() {
        // 偏黄的牙齿（Lab ≈ 72/18/27）拉力 0.8：a → 11.6、b → 13.4，亮度 +4
        let teeth = lab_to_rgb([72.0, 18.0, 27.0]);
        let [l, a, b] = rgb_to_lab(whiten(teeth, 0.8));
        assert!((l - 76.0).abs() < 0.5, "{l}");
        assert!((a - 11.6).abs() < 0.6, "{a}");
        assert!((b - 13.4).abs() < 0.6, "{b}");
        // 已经很白的牙齿不被拉黄、拉红，只提一点亮度
        let white = lab_to_rgb([88.0, 1.0, 2.0]);
        let [l, a, b] = rgb_to_lab(whiten(white, 1.0));
        assert!((l - 90.2).abs() < 0.5, "{l}");
        assert!((a - 1.0).abs() < 0.3 && (b - 2.0).abs() < 0.3, "{a} {b}");
    }

    #[test]
    fn colour_gate_skips_gums_and_the_dark_mouth() {
        // 参考红度 16（暖色调后的牙齿）：映红的牙齿（a 24）大部分保留，牙龈、舌头（a 35）不动，暗处不动
        assert!(gate(80.0, 0.0, A_OUT) > 0.99, "teeth");
        assert!(
            gate(60.0, 24.0 - 16.0, A_OUT) > 0.99,
            "reddish teeth near the gums"
        );
        assert_eq!(gate(55.0, 35.0 - 16.0, A_OUT), 0.0, "gums / tongue");
        assert_eq!(gate(25.0, 0.0, A_OUT), 0.0, "dark cavity");
        // 牙齿边缘那一圈放宽，嘴唇（a 45）仍不动
        assert!(
            gate(50.0, 30.0 - 16.0, A_RIM_OUT) > 0.5,
            "gum line next to the teeth"
        );
        assert_eq!(gate(50.0, 45.0 - 16.0, A_RIM_OUT), 0.0, "lips");
    }

    #[test]
    fn reference_redness_follows_the_brightest_mouth_pixels() {
        // 亮牙（L 80，a 16）、映红的暗牙（L 60，a 26）、口腔深处（L 20）；遮罩边缘（m 0.3）的唇色不参与
        let mut px = vec![(1.0, 80.0, 16.0); 40];
        px.extend(vec![(1.0, 60.0, 26.0); 120]);
        px.extend(vec![(1.0, 20.0, 30.0); 100]);
        px.extend(vec![(0.3, 85.0, 40.0); 50]);
        assert_eq!(reference_redness(&px, 10.0), Some(16.0));
        // 够亮的像素太少：没露齿
        assert_eq!(reference_redness(&px, 1000.0), None);
    }
}
