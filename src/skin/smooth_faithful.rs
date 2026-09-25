//! 磨皮方案 A：忠实还原 `BBGPUImageBeautifyFilter`
//! （双边 → Sobel → 三输入组合 → log 提亮 → HSB）。
//!
//! 分辨率策略：双边与边缘在"短边 720"的工作副本上计算，再上采样到全分辨率与原图组合，
//! 保证任意分辨率下观感与原 720×1280 Demo 一致，且预览/导出一致。

use crate::buffer::{GrayF32, ImgF32};
use crate::color::hsb::hsb_brightness_saturation;
use crate::skin::bilateral::{bilateral_gpuimage, DEFAULT_DNF, DEFAULT_SPACING};
use crate::skin::edge::sobel_edge;
use crate::skin::mask::is_skin_rgb;
use rayon::prelude::*;

/// 只依赖原图、不依赖滑块的中间结果（可缓存）。
#[derive(Clone, Debug)]
pub struct FaithfulPrecomp {
    /// 全分辨率双边结果
    pub bilateral: ImgF32,
    /// 全分辨率边缘图
    pub edge: GrayF32,
}

/// 在短边 `work_short` 的工作副本上计算双边 + 边缘并上采样。
pub fn precompute_faithful(orig: &ImgF32, work_short: f32) -> FaithfulPrecomp {
    let (work, scale) = orig.work_copy(work_short);
    let bl_w = bilateral_gpuimage(&work, DEFAULT_SPACING, DEFAULT_DNF);
    let edge_w = sobel_edge(&work);
    if scale < 1.0 {
        FaithfulPrecomp {
            bilateral: bl_w.resize(orig.w, orig.h),
            edge: edge_w.resize(orig.w, orig.h),
        }
    } else {
        FaithfulPrecomp {
            bilateral: bl_w,
            edge: edge_w,
        }
    }
}

/// 三输入组合 shader 直译 + 可选人脸遮罩 + log 提亮。
/// `face_mask` 为 None 表示不限制（忠实模式）。
pub fn combine_bb(
    origin: &ImgF32,
    bilateral: &ImgF32,
    edge: &GrayF32,
    face_mask: Option<&GrayF32>,
    smooth_degree: f32,
    apply_log: bool,
) -> ImgF32 {
    assert_eq!((origin.w, origin.h), (bilateral.w, bilateral.h));
    assert_eq!((origin.w, origin.h), (edge.w, edge.h));
    let ln12 = 1.2f32.ln();
    let mut out = ImgF32::new(origin.w, origin.h);
    out.data.par_iter_mut().enumerate().for_each(|(i, px)| {
        let o = origin.data[i];
        let bl = bilateral.data[i];
        let m = face_mask.map(|m| m.data[i]).unwrap_or(1.0);
        let mut s = if edge.data[i] < 0.2 && is_skin_rgb(o) && m > 0.0 {
            let mut t = [0.0f32; 3];
            for c in 0..3 {
                let v = bl[c] + (1.0 - smooth_degree) * (o[c] - bl[c]);
                t[c] = o[c] + (v - o[c]) * m;
            }
            t
        } else {
            o
        };
        if apply_log {
            for v in s.iter_mut() {
                *v = (1.0 + 0.2 * *v).ln() / ln12;
            }
        }
        *px = s;
    });
    out
}

/// 方案 A 组装：组合 + log + HSB。
pub fn smooth_faithful(
    orig: &ImgF32,
    pre: &FaithfulPrecomp,
    face_mask: Option<&GrayF32>,
    smooth: f32,
    apply_log: bool,
    brightness: f32,
    saturation: f32,
) -> ImgF32 {
    let mut out = combine_bb(
        orig,
        &pre.bilateral,
        &pre.edge,
        face_mask,
        smooth,
        apply_log,
    );
    hsb_brightness_saturation(&mut out, brightness, saturation);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_one_equals_bilateral_on_skin_and_origin_elsewhere() {
        let mut img = ImgF32::new(32, 8);
        for y in 0..8 {
            for x in 0..32 {
                // 左半肤色渐变，右半灰色（非肤色）
                let v = 0.6 + 0.1 * (x % 4) as f32 / 4.0;
                img.data[y * 32 + x] = if x < 16 {
                    [v, 0.55, 0.45]
                } else {
                    [0.5, 0.5, 0.5]
                };
            }
        }
        let pre = precompute_faithful(&img, 720.0);
        let out = combine_bb(&img, &pre.bilateral, &pre.edge, None, 1.0, false);
        let mut checked = 0;
        for y in 0..8 {
            for x in 0..32 {
                let i = y * 32 + x;
                let o = img.data[i];
                if is_skin_rgb(o) && pre.edge.data[i] < 0.2 {
                    assert!((out.data[i][0] - pre.bilateral.data[i][0]).abs() < 1e-6);
                    checked += 1;
                } else {
                    assert_eq!(out.data[i], o);
                }
            }
        }
        assert!(checked > 0);
    }
}
