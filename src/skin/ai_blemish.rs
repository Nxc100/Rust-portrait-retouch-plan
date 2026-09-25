//! AI 瑕疵祛除：ModelScope `iic/cv_unet_skin-retouching`（ABPN 家族，Apache-2.0；Lei 等 CVPR 2022）
//! 的"去瑕疵"两段网络——瑕疵分割 UNet（768×768 输入）+ 门控卷积修复网络（576 窗口、步长 512、32 px 重叠）。
//!
//! 与 ModelScope `SkinRetouchingPipeline.retouch_local` 逐步一致（分析见 doc/analysis/abpn.md）：
//! 1. ROI：以人脸框中心为中心、边长 `int(1.5·max(w,h))` 的正方形，裁到图像范围（不补边）；
//! 2. ROI 双线性（align_corners=True）缩放到 768，`(x/255 − 0.5)·2`，分割网 → sigmoid → 最近邻放回 ROI 尺寸；
//! 3. `keep`：p ≥ 0.5 → 0（洞）；0.35 ≤ p < 0.5 → 1 − p（软洞）；否则 1；
//! 4. ROI 补零到 512 的倍数、四周再补 32 px，按 576 窗口（步长 512）送修复网：
//!    `comp = img·keep + (1 − keep)·net(img·keep, keep)`；有效区无洞的窗口直接跳过（结果按位相同）；
//! 5. 只把 `keep < 1` 的像素写回（其余像素严格不变）。
//!
//! 模型：`models/abpn_blemish_detect.onnx`（`image[1,3,H,W]` → `logits[1,1,H,W]`）、
//! `models/abpn_blemish_inpaint.onnx`（`image[1,3,H,W]`（已乘 mask）、`mask[1,1,H,W]` → `inpainted[1,3,H,W]`，tanh），
//! 由 `tools/abpn/export_onnx.py` 从 ModelScope 权重导出（H、W 动态，须为 64 的倍数）。

use crate::buffer::{GrayF32, ImgF32};
use crate::face::semantic::FaceBox;
use std::path::Path;
use std::time::Instant;

pub const DET_SIZE: usize = 768;
pub const PATCH: usize = 512;
pub const PAD: usize = 32;
pub const WIN: usize = PATCH + 2 * PAD;
pub const ROI_SCALE: f32 = 1.5;
pub const T_LOW: f32 = 0.35;
pub const T_HIGH: f32 = 0.5;

pub struct AiBlemish {
    detect: ort::session::Session,
    inpaint: ort::session::Session,
    det_in: String,
    inp_img: String,
    inp_mask: String,
}

/// 一次 ROI 清理的统计。
#[derive(Clone, Copy, Debug, Default)]
pub struct AiCleanStats {
    pub roi: (usize, usize, usize, usize),
    /// keep < 1 的像素数
    pub holes: usize,
    pub windows_run: usize,
    pub windows_total: usize,
    pub ms_detect: f32,
    pub ms_inpaint: f32,
}

/// 稀疏补丁：只记录被修改的像素（清理后的值，强度 1）。
#[derive(Clone, Debug, Default)]
pub struct AiPatch {
    pub pixels: Vec<(u32, u32, [f32; 3])>,
    pub stats: AiCleanStats,
}

/// 一张图所有人脸的补丁。
#[derive(Clone, Debug, Default)]
pub struct AiPatches {
    pub patches: Vec<AiPatch>,
}

impl AiPatches {
    pub fn is_empty(&self) -> bool {
        self.patches.iter().all(|p| p.pixels.is_empty())
    }
    pub fn holes(&self) -> usize {
        self.patches.iter().map(|p| p.stats.holes).sum()
    }
    /// 以强度 `strength`（0..1）把补丁混合到 `img` 上。
    pub fn apply_inplace(&self, img: &mut ImgF32, strength: f32) {
        let s = strength.clamp(0.0, 1.0);
        if s <= 0.0 {
            return;
        }
        for p in &self.patches {
            for &(x, y, v) in &p.pixels {
                let (x, y) = (x as usize, y as usize);
                if x >= img.w || y >= img.h {
                    continue;
                }
                let i = y * img.w + x;
                let o = img.data[i];
                img.data[i] = [
                    o[0] + (v[0] - o[0]) * s,
                    o[1] + (v[1] - o[1]) * s,
                    o[2] + (v[2] - o[2]) * s,
                ];
            }
        }
    }
    pub fn apply_to(&self, img: &ImgF32, strength: f32) -> ImgF32 {
        let mut out = img.clone();
        self.apply_inplace(&mut out, strength);
        out
    }
    /// 被修改像素的遮罩（调试用）。
    pub fn hole_mask(&self, w: usize, h: usize) -> GrayF32 {
        let mut m = GrayF32::new(w, h);
        for p in &self.patches {
            for &(x, y, _) in &p.pixels {
                let (x, y) = (x as usize, y as usize);
                if x < w && y < h {
                    m.data[y * w + x] = 1.0;
                }
            }
        }
        m
    }
}

/// ModelScope 的 1.5× 正方形 ROI（`get_crop_bbox` + `get_roi_without_padding`）：
/// 返回 (x0, y0, x1, y1)，右下开区间；太小返回 None。
pub fn face_roi(bx: &FaceBox, w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let side = (ROI_SCALE * bx.width().abs().max(bx.height().abs())) as i64;
    let cx = (bx.x1 + bx.x2) * 0.5;
    let cy = (bx.y1 + bx.y2) * 0.5;
    let half = side as f32 / 2.0;
    let x0 = ((cx - half) as i64).max(0);
    let y0 = ((cy - half) as i64).max(0);
    let x1 = ((cx + half) as i64).min(w as i64);
    let y1 = ((cy + half) as i64).min(h as i64);
    if x1 < x0 + 16 || y1 < y0 + 16 {
        return None;
    }
    Some((x0 as usize, y0 as usize, x1 as usize, y1 as usize))
}

#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

impl AiBlemish {
    pub fn new(detect_model: &Path, inpaint_model: &Path, threads: usize) -> anyhow::Result<Self> {
        let detect = super::super::face::ort_util::make_session(detect_model, threads)?;
        let inpaint = super::super::face::ort_util::make_session(inpaint_model, threads)?;
        let det_in = detect
            .inputs()
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "image".into());
        let names: Vec<String> = inpaint
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        anyhow::ensure!(
            names.len() == 2,
            "inpaint model must have 2 inputs (image, mask), got {:?}",
            names
        );
        // 按名字识别（导出时命名为 image / mask；否则按顺序）
        let (inp_img, inp_mask) = if names[0].contains("mask") {
            (names[1].clone(), names[0].clone())
        } else {
            (names[0].clone(), names[1].clone())
        };
        Ok(Self {
            detect,
            inpaint,
            det_in,
            inp_img,
            inp_mask,
        })
    }

    /// 对 `img`（0..1 RGB）的一个 ROI 做检测 + 修复，返回稀疏补丁（不修改 `img`）。
    pub fn clean_roi(
        &mut self,
        img: &ImgF32,
        roi: (usize, usize, usize, usize),
    ) -> anyhow::Result<AiPatch> {
        let _ftz = crate::face::ort_util::DenormalGuard::new();
        let (x0, y0, x1, y1) = roi;
        let (rw, rh) = (x1 - x0, y1 - y0);
        let mut stats = AiCleanStats {
            roi,
            ..Default::default()
        };
        if rw < 16 || rh < 16 {
            return Ok(AiPatch {
                pixels: Vec::new(),
                stats,
            });
        }
        // 1. 归一化 ROI（[-1, 1]）
        let norm: Vec<[f32; 3]> = (0..rw * rh)
            .map(|i| {
                let p = img.data[(y0 + i / rw) * img.w + x0 + i % rw];
                [(p[0] - 0.5) * 2.0, (p[1] - 0.5) * 2.0, (p[2] - 0.5) * 2.0]
            })
            .collect();
        // 2. 768 双线性（align_corners=True）
        let t = Instant::now();
        let mut det_in = ndarray::Array4::<f32>::zeros((1, 3, DET_SIZE, DET_SIZE));
        let coord = |o: usize, n: usize| -> (usize, usize, f32) {
            if n <= 1 {
                return (0, 0, 0.0);
            }
            let s = o as f32 * (n - 1) as f32 / (DET_SIZE - 1) as f32;
            let i0 = (s.floor() as usize).min(n - 1);
            let i1 = (i0 + 1).min(n - 1);
            (i0, i1, s - i0 as f32)
        };
        for oy in 0..DET_SIZE {
            let (ya, yb, ty) = coord(oy, rh);
            for ox in 0..DET_SIZE {
                let (xa, xb, tx) = coord(ox, rw);
                let p00 = norm[ya * rw + xa];
                let p01 = norm[ya * rw + xb];
                let p10 = norm[yb * rw + xa];
                let p11 = norm[yb * rw + xb];
                for c in 0..3 {
                    let top = p00[c] + (p01[c] - p00[c]) * tx;
                    let bot = p10[c] + (p11[c] - p10[c]) * tx;
                    det_in[[0, c, oy, ox]] = top + (bot - top) * ty;
                }
            }
        }
        let outputs = self
            .detect
            .run(ort::inputs![self.det_in.as_str() => ort::value::Tensor::from_array(det_in)?])?;
        let (shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.len() == 4 && shape[2] as usize == DET_SIZE && shape[3] as usize == DET_SIZE,
            "unexpected blemish-detect output shape {:?}",
            shape
        );
        // 3. keep（最近邻放回 ROI 尺寸）
        let keep: Vec<f32> = (0..rw * rh)
            .map(|i| {
                let (x, y) = (i % rw, i / rw);
                let sy = ((y as f32 * DET_SIZE as f32 / rh as f32) as usize).min(DET_SIZE - 1);
                let sx = ((x as f32 * DET_SIZE as f32 / rw as f32) as usize).min(DET_SIZE - 1);
                let p = sigmoid(logits[sy * DET_SIZE + sx]);
                if p >= T_HIGH {
                    0.0
                } else if p >= T_LOW {
                    1.0 - p
                } else {
                    1.0
                }
            })
            .collect();
        stats.ms_detect = t.elapsed().as_secs_f32() * 1e3;
        stats.holes = keep.iter().filter(|k| **k < 1.0).count();
        if stats.holes == 0 {
            return Ok(AiPatch {
                pixels: Vec::new(),
                stats,
            });
        }
        // 4. 窗口修复
        let t = Instant::now();
        let hs = rh.div_ceil(PATCH) * PATCH;
        let ws = rw.div_ceil(PATCH) * PATCH;
        let (nh, nw) = (hs / PATCH, ws / PATCH);
        stats.windows_total = nh * nw;
        let mut result = norm.clone();
        for i in 0..nh {
            for j in 0..nw {
                let (vy0, vy1) = (i * PATCH, ((i + 1) * PATCH).min(rh));
                let (vx0, vx1) = (j * PATCH, ((j + 1) * PATCH).min(rw));
                if vy1 <= vy0 || vx1 <= vx0 {
                    continue;
                }
                let has_hole =
                    (vy0..vy1).any(|y| keep[y * rw + vx0..y * rw + vx1].iter().any(|k| *k < 1.0));
                if !has_hole {
                    continue;
                }
                let mut im = ndarray::Array4::<f32>::zeros((1, 3, WIN, WIN));
                let mut mk = ndarray::Array4::<f32>::zeros((1, 1, WIN, WIN));
                for wy in 0..WIN {
                    let cy = i * PATCH + wy;
                    if cy < PAD || cy - PAD >= rh {
                        continue;
                    }
                    let ry = cy - PAD;
                    for wx in 0..WIN {
                        let cx = j * PATCH + wx;
                        if cx < PAD || cx - PAD >= rw {
                            continue;
                        }
                        let rx = cx - PAD;
                        let k = keep[ry * rw + rx];
                        mk[[0, 0, wy, wx]] = k;
                        let p = norm[ry * rw + rx];
                        im[[0, 0, wy, wx]] = p[0] * k;
                        im[[0, 1, wy, wx]] = p[1] * k;
                        im[[0, 2, wy, wx]] = p[2] * k;
                    }
                }
                let outputs = self.inpaint.run(ort::inputs![
                    self.inp_img.as_str() => ort::value::Tensor::from_array(im)?,
                    self.inp_mask.as_str() => ort::value::Tensor::from_array(mk)?
                ])?;
                let (shape, out) = outputs[0].try_extract_tensor::<f32>()?;
                anyhow::ensure!(
                    shape.len() == 4 && shape[2] as usize == WIN && shape[3] as usize == WIN,
                    "unexpected blemish-inpaint output shape {:?}",
                    shape
                );
                let plane = WIN * WIN;
                for wy in PAD..PAD + PATCH {
                    let ry = i * PATCH + wy - PAD;
                    if ry >= rh {
                        break;
                    }
                    for wx in PAD..PAD + PATCH {
                        let rx = j * PATCH + wx - PAD;
                        if rx >= rw {
                            break;
                        }
                        let k = keep[ry * rw + rx];
                        if k < 1.0 {
                            let p = norm[ry * rw + rx];
                            let r = &mut result[ry * rw + rx];
                            for c in 0..3 {
                                r[c] = p[c] * k + (1.0 - k) * out[c * plane + wy * WIN + wx];
                            }
                        }
                    }
                }
                stats.windows_run += 1;
            }
        }
        stats.ms_inpaint = t.elapsed().as_secs_f32() * 1e3;
        // 5. 稀疏补丁
        let mut pixels = Vec::with_capacity(stats.holes);
        for i in 0..rw * rh {
            if keep[i] < 1.0 {
                let r = result[i];
                pixels.push((
                    (x0 + i % rw) as u32,
                    (y0 + i / rw) as u32,
                    [
                        (r[0] * 0.5 + 0.5).clamp(0.0, 1.0),
                        (r[1] * 0.5 + 0.5).clamp(0.0, 1.0),
                        (r[2] * 0.5 + 0.5).clamp(0.0, 1.0),
                    ],
                ));
            }
        }
        Ok(AiPatch { pixels, stats })
    }

    /// 对所有人脸 ROI 做清理。
    pub fn clean_faces(&mut self, img: &ImgF32, boxes: &[FaceBox]) -> anyhow::Result<AiPatches> {
        let mut patches = Vec::with_capacity(boxes.len());
        for b in boxes {
            if let Some(roi) = face_roi(b, img.w, img.h) {
                patches.push(self.clean_roi(img, roi)?);
            }
        }
        Ok(AiPatches { patches })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roi_matches_modelscope_crop_rule() {
        let b = FaceBox {
            x1: 100.0,
            y1: 200.0,
            x2: 300.0,
            y2: 500.0,
            score: 1.0,
        };
        // side = int(1.5 * 300) = 450, center (200, 350)
        let r = face_roi(&b, 1000, 1000).unwrap();
        assert_eq!(r, (0, 125, 425, 575));
        let r = face_roi(&b, 10000, 10000).unwrap();
        assert_eq!(r, (0, 125, 425, 575));
        assert!(face_roi(&b, 5, 5).is_none());
    }

    #[test]
    fn patches_apply_with_strength() {
        let mut img = ImgF32::filled(4, 4, [0.2, 0.2, 0.2]);
        let p = AiPatches {
            patches: vec![AiPatch {
                pixels: vec![(1, 1, [1.0, 0.0, 0.2])],
                stats: AiCleanStats::default(),
            }],
        };
        p.apply_inplace(&mut img, 0.5);
        let v = img.get(1, 1);
        assert!(
            (v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.1).abs() < 1e-6 && (v[2] - 0.2).abs() < 1e-6
        );
        assert_eq!(img.get(0, 0), [0.2, 0.2, 0.2]);
        assert!(!p.is_empty());
    }
}
