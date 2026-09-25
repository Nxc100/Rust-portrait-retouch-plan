//! 语义皮肤分割（身体皮肤遮罩的门控）。
//!
//! 肤色规则 + 人像 alpha 会把米色缎面婚纱、粉色团扇、金饰等"肤色"物体当成皮肤；
//! 用一个语义分割模型给出"皮肤概率图"，与颜色规则取交集后即可排除衣物与饰品。
//!
//! 支持两种模型（按输入张量的秩自动识别，见 `Flavor`）：
//! - `models/skin_seg.onnx`（推荐，`scripts/fetch_models.py --only skinseg`）：ModelScope
//!   `iic/cv_unet_skin-retouching` 的皮肤分割 TF 图 `tf_graph.pb`（Apache-2.0，`tools/abpn/convert_skinseg.py`
//!   用 tf2onnx 转换）——输入 `[H, W, 3]` float RGB 0..255（长边 800），输出 `[H, W]` uint8 0..255 皮肤掩码；
//! - `models/skin_seg_lite.onnx`（备选，`--only skinseg-lite`）：Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP
//!   （MIT）的 DeepLabV3+——输入 `[1,3,512,512]` float、ImageNet 归一化，输出 `[1,3,512,512]` 三类 sigmoid
//!   （0 皮肤、1 衣物、2 头发）。
//!
//! 输出统一为与原图同尺寸的 0..1 概率图。

use crate::buffer::GrayF32;
use std::path::Path;

/// 模型输入约定。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavor {
    /// `[1,3,H,W]` float，ImageNet 均值 / 方差归一化
    NchwImagenet,
    /// `[H,W,3]` float/uint8，RGB 0..255，长边 `long_side`
    HwcRaw,
}

pub struct SkinSeg {
    session: ort::session::Session,
    input_name: String,
    input_is_u8: bool,
    /// 固定输入尺寸（宽, 高）；None 表示动态
    fixed: Option<(usize, usize)>,
    long_side: usize,
    flavor: Flavor,
    skin_channel: usize,
}

impl SkinSeg {
    pub fn new(model: &Path, threads: usize) -> anyhow::Result<Self> {
        let session = super::ort_util::make_session(model, threads)?;
        let input = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow::anyhow!("skin-seg model has no input"))?;
        let input_name = input.name().to_string();
        let dtype = input.dtype();
        let input_is_u8 = matches!(
            dtype.tensor_type(),
            Some(ort::value::TensorElementType::Uint8)
        );
        let dims: Vec<i64> = dtype
            .tensor_shape()
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();
        let (flavor, fixed) = match dims.len() {
            3 => (Flavor::HwcRaw, None),
            4 => (
                Flavor::NchwImagenet,
                if dims[2] > 0 && dims[3] > 0 {
                    Some((dims[3] as usize, dims[2] as usize))
                } else {
                    None
                },
            ),
            n => anyhow::bail!("unsupported skin-seg input rank {n} (dims {dims:?})"),
        };
        Ok(Self {
            session,
            input_name,
            input_is_u8,
            fixed,
            long_side: if flavor == Flavor::HwcRaw { 800 } else { 512 },
            flavor,
            skin_channel: 0,
        })
    }

    pub fn flavor(&self) -> Flavor {
        self.flavor
    }

    fn input_size(&self, w: usize, h: usize) -> (usize, usize) {
        if let Some(s) = self.fixed {
            return s;
        }
        let s = self.long_side as f32 / w.max(h) as f32;
        match self.flavor {
            // 与 ModelScope `resize_on_long_side` 逐位一致：长边 = long_side，短边 = int(短边 · (long_side / 长边))（f64）
            Flavor::HwcRaw => {
                let l = self.long_side;
                if h > w {
                    let scale = l as f64 / h as f64;
                    (((w as f64 * scale) as usize).max(1), l)
                } else {
                    let scale = l as f64 / w as f64;
                    (l, ((h as f64 * scale) as usize).max(1))
                }
            }
            // 动态 NCHW：取 32 的倍数
            Flavor::NchwImagenet => (
                (((w as f32 * s) as usize / 32).max(1)) * 32,
                (((h as f32 * s) as usize / 32).max(1)) * 32,
            ),
        }
    }

    /// 返回与原图同尺寸的皮肤概率图（0..1）。
    pub fn skin_prob(&mut self, img: &image::RgbImage) -> anyhow::Result<GrayF32> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let (iw, ih) = self.input_size(w, h);
        let small = image::imageops::resize(
            img,
            iw as u32,
            ih as u32,
            image::imageops::FilterType::Triangle,
        );
        let outputs = match self.flavor {
            Flavor::NchwImagenet => {
                let mean = [0.485f32, 0.456, 0.406];
                let std = [0.229f32, 0.224, 0.225];
                let mut input = ndarray::Array4::<f32>::zeros((1, 3, ih, iw));
                for (x, y, p) in small.enumerate_pixels() {
                    for c in 0..3 {
                        input[[0, c, y as usize, x as usize]] =
                            (p[c] as f32 / 255.0 - mean[c]) / std[c];
                    }
                }
                let tensor = ort::value::Tensor::from_array(input)?;
                self.session
                    .run(ort::inputs![self.input_name.as_str() => tensor])?
            }
            Flavor::HwcRaw => {
                if self.input_is_u8 {
                    let mut input = ndarray::Array3::<u8>::zeros((ih, iw, 3));
                    for (x, y, p) in small.enumerate_pixels() {
                        for c in 0..3 {
                            input[[y as usize, x as usize, c]] = p[c];
                        }
                    }
                    let tensor = ort::value::Tensor::from_array(input)?;
                    self.session
                        .run(ort::inputs![self.input_name.as_str() => tensor])?
                } else {
                    let mut input = ndarray::Array3::<f32>::zeros((ih, iw, 3));
                    for (x, y, p) in small.enumerate_pixels() {
                        for c in 0..3 {
                            input[[y as usize, x as usize, c]] = p[c] as f32;
                        }
                    }
                    let tensor = ort::value::Tensor::from_array(input)?;
                    self.session
                        .run(ort::inputs![self.input_name.as_str() => tensor])?
                }
            }
        };
        // 输出：f32 或 u8；秩 2 `[H,W]`、秩 3 `[H,W,C]`（取末通道）或秩 4 `[1,C,H,W]`（取 skin_channel）
        let (shape, data): (Vec<i64>, Vec<f32>) = match outputs[0].try_extract_tensor::<f32>() {
            Ok((s, d)) => (s.iter().copied().collect(), d.to_vec()),
            Err(_) => {
                let (s, d) = outputs[0].try_extract_tensor::<u8>()?;
                (
                    s.iter().copied().collect(),
                    d.iter().map(|v| *v as f32).collect(),
                )
            }
        };
        let (ow, oh, plane): (usize, usize, Vec<f32>) = match shape.len() {
            2 => (shape[1] as usize, shape[0] as usize, data),
            3 => {
                let (oh, ow, nc) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
                let ch = nc - 1;
                (ow, oh, (0..oh * ow).map(|i| data[i * nc + ch]).collect())
            }
            4 => {
                let (nc, oh, ow) = (shape[1] as usize, shape[2] as usize, shape[3] as usize);
                let ch = self.skin_channel.min(nc.saturating_sub(1));
                let n = oh * ow;
                (ow, oh, data[ch * n..(ch + 1) * n].to_vec())
            }
            n => anyhow::bail!("unexpected skin-seg output rank {n}"),
        };
        anyhow::ensure!(ow > 0 && oh > 0, "empty skin-seg output");
        let maxv = plane.iter().copied().fold(0.0f32, f32::max);
        let minv = plane.iter().copied().fold(f32::MAX, f32::min);
        let mapped: Vec<f32> = if maxv > 1.5 {
            plane.iter().map(|v| (v / 255.0).clamp(0.0, 1.0)).collect()
        } else if minv < -0.01 {
            // logits → sigmoid
            plane.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
        } else {
            plane.iter().map(|v| v.clamp(0.0, 1.0)).collect()
        };
        Ok(GrayF32::from_vec(ow, oh, mapped).resize(w, h))
    }
}
