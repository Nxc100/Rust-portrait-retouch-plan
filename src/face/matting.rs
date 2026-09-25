//! MODNet 人像抠图（ZHKKKe/MODNet，Apache-2.0；ONNX 由 yakhyo/modnet 提供）。
//!
//! 输入：短边缩放到 512（宽高取 32 的倍数），RGB，`(x/255 − 0.5)/0.5`；输出 `[1,1,h,w]` alpha（0..1）。
//! 用途：身体皮肤遮罩的"人物"门控，避免颜色规则把礁石、木头等判为皮肤。

use crate::buffer::GrayF32;
use std::path::Path;

pub const REF_SIZE: u32 = 512;

pub struct PersonMatting {
    session: ort::session::Session,
    input_name: String,
}

impl PersonMatting {
    pub fn new(model: &Path, threads: usize) -> anyhow::Result<Self> {
        let session = super::ort_util::make_session(model, threads)?;
        let input_name = session
            .inputs()
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "input".into());
        Ok(Self {
            session,
            input_name,
        })
    }

    /// 返回与原图同尺寸的 alpha 遮罩。
    pub fn matte(&mut self, img: &image::RgbImage) -> anyhow::Result<GrayF32> {
        let (w, h) = (img.width(), img.height());
        let (mut nw, mut nh) = if w.max(h) < REF_SIZE || w.min(h) > REF_SIZE {
            if w >= h {
                (((w as f32 / h as f32) * REF_SIZE as f32) as u32, REF_SIZE)
            } else {
                (REF_SIZE, ((h as f32 / w as f32) * REF_SIZE as f32) as u32)
            }
        } else {
            (w, h)
        };
        nw -= nw % 32;
        nh -= nh % 32;
        let nw = nw.max(32);
        let nh = nh.max(32);
        let small = image::imageops::resize(img, nw, nh, image::imageops::FilterType::Triangle);
        let mut input = ndarray::Array4::<f32>::zeros((1, 3, nh as usize, nw as usize));
        for (x, y, p) in small.enumerate_pixels() {
            for c in 0..3 {
                input[[0, c, y as usize, x as usize]] = (p[c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        let tensor = ort::value::Tensor::from_array(input)?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        anyhow::ensure!(shape.len() == 4, "unexpected matte rank {}", shape.len());
        let (oh, ow) = (shape[2] as usize, shape[3] as usize);
        let mut m = GrayF32::from_vec(ow, oh, data[..ow * oh].to_vec());
        m.map_inplace(|v| v.clamp(0.0, 1.0));
        Ok(m.resize(w as usize, h as usize))
    }
}
