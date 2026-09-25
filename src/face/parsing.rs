//! BiSeNet 人脸解析（yakhyo/face-parsing，MIT；CelebAMask-HQ 19 类）。
//!
//! 输入 512×512 RGB，ImageNet 归一化 `(x/255 − mean)/std`；输出 `[1,19,512,512]` logits，argmax 得类别。
//! 裁剪：以人脸框中心为中心、边长 `max(w,h) × 1.8` 的正方形（包含发际线与脖颈）。

use super::crop::warp_affine_rgb8;
use super::semantic::FaceBox;
use crate::buffer::GrayF32;
use crate::geom::{Affine, P};
use std::path::Path;

pub const INPUT_SIZE: usize = 512;
pub const CROP_SCALE: f32 = 1.8;
pub const NUM_CLASSES: usize = 19;

pub const CLS_BACKGROUND: u8 = 0;
pub const CLS_SKIN: u8 = 1;
pub const CLS_L_BROW: u8 = 2;
pub const CLS_R_BROW: u8 = 3;
pub const CLS_L_EYE: u8 = 4;
pub const CLS_R_EYE: u8 = 5;
pub const CLS_EYE_G: u8 = 6;
pub const CLS_L_EAR: u8 = 7;
pub const CLS_R_EAR: u8 = 8;
pub const CLS_EAR_R: u8 = 9;
pub const CLS_NOSE: u8 = 10;
pub const CLS_MOUTH: u8 = 11;
pub const CLS_U_LIP: u8 = 12;
pub const CLS_L_LIP: u8 = 13;
pub const CLS_NECK: u8 = 14;
pub const CLS_NECK_L: u8 = 15;
pub const CLS_CLOTH: u8 = 16;
pub const CLS_HAIR: u8 = 17;
pub const CLS_HAT: u8 = 18;

/// 脸部皮肤类（含鼻、耳）。
pub const FACE_SKIN_CLASSES: [u8; 4] = [CLS_SKIN, CLS_NOSE, CLS_L_EAR, CLS_R_EAR];
/// 身体皮肤类（脖颈）。
pub const NECK_CLASSES: [u8; 1] = [CLS_NECK];
/// 不做皮肤处理的类（头发、衣物、五官、饰品）。
pub const EXCLUDE_CLASSES: [u8; 11] = [
    CLS_L_BROW, CLS_R_BROW, CLS_L_EYE, CLS_R_EYE, CLS_EYE_G, CLS_EAR_R, CLS_MOUTH, CLS_U_LIP,
    CLS_L_LIP, CLS_CLOTH, CLS_HAIR,
];

/// 单张脸的解析结果（裁剪分辨率）。
#[derive(Clone, Debug)]
pub struct ParseMap {
    pub size: usize,
    /// 类别图，行优先
    pub classes: Vec<u8>,
    /// 原图索引坐标 → 裁剪索引坐标
    pub affine: Affine,
    /// 该裁剪在原图中的覆盖范围（像素，含）
    pub bbox: (i32, i32, i32, i32),
}

impl ParseMap {
    /// 原图像素中心制坐标处的类别（裁剪外为背景）。
    #[inline]
    pub fn class_at(&self, x: f32, y: f32) -> u8 {
        let q = self.affine.apply(P::new(x - 0.5, y - 0.5));
        let (u, v) = (q.x.round() as i64, q.y.round() as i64);
        if u < 0 || v < 0 || u >= self.size as i64 || v >= self.size as i64 {
            CLS_BACKGROUND
        } else {
            self.classes[v as usize * self.size + u as usize]
        }
    }

    /// 在原图 `w×h` 上生成给定类别集合的二值遮罩（最近邻），只在裁剪覆盖区域内非零。
    pub fn mask_of(&self, w: usize, h: usize, classes: &[u8]) -> GrayF32 {
        let mut m = GrayF32::new(w, h);
        let (x0, y0, x1, y1) = self.bbox;
        let (x0, y0) = (x0.max(0) as usize, y0.max(0) as usize);
        let (x1, y1) = (
            (x1.max(0) as usize).min(w.saturating_sub(1)),
            (y1.max(0) as usize).min(h.saturating_sub(1)),
        );
        if x0 > x1 || y0 > y1 {
            return m;
        }
        use rayon::prelude::*;
        m.data[y0 * w..(y1 + 1) * w]
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(dy, row)| {
                let y = (y0 + dy) as f32 + 0.5;
                for (x, v) in row.iter_mut().enumerate().take(x1 + 1).skip(x0) {
                    let c = self.class_at(x as f32 + 0.5, y);
                    if classes.contains(&c) {
                        *v = 1.0;
                    }
                }
            });
        m
    }

    /// 坐标整体缩放 `s` 后的解析图（预览缩略图用）：仿射左乘 1/s。
    pub fn scaled(&self, s: f32) -> ParseMap {
        let inv = 1.0 / s.max(1e-6);
        let a = &self.affine;
        let affine = Affine {
            a: a.a * inv,
            b: a.b * inv,
            c: a.c,
            d: a.d * inv,
            e: a.e * inv,
            f: a.f,
        };
        let (x0, y0, x1, y1) = self.bbox;
        ParseMap {
            size: self.size,
            classes: self.classes.clone(),
            affine,
            bbox: (
                (x0 as f32 * s).floor() as i32,
                (y0 as f32 * s).floor() as i32,
                (x1 as f32 * s).ceil() as i32,
                (y1 as f32 * s).ceil() as i32,
            ),
        }
    }

    /// 类别直方图（调试用）。
    pub fn histogram(&self) -> [usize; NUM_CLASSES] {
        let mut h = [0usize; NUM_CLASSES];
        for c in &self.classes {
            if (*c as usize) < NUM_CLASSES {
                h[*c as usize] += 1;
            }
        }
        h
    }
}

pub struct FaceParser {
    session: ort::session::Session,
    input_name: String,
}

impl FaceParser {
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

    pub fn parse(&mut self, img: &image::RgbImage, face_box: &FaceBox) -> anyhow::Result<ParseMap> {
        let side = face_box.width().max(face_box.height()).max(8.0) * CROP_SCALE;
        let c = face_box.center();
        let scale = INPUT_SIZE as f32 / side;
        let m = Affine::crop(P::new(c.x - 0.5, c.y - 0.5), scale, 0.0, INPUT_SIZE as f32);
        let hwc = warp_affine_rgb8(img, &m, INPUT_SIZE);
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let mut input = ndarray::Array4::<f32>::zeros((1, 3, INPUT_SIZE, INPUT_SIZE));
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let p = hwc[y * INPUT_SIZE + x];
                for ch in 0..3 {
                    input[[0, ch, y, x]] = (p[ch] / 255.0 - mean[ch]) / std[ch];
                }
            }
        }
        let tensor = ort::value::Tensor::from_array(input)?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])?;
        let (shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.len() == 4,
            "unexpected parsing output rank {}",
            shape.len()
        );
        let (nc, oh, ow) = (shape[1] as usize, shape[2] as usize, shape[3] as usize);
        anyhow::ensure!(
            oh == INPUT_SIZE && ow == INPUT_SIZE,
            "unexpected parsing output size {ow}x{oh}"
        );
        let plane = oh * ow;
        let mut classes = vec![0u8; plane];
        for i in 0..plane {
            let mut best = 0usize;
            let mut bv = f32::MIN;
            for c in 0..nc {
                let v = logits[c * plane + i];
                if v > bv {
                    bv = v;
                    best = c;
                }
            }
            classes[i] = best as u8;
        }
        let half = side * 0.5;
        let bbox = (
            (c.x - half).floor() as i32,
            (c.y - half).floor() as i32,
            (c.x + half).ceil() as i32,
            (c.y + half).ceil() as i32,
        );
        Ok(ParseMap {
            size: INPUT_SIZE,
            classes,
            affine: m,
            bbox,
        })
    }
}
