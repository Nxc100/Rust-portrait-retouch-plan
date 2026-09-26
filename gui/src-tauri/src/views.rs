//! 可查看的图像：原图 / 结果，以及用于排查问题的调试视图（关键点、皮肤遮罩、人像抠图、皮肤概率、AI 修复区）。

use crate::session::Working;
use image::imageops::FilterType;
use image::RgbImage;
use portrait_retouch::{debug, Engine, GrayF32, PhotoMetadata};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// 视图种类（URL 路径与前端按钮使用同样的名字）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewKind {
    Original,
    Result,
    /// 人脸框与语义关键点
    Landmarks,
    /// 皮肤遮罩叠加在原图上：脸部（粉）与身体（橙）
    Skin,
    /// 人像抠图 alpha
    Matte,
    /// 语义皮肤分割概率
    SkinProb,
    /// AI 瑕疵模型要修复的区域
    AiHoles,
}

impl ViewKind {
    pub const ALL: [ViewKind; 7] = [
        ViewKind::Original,
        ViewKind::Result,
        ViewKind::Landmarks,
        ViewKind::Skin,
        ViewKind::Matte,
        ViewKind::SkinProb,
        ViewKind::AiHoles,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ViewKind::Original => "original",
            ViewKind::Result => "result",
            ViewKind::Landmarks => "landmarks",
            ViewKind::Skin => "skin",
            ViewKind::Matte => "matte",
            ViewKind::SkinProb => "skin_prob",
            ViewKind::AiHoles => "ai_holes",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.as_str() == s)
    }

    /// 生成该视图是否需要模型（关键点视图只用已检测的人脸）。
    pub fn needs_engine(self) -> bool {
        matches!(
            self,
            ViewKind::Skin | ViewKind::Matte | ViewKind::SkinProb | ViewKind::AiHoles
        )
    }
}

/// 皮肤遮罩视图的颜色：脸部 / 身体。
const FACE_SKIN_COLOR: [u8; 3] = [255, 60, 110];
const BODY_SKIN_COLOR: [u8; 3] = [255, 150, 30];

/// 渲染调试视图（原图 / 结果不走这里，由会话直接提供）。皮肤遮罩总是连同身体一起计算，
/// 与"处理身体皮肤"打开时奶油肌使用的遮罩相同。
pub fn render(
    kind: ViewKind,
    engine: Option<&Engine>,
    working: &Working,
) -> anyhow::Result<RgbImage> {
    let img = working.image.as_ref();
    let need_engine =
        || engine.ok_or_else(|| anyhow::anyhow!("没有加载模型，无法生成该视图（见\"引擎设置\"）"));
    match kind {
        ViewKind::Original | ViewKind::Result => {
            anyhow::bail!("{} 不是调试视图", kind.as_str())
        }
        ViewKind::Landmarks => {
            let mut vis = img.clone();
            for f in working.faces.iter() {
                debug::draw_face_debug(&mut vis, f, false);
            }
            Ok(vis)
        }
        ViewKind::Skin => {
            let masks = need_engine()?.skin_masks(img, &working.faces, true)?;
            let mut vis = match &masks.body {
                Some(body) => overlay(img, body, BODY_SKIN_COLOR, 0.55),
                None => img.clone(),
            };
            for face in &masks.faces {
                vis = overlay(&vis, face, FACE_SKIN_COLOR, 0.55);
            }
            Ok(vis)
        }
        ViewKind::Matte => {
            let m = need_engine()?.person_matte(img)?.ok_or_else(|| {
                anyhow::anyhow!("未加载人像抠图模型（models/modnet_photographic.onnx）")
            })?;
            Ok(gray_to_rgb(&m))
        }
        ViewKind::SkinProb => {
            let m = need_engine()?.skin_prob(img)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "未加载语义皮肤分割模型（models/skin_seg.onnx 或 skin_seg_lite.onnx）"
                )
            })?;
            Ok(gray_to_rgb(&m))
        }
        ViewKind::AiHoles => {
            let e = need_engine()?;
            anyhow::ensure!(
                !working.faces.is_empty(),
                "没有检测到人脸，AI 瑕疵模型只处理人脸区域"
            );
            let patches = e.ai_patches(&e.precomp(img), &working.faces)?.ok_or_else(|| {
                anyhow::anyhow!("未加载 AI 瑕疵模型（models/abpn_blemish_detect.onnx / abpn_blemish_inpaint.onnx）")
            })?;
            let (w, h) = img.dimensions();
            let holes = patches.hole_mask(w as usize, h as usize);
            Ok(overlay(img, &holes, [0, 200, 255], 0.8))
        }
    }
}

/// 把遮罩（0..1）以半透明颜色叠加到图像上：`out = img·(1 − a·m) + color·a·m`。
pub fn overlay(img: &RgbImage, mask: &GrayF32, color: [u8; 3], alpha: f32) -> RgbImage {
    let mut out = img.clone();
    for (i, p) in out.pixels_mut().enumerate() {
        let m = (mask.data.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0)) * alpha;
        for c in 0..3 {
            p[c] = (p[c] as f32 * (1.0 - m) + color[c] as f32 * m).round() as u8;
        }
    }
    out
}

/// 灰度遮罩（0..1）→ RGB。
pub fn gray_to_rgb(m: &GrayF32) -> RgbImage {
    let luma = m.to_luma8();
    RgbImage::from_fn(luma.width(), luma.height(), |x, y| {
        let v = luma.get_pixel(x, y)[0];
        image::Rgb([v, v, v])
    })
}

/// 显示用：长边超过 `max`（> 0）时缩小。
pub fn fit_long_side(img: &RgbImage, max: u32) -> Cow<'_, RgbImage> {
    let (w, h) = img.dimensions();
    let longest = w.max(h);
    if max == 0 || longest <= max {
        return Cow::Borrowed(img);
    }
    let s = max as f32 / longest as f32;
    Cow::Owned(image::imageops::resize(
        img,
        ((w as f32 * s).round() as u32).max(1),
        ((h as f32 * s).round() as u32).max(1),
        FilterType::Triangle,
    ))
}

/// 显示用的 JPEG 编码（质量 90，不带元数据）。
pub fn encode_for_display(img: &RgbImage) -> anyhow::Result<Vec<u8>> {
    portrait_retouch::photo::encode_jpeg(img, 90, &PhotoMetadata::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn view_names_roundtrip() {
        for k in ViewKind::ALL {
            assert_eq!(ViewKind::parse(k.as_str()), Some(k));
            let json = serde_json::to_string(&k).unwrap();
            assert_eq!(json, format!("\"{}\"", k.as_str()));
        }
        assert_eq!(ViewKind::parse("nope"), None);
    }

    #[test]
    fn overlay_tints_only_masked_pixels() {
        let img = RgbImage::from_pixel(2, 1, image::Rgb([100, 100, 100]));
        let mask = GrayF32::from_vec(2, 1, vec![0.0, 1.0]);
        let out = overlay(&img, &mask, [255, 0, 0], 0.5);
        assert_eq!(out.get_pixel(0, 0).0, [100, 100, 100]);
        assert_eq!(out.get_pixel(1, 0).0, [178, 50, 50]);
    }

    #[test]
    fn fit_and_encode_for_display() {
        let img = RgbImage::from_pixel(3000, 1000, image::Rgb([10, 20, 30]));
        let small = fit_long_side(&img, 1500);
        assert_eq!(small.dimensions(), (1500, 500));
        assert!(matches!(fit_long_side(&img, 0), Cow::Borrowed(_)));
        let jpeg = encode_for_display(&small).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn debug_views_without_engine_report_a_clear_error() {
        let working = Working {
            long_side: 0,
            scale: 1.0,
            image: Arc::new(RgbImage::new(8, 8)),
            faces: Arc::new(vec![]),
        };
        // 关键点视图不需要模型
        assert!(!ViewKind::Landmarks.needs_engine());
        assert!(render(ViewKind::Landmarks, None, &working).is_ok());
        for kind in ViewKind::ALL.into_iter().filter(|k| k.needs_engine()) {
            let err = render(kind, None, &working).unwrap_err();
            assert!(err.to_string().contains("没有加载模型"), "{kind:?}");
        }
        assert!(render(ViewKind::Result, None, &working).is_err());
    }
}
