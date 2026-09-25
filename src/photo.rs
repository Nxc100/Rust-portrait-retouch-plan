//! 照片文件读写：解码（按 EXIF 方向转正）、元数据（ICC / EXIF / 内嵌冲印设置）与编码保存。
//!
//! - **方向**：相机直出的竖拍 JPEG 常以横向像素 + EXIF Orientation 存储，人脸检测只认正立的脸。
//!   读入时按方向标记把像素转正，并把要写回的 EXIF 中的方向置为 1（否则查看器会再转一次）。
//! - **JPEG 编码**用 jpeg-encoder（量化正确舍入，质量 ≥ 90 时 4:4:4）：image 0.25 自带的编码器
//!   在量化前把 DCT 系数整数截断（`dct / 8`），低幅度的细纹理被系统性削弱，q100 时最细一档也只剩约 93%。
//!   默认质量 98 与像素蛋糕导出的量化表一致（亮度表最大 5）。霍夫曼表用标准表（Annex K）而不是优化表：
//!   jpeg-encoder 0.7 的优化表在退化的符号分布上（如 16×16 纯色）会被 zune-jpeg（image 库的解码器）
//!   解成错误颜色，libjpeg 却正常——标准表处处兼容，代价是文件大几个百分点。
//! - **元数据**：ICC 配置文件与 EXIF 原样写回；XMP 不写回，其中的 Camera Raw 冲印设置（`crs:AlreadyApplied`）
//!   可能被下游软件再应用一次。
//! - **原子写入**：先写同目录下的临时文件再改名，进程中断时不会留下半截输出，
//!   批处理"已存在即跳过"因此可靠。

use crate::color::develop::DevelopSettings;
use anyhow::Context;
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, RgbImage};
use std::path::{Path, PathBuf};

/// 默认 JPEG 质量：98 与像素蛋糕导出的量化表一致。
pub const DEFAULT_JPEG_QUALITY: u8 = 98;

/// 随照片携带、写出时需要保留的元数据。
#[derive(Clone, Debug, Default)]
pub struct PhotoMetadata {
    /// ICC 配置文件（原样写回）
    pub icc: Option<Vec<u8>>,
    /// EXIF（TIFF 数据，不含 `Exif\0\0` 头）。像素已按方向转正，其中的方向标记已置为 1
    pub exif: Option<Vec<u8>>,
    /// 原文件的 EXIF 方向（1..=8，无标记时为 1）
    pub orientation: u8,
    /// 内嵌的 Camera Raw 冲印设置（XMP `crs:HasSettings="True"` 时）
    pub develop: Option<DevelopSettings>,
}

/// 解码后的照片：正立的 8 位 RGB 像素 + 元数据。
#[derive(Clone, Debug)]
pub struct Photo {
    pub image: RgbImage,
    pub meta: PhotoMetadata,
}

impl Photo {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::decode(&bytes).with_context(|| format!("decode {}", path.display()))
    }

    /// 从文件字节解码（格式按内容识别）。
    pub fn decode(bytes: &[u8]) -> anyhow::Result<Self> {
        let mut decoder = ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()?
            .into_decoder()?;
        let icc = decoder.icc_profile().ok().flatten();
        let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
        let mut exif = decoder.exif_metadata().ok().flatten();
        if let Some(chunk) = exif.as_mut() {
            // 像素会被转正，写回的 EXIF 方向必须置为 1（无方向标记时不改动）
            let _ = Orientation::remove_from_exif_chunk(chunk);
        }
        let mut image = DynamicImage::from_decoder(decoder)?;
        image.apply_orientation(orientation);
        Ok(Self {
            image: image.to_rgb8(),
            meta: PhotoMetadata {
                icc,
                exif,
                orientation: orientation.to_exif(),
                develop: DevelopSettings::from_file_bytes(bytes),
            },
        })
    }

    /// 再应用内嵌的 Camera Raw 冲印设置（见 `color::develop`）。有设置且会改变图像时应用并返回该设置。
    pub fn apply_embedded_develop(&mut self) -> Option<&DevelopSettings> {
        let settings = self.meta.develop.as_ref().filter(|s| !s.is_noop())?;
        settings.apply_rgb8(&mut self.image);
        Some(settings)
    }
}

/// 按扩展名判断是否写 JPEG。
pub fn is_jpeg_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
}

/// 编码为 JPEG 字节（jpeg-encoder；质量 ≥ 90 时 4:4:4；标准霍夫曼表）。
pub fn encode_jpeg(img: &RgbImage, quality: u8, meta: &PhotoMetadata) -> anyhow::Result<Vec<u8>> {
    let (Ok(w), Ok(h)) = (u16::try_from(img.width()), u16::try_from(img.height())) else {
        anyhow::bail!("JPEG 尺寸上限 65535，实际 {}x{}", img.width(), img.height());
    };
    let quality = quality.clamp(1, 100);
    let mut out = Vec::with_capacity(img.as_raw().len() / 4);
    let mut enc = jpeg_encoder::Encoder::new(&mut out, quality);
    if quality >= 90 {
        enc.set_sampling_factor(jpeg_encoder::SamplingFactor::F_1_1);
    }
    if let Some(icc) = &meta.icc {
        enc.add_icc_profile(icc)?;
    }
    // APP1 段上限 65535 字节（含 6 字节 Exif 头与 2 字节长度）；超长的 EXIF（大缩略图）放弃写回
    if let Some(exif) = meta.exif.as_ref().filter(|e| e.len() <= 65_528) {
        enc.add_exif_metadata(exif)?;
    }
    enc.encode(img.as_raw(), w, h, jpeg_encoder::ColorType::Rgb)?;
    Ok(out)
}

/// 保存照片：格式由扩展名决定（jpg / jpeg 用 jpeg-encoder 并写回元数据，其余交给 image 库）。
/// 先写同目录的临时文件再改名（覆盖已存在的目标）。
pub fn save(
    img: &RgbImage,
    path: &Path,
    jpeg_quality: u8,
    meta: &PhotoMetadata,
) -> anyhow::Result<()> {
    let tmp = partial_path(path);
    let written = if is_jpeg_path(path) {
        encode_jpeg(img, jpeg_quality, meta)
            .and_then(|bytes| std::fs::write(&tmp, bytes).map_err(Into::into))
    } else {
        ImageFormat::from_path(path)
            .map_err(anyhow::Error::from)
            .and_then(|fmt| img.save_with_format(&tmp, fmt).map_err(Into::into))
    };
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.context(format!("write {}", path.display())));
    }
    std::fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))
}

/// 原子写入用的临时文件：`<dir>/.<name>.partial`（扩展名不是图片格式，批处理扫描时会被忽略）。
pub fn partial_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!(".{name}.partial"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 只含 Orientation 一个条目的最小 EXIF（小端 TIFF）。
    fn exif_with_orientation(o: u16) -> Vec<u8> {
        let mut v = b"II*\0".to_vec();
        v.extend_from_slice(&8u32.to_le_bytes()); // IFD0 偏移
        v.extend_from_slice(&1u16.to_le_bytes()); // 条目数
        v.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
        v.extend_from_slice(&3u16.to_le_bytes()); // SHORT
        v.extend_from_slice(&1u32.to_le_bytes()); // count
        v.extend_from_slice(&o.to_le_bytes());
        v.extend_from_slice(&[0, 0]);
        v.extend_from_slice(&0u32.to_le_bytes()); // 下一个 IFD
        v
    }

    /// 3×2 图，每个像素颜色唯一（用 JPEG q100 编码后仍可区分：相邻值相差 ≥ 80）。
    fn test_image() -> RgbImage {
        let mut img = RgbImage::new(3, 2);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 100) as u8, (y * 200) as u8, 30]);
        }
        img
    }

    fn near(a: image::Rgb<u8>, b: image::Rgb<u8>) -> bool {
        a.0.iter()
            .zip(b.0)
            .all(|(x, y)| (*x as i32 - y as i32).abs() <= 24)
    }

    #[test]
    fn decode_applies_exif_orientation_and_resets_tag() {
        let src = test_image();
        let meta = PhotoMetadata {
            exif: Some(exif_with_orientation(6)),
            ..Default::default()
        };
        let bytes = encode_jpeg(&src, 100, &meta).unwrap();
        let photo = Photo::decode(&bytes).unwrap();
        // Orientation 6：显示时需顺时针转 90°，3×2 → 2×3；逐点与 rotate90 的结果核对
        assert_eq!(photo.meta.orientation, 6);
        assert_eq!(photo.image.dimensions(), (2, 3));
        let rot = image::imageops::rotate90(&src);
        for (x, y, p) in photo.image.enumerate_pixels() {
            assert!(
                near(*p, *rot.get_pixel(x, y)),
                "({x},{y}) {p:?} vs {:?}",
                rot.get_pixel(x, y)
            );
        }
        let exif = photo.meta.exif.as_deref().unwrap();
        assert_eq!(
            Orientation::from_exif_chunk(exif),
            Some(Orientation::NoTransforms)
        );
        // 重新编码后再读：不会被再转一次
        let again = Photo::decode(&encode_jpeg(&photo.image, 100, &photo.meta).unwrap()).unwrap();
        assert_eq!(again.meta.orientation, 1);
        assert_eq!(again.image.dimensions(), (2, 3));
    }

    #[test]
    fn jpeg_roundtrip_keeps_icc_and_skips_xmp() {
        // 16×16 纯色：曾触发 jpeg-encoder 优化霍夫曼表与 zune-jpeg 的不兼容（解出 (24, 32, 0)）
        let icc = vec![7u8; 300];
        let meta = PhotoMetadata {
            icc: Some(icc.clone()),
            exif: Some(exif_with_orientation(1)),
            ..Default::default()
        };
        let img = RgbImage::from_pixel(16, 16, image::Rgb([120, 90, 60]));
        let bytes = encode_jpeg(&img, 98, &meta).unwrap();
        let photo = Photo::decode(&bytes).unwrap();
        assert_eq!(photo.meta.icc.as_deref(), Some(&icc[..]));
        assert_eq!(photo.meta.orientation, 1);
        assert!(photo.meta.develop.is_none());
        assert!(near(
            *photo.image.get_pixel(8, 8),
            image::Rgb([120, 90, 60])
        ));
    }

    #[test]
    fn save_is_atomic_and_picks_format_by_extension() {
        let dir = std::env::temp_dir().join(format!("pr_photo_save_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let img = test_image();
        let meta = PhotoMetadata::default();
        for name in ["a.JPG", "b.png"] {
            let p = dir.join(name);
            save(&img, &p, 95, &meta).unwrap();
            assert!(p.exists());
            assert!(!partial_path(&p).exists());
            assert_eq!(Photo::load(&p).unwrap().image.dimensions(), (3, 2));
        }
        // PNG 无损
        assert_eq!(Photo::load(&dir.join("b.png")).unwrap().image, img);
        // 不支持的扩展名：报错且不留临时文件
        let bad = dir.join("c.unknownext");
        assert!(save(&img, &bad, 95, &meta).is_err());
        assert!(!partial_path(&bad).exists() && !bad.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
