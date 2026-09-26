//! 当前照片的会话：原图（EXIF 已转正）、人脸、工作分辨率下的图像、处理结果与调试视图缓存。
//!
//! "工作分辨率"：为了能边拖滑块边看效果，默认在长边 1600 px 的缩小图上处理（人脸关键点同比缩放）；
//! 选"原图"则在全分辨率上处理。保存时总是按原图分辨率重新处理。

use crate::views::ViewKind;
use image::imageops::FilterType;
use image::RgbImage;
use portrait_retouch::batch::FaceSummary;
use portrait_retouch::color::develop::DevelopSettings;
use portrait_retouch::{Engine, FaceKeyPoints, ParamOptions, Photo, PhotoMetadata, RetouchParams};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// 工作分辨率下的图像与人脸。
#[derive(Clone)]
pub struct Working {
    /// 长边上限（0 = 原图分辨率）
    pub long_side: u32,
    /// 相对原图的缩放
    pub scale: f32,
    pub image: Arc<RgbImage>,
    pub faces: Arc<Vec<FaceKeyPoints>>,
}

impl Working {
    /// 按长边上限缩小原图（不放大），人脸关键点同比缩放。
    pub fn build(original: &Arc<RgbImage>, faces: &[FaceKeyPoints], long_side: u32) -> Self {
        let (w, h) = original.dimensions();
        let longest = w.max(h).max(1);
        if long_side == 0 || long_side >= longest {
            return Self {
                long_side,
                scale: 1.0,
                image: original.clone(),
                faces: Arc::new(faces.to_vec()),
            };
        }
        let s = long_side as f32 / longest as f32;
        let (nw, nh) = (
            ((w as f32 * s).round() as u32).max(1),
            ((h as f32 * s).round() as u32).max(1),
        );
        Self {
            long_side,
            scale: s,
            image: Arc::new(image::imageops::resize(
                original.as_ref(),
                nw,
                nh,
                FilterType::Triangle,
            )),
            faces: Arc::new(faces.iter().map(|f| f.scaled(s)).collect()),
        }
    }

    /// 同一工作图像换一组（原图坐标的）人脸。
    pub fn with_faces(&self, faces: &[FaceKeyPoints]) -> Self {
        Self {
            long_side: self.long_side,
            scale: self.scale,
            image: self.image.clone(),
            faces: Arc::new(faces.iter().map(|f| f.scaled(self.scale)).collect()),
        }
    }
}

/// 在工作图像上修图：需要时先再应用内嵌冲印设置；没有引擎（模型缺失）时只做与人脸无关的处理。
pub fn retouch_working(
    engine: Option<&Engine>,
    working: &Working,
    develop: Option<&DevelopSettings>,
    params: &RetouchParams,
) -> (RgbImage, bool) {
    let developed;
    let (img, develop_applied): (&RgbImage, bool) = match develop {
        Some(s) if params.embedded_develop && !s.is_noop() => {
            let mut copy = working.image.as_ref().clone();
            s.apply_rgb8(&mut copy);
            developed = copy;
            (&developed, true)
        }
        _ => (working.image.as_ref(), false),
    };
    let out = match engine {
        Some(e) => e.retouch(img, &working.faces, params),
        None => portrait_retouch::retouch_impl(img, &[], params),
    };
    (out, develop_applied)
}

/// 下一个图像版本号：整个进程内唯一递增。换一张照片后版本号也不会与上一张重复，
/// 前端按 URL 判断"图是否变了"时不会把上一张照片的图当成当前的。
fn next_version() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// 一张打开的照片。
pub struct Session {
    pub path: PathBuf,
    pub original: Arc<RgbImage>,
    pub meta: PhotoMetadata,
    /// 全分辨率上检测到的人脸
    pub faces: Vec<FaceKeyPoints>,
    pub detect_ms: f64,
    /// 没有做人脸检测的原因（如模型缺失）
    pub detect_note: Option<String>,
    working: Working,
    result: Option<Arc<RgbImage>>,
    /// 得到 `result` 所用的参数（保存时按同样的参数在原图分辨率上重做）
    result_options: Option<ParamOptions>,
    views: HashMap<ViewKind, Arc<RgbImage>>,
    version: u64,
}

impl Session {
    pub fn new(
        path: PathBuf,
        photo: Photo,
        faces: Vec<FaceKeyPoints>,
        detect_ms: f64,
        detect_note: Option<String>,
        long_side: u32,
    ) -> Self {
        let original = Arc::new(photo.image);
        let working = Working::build(&original, &faces, long_side);
        Self {
            path,
            original,
            meta: photo.meta,
            faces,
            detect_ms,
            detect_note,
            working,
            result: None,
            result_options: None,
            views: HashMap::new(),
            version: next_version(),
        }
    }

    pub fn working(&self) -> &Working {
        &self.working
    }

    /// 全分辨率的工作图像（保存用）。
    pub fn full_working(&self) -> Working {
        Working::build(&self.original, &self.faces, 0)
    }

    /// 图像版本号：任何图像变化都会递增（进程内唯一），前端据此刷新图片 URL。
    pub fn version(&self) -> u64 {
        self.version
    }

    /// 工作图像就是原图（保存时可直接用当前结果）。
    pub fn is_full_resolution(&self) -> bool {
        Arc::ptr_eq(&self.working.image, &self.original)
    }

    /// 切换工作分辨率：处理结果与调试视图随之失效。
    pub fn set_long_side(&mut self, long_side: u32) {
        if long_side == self.working.long_side {
            return;
        }
        self.working = Working::build(&self.original, &self.faces, long_side);
        self.invalidate();
    }

    /// 换成重新检测的人脸：处理结果与调试视图随之失效。
    pub fn set_faces(
        &mut self,
        faces: Vec<FaceKeyPoints>,
        detect_ms: f64,
        detect_note: Option<String>,
    ) {
        self.working = self.working.with_faces(&faces);
        self.faces = faces;
        self.detect_ms = detect_ms;
        self.detect_note = detect_note;
        self.invalidate();
    }

    fn invalidate(&mut self) {
        self.result = None;
        self.result_options = None;
        self.views.clear();
        self.version = next_version();
    }

    pub fn set_result(&mut self, image: RgbImage, options: ParamOptions) {
        self.result = Some(Arc::new(image));
        self.result_options = Some(options);
        self.version = next_version();
    }

    pub fn result(&self) -> Option<Arc<RgbImage>> {
        self.result.clone()
    }

    pub fn result_options(&self) -> Option<&ParamOptions> {
        self.result_options.as_ref()
    }

    /// 可直接显示的图像：原图 / 结果 / 已缓存的调试视图。
    pub fn view(&self, kind: ViewKind) -> Option<Arc<RgbImage>> {
        match kind {
            ViewKind::Original => Some(self.working.image.clone()),
            ViewKind::Result => self.result.clone(),
            _ => self.views.get(&kind).cloned(),
        }
    }

    pub fn cache_view(&mut self, kind: ViewKind, image: RgbImage) {
        self.views.insert(kind, Arc::new(image));
    }

    pub fn info(&self) -> PhotoInfo {
        let (w, h) = self.original.dimensions();
        let (ww, wh) = self.working.image.dimensions();
        PhotoInfo {
            path: self.path.clone(),
            file_name: file_name(&self.path),
            width: w,
            height: h,
            orientation: self.meta.orientation,
            has_icc: self.meta.icc.is_some(),
            develop_settings: self.meta.develop.as_ref().map(|s| s.summary()),
            faces: self.faces.iter().map(FaceSummary::from).collect(),
            detect_ms: self.detect_ms,
            detect_note: self.detect_note.clone(),
            working: WorkingInfo {
                long_side: self.working.long_side,
                width: ww,
                height: wh,
                scale: self.working.scale,
            },
            has_result: self.result.is_some(),
            version: self.version,
        }
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// 前端看到的照片信息。
#[derive(Clone, Debug, Serialize)]
pub struct PhotoInfo {
    pub path: PathBuf,
    pub file_name: String,
    pub width: u32,
    pub height: u32,
    /// 原文件的 EXIF 方向（已转正）
    pub orientation: u8,
    pub has_icc: bool,
    /// 内嵌 Camera Raw 设置摘要
    pub develop_settings: Option<String>,
    pub faces: Vec<FaceSummary>,
    pub detect_ms: f64,
    pub detect_note: Option<String>,
    pub working: WorkingInfo,
    pub has_result: bool,
    pub version: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorkingInfo {
    pub long_side: u32,
    pub width: u32,
    pub height: u32,
    pub scale: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn photo(w: u32, h: u32) -> Photo {
        Photo {
            image: RgbImage::from_pixel(w, h, image::Rgb([120, 100, 90])),
            meta: PhotoMetadata::default(),
        }
    }

    #[test]
    fn working_image_is_downscaled_but_never_upscaled() {
        let original = Arc::new(photo(4000, 3000).image);
        let w = Working::build(&original, &[], 1600);
        assert_eq!(w.image.dimensions(), (1600, 1200));
        assert!((w.scale - 0.4).abs() < 1e-6);
        let full = Working::build(&original, &[], 0);
        assert!(
            Arc::ptr_eq(&full.image, &original),
            "full resolution shares the original"
        );
        let small = Working::build(&original, &[], 8000);
        assert_eq!(small.image.dimensions(), (4000, 3000));
    }

    #[test]
    fn changing_resolution_invalidates_result_and_bumps_version() {
        let mut s = Session::new(
            PathBuf::from("a.jpg"),
            photo(3000, 2000),
            vec![],
            0.0,
            None,
            1600,
        );
        let v0 = s.version();
        s.set_result(RgbImage::new(1600, 1067), ParamOptions::default());
        assert!(s.result().is_some() && s.version() > v0);
        s.cache_view(ViewKind::Landmarks, RgbImage::new(2, 2));
        assert!(s.view(ViewKind::Landmarks).is_some());
        let v1 = s.version();
        s.set_long_side(0);
        assert!(s.result().is_none());
        assert!(s.view(ViewKind::Landmarks).is_none());
        assert!(s.version() > v1);
        assert_eq!(s.info().working.width, 3000);
        // 同一分辨率不重复失效
        let v2 = s.version();
        s.set_long_side(0);
        assert_eq!(s.version(), v2);
    }

    #[test]
    fn versions_are_unique_across_sessions() {
        let a = Session::new(PathBuf::from("a.jpg"), photo(8, 8), vec![], 0.0, None, 0);
        let b = Session::new(PathBuf::from("b.jpg"), photo(8, 8), vec![], 0.0, None, 0);
        assert_ne!(
            a.version(),
            b.version(),
            "a new photo never reuses an image URL"
        );
    }

    #[test]
    fn new_faces_keep_the_working_image_but_drop_results() {
        let mut s = Session::new(
            PathBuf::from("a.jpg"),
            photo(800, 600),
            vec![],
            0.0,
            None,
            400,
        );
        assert!(!s.is_full_resolution());
        let image = s.working().image.clone();
        s.set_result(RgbImage::new(400, 300), ParamOptions::default());
        let v = s.version();
        s.set_faces(vec![], 3.0, Some("没有检测到人脸".into()));
        assert!(
            Arc::ptr_eq(&s.working().image, &image),
            "no resampling needed"
        );
        assert!(s.result().is_none() && s.result_options().is_none());
        assert!(s.version() > v);
        assert_eq!(s.info().detect_note.as_deref(), Some("没有检测到人脸"));
        s.set_long_side(0);
        assert!(s.is_full_resolution());
    }

    #[test]
    fn retouch_without_engine_applies_global_adjustments() {
        let original = Arc::new(photo(64, 48).image);
        let working = Working::build(&original, &[], 0);
        let params = ParamOptions {
            brightness: 1.3,
            ..Default::default()
        }
        .build()
        .unwrap();
        let (out, developed) = retouch_working(None, &working, None, &params);
        assert!(!developed);
        assert_eq!(out.dimensions(), (64, 48));
        assert!(
            out.get_pixel(10, 10)[0] > 120,
            "brightness applied without faces"
        );
    }
}
