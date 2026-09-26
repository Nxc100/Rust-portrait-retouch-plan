//! 单张照片：打开（检测人脸）→ 在工作分辨率上调参处理 → 查看调试视图 → 按原图分辨率保存。

use super::blocking;
use crate::error::{CmdResult, CommandError};
use crate::session::{retouch_working, PhotoInfo, Session};
use crate::state::AppState;
use crate::sync::lock;
use crate::views::{self, ViewKind};
use image::RgbImage;
use portrait_retouch::{photo, FaceKeyPoints, ParamOptions, Photo};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, Manager};

/// 一次处理的摘要。
#[derive(Debug, Serialize)]
pub struct ProcessInfo {
    /// 会话图像版本（前端据此刷新图片 URL）
    pub version: u64,
    pub ms: f64,
    pub width: u32,
    pub height: u32,
    /// 是否再应用了内嵌冲印设置
    pub develop_applied: bool,
    /// 没有加载模型时的提示（只做了与人脸无关的调整）
    pub note: Option<String>,
}

/// 视图摘要。
#[derive(Debug, Serialize)]
pub struct ViewInfo {
    pub version: u64,
    pub width: u32,
    pub height: u32,
    /// 生成耗时（取缓存时为 0）
    pub ms: f64,
}

/// 保存结果。
#[derive(Debug, Serialize)]
pub struct SaveInfo {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub ms: f64,
    /// 是否按原图分辨率重新处理（工作分辨率小于原图时）
    pub reprocessed: bool,
}

/// 打开照片：解码（EXIF 转正）、检测人脸、按工作分辨率缩小。模型缺失时照样打开，只是没有人脸。
#[tauri::command]
pub async fn open_image(app: AppHandle, path: PathBuf, long_side: u32) -> CmdResult<PhotoInfo> {
    blocking(move || {
        let state = app.state::<AppState>();
        let _work = state.begin_work()?;
        let photo = Photo::load(&path)?;
        if let Some(engine) = state.engine.loaded() {
            engine.clear_cache(); // 释放上一张照片的中间结果
        }
        let d = detect(&state, &photo.image);
        let session = Session::new(path, photo, d.faces, d.ms, d.note, long_side);
        let info = session.info();
        *lock(&state.session) = Some(session);
        Ok(info)
    })
    .await
}

/// 当前照片（界面刷新后恢复显示）。
#[tauri::command]
pub fn current_photo(app: AppHandle) -> Option<PhotoInfo> {
    lock(&app.state::<AppState>().session)
        .as_ref()
        .map(Session::info)
}

/// 切换工作分辨率（0 = 原图）；已有的结果与调试视图作废。
#[tauri::command]
pub async fn set_working_size(app: AppHandle, long_side: u32) -> CmdResult<PhotoInfo> {
    blocking(move || {
        let state = app.state::<AppState>();
        let _work = state.begin_work()?;
        let mut slot = lock(&state.session);
        let s = current_mut(&mut slot)?;
        s.set_long_side(long_side);
        Ok(s.info())
    })
    .await
}

/// 用当前引擎重新检测人脸（改了关键点模型或模型目录之后）。
#[tauri::command]
pub async fn redetect_faces(app: AppHandle) -> CmdResult<PhotoInfo> {
    blocking(move || {
        let state = app.state::<AppState>();
        let _work = state.begin_work()?;
        let original = current(&lock(&state.session))?.original.clone();
        let d = detect(&state, &original);
        let mut slot = lock(&state.session);
        let s = current_mut(&mut slot)?;
        s.set_faces(d.faces, d.ms, d.note);
        Ok(s.info())
    })
    .await
}

/// 按参数处理当前照片（工作分辨率）。
#[tauri::command]
pub async fn process_photo(app: AppHandle, options: ParamOptions) -> CmdResult<ProcessInfo> {
    blocking(move || {
        let state = app.state::<AppState>();
        let params = options.build()?;
        let _work = state.begin_work()?;
        let (working, develop) = {
            let slot = lock(&state.session);
            let s = current(&slot)?;
            (s.working().clone(), s.meta.develop.clone())
        };
        let engine = state.engine.get().ok();
        let t = Instant::now();
        let (image, develop_applied) =
            retouch_working(engine.as_deref(), &working, develop.as_ref(), &params);
        let ms = elapsed_ms(t);
        let (width, height) = image.dimensions();
        let mut slot = lock(&state.session);
        let s = current_mut(&mut slot)?;
        s.set_result(image, options);
        Ok(ProcessInfo {
            version: s.version(),
            ms,
            width,
            height,
            develop_applied,
            note: engine
                .is_none()
                .then(|| "没有加载模型：只做了与人脸无关的调整（见\"引擎设置\"）".to_string()),
        })
    })
    .await
}

/// 准备一个视图（调试视图首次请求时生成并缓存），之后前端用 `photo://` 取图。
#[tauri::command]
pub async fn prepare_view(app: AppHandle, kind: ViewKind) -> CmdResult<ViewInfo> {
    blocking(move || {
        let state = app.state::<AppState>();
        let _work = state.begin_work()?;
        let working = {
            let slot = lock(&state.session);
            let s = current(&slot)?;
            if let Some(image) = s.view(kind) {
                return Ok(view_info(&image, s.version(), 0.0));
            }
            if kind == ViewKind::Result {
                return Err(CommandError::msg("还没有处理结果"));
            }
            s.working().clone()
        };
        let engine = if kind.needs_engine() {
            Some(state.engine.get()?)
        } else {
            None
        };
        let t = Instant::now();
        let image = views::render(kind, engine.as_deref(), &working)?;
        let ms = elapsed_ms(t);
        let mut slot = lock(&state.session);
        let s = current_mut(&mut slot)?;
        let info = view_info(&image, s.version(), ms);
        s.cache_view(kind, image);
        Ok(info)
    })
    .await
}

/// 保存结果：工作分辨率小于原图时，按同样的参数在原图分辨率上重新处理后保存（JPEG 质量 98，写回 ICC / EXIF）。
#[tauri::command]
pub async fn save_result(app: AppHandle, path: PathBuf) -> CmdResult<SaveInfo> {
    blocking(move || {
        let state = app.state::<AppState>();
        let _work = state.begin_work()?;
        let t = Instant::now();
        let (options, full_result, full, meta) = {
            let slot = lock(&state.session);
            let s = current(&slot)?;
            let options = s
                .result_options()
                .cloned()
                .ok_or_else(|| CommandError::msg("还没有处理结果，请先处理"))?;
            let full_result = s.result().filter(|_| s.is_full_resolution());
            (options, full_result, s.full_working(), s.meta.clone())
        };
        let (image, reprocessed) = match full_result {
            Some(image) => (image, false),
            None => {
                let params = options.build()?;
                let engine = state.engine.get().ok();
                let (image, _) =
                    retouch_working(engine.as_deref(), &full, meta.develop.as_ref(), &params);
                (Arc::new(image), true)
            }
        };
        photo::save(&image, &path, photo::DEFAULT_JPEG_QUALITY, &meta)?;
        Ok(SaveInfo {
            path,
            width: image.width(),
            height: image.height(),
            ms: elapsed_ms(t),
            reprocessed,
        })
    })
    .await
}

/// 人脸检测结果；模型加载失败或检测失败时人脸为空并附上原因（照片照样能打开和做全局调整）。
struct Detection {
    faces: Vec<FaceKeyPoints>,
    ms: f64,
    note: Option<String>,
}

fn detect(state: &AppState, image: &RgbImage) -> Detection {
    let engine = match state.engine.get() {
        Ok(e) => e,
        Err(e) => {
            return Detection {
                faces: Vec::new(),
                ms: 0.0,
                note: Some(format!("{e:#}")),
            }
        }
    };
    let t = Instant::now();
    match engine.detect_faces(image) {
        Ok(faces) => Detection {
            note: faces.is_empty().then(|| "没有检测到人脸".to_string()),
            faces,
            ms: elapsed_ms(t),
        },
        Err(e) => Detection {
            faces: Vec::new(),
            ms: elapsed_ms(t),
            note: Some(format!("人脸检测失败：{e:#}")),
        },
    }
}

fn current(slot: &Option<Session>) -> CmdResult<&Session> {
    slot.as_ref().ok_or_else(no_photo)
}

fn current_mut(slot: &mut Option<Session>) -> CmdResult<&mut Session> {
    slot.as_mut().ok_or_else(no_photo)
}

fn no_photo() -> CommandError {
    CommandError::msg("请先打开一张照片")
}

fn view_info(image: &RgbImage, version: u64, ms: f64) -> ViewInfo {
    ViewInfo {
        version,
        width: image.width(),
        height: image.height(),
        ms,
    }
}

fn elapsed_ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}
