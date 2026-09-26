//! 应用级命令：版本与默认参数、引擎设置 / 加载、文件对话框、在资源管理器中显示。

use super::blocking;
use crate::engine_host::{EngineSettings, EngineStatus};
use crate::error::{CmdResult, CommandError};
use crate::state::AppState;
use portrait_retouch::batch::DEFAULT_EXTENSIONS;
use portrait_retouch::{ParamOptions, Preset};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{AppHandle, Manager, State, WebviewWindow};
use tauri_plugin_dialog::{DialogExt, FileDialogBuilder, FilePath};
use tauri_plugin_opener::OpenerExt;

/// 内置预设（界面下拉框）。
#[derive(Debug, Serialize)]
pub struct PresetInfo {
    /// 传给 `ParamOptions::preset` 的名字
    pub id: String,
    pub name: String,
    pub description: String,
}

/// 界面启动时取的信息。
#[derive(Debug, Serialize)]
pub struct AppInfo {
    pub version: &'static str,
    pub presets: Vec<PresetInfo>,
    /// 修图参数默认值（与命令行一致）
    pub defaults: ParamOptions,
    /// 引擎设置默认值（"恢复默认"用）
    pub engine_defaults: EngineSettings,
    pub engine: EngineStatus,
    pub batch_running: bool,
}

#[tauri::command]
pub fn app_info(state: State<'_, AppState>) -> AppInfo {
    AppInfo {
        version: env!("CARGO_PKG_VERSION"),
        presets: builtin_presets(),
        defaults: ParamOptions::default(),
        engine_defaults: EngineSettings::default(),
        engine: state.engine.status(),
        batch_running: state.batch.is_running(),
    }
}

fn builtin_presets() -> Vec<PresetInfo> {
    Preset::builtin_names()
        .iter()
        .filter_map(|id| {
            let p = Preset::load_named(id).ok()?;
            Some(PresetInfo {
                id: id.to_string(),
                name: p.name,
                description: p.description,
            })
        })
        .collect()
}

/// 预设的完整内容（JSON），供界面查看可以用 `sets` 覆盖的字段。
#[tauri::command]
pub fn preset_json(name: String) -> CmdResult<String> {
    let preset = Preset::load_named(&name)?;
    serde_json::to_string_pretty(&preset).map_err(|e| CommandError::msg(e.to_string()))
}

#[tauri::command]
pub fn engine_status(state: State<'_, AppState>) -> EngineStatus {
    state.engine.status()
}

/// 立即加载模型（否则在第一次需要时加载）。
#[tauri::command]
pub async fn load_engine(app: AppHandle) -> CmdResult<EngineStatus> {
    blocking(move || {
        let state = app.state::<AppState>();
        state.engine.get()?;
        Ok(state.engine.status())
    })
    .await
}

/// 修改引擎设置：丢弃已加载的引擎，下次使用时按新设置加载。
#[tauri::command]
pub async fn set_engine_settings(
    app: AppHandle,
    settings: EngineSettings,
) -> CmdResult<EngineStatus> {
    blocking(move || {
        let state = app.state::<AppState>();
        let _work = state.begin_work()?;
        state.engine.update(settings);
        Ok(state.engine.status())
    })
    .await
}

/// 文件选择对话框的文件类型。
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileKind {
    /// 照片
    Image,
    /// 风格 LUT：.cube 或 512×512 查找图
    Lut,
    /// 512×512 查找图（美白）
    Lookup,
    /// JSON（预设、形变系数）
    Json,
}

impl FileKind {
    fn filter(self) -> (&'static str, &'static [&'static str]) {
        match self {
            FileKind::Image => ("图片", &DEFAULT_EXTENSIONS),
            FileKind::Lut => ("LUT（.cube / 512×512 查找图）", &["cube", "png"]),
            FileKind::Lookup => ("512×512 查找图", &["png"]),
            FileKind::Json => ("JSON", &["json"]),
        }
    }
}

/// 以主窗口为父窗口的（模态）文件对话框。
fn dialog(window: &WebviewWindow) -> FileDialogBuilder<tauri::Wry> {
    window.dialog().file().set_parent(window)
}

fn to_path(p: FilePath) -> Option<PathBuf> {
    p.into_path().ok()
}

#[tauri::command]
pub async fn pick_file(window: WebviewWindow, kind: FileKind) -> CmdResult<Option<PathBuf>> {
    blocking(move || {
        let (name, exts) = kind.filter();
        Ok(dialog(&window)
            .add_filter(name, exts)
            .blocking_pick_file()
            .and_then(to_path))
    })
    .await
}

/// 选多张照片（批处理输入）。
#[tauri::command]
pub async fn pick_files(window: WebviewWindow) -> CmdResult<Vec<PathBuf>> {
    blocking(move || {
        let (name, exts) = FileKind::Image.filter();
        Ok(dialog(&window)
            .add_filter(name, exts)
            .blocking_pick_files()
            .unwrap_or_default()
            .into_iter()
            .filter_map(to_path)
            .collect())
    })
    .await
}

#[tauri::command]
pub async fn pick_folder(window: WebviewWindow) -> CmdResult<Option<PathBuf>> {
    blocking(move || Ok(dialog(&window).blocking_pick_folder().and_then(to_path))).await
}

/// 保存对话框（JPEG / PNG，按扩展名决定格式）。
#[tauri::command]
pub async fn pick_save_path(
    window: WebviewWindow,
    file_name: String,
    directory: Option<PathBuf>,
) -> CmdResult<Option<PathBuf>> {
    blocking(move || {
        let mut d = dialog(&window)
            .add_filter("JPEG", &["jpg", "jpeg"])
            .add_filter("PNG", &["png"])
            .set_file_name(file_name);
        if let Some(dir) = directory.filter(|d| d.is_dir()) {
            d = d.set_directory(dir);
        }
        Ok(d.blocking_save_file().and_then(to_path))
    })
    .await
}

/// 在资源管理器中显示（选中）文件或目录。
#[tauri::command]
pub fn reveal_path(app: AppHandle, path: PathBuf) -> CmdResult<()> {
    app.opener()
        .reveal_item_in_dir(&path)
        .map_err(|e| CommandError::msg(format!("无法显示 {}：{e}", path.display())))
}

/// 用系统默认程序打开文件或目录。
#[tauri::command]
pub fn open_path(app: AppHandle, path: PathBuf) -> CmdResult<()> {
    app.opener()
        .open_path(path.to_string_lossy(), None::<&str>)
        .map_err(|e| CommandError::msg(format!("无法打开 {}：{e}", path.display())))
}
