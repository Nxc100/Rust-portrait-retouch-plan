//! 批量处理：列出任务 → 后台线程执行（逐张推送进度事件）→ 写报告；可随时取消。
//!
//! 事件：
//! - [`PROGRESS_EVENT`]：[`Progress`]，每张开始 / 结束各一次；
//! - [`FINISHED_EVENT`]：[`BatchSummary`]，整批结束（完成、取消或无法进行）时一次。

use super::blocking;
use crate::error::{CmdResult, CommandError};
use crate::state::{AppState, BatchTicket};
use portrait_retouch::batch::{
    self, BatchConfig, BatchEvent, BatchItem, BatchReport, ItemReport, ItemStatus, OutputFormat,
};
use portrait_retouch::{photo, ParamOptions, RetouchParams};
use serde::{Deserialize, Serialize};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Manager, State};

pub const PROGRESS_EVENT: &str = "batch-progress";
pub const FINISHED_EVENT: &str = "batch-finished";

/// 界面的批处理请求。
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct BatchRequest {
    /// 目录或文件（可混合）
    pub inputs: Vec<PathBuf>,
    pub output_dir: PathBuf,
    /// 递归子目录（输出保留相对目录结构）
    pub recursive: bool,
    /// 覆盖已存在的输出（默认跳过）
    pub overwrite: bool,
    pub format: OutputFormat,
    pub jpeg_quality: u8,
    /// 同时写出人脸关键点 JSON（输出目录下的 `landmarks/`）
    pub write_landmarks: bool,
    pub options: ParamOptions,
}

impl Default for BatchRequest {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            output_dir: PathBuf::new(),
            recursive: false,
            overwrite: false,
            format: OutputFormat::Jpeg,
            jpeg_quality: photo::DEFAULT_JPEG_QUALITY,
            write_landmarks: false,
            options: ParamOptions::default(),
        }
    }
}

impl BatchRequest {
    fn to_config(&self) -> CmdResult<BatchConfig> {
        if self.inputs.is_empty() {
            return Err(CommandError::msg("请添加输入目录或照片"));
        }
        if self.output_dir.as_os_str().is_empty() {
            return Err(CommandError::msg("请选择输出目录"));
        }
        Ok(BatchConfig {
            recursive: self.recursive,
            format: self.format,
            jpeg_quality: self.jpeg_quality.clamp(1, 100),
            overwrite: self.overwrite,
            landmarks_dir: self
                .write_landmarks
                .then(|| self.output_dir.join("landmarks")),
            ..BatchConfig::new(self.inputs.clone(), self.output_dir.clone())
        })
    }
}

/// 一项任务（界面表格的一行）。
#[derive(Clone, Debug, Serialize)]
pub struct PlannedItem {
    pub input: PathBuf,
    pub output: PathBuf,
    /// 输出已存在（不覆盖时会跳过）
    pub exists: bool,
}

fn planned(items: &[BatchItem]) -> Vec<PlannedItem> {
    items
        .iter()
        .map(|i| PlannedItem {
            input: i.input.clone(),
            output: i.output.clone(),
            exists: i.output.exists(),
        })
        .collect()
}

/// 进度事件。
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Progress {
    Started {
        index: usize,
        total: usize,
    },
    Finished {
        index: usize,
        total: usize,
        report: ItemReport,
    },
}

impl From<BatchEvent<'_>> for Progress {
    fn from(e: BatchEvent<'_>) -> Self {
        match e {
            BatchEvent::Started { index, total, .. } => Progress::Started { index, total },
            BatchEvent::Finished {
                index,
                total,
                report,
            } => Progress::Finished {
                index,
                total,
                report: report.clone(),
            },
        }
    }
}

/// 整批结束的摘要。
#[derive(Clone, Debug, Default, Serialize)]
pub struct BatchSummary {
    pub total: usize,
    pub done: usize,
    pub skipped: usize,
    pub failed: usize,
    pub cancelled: bool,
    pub total_ms: f64,
    pub output_dir: PathBuf,
    pub report_csv: Option<PathBuf>,
    pub report_json: Option<PathBuf>,
    /// 整批无法进行（模型加载失败、建不了输出目录……）或报告写出失败的原因
    pub error: Option<String>,
}

impl BatchSummary {
    fn from_report(report: &BatchReport, output_dir: PathBuf) -> Self {
        Self {
            total: report.items.len(),
            done: report.count(ItemStatus::Done),
            skipped: report.count(ItemStatus::Skipped),
            failed: report.count(ItemStatus::Failed),
            cancelled: report.cancelled,
            total_ms: report.total_ms,
            output_dir,
            ..Default::default()
        }
    }

    fn failed_to_run(total: usize, output_dir: PathBuf, error: String) -> Self {
        Self {
            total,
            output_dir,
            error: Some(error),
            ..Default::default()
        }
    }
}

/// 只列出任务，不执行（界面预览"将处理哪些照片"）。
#[tauri::command]
pub async fn preview_batch(request: BatchRequest) -> CmdResult<Vec<PlannedItem>> {
    blocking(move || Ok(planned(&batch::plan(&request.to_config()?)?))).await
}

/// 开始批处理：参数与任务先在这里校验，出错直接返回；之后在后台线程执行，通过事件报告进度。
/// 返回任务列表（界面据此建表，进度事件里的 `index` 即表中的行号）。
#[tauri::command]
pub async fn start_batch(app: AppHandle, request: BatchRequest) -> CmdResult<Vec<PlannedItem>> {
    blocking(move || {
        let params = request.options.build()?;
        let mut config = request.to_config()?;
        let state = app.state::<AppState>();
        let ticket = state
            .batch
            .try_start()
            .ok_or_else(|| CommandError::msg("已有批处理在运行"))?;
        config.cancel = Some(ticket.cancel_flag());
        let items = batch::plan(&config)?;
        if items.is_empty() {
            return Err(CommandError::msg("输入里没有可处理的图片"));
        }
        let list = planned(&items);
        let app = app.clone();
        std::thread::Builder::new()
            .name("batch".into())
            .spawn(move || run_batch(app, ticket, params, config, items))
            .map_err(|e| CommandError::msg(format!("无法启动批处理线程：{e}")))?;
        Ok(list)
    })
    .await
}

/// 请求取消：正在处理的那张照常完成，其余记为"已取消"。没有在运行的批处理时返回 false。
#[tauri::command]
pub fn cancel_batch(state: State<'_, AppState>) -> bool {
    state.batch.cancel()
}

/// 批处理线程主体。凭据在发出结束事件前释放，界面收到事件时已可开始新的操作。
fn run_batch(
    app: AppHandle,
    ticket: BatchTicket,
    params: RetouchParams,
    config: BatchConfig,
    items: Vec<BatchItem>,
) {
    let summary = catch_unwind(AssertUnwindSafe(|| execute(&app, &params, &config, &items)))
        .unwrap_or_else(|_| {
            BatchSummary::failed_to_run(
                items.len(),
                config.output_dir.clone(),
                "批处理异常中止".into(),
            )
        });
    drop(ticket);
    let _ = app.emit(FINISHED_EVENT, summary);
}

fn execute(
    app: &AppHandle,
    params: &RetouchParams,
    config: &BatchConfig,
    items: &[BatchItem],
) -> BatchSummary {
    let state = app.state::<AppState>();
    let _work = state.lock_work_for_batch();
    let fail = |e: anyhow::Error| {
        BatchSummary::failed_to_run(items.len(), config.output_dir.clone(), format!("{e:#}"))
    };
    let engine = match state.engine.get() {
        Ok(e) => e,
        Err(e) => return fail(e),
    };
    let mut on_event = |e: BatchEvent| {
        let _ = app.emit(PROGRESS_EVENT, Progress::from(e));
    };
    let report = match batch::run(Some(&engine), params, config, items, &mut on_event) {
        Ok(r) => r,
        Err(e) => return fail(e),
    };
    let mut summary = BatchSummary::from_report(&report, config.output_dir.clone());
    match report.write(&config.output_dir) {
        Ok((csv, json)) => {
            summary.report_csv = Some(csv);
            summary.report_json = Some(json);
        }
        Err(e) => summary.error = Some(format!("报告写出失败：{e:#}")),
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults_and_validation() {
        let r: BatchRequest =
            serde_json::from_str(r#"{"inputs":["in"],"output_dir":"out","format":"png"}"#).unwrap();
        assert_eq!(r.format, OutputFormat::Png);
        assert_eq!(r.jpeg_quality, photo::DEFAULT_JPEG_QUALITY);
        let cfg = r.to_config().unwrap();
        assert!(!cfg.recursive && !cfg.overwrite && cfg.landmarks_dir.is_none());

        let with_landmarks = BatchRequest {
            write_landmarks: true,
            ..r.clone()
        };
        assert_eq!(
            with_landmarks.to_config().unwrap().landmarks_dir,
            Some(PathBuf::from("out").join("landmarks"))
        );
        let no_output = BatchRequest {
            output_dir: PathBuf::new(),
            ..r.clone()
        };
        assert!(no_output.to_config().is_err());
        let no_input = BatchRequest {
            inputs: vec![],
            ..r
        };
        assert!(no_input.to_config().is_err());
    }

    #[test]
    fn progress_events_serialize_with_a_kind_tag() {
        let v = serde_json::to_value(Progress::Started { index: 2, total: 5 }).unwrap();
        assert_eq!(v["kind"], "started");
        assert_eq!(v["index"], 2);
    }
}
