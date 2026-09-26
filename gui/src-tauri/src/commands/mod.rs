//! 前端可调用的命令（`invoke("<名字>", {...})`）。
//!
//! - [`app`]：应用信息、引擎设置 / 加载、文件对话框、在资源管理器中显示；
//! - [`photo`]：单张照片的打开、处理、调试视图、保存；
//! - [`batch`]：批量处理（进度以事件推送，可取消）。
//!
//! 耗时的命令都是 `async` 并把计算放到阻塞线程池（`spawn_blocking`），界面线程不被占用。

pub mod app;
pub mod batch;
pub mod photo;

use crate::error::{CmdResult, CommandError};

/// 在阻塞线程池里执行，并把 panic / 取消转成命令错误。
pub(crate) async fn blocking<T, F>(f: F) -> CmdResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> CmdResult<T> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(f)
        .await
        .map_err(|e| CommandError::msg(format!("后台任务异常结束：{e}")))?
}
