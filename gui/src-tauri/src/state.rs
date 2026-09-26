//! 应用状态（由 Tauri 管理，各命令与图像协议共享）。

use crate::engine_host::{EngineHost, EngineSettings};
use crate::error::{CmdResult, CommandError};
use crate::session::Session;
use crate::sync::lock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

pub struct AppState {
    pub engine: EngineHost,
    /// 当前打开的照片
    pub session: Mutex<Option<Session>>,
    /// 工作锁：串行化使用引擎或改变会话的操作（打开、处理、调试视图、保存、改引擎设置、批处理）。
    /// 界面连续触发的请求依次执行，处理结果不会与照片 / 工作分辨率错配
    work: Mutex<()>,
    pub batch: Arc<BatchControl>,
}

impl AppState {
    pub fn new(settings: EngineSettings) -> Self {
        Self {
            engine: EngineHost::new(settings),
            session: Mutex::new(None),
            work: Mutex::new(()),
            batch: Arc::default(),
        }
    }

    /// 开始一项单张照片的操作：批处理进行中时直接拒绝（不排队等整批结束），否则取得工作锁。
    pub fn begin_work(&self) -> CmdResult<MutexGuard<'_, ()>> {
        self.ensure_no_batch()?;
        let guard = lock(&self.work);
        // 排队期间可能刚好开始了批处理
        self.ensure_no_batch()?;
        Ok(guard)
    }

    /// 批处理线程整批持有工作锁（等正在进行的单张操作完成后才开始）。
    pub fn lock_work_for_batch(&self) -> MutexGuard<'_, ()> {
        lock(&self.work)
    }

    fn ensure_no_batch(&self) -> CmdResult<()> {
        if self.batch.is_running() {
            Err(CommandError::msg("批处理进行中，请等它完成或取消后再操作"))
        } else {
            Ok(())
        }
    }
}

/// 批处理的运行 / 取消控制（同一时间只允许一个批处理）。
#[derive(Default)]
pub struct BatchControl {
    running: AtomicBool,
    cancel: Mutex<Option<Arc<AtomicBool>>>,
}

/// 一次批处理的凭据：持有期间视为"运行中"；丢弃时（包括批处理线程 panic 展开时）自动结束。
pub struct BatchTicket {
    control: Arc<BatchControl>,
    cancel: Arc<AtomicBool>,
}

impl BatchTicket {
    /// 交给 `BatchConfig::cancel` 的取消标志。
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }
}

impl Drop for BatchTicket {
    fn drop(&mut self) {
        *lock(&self.control.cancel) = None;
        self.control.running.store(false, Ordering::SeqCst);
    }
}

impl BatchControl {
    /// 开始一个批处理；已有批处理在运行时返回 None。
    pub fn try_start(self: &Arc<Self>) -> Option<BatchTicket> {
        if self.running.swap(true, Ordering::SeqCst) {
            return None;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        *lock(&self.cancel) = Some(cancel.clone());
        Some(BatchTicket {
            control: self.clone(),
            cancel,
        })
    }

    /// 请求取消；没有在运行的批处理时返回 false。
    pub fn cancel(&self) -> bool {
        match lock(&self.cancel).as_ref() {
            Some(flag) => {
                flag.store(true, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_batch_at_a_time_and_cancel_reaches_the_flag() {
        let b = Arc::new(BatchControl::default());
        assert!(!b.cancel(), "nothing to cancel yet");
        let ticket = b.try_start().expect("first start");
        let flag = ticket.cancel_flag();
        assert!(b.is_running());
        assert!(b.try_start().is_none(), "second start refused");
        assert!(b.cancel());
        assert!(flag.load(Ordering::SeqCst));
        drop(ticket);
        assert!(!b.is_running());
        assert!(!b.cancel(), "finished batch can no longer be cancelled");
        assert!(b.try_start().is_some(), "can start again after finishing");
    }

    #[test]
    fn single_photo_work_is_refused_while_a_batch_runs() {
        let state = AppState::new(EngineSettings::default());
        assert!(state.begin_work().is_ok());
        let ticket = state.batch.try_start().unwrap();
        let err = state.begin_work().unwrap_err();
        assert!(err.to_string().contains("批处理进行中"));
        drop(ticket);
        assert!(state.begin_work().is_ok());
    }

    #[test]
    fn ticket_is_released_when_the_batch_thread_panics() {
        let b = Arc::new(BatchControl::default());
        let ticket = b.try_start().unwrap();
        let joined = std::thread::spawn(move || {
            let _ticket = ticket;
            panic!("boom");
        })
        .join();
        assert!(joined.is_err());
        assert!(!b.is_running());
    }
}
