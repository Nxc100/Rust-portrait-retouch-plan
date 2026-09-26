//! 同步小工具。

use std::sync::{Mutex, MutexGuard, PoisonError};

/// 加锁；锁若因持有者 panic 而中毒，照常取出数据继续使用（状态都是可重建的缓存 / 设置）。
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}
