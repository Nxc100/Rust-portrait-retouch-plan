//! 同步小工具。

use std::sync::{Mutex, MutexGuard, PoisonError};

/// 加锁；锁若因持有者 panic 而"中毒"，照常取出数据继续使用。
///
/// 这些锁只保护模型会话与可重建的缓存。批处理中某张照片的处理 panic 被隔离后，
/// 不应让后续照片因锁中毒而全部失败。
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisoned_lock_is_recovered() {
        let m = std::sync::Arc::new(Mutex::new(1));
        let m2 = m.clone();
        let _ = std::thread::spawn(move || {
            let _g = m2.lock().unwrap();
            panic!("poison");
        })
        .join();
        assert!(m.is_poisoned());
        *lock(&m) += 1;
        assert_eq!(*lock(&m), 2);
    }
}
