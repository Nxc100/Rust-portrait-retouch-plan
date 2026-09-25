//! ONNX Runtime 动态库定位与 Session 创建（`ort` 的 `load-dynamic` 模式）。
//!
//! 查找顺序：显式路径 → `ORT_DYLIB_PATH` 环境变量 → `./runtime/<lib>` → 可执行文件目录及其 `runtime/`
//! → 系统库搜索路径（由 ort 默认处理）。

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[cfg(target_os = "windows")]
pub const DYLIB_NAME: &str = "onnxruntime.dll";
#[cfg(target_os = "macos")]
pub const DYLIB_NAME: &str = "libonnxruntime.dylib";
#[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
pub const DYLIB_NAME: &str = "libonnxruntime.so";

static INIT: OnceLock<Result<Option<PathBuf>, String>> = OnceLock::new();

/// 候选路径列表。
pub fn candidate_paths(explicit: Option<&Path>) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(p) = explicit {
        v.push(p.to_path_buf());
    }
    if let Ok(s) = std::env::var("ORT_DYLIB_PATH") {
        if !s.is_empty() {
            v.push(PathBuf::from(s));
        }
    }
    v.push(PathBuf::from("runtime").join(DYLIB_NAME));
    v.push(PathBuf::from(DYLIB_NAME));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join(DYLIB_NAME));
            v.push(dir.join("runtime").join(DYLIB_NAME));
            // target/{debug,release}/ → 项目根目录
            if let Some(root) = dir.parent().and_then(|p| p.parent()) {
                v.push(root.join("runtime").join(DYLIB_NAME));
            }
        }
    }
    v
}

/// 初始化 ORT（进程内只执行一次）。返回实际加载的库路径（None 表示交由 ort 默认搜索）。
pub fn init_ort(explicit: Option<&Path>) -> anyhow::Result<Option<PathBuf>> {
    let r = INIT.get_or_init(|| {
        let found = candidate_paths(explicit).into_iter().find(|p| p.is_file());
        match &found {
            Some(p) => match ort::init_from(p) {
                Ok(env) => {
                    env.with_name("portrait-retouch").commit();
                    Ok(Some(p.clone()))
                }
                Err(e) => Err(format!(
                    "failed to load ONNX Runtime from {}: {e}",
                    p.display()
                )),
            },
            None => {
                // 交给 ort 默认逻辑（系统路径中的 onnxruntime 动态库）
                ort::init().with_name("portrait-retouch").commit();
                Ok(None)
            }
        }
    });
    match r {
        Ok(p) => Ok(p.clone()),
        Err(e) => anyhow::bail!("{e}"),
    }
}

/// 创建 CPU Session。
pub fn make_session(model: &Path, threads: usize) -> anyhow::Result<ort::session::Session> {
    anyhow::ensure!(model.is_file(), "model file not found: {}", model.display());
    let session = ort::session::Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(ort::Error::<()>::from)?
        .with_intra_threads(threads.max(1))
        .map_err(ort::Error::<()>::from)?
        // 线程池不自旋等待：避免与 rayon 工作线程争抢 CPU
        .with_config_entry("session.intra_op.allow_spinning", "0")
        .map_err(ort::Error::<()>::from)?
        // 非规格化浮点数按 0 处理（FTZ/DAZ）：瑕疵分割网络在个别输入上会产生非规格化激活，
        // 此时同一次推理从 0.4 s 变成 70 s 以上（CPU 满载、并非等待）；按 0 处理对输出无可见影响
        .with_config_entry("session.set_denormal_as_zero", "1")
        .map_err(ort::Error::<()>::from)?
        .commit_from_file(model)
        .map_err(|e| anyhow::anyhow!("failed to load {}: {e}", model.display()))?;
    Ok(session)
}

/// 在当前线程上临时打开 FTZ/DAZ（非规格化浮点数按 0 处理），析构时恢复。
/// ORT 的 `session.set_denormal_as_zero` 只作用于其线程池；调用 `run` 的线程也会分担计算，
/// 需要同样设置，否则个别输入仍会变慢数倍。仅 x86 / x86_64 生效，其余平台为空操作。
pub struct DenormalGuard {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    saved: u32,
}

impl DenormalGuard {
    #[allow(deprecated)]
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
            // SAFETY: 只修改本线程 MXCSR 的 FTZ (bit 15) 与 DAZ (bit 6)，析构时恢复原值。
            let saved = unsafe { _mm_getcsr() };
            unsafe { _mm_setcsr(saved | 0x8040) };
            Self { saved }
        }
        #[cfg(target_arch = "x86")]
        {
            use std::arch::x86::{_mm_getcsr, _mm_setcsr};
            // SAFETY: 同上。
            let saved = unsafe { _mm_getcsr() };
            unsafe { _mm_setcsr(saved | 0x8040) };
            Self { saved }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            Self {}
        }
    }
}

impl Default for DenormalGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DenormalGuard {
    #[allow(deprecated)]
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: 恢复构造时保存的 MXCSR。
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.saved)
        };
        #[cfg(target_arch = "x86")]
        // SAFETY: 同上。
        unsafe {
            std::arch::x86::_mm_setcsr(self.saved)
        };
    }
}
