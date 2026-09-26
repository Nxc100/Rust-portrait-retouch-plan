//! 修图引擎的生命周期：设置（模型目录 / 关键点模型 / 可选模型开关）、按需加载、状态查询。
//!
//! 引擎首次使用时才加载（ONNX 会话约 1 s）；修改设置后丢弃旧引擎，下次使用时按新设置重新加载。
//! ONNX Runtime 动态库每个进程只能初始化一次，按库的默认规则查找（`runtime/onnxruntime.dll` 等），
//! 界面只显示实际加载的路径。

use crate::sync::lock;
use portrait_retouch::{Engine, EngineConfig, LandmarkKind};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// 引擎设置（界面"引擎设置"页）。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineSettings {
    /// 模型目录
    pub models_dir: PathBuf,
    /// 关键点模型：lm2d106（开发期，非商用）| facemesh（发布期）
    pub landmarks: LandmarkKind,
    /// ORT 线程数
    pub threads: usize,
    pub enable_parsing: bool,
    pub enable_matting: bool,
    pub enable_attribute: bool,
    pub enable_ai_blemish: bool,
    pub enable_skin_seg: bool,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            landmarks: LandmarkKind::Lm2d106,
            threads: 4,
            enable_parsing: true,
            enable_matting: true,
            enable_attribute: true,
            enable_ai_blemish: true,
            enable_skin_seg: true,
        }
    }
}

impl EngineSettings {
    fn to_config(&self) -> EngineConfig {
        EngineConfig {
            models_dir: self.models_dir.clone(),
            landmark: self.landmarks,
            threads: self.threads.max(1),
            enable_parsing: self.enable_parsing,
            enable_matting: self.enable_matting,
            enable_attribute: self.enable_attribute,
            enable_ai_blemish: self.enable_ai_blemish,
            enable_skin_seg: self.enable_skin_seg,
            ..Default::default()
        }
    }
}

/// 模型目录的候选位置：当前目录下的 `models/`、可执行文件旁的 `models/`，以及可执行文件往上三级
/// （`target/<profile>/` 或 `target/<profile>/deps/` → 项目根）。
pub fn models_dir_candidates(cwd: Option<&Path>, exe: Option<&Path>) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(d) = cwd {
        v.push(d.join("models"));
    }
    if let Some(dir) = exe.and_then(Path::parent) {
        v.extend(dir.ancestors().take(4).map(|a| a.join("models")));
    }
    v
}

/// 第一个存在的候选目录；都不存在时退回相对路径 `models`。
pub fn default_models_dir() -> PathBuf {
    let cwd = std::env::current_dir().ok();
    let exe = std::env::current_exe().ok();
    models_dir_candidates(cwd.as_deref(), exe.as_deref())
        .into_iter()
        .find(|p| p.is_dir())
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// 引擎状态（界面顶栏徽标与设置页）。
#[derive(Clone, Debug, Serialize)]
pub struct EngineStatus {
    pub loaded: bool,
    /// 最近一次加载失败的原因
    pub error: Option<String>,
    pub settings: EngineSettings,
    /// 已加载模型的标签（`Engine::model_tags`）
    pub models: Vec<String>,
    /// 实际加载的 ONNX Runtime 动态库
    pub ort_library: Option<PathBuf>,
}

/// 引擎宿主：持有设置与（按需加载的）引擎。
///
/// 加载（约 1 s）只持有 `loading` 锁：并发的 `get` 排队等同一次加载，而状态查询只短暂持有 `engine` 锁，
/// 不会被加载卡住（状态查询在界面线程上执行）。
pub struct EngineHost {
    settings: Mutex<EngineSettings>,
    engine: Mutex<Option<Arc<Engine>>>,
    last_error: Mutex<Option<String>>,
    loading: Mutex<()>,
}

impl EngineHost {
    pub fn new(settings: EngineSettings) -> Self {
        Self {
            settings: Mutex::new(settings),
            engine: Mutex::new(None),
            last_error: Mutex::new(None),
            loading: Mutex::new(()),
        }
    }

    pub fn settings(&self) -> EngineSettings {
        lock(&self.settings).clone()
    }

    /// 取引擎；首次调用时按当前设置加载模型。
    pub fn get(&self) -> anyhow::Result<Arc<Engine>> {
        if let Some(e) = self.loaded() {
            return Ok(e);
        }
        let _loading = lock(&self.loading);
        // 排队期间可能已由另一个调用加载好
        if let Some(e) = self.loaded() {
            return Ok(e);
        }
        let settings = self.settings();
        match Engine::new(settings.to_config()) {
            Ok(e) => {
                let e = Arc::new(e);
                *lock(&self.engine) = Some(e.clone());
                *lock(&self.last_error) = None;
                Ok(e)
            }
            Err(err) => {
                *lock(&self.last_error) = Some(format!("{err:#}"));
                Err(err.context(format!(
                    "加载模型失败（模型目录 {}）",
                    settings.models_dir.display()
                )))
            }
        }
    }

    /// 已加载的引擎（不触发加载）。
    pub fn loaded(&self) -> Option<Arc<Engine>> {
        lock(&self.engine).clone()
    }

    /// 更新设置：丢弃当前引擎，下次使用时重新加载（正在加载时等加载结束，避免装回按旧设置加载的引擎）。
    pub fn update(&self, settings: EngineSettings) {
        let _loading = lock(&self.loading);
        *lock(&self.settings) = settings;
        *lock(&self.engine) = None;
        *lock(&self.last_error) = None;
    }

    pub fn status(&self) -> EngineStatus {
        let engine = self.loaded();
        EngineStatus {
            loaded: engine.is_some(),
            error: lock(&self.last_error).clone(),
            settings: self.settings(),
            models: engine.as_ref().map(|e| e.model_tags()).unwrap_or_default(),
            ort_library: engine.as_ref().and_then(|e| e.ort_library_path().cloned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_cover_cwd_exe_dir_and_project_root() {
        let exe = Path::new("/proj/target/release/app.exe");
        let v = models_dir_candidates(Some(Path::new("/work")), Some(exe));
        assert_eq!(v[0], Path::new("/work/models"));
        assert!(v.contains(&PathBuf::from("/proj/target/release/models")));
        // target/release → 项目根
        assert!(v.contains(&PathBuf::from("/proj/models")));
    }

    #[test]
    fn settings_update_drops_engine_and_keeps_values() {
        let host = EngineHost::new(EngineSettings {
            models_dir: PathBuf::from("no/such/dir"),
            ..Default::default()
        });
        assert!(!host.status().loaded);
        let s = EngineSettings {
            threads: 2,
            enable_matting: false,
            ..host.settings()
        };
        host.update(s.clone());
        assert_eq!(host.settings(), s);
        assert!(host.loaded().is_none());
    }

    #[test]
    fn settings_json_uses_lowercase_landmark_names() {
        let s: EngineSettings = serde_json::from_str(r#"{"landmarks":"facemesh"}"#).unwrap();
        assert_eq!(s.landmarks, LandmarkKind::FaceMesh);
        assert!(s.enable_parsing, "unspecified fields fall back to defaults");
    }
}
