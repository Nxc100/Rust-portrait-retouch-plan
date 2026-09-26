//! Portrait Retouch Studio：portrait-retouch 的手动测试界面（Tauri 2）。
//!
//! 界面只是库的一层薄壳，参数组装、读写图、批处理都复用库本身（`ParamOptions`、`photo`、`batch`），
//! 与命令行 `retouch` 的行为一致。
//!
//! - [`commands`]：前端调用的命令（应用 / 单张照片 / 批处理）；
//! - [`state`]：应用状态（引擎宿主、当前照片、工作锁、批处理控制）；
//! - [`session`] / [`views`]：单张照片的工作图像、处理结果与调试视图；
//! - [`protocol`]：`photo://` 图像协议，前端 `<img>` 直接取 JPEG；
//! - [`engine_host`]：引擎设置与按需加载。

mod commands;
mod engine_host;
mod error;
mod protocol;
mod session;
mod state;
mod sync;
mod views;

use engine_host::EngineSettings;
use state::AppState;

/// 启动界面。
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::new(EngineSettings::default()))
        .register_asynchronous_uri_scheme_protocol(protocol::SCHEME, protocol::handle)
        .invoke_handler(tauri::generate_handler![
            commands::app::app_info,
            commands::app::preset_json,
            commands::app::engine_status,
            commands::app::load_engine,
            commands::app::set_engine_settings,
            commands::app::pick_file,
            commands::app::pick_files,
            commands::app::pick_folder,
            commands::app::pick_save_path,
            commands::app::reveal_path,
            commands::app::open_path,
            commands::photo::open_image,
            commands::photo::current_photo,
            commands::photo::set_working_size,
            commands::photo::redetect_faces,
            commands::photo::process_photo,
            commands::photo::prepare_view,
            commands::photo::save_result,
            commands::batch::preview_batch,
            commands::batch::start_batch,
            commands::batch::cancel_batch,
        ])
        .run(tauri::generate_context!())
        .expect("启动界面失败");
}
