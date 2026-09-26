// 发布构建不弹控制台窗口（调试构建保留，便于看日志）
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    portrait_retouch_gui::run();
}
