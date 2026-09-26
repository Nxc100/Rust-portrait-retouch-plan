//! 图像协议 `photo://`：前端用 `<img src>` 取当前会话里的图像（二进制 JPEG，免去 base64 与 IPC 序列化）。
//!
//! URL：`photo://localhost/<视图>?v=<版本>&max=<长边上限>`，Windows 上为 `http://photo.localhost/...`
//! （前端用 `convertFileSrc(视图, "photo")` 生成）。`v` 只用于绕过缓存；`max` = 0 表示原尺寸。
//! 只提供已经准备好的图像（原图、结果、`prepare_view` 生成过的调试视图），不在这里做耗时计算。

use crate::state::AppState;
use crate::sync::lock;
use crate::views::{encode_for_display, fit_long_side, ViewKind};
use tauri::http::{header, Request, Response, StatusCode};
use tauri::{Manager, Runtime, UriSchemeContext, UriSchemeResponder};

pub const SCHEME: &str = "photo";

/// 显示用的默认长边上限（适应窗口时足够清晰，编码也快）。
pub const DEFAULT_MAX: u32 = 2400;

/// 解析后的请求。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewRequest {
    pub kind: ViewKind,
    /// 长边上限，0 = 原尺寸
    pub max: u32,
}

/// 从路径与查询串解析请求（`/result`、`max=0`）。
pub fn parse_request(path: &str, query: Option<&str>) -> Option<ViewRequest> {
    let kind = ViewKind::parse(path.trim_start_matches('/').trim_end_matches('/'))?;
    let max = query
        .unwrap_or("")
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == "max")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(DEFAULT_MAX);
    Some(ViewRequest { kind, max })
}

/// Tauri 协议入口：在后台线程编码，编码大图时不阻塞界面。
pub fn handle<R: Runtime>(
    ctx: UriSchemeContext<'_, R>,
    request: Request<Vec<u8>>,
    responder: UriSchemeResponder,
) {
    let app = ctx.app_handle().clone();
    std::thread::spawn(move || {
        let uri = request.uri();
        let response = match parse_request(uri.path(), uri.query()) {
            Some(req) => serve(&app.state::<AppState>(), req),
            None => text(StatusCode::BAD_REQUEST, "未知的视图"),
        };
        responder.respond(response);
    });
}

fn serve(state: &AppState, req: ViewRequest) -> Response<Vec<u8>> {
    // 只在锁内取出 Arc，编码在锁外进行
    let image = lock(&state.session).as_ref().and_then(|s| s.view(req.kind));
    let Some(image) = image else {
        return text(StatusCode::NOT_FOUND, "该视图尚未生成");
    };
    match encode_for_display(&fit_long_side(&image, req.max)) {
        Ok(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "image/jpeg")
            .header(header::CACHE_CONTROL, "no-store")
            .body(bytes)
            .unwrap_or_else(|_| text(StatusCode::INTERNAL_SERVER_ERROR, "响应构造失败")),
        Err(e) => text(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("编码失败：{e:#}"),
        ),
    }
}

fn text(status: StatusCode, message: &str) -> Response<Vec<u8>> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(message.as_bytes().to_vec())
        .expect("static response")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_view_and_max() {
        assert_eq!(
            parse_request("/result", Some("v=12&max=0")),
            Some(ViewRequest {
                kind: ViewKind::Result,
                max: 0
            })
        );
        assert_eq!(
            parse_request("/skin_prob", None),
            Some(ViewRequest {
                kind: ViewKind::SkinProb,
                max: DEFAULT_MAX
            })
        );
        assert_eq!(
            parse_request("/original/", Some("max=bad")).unwrap().max,
            DEFAULT_MAX
        );
        assert_eq!(parse_request("/unknown", None), None);
    }
}
