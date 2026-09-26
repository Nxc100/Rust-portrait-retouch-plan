// 后端接口：所有命令名、事件名与图像 URL 的拼法都集中在这里。
// 命令失败时 invoke 以一段文字 reject，这里统一转成 Error。

const tauri = window.__TAURI__;

async function call(cmd, args) {
  try {
    return await tauri.core.invoke(cmd, args);
  } catch (e) {
    throw new Error(typeof e === 'string' ? e : e?.message ?? String(e));
  }
}

export const api = {
  // ---- 应用 / 引擎 ----
  appInfo: () => call('app_info'),
  presetJson: (name) => call('preset_json', { name }),
  engineStatus: () => call('engine_status'),
  loadEngine: () => call('load_engine'),
  setEngineSettings: (settings) => call('set_engine_settings', { settings }),

  // ---- 对话框 / 系统 ----
  /** kind: image | lut | lookup | json */
  pickFile: (kind) => call('pick_file', { kind }),
  pickFiles: () => call('pick_files'),
  pickFolder: () => call('pick_folder'),
  pickSavePath: (fileName, directory) => call('pick_save_path', { fileName, directory }),
  revealPath: (path) => call('reveal_path', { path }),
  openPath: (path) => call('open_path', { path }),

  // ---- 单张照片 ----
  openImage: (path, longSide) => call('open_image', { path, longSide }),
  currentPhoto: () => call('current_photo'),
  setWorkingSize: (longSide) => call('set_working_size', { longSide }),
  redetectFaces: () => call('redetect_faces'),
  processPhoto: (options) => call('process_photo', { options }),
  prepareView: (kind) => call('prepare_view', { kind }),
  saveResult: (path) => call('save_result', { path }),

  // ---- 批处理 ----
  previewBatch: (request) => call('preview_batch', { request }),
  startBatch: (request) => call('start_batch', { request }),
  cancelBatch: () => call('cancel_batch'),
};

export const events = {
  BATCH_PROGRESS: 'batch-progress',
  BATCH_FINISHED: 'batch-finished',
  DRAG_DROP: 'tauri://drag-drop',
  DRAG_ENTER: 'tauri://drag-enter',
  DRAG_LEAVE: 'tauri://drag-leave',
  /** 监听后端事件，回调收到 payload；返回取消监听的函数（Promise）。 */
  listen: (name, handler) => tauri.event.listen(name, (e) => handler(e.payload)),
};

/**
 * 会话图像的 URL（`photo://` 协议）。kind: original | result | landmarks | skin | matte | skin_prob | ai_holes。
 * version 只用于绕过缓存；max = 0 表示工作分辨率原尺寸。
 */
export function viewUrl(kind, version, max = 0) {
  return `${tauri.core.convertFileSrc(kind, 'photo')}?v=${version}&max=${max}`;
}
