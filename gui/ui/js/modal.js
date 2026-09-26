// 模态对话框（<dialog>）：查看 JSON 等只读内容。

import { el, toast } from './dom.js';

/** 显示一段 JSON / 文本，带"复制"按钮。 */
export function showJson(title, text) {
  const dialog = el(
    'dialog',
    { class: 'modal' },
    el(
      'div',
      { class: 'modal-head' },
      el('h2', {}, title),
      el('button', { class: 'toast-close', title: '关闭', 'aria-label': '关闭', onclick: () => dialog.close() }, '×'),
    ),
    el('pre', { class: 'modal-pre mono' }, text),
    el(
      'div',
      { class: 'modal-foot' },
      el(
        'button',
        {
          class: 'btn btn-sm',
          onclick: async () => {
            try {
              await navigator.clipboard.writeText(text);
              toast('已复制', { kind: 'success' });
            } catch (e) {
              toast(String(e), { kind: 'error', title: '复制失败' });
            }
          },
        },
        '复制',
      ),
      el('button', { class: 'btn btn-primary btn-sm', onclick: () => dialog.close() }, '关闭'),
    ),
  );
  dialog.addEventListener('close', () => dialog.remove());
  // 点背景关闭
  dialog.addEventListener('click', (e) => {
    if (e.target === dialog) dialog.close();
  });
  document.body.append(dialog);
  dialog.showModal();
}
