// 小工具：建元素、提示条、格式化。

/** document.querySelector 的简写。 */
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/**
 * 建元素：el('button', { class: 'btn', onclick }, '文字', child)。
 * attrs 里 on* 为事件，class / dataset / style(对象) 特殊处理，值为 false / null 的属性不设置；
 * value 总是按属性（property）设置——<textarea> 没有 value 特性（attribute）。
 */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs ?? {})) {
    if (v === false || v === null || v === undefined) continue;
    if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k === 'class') node.className = v;
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else if (k === 'value' || (k in node && typeof v !== 'string')) node[k] = v;
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

// ---- 提示条 ----

const toastRoot = () => $('#toasts');

/**
 * 显示提示：kind = info | success | warn | error；actions = [{ label, onClick }]。
 * 错误提示停留更久，鼠标悬停时不消失。
 */
export function toast(message, { kind = 'info', title, actions = [], timeout } = {}) {
  const ms = timeout ?? (kind === 'error' ? 9000 : kind === 'warn' ? 6000 : 3500);
  const node = el(
    'div',
    { class: `toast toast-${kind}`, role: kind === 'error' ? 'alert' : 'status' },
    el(
      'div',
      { class: 'toast-body' },
      title ? el('div', { class: 'toast-title' }, title) : null,
      el('div', { class: 'toast-msg' }, message),
    ),
    actions.length
      ? el(
          'div',
          { class: 'toast-actions' },
          actions.map((a) =>
            el('button', { class: 'btn btn-ghost btn-sm', onclick: () => { a.onClick(); close(); } }, a.label),
          ),
        )
      : null,
    el('button', { class: 'toast-close', title: '关闭', 'aria-label': '关闭', onclick: () => close() }, '×'),
  );
  let timer = null;
  const arm = () => { timer = setTimeout(close, ms); };
  function close() {
    clearTimeout(timer);
    node.classList.add('leaving');
    setTimeout(() => node.remove(), 180);
  }
  node.addEventListener('mouseenter', () => clearTimeout(timer));
  node.addEventListener('mouseleave', arm);
  toastRoot().append(node);
  arm();
  return close;
}

export const toastError = (title, err) => toast(err?.message ?? String(err), { kind: 'error', title });

// ---- 格式化 ----

export function fmtMs(ms) {
  if (ms == null) return '–';
  return ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} s` : `${Math.round(ms)} ms`;
}

export function fmtDuration(ms) {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s} 秒`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m} 分 ${s % 60} 秒` : `${Math.floor(m / 60)} 小时 ${m % 60} 分`;
}

/** 路径的文件名部分（兼容 / 与 \）。 */
export const basename = (p) => String(p ?? '').split(/[\\/]/).filter(Boolean).pop() ?? '';

/** 路径的目录部分。 */
export function dirname(p) {
  const s = String(p ?? '');
  const i = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));
  return i > 0 ? s.slice(0, i) : '';
}

/** 去掉扩展名的文件名。 */
export function stem(p) {
  const b = basename(p);
  const i = b.lastIndexOf('.');
  return i > 0 ? b.slice(0, i) : b;
}

export const genderLabel = (g) => (g === 'female' ? '女' : g === 'male' ? '男' : '性别未知');

/** 延迟执行（连续调用只执行最后一次）。 */
export function debounce(fn, ms) {
  let t = null;
  const wrapped = (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
  wrapped.cancel = () => clearTimeout(t);
  return wrapped;
}

// ---- 本地偏好（只是方便，读写失败时照常工作）----

export const prefs = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(`prs.${key}`);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(`prs.${key}`, JSON.stringify(value));
    } catch {
      /* 存不了就算了 */
    }
  },
};
