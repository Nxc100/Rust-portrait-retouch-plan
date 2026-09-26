// 图像查看器。
//
// - 模式：split（同一窗格，分割线左边原图、右边结果）、side（两个窗格并排）、single（一个窗格一张图）；
// - 每个窗格两层：底层原图，上层结果 / 调试视图；按住空格隐藏上层 = 看原图；
// - 滚轮以光标为中心缩放，拖动平移，双击在"适应窗口"与 100% 之间切换；并排时两个窗格同步。
// - 换图时先在后台解码新图再替换，处理完成时画面不闪。

import { el } from './dom.js';

const MIN_SCALE = 0.02;
const MAX_SCALE = 16;

class Pane {
  constructor() {
    this.base = el('img', { class: 'layer base', alt: '', draggable: false });
    this.top = el('img', { class: 'layer top', alt: '', draggable: false });
    this.stage = el('div', { class: 'stage' }, this.base, this.top);
    this.labelLeft = el('div', { class: 'pane-label left' });
    this.labelRight = el('div', { class: 'pane-label right' });
    this.node = el('div', { class: 'pane' }, this.stage, this.labelLeft, this.labelRight);
    this.urls = { base: null, top: null };
  }

  /** 换一层的图（null = 隐藏该层）；新图解码完成后才替换。 */
  async setLayer(which, url) {
    if (this.urls[which] === url) return;
    this.urls[which] = url;
    const current = this[which];
    if (!url) {
      current.removeAttribute('src');
      current.hidden = true;
      return;
    }
    const next = el('img', { class: current.className, alt: '', draggable: false });
    next.src = url;
    try {
      await next.decode();
    } catch {
      // 解码失败（如视图已失效）：保留原图层，由调用方提示
      if (this.urls[which] === url) this.urls[which] = null;
      throw new Error('图像加载失败');
    }
    if (this.urls[which] !== url) return; // 期间又换了图
    next.style.clipPath = current.style.clipPath;
    current.replaceWith(next);
    this[which] = next;
  }
}

export class Viewer {
  /**
   * @param {HTMLElement} root
   * @param {{ onZoom?: (scale:number, fit:boolean) => void }} opts
   */
  constructor(root, { onZoom } = {}) {
    this.root = root;
    this.onZoom = onZoom ?? (() => {});
    this.panes = [new Pane(), new Pane()];
    this.handle = el('div', { class: 'split-handle', title: '拖动分割线' }, el('div', { class: 'split-knob' }));
    this.busy = el('div', { class: 'viewer-busy', hidden: true }, el('span', { class: 'spinner' }), el('span', { class: 'busy-text' }));
    this.empty = el('div', { class: 'viewer-empty' });
    this.panesNode = el('div', { class: 'panes' }, this.panes.map((p) => p.node));
    root.classList.add('viewer');
    root.replaceChildren(this.panesNode, this.handle, this.empty, this.busy);

    this.mode = 'single';
    this.split = 0.5;
    this.size = null; // 图像尺寸 {w, h}
    this.view = { fit: true, scale: 1, x: 0, y: 0 };

    this.bindPointer();
    new ResizeObserver(() => this.layout()).observe(root);
  }

  // ---------------------------------------------------------------- 内容

  /** 空白页内容（未打开照片时）。 */
  setEmpty(node) {
    this.empty.replaceChildren(...(node ? [node] : []));
    this.empty.hidden = !node;
    this.panesNode.hidden = !!node;
    this.handle.hidden = !!node || this.mode !== 'split';
  }

  /** 图像尺寸变化（新照片 / 新工作分辨率）：回到适应窗口。 */
  setImageSize(w, h) {
    const changed = !this.size || this.size.w !== w || this.size.h !== h;
    this.size = { w, h };
    for (const p of this.panes) {
      p.stage.style.width = `${w}px`;
      p.stage.style.height = `${h}px`;
    }
    if (changed) this.view.fit = true;
    this.layout();
  }

  /**
   * 显示一个场景。
   * @param {{ mode: 'split'|'side'|'single', layers: {base?:string|null, top?:string|null, left?:string, right?:string}[] }} scene
   */
  async show(scene) {
    this.mode = scene.mode;
    this.root.dataset.mode = scene.mode;
    const count = scene.mode === 'side' ? 2 : 1;
    this.panes.forEach((p, i) => {
      p.node.hidden = i >= count;
    });
    const jobs = [];
    scene.layers.slice(0, count).forEach((layer, i) => {
      const pane = this.panes[i];
      pane.labelLeft.textContent = layer.left ?? '';
      pane.labelRight.textContent = layer.right ?? '';
      pane.labelLeft.hidden = !layer.left;
      pane.labelRight.hidden = !layer.right;
      jobs.push(pane.setLayer('base', layer.base ?? null), pane.setLayer('top', layer.top ?? null));
    });
    const hasTop = !!scene.layers[0]?.top;
    this.handle.hidden = scene.mode !== 'split' || !hasTop || !this.empty.hidden;
    this.layout();
    await Promise.all(jobs);
    this.applyClip();
  }

  setBusy(text) {
    this.busy.hidden = !text;
    this.busy.querySelector('.busy-text').textContent = text ?? '';
  }

  /** 按住看原图：隐藏上层。 */
  peek(on) {
    this.root.classList.toggle('peek', on);
  }

  // ---------------------------------------------------------------- 缩放 / 平移

  fit() {
    this.view.fit = true;
    this.layout();
  }

  /** 100%（以窗格中心为基准）。 */
  actual() {
    const r = this.paneRect();
    this.zoomTo(1, r.width / 2, r.height / 2);
  }

  zoomBy(factor, cx, cy) {
    this.zoomTo(this.view.scale * factor, cx, cy);
  }

  zoomTo(scale, cx, cy) {
    const s = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
    const v = this.view;
    v.x = cx - (cx - v.x) * (s / v.scale);
    v.y = cy - (cy - v.y) * (s / v.scale);
    v.scale = s;
    v.fit = false;
    this.apply();
  }

  paneRect() {
    return this.panes[0].node.getBoundingClientRect();
  }

  layout() {
    if (!this.size) return;
    if (this.view.fit) {
      const r = this.paneRect();
      if (!r.width || !r.height) return;
      const s = Math.min(r.width / this.size.w, r.height / this.size.h) * 0.98;
      this.view.scale = s;
      this.view.x = (r.width - this.size.w * s) / 2;
      this.view.y = (r.height - this.size.h * s) / 2;
    }
    this.apply();
  }

  apply() {
    const { x, y, scale } = this.view;
    for (const p of this.panes) {
      p.stage.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
      p.stage.classList.toggle('pixelated', scale >= 2);
    }
    this.applyClip();
    this.onZoom(scale, this.view.fit);
  }

  // ---------------------------------------------------------------- 分割线

  applyClip() {
    const top = this.panes[0].top;
    top.style.clipPath = this.mode === 'split' ? `inset(0 0 0 ${this.split * 100}%)` : '';
    if (this.mode !== 'split' || !this.size) return;
    const r = this.paneRect();
    const root = this.root.getBoundingClientRect();
    const xImg = this.view.x + this.split * this.size.w * this.view.scale;
    const x = Math.min(Math.max(xImg, 0), r.width);
    this.handle.style.left = `${r.left - root.left + x}px`;
  }

  setSplitFromClient(clientX) {
    if (!this.size) return;
    const r = this.paneRect();
    const xImg = (clientX - r.left - this.view.x) / (this.size.w * this.view.scale);
    this.split = Math.min(1, Math.max(0, xImg));
    this.applyClip();
  }

  // ---------------------------------------------------------------- 指针

  bindPointer() {
    // 分割线拖动
    this.handle.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.handle.setPointerCapture(e.pointerId);
      const move = (ev) => this.setSplitFromClient(ev.clientX);
      const up = () => {
        this.handle.removeEventListener('pointermove', move);
        this.handle.removeEventListener('pointerup', up);
        this.handle.removeEventListener('pointercancel', up);
      };
      this.handle.addEventListener('pointermove', move);
      this.handle.addEventListener('pointerup', up);
      this.handle.addEventListener('pointercancel', up);
    });

    // 平移
    this.panesNode.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || !this.size) return;
      e.preventDefault();
      this.panesNode.setPointerCapture(e.pointerId);
      this.root.classList.add('panning');
      let lastX = e.clientX;
      let lastY = e.clientY;
      const move = (ev) => {
        this.view.x += ev.clientX - lastX;
        this.view.y += ev.clientY - lastY;
        this.view.fit = false;
        lastX = ev.clientX;
        lastY = ev.clientY;
        this.apply();
      };
      const up = () => {
        this.root.classList.remove('panning');
        this.panesNode.removeEventListener('pointermove', move);
        this.panesNode.removeEventListener('pointerup', up);
        this.panesNode.removeEventListener('pointercancel', up);
      };
      this.panesNode.addEventListener('pointermove', move);
      this.panesNode.addEventListener('pointerup', up);
      this.panesNode.addEventListener('pointercancel', up);
    });

    // 缩放（光标所在窗格的坐标系；并排时两个窗格同尺寸，变换相同）
    this.panesNode.addEventListener(
      'wheel',
      (e) => {
        if (!this.size) return;
        e.preventDefault();
        const pane = e.target.closest('.pane') ?? this.panes[0].node;
        const r = pane.getBoundingClientRect();
        const factor = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
        this.zoomBy(factor, e.clientX - r.left, e.clientY - r.top);
      },
      { passive: false },
    );

    this.panesNode.addEventListener('dblclick', (e) => {
      if (!this.size) return;
      if (this.view.fit || this.view.scale < 0.999 || this.view.scale > 1.001) {
        const pane = e.target.closest('.pane') ?? this.panes[0].node;
        const r = pane.getBoundingClientRect();
        this.zoomTo(1, e.clientX - r.left, e.clientY - r.top);
      } else {
        this.fit();
      }
    });
  }
}
