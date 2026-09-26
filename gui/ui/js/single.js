// "单张调试"页：打开照片、调参（自动处理）、对比与调试视图、保存。

import { api, viewUrl } from './api.js';
import { $, basename, debounce, dirname, el, fmtMs, genderLabel, prefs, stem, toast, toastError } from './dom.js';
import { Viewer } from './viewer.js';

/** 视图按钮：对比方式与调试视图。 */
const COMPARE_VIEWS = [
  { id: 'split', label: '对比', title: '分割线左边原图、右边结果（拖动分割线）' },
  { id: 'side', label: '并排', title: '原图与结果并排，同步缩放' },
  { id: 'original', label: '原图', title: '只看原图' },
  { id: 'result', label: '结果', title: '只看结果（按住空格看原图）' },
];

const DEBUG_VIEWS = [
  { id: 'landmarks', label: '关键点', title: '人脸框、轮廓、五官关键点' },
  { id: 'skin', label: '皮肤遮罩', title: '奶油肌使用的皮肤遮罩：粉 = 脸部，橙 = 身体' },
  { id: 'matte', label: '人像抠图', title: 'MODNet 人像 alpha' },
  { id: 'skin_prob', label: '皮肤概率', title: '语义皮肤分割概率' },
  { id: 'ai_holes', label: 'AI 修复区', title: 'AI 瑕疵模型检测并修复的区域（青色）' },
];

const WORKING_SIZES = [
  { value: 1024, label: '1024 px（最快）' },
  { value: 1600, label: '1600 px（推荐）' },
  { value: 2400, label: '2400 px' },
  { value: 0, label: '原图分辨率（最慢）' },
];

const AUTO_DELAY_MS = 350;

export class SingleTab {
  /**
   * @param {{ params: import('./params.js').ParamForm, onEngineUsed: () => void }} deps
   */
  constructor({ params, onEngineUsed }) {
    this.params = params;
    this.onEngineUsed = onEngineUsed;
    this.photo = null; // PhotoInfo
    this.photoVersion = 0; // 原图 / 调试视图的 URL 版本
    this.resultVersion = 0; // 结果的 URL 版本（0 = 没有结果）
    const savedView = prefs.get('view', 'split');
    this.view = [...COMPARE_VIEWS, ...DEBUG_VIEWS].some((v) => v.id === savedView) ? savedView : 'split';
    this.workingSize = prefs.get('workingSize', 1600);
    this.auto = prefs.get('autoProcess', true);
    this.processing = false;
    this.pending = false;
    this.lastNote = null;
    this.prepared = new Set(); // 已生成的调试视图（"视图@版本"）
    this.processText = null; // 状态栏：最近一次处理
    this.viewText = null; // 状态栏：当前调试视图的生成耗时

    this.viewer = new Viewer($('#viewer'), { onZoom: (s, fit) => this.showZoom(s, fit) });
    this.scheduleProcess = debounce(() => this.process(), AUTO_DELAY_MS);
    this.buildToolbar();
    this.renderInfo();
    this.showEmpty();
    this.bindKeys();
  }

  // ---------------------------------------------------------------- 外部接口

  /** 参数变化：自动处理时稍等片刻（拖动滑块停下来、连续改动合并）再处理。 */
  onParamsChanged() {
    if (this.photo && this.auto) this.scheduleProcess();
  }

  /** 界面刷新后恢复当前照片（后端仍持有照片与结果）。 */
  async restore() {
    try {
      const info = await api.currentPhoto();
      if (!info) return;
      this.resultVersion = info.has_result ? info.version : 0;
      await this.adopt(info, { process: false });
    } catch (e) {
      toastError('恢复照片失败', e);
    }
  }

  // ---------------------------------------------------------------- 工具栏

  buildToolbar() {
    const seg = (items, cls) =>
      el(
        'div',
        { class: `segmented ${cls}` },
        items.map((v) =>
          el('button', { class: 'seg-btn', title: v.title, dataset: { view: v.id }, onclick: () => this.setView(v.id) }, v.label),
        ),
      );
    $('#view-compare').replaceWith(seg(COMPARE_VIEWS, 'view-compare'));
    $('#view-debug').replaceWith(seg(DEBUG_VIEWS, 'view-debug'));
    this.markView();

    $('#btn-open').addEventListener('click', () => this.pickAndOpen());
    $('#btn-save').addEventListener('click', () => this.save());
    $('#btn-process').addEventListener('click', () => this.process());
    $('#btn-fit').addEventListener('click', () => this.viewer.fit());
    $('#btn-100').addEventListener('click', () => this.viewer.actual());

    const auto = $('#chk-auto');
    auto.checked = this.auto;
    auto.addEventListener('change', () => {
      this.auto = auto.checked;
      prefs.set('autoProcess', this.auto);
      if (this.auto && this.photo) this.process();
    });
  }

  markView() {
    for (const b of document.querySelectorAll('[data-view]')) b.classList.toggle('active', b.dataset.view === this.view);
  }

  showZoom(scale, fit) {
    $('#zoom-label').textContent = `${Math.round(scale * 100)}%${fit ? ' · 适应' : ''}`;
  }

  // ---------------------------------------------------------------- 照片

  async pickAndOpen() {
    try {
      const path = await api.pickFile('image');
      if (path) await this.open(path);
    } catch (e) {
      toastError('选择照片失败', e);
    }
  }

  async open(path) {
    this.viewer.setBusy(`正在打开 ${basename(path)}（首次会加载模型）…`);
    try {
      const info = await api.openImage(path, this.workingSize);
      this.resultVersion = 0;
      this.lastNote = null;
      if (info.detect_note) toast(info.detect_note, { kind: 'warn', title: '人脸检测' });
      await this.adopt(info, { process: this.auto });
    } catch (e) {
      toastError('打开照片失败', e);
    } finally {
      this.viewer.setBusy(null);
      this.onEngineUsed();
    }
  }

  /** 采用后端返回的照片信息并刷新显示。 */
  async adopt(info, { process }) {
    this.photo = info;
    this.photoVersion = info.version;
    if (!info.has_result) this.resultVersion = 0;
    this.renderInfo();
    this.processText = null;
    this.viewText = null;
    this.renderStatus();
    this.viewer.setEmpty(null);
    this.viewer.setImageSize(info.working.width, info.working.height);
    if (process) await this.process();
    else await this.refresh();
  }

  async setWorkingSize(value) {
    this.workingSize = value;
    prefs.set('workingSize', value);
    if (!this.photo) return;
    this.viewer.setBusy('切换工作分辨率…');
    try {
      const info = await api.setWorkingSize(value);
      this.resultVersion = 0;
      await this.adopt(info, { process: this.auto });
    } catch (e) {
      toastError('切换分辨率失败', e);
    } finally {
      this.viewer.setBusy(null);
    }
  }

  async redetect() {
    if (!this.photo) return;
    this.viewer.setBusy('重新检测人脸…');
    try {
      const info = await api.redetectFaces();
      this.resultVersion = 0;
      if (info.detect_note && info.faces.length === 0) toast(info.detect_note, { kind: 'warn', title: '人脸检测' });
      else toast(`检测到 ${info.faces.length} 张人脸（${fmtMs(info.detect_ms)}）`, { kind: 'success' });
      await this.adopt(info, { process: this.auto });
    } catch (e) {
      toastError('重新检测失败', e);
    } finally {
      this.viewer.setBusy(null);
      this.onEngineUsed();
    }
  }

  // ---------------------------------------------------------------- 处理

  /** 处理当前照片；处理中再次触发时，结束后用最新参数再处理一次。 */
  async process() {
    this.scheduleProcess.cancel();
    if (!this.photo) return;
    if (this.processing) {
      this.pending = true;
      return;
    }
    this.processing = true;
    this.viewer.setBusy('处理中…');
    $('#btn-process').disabled = true;
    try {
      const info = await api.processPhoto(this.params.value);
      this.resultVersion = info.version;
      this.photo.has_result = true;
      this.processText = `处理 ${fmtMs(info.ms)}` + (info.develop_applied ? ' · 已再应用内嵌冲印设置' : '');
      this.renderStatus();
      if (info.note && info.note !== this.lastNote) toast(info.note, { kind: 'warn' });
      this.lastNote = info.note;
      await this.refresh();
    } catch (e) {
      toastError('处理失败', e);
    } finally {
      this.processing = false;
      $('#btn-process').disabled = false;
      this.viewer.setBusy(null);
      if (this.pending) {
        this.pending = false;
        this.process();
      }
    }
  }

  // ---------------------------------------------------------------- 视图

  async setView(id) {
    this.view = id;
    prefs.set('view', id);
    this.markView();
    await this.refresh();
  }

  /** 按当前视图重新组织画面。 */
  async refresh() {
    this.markView();
    if (!this.photo) return;
    if (!DEBUG_VIEWS.some((v) => v.id === this.view)) this.viewText = null;
    const original = viewUrl('original', this.photoVersion);
    const result = this.resultVersion ? viewUrl('result', this.resultVersion) : null;
    const noResult = '（尚未处理）';
    let scene;
    switch (this.view) {
      case 'split':
        scene = { mode: 'split', layers: [{ base: original, top: result, left: '原图', right: result ? '结果' : noResult }] };
        break;
      case 'side':
        scene = {
          mode: 'side',
          layers: [
            { base: original, left: '原图' },
            { base: original, top: result, left: result ? '结果' : `结果${noResult}` },
          ],
        };
        break;
      case 'original':
        scene = { mode: 'single', layers: [{ base: original, left: '原图' }] };
        break;
      case 'result':
        scene = { mode: 'single', layers: [{ base: original, top: result, left: result ? '结果' : `结果${noResult}` }] };
        break;
      default: {
        const def = DEBUG_VIEWS.find((v) => v.id === this.view);
        const url = await this.prepareDebugView(this.view, def.label);
        scene = { mode: 'single', layers: [{ base: original, top: url, left: def.label }] };
      }
    }
    try {
      await this.viewer.show(scene);
    } catch (e) {
      toastError('显示图像失败', e);
    }
  }

  async prepareDebugView(kind, label) {
    const key = `${kind}@${this.photoVersion}`;
    if (this.prepared.has(key)) return viewUrl(kind, this.photoVersion);
    this.viewer.setBusy(`生成${label}…`);
    try {
      const info = await api.prepareView(kind);
      this.viewText = info.ms > 0 ? `${label}：生成 ${fmtMs(info.ms)}` : null;
      this.renderStatus();
      this.prepared.add(key);
      return viewUrl(kind, this.photoVersion);
    } catch (e) {
      toastError(`无法生成${label}`, e);
      return null;
    } finally {
      this.viewer.setBusy(null);
      this.onEngineUsed();
    }
  }

  // ---------------------------------------------------------------- 保存

  async save() {
    if (!this.photo) return toast('请先打开一张照片', { kind: 'warn' });
    if (!this.resultVersion) return toast('还没有处理结果', { kind: 'warn' });
    let path;
    try {
      path = await api.pickSavePath(`${stem(this.photo.path)}_retouched.jpg`, dirname(this.photo.path) || null);
    } catch (e) {
      return toastError('选择保存位置失败', e);
    }
    if (!path) return;
    const full = this.photo.working.scale >= 1;
    this.viewer.setBusy(full ? '保存中…' : '按原图分辨率重新处理并保存…');
    try {
      const info = await api.saveResult(path);
      toast(`${basename(info.path)} · ${info.width}×${info.height} · ${fmtMs(info.ms)}`, {
        kind: 'success',
        title: '已保存',
        actions: [{ label: '在文件夹中显示', onClick: () => api.revealPath(info.path).catch((e) => toastError('无法显示', e)) }],
      });
    } catch (e) {
      toastError('保存失败', e);
    } finally {
      this.viewer.setBusy(null);
    }
  }

  // ---------------------------------------------------------------- 信息面板

  showEmpty() {
    this.viewer.setEmpty(
      el(
        'div',
        { class: 'empty-card' },
        el('div', { class: 'empty-icon' }, '🖼'),
        el('div', { class: 'empty-title' }, '打开一张人像照片开始调试'),
        el('div', { class: 'empty-sub' }, '点击"打开照片"，或把照片拖进窗口（Ctrl+O）'),
        el('button', { class: 'btn btn-primary', onclick: () => this.pickAndOpen() }, '打开照片…'),
      ),
    );
  }

  renderInfo() {
    const p = this.photo;
    const sizeSelect = el(
      'select',
      { class: 'input', onchange: (e) => this.setWorkingSize(Number(e.target.value)) },
      WORKING_SIZES.map((s) => el('option', { value: s.value, selected: s.value === this.workingSize }, s.label)),
    );
    const rows = [];
    if (p) {
      rows.push(
        el('div', { class: 'photo-name', title: p.path }, p.file_name),
        el(
          'div',
          { class: 'photo-meta' },
          `${p.width}×${p.height}`,
          p.orientation !== 1 ? ` · EXIF 方向 ${p.orientation}（已转正）` : '',
          p.has_icc ? ' · 含 ICC' : '',
        ),
      );
    } else {
      rows.push(el('div', { class: 'photo-meta muted' }, '还没有打开照片'));
    }
    rows.push(
      el('div', { class: 'field' }, el('label', { class: 'field-label', title: '在缩小图上处理，调参更快；保存时按原图分辨率重做' }, '工作分辨率', el('span', { class: 'hint-dot' }, '?')), sizeSelect),
    );
    if (p) {
      rows.push(
        el(
          'div',
          { class: 'faces-head' },
          el('span', {}, `人脸 ${p.faces.length} 张`, p.detect_ms ? el('span', { class: 'muted' }, ` · 检测 ${fmtMs(p.detect_ms)}`) : null),
          el('button', { class: 'btn btn-ghost btn-sm', title: '改了模型设置后重新检测', onclick: () => this.redetect() }, '重新检测'),
        ),
        p.faces.length
          ? el(
              'ol',
              { class: 'faces' },
              p.faces.map((f) =>
                el(
                  'li',
                  {},
                  `${genderLabel(f.gender)}${f.age != null ? ` · ${Math.round(f.age)} 岁` : ''}`,
                  el('span', { class: 'muted' }, ` · 瞳距 ${Math.round(f.eye_distance)} px · 侧脸 ${Math.round(f.yaw_deg)}°`),
                ),
              ),
            )
          : el('div', { class: 'muted small' }, p.detect_note ?? '没有检测到人脸'),
        p.develop_settings
          ? el('div', { class: 'develop small', title: p.develop_settings }, el('b', {}, '内嵌冲印设置：'), p.develop_settings)
          : null,
      );
    }
    $('#photo-card').replaceChildren(...rows.filter(Boolean));
    $('#btn-save').disabled = !p;
    $('#btn-process').disabled = !p;
  }

  renderStatus() {
    const w = this.photo?.working;
    const size = w ? `工作图 ${w.width}×${w.height}${w.scale < 1 ? `（原图 ${Math.round(w.scale * 100)}%）` : ''}` : null;
    $('#status-text').textContent = [size, this.processText, this.viewText].filter(Boolean).join(' · ');
  }

  // ---------------------------------------------------------------- 键盘

  bindKeys() {
    const typing = (e) => e.target.closest('input, textarea, select, [contenteditable]');
    document.addEventListener('keydown', (e) => {
      if (!document.body.classList.contains('tab-single')) return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === 'o') {
        e.preventDefault();
        this.pickAndOpen();
      } else if (mod && e.key.toLowerCase() === 's') {
        e.preventDefault();
        this.save();
      } else if (mod && e.key === 'Enter') {
        e.preventDefault();
        this.process();
      } else if (!mod && !typing(e)) {
        if (e.code === 'Space') {
          e.preventDefault();
          this.viewer.peek(true);
        } else if (e.key === 'f' || e.key === 'F') this.viewer.fit();
        else if (e.key === '1') this.viewer.actual();
      }
    });
    document.addEventListener('keyup', (e) => {
      if (e.code === 'Space') this.viewer.peek(false);
    });
    window.addEventListener('blur', () => this.viewer.peek(false));
  }
}
