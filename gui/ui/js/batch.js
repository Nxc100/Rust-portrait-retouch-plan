// "批量处理"页：输入 / 输出 / 选项 → 预览任务 → 执行（逐张进度、可取消）→ 报告。
// 修图参数沿用"单张调试"页的当前参数（同一个 ParamOptions），这里只显示摘要。

import { api, events } from './api.js';
import { $, basename, el, fmtDuration, fmtMs, prefs, toast, toastError } from './dom.js';
import { describeOptions } from './schema.js';

const GENDER_SHORT = { female: '女', male: '男' };

const STATUS_LABEL = {
  pending: '等待',
  running: '处理中',
  // 库在下一张开始时才回收上一张的写盘结果：下一张开始即说明上一张已处理完、结果正在或已经写出
  processed: '已处理',
  done: '完成',
  skipped: '跳过',
  failed: '失败',
  exists: '已存在',
};

export class BatchTab {
  /**
   * @param {{ params: import('./params.js').ParamForm, defaults: object, presets: object[],
   *           openInSingle: (path: string) => void, onRunningChange: (running: boolean) => void }} deps
   */
  constructor({ params, defaults, presets, openInSingle, onRunningChange }) {
    this.params = params;
    this.defaults = defaults;
    this.presets = presets;
    this.openInSingle = openInSingle;
    this.onRunningChange = onRunningChange;

    const saved = prefs.get('batch', {});
    this.request = {
      inputs: Array.isArray(saved.inputs) ? saved.inputs : [],
      output_dir: typeof saved.output_dir === 'string' ? saved.output_dir : '',
      recursive: !!saved.recursive,
      overwrite: !!saved.overwrite,
      format: saved.format === 'png' ? 'png' : 'jpeg',
      jpeg_quality: Number.isInteger(saved.jpeg_quality) ? saved.jpeg_quality : 98,
      write_landmarks: !!saved.write_landmarks,
    };
    this.items = []; // 表格行：PlannedItem + 状态
    this.running = false;
    this.cancelling = false;
    this.startedAt = 0;
    this.summary = null;
    this.ticker = null; // 运行中每秒刷新"已用 / 预计还需"

    this.bindForm();
    this.renderInputs();
    this.renderParams();
    this.renderTable();
    this.renderProgress();
    events.listen(events.BATCH_PROGRESS, (p) => this.onProgress(p));
    events.listen(events.BATCH_FINISHED, (s) => this.onFinished(s));
  }

  // ---------------------------------------------------------------- 外部接口

  onParamsChanged() {
    this.renderParams();
  }

  /** 拖入的文件 / 目录加入输入。 */
  addInputs(paths) {
    const known = new Set(this.request.inputs);
    const added = paths.filter((p) => !known.has(p));
    if (!added.length) return;
    this.request.inputs.push(...added);
    this.save();
    this.renderInputs();
    toast(`已添加 ${added.length} 项输入`, { kind: 'success' });
  }

  /** 界面刷新时后端仍在跑批处理：只能等它结束（进度事件会继续到达）。 */
  adoptRunning() {
    this.startedAt = performance.now();
    this.setRunning(true);
    this.renderProgress();
  }

  // ---------------------------------------------------------------- 表单

  bindForm() {
    $('#batch-add-folder').addEventListener('click', async () => {
      const dir = await api.pickFolder().catch((e) => toastError('选择目录失败', e));
      if (dir) this.addInputs([dir]);
    });
    $('#batch-add-files').addEventListener('click', async () => {
      const files = await api.pickFiles().catch((e) => toastError('选择照片失败', e));
      if (files?.length) this.addInputs(files);
    });
    $('#batch-clear').addEventListener('click', () => {
      this.request.inputs = [];
      this.save();
      this.renderInputs();
    });
    $('#batch-pick-output').addEventListener('click', async () => {
      const dir = await api.pickFolder().catch((e) => toastError('选择目录失败', e));
      if (dir) {
        this.request.output_dir = dir;
        this.save();
        this.renderOutput();
      }
    });
    $('#batch-open-output').addEventListener('click', () => {
      if (this.request.output_dir) api.openPath(this.request.output_dir).catch((e) => toastError('无法打开目录', e));
    });

    const bindCheck = (id, key) => {
      const input = $(id);
      input.checked = this.request[key];
      input.addEventListener('change', () => {
        this.request[key] = input.checked;
        this.save();
      });
    };
    bindCheck('#batch-recursive', 'recursive');
    bindCheck('#batch-overwrite', 'overwrite');
    bindCheck('#batch-landmarks', 'write_landmarks');

    const format = $('#batch-format');
    format.value = this.request.format;
    const quality = $('#batch-quality');
    const qualityValue = $('#batch-quality-value');
    const syncQuality = () => {
      quality.disabled = this.request.format !== 'jpeg';
      qualityValue.textContent = this.request.jpeg_quality;
    };
    format.addEventListener('change', () => {
      this.request.format = format.value;
      this.save();
      syncQuality();
    });
    quality.value = this.request.jpeg_quality;
    quality.addEventListener('input', () => {
      this.request.jpeg_quality = Number(quality.value);
      this.save();
      syncQuality();
    });
    syncQuality();
    this.renderOutput();

    $('#batch-preview').addEventListener('click', () => this.preview());
    $('#batch-start').addEventListener('click', () => this.start());
    $('#batch-cancel').addEventListener('click', () => this.cancel());
    $('#batch-goto-params').addEventListener('click', () => document.querySelector('[data-tab="single"]').click());
  }

  save() {
    prefs.set('batch', this.request);
  }

  renderInputs() {
    const list = $('#batch-inputs');
    if (!this.request.inputs.length) {
      list.replaceChildren(el('li', { class: 'muted small empty-li' }, '添加目录或照片，也可以直接拖进窗口'));
      return;
    }
    list.replaceChildren(
      ...this.request.inputs.map((p, i) =>
        el(
          'li',
          { title: p },
          el('span', { class: 'path' }, p),
          el(
            'button',
            {
              class: 'icon-btn',
              title: '移除',
              'aria-label': '移除',
              onclick: () => {
                this.request.inputs.splice(i, 1);
                this.save();
                this.renderInputs();
              },
            },
            '×',
          ),
        ),
      ),
    );
  }

  renderOutput() {
    const out = $('#batch-output');
    out.textContent = this.request.output_dir || '未选择';
    out.title = this.request.output_dir;
    out.classList.toggle('empty', !this.request.output_dir);
    $('#batch-open-output').disabled = !this.request.output_dir;
  }

  renderParams() {
    $('#batch-params').textContent = describeOptions(this.params.value, this.defaults, this.presets);
  }

  fullRequest() {
    return { ...this.request, options: this.params.value };
  }

  // ---------------------------------------------------------------- 执行

  async preview() {
    try {
      const planned = await api.previewBatch(this.fullRequest());
      this.setItems(planned);
      const exists = planned.filter((p) => p.exists).length;
      toast(
        `共 ${planned.length} 张` + (exists && !this.request.overwrite ? `，其中 ${exists} 张输出已存在，将跳过` : ''),
        { kind: 'info', title: '任务预览' },
      );
    } catch (e) {
      toastError('无法列出任务', e);
    }
  }

  async start() {
    if (this.running) return;
    try {
      const planned = await api.startBatch(this.fullRequest());
      this.setItems(planned);
      this.summary = null;
      this.startedAt = performance.now();
      this.setRunning(true);
      this.renderProgress();
    } catch (e) {
      toastError('无法开始批处理', e);
    }
  }

  async cancel() {
    try {
      if (await api.cancelBatch()) {
        this.cancelling = true;
        $('#batch-cancel').disabled = true;
        this.renderProgress();
      }
    } catch (e) {
      toastError('取消失败', e);
    }
  }

  setRunning(running) {
    this.running = running;
    this.cancelling = false;
    clearInterval(this.ticker);
    this.ticker = running ? setInterval(() => this.renderProgress(), 1000) : null;
    $('#batch-start').disabled = running;
    $('#batch-preview').disabled = running;
    $('#batch-cancel').disabled = !running;
    $('#batch-cancel').hidden = !running;
    this.onRunningChange(running);
  }

  setItems(planned) {
    this.items = planned.map((p) => ({
      ...p,
      status: p.exists && !this.request.overwrite ? 'exists' : 'pending',
      report: null,
    }));
    this.renderTable();
    this.renderProgress();
  }

  onProgress(p) {
    const item = this.items[p.index];
    if (!item) return;
    if (p.kind === 'started') {
      for (const [i, other] of this.items.entries()) {
        if (other.status === 'running' && i !== p.index) {
          other.status = 'processed';
          this.renderRow(i);
        }
      }
      item.status = 'running';
      item.started = true;
    } else {
      item.status = p.report.status;
      item.report = p.report;
    }
    this.renderRow(p.index);
    this.renderProgress();
  }

  onFinished(s) {
    this.setRunning(false);
    this.summary = s;
    this.renderTable(); // 行内按钮随运行状态启用
    this.renderProgress();
    if (s.error) {
      toastError('批处理未能完成', new Error(s.error));
      return;
    }
    const title = s.cancelled ? '批处理已取消' : '批处理完成';
    toast(`完成 ${s.done} · 跳过 ${s.skipped} · 失败 ${s.failed} · 用时 ${fmtDuration(s.total_ms)}`, {
      kind: s.failed ? 'warn' : 'success',
      title,
      timeout: 8000,
      actions: [{ label: '打开输出目录', onClick: () => api.openPath(s.output_dir).catch((e) => toastError('无法打开', e)) }],
    });
  }

  // ---------------------------------------------------------------- 进度与表格

  renderProgress() {
    const total = this.items.length;
    const count = (st) => this.items.filter((i) => i.status === st).length;
    const done = count('done') + count('processed');
    const skipped = count('skipped');
    const failed = count('failed');
    const finished = done + skipped + failed;
    $('#batch-bar').style.width = total ? `${(finished / total) * 100}%` : '0';
    $('#batch-bar').classList.toggle('has-failed', failed > 0);

    let text;
    if (!total) {
      text = this.running
        ? '批处理进行中（在界面刷新前开始，逐张进度不再显示，结束时给出汇总）…'
        : '还没有任务：设置输入与输出目录后点"预览任务"或"开始"';
    } else if (this.running) {
      const elapsed = performance.now() - this.startedAt;
      // 只按本次真正处理过的照片估算速度（已存在而跳过的不算）
      const worked = this.items.filter((i) => i.started && i.status !== 'running').length;
      const remaining = this.items.filter((i) => i.status === 'pending' || i.status === 'running').length;
      const eta = worked ? (elapsed / worked) * remaining : null;
      text = this.cancelling
        ? `${finished} / ${total} · 正在取消：当前这张完成后停止…`
        : `${finished} / ${total} · 已用 ${fmtDuration(elapsed)}` + (eta && remaining ? ` · 预计还需 ${fmtDuration(eta)}` : '');
    } else if (this.summary) {
      const s = this.summary;
      text = s.error
        ? `未能完成：${s.error}`
        : `${s.cancelled ? '已取消' : '已完成'} · 用时 ${fmtDuration(s.total_ms)}`;
    } else text = `共 ${total} 张，等待开始`;
    $('#batch-progress-text').textContent = text;
    $('#batch-counts').replaceChildren(
      chip('完成', done, 'ok'),
      chip('跳过', skipped + (this.running ? 0 : count('exists')), 'muted'),
      chip('失败', failed, failed ? 'bad' : 'muted'),
    );

    const report = $('#batch-report');
    const s = this.summary;
    if (s && !s.error && s.report_csv) {
      report.hidden = false;
      report.replaceChildren(
        el('span', { class: 'muted small' }, '报告：'),
        el('button', { class: 'btn btn-ghost btn-sm', onclick: () => api.openPath(s.report_csv).catch((e) => toastError('无法打开', e)) }, 'batch_report.csv'),
        el('button', { class: 'btn btn-ghost btn-sm', onclick: () => api.revealPath(s.report_json).catch((e) => toastError('无法显示', e)) }, 'batch_report.json'),
        el('button', { class: 'btn btn-ghost btn-sm', onclick: () => api.openPath(s.output_dir).catch((e) => toastError('无法打开', e)) }, '打开输出目录'),
      );
    } else {
      report.hidden = true;
    }
  }

  renderTable() {
    const body = $('#batch-rows');
    if (!this.items.length) {
      body.replaceChildren(el('tr', {}, el('td', { colspan: 7, class: 'muted small empty-row' }, '暂无任务')));
      return;
    }
    body.replaceChildren(...this.items.map((_, i) => this.rowNode(i)));
  }

  renderRow(i) {
    const old = document.getElementById(`batch-row-${i}`);
    if (old) old.replaceWith(this.rowNode(i));
  }

  rowNode(i) {
    const it = this.items[i];
    const r = it.report;
    const faces = r?.status === 'done'
      ? r.faces.map((f) => (GENDER_SHORT[f.gender] ?? '?') + (f.age != null ? Math.round(f.age) : '')).join(' ') || '无'
      : '';
    const time = r && r.status !== 'skipped'
      ? `${fmtMs(r.load_ms)} / ${fmtMs(r.process_ms)} / ${fmtMs(r.save_ms)}`
      : '';
    return el(
      'tr',
      { id: `batch-row-${i}`, class: `row-${it.status}` },
      el('td', { class: 'num-col' }, i + 1),
      el('td', { class: 'path-col', title: `${it.input}\n→ ${it.output}` }, basename(it.input)),
      el('td', {}, el('span', {
        class: `status status-${it.status}`,
        title: it.status === 'processed' ? '已处理完，结果正在或已经写出（下一张处理完时确认）' : '',
      }, STATUS_LABEL[it.status] ?? it.status)),
      el('td', { class: 'small' }, faces),
      el('td', { class: 'small nums', title: '读取 / 处理 / 写出' }, time),
      el('td', { class: 'small msg-col', title: r?.message ?? '' }, r?.message ?? (r?.develop_applied ? '已再应用冲印设置' : '')),
      el(
        'td',
        { class: 'actions-col' },
        el('button', {
          class: 'btn btn-ghost btn-xs',
          title: '用系统程序打开结果',
          disabled: !(it.status === 'done' || it.status === 'exists' || (it.status === 'skipped' && r?.message === '输出已存在')),
          onclick: () => api.openPath(it.output).catch((e) => toastError('无法打开', e)),
        }, '结果'),
        el('button', {
          class: 'btn btn-ghost btn-xs',
          title: '在"单张调试"页打开原片',
          disabled: this.running,
          onclick: () => this.openInSingle(it.input),
        }, '调试'),
      ),
    );
  }
}

function chip(label, n, tone) {
  return el('span', { class: `chip chip-${tone}` }, `${label} ${n}`);
}
