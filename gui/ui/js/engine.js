// "引擎设置"页与顶栏的模型状态徽标。

import { api } from './api.js';
import { $, el, toast, toastError } from './dom.js';

const OPTIONAL_MODELS = [
  { key: 'enable_parsing', label: '人脸解析', hint: 'BiSeNet 人脸解析：精确的脸部皮肤 / 五官遮罩' },
  { key: 'enable_matting', label: '人像抠图', hint: 'MODNet：身体皮肤与背景分离' },
  { key: 'enable_attribute', label: '性别年龄', hint: 'InsightFace genderage（非商用）：按性别选奶油肌 / 美型参数' },
  { key: 'enable_ai_blemish', label: 'AI 瑕疵祛除', hint: 'ABPN 检测 + 修复痘印、斑点' },
  { key: 'enable_skin_seg', label: '语义皮肤分割', hint: '身体区域只在语义皮肤上找瑕疵（排除纹身、首饰）' },
];

const LANDMARK_MODELS = [
  { value: 'lm2d106', label: 'InsightFace 2d106det（开发期，模型非商用）' },
  { value: 'facemesh', label: 'MediaPipe Face Mesh（发布期，Apache-2.0）' },
];

export class EngineTab {
  /**
   * @param {{ status: object, defaults: object, onApplied: () => void }} deps
   *   onApplied：设置生效后（例如重新检测当前照片的人脸）
   */
  constructor({ status, defaults, onApplied }) {
    this.defaults = defaults;
    this.onApplied = onApplied;
    this.status = status;
    this.draft = structuredClone(status.settings);
    this.buildForm();
    this.render();
  }

  async refresh() {
    try {
      this.status = await api.engineStatus();
      this.render();
    } catch (e) {
      toastError('读取引擎状态失败', e);
    }
  }

  buildForm() {
    const models = $('#engine-models-dir');
    $('#engine-pick-dir').addEventListener('click', async () => {
      const dir = await api.pickFolder().catch((e) => toastError('选择目录失败', e));
      if (dir) {
        this.draft.models_dir = dir;
        models.value = dir;
      }
    });
    models.addEventListener('change', () => {
      this.draft.models_dir = models.value.trim();
    });

    const lm = $('#engine-landmarks');
    lm.replaceChildren(...LANDMARK_MODELS.map((m) => el('option', { value: m.value }, m.label)));
    lm.addEventListener('change', () => {
      this.draft.landmarks = lm.value;
    });

    const threads = $('#engine-threads');
    threads.addEventListener('change', () => {
      this.draft.threads = Math.max(1, Math.min(64, Math.round(Number(threads.value) || 1)));
      threads.value = this.draft.threads;
    });

    $('#engine-optional').replaceChildren(
      ...OPTIONAL_MODELS.map((m) =>
        el(
          'label',
          { class: 'switch-row', title: m.hint },
          el(
            'span',
            { class: 'switch' },
            el('input', {
              type: 'checkbox',
              dataset: { key: m.key },
              onchange: (e) => {
                this.draft[m.key] = e.target.checked;
              },
            }),
            el('span', { class: 'switch-track' }),
          ),
          el('span', { class: 'switch-label' }, m.label, el('span', { class: 'hint-dot' }, '?')),
        ),
      ),
    );

    $('#engine-apply').addEventListener('click', () => this.apply());
    $('#engine-load').addEventListener('click', () => this.load());
    $('#engine-reset').addEventListener('click', () => {
      this.draft = structuredClone(this.defaults);
      this.fillForm();
      toast('已恢复默认值，点"应用设置"生效', { kind: 'info' });
    });
    this.fillForm();
  }

  fillForm() {
    const d = this.draft;
    $('#engine-models-dir').value = d.models_dir;
    $('#engine-landmarks').value = d.landmarks;
    $('#engine-threads').value = d.threads;
    for (const input of document.querySelectorAll('#engine-optional input[data-key]')) {
      input.checked = !!d[input.dataset.key];
    }
  }

  async apply() {
    const btn = $('#engine-apply');
    btn.disabled = true;
    try {
      this.status = await api.setEngineSettings(this.draft);
      this.render();
      toast('设置已保存，下次使用时按新设置加载模型', { kind: 'success' });
      this.onApplied();
    } catch (e) {
      toastError('应用设置失败', e);
    } finally {
      btn.disabled = false;
    }
  }

  async load() {
    const btn = $('#engine-load');
    btn.disabled = true;
    btn.textContent = '加载中…';
    try {
      this.status = await api.loadEngine();
      toast(`已加载：${this.status.models.join(' ')}`, { kind: 'success' });
    } catch (e) {
      toastError('加载模型失败', e);
      this.status = await api.engineStatus().catch(() => this.status);
    } finally {
      btn.disabled = false;
      btn.textContent = '立即加载';
      this.render();
    }
  }

  render() {
    const s = this.status;
    const state = s.loaded ? 'ok' : s.error ? 'bad' : 'idle';
    const text = s.loaded ? '模型已加载' : s.error ? '模型加载失败' : '模型未加载';

    // 顶栏徽标
    const badge = $('#engine-badge');
    badge.className = `badge badge-${state}`;
    badge.textContent = text;
    badge.title = s.loaded ? s.models.join(' ') : s.error ?? '首次打开照片时自动加载';

    // 状态卡片
    $('#engine-state').replaceChildren(
      el('span', { class: `dot dot-${state}` }),
      el('b', {}, text),
    );
    $('#engine-tags').replaceChildren(
      ...(s.loaded ? s.models.map((m) => el('span', { class: 'chip chip-ok' }, m)) : [el('span', { class: 'muted small' }, '—')]),
    );
    $('#engine-ort').textContent = s.ort_library ?? '（加载模型后显示）';
    const err = $('#engine-error');
    err.hidden = !s.error;
    err.textContent = s.error ?? '';
  }
}
