// 修图参数表单：按 schema 生成，持有当前参数（ParamOptions 的 JSON 形式），变化时回调。

import { api } from './api.js';
import { basename, el, prefs, toast, toastError } from './dom.js';
import { GROUPS, fieldVisible } from './schema.js';
import { showJson } from './modal.js';

/** 改变后需要重建表单的字段（影响其他字段的显示）。 */
const STRUCTURAL = new Set(['preset', 'smooth_mode', 'style_lut']);

/** 可以为 null 的字段及其非 null 时的类型。 */
const NULLABLE = { preset: 'string', embedded_develop: 'boolean', whiten_lut: 'string', warp_coeffs: 'string', style_lut: 'string' };

export class ParamForm {
  /**
   * @param {HTMLElement} root
   * @param {{ defaults: object, presets: {id,name,description}[], onChange: (options: object) => void }} opts
   */
  constructor(root, { defaults, presets, onChange }) {
    this.root = root;
    this.defaults = defaults;
    this.presets = presets;
    this.onChange = onChange;
    this.options = restore(prefs.get('options', null), defaults);
    this.lastPreset = this.options.preset ?? presets[0]?.id ?? null;
    this.collapsed = new Set(prefs.get('collapsedGroups', GROUPS.filter((g) => g.collapsed).map((g) => g.id)));
    this.render();
  }

  /** 当前参数（副本）。 */
  get value() {
    return structuredClone(this.options);
  }

  get isPresetMode() {
    return this.options.preset != null;
  }

  set(key, value) {
    if (JSON.stringify(this.options[key]) === JSON.stringify(value)) return;
    this.options[key] = value;
    if (key === 'preset' && value != null) this.lastPreset = value;
    prefs.set('options', this.options);
    if (STRUCTURAL.has(key)) this.render();
    else this.refreshResetButtons();
    this.onChange(this.value);
  }

  /** 恢复一个字段的默认值。 */
  reset(key) {
    this.set(key, structuredClone(this.defaults[key]));
    this.render();
  }

  setMode(mode) {
    if ((mode === 'preset') === this.isPresetMode) return;
    this.set('preset', mode === 'preset' ? this.lastPreset ?? this.presets[0]?.id ?? 'cream_skin' : null);
  }

  /** 恢复默认（保留当前模式与所选预设）。 */
  resetAll() {
    this.options = { ...structuredClone(this.defaults), preset: this.options.preset };
    prefs.set('options', this.options);
    this.render();
    this.onChange(this.value);
  }

  // ---------------------------------------------------------------- 渲染

  render() {
    const scroller = this.root.closest('.sidebar') ?? this.root;
    const scrollTop = scroller.scrollTop;
    const nodes = [
      this.renderMode(),
      this.isPresetMode ? this.renderPreset() : null,
      ...GROUPS.map((g) => this.renderGroup(g)),
      el(
        'div',
        { class: 'form-footer' },
        el('button', { class: 'btn btn-ghost btn-sm', onclick: () => this.resetAll() }, '恢复默认参数'),
        el('button', { class: 'btn btn-ghost btn-sm', onclick: () => this.copyJson() }, '复制参数 JSON'),
      ),
    ];
    this.root.replaceChildren(...nodes.filter(Boolean));
    scroller.scrollTop = scrollTop;
    this.refreshResetButtons();
  }

  renderMode() {
    const seg = (mode, label, title) =>
      el(
        'button',
        {
          class: `seg-btn${(mode === 'preset') === this.isPresetMode ? ' active' : ''}`,
          title,
          onclick: () => this.setMode(mode),
        },
        label,
      );
    return el(
      'div',
      { class: 'mode-switch' },
      el(
        'div',
        { class: 'segmented', role: 'tablist' },
        seg('preset', '预设', '一键模板：参数以预设为准，可用覆盖项微调'),
        seg('manual', '手动参数', '逐项调节磨皮、色调、美型'),
      ),
    );
  }

  renderPreset() {
    const o = this.options;
    const builtin = this.presets.find((p) => p.id === o.preset);
    const select = el(
      'select',
      {
        class: 'input',
        onchange: async (e) => {
          const v = e.target.value;
          if (v === '__file__') {
            e.target.value = o.preset;
            const path = await api.pickFile('json').catch((err) => toastError('选择文件失败', err));
            if (path) this.set('preset', path);
          } else {
            this.set('preset', v);
          }
        },
      },
      this.presets.map((p) => el('option', { value: p.id, selected: p.id === o.preset }, p.name)),
      !builtin ? el('option', { value: o.preset, selected: true }, `文件：${basename(o.preset)}`) : null,
      el('option', { value: '__file__' }, '从 JSON 文件加载…'),
    );
    const sets = el('textarea', {
      class: 'input mono',
      rows: 3,
      spellcheck: false,
      placeholder: '每行一项 key.sub=value，例如\ncream_female.detail_smooth=0.5\nreshape.female.chin_lift=0.6',
      value: o.sets.join('\n'),
      onchange: (e) => this.set('sets', splitLines(e.target.value)),
    });
    return el(
      'section',
      { class: 'group' },
      el('div', { class: 'group-head static' }, el('span', { class: 'group-title' }, '预设')),
      el(
        'div',
        { class: 'group-body' },
        el('div', { class: 'field' }, el('label', { class: 'field-label' }, '模板'), select),
        builtin?.description ? el('p', { class: 'preset-desc' }, builtin.description) : null,
        el(
          'div',
          { class: 'field' },
          el(
            'label',
            { class: 'field-label', title: '覆盖预设中的字段（与命令行 --set 相同），失焦后生效' },
            '覆盖项',
            el('span', { class: 'hint-dot' }, '?'),
          ),
          sets,
        ),
        el(
          'button',
          {
            class: 'btn btn-ghost btn-sm',
            onclick: async () => {
              try {
                showJson(`预设内容：${builtin?.name ?? basename(o.preset)}`, await api.presetJson(o.preset));
              } catch (err) {
                toastError('读取预设失败', err);
              }
            },
          },
          '查看预设内容（可覆盖的字段）',
        ),
      ),
    );
  }

  renderGroup(group) {
    const o = this.options;
    const fields = group.fields.filter((f) => fieldVisible(f, group, o));
    if (!fields.length) return null;
    const collapsed = this.collapsed.has(group.id);
    const toggle = () => {
      if (this.collapsed.has(group.id)) this.collapsed.delete(group.id);
      else this.collapsed.add(group.id);
      prefs.set('collapsedGroups', [...this.collapsed]);
      this.render();
    };
    return el(
      'section',
      { class: `group${collapsed ? ' collapsed' : ''}` },
      el(
        'button',
        { class: 'group-head', 'aria-expanded': String(!collapsed), onclick: toggle },
        el('span', { class: 'group-title' }, group.title),
        el('span', { class: 'chevron' }),
      ),
      collapsed ? null : el('div', { class: 'group-body' }, fields.map((f) => this.renderField(f))),
    );
  }

  renderField(f) {
    switch (f.type) {
      case 'range':
        return this.renderRange(f);
      case 'toggle':
        return this.renderToggle(f);
      case 'select':
        return this.renderSelect(f);
      case 'file':
        return this.renderFile(f);
      case 'lines':
        return this.renderLines(f);
      default:
        throw new Error(`unknown field type ${f.type}`);
    }
  }

  label(f) {
    return el(
      'label',
      { class: 'field-label', title: f.hint ? `${f.hint}\n（双击恢复默认）` : '双击恢复默认', ondblclick: () => this.reset(f.key) },
      f.label,
      f.hint ? el('span', { class: 'hint-dot' }, '?') : null,
    );
  }

  resetButton(f) {
    return el('button', {
      class: 'reset-btn',
      title: `恢复默认（${fmtValue(this.defaults[f.key])}）`,
      'aria-label': '恢复默认',
      dataset: { reset: f.key },
      onclick: () => this.reset(f.key),
    }, '↺');
  }

  renderRange(f) {
    const v = this.options[f.key];
    const decimals = f.step < 0.01 ? 3 : f.step < 0.1 ? 2 : 1;
    const range = el('input', { type: 'range', min: f.min, max: f.max, step: f.step, value: v });
    const num = el('input', { class: 'input num', type: 'number', min: f.min, max: f.max, step: f.step, value: fix(v, decimals) });
    range.addEventListener('input', () => {
      num.value = fix(Number(range.value), decimals);
      this.set(f.key, Number(range.value));
    });
    num.addEventListener('change', () => {
      const x = Number(num.value);
      if (!Number.isFinite(x)) {
        num.value = fix(this.options[f.key], decimals);
        return;
      }
      range.value = x;
      this.set(f.key, x);
    });
    return el('div', { class: 'field field-range' }, this.label(f), el('div', { class: 'range-row' }, range, num, this.resetButton(f)));
  }

  renderToggle(f) {
    const input = el('input', {
      type: 'checkbox',
      checked: !!this.options[f.key],
      onchange: (e) => this.set(f.key, e.target.checked),
    });
    return el(
      'div',
      { class: 'field field-toggle' },
      el('label', { class: 'switch-row', title: f.hint ?? '' },
        el('span', { class: 'switch' }, input, el('span', { class: 'switch-track' })),
        el('span', { class: 'switch-label' }, f.label, f.hint ? el('span', { class: 'hint-dot' }, '?') : null),
      ),
      this.resetButton(f),
    );
  }

  renderSelect(f) {
    const v = this.options[f.key];
    const select = el(
      'select',
      { class: 'input', onchange: (e) => this.set(f.key, f.options[Number(e.target.value)].value) },
      f.options.map((opt, i) => el('option', { value: i, selected: opt.value === v }, opt.label)),
    );
    return el('div', { class: 'field' }, this.label(f), el('div', { class: 'select-row' }, select, this.resetButton(f)));
  }

  renderFile(f) {
    const v = this.options[f.key];
    const pick = async () => {
      try {
        const path = await api.pickFile(f.fileKind);
        if (path) this.set(f.key, path);
      } catch (err) {
        toastError('选择文件失败', err);
      }
    };
    return el(
      'div',
      { class: 'field' },
      this.label(f),
      el(
        'div',
        { class: 'file-row' },
        el('span', { class: `file-path${v ? '' : ' empty'}`, title: v ?? '' }, v ? basename(v) : '未选择'),
        el('button', { class: 'btn btn-sm', onclick: pick }, '选择…'),
        v ? el('button', { class: 'btn btn-ghost btn-sm', title: '清除', onclick: () => this.set(f.key, null) }, '清除') : null,
      ),
    );
  }

  renderLines(f) {
    return el(
      'div',
      { class: 'field' },
      this.label(f),
      el('textarea', {
        class: 'input mono',
        rows: 2,
        spellcheck: false,
        placeholder: f.placeholder ?? '',
        value: this.options[f.key].join('\n'),
        onchange: (e) => this.set(f.key, splitLines(e.target.value)),
      }),
    );
  }

  /** 只在值不等于默认值时显示"恢复默认"按钮。 */
  refreshResetButtons() {
    for (const btn of this.root.querySelectorAll('[data-reset]')) {
      const key = btn.dataset.reset;
      btn.classList.toggle('visible', JSON.stringify(this.options[key]) !== JSON.stringify(this.defaults[key]));
    }
  }

  async copyJson() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(this.options, null, 2));
      toast('参数 JSON 已复制到剪贴板', { kind: 'success' });
    } catch (err) {
      toastError('复制失败', err);
    }
  }
}

/** 合并保存的参数与默认值：只取已知字段且类型相同的值（字段增删后旧的保存值也能用）。 */
function restore(saved, defaults) {
  const out = structuredClone(defaults);
  if (!saved || typeof saved !== 'object') return out;
  for (const [key, d] of Object.entries(defaults)) {
    if (!(key in saved)) continue;
    const v = saved[key];
    const ok = key in NULLABLE
      ? v === null || typeof v === NULLABLE[key]
      : Array.isArray(d)
        ? Array.isArray(v) && v.every((x) => typeof x === 'string')
        : typeof v === typeof d;
    if (ok) out[key] = v;
  }
  return out;
}

const splitLines = (text) => text.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);

const fix = (v, decimals) => Number(v).toFixed(decimals);

function fmtValue(v) {
  if (v === null || v === undefined) return '无';
  if (typeof v === 'boolean') return v ? '开' : '关';
  if (Array.isArray(v)) return v.length ? `${v.length} 项` : '空';
  return String(v);
}
