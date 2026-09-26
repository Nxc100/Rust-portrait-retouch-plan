// 入口：取应用信息 → 建各页 → 标签切换、拖放、恢复状态。

import { api, events } from './api.js';
import { $, $$, el, prefs, toast } from './dom.js';
import { BatchTab } from './batch.js';
import { EngineTab } from './engine.js';
import { ParamForm } from './params.js';
import { SingleTab } from './single.js';

const IMAGE_EXT = /\.(jpe?g|png|tiff?|webp|bmp)$/i;
const TABS = ['single', 'batch', 'engine'];

let currentTab = 'single';

function selectTab(id) {
  currentTab = TABS.includes(id) ? id : 'single';
  for (const b of $$('[data-tab]')) {
    const active = b.dataset.tab === currentTab;
    b.classList.toggle('active', active);
    b.setAttribute('aria-selected', String(active));
  }
  for (const t of TABS) {
    $(`#tab-${t}`).hidden = t !== currentTab;
    document.body.classList.toggle(`tab-${t}`, t === currentTab);
  }
  prefs.set('tab', currentTab);
}

function fatal(err) {
  document.body.replaceChildren(
    el(
      'div',
      { class: 'fatal' },
      el('h1', {}, '界面无法启动'),
      el('p', {}, window.__TAURI__ ? String(err?.message ?? err) : '请通过 Tauri 应用打开本页面（cargo run --release -p portrait-retouch-gui）。'),
    ),
  );
}

async function main() {
  let info;
  try {
    info = await api.appInfo();
  } catch (e) {
    fatal(e);
    return;
  }
  $('#app-version').textContent = `v${info.version}`;

  let single = null;
  let batch = null;
  const params = new ParamForm($('#params'), {
    defaults: info.defaults,
    presets: info.presets,
    onChange: () => {
      single?.onParamsChanged();
      batch?.onParamsChanged();
    },
  });
  const engine = new EngineTab({
    status: info.engine,
    defaults: info.engine_defaults,
    onApplied: () => single.redetect(),
  });
  single = new SingleTab({ params, onEngineUsed: () => engine.refresh() });
  batch = new BatchTab({
    params,
    defaults: info.defaults,
    presets: info.presets,
    openInSingle: (path) => {
      selectTab('single');
      single.open(path);
    },
    onRunningChange: (running) => {
      document.body.classList.toggle('batch-running', running);
      $('#batch-indicator').hidden = !running;
      if (!running) engine.refresh();
    },
  });

  for (const b of $$('[data-tab]')) b.addEventListener('click', () => selectTab(b.dataset.tab));
  $('#engine-badge').addEventListener('click', () => selectTab('engine'));
  $('#batch-indicator').addEventListener('click', () => selectTab('batch'));
  selectTab(prefs.get('tab', 'single'));

  // 从资源管理器拖入：单张页打开第一张图片，批处理页加入输入
  events.listen(events.DRAG_ENTER, () => document.body.classList.add('dragging'));
  events.listen(events.DRAG_LEAVE, () => document.body.classList.remove('dragging'));
  events.listen(events.DRAG_DROP, ({ paths }) => {
    document.body.classList.remove('dragging');
    if (!paths?.length) return;
    if (currentTab === 'batch') {
      batch.addInputs(paths);
      return;
    }
    const image = paths.find((p) => IMAGE_EXT.test(p));
    if (!image) {
      toast('请拖入图片文件（jpg / png / tif / webp / bmp）', { kind: 'warn' });
      return;
    }
    selectTab('single');
    single.open(image);
  });

  if (info.batch_running) batch.adoptRunning();
  await single.restore();
}

main();
