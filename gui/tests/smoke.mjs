// 界面冒烟测试：通过 Chrome DevTools 协议（CDP）驱动运行中的界面，把主要流程走一遍。
//
// 需要一个开启了 WebView2 远程调试端口的构建（只用于测试，正式构建不带端口），见 gui/README.md「冒烟测试」。
//
//   node gui/tests/smoke.mjs <照片路径> [--port 9222]
//
// 流程：打开照片（模拟拖入）→ 四种磨皮模式 → 奶油肌预设 → 五个调试视图 → 切换工作分辨率 →
//       单张批处理到临时目录（PNG）。测试前后保存 / 恢复界面的本地偏好，不留下痕迹。
// 需要 Node 22+（内置 fetch / WebSocket）。

import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, basename, extname } from 'node:path';

const args = process.argv.slice(2);
const photo = args.find((a) => !a.startsWith('--'));
const port = args.includes('--port') ? args[args.indexOf('--port') + 1] : '9222';
if (!photo || !existsSync(photo)) {
  console.error('用法：node gui/tests/smoke.mjs <照片路径> [--port 9222]');
  process.exit(2);
}
const photoPath = resolve(photo);

// ---------------------------------------------------------------- CDP

const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const page = targets.find((t) => t.type === 'page');
if (!page) throw new Error('没有找到界面页面（界面是否以带调试端口的构建运行？）');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((res, rej) => {
  ws.onopen = res;
  ws.onerror = rej;
});
let nextId = 0;
const pending = new Map();
const pageErrors = [];
ws.onmessage = (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
  } else if (msg.method === 'Runtime.exceptionThrown') {
    const d = msg.params.exceptionDetails;
    pageErrors.push(d.exception?.description ?? d.text);
  }
};
const send = (method, params = {}) =>
  new Promise((res) => {
    const id = ++nextId;
    pending.set(id, res);
    ws.send(JSON.stringify({ id, method, params }));
  });
await send('Runtime.enable');

async function ev(expr) {
  const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
  const ex = r.result?.exceptionDetails;
  if (ex) throw new Error(ex.exception?.description ?? ex.text);
  return r.result.result.value;
}

async function waitFor(expr, what, timeout = 180000) {
  const t0 = Date.now();
  for (;;) {
    const v = await ev(expr);
    if (v) return v;
    if (Date.now() - t0 > timeout) throw new Error(`等待超时：${what}`);
    await new Promise((r) => setTimeout(r, 200));
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const js = (v) => JSON.stringify(v);

// ---------------------------------------------------------------- 界面探针

/** 当前显示的结果图版本（每次处理递增，进程内唯一）。 */
const RESULT_VERSION = `Math.max(0, ...[...document.querySelectorAll('img.layer.top')]
  .map((i) => i.getAttribute('src') || '').filter((s) => s.includes('/result'))
  .map((s) => +new URL(s).searchParams.get('v')))`;
const IDLE = `document.querySelector('.viewer-busy').hidden`;
const ERROR_TOASTS = `[...document.querySelectorAll('.toast-error')].map((t) => t.innerText.replace(/\\s+/g, ' '))`;
const STATUS = `document.querySelector('#status-text').textContent`;

/** 执行一个会触发处理的操作，等新结果显示出来。 */
async function processAfter(action, what) {
  const before = await ev(RESULT_VERSION);
  await ev(`(() => { ${action} })()`);
  await waitFor(`${RESULT_VERSION} > ${before} && ${IDLE}`, what);
  const errors = await ev(ERROR_TOASTS);
  if (errors.length) throw new Error(`${what}：${errors.join(' | ')}`);
  return ev(STATUS);
}

const selectField = (label, value) => `
  const field = [...document.querySelectorAll('#params .field')]
    .find((f) => f.querySelector('.field-label')?.firstChild?.textContent === ${js(label)});
  const s = field.querySelector('select');
  s.value = ${js(String(value))};
  s.dispatchEvent(new Event('change', { bubbles: true }));`;
/** 切换"预设 / 手动参数"；已经是该模式时不会触发处理，直接返回。 */
async function switchMode(text) {
  const active = await ev(`document.querySelector('.mode-switch .seg-btn.active')?.textContent`);
  if (active === text) return ev(STATUS);
  return processAfter(
    `[...document.querySelectorAll('.mode-switch .seg-btn')].find((b) => b.textContent === ${js(text)}).click();`,
    `切到${text}`,
  );
}

// ---------------------------------------------------------------- 步骤

let failed = 0;
async function step(name, fn) {
  const t0 = Date.now();
  try {
    const detail = await fn();
    console.log(`PASS ${name}（${((Date.now() - t0) / 1000).toFixed(1)} s）${detail ? ` — ${detail}` : ''}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${name} — ${e.message}`);
  }
}

const savedPrefs = await ev(`JSON.stringify(Object.fromEntries(Object.keys(localStorage).map((k) => [k, localStorage.getItem(k)])))`);
const outDir = mkdtempSync(join(tmpdir(), 'prs-smoke-'));

try {
  await step('打开照片（模拟拖入）', async () => {
    await ev(`document.querySelector('[data-tab=single]').click(); document.querySelector('[data-view=split]').click();`);
    await ev(`document.querySelector('#chk-auto').checked || document.querySelector('#chk-auto').click()`);
    const status = await processAfter(
      `window.__TAURI__.event.emit('tauri://drag-drop', { paths: [${js(photoPath)}], position: { x: 0, y: 0 } })`,
      '打开并处理',
    );
    const faces = await ev(`document.querySelector('.faces-head span')?.textContent`);
    return `${faces} · ${status}`;
  });

  await step('手动参数：四种磨皮模式', async () => {
    const out = [];
    await switchMode('手动参数');
    for (const [i, mode] of ['freqsep', 'gpupixel', 'cream', 'faithful'].entries()) {
      const index = ['faithful', 'freqsep', 'gpupixel', 'cream'].indexOf(mode);
      const status = await processAfter(selectField('磨皮模式', index), `磨皮模式 ${mode}`);
      out.push(`${mode} ${status.match(/处理 ([^·]+)/)?.[1]?.trim() ?? '?'}`);
      void i;
    }
    return out.join('，');
  });

  await step('奶油肌预设', async () => {
    return switchMode('预设');
  });

  await step('调试视图', async () => {
    const out = [];
    for (const view of ['landmarks', 'skin', 'matte', 'skin_prob', 'ai_holes']) {
      await ev(`document.querySelector('[data-view=${view}]').click()`);
      await waitFor(
        `${IDLE} && [...document.querySelectorAll('img.layer.top')].some((i) => (i.getAttribute('src') || '').includes('/${view}') && i.complete && i.naturalWidth > 0)`,
        `视图 ${view}`,
        60000,
      );
      const errors = await ev(ERROR_TOASTS);
      if (errors.length) throw new Error(`${view}：${errors.join(' | ')}`);
      out.push(view);
    }
    await ev(`document.querySelector('[data-view=split]').click()`);
    return out.join('，');
  });

  await step('切换工作分辨率', async () => {
    const current = await ev(`document.querySelector('#photo-card select').value`);
    const target = current === '1024' ? 1600 : 1024;
    const status = await processAfter(
      `const s = document.querySelector('#photo-card select'); s.value = ${js(String(target))}; s.dispatchEvent(new Event('change', { bubbles: true }));`,
      '切换工作分辨率',
    );
    const m = status.match(/工作图 (\d+)×(\d+)/);
    if (!m || Math.max(+m[1], +m[2]) !== target) throw new Error(`期望长边 ${target}，状态栏：${status}`);
    return `${current} → ${target}：${status}`;
  });

  await step('批处理（单张，PNG）', async () => {
    await ev(`localStorage.setItem('prs.batch', ${js(js({
      inputs: [photoPath],
      output_dir: outDir,
      recursive: false,
      overwrite: true,
      format: 'png',
      jpeg_quality: 98,
      write_landmarks: true,
    }))}); location.reload();`).catch(() => {});
    await sleep(2500);
    await ev(`document.querySelector('[data-tab=batch]').click(); document.querySelector('#batch-start').click();`);
    await waitFor(`document.body.classList.contains('batch-running')`, '批处理开始', 20000);
    await waitFor(`!document.body.classList.contains('batch-running')`, '批处理结束', 300000);
    const text = await ev(`document.querySelector('#batch-progress-text').textContent`);
    const counts = await ev(`document.querySelector('#batch-counts').innerText.replace(/\\s+/g, ' ')`);
    const output = join(outDir, basename(photoPath, extname(photoPath)) + '.png');
    for (const f of [output, join(outDir, 'batch_report.csv'), join(outDir, 'batch_report.json')]) {
      if (!existsSync(f)) throw new Error(`缺少输出 ${f}`);
    }
    if (!counts.startsWith('完成 1')) throw new Error(`计数：${counts}`);
    return `${text} · ${counts}`;
  });
} finally {
  // 恢复界面偏好并刷新，清理临时输出
  await ev(`(() => { localStorage.clear(); const saved = ${savedPrefs}; for (const [k, v] of Object.entries(saved)) localStorage.setItem(k, v); location.reload(); })()`).catch(() => {});
  rmSync(outDir, { recursive: true, force: true });
  if (pageErrors.length) {
    failed++;
    console.log(`FAIL 页面脚本错误：\n  ${pageErrors.join('\n  ')}`);
  }
  console.log(failed ? `\n${failed} 项失败` : '\n全部通过');
  ws.close();
  process.exitCode = failed ? 1 : 0;
}
