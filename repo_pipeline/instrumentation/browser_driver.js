/**
 * In-container browser driver for agent-driven trajectory capture.
 *
 * A long-running HTTP server holding ONE Playwright page at 1280x720. The host-side agent
 * (repo_pipeline/agent_capture.py) calls these endpoints through `docker exec curl`; every
 * action endpoint records a trajectory step in the repo-wide schema (screenshot taken BEFORE
 * the action + the action dict with the target element's bbox-center coordinates) before
 * performing the action with raw mouse/keyboard input at those exact coordinates. What the
 * student model will be trained to do is therefore exactly what was executed.
 *
 * Element references: /snapshot walks the top-level DOM, tags interactive elements with a
 * data-cua-ref attribute and returns a compact text tree. Action endpoints resolve refs via
 * [data-cua-ref="N"]. Refs survive until the next /snapshot reassigns them; acting on a stale
 * ref after a page change returns an error the agent can recover from by re-snapshotting.
 *
 * Endpoints (all JSON):
 *   GET  /status                                -> {ok, ready, task_id, steps}
 *   POST /session  {task_id, start_url}         -> open a fresh context+page, start recording
 *   POST /snapshot                              -> {url, title, snapshot}
 *   POST /screenshot                            -> {b64}  (PNG of the current viewport)
 *   POST /act      {kind, thought, ...}         -> record + perform one action, see below
 *   POST /navigate {url}                        -> goto (NOT recorded as a step; logged in meta)
 *   POST /wait     {seconds}                    -> settle; no step recorded
 *   POST /finish   {agent_outcome, agent_summary} -> finalize trajectory.json + final screenshot
 *
 * /act kinds:
 *   click   {ref, button?, dblclick?}           -> {type:"click", x, y, button}
 *   type    {ref, text, press_enter?}           -> click + {type:"type", text} (+ Enter keypress)
 *   press   {keys: ["Control","a"]}             -> {type:"keypress", keys}
 *   scroll  {dy}                                -> {type:"scroll", x, y, scroll_x, scroll_y}
 *   drag    {ref, to_ref}                       -> {type:"drag", path:[{x,y},{x,y}]}
 *   select  {ref, value}                        -> {type:"click", x, y} (native <select>)
 */
'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = parseInt(process.env.DRIVER_PORT || '8765', 10);
const RESULTS_ROOT = process.env.TRAJECTORY_DIR || '/workspace/results';
const VIEWPORT = { width: 1280, height: 720 };
const SNAPSHOT_MAX_ELEMENTS = 150;
const LOOP_GUARD_REPEATS = 3;

function requirePlaywright() {
  const candidates = [
    '/opt/driver/node_modules/playwright-core',
    '/workspace/driver/node_modules/playwright-core',
    'playwright-core',
    '/workspace/repo/node_modules/playwright-core',
    '/workspace/repo/node_modules/playwright',
  ];
  for (const c of candidates) {
    try { return require(c); } catch (e) { /* try next */ }
  }
  throw new Error('playwright-core not found in any known location');
}

function findChromium() {
  const roots = [process.env.PLAYWRIGHT_BROWSERS_PATH, '/ms-playwright'].filter(Boolean);
  for (const root of roots) {
    let entries;
    try { entries = fs.readdirSync(root); } catch (e) { continue; }
    for (const entry of entries.sort().reverse()) {
      if (!entry.startsWith('chromium')) continue;
      for (const sub of ['chrome-linux/chrome', 'chrome-linux/headless_shell']) {
        const p = path.join(root, entry, sub);
        if (fs.existsSync(p)) return p;
      }
    }
  }
  return null; // let playwright-core resolve its own bundled revision
}

const playwright = requirePlaywright();

const state = {
  browser: null,
  context: null,
  page: null,
  task: null, // {id, dir, startUrl, steps, meta, navigations, recentActions}
};

/* ---------------------------- snapshot (in-page) ---------------------------- */

// Serialized into the page; walks the DOM, tags interactive elements with data-cua-ref and
// returns {elements: [...], texts: [...]} for the compact text tree rendered host-side below.
function pageCollectSnapshot(maxElements) {
  const INTERACTIVE = 'a[href], button, input, select, textarea, summary, [role], [onclick], [contenteditable="true"], [tabindex]';
  const out = [];
  let refCounter = 1;
  document.querySelectorAll('[data-cua-ref]').forEach((el) => el.removeAttribute('data-cua-ref'));

  function visible(el) {
    const rect = el.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return null;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0) return null;
    return rect;
  }

  function accessibleName(el) {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria;
    if (el.labels && el.labels.length) return el.labels[0].innerText.trim().slice(0, 80);
    const ph = el.getAttribute('placeholder');
    if (ph) return ph;
    const title = el.getAttribute('title');
    if (title) return title;
    const alt = el.getAttribute('alt');
    if (alt) return alt;
    const text = (el.innerText || el.value || '').trim().replace(/\s+/g, ' ');
    if (text) return text.slice(0, 80);
    return el.getAttribute('name') || el.id || '';
  }

  function roleOf(el) {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'button' || tag === 'summary') return 'button';
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'input') {
      const t = (el.getAttribute('type') || 'text').toLowerCase();
      if (['button', 'submit', 'reset', 'image'].includes(t)) return 'button';
      if (t === 'checkbox') return 'checkbox';
      if (t === 'radio') return 'radio';
      if (t === 'range') return 'slider';
      return 'textbox';
    }
    return tag;
  }

  const seen = new Set();
  for (const el of document.querySelectorAll(INTERACTIVE)) {
    if (out.length >= maxElements) break;
    if (seen.has(el)) continue;
    seen.add(el);
    const rect = visible(el);
    if (!rect) continue;
    const ref = refCounter++;
    el.setAttribute('data-cua-ref', String(ref));
    const role = roleOf(el);
    const entry = {
      ref,
      role,
      name: accessibleName(el),
      inViewport: rect.bottom > 0 && rect.top < window.innerHeight,
    };
    if (role === 'textbox' && 'value' in el) entry.value = String(el.value || '').slice(0, 60);
    if (role === 'checkbox' || role === 'radio') entry.checked = !!el.checked;
    if (el.disabled) entry.disabled = true;
    if (role === 'combobox' && el.tagName.toLowerCase() === 'select') {
      entry.options = Array.from(el.options).slice(0, 20).map((o) => o.value || o.text);
    }
    out.push(entry);
  }

  const texts = [];
  for (const el of document.querySelectorAll('h1, h2, h3, [role="alert"], [role="status"], .error, .alert')) {
    if (texts.length >= 25) break;
    const rect = visible(el);
    if (!rect) continue;
    const t = (el.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 120);
    if (t) texts.push(t);
  }

  return {
    elements: out,
    texts,
    scrollY: Math.round(window.scrollY),
    scrollHeight: Math.round(document.documentElement.scrollHeight),
    innerHeight: window.innerHeight,
  };
}

function renderSnapshot(data) {
  const lines = [];
  const aboveFold = data.elements.filter((e) => e.inViewport);
  const belowFold = data.elements.filter((e) => !e.inViewport);

  if (data.texts.length) {
    lines.push('## Page text');
    for (const t of data.texts) lines.push(`- ${t}`);
  }
  lines.push('## Interactive elements (in viewport)');
  for (const e of aboveFold) lines.push(renderElement(e));
  if (belowFold.length) {
    lines.push(`## Below the fold (scroll down to interact) — ${belowFold.length} elements`);
    for (const e of belowFold.slice(0, 40)) lines.push(renderElement(e));
  }
  const maxScroll = Math.max(0, data.scrollHeight - data.innerHeight);
  lines.push(`## Scroll position: ${data.scrollY}/${maxScroll}px`);
  return lines.join('\n');
}

function renderElement(e) {
  let line = `[${e.ref}] ${e.role} "${e.name}"`;
  if (e.value !== undefined) line += ` value="${e.value}"`;
  if (e.checked !== undefined) line += e.checked ? ' (checked)' : ' (unchecked)';
  if (e.disabled) line += ' (disabled)';
  if (e.options) line += ` options=[${e.options.join(', ')}]`;
  return line;
}

// Dev-server tooling injects full-screen overlays (webpack/CRA eslint walls, Vite error
// overlays, Next.js portals) that are NOT part of the app a user would see in production.
// They contaminate screenshots and snapshots, so every observation scrubs them first; the
// count is recorded in capture_meta for observability. Extend the registry per framework.
const DEV_OVERLAY_SELECTORS = [
  'iframe#webpack-dev-server-client-overlay',
  '#webpack-dev-server-client-overlay-div',
  'vite-error-overlay',
  'nextjs-portal',
  '#react-refresh-overlay',
];

async function scrubDevOverlays() {
  const removed = await state.page.evaluate((selectors) => {
    let n = 0;
    for (const sel of selectors) {
      document.querySelectorAll(sel).forEach((el) => { el.remove(); n += 1; });
    }
    return n;
  }, DEV_OVERLAY_SELECTORS).catch(() => 0);
  if (removed && state.task) {
    state.task.meta.dev_overlays_scrubbed = (state.task.meta.dev_overlays_scrubbed || 0) + removed;
  }
  return removed;
}

async function takeSnapshot() {
  await scrubDevOverlays();
  const data = await state.page.evaluate(pageCollectSnapshot, SNAPSHOT_MAX_ELEMENTS);
  return renderSnapshot(data);
}

/* ------------------------------ recording ------------------------------ */

function trajectoryPath() { return path.join(state.task.dir, 'trajectory.json'); }

function writeTrajectory(success) {
  const t = state.task;
  const trajectory = {
    task_id: t.id,
    instruction: t.instruction || '',
    start_url: t.startUrl,
    steps: t.steps,
    total_steps: t.steps.length,
    success: !!success,
  };
  fs.writeFileSync(trajectoryPath(), JSON.stringify(trajectory, null, 2));
  fs.writeFileSync(path.join(t.dir, 'capture_meta.json'), JSON.stringify(t.meta, null, 2));
}

async function recordStep(thought, actions, target) {
  const t = state.task;
  const stepNum = t.steps.length + 1;
  const screenshotName = `step_${String(stepNum).padStart(3, '0')}_screenshot.png`;
  await scrubDevOverlays();
  await state.page.screenshot({ path: path.join(t.dir, screenshotName) });

  const step = { step: stepNum, screenshot: screenshotName, thought: thought || '', actions };
  if (target) {
    step.role = target.role || null;
    step.name = target.name || null;
    step.bbox = target.bbox || null;
  }
  t.steps.push(step);
  t.meta.steps[String(stepNum)] = {
    role: target ? target.role : null,
    name: target ? target.name : null,
    bbox: target ? target.bbox : null,
    url: state.page.url(),
    raw_thought: thought || '',
  };
  writeTrajectory(false);
  return stepNum;
}

/* ------------------------------ element resolution ------------------------------ */

async function resolveRef(ref) {
  const locator = state.page.locator(`[data-cua-ref="${ref}"]`).first();
  if ((await locator.count()) === 0) {
    throw new Error(`ref ${ref} not found on the current page — it may be stale. Take a new snapshot.`);
  }
  await locator.scrollIntoViewIfNeeded({ timeout: 5000 }).catch(() => {});
  const box = await locator.boundingBox();
  if (!box) throw new Error(`ref ${ref} has no visible bounding box. Take a new snapshot.`);
  let cx = Math.round(box.x + box.width / 2);
  let cy = Math.round(box.y + box.height / 2);
  // Clamp the click point into the viewport for oversized elements (e.g. full-page sections).
  cx = Math.min(Math.max(cx, 1), VIEWPORT.width - 1);
  cy = Math.min(Math.max(cy, 1), VIEWPORT.height - 1);
  const info = await locator.evaluate((el) => ({
    role: el.getAttribute('role') || el.tagName.toLowerCase(),
    name: (el.getAttribute('aria-label') || el.getAttribute('placeholder') || (el.innerText || el.value || '').trim()).replace(/\s+/g, ' ').slice(0, 80),
  })).catch(() => ({ role: null, name: null }));
  return {
    locator,
    cx,
    cy,
    role: info.role,
    name: info.name,
    bbox: { x: Math.round(box.x), y: Math.round(box.y), width: Math.round(box.width), height: Math.round(box.height) },
  };
}

async function settle() {
  await state.page.waitForLoadState('domcontentloaded', { timeout: 5000 }).catch(() => {});
  await state.page.waitForTimeout(400);
}

/* ------------------------------ loop guard ------------------------------ */

function loopGuard(signature) {
  const t = state.task;
  t.recentActions.push(signature);
  if (t.recentActions.length > LOOP_GUARD_REPEATS) t.recentActions.shift();
  if (
    t.recentActions.length === LOOP_GUARD_REPEATS &&
    t.recentActions.every((s) => s === signature)
  ) {
    t.recentActions.length = 0;
    throw new Error(
      `loop guard: the same action was attempted ${LOOP_GUARD_REPEATS} times in a row with no progress. ` +
      'The action is refused — re-snapshot, reconsider the page state, and try a DIFFERENT approach.'
    );
  }
}

/* ------------------------------ action handlers ------------------------------ */

async function handleAct(body) {
  if (!state.page) throw new Error('no active session — call /session first');
  const { kind, thought } = body;
  if (!thought || !String(thought).trim()) {
    throw new Error("every action requires a non-empty 'thought' explaining the intent behind it");
  }

  if (kind === 'click' || kind === 'select' || kind === 'drag' || kind === 'type') {
    if (body.ref === undefined || body.ref === null) throw new Error(`kind '${kind}' requires a 'ref'`);
  }

  let actions;
  let target = null;

  if (kind === 'click') {
    loopGuard(`click:${body.ref}`);
    target = await resolveRef(body.ref);
    const button = body.button === 'right' ? 'right' : 'left';
    actions = [{ type: 'click', x: target.cx, y: target.cy, button }];
    await recordStep(thought, actions, target);
    if (body.dblclick) await state.page.mouse.dblclick(target.cx, target.cy, { button });
    else await state.page.mouse.click(target.cx, target.cy, { button });
  } else if (kind === 'type') {
    loopGuard(`type:${body.ref}:${body.text}`);
    if (typeof body.text !== 'string') throw new Error("kind 'type' requires 'text'");
    target = await resolveRef(body.ref);
    actions = [
      { type: 'click', x: target.cx, y: target.cy, button: 'left' },
      { type: 'type', text: body.text },
    ];
    if (body.press_enter) actions.push({ type: 'keypress', keys: ['Enter'] });
    await recordStep(thought, actions, target);
    await state.page.mouse.click(target.cx, target.cy);
    // Clear any existing value the way a user would: select-all then retype.
    await state.page.keyboard.press('ControlOrMeta+a').catch(() => {});
    await state.page.keyboard.type(body.text, { delay: 15 });
    if (body.press_enter) await state.page.keyboard.press('Enter');
  } else if (kind === 'press') {
    const keys = Array.isArray(body.keys) ? body.keys : [body.keys];
    loopGuard(`press:${keys.join('+')}`);
    actions = [{ type: 'keypress', keys }];
    await recordStep(thought, actions, null);
    await state.page.keyboard.press(keys.join('+'));
  } else if (kind === 'scroll') {
    const dy = Number.isFinite(body.dy) ? Math.round(body.dy) : 500;
    loopGuard(`scroll:${dy}:${await state.page.evaluate(() => Math.round(window.scrollY))}`);
    const cx = Math.round(VIEWPORT.width / 2);
    const cy = Math.round(VIEWPORT.height / 2);
    actions = [{ type: 'scroll', x: cx, y: cy, scroll_x: 0, scroll_y: dy }];
    await recordStep(thought, actions, null);
    await state.page.mouse.move(cx, cy);
    await state.page.mouse.wheel(0, dy);
  } else if (kind === 'drag') {
    if (body.to_ref === undefined || body.to_ref === null) throw new Error("kind 'drag' requires 'to_ref'");
    loopGuard(`drag:${body.ref}:${body.to_ref}`);
    target = await resolveRef(body.ref);
    const dest = await resolveRef(body.to_ref);
    actions = [{ type: 'drag', path: [{ x: target.cx, y: target.cy }, { x: dest.cx, y: dest.cy }] }];
    await recordStep(thought, actions, target);
    await state.page.mouse.move(target.cx, target.cy);
    await state.page.mouse.down();
    await state.page.mouse.move(dest.cx, dest.cy, { steps: 12 });
    await state.page.mouse.up();
  } else if (kind === 'select') {
    if (body.value === undefined) throw new Error("kind 'select' requires 'value'");
    loopGuard(`select:${body.ref}:${body.value}`);
    target = await resolveRef(body.ref);
    actions = [{ type: 'click', x: target.cx, y: target.cy, button: 'left' }];
    await recordStep(thought, actions, target);
    await target.locator.selectOption(String(body.value));
    state.task.meta.steps[String(state.task.steps.length)].select_value = String(body.value);
    writeTrajectory(false);
  } else {
    throw new Error(`unknown action kind '${kind}'`);
  }

  await settle();
  const snapshot = await takeSnapshot();
  return {
    ok: true,
    step: state.task.steps.length,
    url: state.page.url(),
    snapshot: snapshot.slice(0, 8000),
  };
}

/* ------------------------------ session lifecycle ------------------------------ */

async function ensureBrowser() {
  if (state.browser) return;
  const executablePath = findChromium();
  state.browser = await playwright.chromium.launch({
    headless: true,
    executablePath: executablePath || undefined,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'],
  });
}

async function handleSession(body) {
  if (!body.task_id || !body.start_url) throw new Error('/session requires task_id and start_url');
  await ensureBrowser();
  if (state.context) await state.context.close().catch(() => {});

  state.context = await state.browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: 1,
  });
  // Follow popups/new tabs, and auto-accept JS dialogs on every page (they block otherwise).
  state.context.on('page', (p) => {
    state.page = p;
    p.on('dialog', (d) => d.accept().catch(() => {}));
  });
  state.page = await state.context.newPage();
  state.page.on('dialog', (d) => d.accept().catch(() => {}));

  const safeId = String(body.task_id).replace(/[^a-zA-Z0-9_.-]+/g, '_').slice(0, 120);
  const dir = path.join(RESULTS_ROOT, `task_${safeId}`);
  fs.mkdirSync(dir, { recursive: true });

  state.task = {
    id: safeId,
    dir,
    instruction: body.instruction || '',
    startUrl: body.start_url,
    steps: [],
    meta: { recorder: 'browser_driver', framework: 'agent', steps: {}, navigations: [] },
    recentActions: [],
  };

  await state.page.goto(body.start_url, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await settle();
  const snapshot = await takeSnapshot();
  return { ok: true, url: state.page.url(), snapshot: snapshot.slice(0, 15000) };
}

async function handleFinish(body) {
  if (!state.task) throw new Error('no active session');
  if (state.page) {
    await scrubDevOverlays();
    await state.page.screenshot({ path: path.join(state.task.dir, 'final_screenshot.png') }).catch(() => {});
  }
  state.task.meta.agent_outcome = body.agent_outcome || '';
  state.task.meta.agent_summary = body.agent_summary || '';
  // success stays false until the host-side verifier confirms it (convert skips success:false).
  writeTrajectory(false);
  const steps = state.task.steps.length;
  const dir = state.task.dir;
  if (state.context) await state.context.close().catch(() => {});
  state.context = null;
  state.page = null;
  state.task = null;
  return { ok: true, steps, dir };
}

/* ------------------------------ HTTP plumbing ------------------------------ */

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => {
      data += chunk;
      if (data.length > 2_000_000) reject(new Error('body too large'));
    });
    req.on('end', () => {
      if (!data) return resolve({});
      try { resolve(JSON.parse(data)); } catch (e) { reject(new Error('invalid JSON body')); }
    });
    req.on('error', reject);
  });
}

const routes = {
  'POST /session': handleSession,
  'POST /act': handleAct,
  'POST /finish': handleFinish,
  'POST /snapshot': async () => {
    if (!state.page) throw new Error('no active session');
    return { ok: true, url: state.page.url(), title: await state.page.title(), snapshot: (await takeSnapshot()).slice(0, 15000) };
  },
  'POST /screenshot': async () => {
    if (!state.page) throw new Error('no active session');
    await scrubDevOverlays();
    const buf = await state.page.screenshot();
    return { ok: true, b64: buf.toString('base64') };
  },
  'POST /navigate': async (body) => {
    if (!state.page) throw new Error('no active session');
    if (!body.url) throw new Error('/navigate requires url');
    if (state.task.steps.length > 0) {
      state.task.meta.navigations.push({ after_step: state.task.steps.length, url: body.url });
    }
    await state.page.goto(body.url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await settle();
    return { ok: true, url: state.page.url(), snapshot: (await takeSnapshot()).slice(0, 8000) };
  },
  'POST /wait': async (body) => {
    if (!state.page) throw new Error('no active session');
    const seconds = Math.min(Math.max(Number(body.seconds) || 1, 0.2), 10);
    await state.page.waitForTimeout(seconds * 1000);
    return { ok: true, url: state.page.url(), snapshot: (await takeSnapshot()).slice(0, 8000) };
  },
  'GET /status': async () => ({
    ok: true,
    ready: true,
    task_id: state.task ? state.task.id : null,
    steps: state.task ? state.task.steps.length : 0,
  }),
};

const server = http.createServer(async (req, res) => {
  const key = `${req.method} ${req.url.split('?')[0]}`;
  const handler = routes[key];
  res.setHeader('Content-Type', 'application/json');
  if (!handler) {
    res.statusCode = 404;
    res.end(JSON.stringify({ ok: false, error: `no route ${key}` }));
    return;
  }
  try {
    const body = await readBody(req);
    const result = await handler(body);
    res.statusCode = 200;
    res.end(JSON.stringify(result));
  } catch (e) {
    res.statusCode = 400;
    res.end(JSON.stringify({ ok: false, error: String(e && e.message ? e.message : e).slice(0, 500) }));
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`browser_driver listening on 127.0.0.1:${PORT}`);
});
