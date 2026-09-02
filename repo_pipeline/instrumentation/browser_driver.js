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
 *   POST /observe                               -> {b64 (PNG), url, title, step_next}; the PNG becomes the next recorded step's screenshot
 *
 * /act kinds:
 *   click     {ref, button?, dblclick?}         -> {type:"click", x, y, button}
 *   type      {ref, text, press_enter?}         -> click + {type:"type", text} (+ Enter keypress)
 *   press     {keys: ["Control","a"]}           -> {type:"keypress", keys}
 *   scroll    {dy}                              -> {type:"scroll", x, y, scroll_x, scroll_y}
 *   drag      {ref, to_ref}                     -> {type:"drag", path} (HTML5-DnD fallback)
 *   hover     {ref}                             -> {type:"move", x, y} (hover-revealed UI)
 *   click_xy  {x, y, button?, dblclick?}        -> {type:"click", x, y} (canvas/maps, no DOM ref)
 *   drag_path {path:[{x,y},...]}                -> {type:"drag", path} (canvas drawing/dragging)
 *   move_xy   {x, y}                            -> {type:"move", x, y} (hover at raw coordinates)
 *   type_text {text, press_enter?}              -> {type:"type", text} (types into the focused element)
 *   scroll_xy {x, y, dx?, dy?}                  -> {type:"scroll", x, y, scroll_x, scroll_y}
 *   wait      {seconds}                         -> {type:"wait"} (recorded: waiting is learnable)
 *   select    {ref, value}                      -> {type:"click", x, y} (native <select>)
 *   upload    {ref, filename?, content?}        -> meta only (no replayable pixel equivalent)
 *   sequence  {actions:[contract dicts]}         -> records ONE step with those dicts verbatim, executes them in order (RL policy path)
 *
 * POST /affordances reports counts of sophistication signals (draggables, sliders, canvases,
 * rich-text editors...) for boot-gate profiling and task-proposer quotas.
 */
'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = parseInt(process.env.DRIVER_PORT || '8765', 10);
const RESULTS_ROOT = process.env.TRAJECTORY_DIR || '/workspace/results';
const VIEWPORT = { width: 1280, height: 720 };
const SNAPSHOT_MAX_ELEMENTS = 150;
// 0 disables the guard: student eval must let a stuck model actually repeat itself.
const LOOP_GUARD_REPEATS = parseInt(process.env.DRIVER_LOOP_GUARD_REPEATS || '3', 10);

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
function pageCollectSnapshot(opts) {
  // `startRef` offsets this frame's ref numbering so refs stay globally unique when the host
  // collects across multiple frames (main page + iframes, e.g. a webmail message body).
  const maxElements = (opts && opts.maxElements) || 60;
  const startRef = (opts && opts.startRef) || 0;
  const INTERACTIVE = 'a[href], button, input, select, textarea, summary, [role], [onclick], '
    + '[contenteditable="true"], [tabindex], [draggable="true"], [data-rbd-draggable-id], canvas';
  const out = [];
  let refCounter = startRef + 1;
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
  // Collect across the main frame AND every child iframe, with a running ref offset so refs are
  // globally unique. Elements inside iframes (e.g. a webmail message body with a verification or
  // password-reset link) become first-class, ref-clickable targets — Playwright's frame locators
  // and boundingBox() resolve them in main-frame coordinates, so clicks land correctly.
  const frames = state.page.frames();
  let startRef = 0;
  let merged = null;
  for (const frame of frames) {
    let data;
    try {
      data = await frame.evaluate(pageCollectSnapshot, { maxElements: SNAPSHOT_MAX_ELEMENTS, startRef });
    } catch (e) {
      continue;  // detached or unreachable frame — skip
    }
    if (merged === null) {
      merged = data;  // main frame: keep its scroll geometry + page text
    } else {
      merged.elements.push(...data.elements);
      merged.texts.push(...data.texts);
    }
    startRef += data.elements.length;
  }
  return renderSnapshot(merged || { elements: [], texts: [], scrollY: 0, scrollHeight: 0, innerHeight: 0 });
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

async function recordStep(thought, actions, target, event) {
  const t = state.task;
  const stepNum = t.steps.length + 1;
  const screenshotName = `step_${String(stepNum).padStart(3, '0')}_screenshot.png`;
  await scrubDevOverlays();
  const pending = t.pendingShot;
  t.pendingShot = null;
  if (pending) fs.writeFileSync(path.join(t.dir, screenshotName), pending);
  else await state.page.screenshot({ path: path.join(t.dir, screenshotName) });

  const step = { step: stepNum, screenshot: screenshotName, thought: thought || '', actions };
  if (target) {
    step.role = target.role || null;
    step.name = target.name || null;
    step.bbox = target.bbox || null;
  }
  // Watchdog verdict observation (login-gated capture): an event step has empty actions; convert
  // excludes it as an SFT target but threads it into history.
  if (event) step.event = event;
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
  // The ref may live in the main frame or any child iframe (page.locator does not pierce iframes,
  // so search each frame). boundingBox() on a frame locator is already main-frame-relative, so the
  // recorded coordinates and the page.mouse clicks line up regardless of which frame owns the ref.
  let locator = null;
  for (const frame of state.page.frames()) {
    const cand = frame.locator(`[data-cua-ref="${ref}"]`).first();
    try {
      if ((await cand.count()) > 0) { locator = cand; break; }
    } catch (e) {
      continue;
    }
  }
  if (locator === null) {
    throw new Error(`ref ${ref} not found on the current page — it may be stale. Take a new snapshot.`);
  }
  await locator.scrollIntoViewIfNeeded({ timeout: 5000 }).catch(() => {});
  let box = await locator.boundingBox();
  if (!box) throw new Error(`ref ${ref} has no visible bounding box. Take a new snapshot.`);
  // scrollIntoViewIfNeeded only scrolls within the element's OWN frame; an element inside an iframe
  // (e.g. a link in a webmail message body) can still sit outside the MAIN viewport. boundingBox()
  // is main-frame-relative, so if the center is off-screen, scroll the main window to it and re-read
  // — otherwise the clamp below would push the click to the viewport edge and miss the element.
  let centerY = box.y + box.height / 2;
  if (centerY < 0 || centerY > VIEWPORT.height) {
    await state.page.evaluate((dy) => window.scrollBy(0, dy), Math.round(centerY - VIEWPORT.height / 2)).catch(() => {});
    await state.page.waitForTimeout(200);
    box = (await locator.boundingBox()) || box;
  }
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
  // Animation quiescence: screenshots taken mid-transition (sliding panels, fading modals)
  // are incoherent observations. Sample the layout twice per check and wait until two
  // consecutive samples match, capped so a perpetual animation (carousel, spinner) can't
  // stall the driver.
  const deadline = Date.now() + 1500;
  let prev = '';
  while (Date.now() < deadline) {
    const sample = await state.page.evaluate(() => {
      const rects = [];
      const els = document.querySelectorAll('body *');
      for (let i = 0; i < els.length && rects.length < 60; i += Math.ceil(els.length / 60) || 1) {
        const r = els[i].getBoundingClientRect();
        rects.push(`${Math.round(r.x)},${Math.round(r.y)},${Math.round(r.width)}`);
      }
      return rects.join(';');
    }).catch(() => '');
    if (sample && sample === prev) break;
    prev = sample;
    await state.page.waitForTimeout(150);
  }
}

/* ------------------------------ drag support ------------------------------ */

// Light DOM fingerprint to detect whether a mouse-path drag actually changed anything.
// Pointer-event libraries (dnd-kit, react-dnd pointer sensors) respond to raw mouse input,
// but native HTML5 drag-and-drop ignores it in headless Chromium — those need synthesized
// DragEvents. The fallback fires only when the mouse drag provably did nothing.
async function domFingerprint() {
  return state.page.evaluate(() => {
    const els = document.querySelectorAll('body *');
    let order = '';
    for (let i = 0; i < Math.min(els.length, 120); i += 1) order += els[i].tagName[0];
    const text = document.body ? document.body.innerText : '';
    return `${els.length}:${text.length}:${order}`;
  }).catch(() => '');
}

async function html5DragFallback(sourceLocator, destLocator) {
  const source = await sourceLocator.elementHandle();
  const dest = await destLocator.elementHandle();
  if (!source || !dest) return false;
  await state.page.evaluate(([src, dst]) => {
    const dataTransfer = new DataTransfer();
    const fire = (el, type) => el.dispatchEvent(
      new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer })
    );
    fire(src, 'dragstart');
    fire(dst, 'dragenter');
    fire(dst, 'dragover');
    fire(dst, 'drop');
    fire(src, 'dragend');
  }, [source, dest]);
  return true;
}

/* ------------------------------ loop guard ------------------------------ */

function loopGuard(signature) {
  if (!LOOP_GUARD_REPEATS) return;
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

function requireInViewport(x, y, label) {
  if (!(x >= 0 && x < VIEWPORT.width && y >= 0 && y < VIEWPORT.height)) {
    throw new Error(`${label} (${x},${y}) outside the ${VIEWPORT.width}x${VIEWPORT.height} viewport`);
  }
}

// Substitute `<secret:NAME>` placeholders with the session's real (burner) credential values at
// TYPE time only. The placeholder is what gets recorded into the trajectory; the resolved value is
// only ever keystroked into the page, never written to disk or returned. Unknown keys are left as
// the literal placeholder (fail-visible, not fail-leak).
function resolveSecrets(text) {
  const secrets = state.secrets || {};
  if (typeof text !== 'string') return text;
  return text.replace(/<secret:([^>]+)>/g, (m, key) => (key in secrets ? secrets[key] : m));
}

/* ------------------------------ RL sequence path ------------------------------ */

const SEQUENCE_TYPES = new Set(['click', 'double_click', 'move', 'type', 'keypress', 'scroll', 'drag', 'wait', 'navigate', 'done', 'handoff', 'confirm', 'screenshot']);
const TERMINAL_TYPES = new Set(['done', 'handoff', 'confirm']);
const SEQUENCE_WAIT_MS = parseInt(process.env.SEQUENCE_WAIT_MS || '2000', 10);

function inViewport(x, y) {
  return x >= 0 && x < VIEWPORT.width && y >= 0 && y < VIEWPORT.height;
}

// Validate + normalize one contract action dict; the recorded dict is exactly what the student emitted.
function normalizeContractAction(a) {
  if (!a || typeof a !== 'object' || !SEQUENCE_TYPES.has(a.type)) {
    throw new Error(`sequence: unsupported action ${JSON.stringify(a).slice(0, 80)}`);
  }
  const out = { type: a.type };
  if (['click', 'double_click', 'move', 'scroll'].includes(a.type)) {
    out.x = Math.round(Number(a.x));
    out.y = Math.round(Number(a.y));
    if (!inViewport(out.x, out.y)) throw new Error(`sequence: ${a.type} (${out.x},${out.y}) outside the ${VIEWPORT.width}x${VIEWPORT.height} viewport`);
  }
  if (a.type === 'click') out.button = a.button === 'right' ? 'right' : 'left';
  if (a.type === 'type') {
    if (typeof a.text !== 'string') throw new Error('sequence: type requires text');
    out.text = a.text;
  }
  if (a.type === 'keypress') {
    out.keys = (Array.isArray(a.keys) ? a.keys : [a.keys]).map(String);
    if (!out.keys.length) throw new Error('sequence: keypress requires keys');
  }
  if (a.type === 'scroll') {
    out.scroll_x = Math.round(Number(a.scroll_x) || 0);
    out.scroll_y = Math.round(Number(a.scroll_y) || 0);
  }
  if (a.type === 'drag') {
    out.path = (Array.isArray(a.path) ? a.path : []).map((p) => ({ x: Math.round(Number(p.x)), y: Math.round(Number(p.y)) }));
    if (out.path.length < 2 || !out.path.every((p) => inViewport(p.x, p.y))) throw new Error('sequence: drag requires >=2 in-viewport points');
  }
  if (a.type === 'navigate') {
    if (!a.url) throw new Error('sequence: navigate requires url');
    out.url = String(a.url);
  }
  if (a.type === 'handoff') out.reason = String(a.reason || 'login');
  if (a.type === 'confirm') {
    out.proposed_action = a.proposed_action || {};
    out.risk_category = String(a.risk_category || '');
  }
  return out;
}

async function focusedIsEditable() {
  return state.page.evaluate(() => {
    const el = document.activeElement;
    if (!el) return false;
    return el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable === true;
  }).catch(() => false);
}

const SEQUENCE_EXECUTORS = {
  click: async (a) => state.page.mouse.click(a.x, a.y, { button: a.button }),
  double_click: async (a) => state.page.mouse.dblclick(a.x, a.y),
  move: async (a) => { await state.page.mouse.move(a.x, a.y); await state.page.waitForTimeout(500); },
  type: async (a) => {
    // Ref-free typing goes to the focused element; `replace` mirrors the teacher's select-all-then-type.
    if (state.typeMode !== 'append' && await focusedIsEditable()) await state.page.keyboard.press('ControlOrMeta+a').catch(() => {});
    await state.page.keyboard.type(resolveSecrets(a.text), { delay: 15 });
  },
  keypress: async (a) => state.page.keyboard.press(a.keys.join('+')),
  scroll: async (a) => { await state.page.mouse.move(a.x, a.y); await state.page.mouse.wheel(a.scroll_x, a.scroll_y); },
  drag: async (a) => {
    await state.page.mouse.move(a.path[0].x, a.path[0].y);
    await state.page.mouse.down();
    for (const p of a.path.slice(1)) await state.page.mouse.move(p.x, p.y, { steps: 8 });
    await state.page.mouse.up();
  },
  wait: async () => state.page.waitForTimeout(SEQUENCE_WAIT_MS),
  navigate: async (a) => { await state.page.goto(a.url, { waitUntil: 'domcontentloaded', timeout: 30000 }); await settle(); },
  done: async () => {},
  handoff: async () => {},
  confirm: async () => {},
  screenshot: async () => {},
};

async function executeSequence(actions) {
  let executed = 0;
  for (const a of actions) {
    try {
      await SEQUENCE_EXECUTORS[a.type](a);
      executed += 1;
    } catch (e) {
      return { executed, error_at: executed, error: String(e && e.message ? e.message : e).slice(0, 300) };
    }
    if (TERMINAL_TYPES.has(a.type)) break;
  }
  return { executed, error_at: null, error: null };
}

async function handleObserve() {
  if (!state.page) throw new Error('no active session — call /session first');
  let last;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await scrubDevOverlays();
      await settle();
      const buf = await state.page.screenshot();
      const title = await state.page.title();
      state.task.pendingShot = buf;
      return { ok: true, b64: buf.toString('base64'), mime: 'image/png', url: state.page.url(), title, step_next: state.task.steps.length + 1 };
    } catch (e) {
      // A navigation committing right after the action destroys the execution context mid-read; settle again and retry.
      last = e;
      await state.page.waitForTimeout(500);
    }
  }
  throw last;
}

async function handleAct(body) {
  if (!state.page) throw new Error('no active session — call /session first');
  const { kind, thought } = body;
  // Upload is an unrecorded enabler; approval_event is a watchdog observation, not an agent action.
  // Every other recorded action requires a thought.
  if (kind !== 'upload' && kind !== 'approval_event' && (!thought || !String(thought).trim())) {
    throw new Error("every action requires a non-empty 'thought' explaining the intent behind it");
  }

  if (['click', 'select', 'drag', 'type', 'hover', 'upload'].includes(kind)) {
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
    if (body.dblclick) {
      await state.page.mouse.dblclick(target.cx, target.cy, { button });
    } else {
      // Prefer the locator click: it dispatches a real DOM click through nested iframes (e.g. a
      // link in a webmail message body, which a raw coordinate click may not activate) and follows
      // target=_blank into a new tab. Fall back to the coordinate click if the element isn't
      // actionable (covered/animating) so behavior never regresses for the main page.
      try {
        await target.locator.click({ button, timeout: 4000 });
      } catch (e) {
        await state.page.mouse.click(target.cx, target.cy, { button });
      }
    }
  } else if (kind === 'type') {
    loopGuard(`type:${body.ref}:${body.text}`);
    if (typeof body.text !== 'string') throw new Error("kind 'type' requires 'text'");
    target = await resolveRef(body.ref);
    // Record the placeholder (body.text); type the resolved real value. For non-secret text the
    // two are identical.
    actions = [
      { type: 'click', x: target.cx, y: target.cy, button: 'left' },
      { type: 'type', text: body.text },
    ];
    if (body.press_enter) actions.push({ type: 'keypress', keys: ['Enter'] });
    await recordStep(thought, actions, target);
    await state.page.mouse.click(target.cx, target.cy);
    // Clear any existing value the way a user would: select-all then retype.
    await state.page.keyboard.press('ControlOrMeta+a').catch(() => {});
    await state.page.keyboard.type(resolveSecrets(body.text), { delay: 15 });
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
    // Baseline AFTER the cursor reaches the drag origin: moving there can itself mutate the
    // DOM (hover menus collapsing/opening), which must not mask a no-op drag.
    await state.page.mouse.move(target.cx, target.cy);
    await state.page.waitForTimeout(200);
    const before = await domFingerprint();
    await state.page.mouse.down();
    await state.page.mouse.move(dest.cx, dest.cy, { steps: 12 });
    await state.page.mouse.up();
    await state.page.waitForTimeout(300);
    if ((await domFingerprint()) === before) {
      const fired = await html5DragFallback(target.locator, dest.locator).catch(() => false);
      if (fired) {
        state.task.meta.steps[String(state.task.steps.length)].drag_fallback = 'html5';
        writeTrajectory(false);
      }
    }
  } else if (kind === 'hover') {
    loopGuard(`hover:${body.ref}`);
    target = await resolveRef(body.ref);
    actions = [{ type: 'move', x: target.cx, y: target.cy }];
    await recordStep(thought, actions, target);
    await state.page.mouse.move(target.cx, target.cy);
    // Hover-revealed UI (menus, row buttons) typically appears after a short delay.
    await state.page.waitForTimeout(500);
  } else if (kind === 'click_xy') {
    // Coordinate-grounded click for surfaces with no DOM to reference (canvas, maps,
    // drawing apps). The agent supplies coordinates read from a screenshot.
    const x = Math.round(body.x), y = Math.round(body.y);
    requireInViewport(x, y, 'click_xy');
    loopGuard(`click_xy:${x},${y}`);
    const button = body.button === 'right' ? 'right' : 'left';
    actions = [{ type: 'click', x, y, button }];
    await recordStep(thought, actions, null);
    if (body.dblclick) await state.page.mouse.dblclick(x, y, { button });
    else await state.page.mouse.click(x, y, { button });
  } else if (kind === 'drag_path') {
    const path = Array.isArray(body.path) ? body.path.map((p) => ({ x: Math.round(p.x), y: Math.round(p.y) })) : [];
    if (path.length < 2) throw new Error("kind 'drag_path' requires a 'path' of at least 2 points");
    for (const p of path) requireInViewport(p.x, p.y, 'drag_path point');
    loopGuard(`drag_path:${path[0].x},${path[0].y}:${path[path.length - 1].x},${path[path.length - 1].y}`);
    actions = [{ type: 'drag', path }];
    await recordStep(thought, actions, null);
    await state.page.mouse.move(path[0].x, path[0].y);
    await state.page.mouse.down();
    for (const p of path.slice(1)) await state.page.mouse.move(p.x, p.y, { steps: 8 });
    await state.page.mouse.up();
  } else if (kind === 'move_xy') {
    const x = Math.round(body.x), y = Math.round(body.y);
    requireInViewport(x, y, 'move_xy');
    loopGuard(`move_xy:${x},${y}`);
    actions = [{ type: 'move', x, y }];
    await recordStep(thought, actions, null);
    await state.page.mouse.move(x, y);
    await state.page.waitForTimeout(500);
  } else if (kind === 'type_text') {
    // Coordinate-emitting agents type into whatever is focused; there is no ref to click first.
    if (typeof body.text !== 'string') throw new Error("kind 'type_text' requires 'text'");
    loopGuard(`type_text:${body.text}`);
    actions = [{ type: 'type', text: body.text }];
    if (body.press_enter) actions.push({ type: 'keypress', keys: ['Enter'] });
    await recordStep(thought, actions, null);
    await state.page.keyboard.type(resolveSecrets(body.text), { delay: 15 });
    if (body.press_enter) await state.page.keyboard.press('Enter');
  } else if (kind === 'scroll_xy') {
    const x = Math.round(body.x), y = Math.round(body.y);
    requireInViewport(x, y, 'scroll_xy');
    const dx = Number.isFinite(body.dx) ? Math.round(body.dx) : 0;
    const dy = Number.isFinite(body.dy) ? Math.round(body.dy) : 500;
    loopGuard(`scroll_xy:${x},${y}:${dx},${dy}:${await state.page.evaluate(() => Math.round(window.scrollY))}`);
    actions = [{ type: 'scroll', x, y, scroll_x: dx, scroll_y: dy }];
    await recordStep(thought, actions, null);
    await state.page.mouse.move(x, y);
    await state.page.mouse.wheel(dx, dy);
  } else if (kind === 'navigate') {
    // Recorded cross-site navigation: the agent deliberately goes somewhere — e.g. to its webmail
    // to read a verification / reset email mid-flow. Unlike the setup-only POST /navigate, this IS
    // a recorded step (the "before" screenshot is the page where the decision is made), so the
    // student learns that checking email is part of the flow. The destination url is the target.
    if (!body.url) throw new Error("kind 'navigate' requires a 'url'");
    loopGuard(`navigate:${body.url}`);
    actions = [{ type: 'navigate', url: String(body.url) }];
    await recordStep(thought, actions, null);
    await state.page.goto(String(body.url), { waitUntil: 'domcontentloaded', timeout: 30000 });
    await settle();
  } else if (kind === 'done') {
    // Terminal step: the screen state that justifies declaring the goal met, plus a
    // {type:"done"} action — the student must learn WHEN to stop, not just how to act.
    actions = [{ type: 'done' }];
    await recordStep(thought, actions, null);
    state.task.meta.done_recorded = true;
    return { ok: true, step: state.task.steps.length, url: state.page.url(), snapshot: '(task ended)' };
  } else if (kind === 'wait') {
    // An explicit, recorded wait: the student must learn WHEN waiting is the right action
    // (loading spinners, async updates), so it lands as a real step.
    const seconds = Math.min(Math.max(Number(body.seconds) || 1, 0.2), 10);
    loopGuard(`wait:${await state.page.evaluate(() => document.body ? document.body.innerText.length : 0)}`);
    actions = [{ type: 'wait' }];
    await recordStep(thought, actions, null);
    await state.page.waitForTimeout(seconds * 1000);
  } else if (kind === 'upload') {
    // Enabler, not supervision: programmatic file attachment has no pixel-action equivalent
    // a student could replay (a real click opens an OS dialog), so it is logged in meta only.
    target = await resolveRef(body.ref);
    const filename = String(body.filename || 'upload.txt').replace(/[^a-zA-Z0-9_.-]+/g, '_');
    const filePath = `/tmp/${filename}`;
    fs.writeFileSync(filePath, body.content !== undefined ? String(body.content) : 'sample upload content\n');
    await target.locator.setInputFiles(filePath);
    state.task.meta.uploads = state.task.meta.uploads || [];
    state.task.meta.uploads.push({ after_step: state.task.steps.length, ref: body.ref, filename });
    await settle();
    return { ok: true, step: state.task.steps.length, url: state.page.url(), snapshot: (await takeSnapshot()).slice(0, 8000), note: 'file attached (not recorded as a step)' };
  } else if (kind === 'select') {
    if (body.value === undefined) throw new Error("kind 'select' requires 'value'");
    loopGuard(`select:${body.ref}:${body.value}`);
    target = await resolveRef(body.ref);
    actions = [{ type: 'click', x: target.cx, y: target.cy, button: 'left' }];
    await recordStep(thought, actions, target);
    await target.locator.selectOption(String(body.value));
    state.task.meta.steps[String(state.task.steps.length)].select_value = String(body.value);
    writeTrajectory(false);
  } else if (kind === 'handoff') {
    // Guarded mode: the agent yields control for login / a fresh secret / CAPTCHA / 2FA. Recorded
    // as a student target; no page action (auth is provisioned out-of-band).
    actions = [{ type: 'handoff', reason: String(body.reason || 'login') }];
    await recordStep(thought, actions, null);
  } else if (kind === 'confirm') {
    // Guarded mode: the agent proposes a consequential action INSTEAD of executing it. Recorded as
    // the student target; the watchdog verdict lands next as an approval_event step. No page action.
    if (!body.proposed_action || typeof body.proposed_action !== 'object') {
      throw new Error("kind 'confirm' requires a 'proposed_action' object");
    }
    let proposed = body.proposed_action;
    // Resolve a proposed snapshot ref to pixel coordinates so the training target carries real
    // coords (<ref>(x,y)</ref>), matching every other recorded action. Best-effort: a stale ref
    // keeps the proposal as the agent gave it.
    if (proposed.ref !== undefined && proposed.ref !== null) {
      try {
        target = await resolveRef(proposed.ref);
        proposed = { ...proposed, x: target.cx, y: target.cy };
      } catch (e) { target = null; }
    }
    actions = [{ type: 'confirm', proposed_action: proposed, risk_category: String(body.risk_category || '') }];
    await recordStep(thought, actions, target);
  } else if (kind === 'approval_event') {
    // Host-internal observation (not an agent action): the watchdog verdict after a confirm. Empty
    // actions + an `event` payload; convert excludes it as an SFT target and threads it into history.
    await recordStep('', [], null, body.event || {});
    return { ok: true, step: state.task.steps.length, url: state.page.url(), snapshot: '(verdict recorded)' };
  } else if (kind === 'sequence') {
    const list = Array.isArray(body.actions) ? body.actions : [];
    if (!list.length) throw new Error("kind 'sequence' requires a non-empty 'actions' list");
    actions = list.map(normalizeContractAction);
    loopGuard(`sequence:${JSON.stringify(actions)}`);
    await recordStep(thought, actions, null);
    const outcome = await executeSequence(actions);
    if (actions.some((a) => a.type === 'done')) state.task.meta.done_recorded = true;
    await settle();
    return {
      ok: true, step: state.task.steps.length, url: state.page.url(),
      executed: outcome.executed, error_at: outcome.error_at, error: outcome.error,
      terminal: actions.some((a) => TERMINAL_TYPES.has(a.type)),
    };
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

async function mailPreLogin(cfg) {
  // Log the scenario mailbox into Roundcube (standard form: #rcmloginuser / #rcmloginpwd /
  // #rcmloginsubmit). GreenMail has auth disabled, so any password authenticates the address.
  // Best-effort: on any failure the agent can still drive the webmail login itself.
  try {
    await state.page.goto(cfg.url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await state.page.fill('#rcmloginuser', cfg.address, { timeout: 8000 }).catch(() => {});
    await state.page.fill('#rcmloginpwd', cfg.password || 'qa', { timeout: 8000 }).catch(() => {});
    await Promise.all([
      state.page.waitForLoadState('domcontentloaded').catch(() => {}),
      state.page.click('#rcmloginsubmit, button[type=submit], input[type=submit]', { timeout: 8000 }).catch(() => {}),
    ]);
    await state.page.waitForTimeout(1200);
  } catch (e) {
    console.error(`mail pre-login failed (continuing): ${e.message}`);
  }
}

async function handleSession(body) {
  if (!body.task_id || !body.start_url) throw new Error('/session requires task_id and start_url');
  await ensureBrowser();
  if (state.context) await state.context.close().catch(() => {});

  const contextOpts = { viewport: VIEWPORT, deviceScaleFactor: 1 };
  // pre_session auth mode: inject a saved (burner) storage_state so the agent starts logged in.
  if (body.auth_config && body.auth_config.storage_state) {
    contextOpts.storageState = body.auth_config.storage_state;
  }
  state.context = await state.browser.newContext(contextOpts);
  // Session-scoped burner credentials for masked typing (resolveSecrets); held in memory only,
  // never written to a trajectory. Reset every session.
  state.secrets = body.secrets || {};
  state.typeMode = body.type_mode === 'append' ? 'append' : 'replace';
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
    meta: {
      recorder: 'browser_driver',
      framework: 'agent',
      steps: {},
      navigations: [],
      task_meta: body.task_meta || {},
    },
    recentActions: [],
    pendingShot: null,
  };

  // Auth-flow capture: pre-authenticate the scenario webmail so the agent finds a ready inbox when
  // it navigates there to verify an email or reset a password. Non-recorded setup, like the goto
  // below; the resulting Roundcube session cookie lives in this context for the whole task.
  if (body.mail_config && body.mail_config.url && body.mail_config.address) {
    await mailPreLogin(body.mail_config);
    state.task.meta.mail_webmail_url = body.mail_config.url;
  }

  await state.page.goto(body.start_url, { waitUntil: 'domcontentloaded', timeout: 30000 });
  // SPAs (React/Vue/Vite/Next) render their UI only after hydration, which lands well after
  // `domcontentloaded` fires on the near-empty `<div id="root">` shell — snapshotting immediately
  // saw 0 interactive elements and the boot gate wrongly rejected a perfectly-good app. Poll the
  // snapshot until the hydrated UI appears (or give up), so the probe judges the RENDERED page.
  // (networkidle is deliberately avoided: dev servers hold an open HMR websocket, so it never fires.)
  let snapshot = '';
  for (let i = 0; i < 8; i++) {
    await settle();
    snapshot = await takeSnapshot();
    if ((snapshot.match(/^\[\d+\] /gm) || []).length >= 5) break;
    await state.page.waitForTimeout(800);
  }
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
  'POST /observe': handleObserve,
  'POST /finish': handleFinish,
  'POST /snapshot': async () => {
    if (!state.page) throw new Error('no active session');
    return { ok: true, url: state.page.url(), title: await state.page.title(), snapshot: (await takeSnapshot()).slice(0, 15000) };
  },
  // Whole-page rendered text: /snapshot carries only headings and control names, so a computed result is invisible to it; never part of an agent observation.
  'POST /page_text': async () => {
    if (!state.page) throw new Error('no active session');
    await scrubDevOverlays();
    const frames = await Promise.all(state.page.frames().map(async (frame) => {
      try {
        return await frame.evaluate(() => (document.body ? document.body.innerText : ''));
      } catch (e) {
        return '';
      }
    }));
    return { ok: true, url: state.page.url(), text: frames.join('\n').replace(/\n{3,}/g, '\n\n').slice(0, 40000) };
  },
  'POST /screenshot': async (body) => {
    if (!state.page) throw new Error('no active session');
    await scrubDevOverlays();
    // JPEG for the AGENT'S eyes only: image-heavy pages (map tiles, photo grids) produce
    // PNGs whose base64 exceeds the SDK's 1MB message buffer and kills the whole query.
    // Recorded training screenshots (recordStep) remain lossless PNG per the contract.
    // format:'png' is for student eval, whose observations must match the trained PNG inputs.
    const png = String((body && body.format) || 'jpeg').toLowerCase() === 'png';
    const buf = png
      ? await state.page.screenshot({ type: 'png' })
      : await state.page.screenshot({ type: 'jpeg', quality: 60 });
    return { ok: true, b64: buf.toString('base64'), mime: png ? 'image/png' : 'image/jpeg' };
  },
  'POST /affordances': async () => {
    if (!state.page) throw new Error('no active session');
    const counts = await state.page.evaluate(() => {
      const visible = (el) => {
        const r = el.getBoundingClientRect();
        return r.width > 2 && r.height > 2;
      };
      const count = (selector, min) => {
        let n = 0;
        document.querySelectorAll(selector).forEach((el) => {
          if (!visible(el)) return;
          if (min && (el.getBoundingClientRect().width < min || el.getBoundingClientRect().height < min)) return;
          n += 1;
        });
        return n;
      };
      return {
        draggables: count('[draggable="true"], [data-rbd-draggable-id], .react-draggable'),
        sliders: count('input[type="range"], [role="slider"]'),
        date_inputs: count('input[type="date"], input[type="datetime-local"], input[type="time"], input[type="month"], input[type="week"]'),
        color_inputs: count('input[type="color"]'),
        file_inputs: document.querySelectorAll('input[type="file"]').length,
        rich_text_editors: count('[contenteditable="true"], [role="textbox"][aria-multiline="true"]'),
        canvases: count('canvas', 100),
        multi_selects: count('select[multiple]'),
        autocompletes: count('[role="combobox"], [aria-autocomplete="list"]'),
        media_players: count('video, audio'),
        iframes: count('iframe'),
      };
    });
    return { ok: true, affordances: counts };
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
  'POST /storage_state': async () => {
    // Export the current context's cookies + localStorage + IndexedDB so a register/login
    // trajectory's authenticated session can be injected into later tasks (auth_mode=pre_session) —
    // letting most downstream tasks start already signed in instead of repeating a login. IndexedDB
    // is required for SPA apps (workflowy/appflowy/…) that keep their auth token there rather than in
    // cookies; without it the injected session is empty and every reuse task re-hits the login wall.
    // `{indexedDB:true}` needs Playwright >=1.51 (the sandbox image is pinned to 1.55); newContext
    // restores IndexedDB automatically when the storage_state carries it.
    if (!state.context) throw new Error('no active session');
    return { ok: true, storage_state: await state.context.storageState({ indexedDB: true }) };
  },
  'GET /status': async () => ({
    ok: true,
    ready: true,
    // ensureBrowser() returns early on a truthy state.browser, so a crashed Chromium stays
    // 'ready' forever; the pool needs the real connection state to recycle the slot.
    browser: !!(state.browser && state.browser.isConnected()),
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
