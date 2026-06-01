/**
 * Drop-in Playwright fixture for TypeScript/JavaScript repos.
 *
 * Usage: imports of `@playwright/test` in test files are redirected here in code
 * (see instrumentation/__init__._wire_ts_fixture). This module re-exports everything
 * from @playwright/test (drop-in superset) and overrides the built-in `page` fixture
 * so every test is instrumented automatically.
 *
 * Before each interaction it captures a screenshot + the target element's bounding box
 * + a DOM probe (tag, role, accessible name, attributes, visibility) + page state (url,
 * scroll, console errors), and writes three files per test (see StepRecorder.save):
 *   trajectory.json   — the minimal training contract + inline bbox/role/name
 *   capture_meta.json  — full per-step diagnostics keyed by step number
 *   metadata.json      — trajectory-level fields (aggregates filled later in stage 6)
 *
 * Environment variables:
 *   TRAJECTORY_DIR    — output directory (default: /workspace/results)
 *   SOURCE_REPO       — repo identifier for metadata
 *   TEST_NAME_PREFIX  — prefix for task IDs
 *   FRAMEWORK / FRAMEWORK_LANGUAGE — recorded into capture_meta.json
 */

import { test as base, Page, Locator, BoundingBox } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

const TRAJECTORY_DIR = process.env.TRAJECTORY_DIR || '/workspace/results';
const VIEWPORT_WIDTH = 1280;
const VIEWPORT_HEIGHT = 720;
const TEST_SOURCE_MAX_CHARS = 2000;
const TEST_SOURCE_MAX_LINES = 60;

// Single-round-trip DOM probe run against the target element. Keep in sync with the
// equivalent PROBE_JS constants in pw_python_conftest.py and selenium_python_conftest.py.
const PROBE_FN = (el: any) => {
  if (!el) return null;
  const cs = getComputedStyle(el);
  const r = el.getBoundingClientRect();
  const attr = (n: string) => (el.getAttribute ? el.getAttribute(n) : null);
  let role = attr('role');
  if (!role) {
    const tag = (el.tagName || '').toLowerCase();
    const map: Record<string, string> = { a: 'link', button: 'button', input: 'textbox', textarea: 'textbox', select: 'combobox', img: 'img' };
    role = map[tag] || tag || null;
    if (tag === 'input') {
      const t = (attr('type') || '').toLowerCase();
      if (t === 'checkbox') role = 'checkbox';
      else if (t === 'radio') role = 'radio';
      else if (t === 'submit' || t === 'button') role = 'button';
    }
  }
  let name = attr('aria-label') || (el.textContent || '').trim();
  if (name) name = name.slice(0, 120);
  return {
    tag: (el.tagName || '').toLowerCase(),
    role: role || null,
    name: name || null,
    attrs: {
      type: attr('type'), placeholder: attr('placeholder'), 'aria-label': attr('aria-label'),
      title: attr('title'), href: attr('href'), value: el.value !== undefined ? el.value : attr('value'),
    },
    visible: cs.visibility !== 'hidden' && cs.display !== 'none' && r.width > 0 && r.height > 0,
    enabled: !el.disabled,
    in_viewport: r.bottom > 0 && r.right > 0 && r.top < window.innerHeight && r.left < window.innerWidth,
    scroll: { x: Math.round(window.scrollX), y: Math.round(window.scrollY) },
    ready_state: document.readyState,
  };
};

interface Step {
  step: number;
  screenshot: string;
  thought: string;
  actions: Record<string, unknown>[];
}

interface Pending {
  screenshot: Buffer;
  cx: number;
  cy: number;
  meta: Record<string, any>;
}

// Console / page errors counted globally, snapshotted per action as a delta.
let consoleErrorCount = 0;

function sha256(buf: Buffer): string {
  return crypto.createHash('sha256').update(buf).digest('hex');
}

function pngDims(buf: Buffer): [number, number] {
  try {
    if (buf.length >= 24 && buf[0] === 0x89 && buf[1] === 0x50) {
      return [buf.readUInt32BE(16), buf.readUInt32BE(20)];
    }
  } catch {}
  return [0, 0];
}

class StepRecorder {
  steps: Step[] = [];
  metaByStep: Record<number, Record<string, any>> = {};
  stepCounter = 0;
  outputDir = '';
  testName = '';
  prevSha: string | null = null;
  lastTime: number | null = null;
  lastConsole = 0;

  setTest(testName: string) {
    this.testName = testName;
    const safeName = testName.replace(/[/\\:]/g, '_').replace(/\s+/g, '_');
    const prefix = process.env.TEST_NAME_PREFIX || 'repo';
    this.outputDir = path.join(TRAJECTORY_DIR, `task_${prefix}_${safeName}`);
    fs.mkdirSync(this.outputDir, { recursive: true });
    this.steps = [];
    this.metaByStep = {};
    this.stepCounter = 0;
    this.prevSha = null;
    this.lastTime = null;
    this.lastConsole = 0;
  }

  record(
    actionType: string,
    x: number,
    y: number,
    screenshotBuffer: Buffer,
    extra: Record<string, unknown> = {}
  ) {
    this.stepCounter++;
    const screenshotFilename = `step_${String(this.stepCounter).padStart(3, '0')}_screenshot.png`;
    fs.writeFileSync(path.join(this.outputDir, screenshotFilename), screenshotBuffer);

    const action: Record<string, unknown> = { type: actionType };
    if (actionType === 'click' || actionType === 'double_click') {
      action.x = x;
      action.y = y;
      if (actionType === 'click') {
        action.button = (extra.button as string) || 'left';
      }
    } else if (actionType === 'type') {
      action.text = extra.value || '';
    } else if (actionType === 'keypress') {
      action.keys = extra.keys || [];
    } else if (actionType === 'scroll') {
      action.x = x;
      action.y = y;
      action.scroll_x = 0;
      action.scroll_y = extra.scrollY || 0;
    }

    this.steps.push({
      step: this.stepCounter,
      screenshot: screenshotFilename,
      thought: '',
      actions: [action],
    });

    const m: Record<string, any> = { ...((extra.meta as Record<string, any>) || {}) };
    const sha = sha256(screenshotBuffer);
    if (m.dom_changed === undefined || m.dom_changed === null) {
      m.dom_changed = this.prevSha ? sha !== this.prevSha : null;
    }
    this.prevSha = sha;
    const [w, h] = pngDims(screenshotBuffer);
    m.screenshot_sha = sha;
    m.screenshot_width = w;
    m.screenshot_height = h;
    const now = Date.now();
    m.dt_ms = this.lastTime ? now - this.lastTime : 0;
    this.lastTime = now;
    const delta = consoleErrorCount - this.lastConsole;
    m.console_errors = delta > 0 ? delta : 0;
    this.lastConsole = consoleErrorCount;
    this.metaByStep[this.stepCounter] = m;
  }

  save(startUrl: string, success: boolean) {
    if (!this.outputDir || this.steps.length === 0) return;

    const cleanSteps = this.steps.map((s) => {
      const m = this.metaByStep[s.step] || {};
      return {
        step: s.step,
        screenshot: s.screenshot,
        thought: s.thought,
        actions: s.actions,
        bbox: m.bbox ?? null,
        role: m.role ?? null,
        name: m.name ?? null,
      };
    });

    const trajectory = {
      task_id: this.testName,
      instruction: '',
      start_url: startUrl,
      steps: cleanSteps,
      total_steps: cleanSteps.length,
      success,
    };
    fs.writeFileSync(path.join(this.outputDir, 'trajectory.json'), JSON.stringify(trajectory, null, 2));

    const captureSteps: Record<string, any> = {};
    for (const s of this.steps) captureSteps[String(s.step)] = this.metaByStep[s.step] || {};
    const captureMeta = {
      schema_version: 1,
      task_id: this.testName,
      framework: process.env.FRAMEWORK || '',
      framework_language: process.env.FRAMEWORK_LANGUAGE || '',
      steps: captureSteps,
    };
    fs.writeFileSync(path.join(this.outputDir, 'capture_meta.json'), JSON.stringify(captureMeta, null, 2));

    const metadata = {
      task_id: this.testName,
      instruction: '',
      start_url: startUrl,
      domain: '',
      website: '',
      total_steps: cleanSteps.length,
      success,
      viewport: { width: VIEWPORT_WIDTH, height: VIEWPORT_HEIGHT },
      source_repo: process.env.SOURCE_REPO || '',
      framework: process.env.FRAMEWORK || '',
      framework_language: process.env.FRAMEWORK_LANGUAGE || '',
    };
    fs.writeFileSync(path.join(this.outputDir, 'metadata.json'), JSON.stringify(metadata, null, 2));
  }
}

function getCenter(box: BoundingBox | null): [number, number] {
  if (!box) return [0, 0];
  return [Math.round(box.x + box.width / 2), Math.round(box.y + box.height / 2)];
}

function safeUrl(loc: Locator): string | null {
  try {
    return loc.page().url();
  } catch {
    return null;
  }
}

async function buildMeta(loc: Locator, box: BoundingBox | null): Promise<Record<string, any>> {
  let probe: any = {};
  try {
    probe = (await loc.evaluate(PROBE_FN)) || {};
  } catch {}
  return {
    selector: String(loc),
    tag: probe.tag ?? null,
    role: probe.role ?? null,
    name: probe.name ?? null,
    attrs: probe.attrs || {},
    bbox: box ? { x: box.x, y: box.y, width: box.width, height: box.height } : null,
    is_visible: probe.visible ?? null,
    is_enabled: probe.enabled ?? null,
    in_viewport: probe.in_viewport ?? null,
    scroll_position: probe.scroll || { x: 0, y: 0 },
    wait_state: probe.ready_state ?? null,
    url_before: safeUrl(loc),
    url_after: null,
    caused_navigation: false,
    dom_changed: null,
    error: null,
    console_errors: 0,
  };
}

function finalizeOutcome(meta: Record<string, any>, loc: Locator, err: string | null) {
  meta.url_after = safeUrl(loc);
  meta.caused_navigation = !!(meta.url_before && meta.url_after && meta.url_before !== meta.url_after);
  if (err) meta.error = err;
}

// The active test's recorder. Set per test in the `page` fixture; the (global) Locator
// prototype patch records into it. Playwright runs tests serially within a worker process,
// so a single module-level recorder is safe.
let currentRecorder: StepRecorder | null = null;
let locatorPrototypePatched = false;

async function prepareLocator(loc: Locator, withBox = true): Promise<Pending | null> {
  if (!currentRecorder) return null;
  try {
    const box = withBox ? await loc.boundingBox() : null;
    const [cx, cy] = getCenter(box);
    const meta = await buildMeta(loc, box);
    const screenshot = await loc.page().screenshot({ type: 'png' });
    return { screenshot, cx, cy, meta };
  } catch {
    return null;
  }
}

function commitRecord(type: string, ctx: Pending, extra: Record<string, unknown> = {}) {
  if (!currentRecorder) return;
  currentRecorder.record(type, ctx.cx, ctx.cy, ctx.screenshot, { ...extra, meta: ctx.meta });
}

// Patch the Locator prototype ONCE so locator-based actions in any test are instrumented.
// We obtain the prototype from a live locator's constructor (the runtime class isn't a value
// export of @playwright/test). Mirrors the Python conftest's Locator-class patching.
function patchLocatorPrototype(sample: Locator) {
  const proto: any = Object.getPrototypeOf(sample);
  if (!proto || proto.__trajInstrumented) return;
  proto.__trajInstrumented = true;

  // Run `prep` before the original action (capture before-state), then the original wrapped
  // so we can record the outcome (url_after / error), then `commit` to write the step(s).
  const patch = (
    method: string,
    prep: (loc: Locator, args: any[]) => Promise<Pending | null>,
    commit: (loc: Locator, ctx: Pending | null, args: any[], err: string | null) => void
  ) => {
    const orig = proto[method];
    if (typeof orig !== 'function') return;
    proto[method] = async function (this: Locator, ...args: any[]) {
      const ctx = await prep(this, args);
      let err: string | null = null;
      try {
        return await orig.apply(this, args);
      } catch (e) {
        err = String(e).slice(0, 200);
        throw e;
      } finally {
        try {
          commit(this, ctx, args, err);
        } catch {}
      }
    };
  };

  const clickLike: Record<string, string> = {
    click: 'click', dblclick: 'double_click', tap: 'click',
    check: 'click', uncheck: 'click', setChecked: 'click', selectOption: 'click',
  };
  for (const [method, type] of Object.entries(clickLike)) {
    patch(
      method,
      (loc) => prepareLocator(loc, true),
      (loc, ctx, args, err) => {
        if (!ctx) return;
        finalizeOutcome(ctx.meta, loc, err);
        if (ctx.cx === 0 && ctx.cy === 0) return; // not visible / no box → no usable coords
        commitRecord(type, ctx, { button: args?.[0]?.button });
      }
    );
  }

  patch(
    'fill',
    (loc) => prepareLocator(loc, true),
    (loc, ctx, args, err) => {
      if (!ctx) return;
      finalizeOutcome(ctx.meta, loc, err);
      if (ctx.cx === 0 && ctx.cy === 0) return;
      commitRecord('click', ctx);
      commitRecord('type', ctx, { value: args[0] });
    }
  );

  for (const method of ['type', 'pressSequentially']) {
    patch(
      method,
      (loc) => prepareLocator(loc, true),
      (loc, ctx, args, err) => {
        if (!ctx) return;
        finalizeOutcome(ctx.meta, loc, err);
        if (ctx.cx === 0 && ctx.cy === 0) return;
        commitRecord('type', ctx, { value: args[0] });
      }
    );
  }

  patch(
    'press',
    (loc) => prepareLocator(loc, false),
    (loc, ctx, args, err) => {
      if (!ctx) return;
      finalizeOutcome(ctx.meta, loc, err);
      commitRecord('keypress', ctx, { keys: [args[0]] });
    }
  );
}

function captureTestSource(outputDir: string, testFile: string, testLine: number): void {
  // Persist a slice of the test file's source (test body + surrounding lines) next to the
  // trajectory so the stage-6 instruction generator can infer intent from it instead of
  // just the function name. Best-effort: any failure is swallowed — older trajectories
  // simply have no test_source.txt and the instruction prompt has an explicit fallback.
  if (!outputDir || !testFile) return;
  try {
    const fileText = fs.readFileSync(testFile, 'utf-8');
    const lines = fileText.split('\n');
    const startIdx = Math.max(0, (testLine || 1) - 1);
    const endIdx = Math.min(lines.length, startIdx + TEST_SOURCE_MAX_LINES);
    let slice = lines.slice(startIdx, endIdx).join('\n');
    if (slice.length > TEST_SOURCE_MAX_CHARS) {
      slice = slice.slice(0, TEST_SOURCE_MAX_CHARS) + '\n... (truncated)';
    }
    fs.writeFileSync(path.join(outputDir, 'test_source.txt'), slice);
  } catch {}
}


export const test = base.extend({
  page: async ({ page }, use, testInfo) => {
    const recorder = new StepRecorder();
    recorder.setTest(testInfo.titlePath.join(' > '));
    captureTestSource(recorder.outputDir, testInfo.file, testInfo.line);
    currentRecorder = recorder;

    await page.setViewportSize({ width: VIEWPORT_WIDTH, height: VIEWPORT_HEIGHT });

    consoleErrorCount = 0;
    try {
      page.on('console', (msg) => {
        if (msg.type() === 'error') consoleErrorCount++;
      });
      page.on('pageerror', () => {
        consoleErrorCount++;
      });
    } catch {}

    if (!locatorPrototypePatched) {
      patchLocatorPrototype(page.locator('body'));
      locatorPrototypePatched = true;
    }

    let startUrl = '';
    const originalGoto = page.goto.bind(page);
    page.goto = async (url: string, options?: any) => {
      if (!startUrl) startUrl = url;
      return originalGoto(url, options);
    };

    // Legacy page-level action API (page.click(selector), page.fill(...), ...) — these go
    // through the frame channel, not the Locator class, so they need their own patches.
    // Record via a locator built from the selector, around the original page call.
    const patchPageAction = (method: string, type: string) => {
      const fn = (page as any)[method];
      if (typeof fn !== 'function') return;
      const original = fn.bind(page);
      (page as any)[method] = async (selector: string, ...rest: any[]) => {
        const ctx = await prepareLocator(page.locator(selector).first(), true);
        let err: string | null = null;
        try {
          return await original(selector, ...rest);
        } catch (e) {
          err = String(e).slice(0, 200);
          throw e;
        } finally {
          if (ctx) {
            finalizeOutcome(ctx.meta, page.locator(selector).first(), err);
            if (!(ctx.cx === 0 && ctx.cy === 0)) commitRecord(type, ctx, { button: rest?.[0]?.button });
          }
        }
      };
    };
    patchPageAction('click', 'click');
    patchPageAction('dblclick', 'double_click');
    patchPageAction('check', 'click');
    patchPageAction('uncheck', 'click');
    patchPageAction('selectOption', 'click');
    if (typeof page.fill === 'function') {
      const originalPageFill = page.fill.bind(page);
      (page as any).fill = async (selector: string, value: string, options?: any) => {
        const ctx = await prepareLocator(page.locator(selector).first(), true);
        let err: string | null = null;
        try {
          return await originalPageFill(selector, value, options);
        } catch (e) {
          err = String(e).slice(0, 200);
          throw e;
        } finally {
          if (ctx && !(ctx.cx === 0 && ctx.cy === 0)) {
            finalizeOutcome(ctx.meta, page.locator(selector).first(), err);
            commitRecord('click', ctx);
            commitRecord('type', ctx, { value });
          }
        }
      };
    }

    await use(page);

    const success = testInfo.status === 'passed';
    recorder.save(startUrl, success);
    currentRecorder = null;
  },
});

// Re-export everything from @playwright/test so this module is a drop-in superset:
// redirecting a test's `@playwright/test` import here preserves expect/devices/types,
// while the `test` above (instrumented page override) shadows the original.
export * from '@playwright/test';
