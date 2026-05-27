/**
 * Drop-in Playwright fixture for TypeScript/JavaScript repos.
 *
 * Usage: imports of `@playwright/test` in test files are redirected here in code
 * (see instrumentation/__init__._wire_ts_fixture). This module re-exports everything
 * from @playwright/test (drop-in superset) and overrides the built-in `page` fixture
 * so every test is instrumented automatically.
 *
 * It captures BOTH calling styles by patching the Locator prototype (modern API:
 * page.getByRole(...).click(), page.locator(...).fill(), ...) AND the legacy page API
 * (page.click(selector), page.fill(selector, value), ...). Patching the prototype keeps
 * locators as real objects so Playwright's expect() still recognises them (a Proxy wrapper
 * does not).
 *
 * Environment variables:
 *   TRAJECTORY_DIR    — output directory (default: /workspace/results)
 *   SOURCE_REPO       — repo identifier for metadata
 *   TEST_NAME_PREFIX  — prefix for task IDs
 */

import { test as base, Page, Locator, BoundingBox } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const TRAJECTORY_DIR = process.env.TRAJECTORY_DIR || '/workspace/results';
const VIEWPORT_WIDTH = 1280;
const VIEWPORT_HEIGHT = 720;

interface Step {
  step: number;
  screenshot: string;
  thought: string;
  actions: Record<string, unknown>[];
}

class StepRecorder {
  steps: Step[] = [];
  stepCounter = 0;
  outputDir = '';
  testName = '';

  setTest(testName: string) {
    this.testName = testName;
    const safeName = testName.replace(/[/\\:]/g, '_').replace(/\s+/g, '_');
    const prefix = process.env.TEST_NAME_PREFIX || 'repo';
    this.outputDir = path.join(TRAJECTORY_DIR, `task_${prefix}_${safeName}`);
    fs.mkdirSync(this.outputDir, { recursive: true });
    this.steps = [];
    this.stepCounter = 0;
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
  }

  save(startUrl: string, success: boolean) {
    if (!this.outputDir || this.steps.length === 0) return;

    const trajectory = {
      task_id: this.testName,
      instruction: '',
      start_url: startUrl,
      steps: this.steps,
      total_steps: this.steps.length,
      success,
    };

    fs.writeFileSync(
      path.join(this.outputDir, 'trajectory.json'),
      JSON.stringify(trajectory, null, 2)
    );

    const metadata = {
      task_id: this.testName,
      instruction: '',
      start_url: startUrl,
      domain: '',
      website: '',
      total_steps: this.steps.length,
      success,
      viewport: { width: VIEWPORT_WIDTH, height: VIEWPORT_HEIGHT },
      source_repo: process.env.SOURCE_REPO || '',
    };

    fs.writeFileSync(
      path.join(this.outputDir, 'metadata.json'),
      JSON.stringify(metadata, null, 2)
    );
  }
}

function getCenter(box: BoundingBox | null): [number, number] {
  if (!box) return [0, 0];
  return [Math.round(box.x + box.width / 2), Math.round(box.y + box.height / 2)];
}

// The active test's recorder. Set per test in the `page` fixture; the (global) Locator
// prototype patch records into it. Playwright runs tests serially within a worker process,
// so a single module-level recorder is safe.
let currentRecorder: StepRecorder | null = null;
let locatorPrototypePatched = false;

// Record an action performed on a specific locator: screenshot + center of its bounding box.
async function recordLocatorAction(loc: Locator, type: string, extra: Record<string, unknown> = {}) {
  if (!currentRecorder) return;
  try {
    if (type === 'keypress') {
      const screenshot = await loc.page().screenshot({ type: 'png' });
      currentRecorder.record('keypress', 0, 0, screenshot, extra);
      return;
    }
    const box = await loc.boundingBox();
    const [cx, cy] = getCenter(box);
    if (cx === 0 && cy === 0) return; // not visible / no box → skip (training needs real coords)
    const screenshot = await loc.page().screenshot({ type: 'png' });
    currentRecorder.record(type, cx, cy, screenshot, extra);
  } catch {}
}

// Patch the Locator prototype ONCE so locator-based actions in any test are instrumented.
// We obtain the prototype from a live locator's constructor (the runtime class isn't a value
// export of @playwright/test). Mirrors the Python conftest's Locator-class patching.
function patchLocatorPrototype(sample: Locator) {
  const proto: any = Object.getPrototypeOf(sample);
  if (!proto || proto.__trajInstrumented) return;
  proto.__trajInstrumented = true;

  const patch = (method: string, record: (self: Locator, args: any[]) => Promise<void>) => {
    const orig = proto[method];
    if (typeof orig !== 'function') return;
    proto[method] = async function (this: Locator, ...args: any[]) {
      await record(this, args);
      return orig.apply(this, args);
    };
  };

  const clickLike: Record<string, string> = {
    click: 'click', dblclick: 'double_click', tap: 'click',
    check: 'click', uncheck: 'click', setChecked: 'click', selectOption: 'click',
  };
  for (const [m, type] of Object.entries(clickLike)) {
    patch(m, async (self, args) => recordLocatorAction(self, type, { button: args?.[0]?.button }));
  }
  patch('fill', async (self, args) => {
    await recordLocatorAction(self, 'click');
    await recordLocatorAction(self, 'type', { value: args[0] });
  });
  patch('type', async (self, args) => recordLocatorAction(self, 'type', { value: args[0] }));
  patch('pressSequentially', async (self, args) => recordLocatorAction(self, 'type', { value: args[0] }));
  patch('press', async (self, args) => recordLocatorAction(self, 'keypress', { keys: [args[0]] }));
}

export const test = base.extend({
  page: async ({ page }, use, testInfo) => {
    const recorder = new StepRecorder();
    recorder.setTest(testInfo.titlePath.join(' > '));
    currentRecorder = recorder;

    await page.setViewportSize({ width: VIEWPORT_WIDTH, height: VIEWPORT_HEIGHT });

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
    const patchPageAction = (method: string, type: string) => {
      const fn = (page as any)[method];
      if (typeof fn !== 'function') return;
      const original = fn.bind(page);
      (page as any)[method] = async (selector: string, ...rest: any[]) => {
        await recordLocatorAction(page.locator(selector).first(), type, { button: rest?.[0]?.button });
        return original(selector, ...rest);
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
        await recordLocatorAction(page.locator(selector).first(), 'click');
        await recordLocatorAction(page.locator(selector).first(), 'type', { value });
        return originalPageFill(selector, value, options);
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
