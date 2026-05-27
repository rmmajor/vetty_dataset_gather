/**
 * Drop-in Playwright fixture for TypeScript/JavaScript repos.
 *
 * Usage: Import this fixture instead of @playwright/test in test files,
 * or configure playwright.config.ts to use this as a global setup.
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

export const test = base.extend({
  page: async ({ page }, use, testInfo) => {
    const recorder = new StepRecorder();
    recorder.setTest(testInfo.titlePath.join(' > '));

    await page.setViewportSize({ width: VIEWPORT_WIDTH, height: VIEWPORT_HEIGHT });

    let startUrl = '';

    const originalGoto = page.goto.bind(page);
    page.goto = async (url: string, options?: any) => {
      if (!startUrl) startUrl = url;
      return originalGoto(url, options);
    };

    const originalClick = page.click.bind(page);
    page.click = async (selector: string, options?: any) => {
      try {
        const element = page.locator(selector).first();
        const box = await element.boundingBox();
        const [cx, cy] = getCenter(box);
        const screenshot = await page.screenshot({ type: 'png' });
        recorder.record('click', cx, cy, screenshot, {
          button: options?.button || 'left',
        });
      } catch {}
      return originalClick(selector, options);
    };

    const originalFill = page.fill.bind(page);
    page.fill = async (selector: string, value: string, options?: any) => {
      try {
        const element = page.locator(selector).first();
        const box = await element.boundingBox();
        const [cx, cy] = getCenter(box);
        const screenshot = await page.screenshot({ type: 'png' });
        recorder.record('click', cx, cy, screenshot);
        recorder.record('type', cx, cy, screenshot, { value });
      } catch {}
      return originalFill(selector, value, options);
    };

    const originalPress = page.press.bind(page);
    page.press = async (selector: string, key: string, options?: any) => {
      try {
        const screenshot = await page.screenshot({ type: 'png' });
        recorder.record('keypress', 0, 0, screenshot, { keys: [key] });
      } catch {}
      return originalPress(selector, key, options);
    };

    const originalSelectOption = page.selectOption.bind(page);
    page.selectOption = async (selector: string, values: any, options?: any) => {
      try {
        const element = page.locator(selector).first();
        const box = await element.boundingBox();
        const [cx, cy] = getCenter(box);
        const screenshot = await page.screenshot({ type: 'png' });
        recorder.record('click', cx, cy, screenshot);
      } catch {}
      return originalSelectOption(selector, values, options);
    };

    const originalDblclick = page.dblclick.bind(page);
    page.dblclick = async (selector: string, options?: any) => {
      try {
        const element = page.locator(selector).first();
        const box = await element.boundingBox();
        const [cx, cy] = getCenter(box);
        const screenshot = await page.screenshot({ type: 'png' });
        recorder.record('double_click', cx, cy, screenshot);
      } catch {}
      return originalDblclick(selector, options);
    };

    const originalCheck = page.check.bind(page);
    page.check = async (selector: string, options?: any) => {
      try {
        const element = page.locator(selector).first();
        const box = await element.boundingBox();
        const [cx, cy] = getCenter(box);
        const screenshot = await page.screenshot({ type: 'png' });
        recorder.record('click', cx, cy, screenshot);
      } catch {}
      return originalCheck(selector, options);
    };

    const originalUncheck = page.uncheck.bind(page);
    page.uncheck = async (selector: string, options?: any) => {
      try {
        const element = page.locator(selector).first();
        const box = await element.boundingBox();
        const [cx, cy] = getCenter(box);
        const screenshot = await page.screenshot({ type: 'png' });
        recorder.record('click', cx, cy, screenshot);
      } catch {}
      return originalUncheck(selector, options);
    };

    await use(page);

    const success = testInfo.status === 'passed';
    recorder.save(startUrl, success);
  },
});

// Re-export everything from @playwright/test so this module is a drop-in superset:
// redirecting a test's `@playwright/test` import here preserves expect/devices/types,
// while the `test` above (instrumented page override) shadows the original.
export * from '@playwright/test';
