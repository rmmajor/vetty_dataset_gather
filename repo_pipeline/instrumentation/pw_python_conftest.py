"""Drop-in conftest.py for instrumenting Python Playwright test repos.

Place this file in the test directory of a Playwright Python repo. It
monkey-patches Page and Locator methods to capture screenshots and bounding
boxes before each interaction, recording the data into trajectory files.

Environment variables:
    TRAJECTORY_DIR  — directory to write results (default: /workspace/results)
    SOURCE_REPO     — repo identifier for metadata
    TEST_NAME_PREFIX — prefix for task IDs (default: repo name)
"""
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

TRAJECTORY_DIR = os.environ.get("TRAJECTORY_DIR", "/workspace/results")
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720


class StepRecorder:
    def __init__(self):
        self.steps = []
        self.step_counter = 0
        self.output_dir = None
        self.test_name = None

    def set_test(self, test_name: str):
        self.test_name = test_name
        safe_name = test_name.replace("/", "_").replace("::", "_").replace(" ", "_")
        prefix = os.environ.get("TEST_NAME_PREFIX", "repo")
        self.output_dir = Path(TRAJECTORY_DIR) / f"task_{prefix}_{safe_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps = []
        self.step_counter = 0

    def record(self, action_type, x, y, screenshot_bytes, selector="", value="", button="left"):
        self.step_counter += 1
        screenshot_filename = f"step_{self.step_counter:03d}_screenshot.png"
        screenshot_path = self.output_dir / screenshot_filename
        screenshot_path.write_bytes(screenshot_bytes)

        action = {"type": action_type}
        if action_type in ("click", "double_click"):
            action["x"] = x
            action["y"] = y
            if action_type == "click":
                action["button"] = button
        elif action_type == "type":
            action["text"] = value
        elif action_type == "keypress":
            action["keys"] = value if isinstance(value, list) else [value]
        elif action_type == "scroll":
            action["x"] = x
            action["y"] = y
            action["scroll_x"] = 0
            action["scroll_y"] = int(value) if value else 0

        self.steps.append({
            "step": self.step_counter,
            "screenshot": screenshot_filename,
            "thought": "",
            "actions": [action],
        })

    def save(self, start_url="", success=True):
        if not self.output_dir or not self.steps:
            return

        trajectory = {
            "task_id": self.test_name or "unknown",
            "instruction": "",
            "start_url": start_url,
            "steps": self.steps,
            "total_steps": len(self.steps),
            "success": success,
        }
        with open(self.output_dir / "trajectory.json", "w") as f:
            json.dump(trajectory, f, indent=2)

        metadata = {
            "task_id": self.test_name or "unknown",
            "instruction": "",
            "start_url": start_url,
            "domain": "",
            "website": "",
            "total_steps": len(self.steps),
            "success": success,
            "viewport": {"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            "source_repo": os.environ.get("SOURCE_REPO", ""),
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


_recorder = StepRecorder()


def _get_center(bounding_box):
    if not bounding_box:
        return 0, 0
    x = int(bounding_box["x"] + bounding_box["width"] / 2)
    y = int(bounding_box["y"] + bounding_box["height"] / 2)
    return x, y


def _patch_page_sync(page):
    """Patch synchronous Page methods."""

    original_click = page.click
    original_fill = page.fill
    original_press = page.press
    original_select_option = page.select_option
    original_check = page.check
    original_uncheck = page.uncheck
    original_dblclick = page.dblclick

    def instrumented_click(selector, **kwargs):
        try:
            element = page.query_selector(selector)
            box = element.bounding_box() if element else None
            cx, cy = _get_center(box)
            screenshot = page.screenshot(type="png")
            _recorder.record("click", cx, cy, screenshot, selector=selector, button=kwargs.get("button", "left"))
        except Exception:
            pass
        return original_click(selector, **kwargs)

    def instrumented_fill(selector, value, **kwargs):
        try:
            element = page.query_selector(selector)
            box = element.bounding_box() if element else None
            cx, cy = _get_center(box)
            screenshot = page.screenshot(type="png")
            _recorder.record("click", cx, cy, screenshot, selector=selector)
            _recorder.record("type", cx, cy, screenshot, value=value)
        except Exception:
            pass
        return original_fill(selector, value, **kwargs)

    def instrumented_press(selector, key, **kwargs):
        try:
            screenshot = page.screenshot(type="png")
            _recorder.record("keypress", 0, 0, screenshot, value=key)
        except Exception:
            pass
        return original_press(selector, key, **kwargs)

    def instrumented_select_option(selector, **kwargs):
        try:
            element = page.query_selector(selector)
            box = element.bounding_box() if element else None
            cx, cy = _get_center(box)
            screenshot = page.screenshot(type="png")
            _recorder.record("click", cx, cy, screenshot, selector=selector)
        except Exception:
            pass
        return original_select_option(selector, **kwargs)

    def instrumented_check(selector, **kwargs):
        try:
            element = page.query_selector(selector)
            box = element.bounding_box() if element else None
            cx, cy = _get_center(box)
            screenshot = page.screenshot(type="png")
            _recorder.record("click", cx, cy, screenshot, selector=selector)
        except Exception:
            pass
        return original_check(selector, **kwargs)

    def instrumented_uncheck(selector, **kwargs):
        try:
            element = page.query_selector(selector)
            box = element.bounding_box() if element else None
            cx, cy = _get_center(box)
            screenshot = page.screenshot(type="png")
            _recorder.record("click", cx, cy, screenshot, selector=selector)
        except Exception:
            pass
        return original_uncheck(selector, **kwargs)

    def instrumented_dblclick(selector, **kwargs):
        try:
            element = page.query_selector(selector)
            box = element.bounding_box() if element else None
            cx, cy = _get_center(box)
            screenshot = page.screenshot(type="png")
            _recorder.record("double_click", cx, cy, screenshot, selector=selector)
        except Exception:
            pass
        return original_dblclick(selector, **kwargs)

    page.click = instrumented_click
    page.fill = instrumented_fill
    page.press = instrumented_press
    page.select_option = instrumented_select_option
    page.check = instrumented_check
    page.uncheck = instrumented_uncheck
    page.dblclick = instrumented_dblclick


def _patch_locator_sync(page):
    """Patch the Locator class so locator.click(), locator.fill(), etc. are instrumented."""
    from playwright.sync_api import Locator

    original_locator_click = Locator.click
    original_locator_fill = Locator.fill
    original_locator_press = Locator.press
    original_locator_select_option = Locator.select_option
    original_locator_check = Locator.check
    original_locator_uncheck = Locator.uncheck
    original_locator_dblclick = Locator.dblclick

    def instrumented_click(self, **kwargs):
        try:
            box = self.bounding_box()
            cx, cy = _get_center(box)
            screenshot = self.page.screenshot(type="png")
            _recorder.record("click", cx, cy, screenshot, selector=str(self), button=kwargs.get("button", "left"))
        except Exception:
            pass
        return original_locator_click(self, **kwargs)

    def instrumented_fill(self, value, **kwargs):
        try:
            box = self.bounding_box()
            cx, cy = _get_center(box)
            screenshot = self.page.screenshot(type="png")
            _recorder.record("click", cx, cy, screenshot, selector=str(self))
            _recorder.record("type", cx, cy, screenshot, value=value)
        except Exception:
            pass
        return original_locator_fill(self, value, **kwargs)

    def instrumented_press(self, key, **kwargs):
        try:
            screenshot = self.page.screenshot(type="png")
            _recorder.record("keypress", 0, 0, screenshot, value=key)
        except Exception:
            pass
        return original_locator_press(self, key, **kwargs)

    def instrumented_select_option(self, **kwargs):
        try:
            box = self.bounding_box()
            cx, cy = _get_center(box)
            screenshot = self.page.screenshot(type="png")
            _recorder.record("click", cx, cy, screenshot, selector=str(self))
        except Exception:
            pass
        return original_locator_select_option(self, **kwargs)

    def instrumented_check(self, **kwargs):
        try:
            box = self.bounding_box()
            cx, cy = _get_center(box)
            screenshot = self.page.screenshot(type="png")
            _recorder.record("click", cx, cy, screenshot, selector=str(self))
        except Exception:
            pass
        return original_locator_check(self, **kwargs)

    def instrumented_uncheck(self, **kwargs):
        try:
            box = self.bounding_box()
            cx, cy = _get_center(box)
            screenshot = self.page.screenshot(type="png")
            _recorder.record("click", cx, cy, screenshot, selector=str(self))
        except Exception:
            pass
        return original_locator_uncheck(self, **kwargs)

    def instrumented_dblclick(self, **kwargs):
        try:
            box = self.bounding_box()
            cx, cy = _get_center(box)
            screenshot = self.page.screenshot(type="png")
            _recorder.record("double_click", cx, cy, screenshot, selector=str(self))
        except Exception:
            pass
        return original_locator_dblclick(self, **kwargs)

    Locator.click = instrumented_click
    Locator.fill = instrumented_fill
    Locator.press = instrumented_press
    Locator.select_option = instrumented_select_option
    Locator.check = instrumented_check
    Locator.uncheck = instrumented_uncheck
    Locator.dblclick = instrumented_dblclick


@pytest.fixture(autouse=True)
def _instrument_playwright(request, page):
    """Auto-use fixture that instruments every Playwright page."""
    test_name = request.node.nodeid
    _recorder.set_test(test_name)

    page.set_viewport_size({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})

    _patch_page_sync(page)
    _patch_locator_sync(page)

    start_url = ""
    original_goto = page.goto

    def capture_goto(url, **kwargs):
        nonlocal start_url
        if not start_url:
            start_url = url
        return original_goto(url, **kwargs)

    page.goto = capture_goto

    yield

    success = request.node.rep_call.passed if hasattr(request.node, "rep_call") else True
    _recorder.save(start_url=start_url, success=success)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
