"""Drop-in conftest.py for instrumenting Python Selenium test repos.

Place this file in the test directory of a Selenium Python repo. Unlike the Playwright
conftest (which patches a fixture-provided `page`), Selenium repos build their own
WebDriver, so this module patches the Selenium classes themselves at import time —
`WebElement`, `Select`, and `WebDriver` — and captures a screenshot + the element's
viewport-centre coordinate before each interaction. Naming/saving is driven by global
pytest hooks (not an autouse fixture) so both pytest-function and unittest.TestCase
style tests are captured.

Coordinates come from getBoundingClientRect (viewport-relative CSS pixels, matching
Playwright's bounding_box). The viewport is locked to 1280x720 at deviceScaleFactor=1
via CDP so the recorded coordinates align with the captured screenshot pixels.

Environment variables:
    TRAJECTORY_DIR  — directory to write results (default: /workspace/results)
    SOURCE_REPO     — repo identifier for metadata
    TEST_NAME_PREFIX — prefix for task IDs (default: repo name)
"""
import json
import os
from pathlib import Path

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
_start_url = ""
_viewport_locked = set()

_RECT_JS = "var r = arguments[0].getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2];"


def _driver_of(element):
    return getattr(element, "_parent", None) or getattr(element, "parent", None)


def _lock_viewport(driver):
    """Pin the rendering viewport to 1280x720 at DSF=1 so coordinates align with the
    screenshot pixels. CDP sets the viewport (not the outer window) precisely; fall back
    to set_window_size for non-Chrome drivers."""
    key = id(driver)
    if key in _viewport_locked:
        return
    _viewport_locked.add(key)
    try:
        driver.execute_cdp_cmd(
            "Emulation.setDeviceMetricsOverride",
            {"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT, "deviceScaleFactor": 1, "mobile": False},
        )
    except Exception:
        try:
            driver.set_window_size(VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
        except Exception:
            pass


def _grab(driver, element):
    """Scroll the element to the viewport centre, then return (cx, cy, screenshot_bytes)
    with the coordinate and the screenshot taken from the same post-scroll state."""
    _lock_viewport(driver)
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element)
    except Exception:
        pass
    try:
        rect = driver.execute_script(_RECT_JS, element)
        cx, cy = int(rect[0]), int(rect[1])
    except Exception:
        cx, cy = 0, 0
    return cx, cy, driver.get_screenshot_as_png()


def _split_keys(args):
    """Split send_keys arguments into typed text and named special keys (Enter, Tab...)."""
    combined = "".join(str(a) for a in args)
    text = "".join(ch for ch in combined if ch not in _KEYS_BY_VALUE)
    keys = [_KEYS_BY_VALUE[ch] for ch in combined if ch in _KEYS_BY_VALUE]
    return text, keys


def _patch_selenium():
    """Monkey-patch the Selenium classes. Wrapped methods capture a screenshot + coords
    before delegating to the original, and never raise (instrumentation must not break a
    test). Returns False if Selenium is not importable (instrumentation no-ops)."""
    try:
        from selenium.webdriver.remote.webelement import WebElement
        from selenium.webdriver.remote.webdriver import WebDriver
        from selenium.webdriver.support.select import Select
        from selenium.webdriver.common.keys import Keys
    except Exception:
        return False

    global _KEYS_BY_VALUE
    _KEYS_BY_VALUE = {
        v: k for k, v in vars(Keys).items()
        if isinstance(v, str) and not k.startswith("_")
    }

    original_click = WebElement.click
    original_send_keys = WebElement.send_keys
    original_clear = WebElement.clear
    original_submit = WebElement.submit
    original_get = WebDriver.get

    def instrumented_click(self):
        driver = _driver_of(self)
        if driver is not None:
            try:
                cx, cy, shot = _grab(driver, self)
                _recorder.record("click", cx, cy, shot, button="left")
            except Exception:
                pass
        return original_click(self)

    def instrumented_send_keys(self, *value):
        driver = _driver_of(self)
        if driver is not None:
            try:
                cx, cy, shot = _grab(driver, self)
                text, keys = _split_keys(value)
                _recorder.record("click", cx, cy, shot)
                if text:
                    _recorder.record("type", cx, cy, shot, value=text)
                for key in keys:
                    _recorder.record("keypress", cx, cy, shot, value=key)
            except Exception:
                pass
        return original_send_keys(self, *value)

    def instrumented_clear(self):
        driver = _driver_of(self)
        if driver is not None:
            try:
                cx, cy, shot = _grab(driver, self)
                _recorder.record("click", cx, cy, shot)
            except Exception:
                pass
        return original_clear(self)

    def instrumented_submit(self):
        driver = _driver_of(self)
        if driver is not None:
            try:
                cx, cy, shot = _grab(driver, self)
                _recorder.record("keypress", cx, cy, shot, value="Enter")
            except Exception:
                pass
        return original_submit(self)

    def instrumented_get(self, url):
        global _start_url
        if not _start_url:
            _start_url = url
        _lock_viewport(self)
        return original_get(self, url)

    WebElement.click = instrumented_click
    WebElement.send_keys = instrumented_send_keys
    WebElement.clear = instrumented_clear
    WebElement.submit = instrumented_submit
    WebDriver.get = instrumented_get

    def _wrap_select(original):
        def wrapper(self, value):
            element = getattr(self, "_el", None)
            driver = _driver_of(element) if element is not None else None
            if driver is not None:
                try:
                    cx, cy, shot = _grab(driver, element)
                    _recorder.record("click", cx, cy, shot)
                except Exception:
                    pass
            return original(self, value)
        return wrapper

    Select.select_by_value = _wrap_select(Select.select_by_value)
    Select.select_by_visible_text = _wrap_select(Select.select_by_visible_text)
    Select.select_by_index = _wrap_select(Select.select_by_index)
    return True


_KEYS_BY_VALUE = {}
_patch_selenium()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Reset the recorder for each test before any driver fixtures run."""
    global _start_url
    _start_url = ""
    _recorder.set_test(item.nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        _recorder.save(start_url=_start_url, success=rep.passed)
