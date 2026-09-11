"""Drop-in conftest.py for instrumenting Python Selenium test repos.

Place this file in the test directory of a Selenium Python repo. Unlike the Playwright
conftest (which patches a fixture-provided `page`), Selenium repos build their own
WebDriver, so this module patches the Selenium classes themselves at import time —
`WebElement`, `Select`, and `WebDriver` — and captures a screenshot + the element's
viewport-centre coordinate + a DOM probe (tag, role, accessible name, attributes,
visibility) + page state before each interaction. Naming/saving is driven by global
pytest hooks (not an autouse fixture) so both pytest-function and unittest.TestCase
style tests are captured.

Coordinates and bbox come from getBoundingClientRect (viewport-relative CSS pixels,
matching Playwright's bounding_box). The viewport is locked to 1280x720 at
deviceScaleFactor=1 via CDP so the recorded coordinates align with the screenshot pixels.

Three files are written per test (see save()): trajectory.json (training contract +
inline bbox/role/name), capture_meta.json (full per-step diagnostics), metadata.json.

Environment variables:
    TRAJECTORY_DIR  — directory to write results (default: /workspace/results)
    SOURCE_REPO     — repo identifier for metadata
    TEST_NAME_PREFIX — prefix for task IDs (default: repo)
    FRAMEWORK / FRAMEWORK_LANGUAGE — recorded into capture_meta.json
"""

import hashlib
import inspect
import json
import os
import struct
import time
from pathlib import Path

import pytest

TRAJECTORY_DIR = os.environ.get("TRAJECTORY_DIR", "/workspace/results")
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720
TEST_SOURCE_MAX_CHARS = 2000

# DOM probe run against the target element via execute_script. Returns bbox too, since
# Selenium has no boundingBox() method. Keep in sync with PROBE_JS in pw_python_conftest.py.
_PROBE_JS = """
var el = arguments[0];
if (!el) { return null; }
var cs = getComputedStyle(el);
var r = el.getBoundingClientRect();
function attr(n){ return el.getAttribute ? el.getAttribute(n) : null; }
var role = attr('role');
if (!role) {
  var tag = (el.tagName || '').toLowerCase();
  var map = {a:'link', button:'button', input:'textbox', textarea:'textbox', select:'combobox', img:'img'};
  role = map[tag] || tag || null;
  if (tag === 'input') {
    var t = (attr('type') || '').toLowerCase();
    if (t === 'checkbox') role = 'checkbox';
    else if (t === 'radio') role = 'radio';
    else if (t === 'submit' || t === 'button') role = 'button';
  }
}
var name = attr('aria-label') || (el.textContent || '').trim();
if (name) { name = name.slice(0, 120); }
return {
  tag: (el.tagName || '').toLowerCase(), role: role || null, name: name || null,
  attrs: {type: attr('type'), placeholder: attr('placeholder'), 'aria-label': attr('aria-label'),
          title: attr('title'), href: attr('href'), value: (el.value !== undefined ? el.value : attr('value'))},
  visible: cs.visibility !== 'hidden' && cs.display !== 'none' && r.width > 0 && r.height > 0,
  enabled: !el.disabled,
  in_viewport: r.bottom > 0 && r.right > 0 && r.top < window.innerHeight && r.left < window.innerWidth,
  bbox: {x: r.left, y: r.top, width: r.width, height: r.height},
  scroll: {x: Math.round(window.scrollX), y: Math.round(window.scrollY)},
  ready_state: document.readyState
};
"""


def _png_dims(b):
    try:
        if len(b) >= 24 and b[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", b[16:24])
            return int(w), int(h)
    except Exception:
        pass
    return 0, 0


class StepRecorder:
    def __init__(self):
        self.steps = []
        self.meta_by_step = {}
        self.step_counter = 0
        self.output_dir = None
        self.test_name = None
        self._prev_sha = None
        self._last_time = None

    def set_test(self, test_name: str):
        self.test_name = test_name
        safe_name = test_name.replace("/", "_").replace("::", "_").replace(" ", "_")
        prefix = os.environ.get("TEST_NAME_PREFIX", "repo")
        self.output_dir = Path(TRAJECTORY_DIR) / f"task_{prefix}_{safe_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps = []
        self.meta_by_step = {}
        self.step_counter = 0
        self._prev_sha = None
        self._last_time = None

    def record(self, action_type, x, y, screenshot_bytes, selector="", value="", button="left", meta=None):
        self.step_counter += 1
        screenshot_filename = f"step_{self.step_counter:03d}_screenshot.png"
        (self.output_dir / screenshot_filename).write_bytes(screenshot_bytes)

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

        self.steps.append(
            {
                "step": self.step_counter,
                "screenshot": screenshot_filename,
                "thought": "",
                "actions": [action],
            }
        )

        m = dict(meta) if meta else {}
        sha = hashlib.sha256(screenshot_bytes).hexdigest()
        if m.get("dom_changed") is None:
            m["dom_changed"] = (sha != self._prev_sha) if self._prev_sha else None
        self._prev_sha = sha
        w, h = _png_dims(screenshot_bytes)
        m["screenshot_sha"] = sha
        m["screenshot_width"] = w
        m["screenshot_height"] = h
        now = time.monotonic()
        m["dt_ms"] = int((now - self._last_time) * 1000) if self._last_time else 0
        self._last_time = now
        m["selector"] = m.get("selector") or (str(selector) if selector else "")
        self.meta_by_step[self.step_counter] = m

    def save(self, start_url="", success=True):
        if not self.output_dir or not self.steps:
            return

        clean_steps = []
        for step in self.steps:
            m = self.meta_by_step.get(step["step"], {})
            clean_steps.append(
                {
                    "step": step["step"],
                    "screenshot": step["screenshot"],
                    "thought": step["thought"],
                    "actions": step["actions"],
                    "bbox": m.get("bbox"),
                    "role": m.get("role"),
                    "name": m.get("name"),
                }
            )

        trajectory = {
            "task_id": self.test_name or "unknown",
            "instruction": "",
            "start_url": start_url,
            "steps": clean_steps,
            "total_steps": len(clean_steps),
            "success": success,
        }
        with open(self.output_dir / "trajectory.json", "w") as f:
            json.dump(trajectory, f, indent=2)

        capture_meta = {
            "schema_version": 1,
            "task_id": self.test_name or "unknown",
            "framework": os.environ.get("FRAMEWORK", ""),
            "framework_language": os.environ.get("FRAMEWORK_LANGUAGE", ""),
            "steps": {str(step["step"]): self.meta_by_step.get(step["step"], {}) for step in self.steps},
        }
        with open(self.output_dir / "capture_meta.json", "w") as f:
            json.dump(capture_meta, f, indent=2)

        metadata = {
            "task_id": self.test_name or "unknown",
            "instruction": "",
            "start_url": start_url,
            "domain": "",
            "website": "",
            "total_steps": len(clean_steps),
            "success": success,
            "viewport": {"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            "source_repo": os.environ.get("SOURCE_REPO", ""),
            "framework": os.environ.get("FRAMEWORK", ""),
            "framework_language": os.environ.get("FRAMEWORK_LANGUAGE", ""),
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


_recorder = StepRecorder()
_start_url = ""
_viewport_locked = set()

_RECT_JS = "var r = arguments[0].getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2];"


def _driver_of(element):
    return getattr(element, "_parent", None) or getattr(element, "parent", None)


def _safe_url(driver):
    try:
        return driver.current_url
    except Exception:
        return None


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


def _build_meta(driver, probe, bbox):
    return {
        "selector": "",
        "tag": probe.get("tag"),
        "role": probe.get("role"),
        "name": probe.get("name"),
        "attrs": probe.get("attrs") or {},
        "bbox": ({"x": bbox["x"], "y": bbox["y"], "width": bbox["width"], "height": bbox["height"]} if bbox else None),
        "is_visible": probe.get("visible"),
        "is_enabled": probe.get("enabled"),
        "in_viewport": probe.get("in_viewport"),
        "scroll_position": probe.get("scroll") or {"x": 0, "y": 0},
        "wait_state": probe.get("ready_state"),
        "url_before": _safe_url(driver),
        "url_after": None,
        "caused_navigation": False,
        "dom_changed": None,
        "error": None,
        "console_errors": 0,
    }


def _prepare(driver, element):
    """Scroll the element to the viewport centre, probe it, then take the screenshot from
    the same post-scroll state. Returns None on any failure (instrumentation must not raise)."""
    try:
        _lock_viewport(driver)
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element)
        except Exception:
            pass
        probe = {}
        try:
            probe = driver.execute_script(_PROBE_JS, element) or {}
        except Exception:
            probe = {}
        bbox = probe.get("bbox")
        cx = cy = 0
        if bbox:
            cx = int(bbox["x"] + bbox["width"] / 2)
            cy = int(bbox["y"] + bbox["height"] / 2)
        if cx == 0 and cy == 0:
            # bbox missing or degenerate — fall back to a direct rect probe before giving up.
            try:
                rect = driver.execute_script(_RECT_JS, element)
                cx, cy = int(rect[0]), int(rect[1])
            except Exception:
                cx = cy = 0
        if cx == 0 and cy == 0:
            # No usable on-screen coordinate. Skip the action entirely rather than record a (0,0)
            # step — mirrors the Playwright TS hook and keeps invalid points out of the trajectory,
            # the main cause of Selenium's low capture yield.
            return None
        meta = _build_meta(driver, probe, bbox)
        shot = driver.get_screenshot_as_png()
        return {"cx": cx, "cy": cy, "shot": shot, "meta": meta}
    except Exception:
        return None


def _finalize_outcome(meta, driver, err):
    meta["url_after"] = _safe_url(driver)
    ub, ua = meta.get("url_before"), meta.get("url_after")
    meta["caused_navigation"] = bool(ub and ua and ub != ua)
    if err:
        meta["error"] = err


def _commit(pending, driver, action_type, err, value="", button="left"):
    if not pending:
        return
    meta = pending["meta"]
    _finalize_outcome(meta, driver, err)
    try:
        _recorder.record(
            action_type, pending["cx"], pending["cy"], pending["shot"], value=value, button=button, meta=meta
        )
    except Exception:
        pass


def _commit_send_keys(pending, driver, value_args, err):
    if not pending:
        return
    meta = pending["meta"]
    _finalize_outcome(meta, driver, err)
    text, keys = _split_keys(value_args)
    try:
        _recorder.record("click", pending["cx"], pending["cy"], pending["shot"], meta=dict(meta))
        if text:
            _recorder.record("type", pending["cx"], pending["cy"], pending["shot"], value=text, meta=dict(meta))
        for key in keys:
            _recorder.record("keypress", pending["cx"], pending["cy"], pending["shot"], value=key, meta=dict(meta))
    except Exception:
        pass


def _split_keys(args):
    """Split send_keys arguments into typed text and named special keys (Enter, Tab...)."""
    combined = "".join(str(a) for a in args)
    text = "".join(ch for ch in combined if ch not in _KEYS_BY_VALUE)
    keys = [_KEYS_BY_VALUE[ch] for ch in combined if ch in _KEYS_BY_VALUE]
    return text, keys


def _patch_selenium():
    """Monkey-patch the Selenium classes. Wrapped methods capture before-state, run the
    original (recording its outcome), and never raise (instrumentation must not break a
    test). Returns False if Selenium is not importable (instrumentation no-ops)."""
    try:
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.remote.webdriver import WebDriver
        from selenium.webdriver.remote.webelement import WebElement
        from selenium.webdriver.support.select import Select
    except Exception:
        return False

    global _KEYS_BY_VALUE
    _KEYS_BY_VALUE = {v: k for k, v in vars(Keys).items() if isinstance(v, str) and not k.startswith("_")}

    original_click = WebElement.click
    original_send_keys = WebElement.send_keys
    original_clear = WebElement.clear
    original_submit = WebElement.submit
    original_get = WebDriver.get

    def instrumented_click(self):
        driver = _driver_of(self)
        pending = _prepare(driver, self) if driver is not None else None
        err = None
        try:
            return original_click(self)
        except Exception as e:
            err = repr(e)[:200]
            raise
        finally:
            _commit(pending, driver, "click", err, button="left")

    def instrumented_send_keys(self, *value):
        driver = _driver_of(self)
        pending = _prepare(driver, self) if driver is not None else None
        err = None
        try:
            return original_send_keys(self, *value)
        except Exception as e:
            err = repr(e)[:200]
            raise
        finally:
            _commit_send_keys(pending, driver, value, err)

    def instrumented_clear(self):
        driver = _driver_of(self)
        pending = _prepare(driver, self) if driver is not None else None
        err = None
        try:
            return original_clear(self)
        except Exception as e:
            err = repr(e)[:200]
            raise
        finally:
            _commit(pending, driver, "click", err)

    def instrumented_submit(self):
        driver = _driver_of(self)
        pending = _prepare(driver, self) if driver is not None else None
        err = None
        try:
            return original_submit(self)
        except Exception as e:
            err = repr(e)[:200]
            raise
        finally:
            _commit(pending, driver, "keypress", err, value="Enter")

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
            pending = _prepare(driver, element) if driver is not None else None
            err = None
            try:
                return original(self, value)
            except Exception as e:
                err = repr(e)[:200]
                raise
            finally:
                _commit(pending, driver, "click", err)

        return wrapper

    Select.select_by_value = _wrap_select(Select.select_by_value)
    Select.select_by_visible_text = _wrap_select(Select.select_by_visible_text)
    Select.select_by_index = _wrap_select(Select.select_by_index)
    return True


_KEYS_BY_VALUE = {}
_patch_selenium()


def _capture_test_source(item, output_dir):
    """Persist the test function's source (body + docstring) next to the trajectory so the
    stage-6 instruction generator can infer intent from it instead of just the function name.
    Best-effort: any failure is swallowed — older trajectories simply have no test_source.txt
    and the instruction prompt has an explicit fallback for that case."""
    if not output_dir:
        return
    try:
        func = getattr(item, "function", None) or getattr(item, "obj", None)
        if func is None:
            return
        source = inspect.getsource(func)
    except Exception:
        return
    if len(source) > TEST_SOURCE_MAX_CHARS:
        source = source[:TEST_SOURCE_MAX_CHARS] + "\n... (truncated)"
    try:
        (output_dir / "test_source.txt").write_text(source)
    except Exception:
        pass


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Reset the recorder for each test before any driver fixtures run."""
    global _start_url
    _start_url = ""
    _recorder.set_test(item.nodeid)
    _capture_test_source(item, _recorder.output_dir)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        _recorder.save(start_url=_start_url, success=rep.passed)
