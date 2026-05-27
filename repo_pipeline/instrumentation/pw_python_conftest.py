"""Drop-in conftest.py for instrumenting Python Playwright test repos.

Place this file in the test directory of a Playwright Python repo. It
monkey-patches Page and Locator methods to capture, before each interaction, a
screenshot + the target element's bounding box + a rich DOM probe (tag, role,
accessible name, attributes, visibility) + page state (url, scroll, console
errors), recording all of it into the trajectory files.

Three files are written per test (see save()):
    trajectory.json   — the minimal training contract + inline bbox/role/name
    capture_meta.json  — full per-step diagnostics, keyed by step number
    metadata.json      — trajectory-level fields (aggregates filled later in stage 6)

Environment variables:
    TRAJECTORY_DIR    — directory to write results (default: /workspace/results)
    SOURCE_REPO       — repo identifier for metadata
    TEST_NAME_PREFIX  — prefix for task IDs (default: repo)
    FRAMEWORK / FRAMEWORK_LANGUAGE — recorded into capture_meta.json
"""
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import pytest

TRAJECTORY_DIR = os.environ.get("TRAJECTORY_DIR", "/workspace/results")
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720

# Single-round-trip DOM probe run against the target element. Keep in sync with the
# equivalent constants in pw_ts_fixture.ts and selenium_python_conftest.py.
PROBE_JS = """(el) => {
  if (!el) return null;
  const cs = getComputedStyle(el);
  const r = el.getBoundingClientRect();
  const attr = (n) => el.getAttribute ? el.getAttribute(n) : null;
  let role = attr('role');
  if (!role) {
    const tag = (el.tagName || '').toLowerCase();
    const map = {a: 'link', button: 'button', input: 'textbox', textarea: 'textbox', select: 'combobox', img: 'img'};
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
    attrs: {type: attr('type'), placeholder: attr('placeholder'), 'aria-label': attr('aria-label'),
            title: attr('title'), href: attr('href'), value: (el.value !== undefined ? el.value : attr('value'))},
    visible: cs.visibility !== 'hidden' && cs.display !== 'none' && r.width > 0 && r.height > 0,
    enabled: !el.disabled,
    in_viewport: r.bottom > 0 && r.right > 0 && r.top < window.innerHeight && r.left < window.innerWidth,
    scroll: {x: Math.round(window.scrollX), y: Math.round(window.scrollY)},
    ready_state: document.readyState,
  };
}"""

PAGE_STATE_JS = """() => ({scroll: {x: Math.round(window.scrollX), y: Math.round(window.scrollY)},
                          ready_state: document.readyState})"""

# Console / page errors are counted globally and snapshotted per action as a delta.
_console = {"count": 0}


def _on_console(msg):
    try:
        if getattr(msg, "type", "") == "error":
            _console["count"] += 1
    except Exception:
        pass


def _on_pageerror(exc):
    try:
        _console["count"] += 1
    except Exception:
        pass


class StepRecorder:
    def __init__(self):
        self.steps = []
        self.meta_by_step = {}
        self.step_counter = 0
        self.output_dir = None
        self.test_name = None
        self._prev_sha = None
        self._last_time = None
        self._last_console = 0

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
        self._last_console = 0

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

        self.steps.append({
            "step": self.step_counter,
            "screenshot": screenshot_filename,
            "thought": "",
            "actions": [action],
        })

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
        delta = _console["count"] - self._last_console
        m["console_errors"] = delta if delta > 0 else 0
        self._last_console = _console["count"]
        m["selector"] = m.get("selector") or (str(selector) if selector else "")
        self.meta_by_step[self.step_counter] = m

    def save(self, start_url="", success=True):
        if not self.output_dir or not self.steps:
            return

        clean_steps = []
        for step in self.steps:
            m = self.meta_by_step.get(step["step"], {})
            clean_steps.append({
                "step": step["step"],
                "screenshot": step["screenshot"],
                "thought": step["thought"],
                "actions": step["actions"],
                "bbox": m.get("bbox"),
                "role": m.get("role"),
                "name": m.get("name"),
            })

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


def _png_dims(b):
    try:
        if len(b) >= 24 and b[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", b[16:24])
            return int(w), int(h)
    except Exception:
        pass
    return 0, 0


def _get_center(bounding_box):
    if not bounding_box:
        return 0, 0
    x = int(bounding_box["x"] + bounding_box["width"] / 2)
    y = int(bounding_box["y"] + bounding_box["height"] / 2)
    return x, y


def _box_dict(box):
    if not box:
        return None
    return {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}


def _safe_url(page):
    try:
        return page.url
    except Exception:
        return None


def _build_meta(page, evaluatable, selector, box):
    """Assemble the pre-action half of a step's capture metadata. url_after /
    caused_navigation / error are filled in _commit after the original action runs."""
    probe = {}
    if evaluatable is not None:
        try:
            probe = evaluatable.evaluate(PROBE_JS) or {}
        except Exception:
            probe = {}
    if not probe:
        try:
            probe = page.evaluate(PAGE_STATE_JS) or {}
        except Exception:
            probe = {}
    return {
        "selector": str(selector) if selector else "",
        "tag": probe.get("tag"),
        "role": probe.get("role"),
        "name": probe.get("name"),
        "attrs": probe.get("attrs") or {},
        "bbox": _box_dict(box),
        "is_visible": probe.get("visible"),
        "is_enabled": probe.get("enabled"),
        "in_viewport": probe.get("in_viewport"),
        "scroll_position": probe.get("scroll") or {"x": 0, "y": 0},
        "wait_state": probe.get("ready_state"),
        "url_before": _safe_url(page),
        "url_after": None,
        "caused_navigation": False,
        "dom_changed": None,
        "error": None,
        "console_errors": 0,
    }


def _prepare(page, element, selector):
    try:
        box = element.bounding_box() if element is not None else None
        cx, cy = _get_center(box)
        meta = _build_meta(page, element, selector, box)
        screenshot = page.screenshot(type="png")
        return {"screenshot": screenshot, "cx": cx, "cy": cy, "meta": meta, "selector": str(selector) if selector else ""}
    except Exception:
        return None


def _prepare_locator(loc, with_box=True):
    try:
        box = loc.bounding_box() if with_box else None
        cx, cy = _get_center(box)
        meta = _build_meta(loc.page, loc, str(loc), box)
        screenshot = loc.page.screenshot(type="png")
        return {"screenshot": screenshot, "cx": cx, "cy": cy, "meta": meta, "selector": str(loc)}
    except Exception:
        return None


def _finalize_outcome(meta, page, err):
    try:
        meta["url_after"] = _safe_url(page)
    except Exception:
        pass
    ub, ua = meta.get("url_before"), meta.get("url_after")
    meta["caused_navigation"] = bool(ub and ua and ub != ua)
    if err:
        meta["error"] = err


def _commit(pending, page, action_type, err, selector="", value="", button="left"):
    if not pending:
        return
    meta = pending["meta"]
    _finalize_outcome(meta, page, err)
    try:
        _recorder.record(action_type, pending["cx"], pending["cy"], pending["screenshot"],
                         selector=pending["selector"] or selector, value=value, button=button, meta=meta)
    except Exception:
        pass


def _commit_fill(pending, page, value, err):
    if not pending:
        return
    meta = pending["meta"]
    _finalize_outcome(meta, page, err)
    try:
        _recorder.record("click", pending["cx"], pending["cy"], pending["screenshot"],
                         selector=pending["selector"], meta=dict(meta))
        _recorder.record("type", pending["cx"], pending["cy"], pending["screenshot"],
                         value=value, meta=dict(meta))
    except Exception:
        pass


# (method name, recorded action type) for the click-like family that share one shape.
_CLICK_LIKE = [("click", "click"), ("select_option", "click"), ("check", "click"),
               ("uncheck", "click"), ("dblclick", "double_click")]


def _patch_page_sync(page):
    """Patch synchronous Page methods (page.click(selector), page.fill(...), ...)."""

    def make(method_name, action_type):
        original = getattr(page, method_name)

        def wrapper(selector, *args, **kwargs):
            pending = _prepare(page, _safe_query(page, selector), selector)
            err = None
            try:
                return original(selector, *args, **kwargs)
            except Exception as e:
                err = repr(e)[:200]
                raise
            finally:
                _commit(pending, page, action_type, err, selector=selector, button=kwargs.get("button", "left"))

        return wrapper

    for method_name, action_type in _CLICK_LIKE:
        if hasattr(page, method_name):
            setattr(page, method_name, make(method_name, action_type))

    original_fill = page.fill

    def instrumented_fill(selector, value, **kwargs):
        pending = _prepare(page, _safe_query(page, selector), selector)
        err = None
        try:
            return original_fill(selector, value, **kwargs)
        except Exception as e:
            err = repr(e)[:200]
            raise
        finally:
            _commit_fill(pending, page, value, err)

    page.fill = instrumented_fill

    original_press = page.press

    def instrumented_press(selector, key, **kwargs):
        pending = _prepare(page, _safe_query(page, selector), selector)
        err = None
        try:
            return original_press(selector, key, **kwargs)
        except Exception as e:
            err = repr(e)[:200]
            raise
        finally:
            _commit(pending, page, "keypress", err, selector=selector, value=key)

    page.press = instrumented_press


def _safe_query(page, selector):
    try:
        return page.query_selector(selector)
    except Exception:
        return None


def _patch_locator_sync(page):
    """Patch the Locator class so locator.click(), locator.fill(), etc. are instrumented."""
    from playwright.sync_api import Locator

    def make(method_name, action_type):
        original = getattr(Locator, method_name)

        def wrapper(self, *args, **kwargs):
            pending = _prepare_locator(self)
            err = None
            try:
                return original(self, *args, **kwargs)
            except Exception as e:
                err = repr(e)[:200]
                raise
            finally:
                _commit(pending, self.page, action_type, err, button=kwargs.get("button", "left"))

        return wrapper

    for method_name, action_type in _CLICK_LIKE:
        setattr(Locator, method_name, make(method_name, action_type))

    original_fill = Locator.fill

    def instrumented_fill(self, value, **kwargs):
        pending = _prepare_locator(self)
        err = None
        try:
            return original_fill(self, value, **kwargs)
        except Exception as e:
            err = repr(e)[:200]
            raise
        finally:
            _commit_fill(pending, self.page, value, err)

    Locator.fill = instrumented_fill

    original_press = Locator.press

    def instrumented_press(self, key, **kwargs):
        pending = _prepare_locator(self, with_box=False)
        err = None
        try:
            return original_press(self, key, **kwargs)
        except Exception as e:
            err = repr(e)[:200]
            raise
        finally:
            _commit(pending, self.page, "keypress", err, value=key)

    Locator.press = instrumented_press


@pytest.fixture(autouse=True)
def _instrument_playwright(request, page):
    """Auto-use fixture that instruments every Playwright page."""
    test_name = request.node.nodeid
    _recorder.set_test(test_name)

    page.set_viewport_size({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})

    _console["count"] = 0
    try:
        page.on("console", _on_console)
        page.on("pageerror", _on_pageerror)
    except Exception:
        pass

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
