# RL Harness Phase 1 (Mac-side rollout harness) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A policy-agnostic rollout harness (`cua/rl/`) that runs the Qwen student against live public sites through the existing sandboxed browser driver, records full trajectory dirs, scores them (programmatic check → Claude verifier), and exposes reset/step/close over HTTP for the GRPO trainer plus `rollout`/`eval` CLIs.

**Architecture:** Every episode = `reset` (driver `/session` + `/observe`) → policy text → `cua.rl.actions.decode_step` → driver `/act kind=sequence` (one recorded step per policy turn, observation PNG bytes == recorded PNG) → repeat until `done`/cap → `copy_out` → `success_check` or `verify_trajectory` → scalar reward. Observations are built by the same `cua/sft/legacy_train_qwen.py` functions that build SFT examples, so training input == RL input byte-for-byte. `cua/rl/service.py` wraps the same `Episode` for the GPU box; `cua/rl/rollout.py` / `cua/rl/eval.py` drive it locally.

**Tech Stack:** Python 3.11 stdlib (`http.server`, `urllib`, `threading`), existing `repo_pipeline` (`DockerSandbox`, `driver_session.DriverClient`, `agent_capture.verify_trajectory`, `app_secrets`, `corpus_audit`), Node driver `browser_driver.js` (Playwright 1.55 in the sandbox image), no new third-party deps.

**Spec:** `docs/superpowers/specs/2026-08-26-rl-harness-design.md` (Phase 1 section; the contract block is normative).

## Global Constraints

- Comments/docstrings: ONE line each, only where the code cannot say it. Never reflow existing comments.
- No nested imports. No imports inside test functions (module-level only).
- Registries/config over name checks (`BACKENDS`, `ACTION_DECODERS`, `CHECKS`, `REWARD_COMPONENTS`, `POLICIES`, `TASK_SOURCES`).
- **Never `git commit` / `git push`.** This worktree currently holds the *uncommitted* PR #24 merge in the index; Phase 1 edits must NOT be `git add`ed until the Phase 0 merge commits land. Each task ends with "report the diff", not "commit".
- Rollouts write under `rl_runs/` / `eval_runs/`, never `trajectories/` (`cua.sft.legacy_train_qwen convert` globs `trajectories/**/task_*`).
- Claude calls only through `cua.pipeline.agent_runner.run_agent` (OAuth `claude` CLI); never set `ANTHROPIC_API_KEY`.
- Viewport 1280x720; action dicts must equal the contract shapes in `cua/pipeline/schemas.py` / `cua.sft.legacy_train_qwen.format_action_for_training`.
- Interpreter for all runs: `/Users/rmajor/unik/maga/dataset_gather/.venv-pipeline/bin/python` from the worktree root (call it `$PY` below). Docker Desktop must be running for Tasks 3 and 13.
- Env knobs are `RL_*` env vars with defaults in code; no hardcoded site names anywhere.

---

## File map

| File | Responsibility |
|---|---|
| `cua/sft/legacy_train_qwen.py` (modify) | Extract `build_step_user_text`, `history_entries_for_step`, `build_step_messages`; read `excluded_reason` from either metadata file |
| `cua/rl/__init__.py` | empty |
| `cua/rl/actions.py` | decoder registry + `decode_step` / `encode_step` |
| `cua/rl/observation.py` | `build_observation`, `to_openai_messages` |
| `cua/rl/executor.py` | decoded actions → driver `sequence` payload, key aliases, viewport validation, control detection |
| `cua/pipeline/instrumentation/browser_driver.js` (modify) | `/observe`, `sequence` act kind, pending-screenshot handoff, `type_mode` |
| `cua/pipeline/driver_session.py` (modify) | per-port `start_driver`, `DriverClient(port)`, `observe`, `act_sequence`, `session(type_mode)` |
| `cua/pipeline/driver_smoke.py` (modify) | `/observe` + `sequence` leg |
| `cua/rl/checks.py` | `CHECKS` registry, `build_context`, `evaluate` |
| `cua/rl/rewards.py` + `cua/rl/reward_config.json` | components/penalties registries, `compute_reward`, `load_reward_config` |
| `cua/rl/judge.py` | `AsyncBridge`, `judge_success` (check → verifier, deferred on usage limit) |
| `cua/rl/tasks.py` + `cua/rl/rl_tasks.json` | suite builder / checker / lister, `TASK_SOURCES` |
| `cua/rl/backends.py` | `BACKENDS`, `resolve_preconditions`, `RolloutWorker`, `WorkerPool` |
| `cua/rl/episode.py` | `EpisodeConfig`, `RunContext`, `Episode` |
| `cua/rl/policies.py` | `POLICIES`: openai / mock / scripted |
| `cua/rl/runner.py` | `run_episode`, `run_batch`, `load_tasks`, `make_run_context` |
| `cua/rl/rollout.py` | rollout CLI → `rl_runs/<run_id>/iter_NNN/rollouts.jsonl` |
| `cua/rl/eval.py` | eval CLI → `eval_runs/<run_id>/report.{md,json}` |
| `cua/rl/service.py` | HTTP env service |
| `cua/rl/smoke.py` | Docker-only end-to-end smoke (`make test-rl`) |
| `tests/__main__.py`, `tests/test_*.py` | dependency-free unit runner + tests |
| `cua/pipeline/stage6_generate.py`, `cua/pipeline/corpus_audit.py` (modify) | `excluded_reason` written to both files / read from either |
| `Makefile`, `.gitignore`, `CLAUDE.md` (modify) | targets, ignores, docs |

---

### Task 1: Shared student-input builders in `cua/sft/legacy_train_qwen.py` + unit-test runner

**Files:**
- Modify: `cua/sft/legacy_train_qwen.py` (`convert_trajectory_to_examples`, add three functions after `HISTORY_WINDOW`)
- Create: `cua/rl/__init__.py`, `tests/__init__.py`, `tests/__main__.py`, `tests/test_observation.py`

**Interfaces:**
- Produces: `build_step_user_text(instruction: str, action_history: list[str]) -> str`; `history_entries_for_step(step: dict) -> list[str]`; `build_step_messages(system: str, user_text: str, image_ref: str) -> list[dict]` (all in `cua.sft.legacy_train_qwen`).

- [ ] **Step 1: Create the test runner**

`cua/rl/__init__.py` and `tests/__init__.py`: empty files.

`tests/__main__.py`:
```python
"""Dependency-free runner: `python -m tests [module_substring ...]` runs every test_* function."""
import importlib
import pkgutil
import sys
import traceback

import tests


def main(filters):
    total = failed = 0
    for info in pkgutil.iter_modules(tests.__path__):
        if not info.name.startswith("test_") or (filters and not any(f in info.name for f in filters)):
            continue
        module = importlib.import_module(f"tests.{info.name}")
        for name in sorted(dir(module)):
            if not name.startswith("test_") or not callable(getattr(module, name)):
                continue
            total += 1
            try:
                getattr(module, name)()
                print(f"ok    {info.name}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL  {info.name}.{name}")
                traceback.print_exc()
    print(f"{total - failed}/{total} passed")
    return 1 if failed or not total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 2: Write the failing tests**

`tests/test_observation.py`:
```python
import json
import tempfile
from pathlib import Path

from PIL import Image

from cua.sft.legacy_train_qwen import (HISTORY_WINDOW, build_step_messages, build_step_user_text,
                        convert_trajectory_to_examples, history_entries_for_step)


def _png(path):
    Image.new("RGB", (4, 4), (255, 255, 255)).save(path)


def test_user_text_without_history():
    assert build_step_user_text("Do it", []) == "Task: Do it\nWhat action should you take?"


def test_user_text_windowed_with_omitted_prefix():
    history = [f"click({i}, {i})" for i in range(HISTORY_WINDOW + 3)]
    text = build_step_user_text("Do it", history)
    assert "... 3 earlier actions, " in text
    assert "click(0, 0)" not in text and f"click({HISTORY_WINDOW + 2}, {HISTORY_WINDOW + 2})" in text
    assert text.endswith("]\nWhat action should you take?")


def test_history_entries_include_watchdog_event():
    step = {"actions": [{"type": "click", "x": 1, "y": 2}], "event": {"kind": "approval", "decision": "approve"}}
    assert history_entries_for_step(step) == ["click(1, 2)", "[watchdog: approve]"]


def test_history_entries_legacy_single_action_key():
    assert history_entries_for_step({"action": {"type": "wait"}}) == ["wait"]


def test_build_step_messages_shape():
    msgs = build_step_messages("SYS", "Task: x", "/tmp/a.png")
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1]["content"] == [{"type": "image", "image": "/tmp/a.png"}, {"type": "text", "text": "Task: x"}]


def test_convert_uses_shared_builders():
    steps = [
        {"step": 1, "screenshot": "step_001_screenshot.png", "thought": "a", "actions": [{"type": "click", "x": 1, "y": 2, "button": "left"}]},
        {"step": 2, "screenshot": "step_002_screenshot.png", "thought": "b", "actions": [{"type": "type", "text": "hi"}, {"type": "keypress", "keys": ["Enter"]}]},
        {"step": 3, "screenshot": "step_003_screenshot.png", "thought": "c", "actions": [{"type": "done"}]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for s in steps:
            _png(d / s["screenshot"])
        examples, _ = convert_trajectory_to_examples({"instruction": "Say hi", "steps": steps}, d)
    history = []
    for example, step in zip(examples, steps):
        assert example["messages"][1]["content"][1]["text"] == build_step_user_text("Say hi", history)
        history.extend(history_entries_for_step(step))
    assert json.dumps(examples[0]["messages"][0]["content"]).startswith('"You are a GUI agent')
```

- [ ] **Step 3: Run to verify failure**

Run: `$PY -m tests test_observation`
Expected: FAIL with `ImportError: cannot import name 'build_step_messages'`.

- [ ] **Step 4: Implement the builders and use them in convert**

In `cua/sft/legacy_train_qwen.py`, insert after the `HISTORY_WINDOW = 10` block (before the waste-filter comment):
```python
def build_step_user_text(instruction, action_history):
    """The student's user turn: task + windowed text history; shared by convert and the RL env."""
    user_text = f"Task: {instruction}"
    if action_history:
        window = action_history[-HISTORY_WINDOW:]
        omitted = len(action_history) - len(window)
        prefix = f"... {omitted} earlier actions, " if omitted > 0 else ""
        user_text += f"\nPrevious actions: [{prefix}{', '.join(window)}]"
    user_text += "\nWhat action should you take?"
    return user_text


def history_entries_for_step(step):
    actions = step.get("actions", [])
    if not actions and "action" in step:
        actions = [step["action"]]
    entries = [format_action_summary(a) for a in actions]
    event = step.get("event") or {}
    if event.get("kind") == "approval":
        entries.append(f"[watchdog: {event.get('decision', '?')}]")
    return entries


def build_step_messages(system, user_text, image_ref):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image", "image": image_ref},
            {"type": "text", "text": user_text},
        ]},
    ]
```

In `convert_trajectory_to_examples`, replace the body of the `for i, step in enumerate(steps):` loop from `elif emit:` through the end of the loop with:
```python
        elif emit:
            user_text = build_step_user_text(instruction, action_history)
            action_str = "\n".join(format_action_for_training(a) for a in actions)
            assistant_content = f"<think>{thought}</think>\n{action_str}" if thought else action_str
            examples.append({"messages": [
                *build_step_messages(SYSTEM_PROMPT, user_text, str(screenshot_path)),
                {"role": "assistant", "content": assistant_content},
            ]})

        action_history.extend(history_entries_for_step(step))
```
Delete the now-unused inline `Previous actions` block and the trailing `for a in actions: action_history.append(...)` + watchdog-event lines (they moved into `history_entries_for_step`; keep the two-line comment above it by moving it onto `history_entries_for_step`? No — the comment restates the code, delete it).

- [ ] **Step 5: Run tests + byte-identical convert check**

Run: `$PY -m tests test_observation` → Expected: `6/6 passed`.

Run (parent corpus, both interpreters from the worktree):
```bash
$PY - <<'EOF'
import json, sys
from pathlib import Path
from cua.sft.legacy_train_qwen import convert_trajectory_to_examples
out = []
for d in sorted(Path('/Users/rmajor/unik/maga/dataset_gather/trajectories').glob('**/task_*'))[:200]:
    if (d / 'trajectory.json').exists():
        ex, _ = convert_trajectory_to_examples(json.loads((d / 'trajectory.json').read_text()), d)
        out.append([e['messages'][1]['content'][1]['text'] for e in ex])
Path('/private/tmp/claude-501/-Users-rmajor-unik-maga-dataset-gather/e17b455f-c19b-4ab5-b159-8ba497af78ae/scratchpad/convert_after.json').write_text(json.dumps(out))
EOF
```
Before making the edit, run the same snippet with the output named `convert_before.json` (stash-free: run it first, then edit). Expected: `cmp convert_before.json convert_after.json` prints nothing.

- [ ] **Step 6: Report** — `git diff --stat cua/sft/legacy_train_qwen.py`; list new files. Do not `git add`.

---

### Task 2: Action decoder `cua/rl/actions.py`

**Files:**
- Create: `cua/rl/actions.py`, `tests/test_actions.py`

**Interfaces:**
- Consumes: `cua.sft.legacy_train_qwen.format_action_for_training`.
- Produces: `StepDecode(thought, actions, errors, canonical, raw)` with `.ok`; `decode_step(text: str) -> StepDecode`; `encode_step(thought: str, actions: list[dict]) -> str`; `ACTION_DECODERS: dict[str, Callable[[str], dict | None]]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_actions.py`:
```python
from cua.rl.actions import ACTION_DECODERS, decode_step, encode_step

FIXTURES = [
    {"type": "click", "x": 10, "y": 20, "button": "left"},
    {"type": "click", "x": 1279, "y": 719, "button": "right"},
    {"type": "double_click", "x": 5, "y": 6},
    {"type": "move", "x": 640, "y": 360},
    {"type": "type", "text": "hello world"},
    {"type": "type", "text": 'say "hi" (now)'},
    {"type": "type", "text": "ünïcødé ✓"},
    {"type": "type", "text": ""},
    {"type": "keypress", "keys": ["Enter"]},
    {"type": "keypress", "keys": ["Control", "a"]},
    {"type": "scroll", "x": 640, "y": 360, "scroll_x": 0, "scroll_y": -500},
    {"type": "drag", "path": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]},
    {"type": "wait"},
    {"type": "done"},
    {"type": "screenshot"},
    {"type": "navigate", "url": "https://example.com/a?b=c&d=(e)"},
    {"type": "handoff", "reason": "login"},
    {"type": "confirm", "proposed_action": {"type": "click", "x": 880, "y": 540}, "risk_category": "purchase"},
    {"type": "confirm", "proposed_action": {"type": "type"}, "risk_category": "send"},
    {"type": "confirm", "proposed_action": {}, "risk_category": ""},
]


def test_round_trip_every_action_shape():
    for action in FIXTURES:
        decoded = decode_step(encode_step("thinking", [action]))
        assert decoded.actions == [action], (action, decoded)
        assert decoded.canonical and decoded.ok and not decoded.errors, (action, decoded)
        assert decoded.thought == "thinking"


def test_every_decoder_is_covered_by_a_fixture():
    assert set(ACTION_DECODERS) == {a["type"] for a in FIXTURES}


def test_multi_action_step_keeps_order():
    actions = [FIXTURES[0], FIXTURES[4], FIXTURES[8]]
    decoded = decode_step(encode_step("go", actions))
    assert decoded.actions == actions and decoded.canonical


def test_multiline_type_text_round_trips():
    action = {"type": "type", "text": "line one\nline two"}
    decoded = decode_step(encode_step("t", [action, {"type": "done"}]))
    assert decoded.actions == [action, {"type": "done"}]


def test_drag_keeps_endpoints_only():
    action = {"type": "drag", "path": [{"x": 1, "y": 1}, {"x": 5, "y": 5}, {"x": 9, "y": 9}]}
    assert decode_step(encode_step("", [action])).actions == [{"type": "drag", "path": [{"x": 1, "y": 1}, {"x": 9, "y": 9}]}]


def test_missing_think_is_allowed():
    decoded = decode_step("action: done()")
    assert decoded.thought == "" and decoded.actions == [{"type": "done"}] and decoded.ok


def test_python_repr_keypress_decodes_but_is_not_canonical():
    decoded = decode_step("action: keypress(['Enter'])")
    assert decoded.actions == [{"type": "keypress", "keys": ["Enter"]}] and not decoded.canonical


def test_click_without_button_defaults_left_not_canonical():
    decoded = decode_step("action: click(<ref>(1, 2)</ref>)")
    assert decoded.actions == [{"type": "click", "x": 1, "y": 2, "button": "left"}] and not decoded.canonical


def test_stray_text_and_unknown_action_are_errors():
    decoded = decode_step("<think>x</think>\nI will click now\naction: fly()")
    assert decoded.actions == [] and not decoded.ok
    assert any(e.startswith("stray_text") for e in decoded.errors)
    assert "unknown_action:fly" in decoded.errors


def test_no_actions_is_an_error():
    assert decode_step("<think>hmm</think>").errors == ["no_actions"]


def test_history_summary_grammar_is_rejected():
    assert "bad_args:click" in decode_step("action: click(10, 20)").errors


def test_unclosed_think_is_malformed():
    decoded = decode_step("<think>never closed\naction: done()")
    assert "think_malformed" in decoded.errors and decoded.actions == [{"type": "done"}]


def test_trailing_prose_after_actions_is_stray():
    decoded = decode_step("<think>t</think>\naction: done()\nThat completes the task.")
    assert decoded.actions == [{"type": "done"}] and any(e.startswith("stray_text") for e in decoded.errors)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m tests test_actions` → Expected: `ModuleNotFoundError: No module named 'cua.rl.actions'`.

- [ ] **Step 3: Implement**

`cua/rl/actions.py`:
```python
"""Decode the student's `action:` lines into trajectory action dicts — the inverse of cua.sft.legacy_train_qwen.format_action_for_training."""
import ast
import json
import re
from dataclasses import dataclass, field

from cua.sft.legacy_train_qwen import format_action_for_training

_REF = r"<ref>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)</ref>"
_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>[ \t]*\n?", re.DOTALL)
_LINE_RE = re.compile(r"^action:\s*([a-z_]+)\((.*)\)\s*$", re.DOTALL)


@dataclass
class StepDecode:
    thought: str
    actions: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    canonical: bool = True
    raw: str = ""

    @property
    def ok(self):
        return bool(self.actions)


def _xy(match, first=1):
    return int(match.group(first)), int(match.group(first + 1))


def _decode_click(args):
    match = re.fullmatch(rf"{_REF}(?:,\s*button=(left|right))?", args)
    if not match:
        return None
    x, y = _xy(match)
    return {"type": "click", "x": x, "y": y, "button": match.group(3) or "left"}


def _point_decoder(action_type):
    def decode(args):
        match = re.fullmatch(_REF, args)
        if not match:
            return None
        x, y = _xy(match)
        return {"type": action_type, "x": x, "y": y}
    return decode


def _quoted_decoder(action_type, key):
    def decode(args):
        match = re.fullmatch(r'"(.*)"', args, re.DOTALL)
        return {"type": action_type, key: match.group(1)} if match else None
    return decode


def _decode_keypress(args):
    try:
        keys = json.loads(args)
    except json.JSONDecodeError:
        try:
            keys = ast.literal_eval(args)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        return None
    return {"type": "keypress", "keys": keys}


def _decode_scroll(args):
    match = re.fullmatch(rf"{_REF},\s*scrollX=(-?\d+),\s*scrollY=(-?\d+)", args)
    if not match:
        return None
    x, y = _xy(match)
    return {"type": "scroll", "x": x, "y": y, "scroll_x": int(match.group(3)), "scroll_y": int(match.group(4))}


def _decode_drag(args):
    match = re.fullmatch(rf"{_REF}\s*->\s*{_REF}", args)
    if not match:
        return None
    x1, y1 = _xy(match, 1)
    x2, y2 = _xy(match, 3)
    return {"type": "drag", "path": [{"x": x1, "y": y1}, {"x": x2, "y": y2}]}


def _bare_decoder(action_type):
    return lambda args: {"type": action_type} if args.strip() == "" else None


def _decode_handoff(args):
    match = re.fullmatch(r"reason=([A-Za-z0-9_-]*)", args)
    return {"type": "handoff", "reason": match.group(1) or "login"} if match else None


def _decode_confirm(args):
    match = re.fullmatch(r"propose=(.*),\s*risk=([A-Za-z0-9_-]*)", args, re.DOTALL)
    if not match:
        return None
    inner, risk = match.group(1), match.group(2)
    click = re.fullmatch(rf"click\({_REF}\)", inner)
    if click:
        x, y = _xy(click)
        proposed = {"type": "click", "x": x, "y": y}
    else:
        bare = re.fullmatch(r"([a-z_]+)\(\)", inner)
        if not bare:
            return None
        proposed = {} if bare.group(1) == "action" else {"type": bare.group(1)}
    return {"type": "confirm", "proposed_action": proposed, "risk_category": risk}


ACTION_DECODERS = {
    "click": _decode_click,
    "double_click": _point_decoder("double_click"),
    "move": _point_decoder("move"),
    "type": _quoted_decoder("type", "text"),
    "keypress": _decode_keypress,
    "scroll": _decode_scroll,
    "drag": _decode_drag,
    "wait": _bare_decoder("wait"),
    "done": _bare_decoder("done"),
    "screenshot": _bare_decoder("screenshot"),
    "navigate": _quoted_decoder("navigate", "url"),
    "handoff": _decode_handoff,
    "confirm": _decode_confirm,
}


def _split_action_lines(text):
    """Group lines into `action:` entries; a line that continues an unterminated entry (multi-line type text) is appended to it."""
    entries, stray = [], []
    for line in text.splitlines():
        if line.startswith("action:"):
            entries.append(line.rstrip())
        elif entries and not _LINE_RE.match(entries[-1]):
            entries[-1] += "\n" + line.rstrip()
        elif line.strip():
            stray.append(line.strip())
    return entries, stray


def decode_step(text):
    raw = text or ""
    errors = []
    match = _THINK_RE.match(raw)
    thought = match.group(1).strip() if match else ""
    rest = raw[match.end():] if match else raw
    if not match and "<think>" in raw:
        errors.append("think_malformed")
        rest = re.sub(r"<think>.*?(</think>|$)", "", raw, flags=re.DOTALL)
    entries, stray = _split_action_lines(rest)
    errors.extend(f"stray_text:{s[:40]}" for s in stray)
    actions, canonical = [], True
    for entry in entries:
        line = _LINE_RE.match(entry)
        if not line:
            errors.append(f"bad_line:{entry[:40]}")
            continue
        name, args = line.group(1), line.group(2)
        decoder = ACTION_DECODERS.get(name)
        if decoder is None:
            errors.append(f"unknown_action:{name}")
            continue
        action = decoder(args)
        if action is None:
            errors.append(f"bad_args:{name}")
            continue
        actions.append(action)
        canonical = canonical and format_action_for_training(action) == entry
    if not actions and not errors:
        errors.append("no_actions")
    return StepDecode(thought=thought, actions=actions, errors=errors, canonical=canonical and not errors, raw=raw)


def encode_step(thought, actions):
    """The assistant turn exactly as convert emits it."""
    action_str = "\n".join(format_action_for_training(a) for a in actions)
    return f"<think>{thought}</think>\n{action_str}" if thought else action_str
```
Note the unclosed-think test: `re.sub(r"<think>.*?(</think>|$)", ...)` with DOTALL and a lazy `.*?` reaching `$` at end-of-string would eat the action line; make the fallback strip only the first line of the `<think>` text: replace that `re.sub` call with `rest = "\n".join(l for l in raw.splitlines() if not l.lstrip().startswith("<think>"))`.

- [ ] **Step 4: Run tests** — `$PY -m tests test_actions` → Expected: `13/13 passed`.

- [ ] **Step 5: Report** — list `cua/rl/actions.py`, `tests/test_actions.py`.

---

### Task 3: Driver `/observe` + `sequence` kind + per-port drivers + smoke leg

**Files:**
- Modify: `cua/pipeline/instrumentation/browser_driver.js` (`recordStep`, `handleAct`, `handleSession`, `routes`, new helpers before `handleAct`)
- Modify: `cua/pipeline/driver_session.py` (`start_driver`, `DriverClient`)
- Modify: `cua/pipeline/driver_smoke.py` (new leg before the `done` action)

**Interfaces:**
- Produces (driver): `POST /observe -> {ok, b64 (PNG), mime:"image/png", url, title, step_next}`; `POST /act {kind:"sequence", thought, actions:[contract dicts]} -> {ok, step, url, executed, error_at, error, terminal}`; `POST /session {..., type_mode?: "replace"|"append"}`.
- Produces (Python): `start_driver(sandbox, port=DRIVER_PORT)`; `DriverClient(sandbox, port=DRIVER_PORT)`; `DriverClient.observe() -> dict`; `DriverClient.act_sequence(thought: str, actions: list[dict]) -> dict`; `DriverClient.session(..., type_mode: str | None = None)`.

- [ ] **Step 1: Add the smoke leg (the failing "test")**

In `cua/pipeline/driver_smoke.py`, insert before the `# The loop guard must refuse ...` comment:
```python
        # RL path: /observe hands its PNG to the next recorded step; `sequence` records ONE step with
        # the student's contract actions and executes them in order (ref-free type into the focused field).
        recorded = json.loads(sandbox.read_file(f"{RESULTS_DIR}/task_smoke_test/trajectory.json"))
        name_click = recorded["steps"][0]["actions"][0]
        greet_click = recorded["steps"][1]["actions"][0]
        observed = client.observe()
        assert observed["mime"] == "image/png" and observed["step_next"] == len(recorded["steps"]) + 1
        seq = client.act_sequence("Retyping my name through the coordinate path and submitting.", [
            {"type": "click", "x": name_click["x"], "y": name_click["y"], "button": "left"},
            {"type": "type", "text": "Sequence"},
            {"type": "click", "x": greet_click["x"], "y": greet_click["y"], "button": "left"},
        ])
        assert seq["executed"] == 3 and seq["error"] is None and seq["terminal"] is False, seq
        assert "hello Sequence" in client.snapshot()["snapshot"], "sequence actions did not take effect"
        observed_png = observed["b64"]
```
And inside the `with tempfile.TemporaryDirectory()` block, after the `# approval_event` assertions:
```python
            seq_step = [s for s in traj["steps"] if s["thought"].startswith("Retyping my name")][0]
            assert [a["type"] for a in seq_step["actions"]] == ["click", "type", "click"], seq_step["actions"]
            assert (traj_dir / seq_step["screenshot"]).read_bytes() == base64.b64decode(observed_png), \
                "observation PNG must be byte-identical to the recorded step screenshot"
```
Add `import base64` to the module imports.

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m cua.pipeline.driver_smoke` → Expected: FAIL with `AttributeError: 'DriverClient' object has no attribute 'observe'`.

- [ ] **Step 3: Implement the driver changes**

In `browser_driver.js`:

(a) `recordStep`: replace `await state.page.screenshot({ path: path.join(t.dir, screenshotName) });` with
```js
  const pending = t.pendingShot;
  t.pendingShot = null;
  if (pending) fs.writeFileSync(path.join(t.dir, screenshotName), pending);
  else await state.page.screenshot({ path: path.join(t.dir, screenshotName) });
```

(b) After the `resolveSecrets` function, add:
```js
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
  await scrubDevOverlays();
  await settle();
  const buf = await state.page.screenshot();
  state.task.pendingShot = buf;
  return {
    ok: true, b64: buf.toString('base64'), mime: 'image/png', url: state.page.url(),
    title: await state.page.title(), step_next: state.task.steps.length + 1,
  };
}
```

(c) In `handleAct`, add before the final `} else {` (unknown kind) branch:
```js
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
```

(d) In `handleSession`, after `state.secrets = body.secrets || {};` add `state.typeMode = body.type_mode === 'append' ? 'append' : 'replace';` and in the `state.task = {...}` literal add `pendingShot: null,` after `recentActions: [],`.

(e) In `routes`, add `'POST /observe': handleObserve,` after `'POST /act': handleAct,`. Update the header comment block with two lines: `*   POST /observe                               -> {b64 (PNG), url, title, step_next}; the PNG becomes the next recorded step's screenshot` and under `/act kinds`: `*   sequence  {actions:[contract dicts]}         -> records ONE step with those dicts verbatim, executes them in order (RL policy path)`.

In `cua/pipeline/driver_session.py`:
```python
def start_driver(sandbox, port: int = DRIVER_PORT):
    """Daemonize the driver on `port` and wait for it (one process per port; RL runs several per sandbox)."""
    sandbox.start_background(f"DRIVER_PORT={port} node {DRIVER_DIR}/browser_driver.js", f"{DRIVER_LOG}.{port}")
    if not sandbox.wait_for_port(port, DRIVER_READY_TIMEOUT):
        log_tail, _, _ = sandbox.exec_command(f"tail -20 {DRIVER_LOG}.{port}")
        raise DriverError(f"driver did not open port {port}; log tail: {log_tail[-400:]}")
```
`DriverClient.__init__(self, sandbox, port: int = DRIVER_PORT)` → `self.base = f"http://127.0.0.1:{port}"`. Add `type_mode: str | None = None` to `session()` and `if type_mode: body["type_mode"] = type_mode`. Add:
```python
    def observe(self) -> dict:
        """PNG observation whose bytes become the next recorded step's screenshot (RL parity)."""
        return self._post("/observe")

    def act_sequence(self, thought: str, actions: list[dict]) -> dict:
        return self._post("/act", {"kind": "sequence", "thought": thought, "actions": actions})
```

- [ ] **Step 4: Run the driver smoke**

Run: `$PY -m cua.pipeline.driver_smoke` → Expected: log ends with `Driver smoke test PASSED`, exit 0.

- [ ] **Step 5: Report** — `git diff --stat cua/pipeline/instrumentation/browser_driver.js cua/pipeline/driver_session.py cua/pipeline/driver_smoke.py`.

---

### Task 4: Executor `cua/rl/executor.py`

**Files:**
- Create: `cua/rl/executor.py`, `tests/test_executor.py`

**Interfaces:**
- Consumes: `cua.pipeline.corpus_audit.VIEWPORT`.
- Produces: `normalize_keys(keys) -> list[str]`; `validate_actions(actions) -> list[str]`; `control_index(actions) -> int | None`; `to_sequence_payload(thought, actions) -> {"kind":"sequence","thought","actions"}`; `MISSING_THOUGHT_PLACEHOLDER`; `CONTROL_TYPES`.

- [ ] **Step 1: Write the failing tests**

`tests/test_executor.py`:
```python
from cua.rl.executor import (MISSING_THOUGHT_PLACEHOLDER, control_index, normalize_keys, to_sequence_payload,
                         validate_actions)


def test_normalize_keys_maps_aliases_and_keeps_playwright_names():
    assert normalize_keys(["ctrl", "a"]) == ["Control", "a"]
    assert normalize_keys(["Enter", "ArrowDown", "return"]) == ["Enter", "ArrowDown", "Enter"]


def test_validate_actions_flags_out_of_viewport():
    assert validate_actions([{"type": "click", "x": 1280, "y": 10, "button": "left"}]) == ["out_of_viewport:click"]
    assert validate_actions([{"type": "drag", "path": [{"x": 0, "y": 0}, {"x": 5, "y": 900}]}]) == ["out_of_viewport:drag"]
    assert validate_actions([{"type": "type", "text": "x"}, {"type": "scroll", "x": 1, "y": 1, "scroll_x": 0, "scroll_y": 5}]) == []


def test_control_index():
    assert control_index([{"type": "click", "x": 1, "y": 1, "button": "left"}, {"type": "done"}, {"type": "wait"}]) == 1
    assert control_index([{"type": "wait"}]) is None


def test_payload_cuts_after_control_and_drops_screenshot():
    payload = to_sequence_payload("", [{"type": "screenshot"}, {"type": "keypress", "keys": ["esc"]}, {"type": "handoff", "reason": "login"}, {"type": "wait"}])
    assert payload["kind"] == "sequence" and payload["thought"] == MISSING_THOUGHT_PLACEHOLDER
    assert payload["actions"] == [{"type": "keypress", "keys": ["Escape"]}, {"type": "handoff", "reason": "login"}]


def test_payload_keeps_thought():
    assert to_sequence_payload("  go  ", [{"type": "wait"}])["thought"] == "go"
```

- [ ] **Step 2: Run to verify failure** — `$PY -m tests test_executor` → `ModuleNotFoundError: No module named 'cua.rl.executor'`.

- [ ] **Step 3: Implement**

`cua/rl/executor.py`:
```python
"""Turn a decoded student step into the driver's `sequence` payload."""
from cua.pipeline.corpus_audit import VIEWPORT

MISSING_THOUGHT_PLACEHOLDER = "(no reasoning emitted)"
CONTROL_TYPES = ("done", "handoff", "confirm")
# Uppercased aliases → Playwright key names; local copy because the Path-A recorder that owns the full map imports `openai`.
KEY_ALIASES = {
    "CTRL": "Control", "CONTROL": "Control", "ALT": "Alt", "SHIFT": "Shift", "META": "Meta", "CMD": "Meta",
    "COMMAND": "Meta", "ENTER": "Enter", "RETURN": "Enter", "TAB": "Tab", "ESCAPE": "Escape", "ESC": "Escape",
    "BACKSPACE": "Backspace", "DELETE": "Delete", "SPACE": " ", "UP": "ArrowUp", "DOWN": "ArrowDown",
    "LEFT": "ArrowLeft", "RIGHT": "ArrowRight", "PAGEUP": "PageUp", "PAGEDOWN": "PageDown", "HOME": "Home", "END": "End",
}


def normalize_keys(keys):
    return [KEY_ALIASES.get(k.upper(), k) for k in keys]


def in_viewport(x, y):
    return 0 <= x < VIEWPORT[0] and 0 <= y < VIEWPORT[1]


def validate_actions(actions):
    problems = []
    for a in actions:
        if a["type"] in ("click", "double_click", "move", "scroll") and not in_viewport(a["x"], a["y"]):
            problems.append(f"out_of_viewport:{a['type']}")
        if a["type"] == "drag" and not all(in_viewport(p["x"], p["y"]) for p in a["path"]):
            problems.append("out_of_viewport:drag")
    return problems


def control_index(actions):
    for i, a in enumerate(actions):
        if a["type"] in CONTROL_TYPES:
            return i
    return None


def to_sequence_payload(thought, actions):
    cut = control_index(actions)
    kept = actions if cut is None else actions[: cut + 1]
    out = []
    for a in kept:
        if a["type"] == "screenshot":
            continue
        if a["type"] == "keypress":
            a = {**a, "keys": normalize_keys(a["keys"])}
        out.append(a)
    return {"kind": "sequence", "thought": (thought or "").strip() or MISSING_THOUGHT_PLACEHOLDER, "actions": out}
```

- [ ] **Step 4: Run tests** — `$PY -m tests test_executor` → `5/5 passed`.
- [ ] **Step 5: Report.**

---

### Task 5: Observation `cua/rl/observation.py`

**Files:**
- Create: `cua/rl/observation.py`; extend `tests/test_observation.py`

**Interfaces:**
- Produces: `build_observation(instruction, action_history, image_png_b64, step_index) -> {"system","text","image_b64","step_index"}`; `to_openai_messages(observation) -> list[dict]`.

- [ ] **Step 1: Add failing tests** to `tests/test_observation.py` (add `from cua.rl.observation import build_observation, to_openai_messages` and `from cua.sft.legacy_train_qwen import SYSTEM_PROMPT` at the top):
```python
def test_build_observation_matches_sft_user_turn():
    obs = build_observation("Say hi", ["click(1, 2)"], "QUJD", 1)
    assert obs == {"system": SYSTEM_PROMPT, "text": build_step_user_text("Say hi", ["click(1, 2)"]), "image_b64": "QUJD", "step_index": 1}


def test_openai_messages_carry_data_uri_then_text():
    msgs = to_openai_messages(build_observation("Say hi", [], "QUJD", 0))
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1]["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    assert msgs[1]["content"][1] == {"type": "text", "text": "Task: Say hi\nWhat action should you take?"}
```
- [ ] **Step 2: Run** — `$PY -m tests test_observation` → ImportError.
- [ ] **Step 3: Implement** `cua/rl/observation.py`:
```python
"""The student's observation: the exact SFT user turn plus the PNG, and its OpenAI-message rendering."""
from cua.sft.legacy_train_qwen import SYSTEM_PROMPT, build_step_user_text


def build_observation(instruction, action_history, image_png_b64, step_index):
    return {"system": SYSTEM_PROMPT, "text": build_step_user_text(instruction, action_history),
            "image_b64": image_png_b64, "step_index": step_index}


def to_openai_messages(observation):
    return [
        {"role": "system", "content": observation["system"]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{observation['image_b64']}"}},
            {"type": "text", "text": observation["text"]},
        ]},
    ]
```
- [ ] **Step 4: Run** — `8/8 passed`. **Step 5: Report.**

---

### Task 6: Programmatic checks `cua/rl/checks.py`

**Files:**
- Create: `cua/rl/checks.py`, `tests/test_checks.py`

**Interfaces:**
- Produces: `CHECKS: dict[str, Callable[[Any, dict], bool]]`; `build_context(snapshot: dict, storage_state_loader: Callable[[], dict], record: dict) -> dict`; `evaluate(spec: dict, ctx: dict) -> {"passed": bool, "results": dict}`.

- [ ] **Step 1: Failing tests** `tests/test_checks.py`:
```python
from cua.rl.checks import CHECKS, build_context, evaluate

SNAPSHOT = {"url": "https://example.com/cart?id=7", "title": "Your Cart — Example",
            "snapshot": '## Page text\nOrder confirmed. Thanks!\n## Interactive elements (in viewport)\n[1] button "Checkout"\n[2] textbox "Coupon" value="SAVE10"\n[3] checkbox "Gift wrap" (checked)\n'}


def _ctx(terminal="done"):
    return build_context(SNAPSHOT, lambda: {"cookies": [{"name": "session_id"}]}, {"terminal_reason": terminal})


def test_context_parses_elements():
    ctx = _ctx()
    assert ctx["elements"][1] == {"ref": 2, "role": "textbox", "name": "Coupon", "extra": 'value="SAVE10"'}


def test_each_check():
    ctx = _ctx()
    assert evaluate({"url_matches": r"/cart\?id=\d+"}, ctx)["passed"]
    assert evaluate({"url_contains": "/cart"}, ctx)["passed"]
    assert evaluate({"title_contains": "your cart"}, ctx)["passed"]
    assert evaluate({"text_visible": "order confirmed"}, ctx)["passed"]
    assert not evaluate({"text_absent": "order confirmed"}, ctx)["passed"]
    assert evaluate({"element_present": {"role": "button", "name_regex": "^check"}}, ctx)["passed"]
    assert evaluate({"element_absent": {"name_regex": "Delete"}}, ctx)["passed"]
    assert evaluate({"element_value": {"name_regex": "Coupon", "equals": "SAVE10"}}, ctx)["passed"]
    assert evaluate({"cookie_present": "^session"}, ctx)["passed"]
    assert evaluate({"done_emitted": True}, ctx)["passed"] and not evaluate({"done_emitted": True}, _ctx("step_cap"))["passed"]
    assert evaluate({"terminal_reason_in": ["done", "handoff"]}, ctx)["passed"]


def test_composition_and_unknown_check():
    ctx = _ctx()
    spec = {"all_of": [{"text_visible": "thanks"}, {"any_of": [{"url_contains": "nope"}, {"url_contains": "cart"}]}]}
    assert evaluate(spec, ctx) == {"passed": True, "results": {"all_of": True}}
    assert evaluate({"bogus": 1}, ctx) == {"passed": False, "results": {"bogus": False}}
    assert set(CHECKS) >= {"url_matches", "text_visible", "element_present", "all_of", "any_of"}
```
- [ ] **Step 2: Run** → ModuleNotFoundError.
- [ ] **Step 3: Implement** `cua/rl/checks.py`:
```python
"""Programmatic success checks evaluated against the live page at episode end (registry keyed by check name)."""
import re

_ELEMENT_RE = re.compile(r'^\[(\d+)\] (\S+) "([^"]*)"\s*(.*)$', re.MULTILINE)


def build_context(snapshot, storage_state_loader, record):
    text = snapshot.get("snapshot", "")
    elements = [{"ref": int(m.group(1)), "role": m.group(2), "name": m.group(3), "extra": m.group(4).strip()}
                for m in _ELEMENT_RE.finditer(text)]
    return {"url": snapshot.get("url", ""), "title": snapshot.get("title", ""), "text": text, "elements": elements,
            "storage_state": storage_state_loader, "terminal_reason": record.get("terminal_reason", "")}


def _element_matches(element, spec):
    role_ok = not spec.get("role") or element["role"] == spec["role"]
    name_ok = not spec.get("name_regex") or re.search(spec["name_regex"], element["name"], re.I)
    return bool(role_ok and name_ok)


def _cookie_present(spec, ctx):
    state = ctx["storage_state"]() or {}
    return any(re.search(spec, c.get("name", "")) for c in state.get("cookies", []))


CHECKS = {
    "url_matches": lambda spec, ctx: bool(re.search(spec, ctx["url"])),
    "url_contains": lambda spec, ctx: spec in ctx["url"],
    "title_contains": lambda spec, ctx: spec.lower() in ctx["title"].lower(),
    "text_visible": lambda spec, ctx: spec.lower() in ctx["text"].lower(),
    "text_absent": lambda spec, ctx: spec.lower() not in ctx["text"].lower(),
    "element_present": lambda spec, ctx: any(_element_matches(e, spec) for e in ctx["elements"]),
    "element_absent": lambda spec, ctx: not any(_element_matches(e, spec) for e in ctx["elements"]),
    "element_value": lambda spec, ctx: any(_element_matches(e, spec) and f'value="{spec["equals"]}"' in e["extra"] for e in ctx["elements"]),
    "cookie_present": _cookie_present,
    "done_emitted": lambda spec, ctx: (ctx["terminal_reason"] == "done") == bool(spec),
    "terminal_reason_in": lambda spec, ctx: ctx["terminal_reason"] in spec,
    "all_of": lambda spec, ctx: all(evaluate(s, ctx)["passed"] for s in spec),
    "any_of": lambda spec, ctx: any(evaluate(s, ctx)["passed"] for s in spec),
}


def evaluate(spec, ctx):
    """spec = {check_name: arg, ...}; several keys are an implicit all_of."""
    results = {}
    for name, arg in spec.items():
        check = CHECKS.get(name)
        results[name] = bool(check(arg, ctx)) if check else False
    return {"passed": bool(results) and all(results.values()), "results": results}
```
- [ ] **Step 4: Run** → `3/3 passed`. **Step 5: Report.**

---

### Task 7: Reward composition `cua/rl/rewards.py` + `cua/rl/reward_config.json`

**Files:**
- Create: `cua/rl/rewards.py`, `cua/rl/reward_config.json`, `tests/test_rewards.py`

**Interfaces:**
- Produces: `REWARD_COMPONENTS`, `PENALTIES` (name → `fn(record, task) -> float`); `load_reward_config(path=REWARD_CONFIG_PATH) -> dict`; `compute_reward(record, task, config) -> (float, {"components", "penalties"})`; `REWARD_MIN=-1.0`, `REWARD_MAX=1.5`.
- Record shape consumed: `{"success": bool|None, "turns": [{"clean": bool}], "ui_steps": int, "terminal_reason": str, "navigate_blocked": int}`.

- [ ] **Step 1: Failing tests** `tests/test_rewards.py`:
```python
from cua.rl.rewards import PENALTIES, REWARD_COMPONENTS, REWARD_MAX, compute_reward, load_reward_config

CONFIG = load_reward_config()
TASK = {"lane": "autonomous", "expected_actions": 10}


def _record(**kw):
    base = {"success": True, "turns": [{"clean": True}] * 4, "ui_steps": 8, "terminal_reason": "done", "navigate_blocked": 0}
    return {**base, **kw}


def test_config_has_both_lanes_and_all_names():
    for lane in ("autonomous", "guarded"):
        assert set(CONFIG["lanes"][lane]["weights"]) == set(REWARD_COMPONENTS)
        assert set(CONFIG["lanes"][lane]["penalties"]) == set(PENALTIES)


def test_perfect_episode_scores_full_weights():
    reward, breakdown = compute_reward(_record(), TASK, CONFIG)
    w = CONFIG["lanes"]["autonomous"]["weights"]
    assert reward == round(sum(w.values()), 4) and reward <= REWARD_MAX
    assert breakdown["components"] == {"terminal_success": 1.0, "format_validity": 1.0, "efficiency": 1.0, "terminated_with_done": 1.0}


def test_failed_parse_abort_is_negative():
    reward, breakdown = compute_reward(_record(success=False, terminal_reason="parse_abort", turns=[{"clean": False}] * 3), TASK, CONFIG)
    assert reward < 0 and breakdown["penalties"]["parse_abort"] == 1.0


def test_efficiency_needs_success_and_caps_at_one():
    assert REWARD_COMPONENTS["efficiency"](_record(ui_steps=20), TASK) == 0.5
    assert REWARD_COMPONENTS["efficiency"](_record(success=False), TASK) == 0.0
    assert REWARD_COMPONENTS["efficiency"](_record(ui_steps=3), TASK) == 1.0


def test_unknown_success_counts_as_zero():
    reward, _ = compute_reward(_record(success=None, terminal_reason="step_cap"), TASK, CONFIG)
    assert reward < CONFIG["lanes"]["autonomous"]["weights"]["terminal_success"]
```
- [ ] **Step 2: Run** → ModuleNotFoundError.
- [ ] **Step 3: Implement**

`cua/rl/reward_config.json`:
```json
{
  "verifier_mode": {"train": "fallback", "eval": "always"},
  "lanes": {
    "autonomous": {
      "weights": {"terminal_success": 1.0, "format_validity": 0.2, "efficiency": 0.2, "terminated_with_done": 0.1},
      "penalties": {"parse_abort": 0.5, "loop_guard": 0.3, "step_cap": 0.2, "control_unexpected": 0.3, "navigate_blocked": 0.1}
    },
    "guarded": {
      "weights": {"terminal_success": 1.0, "format_validity": 0.2, "efficiency": 0.1, "terminated_with_done": 0.0},
      "penalties": {"parse_abort": 0.5, "loop_guard": 0.3, "step_cap": 0.2, "control_unexpected": 0.0, "navigate_blocked": 0.1}
    }
  }
}
```
`cua/rl/rewards.py`:
```python
"""Episode reward = weighted components − penalties; weights per lane live in reward_config.json."""
import json
from pathlib import Path

REWARD_CONFIG_PATH = Path(__file__).parent / "reward_config.json"
REWARD_MIN, REWARD_MAX = -1.0, 1.5


def _terminal_success(record, task):
    return 1.0 if record.get("success") else 0.0


def _format_validity(record, task):
    turns = record.get("turns") or []
    return sum(1 for t in turns if t.get("clean")) / len(turns) if turns else 0.0


def _efficiency(record, task):
    expected = task.get("expected_actions") or 0
    ui_steps = record.get("ui_steps") or 0
    if not record.get("success") or not expected or not ui_steps:
        return 0.0
    return min(1.0, expected / ui_steps)


def _reason_is(reason):
    return lambda record, task: 1.0 if record.get("terminal_reason") == reason else 0.0


REWARD_COMPONENTS = {
    "terminal_success": _terminal_success,
    "format_validity": _format_validity,
    "efficiency": _efficiency,
    "terminated_with_done": _reason_is("done"),
}

PENALTIES = {
    "parse_abort": _reason_is("parse_abort"),
    "loop_guard": _reason_is("loop_guard"),
    "step_cap": _reason_is("step_cap"),
    "control_unexpected": _reason_is("control_unexpected"),
    "navigate_blocked": lambda record, task: float(record.get("navigate_blocked") or 0),
}


def load_reward_config(path=REWARD_CONFIG_PATH):
    return json.loads(Path(path).read_text())


def compute_reward(record, task, config):
    lane = config["lanes"][task.get("lane", "autonomous")]
    components = {name: fn(record, task) for name, fn in REWARD_COMPONENTS.items()}
    penalties = {name: fn(record, task) for name, fn in PENALTIES.items()}
    total = sum(lane["weights"].get(n, 0.0) * v for n, v in components.items())
    total -= sum(lane["penalties"].get(n, 0.0) * v for n, v in penalties.items())
    return max(REWARD_MIN, min(REWARD_MAX, round(total, 4))), {"components": components, "penalties": penalties}
```
- [ ] **Step 4: Run** → `5/5 passed`. **Step 5: Report.**

---

### Task 8: Judge `cua/rl/judge.py` (check → verifier → deferred)

**Files:**
- Create: `cua/rl/judge.py`, `tests/test_judge.py`

**Interfaces:**
- Consumes: `cua.pipeline.agent_capture.verify_trajectory(traj_dir, autonomy=, full_name=)`, `cua.pipeline.sdk_runner.rate_limit_tracker`, `UsageLimitReached`.
- Produces: `AsyncBridge()` with `.run(coro, timeout=None)`, `.close()`; `judge_success(traj_dir, task, check_result, mode, bridge, verify=verify_trajectory) -> (success: bool|None, source: str, status: "final"|"deferred")`; `cached_verdict(traj_dir) -> dict|None`.

- [ ] **Step 1: Failing tests** `tests/test_judge.py` (the verifier is injected so no Claude call happens):
```python
import asyncio
import json
import tempfile
from pathlib import Path

from cua.rl.judge import AsyncBridge, judge_success


def _traj_dir(tmp, success=False, verifier=None):
    d = Path(tmp)
    (d / "trajectory.json").write_text(json.dumps({"task_id": "t", "instruction": "i", "start_url": "u", "steps": [], "total_steps": 0, "success": success}))
    (d / "capture_meta.json").write_text(json.dumps({"verifier": verifier} if verifier else {}))
    return d


def test_check_wins_in_fallback_mode_without_verifier_call():
    calls = []
    async def fake_verify(traj_dir, autonomy, full_name):
        calls.append(traj_dir)
    with tempfile.TemporaryDirectory() as tmp:
        result = judge_success(_traj_dir(tmp), {"task_id": "t", "lane": "autonomous"}, {"passed": True}, "fallback", None, verify=fake_verify)
    assert result == (True, "success_check", "final") and calls == []


def test_never_mode_returns_none_without_check():
    with tempfile.TemporaryDirectory() as tmp:
        assert judge_success(_traj_dir(tmp), {"task_id": "t"}, None, "never", None) == (None, "none", "final")


def test_verifier_runs_on_bridge_and_reads_trajectory_success():
    seen = {}
    async def fake_verify(traj_dir, autonomy, full_name):
        seen["autonomy"] = autonomy
        t = json.loads((traj_dir / "trajectory.json").read_text())
        t["success"] = True
        (traj_dir / "trajectory.json").write_text(json.dumps(t))
        (traj_dir / "capture_meta.json").write_text(json.dumps({"verifier": {"success": True}}))
    bridge = AsyncBridge()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = judge_success(_traj_dir(tmp), {"task_id": "t", "lane": "guarded"}, {"passed": False}, "always", bridge, verify=fake_verify)
    finally:
        bridge.close()
    assert result == (True, "verifier", "final") and seen["autonomy"] == "guarded"


def test_cached_verdict_skips_verifier():
    async def boom(traj_dir, autonomy, full_name):
        raise AssertionError("must not run")
    with tempfile.TemporaryDirectory() as tmp:
        d = _traj_dir(tmp, success=False, verifier={"success": False})
        assert judge_success(d, {"task_id": "t"}, None, "fallback", None, verify=boom) == (False, "verifier", "final")
```
- [ ] **Step 2: Run** → ModuleNotFoundError.
- [ ] **Step 3: Implement** `cua/rl/judge.py`:
```python
"""Terminal success for a rollout dir: programmatic check first, Claude verifier as fallback/audit — all SDK calls on one background loop."""
import asyncio
import json
import os
import threading
from pathlib import Path

from cua.pipeline.agent_capture import verify_trajectory
from cua.pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker

VERIFIER_CONCURRENCY = int(os.environ.get("RL_VERIFIER_CONCURRENCY", "2"))
VERIFY_TIMEOUT_S = int(os.environ.get("RL_VERIFY_TIMEOUT_S", "2700"))


class AsyncBridge:
    """One event loop thread for every Claude SDK call so rate_limit_tracker's single-loop assumption holds."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.semaphore = asyncio.Semaphore(VERIFIER_CONCURRENCY)
        self.thread = threading.Thread(target=self.loop.run_forever, name="rl-async-bridge", daemon=True)
        self.thread.start()

    def run(self, coro, timeout=None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


def cached_verdict(traj_dir):
    meta_file = Path(traj_dir) / "capture_meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    return meta.get("verifier")


async def _verify(traj_dir, task, bridge, verify):
    async with bridge.semaphore:
        await verify(traj_dir, autonomy=task.get("lane", "autonomous"), full_name=f"rl:{task.get('task_id', '')}")


def judge_success(traj_dir, task, check_result, mode, bridge, verify=verify_trajectory):
    traj_dir = Path(traj_dir)
    if check_result is not None and mode != "always":
        return bool(check_result["passed"]), "success_check", "final"
    if mode == "never":
        return (bool(check_result["passed"]), "success_check", "final") if check_result else (None, "none", "final")
    if cached_verdict(traj_dir) is None:
        if rate_limit_tracker.limit_reached():
            return None, "verifier", "deferred"
        try:
            bridge.run(_verify(traj_dir, task, bridge, verify), timeout=VERIFY_TIMEOUT_S)
        except UsageLimitReached:
            return None, "verifier", "deferred"
    traj = json.loads((traj_dir / "trajectory.json").read_text())
    return bool(traj.get("success")), "verifier", "final"
```
- [ ] **Step 4: Run** → `4/4 passed`. **Step 5: Report.**

---

### Task 9: Task suite `cua/rl/tasks.py` + `cua/rl/rl_tasks.json`

**Files:**
- Create: `cua/rl/tasks.py`, `tests/test_tasks.py`, `cua/rl/rl_tasks.json` (generated)

**Interfaces:**
- Produces: `TASK_SOURCES: list[dict]` (`{"path", "backend", "env", "lane", "auth_mode"}`); `build_suite(sources, trajectories_dir, seen_config, unseen_site_pct=15) -> {"suite_sha256", "built_from", "tasks": [...]}`; `check_suite(suite, trajectories_dir, seen_config) -> list[str]` (problems); `load_suite(path) -> dict`; `select_tasks(suite, split=None, task_ids=None) -> list[dict]`; `is_idempotent(task_src) -> bool`; `split_for(site, task_index, idempotent, seen_sites, unseen_site_pct) -> str`. CLI `python -m cua.rl.tasks build|check|list`.
- Task record fields (normative): `task_id, site, start_url, instruction, difficulty, expected_actions, action_kinds, backend, env, lane, auth_mode, idempotent, split, preconditions, allowed_hosts, success_check, throttle, source`.

- [ ] **Step 1: Failing tests** `tests/test_tasks.py`:
```python
import json
import tempfile
from pathlib import Path

from cua.rl.tasks import build_suite, check_suite, is_idempotent, select_tasks, split_for

SRC = {"path": None, "backend": "public_site", "env": "live", "lane": "autonomous", "auth_mode": "none"}


def _sites(n_sites=6, tasks_per_site=3):
    return {"sites": [{"name": f"site{i}", "start_url": f"https://site{i}.example/", "domain": "d", "tasks": [
        {"id": f"t{j}", "instruction": f"Open the settings page and change the theme to dark {i}-{j}", "difficulty": "hard",
         "expected_actions": 12, "action_kinds": ["click", "scroll"], "auth_required": False} for j in range(tasks_per_site)]}
        for i in range(n_sites)]}


def _write(tmp, name, data):
    p = Path(tmp) / name
    p.write_text(json.dumps(data))
    return p


def test_idempotent_heuristic():
    assert is_idempotent({"instruction": "Sort the table by price and open the first row", "action_kinds": ["click"]})
    assert not is_idempotent({"instruction": "Publish the draft post", "action_kinds": ["click"]})
    assert not is_idempotent({"instruction": "Fill the form", "action_kinds": ["type"], "requires_upload": True})
    assert is_idempotent({"instruction": "Delete everything", "action_kinds": ["click"], "idempotent": True})


def test_split_is_deterministic_and_covers_labels():
    assert split_for("site0", 0, True, set(), 100) == "eval_unseen_site"
    assert split_for("site0", 0, True, set(), 0) == "eval_seen_site_unseen_task"
    assert split_for("site0", 1, True, set(), 0) == "train"
    assert split_for("site0", 1, False, set(), 0) == "eval_mutating"
    assert split_for("site0", 1, True, {"site0"}, 0) == "eval_seen_site_unseen_task"


def test_build_excludes_trained_tasks_and_checks_disjointness():
    with tempfile.TemporaryDirectory() as tmp:
        sites = _sites()
        src = {**SRC, "path": _write(tmp, "src.json", sites)}
        seen = _write(tmp, "seen.json", {"sites": [{"name": "site1", "tasks": [{"id": "x", "instruction": sites["sites"][1]["tasks"][0]["instruction"]}]}]})
        traj = Path(tmp) / "trajectories" / "site_site2" / "task_t1"
        traj.mkdir(parents=True)
        (traj / "trajectory.json").write_text(json.dumps({"instruction": "irrelevant"}))
        suite = build_suite([src], Path(tmp) / "trajectories", seen, unseen_site_pct=0)
        ids = {t["task_id"] for t in suite["tasks"]}
        assert "site1__t0" not in ids and "site2__t1" not in ids and "site0__t0" in ids
        assert len(suite["suite_sha256"]) == 64 and check_suite(suite, Path(tmp) / "trajectories", seen) == []
        train = select_tasks(suite, split="train")
        assert train and all(t["split"] == "train" and t["idempotent"] for t in train)
        assert select_tasks(suite, task_ids=["site0__t1"])[0]["allowed_hosts"] == ["site0.example"]
```
- [ ] **Step 2: Run** → ModuleNotFoundError.
- [ ] **Step 3: Implement** `cua/rl/tasks.py`:
```python
"""Build / check / list the RL task suite from the site registries, keeping it disjoint from anything ever trained on."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from cua.pipeline import app_secrets
from cua.pipeline.public_capture import _norm_instruction
from cua.pipeline.stage6_generate import _safe_task_id

REPO = Path(__file__).resolve().parent.parent
SUITE_PATH = Path(__file__).parent / "rl_tasks.json"
SEEN_CONFIG = REPO / "repo_pipeline" / "public_sites.json"
TASK_SOURCES = [
    {"path": REPO / "repo_pipeline" / "public_sites_actions_loginfree.json", "backend": "public_site", "env": "live", "lane": "autonomous", "auth_mode": "none"},
    {"path": REPO / "repo_pipeline" / "public_sites_actions_logingated.json", "backend": "public_site", "env": "live", "lane": "autonomous", "auth_mode": "pre_session"},
    {"path": REPO / "repo_pipeline" / "public_sites_human.json", "backend": "public_site", "env": "live", "lane": "autonomous", "auth_mode": "pre_session"},
    {"path": REPO / "repo_pipeline" / "public_sites_saucedemo.json", "backend": "public_site", "env": "test", "lane": "autonomous", "auth_mode": "login"},
]
_MUTATING = re.compile(r"\b(delete|remove|publish|post|send|submit|purchase|buy|pay|order|book|subscribe|register|sign up|"
                       r"create an account|upload|share|invite|tweet|comment|reply|unsubscribe|cancel|transfer|donate)\b", re.I)


def is_idempotent(task):
    if "idempotent" in task:
        return bool(task["idempotent"])
    if task.get("requires_upload"):
        return False
    return not _MUTATING.search(task.get("instruction", ""))


def split_for(site, task_index, idempotent, seen_sites, unseen_site_pct):
    if int(hashlib.sha1(site.encode()).hexdigest()[:8], 16) % 100 < unseen_site_pct:
        return "eval_unseen_site"
    if not idempotent:
        return "eval_mutating"
    if site in seen_sites or task_index == 0:
        return "eval_seen_site_unseen_task"
    return "train"


def _preconditions(site, auth_mode):
    if auth_mode == "none":
        return {}, True
    burner = site.get("burner_account_id")
    ref = site.get("storage_state_ref")
    ok = bool(burner and app_secrets.is_registered_burner(burner) and (auth_mode != "pre_session" or app_secrets.load_storage_state(ref)))
    return {"requires_session": auth_mode == "pre_session", "session_ref": ref, "burner_account_id": burner, "env": site.get("env", "live")}, ok


def _trained_keys(trajectories_dir, seen_config):
    dirs, instructions = set(), set()
    if seen_config and Path(seen_config).exists():
        for s in json.loads(Path(seen_config).read_text())["sites"]:
            for t in s["tasks"]:
                dirs.add((s["name"], _safe_task_id(t["id"])))
                instructions.add(_norm_instruction(t["instruction"]))
    for traj in Path(trajectories_dir).glob("**/task_*/trajectory.json") if trajectories_dir and Path(trajectories_dir).exists() else []:
        site_dir = traj.parent.parent.name
        dirs.add((site_dir[len("site_"):] if site_dir.startswith("site_") else site_dir, traj.parent.name[len("task_"):]))
        try:
            instructions.add(_norm_instruction(json.loads(traj.read_text()).get("instruction", "")))
        except (json.JSONDecodeError, OSError):
            continue
    return dirs, instructions


def build_suite(sources, trajectories_dir, seen_config, unseen_site_pct=15):
    trained_dirs, trained_instructions = _trained_keys(trajectories_dir, seen_config)
    seen_sites = {name for name, _ in trained_dirs}
    tasks, built_from = [], []
    for src in sources:
        path = Path(src["path"])
        if not path.exists():
            continue
        sites = json.loads(path.read_text())["sites"]
        built_from.append({"path": str(path), "sites": len(sites)})
        for site in sites:
            if site.get("skip_capture"):
                continue
            auth_mode = site.get("auth_mode", src["auth_mode"])
            preconditions, available = _preconditions(site, auth_mode)
            if not available:
                continue
            for index, t in enumerate(sorted(site["tasks"], key=lambda t: t["id"])):
                if (site["name"], _safe_task_id(t["id"])) in trained_dirs or _norm_instruction(t["instruction"]) in trained_instructions:
                    continue
                idempotent = is_idempotent(t)
                tasks.append({
                    "task_id": f"{site['name']}__{t['id']}", "site": site["name"], "start_url": site["start_url"],
                    "instruction": t["instruction"], "difficulty": t.get("difficulty", "medium"),
                    "expected_actions": t.get("expected_actions", 8), "action_kinds": t.get("action_kinds", []),
                    "backend": src["backend"], "env": src["env"], "lane": site.get("autonomy", src["lane"]), "auth_mode": auth_mode,
                    "idempotent": idempotent, "split": split_for(site["name"], index, idempotent, seen_sites, unseen_site_pct),
                    "preconditions": preconditions, "allowed_hosts": [urlparse(site["start_url"]).netloc],
                    "success_check": t.get("success_check"), "throttle": {"min_interval_s": site.get("min_interval_s", 30)},
                    "source": path.name,
                })
    tasks.sort(key=lambda t: t["task_id"])
    digest = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()
    return {"suite_sha256": digest, "built_from": built_from, "tasks": tasks}


def check_suite(suite, trajectories_dir, seen_config):
    problems = []
    trained_dirs, trained_instructions = _trained_keys(trajectories_dir, seen_config)
    ids, instructions = set(), {}
    for t in suite["tasks"]:
        if t["task_id"] in ids:
            problems.append(f"duplicate task_id {t['task_id']}")
        ids.add(t["task_id"])
        norm = _norm_instruction(t["instruction"])
        if norm in instructions and instructions[norm] != t["split"]:
            problems.append(f"instruction shared across splits: {t['task_id']} / {instructions[norm]}")
        instructions[norm] = t["split"]
        if (t["site"], _safe_task_id(t["task_id"].split("__", 1)[1])) in trained_dirs or norm in trained_instructions:
            problems.append(f"trained-on task in suite: {t['task_id']}")
        if t["split"] == "train" and not t["idempotent"]:
            problems.append(f"non-idempotent task in train: {t['task_id']}")
    return problems


def load_suite(path=SUITE_PATH):
    return json.loads(Path(path).read_text())


def select_tasks(suite, split=None, task_ids=None):
    tasks = suite["tasks"]
    if task_ids:
        wanted = set(task_ids)
        return [t for t in tasks if t["task_id"] in wanted]
    if split and split != "all":
        return [t for t in tasks if t["split"] == split or (split == "eval" and t["split"].startswith("eval"))]
    return list(tasks)


def main():
    parser = argparse.ArgumentParser(description="RL task suite: build | check | list")
    parser.add_argument("command", choices=["build", "check", "list"])
    parser.add_argument("--out", default=str(SUITE_PATH))
    parser.add_argument("--trajectories", default=str(REPO / "trajectories"))
    parser.add_argument("--seen-config", default=str(SEEN_CONFIG))
    parser.add_argument("--unseen-site-pct", type=int, default=15)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    if args.command == "build":
        suite = build_suite(TASK_SOURCES, args.trajectories, args.seen_config, args.unseen_site_pct)
        Path(args.out).write_text(json.dumps(suite, indent=1))
        print(f"built {len(suite['tasks'])} tasks from {suite['built_from']} sha {suite['suite_sha256'][:12]} -> {args.out}")
        return 0
    suite = load_suite(args.out)
    if args.command == "check":
        problems = check_suite(suite, args.trajectories, args.seen_config)
        counts = {}
        for t in suite["tasks"]:
            counts[t["split"]] = counts.get(t["split"], 0) + 1
        print(f"suite sha256 {suite['suite_sha256']} splits {counts} problems {len(problems)}")
        for p in problems:
            print(f"  - {p}")
        return 1 if problems else 0
    for t in select_tasks(suite, split=args.split):
        print(f"{t['task_id']:50} {t['split']:28} {t['lane']:10} {t['auth_mode']:12} {t['instruction'][:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
Note: `_write` in the test passes `seen_config` as a `Path` to a file with the `{"sites":[{"name","tasks":[{"id","instruction"}]}]}` shape, which `_trained_keys` reads. Note also that `_MUTATING` classifies "sign up"/"register" as mutating: those tasks land in `eval_mutating`, never `train` — intended for the live lane (the burner-account registration flows are capture-only).

- [ ] **Step 4: Run** → `3/3 passed`.
- [ ] **Step 5: Build the real suite** — `$PY -m cua.rl.tasks build --trajectories /Users/rmajor/unik/maga/dataset_gather/trajectories && $PY -m cua.rl.tasks check --trajectories /Users/rmajor/unik/maga/dataset_gather/trajectories` → Expected: `problems 0`, a `train` count > 0 (login-free idempotent tasks), pre_session count 0 until burners exist. Paste the split counts into the report.
- [ ] **Step 6: Report.**

---

### Task 10: Backends + worker pool `cua/rl/backends.py`

**Files:**
- Create: `cua/rl/backends.py`, `tests/test_backends.py`

**Interfaces:**
- Consumes: `DockerSandbox`, `safe_parallel_sandboxes`, `driver_session.install_driver/start_driver/DriverClient/DRIVER_PORT`, `app_secrets`.
- Produces: `resolve_preconditions(task) -> dict | None`; `EnvBackend` (`open(sandbox)`, `start_url(task)`, `pre_episode(sandbox, task)`, `session_kwargs(task)`, `post_episode(sandbox, task, container_task_dir)`); `PublicSiteBackend`, `StaticPageBackend(html)`; `BACKENDS`; `RolloutWorker(worker_id, backend, drivers_per_sandbox)` with `.start() .stop() .recycle() .needs_recycle() .client(slot) .sandbox .live .lock`; `WorkerPool(backend, requested_workers, drivers_per_sandbox)` with `.start() .acquire(timeout) -> (worker, slot) .release(worker, slot) .stop() .slot_count`.

- [ ] **Step 1: Failing tests** `tests/test_backends.py` (pure logic only; Docker paths are covered by the smoke):
```python
from cua.rl.backends import BACKENDS, StaticPageBackend, resolve_preconditions


def test_no_session_needed_is_empty_kwargs():
    assert resolve_preconditions({"task_id": "a", "preconditions": {}}) == {}


def test_unregistered_burner_fails_closed():
    task = {"task_id": "a", "lane": "autonomous", "preconditions": {"requires_session": True, "burner_account_id": "nope", "session_ref": "nope"}}
    assert resolve_preconditions(task) is None


def test_static_backend_serves_local_url_and_needs_no_auth():
    backend = StaticPageBackend("<html></html>")
    assert backend.start_url({"start_url": "ignored"}) == "http://127.0.0.1:3000/"
    assert backend.session_kwargs({"preconditions": {"requires_session": True}}) == {}
    assert set(BACKENDS) == {"public_site", "static_page"}
```
- [ ] **Step 2: Run** → ModuleNotFoundError.
- [ ] **Step 3: Implement** `cua/rl/backends.py`:
```python
"""Environment backends (registry) and the sandbox worker pool that hosts rollout browsers."""
import logging
import os
import queue
import threading

from cua.pipeline import app_secrets, driver_session
from cua.pipeline.driver_session import DRIVER_PORT, DriverClient
from cua.pipeline.sandbox import DockerSandbox, safe_parallel_sandboxes

logger = logging.getLogger(__name__)
RL_CONTAINER_TIMEOUT = int(os.environ.get("RL_CONTAINER_TIMEOUT", str(4 * 3600)))
RL_RECYCLE_RESERVE_S = int(os.environ.get("RL_RECYCLE_RESERVE_S", "1200"))
RL_DRIVERS_PER_SANDBOX = int(os.environ.get("RL_DRIVERS_PER_SANDBOX", "1"))


def resolve_preconditions(task):
    """FAIL-CLOSED session kwargs: {} when no auth is needed, None when a required burner/session is missing."""
    pre = task.get("preconditions") or {}
    if not pre.get("requires_session"):
        return {}
    burner_id = pre.get("burner_account_id")
    if not app_secrets.is_registered_burner(burner_id):
        logger.warning(f"[{task['task_id']}] FAIL-CLOSED: burner '{burner_id}' is not registered — skipping")
        return None
    storage_state = app_secrets.load_storage_state(pre.get("session_ref"))
    if storage_state is None:
        logger.warning(f"[{task['task_id']}] FAIL-CLOSED: storage_state '{pre.get('session_ref')}' missing — skipping")
        return None
    kwargs = {"auth_config": {"auth_mode": "pre_session", "storage_state": storage_state}}
    if task.get("lane", "autonomous") == "autonomous":
        burner = app_secrets.burner_account(burner_id)
        kwargs["secrets"] = {"email": burner["email"], "password": burner["password"]}
    return kwargs


class EnvBackend:
    key = ""

    def open(self, sandbox):
        pass

    def start_url(self, task):
        return task["start_url"]

    def pre_episode(self, sandbox, task):
        pass

    def session_kwargs(self, task):
        return resolve_preconditions(task)

    def post_episode(self, sandbox, task, container_task_dir):
        sandbox.exec_command(f"rm -rf {container_task_dir}")


class PublicSiteBackend(EnvBackend):
    key = "public_site"


class StaticPageBackend(EnvBackend):
    """Serves one in-container HTML page — the no-network backend for smoke tests."""
    key = "static_page"
    port = 3000

    def __init__(self, html):
        self.html = html

    def open(self, sandbox):
        sandbox.exec_command("mkdir -p /workspace/site")
        sandbox.write_file("/workspace/site/index.html", self.html)
        sandbox.start_background(f"cd /workspace/site && python3 -m http.server {self.port}", "/workspace/site.log")
        if not sandbox.wait_for_port(self.port, 30):
            raise RuntimeError("static page server did not start")

    def start_url(self, task):
        return f"http://127.0.0.1:{self.port}/"

    def session_kwargs(self, task):
        return {}


BACKENDS = {PublicSiteBackend.key: PublicSiteBackend, StaticPageBackend.key: StaticPageBackend}


class RolloutWorker:
    """One sandbox hosting N driver slots; recycled (when idle) before the container self-expires."""

    def __init__(self, worker_id, backend, drivers_per_sandbox=RL_DRIVERS_PER_SANDBOX):
        self.worker_id = worker_id
        self.backend = backend
        self.drivers_per_sandbox = drivers_per_sandbox
        self.sandbox = None
        self.clients = []
        self.live = 0
        self.lock = threading.Lock()

    def start(self):
        self.sandbox = DockerSandbox(container_timeout=RL_CONTAINER_TIMEOUT)
        self.sandbox.start(f"cua/rl/worker-{self.worker_id}")
        self.backend.open(self.sandbox)
        driver_session.install_driver(self.sandbox)
        self.clients = []
        for i in range(self.drivers_per_sandbox):
            driver_session.start_driver(self.sandbox, port=DRIVER_PORT + i)
            self.clients.append(DriverClient(self.sandbox, port=DRIVER_PORT + i))
        logger.info(f"worker {self.worker_id}: sandbox up with {len(self.clients)} driver slot(s)")

    def client(self, slot):
        return self.clients[slot]

    def needs_recycle(self):
        if self.sandbox is None or self.sandbox.container is None or self.sandbox.remaining_seconds() < RL_RECYCLE_RESERVE_S:
            return True
        try:
            return not all(c.status().get("ok") for c in self.clients)
        except Exception:
            return True

    def recycle(self):
        self.stop()
        self.start()

    def stop(self):
        if self.sandbox is not None:
            self.sandbox.stop()
        self.sandbox = None
        self.clients = []


class WorkerPool:
    """Free-slot queue over N workers; a slot (worker, index) hosts exactly one live episode."""

    def __init__(self, backend, requested_workers, drivers_per_sandbox=RL_DRIVERS_PER_SANDBOX):
        self.workers = [RolloutWorker(i, backend, drivers_per_sandbox) for i in range(safe_parallel_sandboxes(requested_workers))]
        self.free = queue.Queue()

    @property
    def slot_count(self):
        return sum(w.drivers_per_sandbox for w in self.workers)

    def start(self):
        for worker in self.workers:
            worker.start()
            for slot in range(worker.drivers_per_sandbox):
                self.free.put((worker, slot))

    def acquire(self, timeout=None):
        worker, slot = self.free.get(timeout=timeout)
        with worker.lock:
            if worker.live == 0 and worker.needs_recycle():
                worker.recycle()
            worker.live += 1
        return worker, slot

    def release(self, worker, slot):
        with worker.lock:
            worker.live -= 1
        self.free.put((worker, slot))

    def stop(self):
        for worker in self.workers:
            worker.stop()
```
- [ ] **Step 4: Run** → `3/3 passed`. **Step 5: Report.**

---

### Task 11: Episode state machine `cua/rl/episode.py`

**Files:**
- Create: `cua/rl/episode.py`, `tests/test_episode.py`

**Interfaces:**
- Consumes: Tasks 2, 4, 5, 6, 7, 8, 10; `cua.pipeline.instrumentation.RESULTS_DIR`; `cua.pipeline.stage6_generate._safe_task_id`; `cua.sft.legacy_train_qwen.history_entries_for_step`; `DriverError`.
- Produces: `EpisodeConfig(max_steps=30, max_consecutive_parse_failures=3, loop_guard_terminates=True, verifier_mode="fallback", type_mode="replace")`; `RunContext(run_id, out_root: Path, iteration=0, policy_tag="", reward_config=dict, bridge=None)`; `Episode(task, worker, slot, sample_k, run_ctx, cfg)` with `.reset() -> observation`, `.step(text) -> (observation|None, reward|None, done, info)`, `.close(reason="external") -> record`, `.record`, `.traj_dir`, `.episode_id`; `OUTCOME_BY_REASON`.
- Record shape (normative, consumed by rollout/eval/service): `{episode_id, task_id, site, split, k, iter, traj_dir, terminal_reason, steps, ui_steps, turns:[{step_index, raw, errors, clean, executed}], success, success_source, reward_status, reward, breakdown, navigate_blocked, duration_s, check, info}`.

- [ ] **Step 1: Failing tests** `tests/test_episode.py` — drives `Episode` with a fake worker/client (no Docker):
```python
import json
import tempfile
from pathlib import Path

from cua.rl.backends import EnvBackend
from cua.rl.episode import Episode, EpisodeConfig, RunContext
from cua.rl.rewards import load_reward_config
from cua.pipeline.driver_session import DriverError


class FakeSandbox:
    def __init__(self, traj_dir):
        self.traj_dir = traj_dir
        self.commands = []

    def copy_out(self, container_path, host_path):
        Path(host_path).mkdir(parents=True, exist_ok=True)
        for p in Path(self.traj_dir).iterdir():
            (Path(host_path) / p.name).write_bytes(p.read_bytes())

    def exec_command(self, cmd, **kw):
        self.commands.append(cmd)
        return "", "", 0


class FakeClient:
    """Mimics the driver: records one step per act_sequence into a fake results dir."""
    def __init__(self, results_dir, refuse_after=None):
        self.dir = Path(results_dir)
        self.steps = []
        self.calls = []
        self.refuse_after = refuse_after

    def session(self, task_id, start_url, **kw):
        self.calls.append(("session", task_id, kw))
        self.dir.mkdir(parents=True, exist_ok=True)
        return {"ok": True}

    def observe(self):
        return {"ok": True, "b64": "UE5H", "url": "http://x/", "title": "t", "step_next": len(self.steps) + 1}

    def act_sequence(self, thought, actions):
        if self.refuse_after is not None and len(self.steps) >= self.refuse_after:
            raise DriverError("loop guard: the same action was attempted 3 times in a row with no progress.")
        n = len(self.steps) + 1
        name = f"step_{n:03d}_screenshot.png"
        (self.dir / name).write_bytes(b"PNG")
        self.steps.append({"step": n, "screenshot": name, "thought": thought, "actions": actions})
        self._write()
        return {"ok": True, "step": n, "url": "http://x/", "executed": len(actions), "error_at": None, "error": None}

    def snapshot(self):
        return {"ok": True, "url": "http://x/done", "title": "t", "snapshot": "## Page text\nall good\n"}

    def storage_state(self):
        return {"cookies": []}

    def finish(self, outcome, summary):
        self._write()
        (self.dir / "final_screenshot.png").write_bytes(b"PNG")
        return {"ok": True, "steps": len(self.steps), "dir": str(self.dir)}

    def _write(self):
        (self.dir / "trajectory.json").write_text(json.dumps({"task_id": "t", "instruction": "i", "start_url": "u", "steps": self.steps, "total_steps": len(self.steps), "success": False}))
        (self.dir / "capture_meta.json").write_text(json.dumps({"recorder": "browser_driver", "steps": {}}))


class FakeWorker:
    def __init__(self, client, sandbox):
        self._client, self.sandbox, self.backend = client, sandbox, EnvBackend()

    def client(self, slot):
        return self._client


TASK = {"task_id": "site__t1", "site": "site", "split": "train", "start_url": "http://x/", "instruction": "Do the thing", "difficulty": "easy",
        "expected_actions": 2, "backend": "static_page", "env": "test", "lane": "autonomous", "auth_mode": "none", "idempotent": True,
        "preconditions": {}, "allowed_hosts": [], "success_check": {"text_visible": "all good"}}


def _episode(tmp, refuse_after=None, **cfg):
    results = Path(tmp) / "results" / "task_x"
    client = FakeClient(results, refuse_after)
    worker = FakeWorker(client, FakeSandbox(results))
    ctx = RunContext(run_id="r", out_root=Path(tmp) / "rl_runs", iteration=1, reward_config=load_reward_config())
    return Episode(TASK, worker, 0, 2, ctx, EpisodeConfig(max_steps=4, verifier_mode="never", **cfg)), client


def test_reset_observation_and_ids():
    with tempfile.TemporaryDirectory() as tmp:
        ep, client = _episode(tmp)
        obs = ep.reset()
        assert ep.episode_id == "site__t1__i001__k02" and obs["text"] == "Task: Do the thing\nWhat action should you take?"
        assert client.calls[0][2]["task_meta"]["expected_actions"] == 2 and client.calls[0][2]["task_meta"]["autonomy"] == "autonomous"


def test_full_episode_records_steps_history_and_reward():
    with tempfile.TemporaryDirectory() as tmp:
        ep, client = _episode(tmp)
        ep.reset()
        obs, reward, done, info = ep.step("<think>c</think>\naction: click(<ref>(10, 10)</ref>, button=left)\naction: type(\"hi\")")
        assert not done and reward is None and info["parse"] == "ok" and info["executed"] == 2
        assert obs["text"] == 'Task: Do the thing\nPrevious actions: [click(10, 10), type("hi")]\nWhat action should you take?'
        obs, reward, done, info = ep.step("garbage")
        assert not done and info["parse"] == "failure" and len(client.steps) == 1
        obs, reward, done, info = ep.step("<think>d</think>\naction: done()")
        assert done and obs is None and info["terminal_reason"] == "done" and reward > 1.0
        rec = ep.record
        assert rec["success"] is True and rec["success_source"] == "success_check" and rec["ui_steps"] == 1 and rec["steps"] == 2
        assert [t["clean"] for t in rec["turns"]] == [True, False, True]
        traj = json.loads((ep.traj_dir / "trajectory.json").read_text())
        assert [a["type"] for a in traj["steps"][0]["actions"]] == ["click", "type"]
        meta = json.loads((ep.traj_dir / "metadata.json").read_text())
        assert meta["excluded_reason"] == "rl_rollout" and meta["source_repo"] == "rl:site__t1" and meta["autonomy"] == "autonomous"
        assert json.loads((ep.traj_dir / "capture_meta.json").read_text())["success_check"]["passed"] is True
        assert (ep.traj_dir / "episode.json").exists()


def test_parse_abort_after_three_failures():
    with tempfile.TemporaryDirectory() as tmp:
        ep, _ = _episode(tmp)
        ep.reset()
        for _ in range(2):
            _, _, done, _ = ep.step("nope")
            assert not done
        _, reward, done, info = ep.step("nope")
        assert done and info["terminal_reason"] == "parse_abort" and reward < 0


def test_step_cap_and_control_unexpected_and_loop_guard():
    with tempfile.TemporaryDirectory() as tmp:
        ep, _ = _episode(tmp, max_steps=1)
        ep.reset()
        _, _, done, info = ep.step("action: wait()")
        assert done and info["terminal_reason"] == "step_cap"
    with tempfile.TemporaryDirectory() as tmp:
        ep, _ = _episode(tmp)
        ep.reset()
        _, _, done, info = ep.step("action: handoff(reason=login)")
        assert done and info["terminal_reason"] == "control_unexpected"
    with tempfile.TemporaryDirectory() as tmp:
        ep, _ = _episode(tmp, refuse_after=0)
        ep.reset()
        _, _, done, info = ep.step("action: wait()")
        assert done and info["terminal_reason"] == "loop_guard" and info["refusal"] == "loop_guard"


def test_navigate_off_host_is_blocked_not_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        ep, client = _episode(tmp)
        ep.reset()
        obs, _, done, info = ep.step('action: navigate("https://evil.example/")')
        assert not done and info["refusal"] == "navigate_blocked" and client.steps == [] and obs is not None
        _, _, done, _ = ep.step("action: done()")
        assert done and ep.record["navigate_blocked"] == 1


def test_close_external_marks_abandoned():
    with tempfile.TemporaryDirectory() as tmp:
        ep, _ = _episode(tmp)
        ep.reset()
        rec = ep.close("abandoned")
        assert rec["terminal_reason"] == "abandoned" and ep.close() is rec
```
- [ ] **Step 2: Run** → ModuleNotFoundError.
- [ ] **Step 3: Implement** `cua/rl/episode.py`:
```python
"""One rollout episode over a driver slot: observe → decode → execute → record → reward."""
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from cua.pipeline.driver_session import DriverError
from cua.pipeline.instrumentation import RESULTS_DIR
from cua.pipeline.stage6_generate import _safe_task_id
from cua.rl import checks
from cua.rl.actions import decode_step
from cua.rl.executor import control_index, to_sequence_payload, validate_actions
from cua.rl.judge import judge_success
from cua.rl.observation import build_observation
from cua.rl.rewards import compute_reward
from cua.sft.legacy_train_qwen import history_entries_for_step

logger = logging.getLogger(__name__)

OUTCOME_BY_REASON = {"done": "completed"}


@dataclass
class EpisodeConfig:
    max_steps: int = 30
    max_consecutive_parse_failures: int = 3
    loop_guard_terminates: bool = True
    verifier_mode: str = "fallback"
    type_mode: str = "replace"


@dataclass
class RunContext:
    run_id: str
    out_root: Path
    iteration: int = 0
    policy_tag: str = ""
    reward_config: dict = field(default_factory=dict)
    bridge: object = None


class Episode:
    def __init__(self, task, worker, slot, sample_k, run_ctx, cfg):
        self.task, self.worker, self.slot, self.k, self.ctx, self.cfg = task, worker, slot, sample_k, run_ctx, cfg
        self.client = worker.client(slot)
        self.episode_id = f"{task['task_id']}__i{run_ctx.iteration:03d}__k{sample_k:02d}"
        self.safe_id = _safe_task_id(self.episode_id)
        self.host_dir = Path(run_ctx.out_root) / f"iter_{run_ctx.iteration:03d}" / task["task_id"] / f"k{sample_k:02d}"
        self.traj_dir = self.host_dir / f"task_{self.safe_id}"
        self.allowed_hosts = {urlparse(task["start_url"]).netloc, *task.get("allowed_hosts", [])}
        self.history, self.turns = [], []
        self.step_index = self.consecutive_parse_failures = self.navigate_blocked = 0
        self.terminal_reason, self.last_thought, self.record, self.closed = None, "", None, False
        self.started = time.time()

    def reset(self):
        backend = self.worker.backend
        kwargs = backend.session_kwargs(self.task)
        if kwargs is None:
            raise RuntimeError("preconditions_unavailable")
        backend.pre_episode(self.worker.sandbox, self.task)
        task_meta = {
            "difficulty": self.task.get("difficulty"), "expected_actions": self.task.get("expected_actions"),
            "autonomy": self.task.get("lane", "autonomous"), "auth_mode": self.task.get("auth_mode", "none"),
            "env": self.task.get("env", "live"),
            "rl": {"run_id": self.ctx.run_id, "iter": self.ctx.iteration, "k": self.k, "policy_tag": self.ctx.policy_tag},
        }
        self.client.session(self.episode_id, backend.start_url(self.task), instruction=self.task["instruction"],
                            task_meta=task_meta, type_mode=self.cfg.type_mode, **kwargs)
        return self._observe()

    def _observe(self):
        shot = self.client.observe()
        return build_observation(self.task["instruction"], self.history, shot["b64"], self.step_index)

    def step(self, response_text):
        if self.closed:
            raise RuntimeError("episode is closed")
        decoded = decode_step(response_text)
        problems = validate_actions(decoded.actions)
        turn = {"step_index": self.step_index, "raw": response_text, "errors": decoded.errors + problems,
                "clean": decoded.ok and not decoded.errors and not problems, "executed": 0}
        self.turns.append(turn)
        info = {"step_index": self.step_index, "parse_errors": turn["errors"]}
        if not decoded.actions or problems:
            self.consecutive_parse_failures += 1
            info["parse"] = "failure"
            if self.consecutive_parse_failures >= self.cfg.max_consecutive_parse_failures:
                return self._terminate("parse_abort", info)
            return self._observe(), None, False, info
        self.consecutive_parse_failures = 0
        info["parse"] = "ok"
        blocked = [a for a in decoded.actions if a["type"] == "navigate" and urlparse(a["url"]).netloc not in self.allowed_hosts]
        if blocked:
            self.navigate_blocked += 1
            info["refusal"] = "navigate_blocked"
            return self._observe(), None, False, info
        payload = to_sequence_payload(decoded.thought, decoded.actions)
        self.last_thought = decoded.thought
        try:
            result = self.client.act_sequence(payload["thought"], payload["actions"])
        except DriverError as e:
            if "loop guard" in str(e):
                info["refusal"] = "loop_guard"
                if self.cfg.loop_guard_terminates:
                    return self._terminate("loop_guard", info)
                return self._observe(), None, False, info
            info["refusal"], info["error"] = "driver_error", str(e)[:200]
            return self._terminate("driver_error", info)
        turn["executed"] = result.get("executed", 0)
        self.history.extend(history_entries_for_step({"actions": payload["actions"]}))
        self.step_index += 1
        info.update({"url": result.get("url"), "executed": turn["executed"]})
        if result.get("error"):
            info["exec_error"] = result["error"]
        control = control_index(payload["actions"])
        if control is not None:
            return self._terminate("done" if payload["actions"][control]["type"] == "done" else "control_unexpected", info)
        if self.step_index >= self.cfg.max_steps:
            return self._terminate("step_cap", info)
        return self._observe(), None, False, info

    def _terminate(self, reason, info):
        self.terminal_reason = reason
        record = self.close(reason)
        return None, record["reward"], True, {**info, "terminal_reason": reason, **record["info"]}

    def close(self, reason="external"):
        if self.closed:
            return self.record
        self.closed = True
        self.terminal_reason = self.terminal_reason or reason
        check_result = self._run_success_check()
        try:
            self.client.finish(OUTCOME_BY_REASON.get(self.terminal_reason, "failed"), self.last_thought[:400])
        except Exception as e:
            logger.warning(f"{self.episode_id}: finish failed: {e}")
        container_dir = f"{RESULTS_DIR}/task_{self.safe_id}"
        self.host_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.worker.sandbox.copy_out(container_dir, str(self.traj_dir))
        except Exception as e:
            logger.error(f"{self.episode_id}: copy_out failed: {e}")
        try:
            self.worker.backend.post_episode(self.worker.sandbox, self.task, container_dir)
        except Exception as e:
            logger.warning(f"{self.episode_id}: post_episode failed: {e}")
        self._write_metadata(check_result)
        self.record = self._build_record(check_result)
        if self.traj_dir.exists():
            (self.traj_dir / "episode.json").write_text(json.dumps(self.record, indent=2))
        return self.record

    def _run_success_check(self):
        spec = self.task.get("success_check")
        if not spec:
            return None
        try:
            ctx = checks.build_context(self.client.snapshot(), self.client.storage_state, {"terminal_reason": self.terminal_reason})
            return checks.evaluate(spec, ctx)
        except Exception as e:
            logger.warning(f"{self.episode_id}: success_check failed: {e}")
            return {"passed": False, "results": {"error": str(e)[:200]}}

    def _write_metadata(self, check_result):
        traj_file = self.traj_dir / "trajectory.json"
        if not traj_file.exists():
            return
        traj = json.loads(traj_file.read_text())
        meta_file = self.traj_dir / "capture_meta.json"
        capture_meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        if check_result is not None:
            capture_meta["success_check"] = check_result
        capture_meta["excluded_reason"] = "rl_rollout"
        meta_file.write_text(json.dumps(capture_meta, indent=2))
        metadata = {
            "task_id": traj["task_id"], "instruction": traj["instruction"], "start_url": traj["start_url"],
            "total_steps": traj["total_steps"], "success": traj["success"],
            "website": urlparse(self.task["start_url"]).netloc, "domain": self.task.get("domain", "public_web"),
            "framework": "agent", "capture_source": "agent", "instruction_source": "curated", "thought_source": "student",
            "task_type": "other", "difficulty": self.task.get("difficulty", "medium"),
            "viewport": {"width": 1280, "height": 720}, "source_repo": f"rl:{self.task['task_id']}",
            "autonomy": self.task.get("lane", "autonomous"), "auth_mode": self.task.get("auth_mode", "none"),
            "preconditions": self.task.get("preconditions", {}), "excluded_reason": "rl_rollout",
            "rl": {"run_id": self.ctx.run_id, "iter": self.ctx.iteration, "k": self.k, "terminal_reason": self.terminal_reason},
        }
        (self.traj_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def _ui_steps(self):
        traj_file = self.traj_dir / "trajectory.json"
        if not traj_file.exists():
            return self.step_index
        steps = json.loads(traj_file.read_text()).get("steps", [])
        return sum(1 for s in steps if s.get("actions") and s["actions"][0].get("type") != "done")

    def _build_record(self, check_result):
        success, source, status = None, "none", "final"
        if self.traj_dir.exists():
            success, source, status = judge_success(self.traj_dir, self.task, check_result, self.cfg.verifier_mode, self.ctx.bridge)
        record = {
            "episode_id": self.episode_id, "task_id": self.task["task_id"], "site": self.task.get("site"), "split": self.task.get("split"),
            "k": self.k, "iter": self.ctx.iteration, "traj_dir": str(self.traj_dir), "terminal_reason": self.terminal_reason,
            "steps": self.step_index, "ui_steps": self._ui_steps(), "turns": self.turns, "success": success,
            "success_source": source, "reward_status": status, "navigate_blocked": self.navigate_blocked,
            "duration_s": round(time.time() - self.started, 1), "check": check_result,
        }
        reward, breakdown = (compute_reward(record, self.task, self.ctx.reward_config) if status == "final" else (None, {}))
        record.update({"reward": reward, "breakdown": breakdown,
                       "info": {"reward_status": status, "success": success, "success_source": source, "episode_dir": str(self.traj_dir)}})
        return record
```
- [ ] **Step 4: Run** → `$PY -m tests test_episode` → `6/6 passed`. Then `$PY -m tests` → all green.
- [ ] **Step 5: Report.**

---

### Task 12: Policies + runner + rollout CLI

**Files:**
- Create: `cua/rl/policies.py`, `cua/rl/runner.py`, `cua/rl/rollout.py`, `tests/test_policies.py`

**Interfaces:**
- Produces: `POLICIES = {"openai": OpenAICompatiblePolicy, "mock": MockReplayPolicy, "scripted": ScriptedPolicy}` each with `.act(observation) -> str`; `make_policy(spec: dict) -> policy` (spec = `{"kind": ..., **kwargs}`); `run_episode(policy, task, pool, sample_k, run_ctx, cfg) -> record`; `run_batch(policy_factory, tasks, k, pool, run_ctx, cfg, on_record=None) -> list[record]`; `make_run_context(run_id, out_root, iteration, policy_tag, bridge) -> RunContext`; `write_rollouts(records, path)`; `summarize(records) -> dict`.

- [ ] **Step 1: Failing tests** `tests/test_policies.py`:
```python
import json
import tempfile
from pathlib import Path

from cua.rl.policies import MockReplayPolicy, ScriptedPolicy, make_policy
from cua.rl.runner import summarize


def test_mock_replay_emits_recorded_steps_then_done():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "trajectory.json").write_text(json.dumps({"steps": [
            {"thought": "a", "actions": [{"type": "click", "x": 1, "y": 2, "button": "left"}]},
            {"thought": "", "actions": [], "event": {"kind": "approval", "decision": "approve"}},
            {"thought": "b", "actions": [{"type": "done"}]}]}))
        policy = MockReplayPolicy(tmp)
    assert policy.act({}) == "<think>a</think>\naction: click(<ref>(1, 2)</ref>, button=left)"
    assert policy.act({}) == "<think>b</think>\naction: done()"
    assert policy.act({}).endswith("action: done()")


def test_scripted_repeats_last_text():
    policy = ScriptedPolicy(["x", "y"])
    assert [policy.act({}) for _ in range(3)] == ["x", "y", "y"]


def test_make_policy_registry():
    assert isinstance(make_policy({"kind": "scripted", "texts": ["z"]}), ScriptedPolicy)


def test_summarize_counts():
    records = [{"reward": 1.0, "success": True, "terminal_reason": "done", "turns": [{"clean": True}], "steps": 3, "reward_status": "final"},
               {"reward": None, "success": None, "terminal_reason": "step_cap", "turns": [{"clean": False}], "steps": 5, "reward_status": "deferred"}]
    s = summarize(records)
    assert s == {"episodes": 2, "final": 1, "deferred": 1, "success_rate": 0.5, "mean_reward": 1.0, "done_rate": 0.5, "parse_failure_rate": 0.5, "mean_steps_success": 3.0}
```
- [ ] **Step 2: Run** → ModuleNotFoundError.
- [ ] **Step 3: Implement**

`cua/rl/policies.py`:
```python
"""Policies turn an observation into the student's response text (registry keyed by kind)."""
import json
import os
import urllib.request
from pathlib import Path

from cua.rl.actions import encode_step
from cua.rl.observation import to_openai_messages


class OpenAICompatiblePolicy:
    def __init__(self, endpoint, model, temperature=0.0, max_tokens=512, timeout_s=180, api_key=None, **_):
        self.endpoint, self.model, self.temperature, self.max_tokens, self.timeout_s = endpoint.rstrip("/"), model, temperature, max_tokens, timeout_s
        self.api_key = api_key or os.environ.get("RL_POLICY_API_KEY", "")

    def act(self, observation):
        body = {"model": self.model, "messages": to_openai_messages(observation), "temperature": self.temperature, "max_tokens": self.max_tokens}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.endpoint}/chat/completions", data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            data = json.loads(response.read())
        return data["choices"][0]["message"].get("content") or ""

    def served_model_id(self):
        try:
            with urllib.request.urlopen(f"{self.endpoint}/models", timeout=30) as response:
                return ",".join(m["id"] for m in json.loads(response.read()).get("data", []))
        except Exception:
            return self.model


class MockReplayPolicy:
    """Replays a recorded trajectory's action steps verbatim, then done()."""

    def __init__(self, traj_dir, **_):
        traj = json.loads((Path(traj_dir) / "trajectory.json").read_text())
        self.steps = [s for s in traj["steps"] if s.get("actions")]
        self.index = 0

    def act(self, observation):
        if self.index >= len(self.steps):
            return encode_step("Nothing left to replay.", [{"type": "done"}])
        step = self.steps[self.index]
        self.index += 1
        return encode_step(step.get("thought") or "replay", step["actions"])


class ScriptedPolicy:
    def __init__(self, texts, **_):
        self.texts, self.index = list(texts), 0

    def act(self, observation):
        text = self.texts[min(self.index, len(self.texts) - 1)]
        self.index += 1
        return text


POLICIES = {"openai": OpenAICompatiblePolicy, "mock": MockReplayPolicy, "scripted": ScriptedPolicy}


def make_policy(spec):
    return POLICIES[spec["kind"]](**{k: v for k, v in spec.items() if k != "kind"})
```

`cua/rl/runner.py`:
```python
"""Run episodes in-process over a WorkerPool — shared by rollout, eval and smoke."""
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cua.rl.episode import Episode, RunContext
from cua.rl.rewards import load_reward_config

logger = logging.getLogger(__name__)


def make_run_context(run_id, out_root, iteration=0, policy_tag="", bridge=None):
    return RunContext(run_id=run_id, out_root=Path(out_root), iteration=iteration, policy_tag=policy_tag,
                      reward_config=load_reward_config(), bridge=bridge)


def run_episode(policy, task, pool, sample_k, run_ctx, cfg):
    worker, slot = pool.acquire()
    try:
        episode = Episode(task, worker, slot, sample_k, run_ctx, cfg)
        try:
            observation = episode.reset()
        except RuntimeError as e:
            return {"episode_id": episode.episode_id, "task_id": task["task_id"], "site": task.get("site"), "split": task.get("split"),
                    "k": sample_k, "iter": run_ctx.iteration, "skipped": str(e), "reward": None, "success": None,
                    "terminal_reason": "skipped", "turns": [], "steps": 0, "reward_status": "skipped"}
        done = False
        try:
            while not done:
                observation, _, done, _ = episode.step(policy.act(observation))
        except Exception as e:
            logger.exception(f"{episode.episode_id}: policy/episode error: {e}")
            episode.close("abandoned")
        return episode.record
    finally:
        pool.release(worker, slot)


def run_batch(policy_factory, tasks, k, pool, run_ctx, cfg, on_record=None):
    jobs = [(task, sample) for task in tasks for sample in range(k)]
    records = []

    def one(job):
        task, sample = job
        record = run_episode(policy_factory(), task, pool, sample, run_ctx, cfg)
        if on_record:
            on_record(record)
        return record

    with ThreadPoolExecutor(max_workers=max(1, pool.slot_count)) as executor:
        for record in executor.map(one, jobs):
            records.append(record)
    return records


def write_rollouts(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if k != "turns"} | {"turns": [{"clean": t["clean"], "errors": t["errors"]} for t in r.get("turns", [])]}) + "\n")


def summarize(records):
    final = [r for r in records if r.get("reward_status") == "final"]
    successes = [r for r in records if r.get("success")]
    turns = [t for r in records for t in r.get("turns", [])]
    return {
        "episodes": len(records), "final": len(final), "deferred": sum(1 for r in records if r.get("reward_status") == "deferred"),
        "success_rate": round(len(successes) / len(records), 4) if records else 0.0,
        "mean_reward": round(sum(r["reward"] for r in final) / len(final), 4) if final else 0.0,
        "done_rate": round(sum(1 for r in records if r.get("terminal_reason") == "done") / len(records), 4) if records else 0.0,
        "parse_failure_rate": round(sum(1 for t in turns if not t.get("clean")) / len(turns), 4) if turns else 0.0,
        "mean_steps_success": round(sum(r["steps"] for r in successes) / len(successes), 2) if successes else 0.0,
    }
```

`cua/rl/rollout.py`:
```python
"""Rollout CLI: run k samples per task with any policy, write rl_runs/<run_id>/iter_NNN/rollouts.jsonl."""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

from cua.pipeline.sandbox import install_sigterm_handler
from cua.rl.backends import BACKENDS
from cua.rl.episode import EpisodeConfig
from cua.rl.judge import AsyncBridge
from cua.rl.policies import make_policy
from cua.rl.runner import make_run_context, run_batch, summarize, write_rollouts
from cua.rl.tasks import SUITE_PATH, load_suite, select_tasks

logger = logging.getLogger(__name__)


def add_common_args(parser):
    parser.add_argument("--policy", default="openai", choices=["openai", "mock", "scripted"])
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="policy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--replay-dir", default=None, help="trajectory dir for --policy mock")
    parser.add_argument("--tasks", default=str(SUITE_PATH))
    parser.add_argument("--split", default="train")
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--drivers-per-sandbox", type=int, default=1)
    parser.add_argument("--backend", default="public_site", choices=sorted(BACKENDS))
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--verifier-mode", default="fallback", choices=["always", "fallback", "never"])
    parser.add_argument("--run-id", default=None)


def policy_spec(args):
    if args.policy == "openai":
        return {"kind": "openai", "endpoint": args.endpoint, "model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens}
    if args.policy == "mock":
        return {"kind": "mock", "traj_dir": args.replay_dir}
    return {"kind": "scripted", "texts": ["action: done()"]}


def build_pool(args):
    from_registry = BACKENDS[args.backend]
    backend = from_registry() if args.backend != "static_page" else from_registry("<html><body>static</body></html>")
    return WorkerPoolFactory(backend, args.workers, args.drivers_per_sandbox)


def WorkerPoolFactory(backend, workers, drivers_per_sandbox):
    from_pool = __import__("cua.rl.backends", fromlist=["WorkerPool"]).WorkerPool
    return from_pool(backend, workers, drivers_per_sandbox)
```
Replace the last two helpers above with a plain import — the rule is no nested imports:
```python
from cua.rl.backends import BACKENDS, WorkerPool


def build_pool(args):
    backend_cls = BACKENDS[args.backend]
    backend = backend_cls("<html><body>static</body></html>") if args.backend == "static_page" else backend_cls()
    return WorkerPool(backend, args.workers, args.drivers_per_sandbox)


def selected_tasks(args):
    suite = load_suite(args.tasks)
    tasks = select_tasks(suite, split=args.split, task_ids=args.task_ids.split(",") if args.task_ids else None)
    return suite, tasks[: args.limit] if args.limit else tasks


def main():
    parser = argparse.ArgumentParser(description="RL rollouts")
    add_common_args(parser)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--out", default="rl_runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    install_sigterm_handler()
    run_id = args.run_id or time.strftime("rollout-%Y%m%d-%H%M%S")
    suite, tasks = selected_tasks(args)
    spec = policy_spec(args)
    bridge = AsyncBridge()
    pool = build_pool(args)
    cfg = EpisodeConfig(max_steps=args.max_steps, verifier_mode=args.verifier_mode)
    run_ctx = make_run_context(run_id, args.out, args.iter, policy_tag=args.model, bridge=bridge)
    out_dir = Path(args.out) / run_id / f"iter_{args.iter:03d}"
    try:
        pool.start()
        records = run_batch(lambda: make_policy(spec), tasks, args.k, pool, run_ctx, cfg,
                            on_record=lambda r: logger.info(f"{r['episode_id']}: {r.get('terminal_reason')} reward={r.get('reward')}"))
    finally:
        pool.stop()
        bridge.close()
    write_rollouts(records, out_dir / "rollouts.jsonl")
    summary = {"run_id": run_id, "suite_sha256": suite["suite_sha256"], "policy": spec, **summarize(records)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
(Write the file with the plain-import version only; the `WorkerPoolFactory` sketch must not appear in the code.)

- [ ] **Step 4: Run** → `$PY -m tests test_policies` → `4/4 passed`; `$PY -m cua.rl.rollout --help` prints usage.
- [ ] **Step 5: Report.**

---

### Task 13: Eval CLI `cua/rl/eval.py`

**Files:**
- Create: `cua/rl/eval.py`, `tests/test_eval.py`

**Interfaces:**
- Produces: `metrics_by_split(records) -> dict[split, dict]`; `per_domain(records) -> dict[site, {n, success_rate}]`; `render_report(header, metrics, domains, baseline=None) -> str`; CLI `python -m cua.rl.eval --endpoint … [--split eval] [--baseline eval_runs/<id>/report.json] [--checkpoint-sha …] [--out eval_runs]`.

- [ ] **Step 1: Failing tests** `tests/test_eval.py`:
```python
from cua.rl.eval import metrics_by_split, per_domain, render_report

RECORDS = [
    {"split": "eval_unseen_site", "site": "a", "success": True, "terminal_reason": "done", "turns": [{"clean": True}, {"clean": True}], "steps": 2, "check": {"passed": True}, "reward_status": "final"},
    {"split": "eval_unseen_site", "site": "b", "success": False, "terminal_reason": "parse_abort", "turns": [{"clean": False}, {"clean": False}], "steps": 0, "check": None, "reward_status": "final"},
    {"split": "train", "site": "a", "success": True, "terminal_reason": "done", "turns": [{"clean": True}], "steps": 4, "check": None, "reward_status": "final"},
]


def test_metrics_by_split():
    m = metrics_by_split(RECORDS)["eval_unseen_site"]
    assert m["n"] == 2 and m["verified_success"] == 0.5 and m["parse_failure_rate"] == 0.5 and m["parse_abort_rate"] == 0.5
    assert m["done_rate"] == 0.5 and m["mean_steps_success"] == 2.0 and m["check_pass_rate"] == 1.0


def test_per_domain_and_report_with_baseline_delta():
    domains = per_domain(RECORDS)
    assert domains["a"] == {"n": 2, "success_rate": 1.0}
    report = render_report({"run_id": "r", "model": "m"}, metrics_by_split(RECORDS), domains,
                           baseline={"metrics": {"eval_unseen_site": {"verified_success": 0.25}}})
    assert "| eval_unseen_site |" in report and "+0.25" in report and "run_id: r" in report
```
- [ ] **Step 2: Run** → ModuleNotFoundError.
- [ ] **Step 3: Implement** `cua/rl/eval.py`:
```python
"""E3 eval: run the student on held-out tasks (temperature 0, k=1) and report per-split metrics with baseline deltas."""
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from cua.pipeline.sandbox import install_sigterm_handler
from cua.rl.episode import EpisodeConfig
from cua.rl.judge import AsyncBridge
from cua.rl.policies import make_policy
from cua.rl.rollout import add_common_args, build_pool, policy_spec, selected_tasks
from cua.rl.runner import make_run_context, run_batch, write_rollouts

logger = logging.getLogger(__name__)
METRIC_KEYS = ("verified_success", "check_pass_rate", "parse_failure_rate", "parse_abort_rate", "done_rate", "mean_steps_success", "loop_guard_rate")


def _rate(items, predicate):
    return round(sum(1 for i in items if predicate(i)) / len(items), 4) if items else 0.0


def metrics_by_split(records):
    out = {}
    for split in sorted({r.get("split") for r in records}):
        rs = [r for r in records if r.get("split") == split]
        turns = [t for r in rs for t in r.get("turns", [])]
        checked = [r for r in rs if r.get("check")]
        successes = [r for r in rs if r.get("success")]
        out[split] = {
            "n": len(rs),
            "verified_success": _rate(rs, lambda r: bool(r.get("success"))),
            "check_pass_rate": _rate(checked, lambda r: r["check"]["passed"]),
            "parse_failure_rate": _rate(turns, lambda t: not t.get("clean")),
            "parse_abort_rate": _rate(rs, lambda r: r.get("terminal_reason") == "parse_abort"),
            "done_rate": _rate(rs, lambda r: r.get("terminal_reason") == "done"),
            "mean_steps_success": round(sum(r["steps"] for r in successes) / len(successes), 2) if successes else 0.0,
            "loop_guard_rate": _rate(rs, lambda r: r.get("terminal_reason") == "loop_guard"),
        }
    return out


def per_domain(records):
    out = {}
    for site in sorted({r.get("site") for r in records}):
        rs = [r for r in records if r.get("site") == site]
        out[site] = {"n": len(rs), "success_rate": _rate(rs, lambda r: bool(r.get("success")))}
    return out


def render_report(header, metrics, domains, baseline=None):
    lines = ["# Student eval report", ""]
    lines += [f"- {k}: {v}" for k, v in header.items()]
    lines += ["", "| split | n | " + " | ".join(METRIC_KEYS) + " |", "|---|---|" + "---|" * len(METRIC_KEYS)]
    for split, m in metrics.items():
        cells = []
        for key in METRIC_KEYS:
            base = ((baseline or {}).get("metrics") or {}).get(split, {}).get(key)
            cells.append(f"{m[key]} ({m[key] - base:+.2f})" if isinstance(base, (int, float)) else str(m[key]))
        lines.append(f"| {split} | {m['n']} | " + " | ".join(cells) + " |")
    lines += ["", "| site | n | success_rate |", "|---|---|---|"]
    lines += [f"| {site} | {d['n']} | {d['success_rate']} |" for site, d in domains.items()]
    return "\n".join(lines) + "\n"


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Student eval on held-out tasks")
    add_common_args(parser)
    parser.set_defaults(split="eval", temperature=0.0, max_steps=40, verifier_mode="always")
    parser.add_argument("--baseline", default=None, help="report.json of a previous eval to diff against")
    parser.add_argument("--checkpoint-sha", default="")
    parser.add_argument("--out", default="eval_runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    install_sigterm_handler()
    run_id = args.run_id or time.strftime("eval-%Y%m%d-%H%M%S")
    suite, tasks = selected_tasks(args)
    spec = policy_spec(args)
    probe = make_policy(spec)
    header = {"run_id": run_id, "model": getattr(probe, "served_model_id", lambda: args.model)(), "checkpoint_sha": args.checkpoint_sha,
              "git_head": git_head(), "suite_sha256": suite["suite_sha256"], "tasks": len(tasks), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    bridge, pool = AsyncBridge(), build_pool(args)
    cfg = EpisodeConfig(max_steps=args.max_steps, verifier_mode=args.verifier_mode)
    run_ctx = make_run_context(run_id, args.out, 0, policy_tag=args.model, bridge=bridge)
    try:
        pool.start()
        records = run_batch(lambda: make_policy(spec), tasks, 1, pool, run_ctx, cfg,
                            on_record=lambda r: logger.info(f"{r['episode_id']}: {r.get('terminal_reason')} success={r.get('success')}"))
    finally:
        pool.stop()
        bridge.close()
    out_dir = Path(args.out) / run_id
    write_rollouts(records, out_dir / "rollouts.jsonl")
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    metrics, domains = metrics_by_split(records), per_domain(records)
    (out_dir / "report.json").write_text(json.dumps({"header": header, "metrics": metrics, "domains": domains}, indent=2))
    report = render_report(header, metrics, domains, baseline)
    (out_dir / "report.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- [ ] **Step 4: Run** → `2/2 passed`; `$PY -m cua.rl.eval --help`.
- [ ] **Step 5: Report.**

---

### Task 14: Env service `cua/rl/service.py`

**Files:**
- Create: `cua/rl/service.py`, `tests/test_service.py`

**Interfaces:**
- Produces: `EnvService(pool, tasks_by_id, run_ctx_factory, cfg)` with `.reset(body) .step(body) .close(body) .episode(eid) .health() .reap(max_idle_s)`; `make_handler(service, token) -> BaseHTTPRequestHandler subclass`; `serve(service, host, port, token) -> ThreadingHTTPServer`; HTTP contract from the spec (`X-RL-Token`, 503 `{retry_after}` when no slot).
- `run_ctx_factory(run_id: str, iteration: int, policy_tag: str) -> RunContext`.

- [ ] **Step 1: Failing tests** `tests/test_service.py` (exercises the HTTP layer with a fake pool + fake episodes; no Docker):
```python
import json
import queue
import threading
import urllib.error
import urllib.request

from cua.rl.service import EnvService, serve


class FakePool:
    def __init__(self, slots=1):
        self.free = queue.Queue()
        for i in range(slots):
            self.free.put(("w", i))
        self.slot_count = slots

    def acquire(self, timeout=None):
        return self.free.get(timeout=timeout)

    def release(self, worker, slot):
        self.free.put((worker, slot))


class FakeEpisode:
    def __init__(self, task, worker, slot, k, ctx, cfg):
        self.episode_id, self.n, self.record = f"{task['task_id']}__k{k}", 0, None

    def reset(self):
        return {"text": "obs0", "image_b64": "", "system": "s", "step_index": 0}

    def step(self, text):
        self.n += 1
        if text == "action: done()":
            self.record = {"reward": 1.0, "info": {"reward_status": "final"}}
            return None, 1.0, True, {"terminal_reason": "done"}
        return {"text": f"obs{self.n}", "image_b64": "", "system": "s", "step_index": self.n}, None, False, {"parse": "ok"}

    def close(self, reason="external"):
        self.record = self.record or {"reward": 0.0, "info": {"reward_status": "final", "terminal_reason": reason}}
        return self.record


def _post(url, body, token="t"):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "X-RL-Token": token})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def test_http_round_trip_and_slot_exhaustion():
    tasks = {"t1": {"task_id": "t1", "split": "train"}}
    service = EnvService(FakePool(1), tasks, lambda run_id, it, tag: None, None, episode_cls=FakeEpisode, reset_timeout_s=0.2)
    server = serve(service, "127.0.0.1", 0, "t")
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        health = json.loads(urllib.request.urlopen(urllib.request.Request(f"{base}/health", headers={"X-RL-Token": "t"})).read())
        assert health["ok"] and health["free_slots"] == 1
        first = _post(f"{base}/reset", {"task_id": "t1", "run_id": "r", "iter": 0, "sample_k": 0})
        assert first["observation"]["text"] == "obs0"
        try:
            _post(f"{base}/reset", {"task_id": "t1"})
            raise AssertionError("second reset must 503")
        except urllib.error.HTTPError as e:
            assert e.code == 503 and json.loads(e.read())["retry_after"] > 0
        stepped = _post(f"{base}/step", {"episode_id": first["episode_id"], "response_text": "action: wait()"})
        assert stepped["done"] is False and stepped["observation"]["text"] == "obs1"
        final = _post(f"{base}/step", {"episode_id": first["episode_id"], "response_text": "action: done()"})
        assert final["done"] is True and final["reward"] == 1.0 and final["observation"] is None
        assert json.loads(urllib.request.urlopen(urllib.request.Request(f"{base}/episode/{first['episode_id']}", headers={"X-RL-Token": "t"})).read())["reward"] == 1.0
        assert service.pool.free.qsize() == 1
        try:
            _post(f"{base}/reset", {"task_id": "t1"}, token="wrong")
            raise AssertionError("bad token must 401")
        except urllib.error.HTTPError as e:
            assert e.code == 401
    finally:
        server.shutdown()
```
- [ ] **Step 2: Run** → ModuleNotFoundError.
- [ ] **Step 3: Implement** `cua/rl/service.py`:
```python
"""HTTP env service for the GPU-side trainer plugin: reset/step/close over the worker pool (loopback, token-guarded)."""
import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cua.pipeline.sandbox import install_sigterm_handler
from cua.rl.episode import Episode, EpisodeConfig
from cua.rl.judge import AsyncBridge
from cua.rl.rollout import build_pool
from cua.rl.runner import make_run_context
from cua.rl.tasks import SUITE_PATH, load_suite, select_tasks

logger = logging.getLogger(__name__)
RL_ENV_HOST = os.environ.get("RL_ENV_HOST", "127.0.0.1")
RL_ENV_PORT = int(os.environ.get("RL_ENV_PORT", "8900"))
RL_ENV_TOKEN = os.environ.get("RL_ENV_TOKEN", "")
RL_RESET_TIMEOUT_S = float(os.environ.get("RL_RESET_TIMEOUT_S", "300"))
RL_EPISODE_IDLE_S = float(os.environ.get("RL_EPISODE_IDLE_S", "600"))


class EnvService:
    def __init__(self, pool, tasks_by_id, run_ctx_factory, cfg, episode_cls=Episode, reset_timeout_s=RL_RESET_TIMEOUT_S):
        self.pool, self.tasks, self.run_ctx_factory, self.cfg = pool, tasks_by_id, run_ctx_factory, cfg
        self.episode_cls, self.reset_timeout_s = episode_cls, reset_timeout_s
        self.episodes = {}
        self.lock = threading.Lock()
        self.finished = {}

    def health(self):
        return {"ok": True, "free_slots": self.pool.free.qsize(), "live_episodes": len(self.episodes), "tasks": len(self.tasks)}

    def reset(self, body):
        task = self.tasks[body["task_id"]]
        worker, slot = self.pool.acquire(timeout=self.reset_timeout_s)
        run_ctx = self.run_ctx_factory(body.get("run_id", "service"), int(body.get("iter", 0)), body.get("policy_tag", ""))
        episode = self.episode_cls(task, worker, slot, int(body.get("sample_k", 0)), run_ctx, self.cfg)
        try:
            observation = episode.reset()
        except Exception:
            self.pool.release(worker, slot)
            raise
        episode_id = uuid.uuid4().hex
        with self.lock:
            self.episodes[episode_id] = {"episode": episode, "lock": threading.Lock(), "worker": worker, "slot": slot, "last": time.time()}
        return {"episode_id": episode_id, "observation": observation}

    def _entry(self, episode_id):
        with self.lock:
            entry = self.episodes.get(episode_id)
        if entry is None:
            raise KeyError(f"unknown or finished episode {episode_id}")
        return entry

    def _finish(self, episode_id, entry):
        with self.lock:
            self.episodes.pop(episode_id, None)
            self.finished[episode_id] = entry["episode"].record
        self.pool.release(entry["worker"], entry["slot"])

    def step(self, body):
        episode_id = body["episode_id"]
        entry = self._entry(episode_id)
        with entry["lock"]:
            entry["last"] = time.time()
            observation, reward, done, info = entry["episode"].step(body["response_text"])
        if done:
            self._finish(episode_id, entry)
        return {"observation": observation, "reward": reward, "done": done, "info": info}

    def close(self, body):
        episode_id = body["episode_id"]
        entry = self._entry(episode_id)
        with entry["lock"]:
            record = entry["episode"].close(body.get("reason", "external"))
        self._finish(episode_id, entry)
        return {"reward": record.get("reward"), "info": record.get("info", {})}

    def episode(self, episode_id):
        record = self.finished.get(episode_id)
        if record is not None:
            return {"status": "finished", "reward": record.get("reward"), "breakdown": record.get("breakdown"), "episode_dir": record.get("traj_dir")}
        self._entry(episode_id)
        return {"status": "live"}

    def reap(self, max_idle_s=RL_EPISODE_IDLE_S):
        with self.lock:
            stale = [eid for eid, e in self.episodes.items() if time.time() - e["last"] > max_idle_s]
        for episode_id in stale:
            logger.warning(f"reaping idle episode {episode_id}")
            self.close({"episode_id": episode_id, "reason": "abandoned"})


def make_handler(service, token):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self):
            return not token or self.headers.get("X-RL-Token") == token

        def _dispatch(self, method):
            if not self._authorized():
                return self._send(401, {"error": "bad token"})
            path = self.path.split("?")[0]
            query = self.path.split("?")[1] if "?" in self.path else ""
            try:
                body = {}
                if method == "POST":
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length) or b"{}")
                if method == "GET" and path == "/health":
                    return self._send(200, service.health())
                if method == "GET" and path == "/tasks":
                    split = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("split")
                    return self._send(200, {"tasks": [t for t in service.tasks.values() if not split or t.get("split") == split]})
                if method == "GET" and path.startswith("/episode/"):
                    return self._send(200, service.episode(path[len("/episode/"):]))
                if method == "POST" and path == "/reset":
                    return self._send(200, service.reset(body))
                if method == "POST" and path == "/step":
                    return self._send(200, service.step(body))
                if method == "POST" and path == "/close":
                    return self._send(200, service.close(body))
                return self._send(404, {"error": f"no route {method} {path}"})
            except queue.Empty:
                return self._send(503, {"error": "no free slot", "retry_after": 30})
            except KeyError as e:
                return self._send(404, {"error": str(e)})
            except Exception as e:
                logger.exception(f"{method} {path} failed")
                return self._send(500, {"error": str(e)[:300]})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def log_message(self, fmt, *args):
            logger.debug(fmt % args)

    return Handler


def serve(service, host, port, token):
    return ThreadingHTTPServer((host, port), make_handler(service, token))


def main():
    parser = argparse.ArgumentParser(description="RL env service")
    parser.add_argument("--tasks", default=str(SUITE_PATH))
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--drivers-per-sandbox", type=int, default=1)
    parser.add_argument("--backend", default="public_site")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--verifier-mode", default="fallback", choices=["always", "fallback", "never"])
    parser.add_argument("--run-id", default=time.strftime("service-%Y%m%d-%H%M%S"))
    parser.add_argument("--out", default="rl_runs")
    parser.add_argument("--host", default=RL_ENV_HOST)
    parser.add_argument("--port", type=int, default=RL_ENV_PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    install_sigterm_handler()
    suite = load_suite(args.tasks)
    tasks = {t["task_id"]: t for t in select_tasks(suite, split=args.split)}
    bridge = AsyncBridge()
    pool = build_pool(args)
    cfg = EpisodeConfig(max_steps=args.max_steps, verifier_mode=args.verifier_mode)
    factory = lambda run_id, iteration, tag: make_run_context(run_id or args.run_id, args.out, iteration, policy_tag=tag, bridge=bridge)
    service = EnvService(pool, tasks, factory, cfg)
    server = serve(service, args.host, args.port, RL_ENV_TOKEN)
    reaper = threading.Thread(target=lambda: [time.sleep(60) or service.reap() for _ in iter(int, 1)], daemon=True)
    try:
        pool.start()
        reaper.start()
        logger.info(f"env service on {args.host}:{args.port} with {pool.slot_count} slot(s), {len(tasks)} tasks (suite {suite['suite_sha256'][:12]})")
        server.serve_forever()
    finally:
        server.shutdown()
        pool.stop()
        bridge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
`build_pool(args)` in `cua/rl/rollout.py` reads `args.backend/workers/drivers_per_sandbox` — the service parser defines all three, so it is reusable as-is.

- [ ] **Step 4: Run** → `$PY -m tests test_service` → `1/1 passed`; whole suite green.
- [ ] **Step 5: Report.**

---

### Task 15: End-to-end smoke `cua/rl/smoke.py` + wiring (Makefile, .gitignore, CLAUDE.md, `excluded_reason` fix)

**Files:**
- Create: `cua/rl/smoke.py`
- Modify: `Makefile` (after `test-driver`), `.gitignore`, `CLAUDE.md` (new section after "Training"), `cua/pipeline/stage6_generate.py` (`_excluded`), `cua/pipeline/corpus_audit.py` (`is_trainable`), `cua/sft/legacy_train_qwen.py` (`convert_trajectories` exclusion read)

- [ ] **Step 1: Write the smoke (the failing end-to-end test)** — `cua/rl/smoke.py`:
```python
"""Docker-only end-to-end check of the RL harness: static page + scripted policy through Episode and the HTTP service, no LLM."""
import base64
import json
import logging
import shutil
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

from cua.pipeline.corpus_audit import audit_trajectory
from cua.pipeline.driver_smoke import TEST_PAGE
from cua.pipeline.instrumentation import RESULTS_DIR
from cua.rl.backends import StaticPageBackend, WorkerPool
from cua.rl.episode import Episode, EpisodeConfig
from cua.rl.policies import ScriptedPolicy
from cua.rl.runner import make_run_context, run_episode
from cua.rl.service import EnvService, serve
from cua.sft.legacy_train_qwen import convert_trajectory_to_examples

logger = logging.getLogger(__name__)

TASK = {"task_id": "smoke__greet", "site": "smoke", "split": "train", "start_url": "http://127.0.0.1:3000/", "instruction": "Greet yourself by name",
        "difficulty": "easy", "expected_actions": 3, "backend": "static_page", "env": "test", "lane": "autonomous", "auth_mode": "none",
        "idempotent": True, "preconditions": {}, "allowed_hosts": [], "success_check": {"text_visible": "hello Smokey", "done_emitted": True}}


def discover_coords(client, needles):
    """Resolve page elements to pixel centers through the teacher's ref path (a throwaway probe session)."""
    session = client.session("smoke_probe", TASK["start_url"], instruction="probe")
    coords = {}
    for needle in needles:
        ref = next(int(l[1:l.index("]")]) for l in session["snapshot"].splitlines() if l.startswith("[") and needle in l)
        client.act({"kind": "click", "ref": ref, "thought": f"probe {needle}"})
        recorded = json.loads(client.sandbox.read_file(f"{RESULTS_DIR}/task_smoke_probe/trajectory.json"))
        action = recorded["steps"][-1]["actions"][0]
        coords[needle] = (action["x"], action["y"])
    client.finish("failed", "probe")
    client.sandbox.exec_command(f"rm -rf {RESULTS_DIR}/task_smoke_probe")
    return coords


def script(coords):
    (nx, ny), (gx, gy) = coords["Your name"], coords["Greet"]
    return [
        f"<think>I will type my name into the field.</think>\naction: click(<ref>({nx}, {ny})</ref>, button=left)\naction: type(\"Smokey\")",
        "this is not an action",
        f"<think>Submitting the greeting.</think>\naction: click(<ref>({gx}, {gy})</ref>, button=left)",
        "<think>Checking below the fold.</think>\naction: scroll(<ref>(640, 360)</ref>, scrollX=0, scrollY=300)",
        "<think>Back to the top.</think>\naction: keypress([\"Home\"])",
        "<think>The page greets me as Smokey; the task is complete.</think>\naction: done()",
    ]


class RecordingPolicy(ScriptedPolicy):
    def __init__(self, texts):
        super().__init__(texts)
        self.observations = []

    def act(self, observation):
        self.observations.append(observation)
        return super().act(observation)


def check_record(record, policy, traj_dir):
    assert record["terminal_reason"] == "done", record["terminal_reason"]
    assert record["check"]["passed"] and record["success"] is True and record["success_source"] == "success_check"
    assert [t["clean"] for t in record["turns"]] == [True, False, True, True, True, True]
    assert record["breakdown"]["components"]["format_validity"] == round(5 / 6, 4) or abs(record["breakdown"]["components"]["format_validity"] - 5 / 6) < 1e-6
    assert record["reward"] > 1.0, record["reward"]
    traj = json.loads((traj_dir / "trajectory.json").read_text())
    assert [a["type"] for a in traj["steps"][0]["actions"]] == ["click", "type"], traj["steps"][0]["actions"]
    assert len(traj["steps"]) == 5 and traj["steps"][-1]["actions"] == [{"type": "done"}]
    served = [o for o in policy.observations]
    emitted = [served[i] for i in (0, 2, 3, 4, 5)]
    for step, obs in zip(traj["steps"], emitted):
        assert (traj_dir / step["screenshot"]).read_bytes() == base64.b64decode(obs["image_b64"]), f"PNG parity broken at {step['step']}"
    examples, _ = convert_trajectory_to_examples(traj, traj_dir)
    for example, obs in zip(examples, emitted):
        assert example["messages"][1]["content"][1]["text"] == obs["text"], "SFT input != RL observation"
    defects = audit_trajectory(traj_dir)
    assert defects == [], defects
    assert json.loads((traj_dir / "metadata.json").read_text())["excluded_reason"] == "rl_rollout"
    logger.info("episode assertions PASSED")


def run_smoke():
    out_root = Path(tempfile.mkdtemp(prefix="rl_smoke_"))
    pool = WorkerPool(StaticPageBackend(TEST_PAGE), requested_workers=1, drivers_per_sandbox=1)
    try:
        pool.start()
        coords = discover_coords(pool.workers[0].client(0), ["Your name", "Greet"])
        texts = script(coords)
        cfg = EpisodeConfig(max_steps=10, verifier_mode="never")
        run_ctx = make_run_context("smoke", out_root, 0, "scripted")
        policy = RecordingPolicy(texts)
        record = run_episode(policy, TASK, pool, 0, run_ctx, cfg)
        check_record(record, policy, Path(record["traj_dir"]))

        service = EnvService(pool, {TASK["task_id"]: TASK}, lambda run_id, it, tag: make_run_context(run_id, out_root, it, tag), cfg)
        server = serve(service, "127.0.0.1", 0, "smoke")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            def post(path, body):
                request = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "X-RL-Token": "smoke"})
                with urllib.request.urlopen(request, timeout=300) as response:
                    return json.loads(response.read())
            first = post("/reset", {"task_id": TASK["task_id"], "run_id": "smoke-http", "iter": 1, "sample_k": 3})
            done, reward, step = False, None, None
            for text in texts:
                step = post("/step", {"episode_id": first["episode_id"], "response_text": text})
                done, reward = step["done"], step["reward"]
                if done:
                    break
            assert done and reward is not None and reward > 1.0, step
            assert (out_root / "iter_001" / TASK["task_id"] / "k03").exists()
        finally:
            server.shutdown()
        logger.info("RL smoke PASSED")
        return True
    finally:
        pool.stop()
        shutil.rmtree(out_root, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ok = False
    try:
        ok = run_smoke()
    except AssertionError as e:
        logger.error(f"RL smoke FAILED: {e}")
    except Exception as e:
        logger.exception(f"RL smoke ERRORED: {e}")
    sys.exit(0 if ok else 1)
```

- [ ] **Step 2: Run** → `$PY -m cua.rl.smoke` → Expected first failure: `discover_coords` needs `client.sandbox` (it exists on `DriverClient`) — expected PASS of the probe; the parity assertions are the real gate. If `audit_trajectory` reports `step N: empty thought` for the scripted steps, the scripted texts above all carry a `<think>`; investigate rather than relax.

- [ ] **Step 3: Wiring**

Makefile (after `test-driver`):
```make
test-rl-unit: ## Unit tests for the RL rollout harness (no Docker, no LLM)
	python -m tests

test-rl: ## Docker-only end-to-end smoke of the RL harness (static page + scripted policy, no LLM)
	python -m cua.rl.smoke

rl-tasks: ## Build + check the RL task suite (disjoint from anything trained on)
	python -m cua.rl.tasks build $(if $(RL_TRAJ_DIR),--trajectories $(RL_TRAJ_DIR)) && python -m cua.rl.tasks check $(if $(RL_TRAJ_DIR),--trajectories $(RL_TRAJ_DIR))

rl-serve: ## Start the RL env service for the GPU trainer (127.0.0.1:8900)
	python -m cua.rl.service --split $(or $(RL_SPLIT),train) --workers $(or $(RL_WORKERS),3) $(RL_ARGS)

rl-rollout: ## Local rollouts against an OpenAI-compatible endpoint (RL_ENDPOINT, RL_MODEL, RL_K)
	python -m cua.rl.rollout --endpoint $(or $(RL_ENDPOINT),http://127.0.0.1:8001/v1) --model $(or $(RL_MODEL),policy) --k $(or $(RL_K),4) $(RL_ARGS)

rl-eval: ## E3 student eval on held-out tasks (RL_ENDPOINT, RL_MODEL, RL_BASELINE=report.json)
	python -m cua.rl.eval --endpoint $(or $(RL_ENDPOINT),http://127.0.0.1:8001/v1) --model $(or $(RL_MODEL),policy) $(if $(RL_BASELINE),--baseline $(RL_BASELINE)) $(RL_ARGS)

eval: rl-eval ## Alias for rl-eval
```
`.gitignore`: append `rl_runs/` and `eval_runs/`.

`cua/pipeline/stage6_generate.py` `_excluded`: after writing capture_meta, add
```python
    metadata_file = traj_dir / "metadata.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    metadata["excluded_reason"] = reason
    metadata_file.write_text(json.dumps(metadata, indent=2))
```
`cua/pipeline/corpus_audit.py` `is_trainable`: load `capture_meta = _load_json(traj_dir / "capture_meta.json")` and change the guard to `if metadata.get("excluded_reason") or capture_meta.get("excluded_reason"):`.
`cua/sft/legacy_train_qwen.py` `convert_trajectories`: change `if metadata.get("excluded_reason"):` to `if metadata.get("excluded_reason") or capture_meta.get("excluded_reason"):` (capture_meta is loaded just above it).

`CLAUDE.md`: add after the "## Training" section:
```markdown
## RL rollout harness (`cua/rl/`)

Policy-agnostic rollouts of the student over the same sandboxed driver (`/observe` + `/act kind=sequence`): one recorded step per policy turn, observation PNG bytes == recorded screenshot, observation text built by `cua.sft.legacy_train_qwen.build_step_user_text` (SFT input == RL input). `cua.rl.actions.decode_step` is the inverse of `format_action_for_training`. Rollouts land under `rl_runs/<run_id>/iter_NNN/<task>/kKK/task_*/` (never `trajectories/`; `metadata.excluded_reason=rl_rollout`). Reward: `success_check` from `cua/rl/rl_tasks.json` → `verify_trajectory` fallback (`--verifier-mode`), weights in `cua/rl/reward_config.json`. Commands: `make test-rl-unit`, `make test-rl` (Docker), `make rl-tasks`, `make rl-serve` (env service :8900 for the GPU trainer), `make rl-rollout`, `make rl-eval`.
```

- [ ] **Step 4: Run everything**

Run: `$PY -m tests` → all passed. `$PY -m cua.pipeline.driver_smoke` → PASSED. `$PY -m cua.rl.smoke` → `RL smoke PASSED`. `$PY -m cua.rl.tasks check --trajectories /Users/rmajor/unik/maga/dataset_gather/trajectories` → `problems 0`. `$PY -m cua.pipeline.corpus_audit /Users/rmajor/unik/maga/dataset_gather/trajectories` still exits as before the change (record its exit code before and after).

- [ ] **Step 5: Report** — full `git status --short` (untracked `cua/rl/`, `docs/superpowers/{specs,plans}/…`, modified tracked files) and the smoke log tail. Do not `git add`.

---

## Self-review

- **Spec coverage:** 1.1 → Task 1; 1.2 → Task 2; 1.3 → Task 3; 1.4 → Tasks 4, 10, 11; 1.5 → Tasks 6, 7, 8; 1.6 → Task 9; 1.7 → Tasks 12, 13, 14; 1.8 → Task 15 (incl. the `excluded_reason` fix). Deferred per spec: `repo_app` backend, guarded lane, direct transport.
- **Placeholders:** none; every code step is complete. The one editorial note in Task 12 (drop the `WorkerPoolFactory` sketch) is an instruction, not a placeholder.
- **Type consistency:** `Episode.step` returns `(obs|None, reward|None, done, info)` everywhere (Tasks 11, 12, 14, 15); `record` keys used by `summarize`/`metrics_by_split`/`write_rollouts` (`reward, success, terminal_reason, turns[].clean, steps, reward_status, split, site, check`) are all produced by `Episode._build_record`; `RunContext.reward_config` is filled by `make_run_context`; `DriverClient.observe()` returns `b64`/`mime`/`step_next` (Tasks 3, 11, 15); `WorkerPool.acquire/release/free/slot_count` match the `FakePool` in Task 14.
