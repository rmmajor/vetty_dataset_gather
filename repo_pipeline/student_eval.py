"""Episode runner that measures a served computer-use model on real browser tasks, before vs after fine-tuning."""
import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from repo_pipeline import driver_session
from repo_pipeline.agent_capture import verify_trajectory
from repo_pipeline.driver_session import DriverClient, DriverError
from repo_pipeline.eval_task_sources import (
    EvalTask,
    TaskLoadConfig,
    TaskSourceUnavailable,
    TASK_SOURCES,
    episode_dir_name,
    load_tasks,
    load_tasks_from_file,
)
from repo_pipeline.instrumentation import RESULTS_DIR
from repo_pipeline.llm_config import MODEL_FOR_REASONING
from repo_pipeline.sandbox import DockerSandbox, install_sigterm_handler, safe_parallel_sandboxes
from repo_pipeline.sdk_runner import UsageLimitReached
from train_qwen import (
    HISTORY_WINDOW,
    SYSTEM_PROMPT,
    format_action_for_training,
    format_action_summary,
)

logger = logging.getLogger(__name__)

EVAL_RUNS_DIR = Path(os.environ.get("EVAL_RUNS_DIR", "eval_runs"))
# Headroom below a container's self-expiry at which the next episode gets a fresh sandbox.
EPISODE_TIME_RESERVE = int(os.environ.get("EVAL_EPISODE_TIME_RESERVE", "900"))
EVAL_CONTAINER_TIMEOUT = int(os.environ.get("EVAL_CONTAINER_TIMEOUT", "10800"))
MAX_CONSECUTIVE_PARSE_FAILURES = int(os.environ.get("EVAL_MAX_PARSE_FAILURES", "3"))
# Steps in a row that execute NOTHING (refused, errored or unparseable) before the episode is
# abandoned. The driver's own loop guard is off for eval, so this is what stops a miscalibrated
# model burning a full step cap on refusals.
MAX_UNPRODUCTIVE_STEPS = int(os.environ.get("EVAL_MAX_UNPRODUCTIVE_STEPS", "5"))
NO_THINK_PLACEHOLDER = "(model emitted no <think> block)"


class StudentError(Exception):
    """The served model could not be reached or returned an unusable response."""


# ---------------------------------------------------------------------------
# Prompt construction — byte-identical to the per-step training example.
# ---------------------------------------------------------------------------


def build_user_text(instruction: str, action_history: list[str]) -> str:
    """The exact user text `train_qwen.convert_trajectory_to_examples` emits per step."""
    user_text = f"Task: {instruction}"
    if action_history:
        window = action_history[-HISTORY_WINDOW:]
        omitted = len(action_history) - len(window)
        prefix = f"... {omitted} earlier actions, " if omitted > 0 else ""
        user_text += f"\nPrevious actions: [{prefix}{', '.join(window)}]"
    user_text += "\nWhat action should you take?"
    return user_text


def build_messages(instruction: str, action_history: list[str], image_b64: str, mime: str) -> list:
    """OpenAI chat messages whose text is identical to training; only the image transport differs."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            {"type": "text", "text": build_user_text(instruction, action_history)},
        ]},
    ]


# ---------------------------------------------------------------------------
# Student backends. Registry, not special cases: a new serving stack is one entry.
# ---------------------------------------------------------------------------


class StudentClient:
    def complete(self, messages: list) -> str:
        raise NotImplementedError


class OpenAIChatStudent(StudentClient):
    """Any OpenAI-compatible /chat/completions endpoint (vLLM, `swift deploy`, SGLang, a hosted API)."""

    def __init__(self, endpoint: str, model: str, api_key: str = "", max_tokens: int = 512,
                 temperature: float = 0.0, timeout_s: int = 180, retries: int = 2):
        self.url = self._chat_url(endpoint)
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.retries = retries

    @staticmethod
    def _chat_url(endpoint: str) -> str:
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            return endpoint
        return f"{endpoint}/chat/completions"

    def complete(self, messages: list) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout_s)
            except requests.RequestException as e:
                last_error = f"transport error: {e}"
                continue
            if response.status_code != 200:
                raise StudentError(f"{self.url} returned HTTP {response.status_code}: {response.text[:300]}")
            try:
                choice = response.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, ValueError) as e:
                raise StudentError(f"unexpected response shape from {self.url}: {e}")
            if isinstance(choice, list):
                choice = "".join(part.get("text", "") for part in choice if isinstance(part, dict))
            return choice or ""
        raise StudentError(f"{self.url} unreachable after {self.retries + 1} attempts ({last_error})")


class ReplayStudent(StudentClient):
    """Zero-model student replaying a recorded trajectory's actions in the training wire format."""

    def __init__(self, trajectory: dict):
        self.steps = trajectory.get("steps", [])
        self.index = 0

    def complete(self, messages: list) -> str:
        if self.index >= len(self.steps):
            return "<think>The recorded run is finished and the final state is on screen.</think>\naction: done()"
        step = self.steps[self.index]
        self.index += 1
        thought = step.get("thought") or "Replaying the next recorded action."
        actions = step.get("actions") or []
        lines = "\n".join(format_action_for_training(a) for a in actions)
        return f"<think>{thought}</think>\n{lines}"


def build_openai_student(cfg: "EvalConfig") -> StudentClient:
    if not cfg.endpoint:
        raise StudentError("--endpoint is required for the 'openai' student backend")
    return OpenAIChatStudent(
        endpoint=cfg.endpoint, model=cfg.model, api_key=cfg.api_key,
        max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        timeout_s=cfg.student_timeout_s, retries=cfg.student_retries,
    )


def build_replay_student(cfg: "EvalConfig") -> StudentClient:
    if not cfg.mock_trajectory:
        raise StudentError("--mock-trajectory PATH is required for the 'mock' student backend")
    return ReplayStudent(json.loads(Path(cfg.mock_trajectory).read_text()))


STUDENT_BACKENDS: dict = {
    "openai": build_openai_student,
    "mock": build_replay_student,
}


def build_student(cfg: "EvalConfig") -> StudentClient:
    if cfg.student not in STUDENT_BACKENDS:
        raise StudentError(f"Unknown student backend {cfg.student!r}. Available: {sorted(STUDENT_BACKENDS)}")
    return STUDENT_BACKENDS[cfg.student](cfg)


# ---------------------------------------------------------------------------
# Action parsing. The grammar is exactly what train_qwen.format_action_for_training emits.
# ---------------------------------------------------------------------------

_REF = r"<ref>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)</ref>"

ACTION_PATTERNS: list = [
    (re.compile(rf"^click\(\s*{_REF}\s*(?:,\s*button\s*=\s*(\w+)\s*)?\)$"),
     lambda m: {"type": "click", "x": int(m.group(1)), "y": int(m.group(2)),
                "button": m.group(3) or "left"}),
    (re.compile(rf"^double_click\(\s*{_REF}\s*\)$"),
     lambda m: {"type": "double_click", "x": int(m.group(1)), "y": int(m.group(2))}),
    (re.compile(rf"^move\(\s*{_REF}\s*\)$"),
     lambda m: {"type": "move", "x": int(m.group(1)), "y": int(m.group(2))}),
    (re.compile(rf"^drag\(\s*{_REF}\s*->\s*{_REF}\s*\)$"),
     lambda m: {"type": "drag", "path": [{"x": int(m.group(1)), "y": int(m.group(2))},
                                         {"x": int(m.group(3)), "y": int(m.group(4))}]}),
    (re.compile(rf"^scroll\(\s*{_REF}\s*,\s*scrollX\s*=\s*(-?\d+)\s*,\s*scrollY\s*=\s*(-?\d+)\s*\)$"),
     lambda m: {"type": "scroll", "x": int(m.group(1)), "y": int(m.group(2)),
                "scroll_x": int(m.group(3)), "scroll_y": int(m.group(4))}),
    (re.compile(r'^type\("(.*)"\)$', re.DOTALL),
     lambda m: {"type": "type", "text": m.group(1)}),
    (re.compile(r"^keypress\((\[.*\])\)$", re.DOTALL),
     lambda m: {"type": "keypress", "keys": json.loads(m.group(1))}),
    (re.compile(r"^wait\(\)$"), lambda m: {"type": "wait"}),
    (re.compile(r"^done\(\)$"), lambda m: {"type": "done"}),
    (re.compile(r"^screenshot\(\)$"), lambda m: {"type": "screenshot"}),
]

_BULLET_RE = re.compile(r"^[-*]\s+")
_ACTION_PREFIX_RE = re.compile(r"^action\s*:\s*(.*)$", re.IGNORECASE)
# A bare call counts only when it is the WHOLE line; this hint spots a line that gestures at an
# action without being one, so leading prose is reported as malformed rather than silently ignored.
_ACTION_CALL_HINT_RE = re.compile(
    r"\b(?:click|double_click|move|type|keypress|scroll|drag|wait|done|screenshot)\s*\(")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Reasoning chat templates PREFILL the opening <think>, so the completion carries only the
# closing tag; without this the think text is lost and think_rate reads 0% on a model that
# reasoned on every step.
_THINK_PREFILLED_RE = re.compile(r"\A(.*?)</think>", re.DOTALL)


def _parse_action_expression(body: str) -> dict | None:
    """The action grammar itself, with no prefix handling."""
    for pattern, builder in ACTION_PATTERNS:
        matched = pattern.match(body)
        if not matched:
            continue
        try:
            return builder(matched)
        except (ValueError, json.JSONDecodeError):
            return None
    return None


def classify_action_line(line: str) -> tuple:
    """Classify one line as prefixed | bare | malformed | ignored, with its action when it has one."""
    stripped = _BULLET_RE.sub("", line.strip().strip("`").strip(), count=1)
    if not stripped:
        return None, "ignored"
    prefix_match = _ACTION_PREFIX_RE.match(stripped)
    body = prefix_match.group(1).strip() if prefix_match else stripped
    action = _parse_action_expression(body)
    if action is not None:
        return action, "prefixed" if prefix_match else "bare"
    if prefix_match or _ACTION_CALL_HINT_RE.search(stripped):
        return None, "malformed"
    return None, "ignored"


def parse_action_line(line: str) -> dict | None:
    """One action line back into a trajectory action dict, or None if it is not a valid action."""
    return classify_action_line(line)[0]


@dataclass
class ParsedStep:
    think: str
    actions: list
    malformed_lines: int
    raw: str
    bare_actions: int = 0

    @property
    def parse_failed(self) -> bool:
        return not self.actions


def parse_student_output(text: str, accept_bare_actions: bool = True) -> ParsedStep:
    """Split one assistant turn into its `<think>` text and ordered action dicts; a step fails only when NO line parsed."""
    text = text or ""
    think_re = _THINK_RE if "<think>" in text else _THINK_PREFILLED_RE
    think_match = think_re.search(text)
    think = (think_match.group(1).strip() if think_match else "")
    body = think_re.sub("", text, count=1)

    actions = []
    malformed = 0
    bare = 0
    for line in body.splitlines():
        action, kind = classify_action_line(line)
        if kind == "ignored":
            continue
        if kind == "malformed" or (kind == "bare" and not accept_bare_actions):
            malformed += 1
            continue
        bare += kind == "bare"
        actions.append(action)
    return ParsedStep(think=think, actions=actions, malformed_lines=malformed,
                      raw=text, bare_actions=bare)


# ---------------------------------------------------------------------------
# Coordinate-space registry: how the model's numbers relate to viewport pixels.
# PER-MODEL and never auto-detected, mirroring eval_screenspot's ModelSpec.coord_space —
# a stock Qwen answers in normalized_1000, a model fine-tuned here answers in viewport_abs,
# and scoring one in the other's space yields a confident wrong number instead of an error.
# ---------------------------------------------------------------------------

VIEWPORT_W = 1280
VIEWPORT_H = 720

COORD_SPACES: dict = {
    "viewport_abs": lambda x, y: (round(x), round(y)),
    "normalized_1000": lambda x, y: (round(x / 1000 * VIEWPORT_W), round(y / 1000 * VIEWPORT_H)),
}

# Action keys holding a SCREEN POSITION. scroll_x/scroll_y are relative wheel deltas, not
# positions, so they are passed through unscaled — scaling them would be a guess.
_POSITION_KEYS = ("x", "y")


def scale_action(action: dict, coord_space: str) -> dict:
    """The action as it will be EXECUTED: model coordinates mapped into viewport pixels."""
    to_pixels = COORD_SPACES[coord_space]
    if action.get("type") == "drag":
        return {**action, "path": [dict(zip(_POSITION_KEYS, to_pixels(p["x"], p["y"])))
                                   for p in action["path"]]}
    if "x" not in action or "y" not in action:
        return action
    x, y = to_pixels(action["x"], action["y"])
    return {**action, "x": x, "y": y}


def action_coordinates(action: dict) -> list:
    """Every (x, y) an action carries, for viewport-bounds diagnostics."""
    if action.get("type") == "drag":
        return [(p["x"], p["y"]) for p in action.get("path", [])]
    if "x" in action and "y" in action:
        return [(action["x"], action["y"])]
    return []


def outside_viewport(x, y) -> bool:
    return not (0 <= x < VIEWPORT_W and 0 <= y < VIEWPORT_H)


# ---------------------------------------------------------------------------
# Action -> driver payload. The coordinate-native /act kinds, keyed by action type.
# ---------------------------------------------------------------------------

ACTION_TO_DRIVER: dict = {
    "click": lambda a: {"kind": "click_xy", "x": a["x"], "y": a["y"], "button": a.get("button", "left")},
    "double_click": lambda a: {"kind": "click_xy", "x": a["x"], "y": a["y"], "dblclick": True},
    "move": lambda a: {"kind": "move_xy", "x": a["x"], "y": a["y"]},
    "type": lambda a: {"kind": "type_text", "text": a["text"]},
    "keypress": lambda a: {"kind": "press", "keys": a["keys"]},
    "scroll": lambda a: {"kind": "scroll_xy", "x": a["x"], "y": a["y"],
                         "dx": a.get("scroll_x", 0), "dy": a.get("scroll_y", 0)},
    "drag": lambda a: {"kind": "drag_path", "path": a["path"]},
    "wait": lambda a: {"kind": "wait", "seconds": 1},
    "done": lambda a: {"kind": "done"},
    "screenshot": lambda a: None,
}


def driver_payload(action: dict, thought: str) -> dict | None:
    builder = ACTION_TO_DRIVER.get(action.get("type"))
    if builder is None:
        return None
    payload = builder(action)
    if payload is None:
        return None
    payload["thought"] = thought or NO_THINK_PLACEHOLDER
    return payload


# ---------------------------------------------------------------------------
# Site blocking: scoring a CAPTCHA wall as a model failure measures the wrong thing.
# ---------------------------------------------------------------------------

BLOCK_SIGNATURES: list = [
    ("captcha", re.compile(r"\b(re)?captcha\b|\bhcaptcha\b|press *(&|and) *hold", re.IGNORECASE)),
    ("human_verification", re.compile(
        r"verify (that )?you('re| are)( a)? human|are you a robot|i'm not a robot|"
        r"confirm you are (a )?human|prove you('re| are) (not a robot|human)", re.IGNORECASE)),
    ("cloudflare", re.compile(
        r"checking your browser before accessing|cloudflare ray id|"
        r"enable javascript and cookies to continue|attention required! \| cloudflare", re.IGNORECASE)),
    ("waf_block", re.compile(
        r"access denied|403 forbidden|you (have been|are) blocked|request unsuccessful|"
        r"pardon our interruption|unusual traffic from your computer|"
        r"your request has been blocked|bot detection", re.IGNORECASE)),
]


def detect_block(url: str, snapshot: str) -> str | None:
    """Name of the first blocking signature visible in the current observation, else None."""
    haystack = f"{url or ''}\n{snapshot or ''}"
    for name, pattern in BLOCK_SIGNATURES:
        if pattern.search(haystack):
            return name
    return None


# ---------------------------------------------------------------------------
# Config + episode records
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    task_source: str = "online_mind2web"
    tasks_file: str = ""
    limit: int | None = None
    label: str = "baseline"
    run_id: str = ""
    student: str = "openai"
    endpoint: str = ""
    model: str = ""
    adapter: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 512
    student_timeout_s: int = 180
    student_retries: int = 2
    mock_trajectory: str = ""
    coord_space: str = "viewport_abs"
    accept_bare_actions: bool = True
    max_unproductive_steps: int = 5
    max_parse_failures: int = 3
    max_steps: int = 0                 # >0 pins a FLAT cap; 0 derives one per task (below)
    step_cap_multiple: float = 3.0     # cap = multiple x the task's own reference_length
    step_cap_floor: int = 15
    step_cap_ceiling: int = 50
    stratify: bool = False
    episode_timeout_s: int = 900
    screenshot_format: str = "png"
    judge: bool = True
    judge_mid_screenshots: int = 2
    parallel: int = 1
    resume: bool = True
    dataset_revision: str = "main"
    trajectories_dir: str = "trajectories"
    loop_guard_repeats: int = 0
    sandbox_image: str = ""


@dataclass
class EpisodeRecord:
    task_id: str
    instruction: str
    start_url: str
    website: str = ""
    domain: str = ""
    split_tag: str = ""
    level: str = ""
    reference_length: int = 0
    step_cap: int = 0
    outcome: str = "not_run"
    verified_success: bool = False
    judged_success: bool = False
    judge_deferred: bool = False
    efficiency: str = ""
    judge_reasoning: str = ""
    model_steps: int = 0
    driver_steps: int = 0
    actions_executed: int = 0
    parse_failure_steps: int = 0
    malformed_lines: int = 0
    actions_parsed: int = 0
    actions_missing_prefix: int = 0
    think_steps: int = 0
    driver_errors: int = 0
    out_of_bounds: int = 0
    emitted_coords: int = 0
    emitted_out_of_viewport: int = 0
    coord_space_warning: str = ""
    done_emitted: bool = False
    block_reason: str = ""
    block_step: int = 0
    error: str = ""
    duration_s: float = 0.0
    episode_dir: str = ""
    raw_last_response: str = ""


# Environment failures, not model failures: excluded from the denominator, reported separately.
UNSCORED_OUTCOMES = ("site_blocked", "unreachable", "harness_error")


def step_cap_for(task: EvalTask, cfg: EvalConfig, source_default: int) -> int:
    """Per-task step budget: `--max-steps` when pinned, else the multiple of the task's own reference_length."""
    if cfg.max_steps:
        return cfg.max_steps
    if not task.reference_length:
        return source_default
    derived = int(round(cfg.step_cap_multiple * task.reference_length))
    return max(cfg.step_cap_floor, min(cfg.step_cap_ceiling, derived))


# ---------------------------------------------------------------------------
# Sandbox + driver lifecycle
# ---------------------------------------------------------------------------


async def _run(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


class EvalBrowser:
    """One sandbox holding one browser driver, reused across episodes and recycled before self-expiry."""

    def __init__(self, cfg: EvalConfig, name: str):
        self.cfg = cfg
        self.name = name
        self.sandbox: DockerSandbox | None = None
        self.client: DriverClient | None = None

    async def ensure(self):
        if self.sandbox is not None and self.sandbox.remaining_seconds() > EPISODE_TIME_RESERVE:
            return
        await self.stop()
        kwargs = {"container_timeout": EVAL_CONTAINER_TIMEOUT}
        if self.cfg.sandbox_image:
            kwargs["image"] = self.cfg.sandbox_image
        self.sandbox = DockerSandbox(**kwargs)
        await _run(self.sandbox.start, f"eval/{self.name}")
        await _run(driver_session.install_driver, self.sandbox)
        await _run(driver_session.start_driver, self.sandbox,
                   {"DRIVER_LOOP_GUARD_REPEATS": str(self.cfg.loop_guard_repeats)})
        self.client = DriverClient(self.sandbox)

    async def stop(self):
        if self.sandbox is not None:
            await _run(self.sandbox.stop)
        self.sandbox = None
        self.client = None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def _note_emitted_coordinates(record: EpisodeRecord, actions: list, cfg: EvalConfig, task):
    """Count emitted coordinates outside 1280x720 and name the likely coordinate-space fix once per episode."""
    for action in actions:
        for x, y in action_coordinates(action):
            record.emitted_coords += 1
            if not outside_viewport(x, y):
                continue
            record.emitted_out_of_viewport += 1
            if cfg.coord_space != "viewport_abs" or record.coord_space_warning:
                continue
            record.coord_space_warning = (
                f"emitted ({x}, {y}) is outside the {VIEWPORT_W}x{VIEWPORT_H} viewport under "
                f"coord_space=viewport_abs; the model is likely answering in normalized 0-1000 "
                f"coordinates — re-run with --coord-space normalized_1000"
            )
            logger.warning(f"[{task.task_id}] COORDINATE SPACE: {record.coord_space_warning}")


async def run_episode(browser: EvalBrowser, student: StudentClient, task: EvalTask,
                      cfg: EvalConfig, episode_dir: Path, max_steps: int) -> EpisodeRecord:
    """Run one task start to finish, recording its trajectory into `episode_dir` for the judge."""
    record = EpisodeRecord(
        task_id=task.task_id, instruction=task.instruction, start_url=task.start_url,
        website=task.website, domain=task.domain, split_tag=task.split_tag, level=task.level,
        reference_length=task.reference_length, step_cap=max_steps, episode_dir=str(episode_dir),
    )
    started = time.time()
    action_history: list[str] = []
    consecutive_parse_failures = 0
    consecutive_unproductive = 0
    container_dir = f"{RESULTS_DIR}/{episode_dir_name(task)}"
    session_open = False

    await browser.ensure()
    client = browser.client

    try:
        try:
            session = await _run(client.session, task.task_id, task.start_url, task.instruction,
                                 {"expected_actions": task.reference_length,
                                  "eval_label": cfg.label, "eval_source": task.source})
            session_open = True
        except (DriverError, ConnectionError) as e:
            record.outcome = "unreachable"
            record.error = str(e)[:300]
            return record

        last_url = session.get("url", task.start_url)
        last_snapshot = session.get("snapshot", "")

        blocked = detect_block(last_url, last_snapshot)
        if blocked:
            record.outcome = "site_blocked"
            record.block_reason = blocked
            record.block_step = 0
            return record

        for step in range(1, max_steps + 1):
            if time.time() - started > cfg.episode_timeout_s:
                record.outcome = "timeout"
                break

            try:
                image_b64 = await _run(client.screenshot_b64, cfg.screenshot_format)
            except (DriverError, ConnectionError) as e:
                record.outcome = "harness_error"
                record.error = f"screenshot failed: {str(e)[:250]}"
                break

            mime = "image/png" if cfg.screenshot_format == "png" else "image/jpeg"
            messages = build_messages(task.instruction, action_history, image_b64, mime)
            try:
                raw = await _run(student.complete, messages)
            except StudentError as e:
                record.outcome = "harness_error"
                record.error = f"student: {str(e)[:250]}"
                break

            record.model_steps = step
            record.raw_last_response = (raw or "")[-1500:]
            parsed = parse_student_output(raw, cfg.accept_bare_actions)
            record.malformed_lines += parsed.malformed_lines
            record.actions_parsed += len(parsed.actions)
            record.actions_missing_prefix += parsed.bare_actions
            if parsed.think:
                record.think_steps += 1

            if parsed.parse_failed:
                record.parse_failure_steps += 1
                consecutive_parse_failures += 1
                consecutive_unproductive += 1
                logger.warning(f"[{task.task_id}] step {step}: no parseable action in "
                               f"{(raw or '')[:200]!r}")
                if consecutive_parse_failures >= cfg.max_parse_failures:
                    record.outcome = "parse_failure_abort"
                    break
                if consecutive_unproductive >= cfg.max_unproductive_steps:
                    record.outcome = "stuck"
                    break
                continue
            consecutive_parse_failures = 0

            _note_emitted_coordinates(record, parsed.actions, cfg, task)

            terminated = False
            executed_this_step = 0
            for action in parsed.actions:
                payload = driver_payload(scale_action(action, cfg.coord_space), parsed.think)
                if payload is None:
                    continue
                if action["type"] == "done":
                    record.done_emitted = True
                    try:
                        await _run(client.act, payload)
                    except (DriverError, ConnectionError) as e:
                        record.driver_errors += 1
                        record.error = record.error or f"done failed: {str(e)[:200]}"
                    terminated = True
                    break
                try:
                    result = await _run(client.act, payload)
                except (DriverError, ConnectionError) as e:
                    message = str(e)
                    record.driver_errors += 1
                    if "outside the" in message:
                        record.out_of_bounds += 1
                    logger.info(f"[{task.task_id}] step {step}: driver refused "
                                f"{action['type']}: {message[:160]}")
                    continue
                record.actions_executed += 1
                executed_this_step += 1
                # History stays in the MODEL's coordinate space so its own frame is self-consistent.
                action_history.append(format_action_summary(action))
                last_url = result.get("url", last_url)
                last_snapshot = result.get("snapshot", last_snapshot)

            if terminated:
                record.outcome = "completed"
                break

            consecutive_unproductive = 0 if executed_this_step else consecutive_unproductive + 1
            if consecutive_unproductive >= cfg.max_unproductive_steps:
                record.outcome = "stuck"
                logger.warning(f"[{task.task_id}] aborting: {consecutive_unproductive} consecutive "
                               f"steps executed no action ({record.driver_errors} driver refusals, "
                               f"{record.out_of_bounds} out of viewport)")
                break

            blocked = detect_block(last_url, last_snapshot)
            if blocked:
                record.outcome = "site_blocked"
                record.block_reason = blocked
                record.block_step = step
                break
        else:
            record.outcome = "step_cap"
    finally:
        if session_open:
            summary = f"{record.actions_executed} actions executed; outcome {record.outcome}"
            try:
                finish = await _run(client.finish, record.outcome, summary)
                record.driver_steps = finish.get("steps", 0)
            except Exception as e:
                logger.warning(f"[{task.task_id}] driver finish failed: {e}")
            episode_dir.mkdir(parents=True, exist_ok=True)
            try:
                await _run(browser.sandbox.copy_out, container_dir, str(episode_dir))
                await _run(browser.sandbox.exec_command, f"rm -rf {container_dir}")
            except Exception as e:
                logger.warning(f"[{task.task_id}] copy_out failed: {e}")
        record.duration_s = round(time.time() - started, 1)

    return record


def _write_episode_metadata(episode_dir: Path, record: EpisodeRecord, cfg: EvalConfig):
    """Write metadata whose `excluded_reason` keeps an eval episode untrainable even if it is copied into the corpus."""
    (episode_dir / "metadata.json").write_text(json.dumps({
        "task_id": record.task_id,
        "instruction": record.instruction,
        "start_url": record.start_url,
        "website": record.website,
        "domain": record.domain,
        "total_steps": record.driver_steps,
        "success": record.verified_success,
        "capture_source": "student_eval",
        "excluded_reason": "eval_episode",
        "eval_label": cfg.label,
        "eval_model": cfg.model,
        "viewport": {"width": 1280, "height": 720},
    }, indent=2))


async def judge_episode(record: EpisodeRecord, cfg: EvalConfig):
    """Score one recorded episode with the repo's existing verifier (the WebJudge analogue)."""
    episode_dir = Path(record.episode_dir)
    if not (episode_dir / "trajectory.json").exists():
        record.error = record.error or "no trajectory.json recorded — nothing to judge"
        return
    verdict = await verify_trajectory(
        episode_dir, full_name=f"eval:{cfg.label}:{record.task_id}",
        mid_screenshots=cfg.judge_mid_screenshots,
    )
    record.judged_success = bool(verdict.get("success"))
    record.verified_success = bool(verdict.get("success"))
    record.efficiency = verdict.get("efficiency") or ""
    record.judge_reasoning = (verdict.get("reasoning") or "")[:400]


# ---------------------------------------------------------------------------
# Metrics + reporting
# ---------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _is_scored(record: EpisodeRecord) -> bool:
    """An episode counts toward the success rate only if it both ran and was judged."""
    return record.outcome not in UNSCORED_OUTCOMES and not record.judge_deferred


def _group_metrics(records: list[EpisodeRecord], key) -> dict:
    """Success / blocking / step-efficiency broken out by any grouping key (level, domain)."""
    groups: dict = {}
    for r in records:
        bucket = groups.setdefault(key(r), {"n": 0, "scored": 0, "success": 0, "blocked": 0,
                                            "_success_ratios": [], "_success_steps": []})
        bucket["n"] += 1
        if r.outcome == "site_blocked":
            bucket["blocked"] += 1
        if _is_scored(r):
            bucket["scored"] += 1
            bucket["success"] += int(r.verified_success)
            if r.verified_success:
                bucket["_success_steps"].append(r.model_steps)
                if r.reference_length:
                    bucket["_success_ratios"].append(r.model_steps / r.reference_length)
    for bucket in groups.values():
        bucket["success_rate"] = _rate(bucket["success"], bucket["scored"])
        bucket["blocked_rate"] = _rate(bucket["blocked"], bucket["n"])
        bucket["mean_steps_on_success"] = _mean(bucket.pop("_success_steps"))
        bucket["mean_steps_over_reference"] = _mean(bucket.pop("_success_ratios"))
    return groups


def _mean(values: list) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def compute_metrics(records: list[EpisodeRecord]) -> dict:
    total = len(records)
    scored = [r for r in records if _is_scored(r)]
    successes = [r for r in scored if r.verified_success]
    blocked = [r for r in records if r.outcome == "site_blocked"]
    model_steps = sum(r.model_steps for r in records)
    parse_failures = sum(r.parse_failure_steps for r in records)
    actions_parsed = sum(r.actions_parsed for r in records)
    actions_missing_prefix = sum(r.actions_missing_prefix for r in records)

    by_domain = _group_metrics(records, lambda r: r.domain or r.website or "unknown")
    by_level = _group_metrics(records, lambda r: r.level or "ungraded")

    outcomes: dict = {}
    for r in records:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1

    success_steps = [r.model_steps for r in successes]
    success_ratios = [r.model_steps / r.reference_length for r in successes if r.reference_length]
    step_cap_hits = sum(1 for r in records if r.outcome == "step_cap")
    return {
        "episodes": total,
        "scored_episodes": len(scored),
        "judge_deferred_episodes": sum(1 for r in records if r.judge_deferred),
        "successes": len(successes),
        "verified_success_rate": _rate(len(successes), len(scored)),
        "verified_success_rate_all_episodes": _rate(len(successes), total),
        "efficient_success_rate": _rate(
            sum(1 for r in successes if r.efficiency != "meandering"), len(scored)),
        "site_blocked_rate": _rate(len(blocked), total),
        "parse_failure_rate": _rate(parse_failures, model_steps),
        "malformed_action_lines": sum(r.malformed_lines for r in records),
        "actions_parsed": actions_parsed,
        "actions_missing_prefix": actions_missing_prefix,
        "prefix_adherence_rate": _rate(actions_parsed - actions_missing_prefix, actions_parsed),
        "done_rate": _rate(sum(1 for r in records if r.done_emitted), total),
        "think_rate": _rate(sum(r.think_steps for r in records), model_steps),
        "driver_error_rate": _rate(sum(r.driver_errors for r in records),
                                   sum(r.actions_executed + r.driver_errors for r in records)),
        "out_of_bounds_actions": sum(r.out_of_bounds for r in records),
        "emitted_coords": sum(r.emitted_coords for r in records),
        "emitted_out_of_viewport": sum(r.emitted_out_of_viewport for r in records),
        "emitted_out_of_viewport_rate": _rate(sum(r.emitted_out_of_viewport for r in records),
                                              sum(r.emitted_coords for r in records)),
        "stuck_episodes": sum(1 for r in records if r.outcome == "stuck"),
        "mean_steps_on_success": _mean(success_steps),
        "mean_steps_over_reference_on_success": _mean(success_ratios),
        "mean_steps_all": round(model_steps / total, 2) if total else 0.0,
        "step_cap_hits": step_cap_hits,
        "total_model_steps": model_steps,
        "outcomes": outcomes,
        "by_level": by_level,
        "by_domain": by_domain,
    }


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()[:12]
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               check=True).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def step_cap_policy(cfg: EvalConfig, source) -> str:
    if cfg.max_steps:
        return f"flat {cfg.max_steps}"
    return (f"{cfg.step_cap_multiple}x reference_length, clamped to "
            f"[{cfg.step_cap_floor}, {cfg.step_cap_ceiling}] "
            f"(ungraded tasks: {source.default_step_cap})")


def harness_fingerprint(cfg: EvalConfig, source, provenance: dict) -> dict:
    """Everything that must match for two runs to be comparable. `compare` diffs exactly this."""
    return {
        "task_source": source.name,
        "task_source_description": source.description,
        "dataset": provenance.get("dataset", source.access_url),
        "dataset_revision": provenance.get("revision", ""),
        "dataset_split": provenance.get("split", ""),
        "coord_space": cfg.coord_space,
        # Parity key, deliberately NOT expected-to-differ: prefix tolerance is a property of the
        # harness, so both sides of a delta must have been parsed the same way.
        "accepts_bare_actions": cfg.accept_bare_actions,
        "step_cap_policy": step_cap_policy(cfg, source),
        "max_unproductive_steps": cfg.max_unproductive_steps,
        "max_parse_failures": cfg.max_parse_failures,
        "subset_strategy": (f"first {cfg.limit} by dataset order" if cfg.limit and not cfg.stratify
                            else f"level-stratified {cfg.limit}" if cfg.limit else "full set"),
        "level_distribution": provenance.get("level_distribution", {}),
        "screenshot_format": cfg.screenshot_format,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "student_backend": cfg.student,
        "judge_enabled": cfg.judge,
        "judge_model": MODEL_FOR_REASONING if cfg.judge else "",
        "judge_mid_screenshots": cfg.judge_mid_screenshots,
        "driver_loop_guard_repeats": cfg.loop_guard_repeats,
        "history_window": HISTORY_WINDOW,
        "git_commit": git_commit(),
    }


# A calibrated viewport_abs model essentially never lands outside the viewport, so crossing this
# share of emitted coordinates means the run measured the wrong coordinate space, not the model.
COORD_SPACE_SUSPECT_RATE = float(os.environ.get("EVAL_COORD_SPACE_SUSPECT_RATE", "0.25"))


def coord_space_diagnosis(fingerprint: dict, metrics: dict) -> str:
    """The run-level version of the per-episode warning: is this number even measuring the model?"""
    if fingerprint.get("coord_space") != "viewport_abs":
        return ""
    rate = metrics.get("emitted_out_of_viewport_rate", 0)
    if rate < COORD_SPACE_SUSPECT_RATE or not metrics.get("emitted_coords"):
        return ""
    return (
        f"COORDINATE SPACE SUSPECT: {rate * 100:.0f}% of emitted coordinates "
        f"({metrics['emitted_out_of_viewport']}/{metrics['emitted_coords']}) fall outside "
        f"{VIEWPORT_W}x{VIEWPORT_H} under coord_space=viewport_abs. This model is probably "
        f"answering in normalized 0-1000 coordinates; re-run with --coord-space normalized_1000 "
        f"before reading any success rate below."
    )


def render_report(results: dict) -> str:
    cfg = results["config"]
    fingerprint = results["harness"]
    metrics = results["metrics"]
    source_name = fingerprint["task_source"]

    lines = [
        f"# Student eval — `{results['label']}`",
        "",
        f"**verified task success: {metrics['verified_success_rate'] * 100:.1f}% across "
        f"{metrics['episodes']} {source_name} tasks, "
        f"parse-failure rate {metrics['parse_failure_rate'] * 100:.1f}%, "
        f"site-blocked {metrics['site_blocked_rate'] * 100:.1f}%**",
        "",
        f"Success is over the SCORED episodes ({metrics['successes']} verified of "
        f"{metrics['scored_episodes']}); site-blocked / unreachable / harness-error episodes are "
        f"excluded from that denominator and reported separately, because folding bot-blocking "
        f"into the success rate measures the network, not the model. Over all "
        f"{metrics['episodes']} episodes the rate is "
        f"{metrics['verified_success_rate_all_episodes'] * 100:.1f}%.",
        "",
        "## Run",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| run id | `{results['run_id']}` |",
        f"| label | `{results['label']}` |",
        f"| started (UTC) | {results['started_at']} |",
        f"| finished (UTC) | {results['finished_at']} |",
        f"| endpoint | `{cfg['endpoint'] or '(none — ' + cfg['student'] + ' backend)'}` |",
        f"| model | `{cfg['model'] or '(n/a)'}` |",
        f"| adapter / checkpoint | `{cfg['adapter'] or '(none declared)'}` |",
        f"| task source | `{source_name}` — {fingerprint['task_source_description']} |",
        f"| dataset | `{fingerprint['dataset']}` |",
        f"| dataset revision | `{fingerprint['dataset_revision'] or '(n/a)'}` |",
        f"| coordinate space | `{fingerprint['coord_space']}` |",
        f"| step cap | {fingerprint['step_cap_policy']} |",
        f"| abort rules | {fingerprint['max_unproductive_steps']} unproductive steps / "
        f"{fingerprint['max_parse_failures']} parse failures |",
        f"| task subset | {fingerprint['subset_strategy']} |",
        f"| temperature / max tokens | {fingerprint['temperature']} / {fingerprint['max_tokens']} |",
        f"| observation format | {fingerprint['screenshot_format']} @ 1280x720 |",
        f"| judge | {fingerprint['judge_model'] or 'disabled'} "
        f"(final + {fingerprint['judge_mid_screenshots']} mid screenshots) |",
        f"| history window | {fingerprint['history_window']} |",
        f"| driver loop guard | {fingerprint['driver_loop_guard_repeats'] or 'disabled'} |",
        f"| browser | local DockerSandbox on the invoking host |",
        f"| repo commit | `{fingerprint['git_commit']}` |",
        "",
    ]
    if results.get("aborted_reason"):
        lines += [f"> **Run aborted early: {results['aborted_reason']}** — the numbers below "
                  f"cover only the episodes that ran.", ""]
    warning = coord_space_diagnosis(fingerprint, metrics)
    if warning:
        lines += [f"> **{warning}**", ""]
    if results.get("task_provenance"):
        lines += ["<details><summary>task provenance</summary>", "",
                  "```json", json.dumps(results["task_provenance"], indent=2), "```", "",
                  "</details>", ""]

    lines += [
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| episodes | {metrics['episodes']} |",
        f"| scored episodes | {metrics['scored_episodes']} |",
        f"| judge deferred (usage limit) | {metrics.get('judge_deferred_episodes', 0)} |",
        f"| verified success (scored) | {metrics['verified_success_rate'] * 100:.1f}% |",
        f"| verified success (all episodes) | {metrics['verified_success_rate_all_episodes'] * 100:.1f}% |",
        f"| verified AND non-meandering | {metrics['efficient_success_rate'] * 100:.1f}% |",
        f"| site-blocked | {metrics['site_blocked_rate'] * 100:.1f}% |",
        f"| parse-failure rate (steps) | {metrics['parse_failure_rate'] * 100:.1f}% |",
        f"| malformed action lines | {metrics['malformed_action_lines']} |",
        f"| `action:` prefix adherence | {metrics.get('prefix_adherence_rate', 0) * 100:.1f}% "
        f"({metrics.get('actions_missing_prefix', 0)}/{metrics.get('actions_parsed', 0)} "
        f"actions omitted the prefix) |",
        f"| done() emission rate | {metrics['done_rate'] * 100:.1f}% |",
        f"| <think> emission rate (steps) | {metrics['think_rate'] * 100:.1f}% |",
        f"| driver-refused action rate | {metrics['driver_error_rate'] * 100:.1f}% |",
        f"| emitted coords outside the viewport | {metrics.get('emitted_out_of_viewport', 0)}"
        f"/{metrics.get('emitted_coords', 0)} "
        f"({metrics.get('emitted_out_of_viewport_rate', 0) * 100:.1f}%) |",
        f"| out-of-viewport actions refused | {metrics['out_of_bounds_actions']} |",
        f"| episodes abandoned as stuck | {metrics.get('stuck_episodes', 0)} |",
        f"| mean steps on success | {metrics['mean_steps_on_success']} |",
        f"| mean steps / reference_length on success | {metrics['mean_steps_over_reference_on_success']}x |",
        f"| mean steps (all) | {metrics['mean_steps_all']} |",
        f"| episodes that hit the step cap | {metrics['step_cap_hits']} |",
        "",
        "`verified` is the judge's verdict on the ORIGINAL goal; the efficiency grade is reported "
        "beside it rather than folded into it. `steps / reference_length` compares against the "
        "benchmark's own human step count, which is a more defensible efficiency denominator than "
        "our raw step counts.",
        "",
        "Prefix adherence is reported on its own rather than folded into the success rate: the "
        "parser accepts the action grammar with or without the leading `action:` (it is "
        "unambiguous either way), so a model that never learned the prefix is not scored as unable "
        "to browse. How well each model formats its output stays visible as its own finding, and "
        "the tolerance is a harness parity key so both sides of a delta are parsed identically.",
        "",
        "## Per level",
        "",
        "| level | n | scored | verified | success rate | steps/ref | blocked |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for level in sorted(metrics["by_level"], key=lambda k: ("easy", "medium", "hard").index(k)
                        if k in ("easy", "medium", "hard") else 99):
        bucket = metrics["by_level"][level]
        lines.append(f"| {level} | {bucket['n']} | {bucket['scored']} | {bucket['success']} | "
                     f"{bucket['success_rate'] * 100:.1f}% | "
                     f"{bucket['mean_steps_over_reference']}x | {bucket['blocked']} |")

    lines += [
        "",
        "## Outcomes",
        "",
        "| outcome | episodes |",
        "| --- | --- |",
    ]
    for outcome, count in sorted(metrics["outcomes"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {outcome} | {count} |")

    lines += ["", "## Per-domain", "", "| domain | n | scored | verified | success rate | blocked |",
              "| --- | --- | --- | --- | --- | --- |"]
    for domain, bucket in sorted(metrics["by_domain"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {domain} | {bucket['n']} | {bucket['scored']} | {bucket['success']} | "
                     f"{bucket['success_rate'] * 100:.1f}% | {bucket['blocked']} |")

    lines += ["", "## Episodes", "",
              "| task | level | outcome | verified | efficiency | steps | cap | ref | parse fails | "
              "driver errors | note |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results["episodes"]:
        note = r["block_reason"] or r["error"] or (r["judge_reasoning"] or "")[:120]
        note = note.replace("|", "/").replace("\n", " ")
        lines.append(
            f"| `{r['task_id'][:40]}` | {r.get('level') or '-'} | {r['outcome']} | "
            f"{'yes' if r['verified_success'] else 'no'} | {r['efficiency'] or '-'} | "
            f"{r['model_steps']} | {r.get('step_cap', 0)} | {r['reference_length']} | "
            f"{r['parse_failure_steps']} | {r['driver_errors']} | {note} |")

    lines += ["", "---", "",
              "Episodes are recorded under this run directory and are excluded from training by "
              "construction (outside `trajectories/`, and each carries "
              "`metadata.excluded_reason=eval_episode`).", ""]
    return "\n".join(lines)


# A before/after pair is EXPECTED to differ here: a stock model answers in normalized_1000 and a
# fine-tuned one in viewport_abs, so each side must be measured in its own convention.
EXPECTED_TO_DIFFER_KEYS = ("coord_space",)
# Provenance, not harness settings — these can never make a delta unattributable.
PARITY_EXEMPT_KEYS = ("git_commit", "task_source_description", "level_distribution") + EXPECTED_TO_DIFFER_KEYS


def render_comparison(runs: list[dict]) -> str:
    """Before/after table over two or more runs, with the parity check and per-level and per-domain diffs."""
    lines = ["# Student eval — comparison", "",
             "| label | verified success | all-episode success | parse-failure | prefix adherence | "
             "done() rate | mean steps (success) | steps/ref | site-blocked | n |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in runs:
        m = run["metrics"]
        lines.append(
            f"| `{run['label']}` | {m['verified_success_rate'] * 100:.1f}% | "
            f"{m['verified_success_rate_all_episodes'] * 100:.1f}% | "
            f"{m['parse_failure_rate'] * 100:.1f}% | "
            f"{m.get('prefix_adherence_rate', 0) * 100:.1f}% | {m['done_rate'] * 100:.1f}% | "
            f"{m['mean_steps_on_success']} | "
            f"{m.get('mean_steps_over_reference_on_success', 0)}x | "
            f"{m['site_blocked_rate'] * 100:.1f}% | {m['episodes']} |")

    base = runs[0]
    levels = [level for level in ("easy", "medium", "hard", "ungraded")
              if any(level in run["metrics"].get("by_level", {}) for run in runs)]
    if levels:
        lines += ["", "## By level", "",
                  "| label | " + " | ".join(levels) + " |",
                  "| --- | " + " | ".join("---" for _ in levels) + " |"]
        for run in runs:
            cells = []
            for level in levels:
                bucket = run["metrics"].get("by_level", {}).get(level)
                cells.append(f"{bucket['success_rate'] * 100:.1f}% ({bucket['success']}/{bucket['scored']})"
                             if bucket else "-")
            lines.append(f"| `{run['label']}` | " + " | ".join(cells) + " |")

    if len(runs) > 1:
        lines += ["", f"## Delta vs `{base['label']}`", "",
                  "| label | verified success | parse-failure | prefix adherence | done() rate | "
                  + " | ".join(f"{level} success" for level in levels) + " |",
                  "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in levels) + " |"]
        for run in runs[1:]:
            a, b = base["metrics"], run["metrics"]
            level_cells = []
            for level in levels:
                before = (a.get("by_level", {}).get(level) or {}).get("success_rate")
                after = (b.get("by_level", {}).get(level) or {}).get("success_rate")
                level_cells.append(f"{(after - before) * 100:+.1f} pp"
                                   if before is not None and after is not None else "-")
            lines.append(
                f"| `{run['label']}` | "
                f"{(b['verified_success_rate'] - a['verified_success_rate']) * 100:+.1f} pp | "
                f"{(b['parse_failure_rate'] - a['parse_failure_rate']) * 100:+.1f} pp | "
                f"{(b.get('prefix_adherence_rate', 0) - a.get('prefix_adherence_rate', 0)) * 100:+.1f} pp | "
                f"{(b['done_rate'] - a['done_rate']) * 100:+.1f} pp | "
                + " | ".join(level_cells) + " |")

    annotated = []
    for key in EXPECTED_TO_DIFFER_KEYS:
        values = {run["label"]: run["harness"].get(key) for run in runs}
        if len(set(map(repr, values.values()))) > 1:
            annotated.append(f"- `{key}`: " + ", ".join(f"`{k}`={v!r}" for k, v in values.items()))

    differences = []
    for key, value in base["harness"].items():
        if key in PARITY_EXEMPT_KEYS:
            continue
        for run in runs[1:]:
            if run["harness"].get(key) != value:
                differences.append(f"- `{key}`: `{base['label']}`={value!r} vs "
                                   f"`{run['label']}`={run['harness'].get(key)!r}")
    lines += ["", "## Harness parity", ""]
    if differences:
        lines += ["**These runs are NOT the same harness — the delta above is not attributable to "
                  "the model alone:**", ""] + differences
    else:
        lines.append("All compared runs share the same task subset, step-cap and abort policy, "
                     "prompt settings and judge configuration.")
    if annotated:
        lines += ["", "Expected model-side differences (not parity violations — each model is "
                  "measured in its own convention):", ""] + annotated
    commits = {run["label"]: run["harness"].get("git_commit") for run in runs}
    lines += ["", f"Repo commits: {json.dumps(commits)}", ""]

    domains = sorted({d for run in runs for d in run["metrics"]["by_domain"]})
    lines += ["## Per-domain success rate", "",
              "| domain | " + " | ".join(f"`{run['label']}`" for run in runs) + " |",
              "| --- | " + " | ".join("---" for _ in runs) + " |"]
    for domain in domains:
        cells = []
        for run in runs:
            bucket = run["metrics"]["by_domain"].get(domain)
            cells.append(f"{bucket['success_rate'] * 100:.0f}% ({bucket['success']}/{bucket['scored']})"
                         if bucket else "-")
        lines.append(f"| {domain} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------


def _make_run_id(cfg: EvalConfig) -> str:
    if cfg.run_id:
        return cfg.run_id
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{cfg.label}_{cfg.task_source}_{stamp}"


def _load_existing_episode(episode_dir: Path) -> EpisodeRecord | None:
    record_file = episode_dir / "episode.json"
    if not record_file.exists():
        return None
    try:
        return EpisodeRecord(**json.loads(record_file.read_text()))
    except (json.JSONDecodeError, OSError, TypeError):
        return None


async def _process_task(task: EvalTask, cfg: EvalConfig, browser: EvalBrowser, run_dir: Path,
                        source_default_cap: int, semaphore: asyncio.Semaphore) -> EpisodeRecord:
    episode_dir = run_dir / "episodes" / episode_dir_name(task)
    max_steps = step_cap_for(task, cfg, source_default_cap)
    async with semaphore:
        if cfg.resume:
            existing = _load_existing_episode(episode_dir)
            if existing is not None and not (existing.judge_deferred and cfg.judge):
                logger.info(f"[{task.task_id}] already evaluated ({existing.outcome}) — skipping")
                return existing
            if existing is not None:
                # The browser work is already on disk; only the verdict is owed.
                logger.info(f"[{task.task_id}] re-judging a deferred episode")
                existing.judge_deferred = False
                await judge_episode(existing, cfg)
                _persist_episode(episode_dir, existing, cfg)
                return existing

        # A fresh client per episode: the replay student walks a recorded trajectory statefully.
        record = await run_episode(browser, build_student(cfg), task, cfg, episode_dir, max_steps)
        try:
            if cfg.judge and record.outcome not in UNSCORED_OUTCOMES:
                await judge_episode(record, cfg)
        except UsageLimitReached:
            # Persist before re-raising: the episode ran, only the verdict is owed, and a resume
            # must re-judge from disk instead of re-driving the browser.
            record.error = record.error or "judge skipped: Claude usage limit reached"
            record.judge_deferred = True
            _persist_episode(episode_dir, record, cfg)
            raise
        except Exception as e:
            logger.warning(f"[{task.task_id}] judging failed: {e}")
            record.error = record.error or f"judge failed: {str(e)[:200]}"
        _persist_episode(episode_dir, record, cfg)
        logger.info(f"[{task.task_id}] {record.outcome} | verified={record.verified_success} | "
                    f"{record.model_steps} steps | {record.duration_s}s")
        return record


def _persist_episode(episode_dir: Path, record: EpisodeRecord, cfg: EvalConfig):
    if not episode_dir.exists():
        return
    _write_episode_metadata(episode_dir, record, cfg)
    (episode_dir / "episode.json").write_text(json.dumps(asdict(record), indent=2))


async def run_eval(cfg: EvalConfig) -> dict:
    load_cfg = TaskLoadConfig(
        limit=cfg.limit, revision=cfg.dataset_revision,
        cache_dir=EVAL_RUNS_DIR / ".task_cache",
        trajectories_dir=Path(cfg.trajectories_dir),
        stratify=cfg.stratify,
    )
    if cfg.tasks_file:
        source = TASK_SOURCES[cfg.task_source]
        tasks = load_tasks_from_file(Path(cfg.tasks_file), cfg.limit)
        provenance = {"tasks_file": cfg.tasks_file, "rows_available": len(tasks)}
    else:
        source, tasks, provenance = load_tasks(cfg.task_source, load_cfg)

    run_id = _make_run_id(cfg)
    run_dir = EVAL_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(run_dir / "run.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info(f"run {run_id}: {len(tasks)} tasks from {source.name} "
                f"(levels: {provenance.get('level_distribution', {})}), "
                f"step cap {step_cap_policy(cfg, source)}, coord_space={cfg.coord_space}, "
                f"student={cfg.student} model={cfg.model or '(n/a)'}")

    workers = safe_parallel_sandboxes(max(1, cfg.parallel))
    if workers != cfg.parallel:
        logger.info(f"sandbox parallelism {cfg.parallel} -> {workers}")
    browsers = [EvalBrowser(cfg, f"{run_id}_w{i}") for i in range(workers)]
    semaphores = [asyncio.Semaphore(1) for _ in browsers]

    started_at = datetime.now(timezone.utc).isoformat()
    records: list[EpisodeRecord] = []
    aborted_reason = ""
    try:
        coros = [
            _process_task(task, cfg, browsers[i % workers], run_dir, source.default_step_cap,
                          semaphores[i % workers])
            for i, task in enumerate(tasks)
        ]
        for result in await asyncio.gather(*coros, return_exceptions=True):
            if isinstance(result, UsageLimitReached):
                aborted_reason = f"Claude usage limit reached during judging ({result})"
                logger.error(aborted_reason)
            elif isinstance(result, BaseException):
                logger.exception(f"episode raised: {result}")
            else:
                records.append(result)
    finally:
        for browser in browsers:
            await browser.stop()

    results = {
        "run_id": run_id,
        "label": cfg.label,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg) | {"api_key": "<redacted>" if cfg.api_key else ""},
        "harness": harness_fingerprint(cfg, source, provenance),
        "task_provenance": provenance,
        "aborted_reason": aborted_reason,
        "episodes": [asdict(r) for r in records],
        "metrics": compute_metrics(records),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    (run_dir / "report.md").write_text(render_report(results))
    logger.info(f"wrote {run_dir / 'report.md'} and {run_dir / 'results.json'}")
    return results


def load_run(run_dir: str | Path) -> dict:
    path = Path(run_dir)
    results_file = path if path.suffix == ".json" else path / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"{results_file} not found — is {run_dir} an eval run directory?")
    return json.loads(results_file.read_text())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_run_arguments(parser):
    parser.add_argument("--task-source", default="online_mind2web",
                        choices=sorted(TASK_SOURCES), help="which task set to evaluate on")
    parser.add_argument("--tasks-file", default="",
                        help="JSON/JSONL of EvalTask rows, bypassing the registry loader")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N tasks")
    parser.add_argument("--label", default="baseline",
                        help="run label, e.g. 'before' / 'after' — names the run dir and the report")
    parser.add_argument("--run-id", default="", help="override the generated run directory name")
    parser.add_argument("--student", default="openai", choices=sorted(STUDENT_BACKENDS))
    parser.add_argument("--endpoint", default=os.environ.get("EVAL_ENDPOINT", ""),
                        help="OpenAI-compatible base URL (e.g. http://host:8000/v1)")
    parser.add_argument("--model", default=os.environ.get("EVAL_MODEL", ""), help="served model id")
    parser.add_argument("--adapter", default=os.environ.get("EVAL_ADAPTER", ""),
                        help="LoRA/checkpoint path served behind the endpoint (recorded in the report)")
    parser.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--student-timeout", type=int, default=180)
    parser.add_argument("--student-retries", type=int, default=2,
                        help="retries for transport errors only; HTTP errors are surfaced")
    parser.add_argument("--mock-trajectory", default="",
                        help="trajectory.json replayed by the 'mock' student backend")
    parser.add_argument("--coord-space", default=os.environ.get("EVAL_COORD_SPACE", "viewport_abs"),
                        choices=sorted(COORD_SPACES),
                        help="the convention THIS model answers in: a model fine-tuned here emits "
                             "viewport_abs (1280x720 pixels), a stock Qwen emits normalized_1000. "
                             "Set it explicitly per model — it is never auto-detected")
    parser.add_argument("--require-action-prefix", action="store_true",
                        help="reject action lines that omit the leading `action:`. Off by default: "
                             "the grammar is unambiguous without it, and rejecting bare calls "
                             "scores format adherence rather than browsing ability")
    parser.add_argument("--max-unproductive-steps", type=int, default=MAX_UNPRODUCTIVE_STEPS,
                        help="abort the episode as 'stuck' after this many steps in a row that "
                             "execute no action at all")
    parser.add_argument("--max-parse-failures", type=int, default=MAX_CONSECUTIVE_PARSE_FAILURES,
                        help="abort as 'parse_failure_abort' after this many unparseable steps in a row")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="FLAT step cap per episode; 0 derives one from each task's reference_length")
    parser.add_argument("--step-cap-multiple", type=float, default=3.0,
                        help="derived cap = this x the task's reference_length")
    parser.add_argument("--step-cap-floor", type=int, default=15)
    parser.add_argument("--step-cap-ceiling", type=int, default=50)
    parser.add_argument("--stratify", action="store_true",
                        help="with --limit, keep the task set's easy/medium/hard mix instead of "
                             "taking the first N rows")
    parser.add_argument("--episode-timeout", type=int, default=900)
    parser.add_argument("--screenshot-format", default="png", choices=("png", "jpeg"))
    parser.add_argument("--no-judge", action="store_true",
                        help="skip verification (harness-only run; no Claude calls, no success rate)")
    parser.add_argument("--judge-mid-screenshots", type=int, default=2,
                        help="mid-run frames per verdict on top of the final one; the main cost dial")
    parser.add_argument("--parallel", type=int, default=1,
                        help="concurrent sandboxes (clamped to what the Docker VM can hold)")
    parser.add_argument("--no-resume", action="store_true",
                        help="re-run episodes that already have an episode.json")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--trajectories-dir", default="trajectories",
                        help="corpus checked by the held_out source to prove a task is unseen")
    parser.add_argument("--loop-guard-repeats", type=int, default=0,
                        help="driver loop guard for eval; 0 lets a stuck model actually repeat itself")
    parser.add_argument("--sandbox-image", default="")


def _config_from_args(args) -> EvalConfig:
    return EvalConfig(
        task_source=args.task_source, tasks_file=args.tasks_file, limit=args.limit,
        label=args.label, run_id=args.run_id, student=args.student, endpoint=args.endpoint,
        model=args.model, adapter=args.adapter, api_key=args.api_key,
        temperature=args.temperature, max_tokens=args.max_tokens,
        student_timeout_s=args.student_timeout, student_retries=args.student_retries,
        mock_trajectory=args.mock_trajectory, coord_space=args.coord_space,
        accept_bare_actions=not args.require_action_prefix,
        max_unproductive_steps=args.max_unproductive_steps,
        max_parse_failures=args.max_parse_failures, max_steps=args.max_steps,
        step_cap_multiple=args.step_cap_multiple, step_cap_floor=args.step_cap_floor,
        step_cap_ceiling=args.step_cap_ceiling, stratify=args.stratify,
        episode_timeout_s=args.episode_timeout, screenshot_format=args.screenshot_format,
        judge=not args.no_judge, judge_mid_screenshots=args.judge_mid_screenshots,
        parallel=args.parallel, resume=not args.no_resume,
        dataset_revision=args.dataset_revision, trajectories_dir=args.trajectories_dir,
        loop_guard_repeats=args.loop_guard_repeats, sandbox_image=args.sandbox_image,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a served computer-use student model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a task set against a served model")
    _add_run_arguments(run_parser)

    compare_parser = subparsers.add_parser("compare", help="render a before/after table")
    compare_parser.add_argument("runs", nargs="+", help="two or more eval_runs/<id> directories")
    compare_parser.add_argument("--output", default="",
                                help="write the table here (default: print, plus comparison.md "
                                     "next to the first run)")

    list_parser = subparsers.add_parser("sources", help="list the registered task sources")
    list_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "sources":
        rows = [{"name": s.name, "description": s.description, "default_step_cap": s.default_step_cap,
                 "live_web": s.live_web, "dataset": s.hf_dataset or s.access_url}
                for s in TASK_SOURCES.values()]
        print(json.dumps(rows, indent=2) if args.json else
              "\n".join(f"{r['name']:<20} step cap {r['default_step_cap']:<4} {r['description']}"
                        for r in rows))
        return

    if args.command == "compare":
        runs = [load_run(path) for path in args.runs]
        table = render_comparison(runs)
        print(table)
        output = Path(args.output) if args.output else Path(args.runs[0]).parent / "comparison.md"
        output.write_text(table)
        print(f"\nwrote {output}")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    install_sigterm_handler()
    cfg = _config_from_args(args)
    try:
        results = asyncio.run(run_eval(cfg))
    except TaskSourceUnavailable as e:
        logger.error(str(e))
        raise SystemExit(2)
    metrics = results["metrics"]
    print(f"\nverified task success: {metrics['verified_success_rate'] * 100:.1f}% across "
          f"{metrics['episodes']} {results['harness']['task_source']} tasks, "
          f"parse-failure rate {metrics['parse_failure_rate'] * 100:.1f}%, "
          f"site-blocked {metrics['site_blocked_rate'] * 100:.1f}%")
    warning = coord_space_diagnosis(results["harness"], metrics)
    if warning:
        print(f"\n!! {warning}")
    print(f"report: {EVAL_RUNS_DIR / results['run_id'] / 'report.md'}")


if __name__ == "__main__":
    main()
