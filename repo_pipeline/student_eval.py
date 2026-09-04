"""Episode runner that measures a served computer-use model on real browser tasks, before vs after fine-tuning."""
import argparse
import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from repo_pipeline import driver_session
from repo_pipeline.agent_capture import VERIFIER_UNAVAILABLE_REASON, verify_trajectory
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
from repo_pipeline import remote_sandbox
from repo_pipeline.sandbox import DockerSandbox, install_sigterm_handler, safe_parallel_sandboxes
from repo_pipeline.sdk_runner import UsageLimitReached
from repo_pipeline.site_blocks import BLOCK_SIGNATURES, BLOCK_SIGNATURES_VERSION, detect_block  # noqa: F401  re-exported
from train_qwen import (
    HISTORY_WINDOW,
    MODEL_COORD_MAX,
    PREV_IMAGE_SIZE,
    PROMPT_FORMAT_VERSION,
    SYSTEM_PROMPT,
    build_step_messages,
    build_step_observation,
    downscale_prev,
    format_action_for_training,
    history_entries_for_step,
    screenshot_changed_fraction,
    to_openai_content,
    to_viewport_coords,
    url_path_change,
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
# Prompt construction — train_qwen's own observation builders, with PNG bytes inlined as data URLs.
# ---------------------------------------------------------------------------


def observed_effect(prev_png: bytes, cur_png: bytes, prev_url: str, cur_url: str) -> dict:
    """What the next screenshot/url showed the last step's actions did, in the shape train_qwen annotates with."""
    return {"changed": screenshot_changed_fraction(prev_png, cur_png), "path": url_path_change(prev_url, cur_url)}


# The observation layout is per-ARM, not global: an sft-v1/v2 checkpoint (and any GRPO LoRA over one)
# trained on ONE screenshot, so sending it the sft3 two-image turn measures the prompt, not the model.
OBSERVATION_FORMATS = ("v2", "v3")
DEFAULT_OBSERVATION = "v3"


def build_student_messages(instruction: str, action_history: list, cur_png: bytes,
                           prev_png: bytes | None = None, observation: str = DEFAULT_OBSERVATION) -> list:
    """OpenAI chat messages for one step; v3 adds the previous screenshot from step 2 on, v2 never does."""
    if observation not in OBSERVATION_FORMATS:
        raise ValueError(f"unknown observation format {observation!r}; expected one of {OBSERVATION_FORMATS}")
    if observation == "v2":
        prev_png = None
    images = {"cur": cur_png}
    if prev_png is not None:
        images["prev"] = downscale_prev(prev_png)
    turn = build_step_observation(instruction, action_history, "cur", "prev" if prev_png is not None else None)
    content = to_openai_content(turn["content"], lambda ref: base64.b64encode(images[ref]).decode())
    return build_step_messages(SYSTEM_PROMPT, {**turn, "content": content})


# ---------------------------------------------------------------------------
# Student backends. Registry, not special cases: a new serving stack is one entry.
# ---------------------------------------------------------------------------


class StudentClient:
    def complete(self, messages: list) -> str:
        raise NotImplementedError


class OpenAIChatStudent(StudentClient):
    """Any OpenAI-compatible /chat/completions endpoint (vLLM, `swift deploy`, SGLang, a hosted API)."""

    def __init__(self, endpoint: str, model: str, api_key: str = "", max_tokens: int = 512,
                 temperature: float = 0.0, timeout_s: int = 180, retries: int = 2, retry_backoff_s: int = 90):
        self.url = self._chat_url(endpoint)
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s

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
            if attempt:
                time.sleep(self.retry_backoff_s * attempt)
            try:
                response = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout_s)
            except requests.RequestException as e:
                last_error = f"transport error: {e}"
                continue
            if response.status_code >= 500:
                # A served model that was preempted or is restarting answers 5xx for minutes; that is a retry, not a verdict on the episode.
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
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
# Action grammar: ONE tolerant definition, enumerated in CLAUDE.md's Student eval section.
# Any UNAMBIGUOUS spelling of a call is accepted and how canonical the output was is reported
# as canonical_form_rate, because a base model has only seen this grammar DESCRIBED in the
# system prompt and charging surface deviation to its success rate would measure imitation.
# ---------------------------------------------------------------------------

ACTION_VERBS = ("click", "double_click", "move", "drag", "scroll", "type", "keypress",
                "wait", "done", "screenshot", "navigate", "handoff", "confirm")
NO_ARG_VERBS = ("wait", "done", "screenshot")
# Yield-control verbs: the episode ends there, but as a scored model outcome, not a parse failure.
HANDOFF_VERBS = ("handoff", "confirm")
# The system prompt documents scroll as scrollX=0, scrollY=500, and the driver defaults match.
DEFAULT_SCROLL = (0, 500)

_BULLET_RE = re.compile(r"^[-*]\s+")
_ACTION_PREFIX_RE = re.compile(r"^action\s*:\s*(.*)$", re.IGNORECASE)
_CALL_RE = re.compile(rf"^({'|'.join(ACTION_VERBS)})\s*(?:\((.*)\))?$", re.IGNORECASE | re.DOTALL)
_REF_WRAPPER_RE = re.compile(r"</?ref>")
# A call counts only when it is the WHOLE line; this hint spots a line that gestures at an action
# without being one, so leading prose is reported as malformed rather than silently ignored.
_ACTION_CALL_HINT_RE = re.compile(rf"\b(?:{'|'.join(ACTION_VERBS)})\s*\(")
_WHITESPACE_RE = re.compile(r"\s+")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Reasoning chat templates PREFILL the opening <think>, so the completion carries only the
# closing tag; without this the think text is lost and think_rate reads 0% on a model that
# reasoned on every step.
_THINK_PREFILLED_RE = re.compile(r"\A(.*?)</think>", re.DOTALL)


class ParsedLine(NamedTuple):
    action: dict | None
    kind: str                 # prefixed | bare | malformed | ignored
    deviations: tuple = ()    # how it differed from the canonical form


def _squash(text: str) -> str:
    return _WHITESPACE_RE.sub("", text)


def _split_top_level(text: str, extra_separators: str = "") -> list:
    """Split on commas outside quotes, parens and brackets, so quoted text survives intact."""
    parts, current, depth, quote = [], [], 0, ""
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if depth == 0 and (char == "," or char in extra_separators):
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def _as_number(token: str):
    try:
        return int(round(float(token)))
    except (TypeError, ValueError):
        return None


def _as_quoted_string(token: str):
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return None


def _take_point(tokens: list, index: int):
    """(x, y, next_index) from one `(x, y)` token or two scalar tokens, else None."""
    if index >= len(tokens):
        return None
    token = tokens[index]
    if token.startswith("(") and token.endswith(")"):
        parts = _split_top_level(token[1:-1])
        if len(parts) != 2:
            return None
        x, y = _as_number(parts[0]), _as_number(parts[1])
        return (x, y, index + 1) if x is not None and y is not None else None
    x, y = _as_number(token), _as_number(tokens[index + 1]) if index + 1 < len(tokens) else None
    return (x, y, index + 2) if x is not None and y is not None else None


def _kwarg_number(kwargs: dict, names: tuple):
    for name in names:
        if name in kwargs:
            return _as_number(kwargs[name])
    return None


def _build_click(tokens, kwargs, note):
    point = _take_point(tokens, 0)
    if point is None:
        return None
    x, y, index = point
    button = kwargs.get("button")
    if button is None and index < len(tokens) and tokens[index].lower() in ("left", "right"):
        button = tokens[index]
        index += 1
    if button is None:
        note("default_kwargs")
    return {"type": "click", "x": x, "y": y, "button": (button or "left").lower()}


def _point_only(action_type: str):
    def build(tokens, kwargs, note):
        point = _take_point(tokens, 0)
        return None if point is None else {"type": action_type, "x": point[0], "y": point[1]}
    return build


def _build_drag(tokens, kwargs, note):
    first = _take_point(tokens, 0)
    if first is None:
        return None
    second = _take_point(tokens, first[2])
    if second is None:
        return None
    return {"type": "drag", "path": [{"x": first[0], "y": first[1]},
                                     {"x": second[0], "y": second[1]}]}


def _build_scroll(tokens, kwargs, note):
    point = _take_point(tokens, 0)
    if point is None:
        return None
    x, y, index = point
    scroll_x = _kwarg_number(kwargs, ("scrollx", "scroll_x", "dx"))
    scroll_y = _kwarg_number(kwargs, ("scrolly", "scroll_y", "dy"))
    if scroll_x is None and index < len(tokens):
        scroll_x = _as_number(tokens[index])
        index += 1
    if scroll_y is None and index < len(tokens):
        scroll_y = _as_number(tokens[index])
    if scroll_x is None or scroll_y is None:
        note("default_kwargs")
    return {"type": "scroll", "x": x, "y": y,
            "scroll_x": DEFAULT_SCROLL[0] if scroll_x is None else scroll_x,
            "scroll_y": DEFAULT_SCROLL[1] if scroll_y is None else scroll_y}


def _build_type(tokens, kwargs, note):
    # Exactly one token: a quoted literal keeps its own commas, and an unquoted one is ambiguous.
    if len(tokens) != 1:
        return None
    text = _as_quoted_string(tokens[0])
    if text is None:
        return None
    if not tokens[0].startswith('"'):
        note("alt_quotes")
    return {"type": "type", "text": text}


def _build_keypress(tokens, kwargs, note):
    if not tokens:
        return None
    if len(tokens) == 1 and tokens[0].startswith("[") and tokens[0].endswith("]"):
        try:
            keys = json.loads(tokens[0].replace("'", '"'))
        except json.JSONDecodeError:
            return None
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
            return None
        return {"type": "keypress", "keys": keys}
    keys = []
    for token in tokens:
        literal = _as_quoted_string(token)
        token = literal if literal is not None else token
        combo = [part for part in token.split("+") if part.strip()]
        keys.extend(combo if len(combo) > 1 else [token])
    keys = [key.strip() for key in keys if key.strip()]
    if not keys:
        return None
    note("bare_keys")
    return {"type": "keypress", "keys": keys}


def _no_args(action_type: str):
    def build(tokens, kwargs, note):
        if tokens or kwargs:
            note("extra_args")
        return {"type": action_type}
    return build


def _build_navigate(tokens, kwargs, note):
    token = kwargs.get("url") if "url" in kwargs else (tokens[0] if len(tokens) == 1 else None)
    if token is None:
        return None
    url = _as_quoted_string(token)
    if url is None:
        # An unquoted scheme-less token could be a domain or a search phrase; require the scheme.
        if not re.match(r"https?://", token):
            return None
        url = token
        note("bare_url")
    elif not token.startswith('"'):
        note("alt_quotes")
    return {"type": "navigate", "url": url}


def _build_handoff(tokens, kwargs, note):
    if len(tokens) > 1 or (tokens and "reason" in kwargs):
        return None
    raw = kwargs.get("reason", tokens[0] if tokens else "")
    reason = _as_quoted_string(raw)
    return {"type": "handoff", "reason": (raw if reason is None else reason).strip()}


def _build_confirm(tokens, kwargs, note):
    if tokens:
        return None
    proposed = _parse_call(kwargs["propose"])[0] if kwargs.get("propose") else None
    raw_risk = kwargs.get("risk", "")
    risk = _as_quoted_string(raw_risk)
    action = {"type": "confirm", "risk_category": (raw_risk if risk is None else risk).strip()}
    if proposed is not None:
        action["proposed_action"] = proposed
    return action


ACTION_BUILDERS: dict = {
    "click": _build_click,
    "double_click": _point_only("double_click"),
    "move": _point_only("move"),
    "drag": _build_drag,
    "scroll": _build_scroll,
    "type": _build_type,
    "keypress": _build_keypress,
    "navigate": _build_navigate,
    "wait": _no_args("wait"),
    "done": _no_args("done"),
    "screenshot": _no_args("screenshot"),
    "navigate": _build_navigate,
    "handoff": _build_handoff,
    "confirm": _build_confirm,
}


def _parse_call(body: str):
    """(action, deviations) for one call expression, or (None, ()) when it is not one."""
    match = _CALL_RE.match(body.strip())
    if not match:
        return None, ()
    verb = match.group(1).lower()
    raw_args = match.group(2)
    deviations = []
    note = deviations.append

    if raw_args is None:
        if verb not in NO_ARG_VERBS:
            return None, ()
        note("missing_parens")
        raw_args = ""

    had_ref = "<ref>" in raw_args
    args = _REF_WRAPPER_RE.sub("", raw_args)
    if verb == "drag":
        args = args.replace("->", ",")

    tokens, kwargs = [], {}
    for token in _split_top_level(args):
        key, sep, value = token.partition("=")
        if sep and _as_quoted_string(token) is None and re.fullmatch(r"[A-Za-z_][\w]*", key.strip()):
            kwargs[key.strip().lower()] = value.strip()
        else:
            tokens.append(token)

    action = ACTION_BUILDERS[verb](tokens, kwargs, note)
    if action is None:
        return None, ()
    if not had_ref and any(key in action for key in ("x", "path")):
        note("bare_coords")
    known = {"click": {"button"}, "scroll": {"scrollx", "scrolly", "scroll_x", "scroll_y", "dx", "dy"},
             "navigate": {"url"}, "handoff": {"reason"}, "confirm": {"propose", "risk"}}
    if set(kwargs) - known.get(verb, set()):
        note("extra_args")
    return action, tuple(dict.fromkeys(deviations))


def classify_action_line(line: str) -> ParsedLine:
    """Classify one line as prefixed | bare | malformed | ignored, with its action and deviations."""
    stripped = _BULLET_RE.sub("", line.strip().strip("`").strip(), count=1)
    if not stripped:
        return ParsedLine(None, "ignored")
    prefix_match = _ACTION_PREFIX_RE.match(stripped)
    body = prefix_match.group(1).strip() if prefix_match else stripped
    action, deviations = _parse_call(body)
    if action is None:
        if prefix_match or _ACTION_CALL_HINT_RE.search(stripped):
            return ParsedLine(None, "malformed")
        return ParsedLine(None, "ignored")
    if not prefix_match:
        deviations = ("missing_prefix",) + deviations
    if _squash(stripped) == _squash(format_action_for_training(action)):
        deviations = ()
    elif not deviations:
        deviations = ("other",)
    return ParsedLine(action, "prefixed" if prefix_match else "bare", deviations)


def parse_action_line(line: str) -> dict | None:
    """One action line back into a trajectory action dict, or None if it is not a valid action."""
    return classify_action_line(line).action


@dataclass
class ParsedStep:
    think: str
    actions: list
    malformed_lines: int
    raw: str
    canonical_actions: int = 0
    deviations: dict = field(default_factory=dict)

    @property
    def parse_failed(self) -> bool:
        return not self.actions


def parse_student_output(text: str, tolerant_grammar: bool = True) -> ParsedStep:
    """Split one assistant turn into its `<think>` text and ordered action dicts; a step fails only when NO line parsed."""
    text = text or ""
    think_re = _THINK_RE if "<think>" in text else _THINK_PREFILLED_RE
    think_match = think_re.search(text)
    think = (think_match.group(1).strip() if think_match else "")
    body = think_re.sub("", text, count=1)

    actions, malformed, canonical = [], 0, 0
    deviations: dict = {}
    for line in body.splitlines():
        parsed = classify_action_line(line)
        if parsed.kind == "ignored":
            continue
        if parsed.kind == "malformed" or (parsed.deviations and not tolerant_grammar):
            malformed += 1
            continue
        for deviation in parsed.deviations:
            deviations[deviation] = deviations.get(deviation, 0) + 1
        canonical += not parsed.deviations
        actions.append(parsed.action)
    return ParsedStep(think=think, actions=actions, malformed_lines=malformed, raw=text,
                      canonical_actions=canonical, deviations=deviations)


# ---------------------------------------------------------------------------
# Coordinate-space registry: how the model's numbers relate to viewport pixels.
# PER-MODEL and never auto-detected (stock Qwen and sft3 answer in normalized_1000, sft-v1/v2 in viewport_abs); scoring one in the other's space yields a confident wrong number, not an error.
# ---------------------------------------------------------------------------

VIEWPORT_W = 1280
VIEWPORT_H = 720

COORD_SPACES: dict = {
    "viewport_abs": lambda x, y: (round(x), round(y)),
    "normalized_1000": to_viewport_coords,
}

# Per space: the inclusive range an emitted coordinate must respect, and the diagnosis when it does not.
COORD_SPACE_RANGE: dict = {
    "viewport_abs": (VIEWPORT_W - 1, VIEWPORT_H - 1),
    "normalized_1000": (MODEL_COORD_MAX, MODEL_COORD_MAX),
}
COORD_SPACE_SUSPECT: dict = {
    "viewport_abs": ("the model is likely answering in normalized 0-1000 coordinates", "normalized_1000"),
    "normalized_1000": ("the model answers in pixels: an sft-v1/v2 checkpoint?", "viewport_abs"),
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


def outside_coord_space(x, y, coord_space: str) -> bool:
    max_x, max_y = COORD_SPACE_RANGE[coord_space]
    return not (0 <= x <= max_x and 0 <= y <= max_y)


def outside_viewport(x, y) -> bool:
    return outside_coord_space(x, y, "viewport_abs")


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
    "navigate": lambda a: {"kind": "navigate", "url": a["url"]},
    # Terminal in the episode loop before dispatch; there is no driver action to execute.
    "handoff": lambda a: None,
    "confirm": lambda a: None,
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


SNAPSHOT_RECORD_CHARS = 4000


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
    coord_space: str = "normalized_1000"
    type_mode: str = "replace"
    browser_backend: str = "docker"
    browser_host: str = ""
    egress_note: str = ""
    tolerant_grammar: bool = True
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
    observation: str = DEFAULT_OBSERVATION
    resume: bool = True
    dataset_revision: str = "main"
    trajectories_dir: str = "trajectories"
    loop_guard_repeats: int = 0
    sandbox_image: str = ""

    def __post_init__(self):
        if self.screenshot_format != "png":
            raise ValueError(
                f"screenshot_format={self.screenshot_format!r}: {PROMPT_FORMAT_VERSION} effect annotations compare "
                f"consecutive screenshots pixel-wise and a lossy frame never compares identical — use png")


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
    actions_canonical: int = 0
    form_deviations: dict = field(default_factory=dict)
    think_steps: int = 0
    driver_errors: int = 0
    out_of_bounds: int = 0
    emitted_coords: int = 0
    emitted_out_of_viewport: int = 0
    coord_space_warning: str = ""
    done_emitted: bool = False
    navigations: int = 0
    handoff_kind: str = ""
    handoff_reason: str = ""
    block_reason: str = ""
    block_step: int = 0
    error: str = ""
    duration_s: float = 0.0
    episode_dir: str = ""
    raw_last_response: str = ""
    # The observation the episode ended on. Screenshots alone cannot be re-scanned for block
    # signatures, so without this a finished run can never be re-classified when the set grows.
    final_url: str = ""
    final_snapshot: str = ""


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


def _docker_browser(cfg):
    kwargs = {"container_timeout": EVAL_CONTAINER_TIMEOUT}
    if cfg.sandbox_image:
        kwargs["image"] = cfg.sandbox_image
    return DockerSandbox(**kwargs)


def _remote_ssh_browser(cfg):
    host, _, port = cfg.browser_host.partition(":")
    return remote_sandbox.RemoteSandbox(host=host, port=int(port or 22),
                                        container_timeout=EVAL_CONTAINER_TIMEOUT)


def _vast_residential_browser(cfg):
    return remote_sandbox.VastResidentialSandbox(container_timeout=EVAL_CONTAINER_TIMEOUT)


# Where the browser runs, and therefore what IP live sites see. Adding an egress is a registry
# entry, the same idiom as TASK_SOURCES / PREDICTOR_BACKENDS.
BROWSER_BACKENDS = {
    "docker": _docker_browser,
    "remote_ssh": _remote_ssh_browser,
    "vast_residential": _vast_residential_browser,
}


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
        self.sandbox = BROWSER_BACKENDS[self.cfg.browser_backend](self.cfg)
        await _run(self.sandbox.start, f"eval/{self.name}")
        # Record the egress the sites actually saw, so the fingerprint names the ASN rather than
        # just the backend — two residential rentals are different IPs.
        egress = getattr(self.sandbox, "egress", None)
        if egress and not self.cfg.egress_note:
            self.cfg.egress_note = (f"{egress.get('country', '?')}/{egress.get('city', '?')}/"
                                    f"{egress.get('org', '?')}")
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
    """Count emitted coordinates outside the declared space's range and name the likely coordinate-space fix once per episode."""
    hint, other_space = COORD_SPACE_SUSPECT[cfg.coord_space]
    max_x, max_y = COORD_SPACE_RANGE[cfg.coord_space]
    for action in actions:
        for x, y in action_coordinates(action):
            record.emitted_coords += 1
            if not outside_coord_space(x, y, cfg.coord_space):
                continue
            record.emitted_out_of_viewport += 1
            if record.coord_space_warning:
                continue
            record.coord_space_warning = (
                f"emitted ({x}, {y}) is outside 0-{max_x} x 0-{max_y} under coord_space={cfg.coord_space}; "
                f"{hint} — re-run with --coord-space {other_space}"
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
    prev_png: bytes | None = None
    prev_url = ""
    executed_actions: list = []
    consecutive_parse_failures = 0
    consecutive_unproductive = 0
    container_dir = f"{RESULTS_DIR}/{episode_dir_name(task)}"
    session_open = False
    last_url, last_snapshot = task.start_url, ""

    await browser.ensure()
    client = browser.client

    try:
        try:
            session = await _run(lambda: client.session(
                task.task_id, task.start_url, task.instruction,
                {"expected_actions": task.reference_length, "eval_label": cfg.label, "eval_source": task.source},
                type_mode=cfg.type_mode))
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
                cur_png = base64.b64decode(await _run(client.screenshot_b64, cfg.screenshot_format))
            except (DriverError, ConnectionError) as e:
                record.outcome = "harness_error"
                record.error = f"screenshot failed: {str(e)[:250]}"
                break

            cur_url = last_url
            if prev_png is not None:
                # The last step enters the history only now: its effect is what THIS screenshot/url show.
                action_history.extend(history_entries_for_step(
                    {"actions": executed_actions}, observed_effect(prev_png, cur_png, prev_url, cur_url)))
            messages = build_student_messages(task.instruction, action_history, cur_png, prev_png,
                                              observation=cfg.observation)
            prev_png, prev_url, executed_actions = cur_png, cur_url, []
            try:
                raw = await _run(student.complete, messages)
            except StudentError as e:
                record.outcome = "harness_error"
                record.error = f"student: {str(e)[:250]}"
                break

            record.model_steps = step
            record.raw_last_response = (raw or "")[-1500:]
            parsed = parse_student_output(raw, cfg.tolerant_grammar)
            record.malformed_lines += parsed.malformed_lines
            record.actions_parsed += len(parsed.actions)
            record.actions_canonical += parsed.canonical_actions
            for deviation, count in parsed.deviations.items():
                record.form_deviations[deviation] = record.form_deviations.get(deviation, 0) + count
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
            handed_off = False
            executed_this_step = 0
            for action in parsed.actions:
                if action["type"] in HANDOFF_VERBS:
                    record.handoff_kind = action["type"]
                    record.handoff_reason = action.get("reason") or action.get("risk_category", "")
                    terminated = True
                    handed_off = True
                    break
                executed = scale_action(action, cfg.coord_space)
                payload = driver_payload(executed, parsed.think)
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
                if action["type"] == "navigate":
                    record.navigations += 1
                # History is derived from the EXECUTED pixel action, so its coordinates round-trip as in training.
                executed_actions.append(executed)
                last_url = result.get("url", last_url)
                last_snapshot = result.get("snapshot", last_snapshot)

            if terminated:
                record.outcome = "handoff" if handed_off else "completed"
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
        record.final_url = last_url or ""
        record.final_snapshot = (last_snapshot or "")[:SNAPSHOT_RECORD_CHARS]
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
    # verify_trajectory returns its `success: False` fallback for a wedged CLI, a budget cap or an
    # unparseable reply. Scoring that as a model failure charges the model for an unmeasured episode,
    # so an unavailable verifier defers exactly as a usage limit does.
    if verdict.get("reasoning") == VERIFIER_UNAVAILABLE_REASON:
        record.judge_deferred = True
        record.judge_reasoning = VERIFIER_UNAVAILABLE_REASON
        return
    # `final_success` carries the degenerate-loop override the capture gate applies; `success` is the
    # judge's raw verdict. Reading the raw key would score a loop the corpus refuses as a win.
    record.judged_success = bool(verdict.get("claimed_success", verdict.get("success")))
    record.verified_success = bool(verdict.get("final_success", verdict.get("success")))
    record.efficiency = verdict.get("efficiency") or ""
    record.judge_reasoning = (verdict.get("reasoning") or "")[:400]


# ---------------------------------------------------------------------------
# Metrics + reporting
# ---------------------------------------------------------------------------


def _format_deviations(deviations: dict) -> str:
    if not deviations:
        return "none"
    return ", ".join(f"{kind} {count}" for kind, count in deviations.items())


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
    actions_canonical = sum(r.actions_canonical for r in records)
    form_deviations: dict = {}
    for record in records:
        for deviation, count in (record.form_deviations or {}).items():
            form_deviations[deviation] = form_deviations.get(deviation, 0) + count

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
        "actions_canonical": actions_canonical,
        "canonical_form_rate": _rate(actions_canonical, actions_parsed),
        "action_form_deviations": dict(sorted(form_deviations.items(), key=lambda kv: -kv[1])),
        "done_rate": _rate(sum(1 for r in records if r.done_emitted), total),
        "navigate_actions": sum(r.navigations for r in records),
        "handoff_episodes": sum(1 for r in records if r.handoff_kind),
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
        # Parity keys: the observation layout. A run prompted in an older format measures the prompt, not the model.
        "prompt_format": PROMPT_FORMAT_VERSION,
        "observation": cfg.observation,
        "prev_image_size": f"{PREV_IMAGE_SIZE[0]}x{PREV_IMAGE_SIZE[1]}",
        "effect_annotations": True,
        "type_mode": cfg.type_mode,
        # Parity key, deliberately NOT expected-to-differ: grammar tolerance is a property of
        # the harness, so both sides of a delta must have been parsed the same way.
        "action_grammar": "tolerant" if cfg.tolerant_grammar else "canonical",
        # Also a parity key: adding a verb changes what counts as a parse failure.
        "action_verbs": ",".join(ACTION_VERBS),
        # Parity key: the signature set decides which episodes leave the success denominator.
        "block_signatures": f"v{BLOCK_SIGNATURES_VERSION}:{','.join(n for n, _ in BLOCK_SIGNATURES)}",
        # Parity key: which IP the sites saw. 58% of Online-Mind2Web is reachable from a US
        # residential exit against ~42% from this machine, so comparing runs on different egress
        # measures the network rather than the model.
        "browser_backend": cfg.browser_backend,
        "egress": cfg.egress_note,
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
    space = fingerprint.get("coord_space")
    if space not in COORD_SPACE_SUSPECT:
        return ""
    rate = metrics.get("emitted_out_of_viewport_rate", 0)
    if rate < COORD_SPACE_SUSPECT_RATE or not metrics.get("emitted_coords"):
        return ""
    hint, other_space = COORD_SPACE_SUSPECT[space]
    max_x, max_y = COORD_SPACE_RANGE[space]
    return (
        f"COORDINATE SPACE SUSPECT: {rate * 100:.0f}% of emitted coordinates "
        f"({metrics['emitted_out_of_viewport']}/{metrics['emitted_coords']}) fall outside "
        f"0-{max_x} x 0-{max_y} under coord_space={space}. {hint}; re-run with --coord-space {other_space} "
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
        f"| prompt format | `{fingerprint['prompt_format']}` |",
        f"| previous-screenshot size | {fingerprint['prev_image_size']} (from step 2 on) |",
        f"| effect annotations | {'on' if fingerprint['effect_annotations'] else 'off'} |",
        f"| type mode | {fingerprint['type_mode']} |",
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
        f"| canonical action form | {metrics.get('canonical_form_rate', 0) * 100:.1f}% "
        f"({metrics.get('actions_canonical', 0)}/{metrics.get('actions_parsed', 0)} actions) |",
        f"| form deviations | {_format_deviations(metrics.get('action_form_deviations'))} |",
        f"| done() emission rate | {metrics['done_rate'] * 100:.1f}% |",
        f"| navigate actions executed | {metrics.get('navigate_actions', 0)} |",
        f"| episodes ended by handoff/confirm | {metrics.get('handoff_episodes', 0)} |",
        f"| <think> emission rate (steps) | {metrics['think_rate'] * 100:.1f}% |",
        f"| driver-refused action rate | {metrics['driver_error_rate'] * 100:.1f}% |",
        f"| emitted coords outside the declared coordinate space | {metrics.get('emitted_out_of_viewport', 0)}"
        f"/{metrics.get('emitted_coords', 0)} "
        f"({metrics.get('emitted_out_of_viewport_rate', 0) * 100:.1f}%) |",
        f"| out-of-viewport actions refused | {metrics['out_of_bounds_actions']} |",
        f"| episodes abandoned as stuck | {metrics.get('stuck_episodes', 0)} |",
        f"| mean steps on success | {metrics['mean_steps_on_success']} |",
        f"| mean steps / reference_length on success | {metrics['mean_steps_over_reference_on_success']}x |",
        f"| mean steps (all) | {metrics['mean_steps_all']} |",
        f"| episodes that hit the step cap | {metrics['step_cap_hits']} |",
        "",
        "`verified` is the judge's verdict on the ORIGINAL goal after its own criteria checklist is "
        "enforced (`judged` is the raw claim); the efficiency grade is reported beside it rather than "
        "folded into it. `steps / reference_length` compares against the "
        "benchmark's own human step count, which is a more defensible efficiency denominator than "
        "our raw step counts.",
        "",
        "`canonical action form` is the share of parsed actions spelled exactly as "
        "`format_action_for_training` emits them. The parser accepts any UNAMBIGUOUS spelling of a "
        "call (optional `action:`, bare or `<ref>`-wrapped coordinates, `wait` with or without "
        "`()`, omitted keyword arguments, single or double quotes, bare key names) and rejects "
        "genuinely ambiguous ones. A base model has only seen the grammar described in the system "
        "prompt, so charging surface deviation to its success rate would measure imitation of an "
        "unfamiliar convention rather than browsing. The deviation is reported here instead, and "
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


# A before/after pair MAY differ here: an sft-v1/v2 checkpoint answers in viewport_abs, so each side is measured in its own convention.
EXPECTED_TO_DIFFER_KEYS = ("coord_space",)
# Provenance, not harness settings — these can never make a delta unattributable.
PARITY_EXEMPT_KEYS = ("git_commit", "task_source_description", "level_distribution") + EXPECTED_TO_DIFFER_KEYS


def render_comparison(runs: list[dict]) -> str:
    """Before/after table over two or more runs, with the parity check and per-level and per-domain diffs."""
    lines = ["# Student eval — comparison", "",
             "| label | verified success | all-episode success | parse-failure | canonical form | "
             "done() rate | mean steps (success) | steps/ref | site-blocked | n |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in runs:
        m = run["metrics"]
        lines.append(
            f"| `{run['label']}` | {m['verified_success_rate'] * 100:.1f}% | "
            f"{m['verified_success_rate_all_episodes'] * 100:.1f}% | "
            f"{m['parse_failure_rate'] * 100:.1f}% | "
            f"{m.get('canonical_form_rate', 0) * 100:.1f}% | {m['done_rate'] * 100:.1f}% | "
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
                  "| label | verified success | parse-failure | canonical form | done() rate | "
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
                f"{(b.get('canonical_form_rate', 0) - a.get('canonical_form_rate', 0)) * 100:+.1f} pp | "
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
    parser.add_argument("--coord-space", default=os.environ.get("EVAL_COORD_SPACE", "normalized_1000"),
                        choices=sorted(COORD_SPACES),
                        help="the convention THIS model answers in: a stock Qwen and an sft3 checkpoint emit "
                             "normalized_1000 (the prompt's 0-1000 grid); an sft-v1/v2 checkpoint emits "
                             "viewport_abs (1280x720 pixels). Set it explicitly per model — it is never "
                             "auto-detected")
    parser.add_argument("--type-mode", default="replace", choices=("replace", "append"),
                        help="driver typing mode; replace selects the focused field's text before typing, "
                             "as capture and RL do")
    parser.add_argument("--strict-action-grammar", action="store_true",
                        help="accept ONLY the canonical form format_action_for_training emits. Off "
                             "by default: any unambiguous spelling of a call is accepted, and how "
                             "canonical each model's output was is reported as canonical_form_rate "
                             "instead of being charged to its success rate")
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
    parser.add_argument("--screenshot-format", default="png", choices=("png", "jpeg"),
                        help="png only under sft3: effect annotations compare consecutive frames pixel-wise")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip verification (harness-only run; no Claude calls, no success rate)")
    parser.add_argument("--judge-mid-screenshots", type=int, default=2,
                        help="mid-run frames per verdict on top of the final one; the main cost dial")
    parser.add_argument("--observation", default=DEFAULT_OBSERVATION, choices=OBSERVATION_FORMATS,
                        help="observation layout: v3 sends prev+current screenshots, v2 sends current "
                             "only (required for sft-v1/v2 and any GRPO LoRA over them)")
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
    parser.add_argument("--browser-backend", default="docker", choices=sorted(BROWSER_BACKENDS),
                        help="where the browser runs, i.e. which IP the sites see. "
                             "`vast_residential` rents a US consumer-ISP box and destroys it after.")
    parser.add_argument("--browser-host", default="",
                        help="remote_ssh backend: host or host:port of an already-running box")


def _config_from_args(args) -> EvalConfig:
    return EvalConfig(
        task_source=args.task_source, tasks_file=args.tasks_file, limit=args.limit,
        label=args.label, run_id=args.run_id, student=args.student, endpoint=args.endpoint,
        model=args.model, adapter=args.adapter, api_key=args.api_key,
        temperature=args.temperature, max_tokens=args.max_tokens,
        student_timeout_s=args.student_timeout, student_retries=args.student_retries,
        mock_trajectory=args.mock_trajectory, coord_space=args.coord_space, type_mode=args.type_mode,
        tolerant_grammar=not args.strict_action_grammar,
        max_unproductive_steps=args.max_unproductive_steps,
        max_parse_failures=args.max_parse_failures, max_steps=args.max_steps,
        step_cap_multiple=args.step_cap_multiple, step_cap_floor=args.step_cap_floor,
        step_cap_ceiling=args.step_cap_ceiling, stratify=args.stratify,
        episode_timeout_s=args.episode_timeout, screenshot_format=args.screenshot_format,
        judge=not args.no_judge, judge_mid_screenshots=args.judge_mid_screenshots,
        parallel=args.parallel, observation=args.observation, resume=not args.no_resume,
        dataset_revision=args.dataset_revision, trajectories_dir=args.trajectories_dir,
        loop_guard_repeats=args.loop_guard_repeats, sandbox_image=args.sandbox_image,
        browser_backend=args.browser_backend, browser_host=args.browser_host,
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
    try:
        cfg = _config_from_args(args)
    except ValueError as e:
        parser.error(str(e))
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
