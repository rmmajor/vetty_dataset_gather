"""Decode the student's `action:` lines into trajectory action dicts — the inverse of train_qwen.format_action_for_training."""

import ast
import json
import re
from dataclasses import dataclass, field

from train_qwen import format_action_for_training

_REF = r"<ref>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)</ref>"
_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>[ \t]*\n?", re.DOTALL)
# Reasoning templates PREFILL the opening <think>, so a served completion carries only the closing tag and the thought would otherwise be lost as stray text.
_THINK_PREFILLED_RE = re.compile(r"\A(?!\s*<think>)(.*?)</think>[ \t]*\n?", re.DOTALL)
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
    """Group lines into `action:` entries; a line continuing an unterminated entry (multi-line type text) is appended to it."""
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
    match = _THINK_RE.match(raw) or _THINK_PREFILLED_RE.match(raw)
    thought = match.group(1).strip() if match else ""
    rest = raw[match.end() :] if match else raw
    if not match and "<think>" in raw:
        errors.append("think_malformed")
        rest = "\n".join(line for line in raw.splitlines() if not line.lstrip().startswith("<think>"))
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
