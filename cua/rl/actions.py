"""Parse the student's `action:` lines into trajectory action dicts via the shared cua.trajectory.actions registry."""

import re
from dataclasses import dataclass, field

from cua.trajectory.actions import ACTION_SPECS, format_action_for_training

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


def _decode_entry(entry, actions, errors):
    line = _LINE_RE.match(entry)
    if not line:
        errors.append(f"bad_line:{entry[:40]}")
        return True
    name, args = line.group(1), line.group(2)
    spec = ACTION_SPECS.get(name)
    if spec is None or spec.decode is None:
        errors.append(f"unknown_action:{name}")
        return True
    action = spec.decode(args)
    if action is None:
        errors.append(f"bad_args:{name}")
        return True
    actions.append(action)
    return format_action_for_training(action) == entry


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
        canonical = _decode_entry(entry, actions, errors) and canonical
    if not actions and not errors:
        errors.append("no_actions")
    return StepDecode(thought=thought, actions=actions, errors=errors, canonical=canonical and not errors, raw=raw)


def encode_step(thought, actions):
    """The assistant turn exactly as convert emits it."""
    action_str = "\n".join(format_action_for_training(a) for a in actions)
    return f"<think>{thought}</think>\n{action_str}" if thought else action_str
