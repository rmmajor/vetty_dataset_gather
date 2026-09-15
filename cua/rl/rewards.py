"""Episode reward is weighted components minus penalties, with every weight and slope in reward_config.json."""

import json
from pathlib import Path

REWARD_CONFIG_PATH = Path(__file__).parent / "reward_config.json"
REWARD_MIN, REWARD_MAX = -1.0, 2.0
# The env-side step cap and the trainer's turn cap are the same under-stopping outcome.
CAP_REASONS = {"step_cap", "max_turns"}


def _terminal_success(record, task, lane):
    if not record.get("success"):
        return 0.0
    return float(lane.get("success_by_efficiency_grade", {}).get(record.get("efficiency_grade") or "", 1.0))


def _format_validity(record, task, lane):
    turns = record.get("turns") or []
    return sum(1 for t in turns if t.get("clean")) / len(turns) if turns else 0.0


def _efficiency(record, task, lane):
    expected = task.get("expected_actions") or 0
    ui_steps = record.get("ui_steps") or 0
    if not record.get("success") or not expected or not ui_steps:
        return 0.0
    zero_at = float(lane.get("efficiency_zero_at_ratio", 2.0))
    return max(0.0, min(1.0, (zero_at - ui_steps / expected) / (zero_at - 1.0)))


def _terminated_with_done(record, task, lane):
    return 1.0 if record.get("success") and record.get("terminal_reason") == "done" else 0.0


def _progress(record, task, lane):
    # Milestone partial credit orders failed groups; a success is scored by terminal_success alone.
    if record.get("success"):
        return 0.0
    milestones = task.get("milestones") or []
    passed = record.get("milestones") or {}
    total = sum(float(m.get("weight", 1.0)) for m in milestones)
    if not total:
        return 0.0
    return sum(float(m.get("weight", 1.0)) for m in milestones if passed.get(m.get("name", ""))) / total


def _false_done(record, task, lane):
    return 1.0 if record.get("terminal_reason") == "done" and not record.get("success") else 0.0


def _reason_in(reasons):
    return lambda record, task, lane: 1.0 if record.get("terminal_reason") in reasons else 0.0


REWARD_COMPONENTS = {
    "terminal_success": _terminal_success,
    "format_validity": _format_validity,
    "efficiency": _efficiency,
    "terminated_with_done": _terminated_with_done,
    "progress": _progress,
}

PENALTIES = {
    "parse_abort": _reason_in({"parse_abort"}),
    "loop_guard": _reason_in({"loop_guard"}),
    "step_cap": _reason_in(CAP_REASONS),
    "false_done": _false_done,
    "control_unexpected": _reason_in({"control_unexpected"}),
    "navigate_blocked": lambda record, task, lane: float(record.get("navigate_blocked") or 0),
}


def load_reward_config(path=REWARD_CONFIG_PATH):
    return json.loads(Path(path).read_text())


def compute_reward(record, task, config):
    lane = config["lanes"][task.get("lane", "autonomous")]
    components = {name: fn(record, task, lane) for name, fn in REWARD_COMPONENTS.items()}
    penalties = {name: fn(record, task, lane) for name, fn in PENALTIES.items()}
    total = sum(lane["weights"].get(n, 0.0) * v for n, v in components.items())
    total -= sum(lane["penalties"].get(n, 0.0) * v for n, v in penalties.items())
    return max(REWARD_MIN, min(REWARD_MAX, round(total, 4))), {"components": components, "penalties": penalties}
