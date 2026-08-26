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
