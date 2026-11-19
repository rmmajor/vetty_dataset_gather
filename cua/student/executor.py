"""A decoded student step into driver calls: coordinate spaces, viewport bounds, and the two lanes' /act payloads."""

from cua.prompt.coords import MODEL_COORD_MAX, to_viewport_coords
from cua.trajectory import VIEWPORT
from cua.trajectory.actions import TERMINAL_TYPES

VIEWPORT_W, VIEWPORT_H = VIEWPORT

# Both are recorded in trajectories, so the RL lane's wording and the eval lane's stay as they were written.
MISSING_THOUGHT_PLACEHOLDER = "(no reasoning emitted)"
NO_THINK_PLACEHOLDER = "(model emitted no <think> block)"

# Uppercased aliases → Playwright key names; a local copy because the recorder owning the full map imports `openai`.
KEY_ALIASES = {
    "CTRL": "Control",
    "CONTROL": "Control",
    "ALT": "Alt",
    "SHIFT": "Shift",
    "META": "Meta",
    "CMD": "Meta",
    "COMMAND": "Meta",
    "ENTER": "Enter",
    "RETURN": "Enter",
    "TAB": "Tab",
    "ESCAPE": "Escape",
    "ESC": "Escape",
    "BACKSPACE": "Backspace",
    "DELETE": "Delete",
    "SPACE": " ",
    "UP": "ArrowUp",
    "DOWN": "ArrowDown",
    "LEFT": "ArrowLeft",
    "RIGHT": "ArrowRight",
    "PAGEUP": "PageUp",
    "PAGEDOWN": "PageDown",
    "HOME": "Home",
    "END": "End",
}

# PER-MODEL and never auto-detected (stock Qwen and sft3 answer in normalized_1000, sft-v1/v2 in viewport_abs).
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

# Action keys holding a SCREEN POSITION; scroll_x/scroll_y are relative wheel deltas, so scaling them would be a guess.
_POSITION_KEYS = ("x", "y")


def normalize_keys(keys):
    return [KEY_ALIASES.get(k.upper(), k) for k in keys]


def scale_action(action: dict, coord_space: str) -> dict:
    """The action as it will be EXECUTED: model coordinates mapped into viewport pixels."""
    to_pixels = COORD_SPACES[coord_space]
    if action.get("type") == "drag":
        return {
            **action,
            "path": [dict(zip(_POSITION_KEYS, to_pixels(p["x"], p["y"]), strict=False)) for p in action["path"]],
        }
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


def in_viewport(x, y):
    return 0 <= x < VIEWPORT_W and 0 <= y < VIEWPORT_H


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
        if a["type"] in TERMINAL_TYPES:
            return i
    return None


def to_sequence_payload(thought, actions):
    """The RL lane's payload: one /act sequence per policy turn, cut after the first terminal action."""
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


# Action -> driver payload: the coordinate-native /act kinds, keyed by action type.
ACTION_TO_DRIVER: dict = {
    "click": lambda a: {"kind": "click_xy", "x": a["x"], "y": a["y"], "button": a.get("button", "left")},
    "double_click": lambda a: {"kind": "click_xy", "x": a["x"], "y": a["y"], "dblclick": True},
    "move": lambda a: {"kind": "move_xy", "x": a["x"], "y": a["y"]},
    "type": lambda a: {"kind": "type_text", "text": a["text"]},
    "keypress": lambda a: {"kind": "press", "keys": a["keys"]},
    "scroll": lambda a: {
        "kind": "scroll_xy",
        "x": a["x"],
        "y": a["y"],
        "dx": a.get("scroll_x", 0),
        "dy": a.get("scroll_y", 0),
    },
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
    """The student-eval lane's payload: one /act call per action, dispatched in order."""
    builder = ACTION_TO_DRIVER.get(action.get("type"))
    if builder is None:
        return None
    payload = builder(action)
    if payload is None:
        return None
    payload["thought"] = thought or NO_THINK_PLACEHOLDER
    return payload
