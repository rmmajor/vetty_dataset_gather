"""Turn a decoded student step into the driver's `sequence` payload."""

from cua.trajectory import VIEWPORT

MISSING_THOUGHT_PLACEHOLDER = "(no reasoning emitted)"
CONTROL_TYPES = ("done", "handoff", "confirm")
# Uppercased aliases → Playwright key names; local copy because the Path-A recorder that owns the full map imports `openai`.
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
