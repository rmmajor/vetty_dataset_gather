"""The action grammar: one registry mapping each action type to its training line, history summary and decoder."""

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

_REF = r"<ref>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)</ref>"


@dataclass(frozen=True)
class ActionSpec:
    format_training: Callable
    summary: Callable
    decode: Callable = None


def _format_point(action):
    return f"action: {action['type']}(<ref>({action['x']}, {action['y']})</ref>)"


def _format_bare(action):
    return f"action: {action['type']}()"


def _format_click(action):
    return f"action: click(<ref>({action['x']}, {action['y']})</ref>, button={action.get('button', 'left')})"


def _format_drag(action):
    start, end = action["path"][0], action["path"][-1]
    return f"action: drag(<ref>({start['x']}, {start['y']})</ref> -> <ref>({end['x']}, {end['y']})</ref>)"


def _format_type(action):
    return f'action: type("{action["text"]}")'


def _format_keypress(action):
    return f"action: keypress({json.dumps(action['keys'])})"


def _format_scroll(action):
    return f"action: scroll(<ref>({action['x']}, {action['y']})</ref>, scrollX={action['scroll_x']}, scrollY={action['scroll_y']})"


def _format_navigate(action):
    return f'action: navigate("{action.get("url", "")}")'


def _format_handoff(action):
    return f"action: handoff(reason={action.get('reason', 'login')})"


def _proposed_call(proposed):
    if "x" in proposed and "y" in proposed:
        return f"{proposed.get('type') or 'click'}(<ref>({proposed['x']}, {proposed['y']})</ref>)"
    return f"{proposed['type']}()" if proposed.get("type") else "action()"


def _format_confirm(action):
    inner = _proposed_call(action.get("proposed_action") or {})
    return f"action: confirm(propose={inner}, risk={action.get('risk_category', '')})"


def _summary_point(action):
    return f"{action['type']}({action['x']}, {action['y']})"


def _summary_type_only(action):
    return action["type"]


def _summary_text(action):
    return f'type("{action["text"]}")'


def _summary_keypress(action):
    return f"keypress({action['keys']})"


def _summary_scroll(action):
    return f"scroll({action['x']}, {action['y']})"


def _summary_drag(action):
    path = action["path"]
    return f"drag({path[0]['x']},{path[0]['y']} -> {path[-1]['x']},{path[-1]['y']})"


def _summary_navigate(action):
    return f"navigate({action.get('url', '')})"


def _summary_handoff(action):
    return f"handoff({action.get('reason', '')})"


def _summary_confirm(action):
    return f"confirm(risk={action.get('risk_category', '')})"


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


def _bare_decoder(action_type):
    return lambda args: {"type": action_type} if args.strip() == "" else None


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


def _decode_handoff(args):
    match = re.fullmatch(r"reason=([A-Za-z0-9_-]*)", args)
    return {"type": "handoff", "reason": match.group(1) or "login"} if match else None


def _decode_proposed(inner):
    click = re.fullmatch(rf"click\({_REF}\)", inner)
    if click:
        x, y = _xy(click)
        return {"type": "click", "x": x, "y": y}
    bare = re.fullmatch(r"([a-z_]+)\(\)", inner)
    if not bare:
        return None
    return {} if bare.group(1) == "action" else {"type": bare.group(1)}


def _decode_confirm(args):
    match = re.fullmatch(r"propose=(.*),\s*risk=([A-Za-z0-9_-]*)", args, re.DOTALL)
    if not match:
        return None
    proposed = _decode_proposed(match.group(1))
    if proposed is None:
        return None
    return {"type": "confirm", "proposed_action": proposed, "risk_category": match.group(2)}


ACTION_SPECS = {
    "click": ActionSpec(_format_click, _summary_point, _decode_click),
    "double_click": ActionSpec(_format_point, _summary_point, _point_decoder("double_click")),
    "move": ActionSpec(_format_point, _summary_point, _point_decoder("move")),
    "type": ActionSpec(_format_type, _summary_text, _quoted_decoder("type", "text")),
    "keypress": ActionSpec(_format_keypress, _summary_keypress, _decode_keypress),
    "scroll": ActionSpec(_format_scroll, _summary_scroll, _decode_scroll),
    "drag": ActionSpec(_format_drag, _summary_drag, _decode_drag),
    "navigate": ActionSpec(_format_navigate, _summary_navigate, _quoted_decoder("navigate", "url")),
    "wait": ActionSpec(_format_bare, _summary_type_only, _bare_decoder("wait")),
    "done": ActionSpec(_format_bare, _summary_type_only, _bare_decoder("done")),
    "screenshot": ActionSpec(_format_bare, _summary_type_only, _bare_decoder("screenshot")),
    "handoff": ActionSpec(_format_handoff, _summary_handoff, _decode_handoff),
    "confirm": ActionSpec(_format_confirm, _summary_confirm, _decode_confirm),
    # An observation the watchdog records, never a verb the student may emit, so it has no decoder.
    "approval_event": ActionSpec(_format_bare, _summary_type_only),
}
# An unregistered type still renders, so a new producer verb reaches the corpus report instead of crashing convert.
DEFAULT_SPEC = ActionSpec(_format_bare, _summary_type_only)

KEY_CANON = {
    k.lower(): k
    for k in (
        "Enter",
        "Escape",
        "Tab",
        "Backspace",
        "Delete",
        "Home",
        "End",
        "Space",
        "ArrowDown",
        "ArrowUp",
        "ArrowLeft",
        "ArrowRight",
        "PageDown",
        "PageUp",
        "Control",
        "Shift",
        "Alt",
        "Meta",
    )
}


def spec_for(action_type):
    return ACTION_SPECS.get(action_type, DEFAULT_SPEC)


def format_action_for_training(action):
    return spec_for(action["type"]).format_training(action)


def format_action_summary(action):
    return spec_for(action["type"]).summary(action)


def normalize_trajectory_actions(trajectory):
    """Canonical key casing (ENTER -> Enter) in place, so examples and history agree on one spelling."""
    for step in trajectory.get("steps", []):
        for action in step.get("actions", []):
            if action.get("type") == "keypress" and action.get("keys"):
                action["keys"] = [k if len(k) == 1 else KEY_CANON.get(k.lower(), k) for k in action["keys"]]
