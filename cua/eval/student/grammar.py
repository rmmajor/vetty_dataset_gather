"""ONE tolerant definition of the student action grammar, enumerated in docs/evaluation.md's student-eval section."""

import json
import re
from dataclasses import dataclass, field
from typing import NamedTuple

from cua.trajectory.actions import format_action_for_training

# Any UNAMBIGUOUS spelling is accepted and reported as canonical_form_rate: a base model only saw it DESCRIBED.
ACTION_VERBS = (
    "click",
    "double_click",
    "move",
    "drag",
    "scroll",
    "type",
    "keypress",
    "wait",
    "done",
    "screenshot",
    "navigate",
    "handoff",
    "confirm",
)
NO_ARG_VERBS = ("wait", "done", "screenshot")
# Yield-control verbs: the episode ends there, but as a scored model outcome, not a parse failure.
HANDOFF_VERBS = ("handoff", "confirm")
# The system prompt documents scroll as scrollX=0, scrollY=500, and the driver defaults match.
DEFAULT_SCROLL = (0, 500)

_BULLET_RE = re.compile(r"^[-*]\s+")
_ACTION_PREFIX_RE = re.compile(r"^action\s*:\s*(.*)$", re.IGNORECASE)
_CALL_RE = re.compile(rf"^({'|'.join(ACTION_VERBS)})\s*(?:\((.*)\))?$", re.IGNORECASE | re.DOTALL)
_REF_WRAPPER_RE = re.compile(r"</?ref>")
# A call counts only as a WHOLE line; this hint reports a line that gestures at one as malformed, never ignored.
_ACTION_CALL_HINT_RE = re.compile(rf"\b(?:{'|'.join(ACTION_VERBS)})\s*\(")
_WHITESPACE_RE = re.compile(r"\s+")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Reasoning chat templates PREFILL the opening <think>, so without this think_rate reads 0% on a model that reasoned.
_THINK_PREFILLED_RE = re.compile(r"\A(.*?)</think>", re.DOTALL)


class ParsedLine(NamedTuple):
    action: dict | None
    kind: str  # prefixed | bare | malformed | ignored
    deviations: tuple = ()  # how it differed from the canonical form


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
    return {"type": "drag", "path": [{"x": first[0], "y": first[1]}, {"x": second[0], "y": second[1]}]}


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
    return {
        "type": "scroll",
        "x": x,
        "y": y,
        "scroll_x": DEFAULT_SCROLL[0] if scroll_x is None else scroll_x,
        "scroll_y": DEFAULT_SCROLL[1] if scroll_y is None else scroll_y,
    }


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


# One builder per verb: a new verb is an entry here plus one in ACTION_VERBS, never a branch in the parser.
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
    "handoff": _build_handoff,
    "confirm": _build_confirm,
}

# Keyword arguments each verb understands; anything else parses but is reported as extra_args.
KNOWN_KWARGS: dict = {
    "click": {"button"},
    "scroll": {"scrollx", "scrolly", "scroll_x", "scroll_y", "dx", "dy"},
    "navigate": {"url"},
    "handoff": {"reason"},
    "confirm": {"propose", "risk"},
}


def _split_arguments(args: str):
    """(positional tokens, lowercased kwargs) of one call's argument text."""
    tokens, kwargs = [], {}
    for token in _split_top_level(args):
        key, sep, value = token.partition("=")
        if sep and _as_quoted_string(token) is None and re.fullmatch(r"[A-Za-z_][\w]*", key.strip()):
            kwargs[key.strip().lower()] = value.strip()
        else:
            tokens.append(token)
    return tokens, kwargs


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

    tokens, kwargs = _split_arguments(args)
    action = ACTION_BUILDERS[verb](tokens, kwargs, note)
    if action is None:
        return None, ()
    if not had_ref and any(key in action for key in ("x", "path")):
        note("bare_coords")
    if set(kwargs) - KNOWN_KWARGS.get(verb, set()):
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


def _think_text(text: str):
    """(think text, body with the think block removed), tolerating a template-prefilled opening tag."""
    think_re = _THINK_RE if "<think>" in text else _THINK_PREFILLED_RE
    think_match = think_re.search(text)
    return (think_match.group(1).strip() if think_match else ""), think_re.sub("", text, count=1)


# NOT cua.rl.actions.decode_step, the formatter's strict inverse: this reading is the `action_grammar` parity key.
def parse_student_output(text: str, tolerant_grammar: bool = True) -> ParsedStep:
    """Split one assistant turn into `<think>` text and ordered actions; a step fails only when NO line parsed."""
    text = text or ""
    think, body = _think_text(text)

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
    return ParsedStep(
        think=think,
        actions=actions,
        malformed_lines=malformed,
        raw=text,
        canonical_actions=canonical,
        deviations=deviations,
    )
