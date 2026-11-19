"""Pixel <-> model coordinate space: trajectory.json stays in pixels, the student reads and writes 0-1000."""

from cua.settings import parse_viewport as parse_viewport
from cua.trajectory import VIEWPORT

COORD_SPACE = "normalized_1000"
MODEL_COORD_MAX = 1000

_PATH_KEYS = ("path",)
_NESTED_ACTION_KEYS = ("proposed_action",)


def to_model_coords(x, y):
    return round(x * MODEL_COORD_MAX / VIEWPORT[0]), round(y * MODEL_COORD_MAX / VIEWPORT[1])


def to_viewport_coords(x, y):
    return round(x * VIEWPORT[0] / MODEL_COORD_MAX), round(y * VIEWPORT[1] / MODEL_COORD_MAX)


def is_point(action):
    return isinstance(action.get("x"), (int, float)) and isinstance(action.get("y"), (int, float))


def to_model_action(action):
    """Pixel-space trajectory action -> model-space copy; scroll deltas and every non-coordinate key untouched."""
    converted = dict(action)
    if is_point(action):
        converted["x"], converted["y"] = to_model_coords(action["x"], action["y"])
    for key in _PATH_KEYS:
        if isinstance(action.get(key), list):
            converted[key] = [to_model_action(point) for point in action[key]]
    for key in _NESTED_ACTION_KEYS:
        if isinstance(action.get(key), dict):
            converted[key] = to_model_action(action[key])
    return converted
