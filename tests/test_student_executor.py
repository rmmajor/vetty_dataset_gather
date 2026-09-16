from cua.prompt.coords import to_viewport_coords
from cua.student.executor import (
    ACTION_TO_DRIVER,
    COORD_SPACES,
    action_coordinates,
    control_index,
    driver_payload,
    in_viewport,
    outside_coord_space,
    outside_viewport,
    scale_action,
    to_sequence_payload,
    validate_actions,
)
from cua.trajectory.actions import TERMINAL_TYPES

# Every action shape the grammar emits; order is load-bearing because the sequence test stops at the first terminal.
ACTION_SHAPES = [
    {"type": "click", "x": 640, "y": 360, "button": "left"},
    {"type": "click", "x": 640, "y": 360, "button": "right"},
    {"type": "click", "x": 12, "y": 34},
    {"type": "double_click", "x": 160, "y": 115},
    {"type": "move", "x": 300, "y": 300},
    {"type": "type", "text": "laptop stand"},
    {"type": "type", "text": "<secret:PASSWORD>"},
    {"type": "keypress", "keys": ["Enter"]},
    {"type": "keypress", "keys": ["Control", "a"]},
    {"type": "keypress", "keys": ["ArrowDown", "ArrowDown", "ArrowDown"]},
    {"type": "scroll", "x": 640, "y": 360, "scroll_x": 0, "scroll_y": 500},
    {"type": "scroll", "x": 640, "y": 360, "scroll_x": -120, "scroll_y": -240},
    {"type": "drag", "path": [{"x": 100, "y": 200}, {"x": 400, "y": 500}]},
    {"type": "drag", "path": [{"x": 100, "y": 200}, {"x": 250, "y": 350}, {"x": 400, "y": 500}]},
    {"type": "navigate", "url": "https://synth.test/app/board"},
    {"type": "wait"},
    {"type": "screenshot"},
    {"type": "confirm", "proposed_action": {"type": "click", "x": 640, "y": 400}, "risk_category": "FINANCIAL_RISK"},
    {"type": "confirm", "proposed_action": {"type": "submit"}, "risk_category": "DATA_LOSS"},
    {"type": "confirm"},
    {"type": "handoff", "reason": "login"},
    {"type": "handoff"},
    {"type": "approval_event"},
    {"type": "done"},
]
# Verbs the driver has no call for: the loop stops there, the screenshot is the observation, or a watchdog wrote it.
NO_DRIVER_CALL = ("handoff", "confirm", "screenshot", "approval_event")
DRIVER_KINDS = {"click_xy", "move_xy", "type_text", "press", "scroll_xy", "drag_path", "wait", "done", "navigate"}


def test_every_action_shape_scales_in_both_coordinate_spaces():
    for action in ACTION_SHAPES:
        pixels = scale_action(action, "viewport_abs")
        model = scale_action(action, "normalized_1000")
        assert pixels == action, action
        for key, value in action.items():
            if key not in ("x", "y", "path"):
                assert model[key] == value, (action, key)
        for (x, y), (mx, my) in zip(action_coordinates(action), action_coordinates(model), strict=True):
            assert (mx, my) == to_viewport_coords(x, y), action


def test_scaling_leaves_scroll_deltas_alone_because_they_are_not_positions():
    scrolled = scale_action({"type": "scroll", "x": 500, "y": 500, "scroll_x": 0, "scroll_y": 500}, "normalized_1000")
    assert scrolled == {"type": "scroll", "x": 640, "y": 360, "scroll_x": 0, "scroll_y": 500}


def test_coordinate_space_registry_covers_both_conventions():
    assert set(COORD_SPACES) == {"viewport_abs", "normalized_1000"}
    assert not outside_coord_space(1000, 1000, "normalized_1000") and outside_coord_space(1001, 0, "normalized_1000")
    assert not outside_coord_space(1279, 719, "viewport_abs") and outside_coord_space(1280, 0, "viewport_abs")


def test_the_two_lanes_agree_on_viewport_bounds_for_every_emitted_coordinate():
    """The RL lane asks `in_viewport` and the eval lane `outside_viewport`; on emitted integers they are inverses."""
    for action in ACTION_SHAPES:
        for x, y in action_coordinates(action):
            assert in_viewport(x, y) is not outside_viewport(x, y), action
    for x, y in ((0, 0), (1279, 719), (1280, 719), (1279, 720), (-1, 0)):
        assert in_viewport(x, y) is not outside_viewport(x, y)


def test_every_action_shape_maps_to_a_driver_call_or_is_deliberately_terminal():
    for action in ACTION_SHAPES:
        payload = driver_payload(scale_action(action, "normalized_1000"), "why")
        if action["type"] in NO_DRIVER_CALL:
            assert payload is None, action
            continue
        assert payload["kind"] in DRIVER_KINDS, action
        assert payload["thought"] == "why", action
    assert driver_payload({"type": "not_a_verb"}, "t") is None


def test_driver_payloads_are_the_coordinate_native_kinds():
    assert driver_payload({"type": "click", "x": 5, "y": 6, "button": "right"}, "t") == {
        "kind": "click_xy",
        "x": 5,
        "y": 6,
        "button": "right",
        "thought": "t",
    }
    assert driver_payload({"type": "double_click", "x": 5, "y": 6}, "t")["dblclick"] is True
    assert driver_payload({"type": "scroll", "x": 1, "y": 2, "scroll_x": 3, "scroll_y": 4}, "t") == {
        "kind": "scroll_xy",
        "x": 1,
        "y": 2,
        "dx": 3,
        "dy": 4,
        "thought": "t",
    }
    assert driver_payload({"type": "type", "text": "x"}, "")["thought"] == "(model emitted no <think> block)"
    assert {a["type"] for a in ACTION_SHAPES} - set(ACTION_TO_DRIVER) == {"approval_event"}


def test_a_sequence_payload_stops_at_the_first_terminal_action():
    payload = to_sequence_payload("go", ACTION_SHAPES)
    terminal = [a for a in payload["actions"] if a["type"] in TERMINAL_TYPES]
    assert payload["kind"] == "sequence" and payload["thought"] == "go"
    assert len(terminal) == 1 and payload["actions"][-1] is terminal[0]
    assert not any(a["type"] == "screenshot" for a in payload["actions"])
    assert control_index(payload["actions"]) == len(payload["actions"]) - 1


def test_terminal_types_are_the_registry_and_nothing_else():
    assert TERMINAL_TYPES == frozenset({"done", "handoff", "confirm"})
    assert all(ACTION_TO_DRIVER[name]({"type": name}) is None for name in ("handoff", "confirm"))
    assert control_index([{"type": "wait"}, {"type": "click", "x": 1, "y": 1}]) is None


def test_out_of_viewport_actions_are_reported_not_executed():
    assert validate_actions([a for a in ACTION_SHAPES if a["type"] == "click"]) == []
    assert validate_actions([{"type": "click", "x": 1280, "y": 1, "button": "left"}]) == ["out_of_viewport:click"]
    assert validate_actions([{"type": "drag", "path": [{"x": 0, "y": 0}, {"x": 5, "y": 900}]}]) == [
        "out_of_viewport:drag"
    ]
