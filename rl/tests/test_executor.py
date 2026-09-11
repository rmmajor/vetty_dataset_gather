from rl.executor import (
    MISSING_THOUGHT_PLACEHOLDER,
    control_index,
    normalize_keys,
    to_sequence_payload,
    validate_actions,
)


def test_normalize_keys_maps_aliases_and_keeps_playwright_names():
    assert normalize_keys(["ctrl", "a"]) == ["Control", "a"]
    assert normalize_keys(["Enter", "ArrowDown", "return"]) == ["Enter", "ArrowDown", "Enter"]


def test_validate_actions_flags_out_of_viewport():
    assert validate_actions([{"type": "click", "x": 1280, "y": 10, "button": "left"}]) == ["out_of_viewport:click"]
    assert validate_actions([{"type": "drag", "path": [{"x": 0, "y": 0}, {"x": 5, "y": 900}]}]) == [
        "out_of_viewport:drag"
    ]
    assert (
        validate_actions(
            [{"type": "type", "text": "x"}, {"type": "scroll", "x": 1, "y": 1, "scroll_x": 0, "scroll_y": 5}]
        )
        == []
    )


def test_control_index():
    assert control_index([{"type": "click", "x": 1, "y": 1, "button": "left"}, {"type": "done"}, {"type": "wait"}]) == 1
    assert control_index([{"type": "wait"}]) is None


def test_payload_cuts_after_control_and_drops_screenshot():
    payload = to_sequence_payload(
        "",
        [
            {"type": "screenshot"},
            {"type": "keypress", "keys": ["esc"]},
            {"type": "handoff", "reason": "login"},
            {"type": "wait"},
        ],
    )
    assert payload["kind"] == "sequence" and payload["thought"] == MISSING_THOUGHT_PLACEHOLDER
    assert payload["actions"] == [{"type": "keypress", "keys": ["Escape"]}, {"type": "handoff", "reason": "login"}]


def test_payload_keeps_thought():
    assert to_sequence_payload("  go  ", [{"type": "wait"}])["thought"] == "go"
