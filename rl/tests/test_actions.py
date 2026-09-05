from rl.actions import ACTION_DECODERS, decode_step, encode_step

FIXTURES = [
    {"type": "click", "x": 10, "y": 20, "button": "left"},
    {"type": "click", "x": 1279, "y": 719, "button": "right"},
    {"type": "double_click", "x": 5, "y": 6},
    {"type": "move", "x": 640, "y": 360},
    {"type": "type", "text": "hello world"},
    {"type": "type", "text": 'say "hi" (now)'},
    {"type": "type", "text": "ünïcødé ✓"},
    {"type": "type", "text": ""},
    {"type": "keypress", "keys": ["Enter"]},
    {"type": "keypress", "keys": ["Control", "a"]},
    {"type": "scroll", "x": 640, "y": 360, "scroll_x": 0, "scroll_y": -500},
    {"type": "drag", "path": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]},
    {"type": "wait"},
    {"type": "done"},
    {"type": "screenshot"},
    {"type": "navigate", "url": "https://example.com/a?b=c&d=(e)"},
    {"type": "handoff", "reason": "login"},
    {"type": "confirm", "proposed_action": {"type": "click", "x": 880, "y": 540}, "risk_category": "purchase"},
    {"type": "confirm", "proposed_action": {"type": "type"}, "risk_category": "send"},
    {"type": "confirm", "proposed_action": {}, "risk_category": ""},
]


def test_round_trip_every_action_shape():
    for action in FIXTURES:
        decoded = decode_step(encode_step("thinking", [action]))
        assert decoded.actions == [action], (action, decoded)
        assert decoded.canonical and decoded.ok and not decoded.errors, (action, decoded)
        assert decoded.thought == "thinking"


def test_every_decoder_is_covered_by_a_fixture():
    assert set(ACTION_DECODERS) == {a["type"] for a in FIXTURES}


def test_multi_action_step_keeps_order():
    actions = [FIXTURES[0], FIXTURES[4], FIXTURES[8]]
    decoded = decode_step(encode_step("go", actions))
    assert decoded.actions == actions and decoded.canonical


def test_multiline_type_text_round_trips():
    action = {"type": "type", "text": "line one\nline two"}
    decoded = decode_step(encode_step("t", [action, {"type": "done"}]))
    assert decoded.actions == [action, {"type": "done"}]


def test_drag_keeps_endpoints_only():
    action = {"type": "drag", "path": [{"x": 1, "y": 1}, {"x": 5, "y": 5}, {"x": 9, "y": 9}]}
    assert decode_step(encode_step("", [action])).actions == [{"type": "drag", "path": [{"x": 1, "y": 1}, {"x": 9, "y": 9}]}]


def test_missing_think_is_allowed():
    decoded = decode_step("action: done()")
    assert decoded.thought == "" and decoded.actions == [{"type": "done"}] and decoded.ok


def test_python_repr_keypress_decodes_but_is_not_canonical():
    decoded = decode_step("action: keypress(['Enter'])")
    assert decoded.actions == [{"type": "keypress", "keys": ["Enter"]}] and not decoded.canonical


def test_click_without_button_defaults_left_not_canonical():
    decoded = decode_step("action: click(<ref>(1, 2)</ref>)")
    assert decoded.actions == [{"type": "click", "x": 1, "y": 2, "button": "left"}] and not decoded.canonical


def test_stray_text_and_unknown_action_are_errors():
    decoded = decode_step("<think>x</think>\nI will click now\naction: fly()")
    assert decoded.actions == [] and not decoded.ok
    assert any(e.startswith("stray_text") for e in decoded.errors)
    assert "unknown_action:fly" in decoded.errors


def test_no_actions_is_an_error():
    assert decode_step("<think>hmm</think>").errors == ["no_actions"]


def test_history_summary_grammar_is_rejected():
    assert "bad_args:click" in decode_step("action: click(10, 20)").errors


def test_unclosed_think_is_malformed():
    decoded = decode_step("<think>never closed\naction: done()")
    assert "think_malformed" in decoded.errors and decoded.actions == [{"type": "done"}]


def test_trailing_prose_after_actions_is_stray():
    decoded = decode_step("<think>t</think>\naction: done()\nThat completes the task.")
    assert decoded.actions == [{"type": "done"}] and any(e.startswith("stray_text") for e in decoded.errors)


def test_prefilled_think_keeps_the_thought():
    decoded = decode_step('I will click the field.\n</think>\n\naction: click(<ref>(10, 20)</ref>, button=left)')
    assert decoded.thought == "I will click the field."
    assert decoded.actions == [{"type": "click", "x": 10, "y": 20, "button": "left"}]
    assert not decoded.errors and decoded.ok
