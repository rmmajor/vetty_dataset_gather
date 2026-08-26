import json
import tempfile
from pathlib import Path

from PIL import Image

from rl.observation import build_observation, to_openai_messages
from train_qwen import (HISTORY_WINDOW, SYSTEM_PROMPT, build_step_messages, build_step_user_text,
                        convert_trajectory_to_examples, history_entries_for_step)


def _png(path):
    Image.new("RGB", (4, 4), (255, 255, 255)).save(path)


def test_user_text_without_history():
    assert build_step_user_text("Do it", []) == "Task: Do it\nWhat action should you take?"


def test_user_text_windowed_with_omitted_prefix():
    history = [f"click({i}, {i})" for i in range(HISTORY_WINDOW + 3)]
    text = build_step_user_text("Do it", history)
    assert "... 3 earlier actions, " in text
    assert "click(0, 0)" not in text and f"click({HISTORY_WINDOW + 2}, {HISTORY_WINDOW + 2})" in text
    assert text.endswith("]\nWhat action should you take?")


def test_history_entries_include_watchdog_event():
    step = {"actions": [{"type": "click", "x": 1, "y": 2}], "event": {"kind": "approval", "decision": "approve"}}
    assert history_entries_for_step(step) == ["click(1, 2)", "[watchdog: approve]"]


def test_history_entries_legacy_single_action_key():
    assert history_entries_for_step({"action": {"type": "wait"}}) == ["wait"]


def test_build_step_messages_shape():
    msgs = build_step_messages("SYS", "Task: x", "/tmp/a.png")
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1]["content"] == [{"type": "image", "image": "/tmp/a.png"}, {"type": "text", "text": "Task: x"}]


def test_convert_uses_shared_builders():
    steps = [
        {"step": 1, "screenshot": "step_001_screenshot.png", "thought": "a", "actions": [{"type": "click", "x": 1, "y": 2, "button": "left"}]},
        {"step": 2, "screenshot": "step_002_screenshot.png", "thought": "b", "actions": [{"type": "type", "text": "hi"}, {"type": "keypress", "keys": ["Enter"]}]},
        {"step": 3, "screenshot": "step_003_screenshot.png", "thought": "c", "actions": [{"type": "done"}]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for s in steps:
            _png(d / s["screenshot"])
        examples, _ = convert_trajectory_to_examples({"instruction": "Say hi", "steps": steps}, d)
    history = []
    for example, step in zip(examples, steps):
        assert example["messages"][1]["content"][1]["text"] == build_step_user_text("Say hi", history)
        history.extend(history_entries_for_step(step))
    assert json.dumps(examples[0]["messages"][0]["content"]).startswith('"You are a GUI agent')


def test_build_observation_matches_sft_user_turn():
    obs = build_observation("Say hi", ["click(1, 2)"], "QUJD", 1)
    assert obs == {"system": SYSTEM_PROMPT, "text": build_step_user_text("Say hi", ["click(1, 2)"]), "image_b64": "QUJD", "step_index": 1}


def test_openai_messages_carry_data_uri_then_text():
    msgs = to_openai_messages(build_observation("Say hi", [], "QUJD", 0))
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1]["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    assert msgs[1]["content"][1] == {"type": "text", "text": "Task: Say hi\nWhat action should you take?"}
