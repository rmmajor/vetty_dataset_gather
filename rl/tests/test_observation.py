"""SFT-3 prompt format: coordinate space, effect annotations, two-image observation layout, golden fixture (`--regen` rewrites it)."""

import io
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

from rl.actions import decode_step, encode_step
from rl.observation import build_observation, to_openai_messages
from train_qwen import (
    COORD_SPACE,
    EFFECT_NO_CHANGE_MAX_FRACTION,
    EFFECT_PATH_MAX_CHARS,
    GROUNDING_SYSTEM_PROMPT,
    HISTORY_WINDOW,
    PREV_IMAGE_SIZE,
    PROMPT_FORMAT_VERSION,
    SYSTEM_PROMPT,
    VIEWPORT,
    build_step_messages,
    build_step_observation,
    build_step_user_text,
    convert_trajectory_to_examples,
    downscale_prev,
    effect_annotation,
    flatten_for_swift,
    history_entries_for_step,
    screenshot_changed_fraction,
    step_effects,
    to_model_action,
    to_model_coords,
    to_openai_content,
    to_viewport_coords,
    url_path_change,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "prompt_golden.json"

PIXEL_ACTIONS = {
    "click": {"type": "click", "x": 640, "y": 360, "button": "left"},
    "drag": {"type": "drag", "path": [{"x": 0, "y": 0}, {"x": 1280, "y": 720}]},
    "scroll": {"type": "scroll", "x": 320, "y": 180, "scroll_x": 0, "scroll_y": 500},
    "confirm": {
        "type": "confirm",
        "proposed_action": {"type": "click", "x": 1279, "y": 719},
        "risk_category": "purchase",
    },
    "type": {"type": "type", "text": "hello"},
    "keypress": {"type": "keypress", "keys": ["Enter"]},
    "done": {"type": "done"},
}
CLICK_STEP = {"actions": [PIXEL_ACTIONS["click"]]}
GROUP_STEP = {"actions": [PIXEL_ACTIONS["click"], PIXEL_ACTIONS["type"]]}
NO_CHANGE = {"changed": 0.0, "path": None}
PATH_CHANGE = {"changed": 0.4, "path": "/projects/new"}
BOTH = {"changed": 0.0, "path": "/projects/new"}
INSTRUCTION = "Move the apartment board to the kanban view"


def _png_bytes(size=(4, 4), color=(255, 255, 255), block=None):
    image = Image.new("RGB", size, color)
    if block:
        image.paste((0, 0, 0), block)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _write_png(path, **kwargs):
    Path(path).write_bytes(_png_bytes(**kwargs))


def _observation_case(history, prev):
    obs = build_step_observation(INSTRUCTION, history, "<CUR>", "<PREV>" if prev else None)
    return {"text": obs["text"], "content": obs["content"], "images": obs["images"]}


def golden_cases():
    """Every golden case rendered by the live code; `--regen` stores it, the test diffs against the store."""
    long_history = [f"click({i}, {i})" for i in range(HISTORY_WINDOW + 3)]
    annotated = (
        history_entries_for_step(CLICK_STEP, NO_CHANGE)
        + history_entries_for_step(GROUP_STEP, PATH_CHANGE)
        + history_entries_for_step(CLICK_STEP, BOTH)
    )
    return {
        "constants": {
            "PROMPT_FORMAT_VERSION": PROMPT_FORMAT_VERSION,
            "COORD_SPACE": COORD_SPACE,
            "HISTORY_WINDOW": HISTORY_WINDOW,
            "SYSTEM_PROMPT": SYSTEM_PROMPT,
            "GROUNDING_SYSTEM_PROMPT": GROUNDING_SYSTEM_PROMPT,
        },
        "coords": {name: to_model_action(action) for name, action in PIXEL_ACTIONS.items()},
        "history": {
            "no_change": history_entries_for_step(CLICK_STEP, NO_CHANGE),
            "path": history_entries_for_step(CLICK_STEP, PATH_CHANGE),
            "both": history_entries_for_step(CLICK_STEP, BOTH),
            "group_last_entry": history_entries_for_step(GROUP_STEP, BOTH),
        },
        "observations": {
            "no_history_step1": _observation_case([], prev=False),
            "windowed_23_stepn": _observation_case(long_history, prev=True),
            "annotated_stepn": _observation_case(annotated, prev=True),
        },
    }


def test_prompt_golden_matches_fixture():
    expected = json.loads(FIXTURE_PATH.read_text())
    actual = json.loads(json.dumps(golden_cases()))
    for section in expected:
        for name in expected[section]:
            assert actual[section][name] == expected[section][name], (section, name, actual[section][name])
    assert actual == expected, "golden sections drifted; review and `python -m rl.tests.test_observation --regen`"


def test_constants():
    assert PROMPT_FORMAT_VERSION == "sft3" and COORD_SPACE == "normalized_1000" and VIEWPORT == (1280, 720)
    assert HISTORY_WINDOW == 20 and PREV_IMAGE_SIZE == (640, 360) and EFFECT_PATH_MAX_CHARS == 40
    assert EFFECT_NO_CHANGE_MAX_FRACTION == 0.0005


def test_system_prompt_describes_sft3_inputs():
    assert SYSTEM_PROMPT.startswith(
        "You are a GUI agent that operates a web browser at a 1280x720 resolution. Given a task, the current screenshot"
    )
    assert "[no change]" in SYSTEM_PROMPT and "-> /path" in SYSTEM_PROMPT
    assert (
        "Available actions (x, y are normalized to 0-1000 across the screenshot width and height; scrollX/scrollY are pixel deltas):"
        in SYSTEM_PROMPT
    )
    assert "action: confirm(propose=click(<ref>(x, y)</ref>), risk=purchase)" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.endswith("and wait for approval instead of doing it.")
    assert (
        GROUNDING_SYSTEM_PROMPT.startswith("You are a GUI agent")
        and "No reasoning, no other text." in GROUNDING_SYSTEM_PROMPT
    )


def test_coords_corners_and_inverse():
    assert to_model_coords(0, 0) == (0, 0) and to_model_coords(1280, 720) == (1000, 1000)
    assert to_model_coords(640, 360) == (500, 500)
    assert to_viewport_coords(500, 500) == (640, 360) and to_viewport_coords(1000, 1000) == (1280, 720)
    assert to_viewport_coords(*to_model_coords(1279, 719)) == (1279, 719)


def test_to_model_action_converts_points_and_keeps_deltas():
    click = to_model_action(PIXEL_ACTIONS["click"])
    assert click == {"type": "click", "x": 500, "y": 500, "button": "left"} and PIXEL_ACTIONS["click"]["x"] == 640
    assert to_model_action(PIXEL_ACTIONS["drag"])["path"] == [{"x": 0, "y": 0}, {"x": 1000, "y": 1000}]
    assert to_model_action(PIXEL_ACTIONS["scroll"]) == {
        "type": "scroll",
        "x": 250,
        "y": 250,
        "scroll_x": 0,
        "scroll_y": 500,
    }
    confirm = to_model_action(PIXEL_ACTIONS["confirm"])
    assert (
        confirm["proposed_action"] == {"type": "click", "x": 999, "y": 999} and confirm["risk_category"] == "purchase"
    )
    typed = to_model_action(PIXEL_ACTIONS["type"])
    assert typed == PIXEL_ACTIONS["type"] and typed is not PIXEL_ACTIONS["type"]
    assert to_model_action({"type": "confirm", "proposed_action": {}, "risk_category": ""})["proposed_action"] == {}


def test_decode_step_round_trips_model_space_actions():
    for action in PIXEL_ACTIONS.values():
        model_action = to_model_action(action)
        decoded = decode_step(encode_step("thinking", [model_action]))
        assert decoded.actions == [model_action] and decoded.canonical and not decoded.errors, (action, decoded)


def test_effect_annotation():
    assert effect_annotation(None) == "" and effect_annotation({"changed": None, "path": None}) == ""
    assert effect_annotation({"changed": 0.0, "path": None}) == " [no change]"
    assert effect_annotation({"changed": EFFECT_NO_CHANGE_MAX_FRACTION, "path": None}) == " [no change]"
    assert effect_annotation({"changed": 0.2, "path": "/x"}) == " -> /x"
    assert effect_annotation({"changed": 0.0, "path": "/x"}) == " [no change] -> /x"


def test_url_path_change():
    assert url_path_change("https://a.io/login", "https://a.io/login") is None
    assert url_path_change("https://a.io/login", "https://a.io/login/") is None
    assert url_path_change("https://a.io/login?x=1", "https://a.io/login?x=2#h") is None
    assert url_path_change("https://a.io/register", "https://a.io/login") == "/login"
    assert url_path_change("https://a.io/login", "https://a.io/") == "/"
    assert url_path_change(None, "https://a.io/login") is None and url_path_change("https://a.io/a", "") is None
    long_path = "/" + "p" * 60
    assert url_path_change("https://a.io/", "https://a.io" + long_path) == long_path[:EFFECT_PATH_MAX_CHARS]


def test_screenshot_changed_fraction():
    assert screenshot_changed_fraction(_png_bytes(), _png_bytes()) == 0.0
    assert screenshot_changed_fraction(_png_bytes(size=(8, 8)), _png_bytes(size=(16, 16))) == 0.0
    changed = screenshot_changed_fraction(
        _png_bytes(size=(100, 100)), _png_bytes(size=(100, 100), block=(0, 0, 20, 10))
    )
    assert 0.0005 < changed < 0.05, changed
    assert screenshot_changed_fraction(_png_bytes(color=(255, 255, 255)), _png_bytes(color=(0, 0, 0))) == 1.0


def test_step_effects_synthetic_trajectory():
    steps = [
        {"step": 1, "screenshot": "step_001_screenshot.png"},
        {"step": 2, "screenshot": "step_002_screenshot.png"},
        {"step": 3, "screenshot": "step_003_screenshot.png"},
        {"step": 4, "screenshot": "final_screenshot.png"},
    ]
    meta = {
        "steps": {
            "1": {"url": "https://a.io/login"},
            "2": {"url": "https://a.io/login"},
            "3": {"url": "https://a.io/projects"},
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_png(d / "step_001_screenshot.png")
        _write_png(d / "step_002_screenshot.png")
        _write_png(d / "step_003_screenshot.png", color=(0, 0, 0))
        _write_png(d / "final_screenshot.png", color=(0, 0, 0))
        effects = step_effects(d, steps, meta)
        assert effects == [
            {"changed": 0.0, "path": None},
            {"changed": 1.0, "path": "/projects"},
            {"changed": 0.0, "path": None},
            None,
        ]
        assert [e and e["path"] for e in step_effects(d, steps, {})] == [None, None, None, None]
        assert [e and e["path"] for e in step_effects(d, steps, {"steps": []})] == [None, None, None, None]
        (d / "step_002_screenshot.png").unlink()
        assert step_effects(d, steps, meta)[:2] == [None, None]


def test_history_entries_convert_and_annotate_last():
    assert history_entries_for_step(CLICK_STEP) == ["click(500, 500)"]
    assert history_entries_for_step(GROUP_STEP, BOTH) == [
        "click(500, 500)",
        'type("hello") [no change] -> /projects/new',
    ]
    assert history_entries_for_step({"actions": []}, BOTH) == []


def test_history_entries_include_watchdog_event():
    step = {"actions": [{"type": "click", "x": 1, "y": 2}], "event": {"kind": "approval", "decision": "approve"}}
    assert history_entries_for_step(step) == ["click(1, 3)", "[watchdog: approve]"]


def test_history_entries_legacy_single_action_key():
    assert history_entries_for_step({"action": {"type": "wait"}}) == ["wait"]


def test_user_text_without_history():
    assert build_step_user_text("Do it", []) == "Task: Do it\nWhat action should you take?"


def test_user_text_windowed_with_omitted_prefix():
    history = [f"click({i}, {i})" for i in range(HISTORY_WINDOW + 3)]
    text = build_step_user_text("Do it", history)
    assert "... 3 earlier actions, " in text
    assert "click(0, 0)" not in text and f"click({HISTORY_WINDOW + 2}, {HISTORY_WINDOW + 2})" in text
    assert text.endswith("]\nWhat action should you take?")


def test_observation_layouts():
    user_text = build_step_user_text(INSTRUCTION, ["click(1, 1)"])
    first = build_step_observation(INSTRUCTION, ["click(1, 1)"], "/cur.png")
    assert first["content"] == [
        {"type": "text", "text": "Current screenshot:\n"},
        {"type": "image", "image": "/cur.png"},
        {"type": "text", "text": "\n" + user_text},
    ]
    assert first["images"] == ["/cur.png"] and first["text"] == "Current screenshot:\n<image>\n" + user_text
    later = build_step_observation(INSTRUCTION, ["click(1, 1)"], "/cur.png", "/prev.png")
    assert later["content"] == [
        {"type": "text", "text": "Previous screenshot (before your last action, downscaled):\n"},
        {"type": "image", "image": "/prev.png"},
        {"type": "text", "text": "\nCurrent screenshot:\n"},
        {"type": "image", "image": "/cur.png"},
        {"type": "text", "text": "\n" + user_text},
    ]
    assert later["images"] == ["/prev.png", "/cur.png"]
    assert later["text"] == flatten_for_swift(later["content"]) and later["text"].count("<image>") == 2


def test_build_step_messages_and_openai_content():
    obs = build_step_observation("x", [], "/cur.png", "/prev.png")
    msgs = build_step_messages("SYS", obs)
    assert msgs == [{"role": "system", "content": "SYS"}, {"role": "user", "content": obs["content"]}]
    rendered = to_openai_content(obs["content"], lambda ref: "B64:" + ref)
    assert rendered[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,B64:/prev.png"}}
    assert rendered[3] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,B64:/cur.png"}}
    assert [item["type"] for item in rendered] == ["text", "image_url", "text", "image_url", "text"]
    assert rendered[4] == {"type": "text", "text": "\nTask: x\nWhat action should you take?"}


def test_downscale_prev_size():
    small = downscale_prev(_png_bytes(size=(1280, 720)))
    assert Image.open(io.BytesIO(small)).size == PREV_IMAGE_SIZE
    assert Image.open(io.BytesIO(downscale_prev(_png_bytes(size=(1280, 720)), (64, 36)))).size == (64, 36)


def _synthetic_trajectory(d):
    steps = [
        {
            "step": 1,
            "screenshot": "step_001_screenshot.png",
            "thought": "a",
            "actions": [{"type": "click", "x": 640, "y": 360, "button": "left"}],
        },
        {
            "step": 2,
            "screenshot": "step_002_screenshot.png",
            "thought": "b",
            "actions": [{"type": "type", "text": "hi"}, {"type": "keypress", "keys": ["Enter"]}],
        },
        {"step": 3, "screenshot": "step_003_screenshot.png", "thought": "c", "actions": [{"type": "done"}]},
    ]
    _write_png(d / "step_001_screenshot.png", size=(64, 36))
    _write_png(d / "step_002_screenshot.png", size=(64, 36))
    _write_png(d / "step_003_screenshot.png", size=(64, 36), color=(0, 0, 0))
    meta = {
        "steps": {
            "1": {"url": "https://a.io/login"},
            "2": {"url": "https://a.io/login"},
            "3": {"url": "https://a.io/home"},
        }
    }
    return {"instruction": "Say hi", "steps": steps}, meta


def test_convert_two_image_layout_writes_prev_and_model_coords():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "task_x"
        d.mkdir()
        trajectory, meta = _synthetic_trajectory(d)
        prev_dir = Path(tmp) / "prev" / "group"
        examples, stats = convert_trajectory_to_examples(trajectory, d, capture_meta=meta, prev_dir=prev_dir)
        assert len(examples) == 3
        assert [sum(1 for c in e["messages"][1]["content"] if c["type"] == "image") for e in examples] == [1, 2, 2]
        assert (
            examples[0]["messages"][2]["content"]
            == "<think>a</think>\naction: click(<ref>(500, 500)</ref>, button=left)"
        )
        prev_ref = examples[1]["messages"][1]["content"][1]["image"]
        assert prev_ref == str((prev_dir / "task_x" / "step_001_prev.png").resolve())
        assert Image.open(prev_ref).size == PREV_IMAGE_SIZE and (prev_dir / "task_x" / "step_002_prev.png").is_file()
        assert examples[1]["messages"][1]["content"][3]["image"] == str((d / "step_002_screenshot.png").resolve())
        assert (
            examples[1]["messages"][1]["content"][-1]["text"]
            == "\nTask: Say hi\nPrevious actions: [click(500, 500) [no change]]\nWhat action should you take?"
        )
        assert examples[2]["messages"][1]["content"][-1]["text"].endswith(
            "keypress(['Enter']) -> /home]\nWhat action should you take?"
        )
        assert stats["effect_no_change"] == 1 and stats["effect_url_change"] == 1 and stats["effect_steps"] == 2


def test_convert_default_is_one_image_without_prev_dir():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "task_y"
        d.mkdir()
        trajectory, _ = _synthetic_trajectory(d)
        examples, _ = convert_trajectory_to_examples(trajectory, d)
        assert [sum(1 for c in e["messages"][1]["content"] if c["type"] == "image") for e in examples] == [1, 1, 1]
        assert not (Path(tmp) / "prev").exists()
        assert (
            "[no change]" in examples[1]["messages"][1]["content"][-1]["text"]
            and "->" not in examples[2]["messages"][1]["content"][-1]["text"]
        )


def test_convert_history_matches_shared_builders():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "task_z"
        d.mkdir()
        trajectory, meta = _synthetic_trajectory(d)
        examples, _ = convert_trajectory_to_examples(trajectory, d, capture_meta=meta, prev_dir=Path(tmp) / "prev")
        effects = step_effects(d, trajectory["steps"], meta)
    history = []
    for i, (example, step) in enumerate(zip(examples, trajectory["steps"], strict=False)):
        content = example["messages"][1]["content"]
        assert content[-1]["text"] == "\n" + build_step_user_text("Say hi", history)
        assert flatten_for_swift(content).count("<image>") == len(
            build_step_observation("Say hi", history, "c", "p" if i else None)["images"]
        )
        history.extend(history_entries_for_step(step, effects[i]))
    assert json.dumps(examples[0]["messages"][0]["content"]).startswith('"You are a GUI agent')


def test_worst_case_prompt_stays_under_token_budget():
    path = "/" + "p" * (EFFECT_PATH_MAX_CHARS - 1)
    entry = history_entries_for_step({"actions": [PIXEL_ACTIONS["drag"]]}, {"changed": 0.0, "path": path})[0]
    obs = build_step_observation("x" * 300, [entry] * (HISTORY_WINDOW + 5), "<CUR>", "<PREV>")
    chars = len(SYSTEM_PROMPT) + len(obs["text"].replace("<image>", ""))
    assert chars / 3 + 880 + 220 + 6 < 3000, chars


def test_build_observation_matches_sft_user_turn():
    obs = build_observation("Say hi", ["click(1, 2)"], "QUJD", 1)
    assert obs == {
        "system": SYSTEM_PROMPT,
        "text": build_step_user_text("Say hi", ["click(1, 2)"]),
        "image_b64": "QUJD",
        "step_index": 1,
    }


def test_openai_messages_carry_data_uri_then_text():
    msgs = to_openai_messages(build_observation("Say hi", [], "QUJD", 0))
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1]["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    assert msgs[1]["content"][1] == {"type": "text", "text": "Task: Say hi\nWhat action should you take?"}


if __name__ == "__main__":
    if "--regen" in sys.argv:
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_PATH.write_text(json.dumps(golden_cases(), indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {FIXTURE_PATH}")
    else:
        print("usage: python -m rl.tests.test_observation --regen")
