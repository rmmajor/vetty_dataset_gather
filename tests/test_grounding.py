"""Grounding slice of `python -m cua.sft convert`: element-description to click examples from recorded steps."""

import io
import itertools
import json
import random
import tempfile
from pathlib import Path

from PIL import Image

from cua.prompt.format import GROUNDING_SYSTEM_PROMPT, build_grounding_observation, build_step_observation
from cua.sft.convert import convert_trajectories
from cua.sft.grounding import (
    GROUNDING_PER_GROUP_CAP,
    GROUNDING_ROLE_NOUNS,
    GROUNDING_SMALL_OVERSAMPLE,
    GROUNDING_TEMPLATES,
    grounding_description,
    grounding_example,
    grounding_targets,
    select_grounding_examples,
)

SAVE_BBOX = {"x": 600, "y": 340, "width": 80, "height": 40}
ICON_BBOX = {"x": 1000, "y": 20, "width": 24, "height": 24}


def _png_bytes(size=(64, 36), color=(255, 255, 255)):
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="PNG")
    return out.getvalue()


def _centre(bbox):
    return round(bbox["x"] + bbox["width"] / 2), round(bbox["y"] + bbox["height"] / 2)


def _click_step(number, x, y, action_type="click"):
    action = {"type": action_type, "x": x, "y": y, "button": "left"}
    return {"step": number, "screenshot": f"step_{number:03d}_screenshot.png", "thought": "t", "actions": [action]}


_DIR_IDS = itertools.count()


def _trajectory_dir(tmp, name=None, step_count=1):
    d = Path(tmp) / (name or f"task_{next(_DIR_IDS)}")
    d.mkdir(parents=True)
    for number in range(1, step_count + 1):
        (d / f"step_{number:03d}_screenshot.png").write_bytes(_png_bytes())
    return d


def _targets_for(tmp, role, name, bbox, click=None, meta_extra=None):
    d = _trajectory_dir(tmp)
    x, y = click or _centre(bbox)
    trajectory = {"instruction": "i", "steps": [_click_step(1, x, y)]}
    meta = {"steps": {"1": {"role": role, "name": name, "bbox": bbox, **(meta_extra or {})}}}
    targets, _ = grounding_targets(trajectory, d, meta)
    return targets


def _target(name="Save", noun="button", x=640, y=360, small=False, screenshot="/s.png"):
    return {"screenshot": screenshot, "name": name, "noun": noun, "role": "button", "x": x, "y": y, "small": small}


def test_grounding_observation_is_a_one_image_first_step_layout():
    obs = build_grounding_observation('the "Save" button', "/cur.png")
    assert obs["content"] == [
        {"type": "text", "text": "Current screenshot:\n"},
        {"type": "image", "image": "/cur.png"},
        {"type": "text", "text": '\nTask: Click the "Save" button.\nWhat action should you take?'},
    ]
    assert obs["images"] == ["/cur.png"] and obs["text"].count("<image>") == 1
    assert obs == build_step_observation('Click the "Save" button.', [], "/cur.png")


def test_role_registry_maps_and_skips():
    expected = {
        "button": "button",
        "a": "link",
        "link": "link",
        "input": "text field",
        "textbox": "text field",
        "searchbox": "text field",
        "textarea": "text area",
        "select": "dropdown",
        "combobox": "dropdown",
        "option": "option",
        "menuitem": "menu item",
        "tab": "tab",
        "checkbox": "checkbox",
        "radio": "radio button",
        "spinbutton": "number field",
        "treeitem": "item",
        "listitem": "item",
        "li": "item",
        "gridcell": "item",
        "summary": "item",
        "switch": "toggle",
        "img": "image",
    }
    assert GROUNDING_ROLE_NOUNS == expected
    with tempfile.TemporaryDirectory() as tmp:
        for role in ("div", "span", "main", "presentation", "list", "code", "pre", None):
            assert _targets_for(tmp, role, "Save", SAVE_BBOX) == [], role
        (target,) = _targets_for(tmp, "a", "Docs", SAVE_BBOX)
        assert (target["noun"], target["role"], target["name"]) == ("link", "a", "Docs")
        assert (target["x"], target["y"], target["small"]) == (640, 360, False)


def test_only_leading_click_actions_with_centre_hits_qualify():
    cx, cy = _centre(SAVE_BBOX)
    with tempfile.TemporaryDirectory() as tmp:
        assert len(_targets_for(tmp, "button", "Save", SAVE_BBOX, click=(cx + 2, cy - 2))) == 1
        assert _targets_for(tmp, "button", "Save2", SAVE_BBOX, click=(cx + 3, cy)) == []
        assert _targets_for(tmp, "button", "Save3", SAVE_BBOX, click=(cx, cy - 3)) == []
        d = _trajectory_dir(tmp, step_count=2)
        scroll = {
            "step": 1,
            "screenshot": "step_001_screenshot.png",
            "actions": [{"type": "scroll", "x": cx, "y": cy, "scroll_x": 0, "scroll_y": 500}],
        }
        trajectory = {"instruction": "i", "steps": [scroll, _click_step(2, cx, cy)]}
        meta = {
            "steps": {
                "1": {"role": "button", "name": "Save", "bbox": SAVE_BBOX},
                "2": {"role": "button", "name": "Save", "bbox": SAVE_BBOX},
            }
        }
        targets, skipped = grounding_targets(trajectory, d, meta)
        assert [t["name"] for t in targets] == ["Save"] and skipped["not_click"] == 1


def test_name_secret_password_and_geometry_filters():
    with tempfile.TemporaryDirectory() as tmp:
        assert _targets_for(tmp, "input", "", SAVE_BBOX) == []
        assert _targets_for(tmp, "input", "x" * 61, SAVE_BBOX) == []
        assert len(_targets_for(tmp, "input", "x" * 60, SAVE_BBOX)) == 1
        assert _targets_for(tmp, "input", "<secret:PASSWORD>", SAVE_BBOX) == []
        assert _targets_for(tmp, "input", "key AKIAABCDEFGHIJKLMNOP", SAVE_BBOX) == []
        for masked in ("••••••••", "********", "Password", "Enter your password", "Confirm password"):
            assert _targets_for(tmp, "input", masked, SAVE_BBOX) == [], masked
        assert _targets_for(tmp, "input", "Email", SAVE_BBOX, meta_extra={"attrs": {"type": "password"}}) == []
        assert len(_targets_for(tmp, "input", "Email", SAVE_BBOX, meta_extra={"attrs": {"type": "email"}})) == 1
        assert _targets_for(tmp, "button", "Off", {"x": 1250, "y": 10, "width": 40, "height": 20}) == []
        assert _targets_for(tmp, "button", "Neg", {"x": -5, "y": 10, "width": 40, "height": 20}) == []
        assert _targets_for(tmp, "button", "Flat", {"x": 5, "y": 10, "width": 0, "height": 20}) == []
        assert _targets_for(tmp, "button", "NoBox", None, click=(640, 360)) == []
        d = _trajectory_dir(tmp, step_count=0)
        targets, skipped = grounding_targets(
            {"instruction": "i", "steps": [_click_step(1, 640, 360)]},
            d,
            {"steps": {"1": {"role": "button", "name": "Save", "bbox": SAVE_BBOX}}},
        )
        assert targets == [] and skipped["no_screenshot"] == 1


def test_descriptions_rotate_templates_deterministically():
    assert GROUNDING_TEMPLATES == ('the "{name}" {noun}', 'the {noun} labelled "{name}"', 'the {noun} "{name}"')
    assert grounding_description("Save", "button", 0) == 'the "Save" button'
    assert grounding_description("Save", "button", 1) == 'the button labelled "Save"'
    assert grounding_description("Save", "button", 2) == 'the button "Save"'
    assert grounding_description("Save", "button", 3) == grounding_description("Save", "button", 0)


def test_grounding_example_has_no_think_and_model_space_coords():
    example = grounding_example(_target(x=1279, y=719), 0)
    assert list(example) == ["messages"]
    system, user, assistant = example["messages"]
    assert system == {"role": "system", "content": GROUNDING_SYSTEM_PROMPT}
    assert user["content"] == build_grounding_observation('the "Save" button', "/s.png")["content"]
    assert assistant == {"role": "assistant", "content": "action: click(<ref>(999, 999)</ref>, button=left)"}
    assert "<think>" not in assistant["content"]
    assert (
        grounding_example(_target(x=0, y=0), 1)["messages"][2]["content"]
        == "action: click(<ref>(0, 0)</ref>, button=left)"
    )


def _user_text(example):
    return example["messages"][1]["content"][-1]["text"]


def test_small_targets_oversample_with_distinct_phrasings():
    by_group, stats = select_grounding_examples({"site_a": [_target(small=True)]}, budget=100, rng=random.Random(0))
    texts = [_user_text(e) for e in by_group["site_a"]]
    assert len(texts) == GROUNDING_SMALL_OVERSAMPLE == 3 and len(set(texts)) == 3
    assert stats["grounding_examples"] == 3 and stats["grounding_small_target_share"] == 1.0
    by_group, _ = select_grounding_examples({"site_a": [_target(small=False)]}, budget=100, rng=random.Random(0))
    assert len(by_group["site_a"]) == 1


def test_dedupe_keeps_at_most_two_per_name_and_cell():
    same = [_target(x=640 + i, y=360, screenshot=f"/s{i}.png") for i in range(5)]
    by_group, stats = select_grounding_examples({"site_a": same}, budget=100, rng=random.Random(0))
    assert len(by_group["site_a"]) == 2 and stats["grounding_deduped"] == 3
    far = [_target(x=640, y=360), _target(x=640, y=460), _target(name="Cancel", x=640, y=360)]
    by_group, _ = select_grounding_examples({"site_a": far}, budget=100, rng=random.Random(0))
    assert len(by_group["site_a"]) == 3


def test_per_group_cap_and_share_ceiling_drop_small_targets_last():
    large = [_target(name=f"L{i}", x=10 + i * 20, y=100) for i in range(60)]
    by_group, stats = select_grounding_examples({"site_a": large}, budget=1000, rng=random.Random(0))
    assert len(by_group["site_a"]) == GROUNDING_PER_GROUP_CAP == 40 and stats["grounding_capped"] == 20
    mixed = {
        "site_a": [_target(name=f"L{i}", x=10 + i * 20, y=100) for i in range(30)],
        "site_b": [_target(name=f"S{i}", x=10 + i * 20, y=300, small=True) for i in range(4)],
    }
    by_group, stats = select_grounding_examples(mixed, budget=10, rng=random.Random(0))
    assert stats["grounding_examples"] == 10 and stats["grounding_ceiling_dropped"] == 32
    assert len(by_group["site_b"]) == 10
    assert all("S" in _user_text(e) for e in by_group["site_b"]) and "site_a" not in by_group
    again, _ = select_grounding_examples(mixed, budget=10, rng=random.Random(0))
    assert again == by_group
    empty, stats = select_grounding_examples(mixed, budget=0, rng=random.Random(0))
    assert empty == {} and stats["grounding_examples"] == 0


def _write_corpus(root):
    steps = [
        _click_step(1, *_centre(SAVE_BBOX)),
        {
            "step": 2,
            "screenshot": "step_002_screenshot.png",
            "thought": "b",
            "actions": [{"type": "type", "text": "hi"}],
        },
        {"step": 3, "screenshot": "step_003_screenshot.png", "thought": "c", "actions": [{"type": "done"}]},
    ]
    for site, icon in (("site_a", False), ("site_b", True)):
        d = root / site / "task_1"
        d.mkdir(parents=True)
        for number in range(1, 4):
            (d / f"step_{number:03d}_screenshot.png").write_bytes(_png_bytes(color=(number * 40, 0, 0)))
        bbox = ICON_BBOX if icon else SAVE_BBOX
        steps[0]["actions"][0]["x"], steps[0]["actions"][0]["y"] = _centre(bbox)
        (d / "trajectory.json").write_text(json.dumps({"instruction": "Save it", "steps": steps, "success": True}))
        (d / "metadata.json").write_text(json.dumps({"capture_source": "agent", "thought_source": "teacher"}))
        (d / "capture_meta.json").write_text(
            json.dumps({"steps": {"1": {"role": "button", "name": "Save", "bbox": bbox}}})
        )


def test_convert_action_share_ignores_grounding_and_rows_carry_no_extra_keys():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "corpus", Path(tmp) / "out"
        _write_corpus(root)
        stats = convert_trajectories(root, out, 0.0, health_config=str(Path(tmp) / "none.json"), grounding_share=1.0)
        assert stats["example_action_share"] == {"click": 1 / 3, "type": 1 / 3, "done": 1 / 3}
        assert stats["grounding_examples"] == 4 and stats["grounding_share"] == 4 / 6
        assert stats["grounding_example_share"] == 4 / 10 and stats["grounding_small_target_share"] == 3 / 4
        rows = [json.loads(line) for line in (out / "sft_dataset.jsonl").read_text().splitlines()]
        assert len(rows) == 10 and all(list(row) == ["messages"] for row in rows)
        grounding = [r for r in rows if r["messages"][0]["content"] == GROUNDING_SYSTEM_PROMPT]
        assert len(grounding) == 4 and all("<think>" not in r["messages"][2]["content"] for r in grounding)
        assert all(sum(c["type"] == "image" for c in r["messages"][1]["content"]) == 1 for r in grounding)
        stats = convert_trajectories(root, out, 0.0, health_config=str(Path(tmp) / "none.json"), grounding_share=0.0)
        assert stats["grounding_examples"] == 0 and stats["grounding_share"] == 0.0
        assert len((out / "sft_dataset.jsonl").read_text().splitlines()) == 6
