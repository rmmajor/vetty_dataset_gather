import argparse
import json
import os
import random
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import yaml
from PIL import Image, ImageChops

from cua.pipeline import operator_pii
from cua.pipeline.corpus_audit import corpus_health_violations, trainability

# Unambiguous secret literals that must NEVER reach the training corpus. Masked credentials are
# recorded as `<secret:NAME>` placeholders (driver-side substitution), so any of these appearing as
# a literal is a real leak. Scoped to high-entropy keys/tokens that CANNOT be legitimate page
# content — the gate must never false-positive on normal UI text. Credit-card numbers are
# deliberately NOT matched: we never type real cards (payments are denied/handed-off), so the only
# card-shaped strings in the corpus are page content (year dropdowns, UUIDs) and public test cards
# (4111…/4242…) used in test-env checkout tasks — all false positives. Arbitrary leaked
# passwords/usernames can't be pattern-matched without false positives; the structural guarantee
# (driver records placeholders, asserted by driver_smoke) is their primary defense.
_SECRET_PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("github_token", re.compile(r"\bghp_[A-Za-z0-9]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
]


def _scan_text_for_secrets(text: str) -> list[str]:
    return [name for name, pat in _SECRET_PATTERNS if pat.search(text)] + operator_pii.scan_text(text)


def _scan_examples_for_secrets(examples) -> list[str]:
    """Distinct findings across every message of one trajectory's examples — the text that would ship."""
    findings = []
    for example in examples:
        for message in example["messages"]:
            content = message["content"]
            text = (
                content
                if isinstance(content, str)
                else " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            )
            for finding in _scan_text_for_secrets(text):
                if finding not in findings:
                    findings.append(finding)
    return findings


SYSTEM_PROMPT = """You are a GUI agent that operates a web browser at a 1280x720 resolution. \
Given a task, the current screenshot (and, after the first step, a downscaled screenshot of the \
screen before your last action), and your previous actions annotated with their observed effect \
([no change] when the screen did not change, -> /path when the URL path changed), reason about \
the state in <think>...</think>, then output one or more actions, one per line.

Available actions (x, y are normalized to 0-1000 across the screenshot width and height; \
scrollX/scrollY are pixel deltas):
action: click(<ref>(x, y)</ref>, button=left|right)
action: double_click(<ref>(x, y)</ref>)
action: move(<ref>(x, y)</ref>)
action: type("text")
action: keypress(["Enter"])
action: scroll(<ref>(x, y)</ref>, scrollX=0, scrollY=500)
action: drag(<ref>(x1, y1)</ref> -> <ref>(x2, y2)</ref>)
action: wait()
action: navigate("https://...")
action: handoff(reason=login)
action: confirm(propose=click(<ref>(x, y)</ref>), risk=purchase)
action: done()

Use done() only when the task is fully accomplished and the screen confirms it.
Use handoff(reason=...) to yield control for a login, payment entry, CAPTCHA, or 2FA — never type \
those yourself. Use confirm(propose=..., risk=...) to PAUSE before a consequential, hard-to-undo \
action (purchase, payment, send, publish, delete, consent) and wait for approval instead of doing it."""

GROUNDING_SYSTEM_PROMPT = """You are a GUI agent that operates a web browser at a 1280x720 resolution. \
Given a screenshot and the description of one UI element, output exactly one action that clicks it: \
action: click(<ref>(x, y)</ref>, button=left) with x, y normalized to 0-1000 across the screenshot \
width and height. No reasoning, no other text."""

PROMPT_FORMAT_VERSION = "sft3"
COORD_SPACE = "normalized_1000"
VIEWPORT = (1280, 720)
MODEL_COORD_MAX = 1000


def _parse_size(text):
    width, height = text.lower().split("x")
    return int(width), int(height)


PREV_IMAGE_SIZE = _parse_size(os.environ.get("PREV_IMAGE_SIZE", "640x360"))
EFFECT_DIFF_SIZE = (320, 180)
EFFECT_NO_CHANGE_MAX_FRACTION = float(os.environ.get("EFFECT_NO_CHANGE_MAX_FRACTION", "0.0005"))
EFFECT_PIXEL_DELTA = int(os.environ.get("EFFECT_PIXEL_DELTA", "16"))
EFFECT_PATH_MAX_CHARS = int(os.environ.get("EFFECT_PATH_MAX_CHARS", "40"))


def to_model_coords(x, y):
    return round(x * MODEL_COORD_MAX / VIEWPORT[0]), round(y * MODEL_COORD_MAX / VIEWPORT[1])


def to_viewport_coords(x, y):
    return round(x * VIEWPORT[0] / MODEL_COORD_MAX), round(y * VIEWPORT[1] / MODEL_COORD_MAX)


_PATH_KEYS = ("path",)
_NESTED_ACTION_KEYS = ("proposed_action",)


def _is_point(action):
    return isinstance(action.get("x"), (int, float)) and isinstance(action.get("y"), (int, float))


def to_model_action(action):
    """Pixel-space trajectory action -> model-space copy; scroll deltas and every non-coordinate key untouched."""
    converted = dict(action)
    if _is_point(action):
        converted["x"], converted["y"] = to_model_coords(action["x"], action["y"])
    for key in _PATH_KEYS:
        if isinstance(action.get(key), list):
            converted[key] = [to_model_action(point) for point in action[key]]
    for key in _NESTED_ACTION_KEYS:
        if isinstance(action.get(key), dict):
            converted[key] = to_model_action(action[key])
    return converted


def _diff_plane(png_bytes):
    return Image.open(BytesIO(png_bytes)).convert("L").resize(EFFECT_DIFF_SIZE, Image.Resampling.BOX)


def _changed_fraction(plane_a, plane_b):
    histogram = ImageChops.difference(plane_a, plane_b).histogram()
    return sum(histogram[EFFECT_PIXEL_DELTA + 1 :]) / (plane_a.width * plane_a.height)


def screenshot_changed_fraction(png_a_bytes, png_b_bytes):
    return _changed_fraction(_diff_plane(png_a_bytes), _diff_plane(png_b_bytes))


def url_path_change(url_before, url_after):
    if not url_before or not url_after:
        return None
    before = urlparse(url_before).path.rstrip("/")
    after = urlparse(url_after).path.rstrip("/")
    if before == after:
        return None
    return (after or "/")[:EFFECT_PATH_MAX_CHARS]


def effect_is_no_change(effect):
    return effect.get("changed") is not None and effect["changed"] <= EFFECT_NO_CHANGE_MAX_FRACTION


def effect_annotation(effect):
    if not effect:
        return ""
    text = " [no change]" if effect_is_no_change(effect) else ""
    if effect.get("path"):
        text += f" -> {effect['path']}"
    return text


def step_effects(trajectory_dir, steps, capture_meta):
    """Per step, what the NEXT screenshot/url show the action did; None for the last step or a missing screenshot."""
    trajectory_dir = Path(trajectory_dir)
    meta_steps = (capture_meta or {}).get("steps")
    meta_steps = meta_steps if isinstance(meta_steps, dict) else {}
    planes = {}

    def plane_at(index):
        if index not in planes:
            file = trajectory_dir / steps[index]["screenshot"]
            planes[index] = _diff_plane(file.read_bytes()) if file.is_file() else None
        return planes[index]

    effects = []
    for k, step in enumerate(steps):
        if k + 1 == len(steps) or plane_at(k) is None or plane_at(k + 1) is None:
            effects.append(None)
            continue
        url_before = (meta_steps.get(str(step.get("step"))) or {}).get("url")
        url_after = (meta_steps.get(str(steps[k + 1].get("step"))) or {}).get("url")
        effects.append(
            {"changed": _changed_fraction(plane_at(k), plane_at(k + 1)), "path": url_path_change(url_before, url_after)}
        )
        planes.pop(k, None)
    return effects


def downscale_prev(png_bytes, size=None):
    image = Image.open(BytesIO(png_bytes)).convert("RGB").resize(size or PREV_IMAGE_SIZE, Image.Resampling.LANCZOS)
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def format_action_for_training(action):
    action_type = action["type"]

    if action_type == "click":
        x, y = action["x"], action["y"]
        button = action.get("button", "left")
        return f"action: click(<ref>({x}, {y})</ref>, button={button})"

    elif action_type == "double_click":
        x, y = action["x"], action["y"]
        return f"action: double_click(<ref>({x}, {y})</ref>)"

    elif action_type == "drag":
        path = action["path"]
        start = path[0]
        end = path[-1]
        return f"action: drag(<ref>({start['x']}, {start['y']})</ref> -> <ref>({end['x']}, {end['y']})</ref>)"

    elif action_type == "move":
        x, y = action["x"], action["y"]
        return f"action: move(<ref>({x}, {y})</ref>)"

    elif action_type == "type":
        text = action["text"]
        return f'action: type("{text}")'

    elif action_type == "keypress":
        keys = action["keys"]
        return f"action: keypress({json.dumps(keys)})"

    elif action_type == "scroll":
        x, y = action["x"], action["y"]
        sx, sy = action["scroll_x"], action["scroll_y"]
        return f"action: scroll(<ref>({x}, {y})</ref>, scrollX={sx}, scrollY={sy})"

    elif action_type == "wait":
        return "action: wait()"

    elif action_type == "navigate":
        return f'action: navigate("{action.get("url", "")}")'

    elif action_type == "handoff":
        return f"action: handoff(reason={action.get('reason', 'login')})"

    elif action_type == "confirm":
        risk = action.get("risk_category", "")
        proposed = action.get("proposed_action") or {}
        if "x" in proposed and "y" in proposed:
            inner = f"{proposed.get('type') or 'click'}(<ref>({proposed['x']}, {proposed['y']})</ref>)"
        elif proposed.get("type"):
            inner = f"{proposed['type']}()"
        else:
            inner = "action()"
        return f"action: confirm(propose={inner}, risk={risk})"

    elif action_type == "done":
        return "action: done()"

    elif action_type == "screenshot":
        return "action: screenshot()"

    return f"action: {action_type}()"


def format_action_summary(action):
    action_type = action["type"]

    if action_type in ("click", "double_click", "move"):
        return f"{action_type}({action['x']}, {action['y']})"
    elif action_type == "type":
        return f'type("{action["text"]}")'
    elif action_type == "keypress":
        return f"keypress({action['keys']})"
    elif action_type == "scroll":
        return f"scroll({action['x']}, {action['y']})"
    elif action_type == "drag":
        path = action["path"]
        return f"drag({path[0]['x']},{path[0]['y']} -> {path[-1]['x']},{path[-1]['y']})"
    elif action_type == "navigate":
        return f"navigate({action.get('url', '')})"
    elif action_type == "handoff":
        return f"handoff({action.get('reason', '')})"
    elif action_type == "confirm":
        return f"confirm(risk={action.get('risk_category', '')})"
    return action_type


# Text history shown to the model for each step. Windowed: long trajectories would otherwise
# bloat the prompt, and at inference the runtime keeps a bounded history too.
# 20 covers the corpus's median 24-step auth trajectories; 10 truncated 56.5% of steps.
HISTORY_WINDOW = 20


def build_step_user_text(instruction, action_history):
    """The student's user turn: task + windowed text history; shared by convert and the RL env."""
    user_text = f"Task: {instruction}"
    if action_history:
        window = action_history[-HISTORY_WINDOW:]
        omitted = len(action_history) - len(window)
        prefix = f"... {omitted} earlier actions, " if omitted > 0 else ""
        user_text += f"\nPrevious actions: [{prefix}{', '.join(window)}]"
    user_text += "\nWhat action should you take?"
    return user_text


def history_entries_for_step(step, effect=None):
    """Model-space summaries of a step's (pixel-space) actions; the observed effect annotates the last entry."""
    actions = step.get("actions", [])
    if not actions and "action" in step:
        actions = [step["action"]]
    entries = [format_action_summary(to_model_action(a)) for a in actions]
    event = step.get("event") or {}
    if event.get("kind") == "approval":
        entries.append(f"[watchdog: {event.get('decision', '?')}]")
    if effect is not None and entries:
        entries[-1] += effect_annotation(effect)
    return entries


PREV_IMAGE_HEADER = "Previous screenshot (before your last action, downscaled):\n"
CUR_IMAGE_HEADER = "Current screenshot:\n"
IMAGE_PLACEHOLDER = "<image>"


def _text_item(text):
    return {"type": "text", "text": text}


def _image_item(ref):
    return {"type": "image", "image": ref}


def build_step_observation(instruction, action_history, cur_image, prev_image=None):
    """The student's user turn as interleaved text/image items; `text` is its ms-swift `<image>` flattening."""
    if prev_image is not None:
        content = [_text_item(PREV_IMAGE_HEADER), _image_item(prev_image), _text_item("\n" + CUR_IMAGE_HEADER)]
    else:
        content = [_text_item(CUR_IMAGE_HEADER)]
    content += [_image_item(cur_image), _text_item("\n" + build_step_user_text(instruction, action_history))]
    images = [item["image"] for item in content if item["type"] == "image"]
    return {"content": content, "images": images, "text": flatten_for_swift(content)}


def flatten_for_swift(content):
    return "".join(IMAGE_PLACEHOLDER if item["type"] == "image" else item["text"] for item in content)


def build_step_messages(system, observation):
    return [{"role": "system", "content": system}, {"role": "user", "content": observation["content"]}]


def to_openai_content(content, image_b64_for):
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64_for(item['image'])}"}}
        if item["type"] == "image"
        else _text_item(item["text"])
        for item in content
    ]


def to_claude_content(content, image_b64_for):
    """The same observation in Anthropic's block form, so a frontier arm is prompted identically."""
    return [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64_for(item["image"])}}
        if item["type"] == "image"
        else _text_item(item["text"])
        for item in content
    ]


GROUNDING_INSTRUCTION_TEMPLATE = "Click {description}."


def build_grounding_observation(description, image_ref):
    """A grounding user turn IS a history-free first-step observation whose task is to click one described element."""
    return build_step_observation(GROUNDING_INSTRUCTION_TEMPLATE.format(description=description), [], image_ref)


# Step-level waste filter (opt-in via --exclude-waste-steps). A single-click step whose target
# lands within WASTE_CLICK_RADIUS_PX of a click in the previous WASTE_LOOKBACK_STEPS is a
# retry/backtrack — the "click didn't register, click again" pattern that survives the stuck-loop
# gate (which only drops >=3 IDENTICAL consecutive actions). These steps are dropped as TRAINING
# TARGETS only; they stay in the text history so later steps' context — and their post-retry
# screenshots — remain coherent. Geometry-based, like the loop/viewport gates, not thought-text.
WASTE_CLICK_RADIUS_PX = int(os.environ.get("WASTE_CLICK_RADIUS_PX", "10"))
WASTE_LOOKBACK_STEPS = int(os.environ.get("WASTE_LOOKBACK_STEPS", "4"))


# Meandering trajectories train subsampled (--meandering-mode subsample): the head keeps the setup,
# the tail keeps the goal approach + done(), the wander in between supervises every Nth step only —
# so their 40%+ step mass stops dominating the loss while the stop signal stays intact.
MEANDER_KEEP_HEAD = int(os.environ.get("MEANDER_KEEP_HEAD", "5"))
MEANDER_KEEP_TAIL = int(os.environ.get("MEANDER_KEEP_TAIL", "5"))
# 6 lands the meandering EXAMPLE share at roughly their TRAJECTORY share (~23%): represented, not dominant.
MEANDER_MID_KEEP_EVERY = int(os.environ.get("MEANDER_MID_KEEP_EVERY", "6"))

# A keypress action repeating one key beyond this (10-30-key ArrowDown chains) is dropped as a
# target: the post-chain screenshot no longer matches a sensible per-press supervision signal.
KEYPRESS_MAX_REPEAT = int(os.environ.get("KEYPRESS_MAX_REPEAT", "6"))

_KEY_CANON = {
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


def normalize_trajectory_actions(trajectory):
    """Canonical key casing (ENTER -> Enter) in place, so examples and history agree on one spelling."""
    for step in trajectory.get("steps", []):
        for action in step.get("actions", []):
            if action.get("type") == "keypress" and action.get("keys"):
                action["keys"] = [k if len(k) == 1 else _KEY_CANON.get(k.lower(), k) for k in action["keys"]]


def _meander_drop_indices(steps, head=None, tail=None, keep_every=None):
    """Mid-trajectory step indices to drop as targets in a meandering trajectory (kept in history)."""
    head = MEANDER_KEEP_HEAD if head is None else head
    tail = MEANDER_KEEP_TAIL if tail is None else tail
    keep_every = MEANDER_MID_KEEP_EVERY if keep_every is None else keep_every
    drop = set()
    for i in range(head, max(head, len(steps) - tail)):
        if (i - head) % keep_every == 0:
            continue
        if any(a.get("type") == "done" for a in steps[i].get("actions", [])):
            continue
        drop.add(i)
    return drop


def _absurd_keypress_indices(steps):
    """Indices of steps whose keypress repeats one key more than KEYPRESS_MAX_REPEAT times."""
    absurd = set()
    for i, step in enumerate(steps):
        for action in step.get("actions", []):
            keys = action.get("keys") or []
            if action.get("type") == "keypress" and len(keys) > KEYPRESS_MAX_REPEAT and len(set(keys)) == 1:
                absurd.add(i)
    return absurd


def _waste_step_indices(steps, radius_px=WASTE_CLICK_RADIUS_PX, lookback=WASTE_LOOKBACK_STEPS):
    """Indices of single-click steps that merely retry a recent click (within radius_px of a click
    in the previous `lookback` steps). Only PURE single-click steps qualify — a step that also
    types/scrolls/keypresses does real work and is never waste; scrolls (which legitimately repeat
    x,y with different deltas) are never considered. A step's own click is recorded only after it
    is checked, so a step never matches itself."""
    waste = set()
    recent = []
    for i, step in enumerate(steps):
        actions = step.get("actions") or ([step["action"]] if "action" in step else [])
        clicks = [a for a in actions if a.get("type") == "click" and a.get("x") is not None]
        if len(actions) == 1 and len(clicks) == 1:
            x, y = clicks[0]["x"], clicks[0]["y"]
            if any(i - j <= lookback and abs(x - px) <= radius_px and abs(y - py) <= radius_px for px, py, j in recent):
                waste.add(i)
        for a in clicks:
            recent.append((a["x"], a["y"], i))
    return waste


def _maybe_synthesize_done_step(trajectory, trajectory_dir, capture_meta):
    """Terminal supervision for agent trajectories recorded before the driver emitted `done`
    steps: when the teacher honestly reported completion and the final screenshot exists,
    append a {type:"done"} step whose thought is the teacher's own completion summary. Without
    a terminal step the model is never taught to STOP — it would act forever at inference."""
    steps = trajectory.get("steps", [])
    if any(a.get("type") == "done" for s in steps for a in s.get("actions", [])):
        return
    if capture_meta.get("agent_outcome") != "completed":
        return
    summary = (capture_meta.get("agent_summary") or "").strip()
    final_screenshot = trajectory_dir / "final_screenshot.png"
    if not summary or not final_screenshot.exists():
        return
    steps.append(
        {
            "step": (steps[-1]["step"] + 1) if steps else 1,
            "screenshot": "final_screenshot.png",
            "thought": summary,
            "actions": [{"type": "done"}],
        }
    )


def _write_prev_image(trajectory_dir, steps, index, prev_dir):
    """Downscaled previous-step screenshot for example `index`; None on step 0, without prev_dir, or a missing source."""
    if index == 0 or prev_dir is None:
        return None
    source = Path(trajectory_dir) / steps[index - 1]["screenshot"]
    if not source.is_file():
        return None
    target = Path(prev_dir) / Path(trajectory_dir).name / f"step_{index:03d}_prev.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(downscale_prev(source.read_bytes()))
    return str(target.resolve())


def convert_trajectory_to_examples(
    trajectory, trajectory_dir, exclude_waste_steps=False, extra_drop_indices=(), capture_meta=None, prev_dir=None
):
    """One training example PER STEP: current screenshot + task + windowed text history ->
    <think> + action(s). A whole trajectory as one multi-turn example packs every screenshot
    (~300 visual tokens each) into a single sequence — a 21-step trajectory is ~9k tokens
    against max_length 4096, so the trainer silently truncated exactly the long-horizon tails
    this dataset exists for. Per-step examples match inference conditions (one screenshot +
    text history) and supervise every step uniformly.

    Returns (examples, stats). Steps in `extra_drop_indices` (meandering subsample, absurd
    keypress chains) and — with exclude_waste_steps — retry/backtrack clicks are omitted as
    TARGETS but still flow into action_history, so the trajectory keeps all its steps and
    downstream context stays intact — only the bad supervision target is removed."""
    instruction = trajectory["instruction"]
    steps = trajectory["steps"]
    stats = {
        "waste_dropped": 0,
        "subsample_dropped": 0,
        "target_actions": {},
        "done_emitted": False,
        "effect_steps": 0,
        "effect_no_change": 0,
        "effect_url_change": 0,
    }
    if not steps:
        return [], stats

    examples = []
    action_history = []
    waste = _waste_step_indices(steps) if exclude_waste_steps else set()
    extra_drop = set(extra_drop_indices or ())
    effects = step_effects(trajectory_dir, steps, capture_meta)
    for effect in filter(None, effects):
        stats["effect_steps"] += 1
        stats["effect_no_change"] += int(effect_is_no_change(effect))
        stats["effect_url_change"] += int(bool(effect.get("path")))

    for i, step in enumerate(steps):
        screenshot_path = (trajectory_dir / step["screenshot"]).resolve()
        thought = step.get("thought", "")
        actions = step.get("actions", [])
        if not actions and "action" in step:
            actions = [step["action"]]

        emit = screenshot_path.exists() and bool(actions)
        if emit and i in waste:
            stats["waste_dropped"] += 1
        elif emit and i in extra_drop:
            stats["subsample_dropped"] += 1
        elif emit:
            prev_image = _write_prev_image(trajectory_dir, steps, i, prev_dir)
            observation = build_step_observation(instruction, action_history, str(screenshot_path), prev_image)
            action_str = "\n".join(format_action_for_training(to_model_action(a)) for a in actions)
            assistant_content = f"<think>{thought}</think>\n{action_str}" if thought else action_str
            examples.append(
                {
                    "messages": [
                        *build_step_messages(SYSTEM_PROMPT, observation),
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
            )
            for a in actions:
                t = a.get("type", "?")
                stats["target_actions"][t] = stats["target_actions"].get(t, 0) + 1
                # Any terminal stop verb is stop-supervision — wall demos legitimately end in handoff.
                if t in ("done", "handoff", "confirm"):
                    stats["done_emitted"] = True

        action_history.extend(history_entries_for_step(step, effects[i]))

    return examples, stats


# Grounding slice: recorded clicks land on a named element's bbox centre — a free ScreenSpot-style set on our distribution.
GROUNDING_ROLE_NOUNS = {
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
GROUNDING_TEMPLATES = ('the "{name}" {noun}', 'the {noun} labelled "{name}"', 'the {noun} "{name}"')
GROUNDING_NAME_MAX_CHARS = 60
GROUNDING_CENTRE_TOLERANCE_PX = 2
GROUNDING_SMALL_TARGET_PX = 40
GROUNDING_SMALL_OVERSAMPLE = int(os.environ.get("GROUNDING_SMALL_OVERSAMPLE", "3"))
GROUNDING_PER_GROUP_CAP = int(os.environ.get("GROUNDING_PER_GROUP_CAP", "40"))
GROUNDING_DEDUPE_MAX = 2
GROUNDING_DEDUPE_CELL_PX = 8
GROUNDING_SEED = int(os.environ.get("GROUNDING_SEED", "0"))
GROUNDING_DEFAULT_SHARE = 0.25
# Masked-password placeholders, password labels and secret placeholders: never a grounding target.
GROUNDING_SKIP_NAME_PATTERNS = (re.compile(r"[•*]{3,}"), re.compile(r"passw", re.IGNORECASE), re.compile(r"<secret:"))
GROUNDING_SKIP_INPUT_TYPES = ("password",)


def grounding_description(name, noun, variant=0):
    return GROUNDING_TEMPLATES[variant % len(GROUNDING_TEMPLATES)].format(name=name, noun=noun)


def _bbox_centre(bbox):
    return round(bbox["x"] + bbox["width"] / 2), round(bbox["y"] + bbox["height"] / 2)


def _bbox_in_viewport(bbox):
    x, y, w, h = (bbox.get(k) for k in ("x", "y", "width", "height"))
    if not all(isinstance(v, (int, float)) for v in (x, y, w, h)):
        return False
    return w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= VIEWPORT[0] and y + h <= VIEWPORT[1]


def _grounding_skip_reason(action, meta, screenshot_path):
    """Why a step is not a grounding target, or None; the reason keys are the convert report's histogram."""
    if action.get("type") != "click" or not _is_point(action):
        return "not_click"
    role = meta.get("role")
    if not isinstance(role, str) or role not in GROUNDING_ROLE_NOUNS:
        return "no_role"
    name = (meta.get("name") or "").strip()
    if not name:
        return "no_name"
    if len(name) > GROUNDING_NAME_MAX_CHARS:
        return "name_too_long"
    if (
        any(p.search(name) for p in GROUNDING_SKIP_NAME_PATTERNS)
        or (meta.get("attrs") or {}).get("type") in GROUNDING_SKIP_INPUT_TYPES
    ):
        return "password_or_masked"
    if _scan_text_for_secrets(name):
        return "secret"
    bbox = meta.get("bbox")
    if not isinstance(bbox, dict) or not _bbox_in_viewport(bbox):
        return "bbox_off_viewport"
    cx, cy = _bbox_centre(bbox)
    if abs(action["x"] - cx) > GROUNDING_CENTRE_TOLERANCE_PX or abs(action["y"] - cy) > GROUNDING_CENTRE_TOLERANCE_PX:
        return "off_centre"
    if not screenshot_path.is_file():
        return "no_screenshot"
    return None


def grounding_targets(trajectory, trajectory_dir, capture_meta):
    """Clean grounding targets from a trajectory's click steps, plus a skip-reason histogram for the report."""
    meta_steps = (capture_meta or {}).get("steps")
    meta_steps = meta_steps if isinstance(meta_steps, dict) else {}
    targets, skipped = [], {}
    for step in trajectory.get("steps", []):
        actions = step.get("actions") or ([step["action"]] if "action" in step else [])
        meta = meta_steps.get(str(step.get("step"))) or {}
        screenshot_path = (Path(trajectory_dir) / (step.get("screenshot") or "")).resolve()
        reason = _grounding_skip_reason(actions[0] if actions else {}, meta, screenshot_path)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        bbox = meta["bbox"]
        cx, cy = _bbox_centre(bbox)
        targets.append(
            {
                "screenshot": str(screenshot_path),
                "name": meta["name"].strip(),
                "role": meta["role"],
                "noun": GROUNDING_ROLE_NOUNS[meta["role"]],
                "x": cx,
                "y": cy,
                "small": bbox["width"] <= GROUNDING_SMALL_TARGET_PX and bbox["height"] <= GROUNDING_SMALL_TARGET_PX,
            }
        )
    return targets, skipped


def grounding_example(target, variant=0):
    """Description -> bbox-centre click with no <think>: the grounding slice supervises pointing, not planning."""
    description = grounding_description(target["name"], target["noun"], variant)
    observation = build_grounding_observation(description, target["screenshot"])
    action = to_model_action({"type": "click", "x": target["x"], "y": target["y"], "button": "left"})
    return {
        "messages": [
            *build_step_messages(GROUNDING_SYSTEM_PROMPT, observation),
            {"role": "assistant", "content": format_action_for_training(action)},
        ]
    }


def _dedupe_targets(targets, dedupe_max):
    """At most `dedupe_max` targets per (name, 8px cell): one button clicked in ten trajectories is one element."""
    seen, kept = {}, []
    for target in targets:
        key = (
            target["name"],
            round(target["x"] / GROUNDING_DEDUPE_CELL_PX),
            round(target["y"] / GROUNDING_DEDUPE_CELL_PX),
        )
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= dedupe_max:
            kept.append(target)
    return kept


def _grounding_tier(target, variant):
    """Survival order under a cap: small-target primaries, then their rephrasings, large targets dropped first."""
    return 0 if target["small"] and variant == 0 else 1 if target["small"] else 2


def _ranked(rows, rng):
    rng.shuffle(rows)
    rows.sort(key=lambda row: row["tier"])
    return rows


def select_grounding_examples(targets_by_group, budget, rng, oversample=None, per_group_cap=None, dedupe_max=None):
    """Dedupe, oversample small targets, cap per group, then keep <= budget examples with small targets dropped last."""
    oversample = GROUNDING_SMALL_OVERSAMPLE if oversample is None else oversample
    per_group_cap = GROUNDING_PER_GROUP_CAP if per_group_cap is None else per_group_cap
    dedupe_max = GROUNDING_DEDUPE_MAX if dedupe_max is None else dedupe_max
    stats = {"grounding_candidates": 0, "grounding_deduped": 0, "grounding_capped": 0, "grounding_ceiling_dropped": 0}
    rows = []
    for group, targets in sorted(targets_by_group.items()):
        unique = _dedupe_targets(targets, dedupe_max)
        stats["grounding_candidates"] += len(targets)
        stats["grounding_deduped"] += len(targets) - len(unique)
        group_rows = [
            {
                "group": group,
                "tier": _grounding_tier(t, v),
                "small": t["small"],
                "role": t["role"],
                "example": grounding_example(t, v),
            }
            for t in unique
            for v in range(oversample if t["small"] else 1)
        ]
        group_rows = _ranked(group_rows, rng)
        stats["grounding_capped"] += max(0, len(group_rows) - per_group_cap)
        rows.extend(group_rows[:per_group_cap])
    rows = _ranked(rows, rng)
    stats["grounding_ceiling_dropped"] = max(0, len(rows) - max(0, budget))
    rows = rows[: max(0, budget)]
    examples_by_group, roles = {}, {}
    for row in rows:
        examples_by_group.setdefault(row["group"], []).append(row["example"])
        roles[row["role"]] = roles.get(row["role"], 0) + 1
    stats["grounding_examples"] = len(rows)
    stats["grounding_small_target_share"] = sum(row["small"] for row in rows) / len(rows) if rows else 0.0
    stats["grounding_roles"] = roles
    return examples_by_group, stats


def convert_trajectories(
    input_dir,
    output_dir,
    val_split,
    include_replay=False,
    meandering_mode="keep",
    exclude_waste_steps=False,
    health_config="corpus_health.json",
    skip_health_gate=False,
    grounding_share=GROUNDING_DEFAULT_SHARE,
    grounding_seed=GROUNDING_SEED,
    seed=None,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_examples = []
    leak_sources = []
    examples_by_group = {}
    grounding_targets_by_group = {}
    skipped = {
        "excluded": 0,
        "unverified": 0,
        "replay_gated": 0,
        "empty": 0,
        "meandering": 0,
        "stuck_loop": 0,
        "unreadable": 0,
    }
    report = {
        "trajectories": 0,
        "by_source": {},
        "actions": {},
        "target_actions": {},
        "lengths": [],
        "domains": {},
        "done_steps": 0,
        "done_trajectories": 0,
        "thought_sources": {},
        "meandering": 0,
        "meandering_examples": 0,
        "teacher_raw": 0,
        "waste_steps_dropped": 0,
        "subsample_dropped": 0,
        "autonomy": {},
        "auth_mode": {},
        "effect_steps": 0,
        "effect_no_change": 0,
        "effect_url_change": 0,
        "grounding_skipped": {},
    }

    trajectory_dirs = sorted(d for d in input_path.glob("**/task_*") if d.is_dir() and (d / "trajectory.json").exists())
    print(f"Found {len(trajectory_dirs)} trajectory directories")

    for traj_dir in trajectory_dirs:
        # A trajectory being written by a concurrent capture parses as truncated JSON; skipping it
        # keeps one half-written file from killing a multi-hour training run after the GPU is up.
        try:
            with open(traj_dir / "trajectory.json") as f:
                trajectory = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"SKIP unreadable {traj_dir}: {e}")
            skipped["unreadable"] += 1
            continue

        metadata = {}
        metadata_file = traj_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        capture_meta = {}
        capture_meta_file = traj_dir / "capture_meta.json"
        if capture_meta_file.exists():
            try:
                with open(capture_meta_file) as f:
                    capture_meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        source = metadata.get("capture_source") or ("path_a" if "mind2web" in traj_dir.name else "replay")

        # The ONE trainability gate, shared with cua.pipeline.corpus_audit so they cannot drift.
        skip_reason = trainability(trajectory, metadata, capture_meta, traj_dir.name, include_replay=include_replay)
        if skip_reason:
            skipped[skip_reason] += 1
            continue

        # Efficiency is a quality tag, not a gate: meandering goal-completions train subsampled or
        # in full, or can be dropped for a clean run. `success` already means the goal was met.
        is_meandering = (capture_meta.get("verifier") or {}).get("efficiency") == "meandering" or metadata.get(
            "efficiency"
        ) == "meandering"
        if meandering_mode == "exclude" and is_meandering:
            skipped["meandering"] += 1
            continue

        if source == "agent":
            _maybe_synthesize_done_step(trajectory, traj_dir, capture_meta)
        normalize_trajectory_actions(trajectory)

        steps = trajectory.get("steps", [])
        extra_drop = _absurd_keypress_indices(steps)
        if meandering_mode == "subsample" and is_meandering:
            extra_drop |= _meander_drop_indices(steps)

        # Split unit: the APP/SITE, not the step. Steps of one trajectory (and trajectories of
        # one app) share screenshots and layout — splitting below app level leaks pixels into
        # val and turns val loss into a memorization check.
        parent = traj_dir.parent.name
        group = parent if parent.startswith(("repo_", "site_")) else traj_dir.name

        examples, traj_stats = convert_trajectory_to_examples(
            trajectory,
            traj_dir,
            exclude_waste_steps=exclude_waste_steps,
            extra_drop_indices=extra_drop,
            capture_meta=capture_meta,
            prev_dir=output_path / "prev" / group,
        )
        report["waste_steps_dropped"] += traj_stats["waste_dropped"]
        report["subsample_dropped"] += traj_stats["subsample_dropped"]
        for key in ("effect_steps", "effect_no_change", "effect_url_change"):
            report[key] += traj_stats[key]
        if not examples:
            skipped["empty"] += 1
            continue

        all_examples.extend(examples)
        leaks = _scan_examples_for_secrets(examples)
        if leaks:
            leak_sources.append((traj_dir, leaks))
        examples_by_group.setdefault(group, []).extend(examples)
        targets, grounding_skipped = grounding_targets(trajectory, traj_dir, capture_meta)
        grounding_targets_by_group.setdefault(group, []).extend(targets)
        for reason, n in grounding_skipped.items():
            report["grounding_skipped"][reason] = report["grounding_skipped"].get(reason, 0) + n
        report["trajectories"] += 1
        if is_meandering:
            report["meandering"] += 1
            report["meandering_examples"] += len(examples)
        if traj_stats["done_emitted"]:
            report["done_trajectories"] += 1
        for t, n in traj_stats["target_actions"].items():
            report["target_actions"][t] = report["target_actions"].get(t, 0) + n
        report["by_source"][source] = report["by_source"].get(source, 0) + 1
        report["lengths"].append(len(trajectory.get("steps", [])))
        domain = metadata.get("website") or metadata.get("domain") or "unknown"
        report["domains"][domain] = report["domains"].get(domain, 0) + 1
        ts = metadata.get("thought_source", "unknown")
        if ts == "teacher_raw":
            report["teacher_raw"] += 1
        report["thought_sources"][ts] = report["thought_sources"].get(ts, 0) + 1
        task_meta = capture_meta.get("task_meta") or {}
        au = metadata.get("autonomy") or task_meta.get("autonomy") or "autonomous"
        report["autonomy"][au] = report["autonomy"].get(au, 0) + 1
        am = metadata.get("auth_mode") or task_meta.get("auth_mode") or "none"
        report["auth_mode"][am] = report["auth_mode"].get(am, 0) + 1
        for step in trajectory.get("steps", []):
            for a in step.get("actions", []):
                t = a.get("type", "?")
                report["actions"][t] = report["actions"].get(t, 0) + 1
                if t == "done":
                    report["done_steps"] += 1

    # Grounding joins each site's split group so val stays leak-free; the ceiling is a share of AGENT examples.
    agent_example_count = len(all_examples)
    grounding_by_group, grounding = select_grounding_examples(
        grounding_targets_by_group, int(grounding_share * agent_example_count), random.Random(grounding_seed)
    )
    for group, examples in grounding_by_group.items():
        leaks = _scan_examples_for_secrets(examples)
        if leaks:
            leak_sources.append((f"grounding:{group}", leaks))
        examples_by_group.setdefault(group, []).extend(examples)
        all_examples.extend(examples)

    # ---- Corpus report: the at-a-glance health check that catches a Booking.com-style
    # regression at convert time instead of in a JSON file weeks later. ----
    lengths = sorted(report["lengths"])
    total_actions = sum(report["actions"].values()) or 1
    print("\n=== Corpus report ===")
    print(
        f"Prompt format: {PROMPT_FORMAT_VERSION} ({COORD_SPACE}, prev image {PREV_IMAGE_SIZE[0]}x{PREV_IMAGE_SIZE[1]})"
    )
    print(
        f"Trajectories trained on: {report['trajectories']}  (examples: {len(all_examples)} = "
        f"{agent_example_count} agent steps + {grounding['grounding_examples']} grounding)"
    )
    print(f"By source: {report['by_source']}")
    print(f"Autonomy: {report['autonomy']}  Auth modes: {report['auth_mode']}")
    print(f"Thought sources: {report['thought_sources']}")
    if lengths:
        mid = lengths[len(lengths) // 2]
        print(
            f"Steps/trajectory: min={lengths[0]} median={mid} p90={lengths[int(len(lengths) * 0.9)]} max={lengths[-1]}"
        )
    print(
        "Action mix: "
        + ", ".join(
            f"{t}={n} ({n / total_actions:.0%})" for t, n in sorted(report["actions"].items(), key=lambda kv: -kv[1])
        )
    )
    total_targets = sum(report["target_actions"].values()) or 1
    print(
        "Supervision-target mix: "
        + ", ".join(
            f"{t}={n} ({n / total_targets:.0%})"
            for t, n in sorted(report["target_actions"].items(), key=lambda kv: -kv[1])
        )
    )
    print(
        f"Terminal done() steps: {report['done_steps']}/{report['trajectories']} trajectories "
        f"(done in emitted targets: {report['done_trajectories']})"
    )
    print(
        f"Meandering (mode={meandering_mode}): {report['meandering']} trajectories, "
        f"{report['meandering_examples']} examples kept, {report['subsample_dropped']} step targets subsampled away"
    )
    print(f"Waste steps dropped as targets (--exclude-waste-steps): {report['waste_steps_dropped']}")
    effect_steps = report["effect_steps"] or 1
    print(
        f"History effects: [no change] on {report['effect_no_change']}/{report['effect_steps']} annotated steps "
        f"({report['effect_no_change'] / effect_steps:.1%}), URL path change on "
        f"{report['effect_url_change']}/{report['effect_steps']} ({report['effect_url_change'] / effect_steps:.1%})"
    )
    grounding_share_actual = grounding["grounding_examples"] / agent_example_count if agent_example_count else 0.0
    print(
        f"Grounding slice (ceiling {grounding_share:.0%} of agent examples, seed {grounding_seed}): "
        f"{grounding['grounding_examples']} examples = {grounding_share_actual:.1%} of agent steps, "
        f"{grounding['grounding_examples'] / (len(all_examples) or 1):.1%} of all; "
        f"small (<= {GROUNDING_SMALL_TARGET_PX}px) targets {grounding['grounding_small_target_share']:.1%}"
    )
    print(
        f"Grounding funnel: {grounding['grounding_candidates']} candidates, -{grounding['grounding_deduped']} dedupe, "
        f"-{grounding['grounding_capped']} per-group cap ({GROUNDING_PER_GROUP_CAP}), "
        f"-{grounding['grounding_ceiling_dropped']} ceiling; small targets x{GROUNDING_SMALL_OVERSAMPLE}"
    )
    print(
        "Grounding roles: "
        + (
            ", ".join(f"{r}={n}" for r, n in sorted(grounding["grounding_roles"].items(), key=lambda kv: -kv[1]))
            or "none"
        )
    )
    print(
        "Grounding skipped steps: "
        + (
            ", ".join(f"{r}={n}" for r, n in sorted(report["grounding_skipped"].items(), key=lambda kv: -kv[1]))
            or "none"
        )
    )
    print("Domains: " + ", ".join(f"{d}={n}" for d, n in sorted(report["domains"].items(), key=lambda kv: -kv[1])[:10]))
    print(f"Skipped: {skipped}")
    print("======================\n")

    identities = operator_pii.configured_identities()
    print(
        "Operator-identity patterns: "
        + (
            ", ".join(operator_pii.mask(v) for v in identities)
            or "NONE — the PII gate is inert; set OPERATOR_PII or credentials/operator_pii.txt"
        )
    )

    # Leak gate (FAIL CLOSED): any literal secret or operator identity refuses the whole dataset.
    if leak_sources:
        print(f"SECRET/PII LEAK DETECTED in {len(leak_sources)} trajectories — refusing to write dataset.")
        for traj_dir, findings in leak_sources:
            print(f"  {traj_dir}: {', '.join(findings)}")
        print(
            "Quarantine each one (metadata.json excluded_reason) or re-capture: credentials belong in "
            "<secret:NAME> placeholders, and a real person's identity belongs nowhere in the corpus."
        )
        sys.exit(1)

    # Corpus health gate (FAIL CLOSED): the regressions that produced sft-v1's gaps refuse to ship
    # again. Thresholds live in corpus_health.json; --skip-health-gate is for experiments only.
    health_path = Path(health_config)
    max_group, max_group_examples = max(examples_by_group.items(), key=lambda kv: len(kv[1]), default=("", []))
    stats = {
        "trainable": report["trajectories"],
        "done_coverage": (report["done_trajectories"] / report["trajectories"] if report["trajectories"] else 0.0),
        "teacher_raw": report["teacher_raw"],
        "meandering_example_share": (
            report["meandering_examples"] / agent_example_count if agent_example_count else 0.0
        ),
        "max_group": max_group,
        "max_group_example_share": (len(max_group_examples) / len(all_examples) if all_examples else 0.0),
        # Agent step targets only: grounding clicks would otherwise inflate the click share past every floor.
        "example_action_share": {t: n / total_targets for t, n in report["target_actions"].items()},
        "grounding_examples": grounding["grounding_examples"],
        "grounding_share": grounding_share_actual,
        "grounding_example_share": grounding["grounding_examples"] / len(all_examples) if all_examples else 0.0,
        "grounding_small_target_share": grounding["grounding_small_target_share"],
    }
    if health_path.is_file():
        violations = corpus_health_violations(stats, json.loads(health_path.read_text()))
        if violations:
            print(
                f"CORPUS HEALTH GATE {'(bypassed with --skip-health-gate)' if skip_health_gate else 'FAILED'} "
                f"— {len(violations)} violation(s) against {health_path}:"
            )
            for violation in violations:
                print(f"  - {violation}")
            if not skip_health_gate:
                print("Fix the corpus (or consciously bypass with --skip-health-gate) before shipping a dataset.")
                sys.exit(2)
        else:
            print(f"Corpus health gate: PASS ({health_path})")
    else:
        print(f"Corpus health gate: SKIPPED — no thresholds file at {health_path}")

    # Group-wise split: whole apps go to val, never individual steps.
    groups = list(examples_by_group.items())
    if seed is not None:
        random.seed(seed)
    random.shuffle(groups)
    val_target = int(len(all_examples) * val_split)
    val_examples, train_examples, val_groups = [], [], []
    for group, examples in groups:
        if val_target and len(val_examples) < val_target and len(groups) > 1:
            val_examples.extend(examples)
            val_groups.append(group)
        else:
            train_examples.extend(examples)
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    if val_groups:
        print(f"Validation groups (held-out apps): {', '.join(val_groups)}")

    train_path = output_path / "sft_dataset.jsonl"
    with open(train_path, "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")
    print(f"Wrote {len(train_examples)} training examples to {train_path}")

    if val_examples:
        val_path = output_path / "sft_val.jsonl"
        with open(val_path, "w") as f:
            for example in val_examples:
                f.write(json.dumps(example) + "\n")
        print(f"Wrote {len(val_examples)} validation examples to {val_path}")
    return stats


SFT_FLAG_MAP = {
    "model": "--model",
    "train_type": "--tuner_type",
    "dataset": "--dataset",
    "val_dataset": "--val_dataset",
    "eval_strategy": "--eval_strategy",
    "packing": "--packing",
    "attn_impl": "--attn_impl",
    "torch_dtype": "--torch_dtype",
    "num_train_epochs": "--num_train_epochs",
    "per_device_train_batch_size": "--per_device_train_batch_size",
    "learning_rate": "--learning_rate",
    "lora_rank": "--lora_rank",
    "lora_alpha": "--lora_alpha",
    "target_modules": "--target_modules",
    "gradient_accumulation_steps": "--gradient_accumulation_steps",
    "max_length": "--max_length",
    "warmup_ratio": "--warmup_ratio",
    "freeze_vit": "--freeze_vit",
    "output_dir": "--output_dir",
}
# Config keys that are environment, not flags (train_model exports them); everything else unmapped is forwarded as --key value.
SFT_ENV_KEYS = ("max_pixels",)


def build_sft_command(config, dataset=None, val_dataset=None, passthrough_args=None):
    """The swift sft argv for a config: mapped keys via SFT_FLAG_MAP, unmapped keys forwarded verbatim, then passthrough."""
    merged = dict(config)
    if dataset:
        merged["dataset"] = dataset
    if val_dataset:
        merged["val_dataset"] = val_dataset
    cmd = ["swift", "sft"]
    for key, value in merged.items():
        if key in SFT_ENV_KEYS:
            continue
        flag = SFT_FLAG_MAP.get(key, f"--{key}")
        if isinstance(value, bool):
            value = str(value).lower()
        cmd.extend([flag, str(value)])
    cmd.extend(passthrough_args or [])
    return cmd


def train_model(
    config_path,
    dataset_override=None,
    epochs_override=None,
    lr_override=None,
    batch_size_override=None,
    passthrough_args=None,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if dataset_override:
        config["dataset"] = dataset_override
    if epochs_override:
        config["num_train_epochs"] = epochs_override
    if lr_override:
        config["learning_rate"] = lr_override
    if batch_size_override:
        config["per_device_train_batch_size"] = batch_size_override

    # convert writes sft_val.jsonl beside the train file; without this it was generated and discarded,
    # so no eval_loss was ever computed and overfitting was invisible.
    if not config.get("val_dataset") and config.get("dataset"):
        sibling = Path(config["dataset"]).parent / "sft_val.jsonl"
        if sibling.is_file():
            config["val_dataset"] = str(sibling)
            print(f"Validation split found: {sibling}")
        else:
            # eval_strategy without a val set makes the trainer raise at startup, so honour the
            # warning instead of dying: no split means no evaluation.
            dropped = config.pop("eval_strategy", None)
            print(
                f"No validation split at {sibling} — training will report no eval_loss"
                + (f" (eval_strategy={dropped} dropped)" if dropped else "")
            )

    cmd = build_sft_command(config, passthrough_args=passthrough_args)

    env = os.environ.copy()
    max_pixels = config.get("max_pixels")
    if max_pixels:
        env["MAX_PIXELS"] = str(max_pixels)

    print("Running command:")
    print("  " + " ".join(cmd))
    if max_pixels:
        print(f"  MAX_PIXELS={max_pixels}")

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Convert trajectories and train Qwen 3.5-9B")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert trajectories to ms-swift SFT format")
    convert_parser.add_argument("--input", type=str, required=True, help="Input trajectories directory")
    convert_parser.add_argument("--output", type=str, default="training_data/", help="Output directory")
    convert_parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    convert_parser.add_argument(
        "--exclude-waste-steps",
        action="store_true",
        help="Drop retry/backtrack click steps as training targets (kept in "
        "history); non-destructive, trajectory files untouched",
    )
    convert_parser.add_argument(
        "--meandering-mode",
        choices=("keep", "subsample", "exclude"),
        default="keep",
        help="Verifier-graded meandering trajectories: keep all steps, subsample "
        "mid-wander targets (head/tail kept, MEANDER_* env knobs), or drop whole",
    )
    convert_parser.add_argument(
        "--exclude-meandering", action="store_true", help="Deprecated alias for --meandering-mode exclude"
    )
    convert_parser.add_argument(
        "--include-replay",
        action="store_true",
        help="Also train on legacy test-replay trajectories (QA-style supervision; off by default)",
    )
    convert_parser.add_argument(
        "--health-config",
        type=str,
        default="corpus_health.json",
        help="Corpus health thresholds JSON (missing file skips the gate)",
    )
    convert_parser.add_argument(
        "--skip-health-gate",
        action="store_true",
        help="Report health violations but write the dataset anyway (experiments only)",
    )
    convert_parser.add_argument(
        "--grounding-share",
        type=float,
        default=GROUNDING_DEFAULT_SHARE,
        help="Ceiling on element-grounding click examples as a share of agent step examples "
        "(mined from recorded click targets; GROUNDING_* env knobs tune sampling)",
    )
    convert_parser.add_argument("--no-grounding", action="store_true", help="Alias for --grounding-share 0")
    convert_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the split shuffles and the grounding sampler for a reproducible dataset "
        "(default: unseeded splits, GROUNDING_SEED for grounding)",
    )

    train_parser = subparsers.add_parser("train", help="Launch ms-swift SFT training")
    train_parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    train_parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    train_parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    train_parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")

    args, passthrough = parser.parse_known_args()

    if args.command == "convert":
        meandering_mode = "exclude" if args.exclude_meandering else args.meandering_mode
        convert_trajectories(
            args.input,
            args.output,
            args.val_split,
            include_replay=args.include_replay,
            meandering_mode=meandering_mode,
            exclude_waste_steps=args.exclude_waste_steps,
            health_config=args.health_config,
            skip_health_gate=args.skip_health_gate,
            grounding_share=0.0 if args.no_grounding else args.grounding_share,
            grounding_seed=GROUNDING_SEED if args.seed is None else args.seed,
            seed=args.seed,
        )
    elif args.command == "train":
        train_model(args.config, args.dataset, args.epochs, args.lr, args.batch_size, passthrough)


if __name__ == "__main__":
    main()
