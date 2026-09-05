"""Self-test for the student eval harness: pure checks, a mock replay episode, and a real judge verdict."""
import asyncio
import argparse
import base64
import io
import json
import logging
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from repo_pipeline.student_eval import (
    EXPECTED_TO_DIFFER_KEYS,
    PARITY_EXEMPT_KEYS,
    ACTION_TO_DRIVER,
    BROWSER_BACKENDS,
    EvalBrowser,
    EvalConfig,
    ReplayStudent,
    action_coordinates,
    EpisodeRecord,
    build_student_messages,
    coord_space_diagnosis,
    detect_block,
    driver_payload,
    judge_episode,
    observed_effect,
    outside_coord_space,
    outside_viewport,
    classify_action_line,
    compute_metrics,
    parse_action_line,
    parse_student_output,
    render_comparison,
    harness_fingerprint,
    render_report,
    run_episode,
    scale_action,
)
from repo_pipeline.eval_task_sources import TASK_SOURCES, EvalTask
from repo_pipeline.sandbox import ShellSandbox
from train_qwen import (
    HISTORY_WINDOW,
    PREV_IMAGE_SIZE,
    PROMPT_FORMAT_VERSION,
    SYSTEM_PROMPT,
    format_action_for_training,
    history_entries_for_step,
    to_viewport_coords,
)

logger = logging.getLogger(__name__)

PROMPT_GOLDEN = Path(__file__).resolve().parent.parent / "rl" / "tests" / "fixtures" / "prompt_golden.json"
GOLDEN_INSTRUCTION = "Move the apartment board to the kanban view"

TEST_PAGE = """<!doctype html>
<html><head><title>Student eval smoke</title>
<style>
  body { margin: 0; font-family: sans-serif; }
  #menu-items { display: none; position: absolute; left: 560px; top: 140px; }
  #menu:hover + #menu-items { display: block; }
</style></head><body>
<h1 style="position:absolute;left:100px;top:20px;margin:0">Greeting app</h1>
<p id="status" role="status" style="position:absolute;left:100px;top:60px;margin:0">waiting</p>
<input id="name" type="text" placeholder="Your name"
       style="position:absolute;left:100px;top:100px;width:200px;height:30px">
<button id="greet" style="position:absolute;left:100px;top:160px;width:200px;height:40px"
        onclick="document.getElementById('status').innerText='hello '+document.getElementById('name').value">Greet</button>
<button id="menu" style="position:absolute;left:560px;top:100px;width:120px;height:30px">Open menu</button>
<div id="menu-items">Hidden action revealed</div>
<canvas id="board" width="300" height="150" style="position:absolute;left:100px;top:300px;border:1px solid"></canvas>
<div style="height: 2200px"></div>
</body></html>"""

# Coordinates read off the absolute layout above — exactly how the student addresses the page.
NAME_FIELD = (200, 115)
GREET_BUTTON = (200, 180)
MENU_BUTTON = (620, 115)
CANVAS_FROM = (150, 340)
CANVAS_TO = (350, 420)

REPLAY_STEPS = [
    ({"type": "scroll", "x": 640, "y": 360, "scroll_x": 0, "scroll_y": 400},
     "The page is taller than the window, so I scroll down to see what else is here."),
    ({"type": "scroll", "x": 640, "y": 360, "scroll_x": 0, "scroll_y": -400},
     "Nothing useful below, so I scroll back up to the greeting form."),
    ({"type": "move", "x": MENU_BUTTON[0], "y": MENU_BUTTON[1]},
     "The menu button may hide extra options, so I hover over it to check."),
    ({"type": "click", "x": NAME_FIELD[0], "y": NAME_FIELD[1], "button": "left"},
     "The name field is empty; I click it so I can enter my name."),
    ({"type": "type", "text": "Smokey"},
     "I type my name so the greeting has something to address."),
    ({"type": "keypress", "keys": ["Enter"]},
     "I press Enter in case the form submits on its own."),
    ({"type": "drag", "path": [{"x": CANVAS_FROM[0], "y": CANVAS_FROM[1]},
                               {"x": CANVAS_TO[0], "y": CANVAS_TO[1]}]},
     "I draw a stroke across the sketch area to confirm it responds."),
    ({"type": "wait"},
     "I wait a moment to let the page settle before the final click."),
    ({"type": "click", "x": GREET_BUTTON[0], "y": GREET_BUTTON[1], "button": "left"},
     "My name is entered, so clicking Greet should produce the greeting."),
]

ROUND_TRIP_ACTIONS = [
    {"type": "click", "x": 640, "y": 360, "button": "left"},
    {"type": "click", "x": 12, "y": 7, "button": "right"},
    {"type": "double_click", "x": 100, "y": 200},
    {"type": "move", "x": 5, "y": 719},
    {"type": "type", "text": "hello world"},
    {"type": "type", "text": 'a "quoted" phrase, with punctuation!'},
    {"type": "keypress", "keys": ["Enter"]},
    {"type": "keypress", "keys": ["Control", "a"]},
    {"type": "scroll", "x": 640, "y": 360, "scroll_x": 0, "scroll_y": 500},
    {"type": "scroll", "x": 100, "y": 100, "scroll_x": -50, "scroll_y": -500},
    {"type": "drag", "path": [{"x": 10, "y": 20}, {"x": 300, "y": 400}]},
    {"type": "wait"},
    {"type": "done"},
    {"type": "screenshot"},
    {"type": "navigate", "url": "https://example.com/search?q=visa&page=2"},
    {"type": "handoff", "reason": "login"},
    {"type": "confirm", "risk_category": "purchase",
     "proposed_action": {"type": "click", "x": 200, "y": 300, "button": "left"}},
]


def check_parser_round_trip():
    """Every action type survives format -> parse unchanged, and reads as canonical."""
    seen = set()
    for action in ROUND_TRIP_ACTIONS:
        line = format_action_for_training(action)
        parsed = classify_action_line(line)
        assert parsed.action == action, f"round-trip mismatch for {line!r}: {parsed.action}"
        assert parsed.kind == "prefixed", f"{line!r} classified {parsed.kind}"
        assert parsed.deviations == (), f"canonical line reported deviations {parsed.deviations}"
        seen.add(action["type"])
    required = {"click", "double_click", "move", "type", "keypress", "scroll", "drag", "wait",
                "done", "navigate", "handoff", "confirm"}
    assert required <= seen, f"round-trip missed action types: {required - seen}"
    logger.info(f"parser round-trip OK for {len(ROUND_TRIP_ACTIONS)} canonical actions "
                f"over {len(seen)} types")


# Every accepted surface variant, per verb, against the action it must produce. The canonical form
# is generated from the action itself, so this table only lists DEVIATIONS.
GRAMMAR_MATRIX = [
    ({"type": "click", "x": 217, "y": 925, "button": "left"}, [
        "click(<ref>(217, 925)</ref>, button=left)",
        "click(<ref>(217, 925)</ref>)",
        "click((217, 925))",
        "click(217, 925)",
        "click(217, 925, left)",
        "click(217.0, 925.0)",
        "CLICK(217, 925)",
    ]),
    ({"type": "click", "x": 10, "y": 20, "button": "right"}, [
        "click(<ref>(10, 20)</ref>, button=right)",
        "click(10, 20, right)",
        "click((10, 20), button=RIGHT)",
    ]),
    ({"type": "double_click", "x": 100, "y": 200}, [
        "double_click(<ref>(100, 200)</ref>)",
        "double_click((100, 200))",
        "double_click(100, 200)",
    ]),
    ({"type": "move", "x": 5, "y": 719}, [
        "move(<ref>(5, 719)</ref>)",
        "move((5, 719))",
        "move(5, 719)",
    ]),
    ({"type": "drag", "path": [{"x": 10, "y": 20}, {"x": 300, "y": 400}]}, [
        "drag(<ref>(10, 20)</ref> -> <ref>(300, 400)</ref>)",
        "drag((10, 20) -> (300, 400))",
        "drag(10, 20 -> 300, 400)",
        "drag((10, 20), (300, 400))",
        "drag(10, 20, 300, 400)",
        "drag(<ref>(10, 20)</ref> -> 300, 400)",
        "drag(10, 20 -> <ref>(300, 400)</ref>)",
    ]),
    ({"type": "scroll", "x": 640, "y": 360, "scroll_x": 0, "scroll_y": 500}, [
        "scroll(<ref>(640, 360)</ref>, scrollX=0, scrollY=500)",
        "scroll(<ref>(640, 360)</ref>)",
        "scroll((640, 360), scrollY=500)",
        "scroll(640, 360)",
        "scroll(640, 360, 0, 500)",
        "scroll((640, 360), scroll_x=0, scroll_y=500)",
        "scroll((640, 360), dx=0, dy=500)",
    ]),
    ({"type": "scroll", "x": 100, "y": 100, "scroll_x": -50, "scroll_y": -500}, [
        "scroll(<ref>(100, 100)</ref>, scrollX=-50, scrollY=-500)",
        "scroll(100, 100, -50, -500)",
    ]),
    ({"type": "type", "text": "hello, world"}, [
        'type("hello, world")',
        "type('hello, world')",
    ]),
    ({"type": "keypress", "keys": ["Enter"]}, [
        'keypress(["Enter"])',
        "keypress(['Enter'])",
        'keypress("Enter")',
        "keypress(Enter)",
    ]),
    ({"type": "keypress", "keys": ["Control", "a"]}, [
        'keypress(["Control", "a"])',
        "keypress(Control, a)",
        "keypress(Control+a)",
    ]),
    ({"type": "wait"}, ["wait()", "wait", "wait( )"]),
    ({"type": "done"}, ["done()", "done"]),
    ({"type": "screenshot"}, ["screenshot()", "screenshot"]),
    ({"type": "navigate", "url": "https://example.com/a?q=1"}, [
        'navigate("https://example.com/a?q=1")',
        "navigate('https://example.com/a?q=1')",
        "navigate(https://example.com/a?q=1)",
        'navigate(url="https://example.com/a?q=1")',
    ]),
    ({"type": "handoff", "reason": "login"}, [
        "handoff(reason=login)",
        "handoff(login)",
        'handoff("login")',
        'handoff(reason="login")',
    ]),
    ({"type": "confirm", "risk_category": "purchase",
      "proposed_action": {"type": "click", "x": 200, "y": 300, "button": "left"}}, [
        "confirm(propose=click(<ref>(200, 300)</ref>), risk=purchase)",
        "confirm(propose=click(200, 300), risk=purchase)",
        'confirm(propose=click((200, 300)), risk="purchase")',
    ]),
]

# Shapes rejected because they have more than one reading, not because they are unfamiliar.
AMBIGUOUS_SHAPES = [
    ("type(hello world)", "an unquoted string cannot be told from an argument list"),
    ("type(a, b)", "two readings: one string containing a comma, or two arguments"),
    ("scroll(500)", "one number reads equally as an x coordinate or a scrollY delta"),
    ("click(100)", "half a point"),
    ("drag((10, 20))", "one endpoint"),
    ("keypress()", "no keys"),
    ("click()", "no coordinates"),
    ("navigate()", "no destination"),
    ("navigate(example.com)", "a scheme-less bare token could be a domain or a search phrase"),
    ("navigate(back)", "not a URL — back could be a relative path, a keyword, or prose"),
]


def check_grammar_matrix():
    """Every verb in every accepted surface variant, canonical and deviant, plus the rejections."""
    variants = 0
    for action, forms in GRAMMAR_MATRIX:
        canonical = format_action_for_training(action)
        for form in forms:
            for line in (form, f"action: {form}", f"  {form} ", f"- {form}", f"`{form}`"):
                parsed = classify_action_line(line)
                assert parsed.action == action, f"{line!r} -> {parsed.action}, expected {action}"
                assert parsed.kind in ("prefixed", "bare"), f"{line!r} classified {parsed.kind}"
                # Whitespace is not a meaningful deviation, so canonicality ignores it.
                is_canonical = "".join(line.split()) == "".join(canonical.split())
                assert bool(parsed.deviations) != is_canonical, \
                    f"{line!r} deviations={parsed.deviations} but canonical={is_canonical}"
                variants += 1

    for shape, why in AMBIGUOUS_SHAPES:
        parsed = classify_action_line(shape)
        assert parsed.action is None and parsed.kind == "malformed", \
            f"{shape!r} was accepted as {parsed.action} but is ambiguous: {why}"
    logger.info(f"grammar matrix OK: {variants} accepted spellings over "
                f"{len(GRAMMAR_MATRIX)} action shapes, {len(AMBIGUOUS_SHAPES)} ambiguous rejected")


def check_line_discipline():
    """Loosening the SHAPE of a call must not loosen the rule that the line IS a call."""
    for prose in [
        "The next logical step is to type a query into the search bar.",
        "I should click the search bar and then wait for results.",
        "I will scroll down to find the done button.",
        "I am done with this task.",
    ]:
        assert classify_action_line(prose) == (None, "ignored", ()), f"narration picked up: {prose!r}"

    for near_miss in [
        'I should type("UK visa requirements") now.',
        "Next I will click(<ref>(100, 200)</ref>) on the field.",
        "action: I will search for visa requirements",
        "First, click(217, 925) to focus it.",
    ]:
        parsed = classify_action_line(near_miss)
        assert (parsed.action, parsed.kind) == (None, "malformed"), \
            f"{near_miss!r} -> {parsed.kind}, expected malformed"
    logger.info("line discipline OK (prose with a call is malformed, narration is ignored)")


def check_live_response_shapes():
    """The three surface variants the live baseline actually emitted, end to end through a step."""
    step = parse_student_output(
        "I have clicked into the search bar. The next step is to type a query.\n</think>\n\n"
        'type("UK visa requirements for American citizens working in healthcare")')
    assert step.actions == [
        {"type": "type", "text": "UK visa requirements for American citizens working in healthcare"}
    ], step.actions
    assert step.deviations == {"missing_prefix": 1}, step.deviations
    assert step.canonical_actions == 0 and not step.parse_failed, step
    assert step.think.startswith("I have clicked into the search bar"), step.think

    bare_coords = parse_student_output("<think>t</think>\nclick(217, 925)")
    assert bare_coords.actions == [{"type": "click", "x": 217, "y": 925, "button": "left"}]
    assert set(bare_coords.deviations) == {"missing_prefix", "bare_coords", "default_kwargs"}, \
        bare_coords.deviations

    no_parens = parse_student_output("<think>t</think>\naction: wait")
    assert no_parens.actions == [{"type": "wait"}], no_parens.actions
    assert no_parens.deviations == {"missing_parens": 1}, no_parens.deviations

    mixed = parse_student_output(
        "<think>t</think>\naction: click(<ref>(10, 20)</ref>, button=left)\nwait()\nnonsense(")
    assert [a["type"] for a in mixed.actions] == ["click", "wait"], mixed.actions
    assert mixed.canonical_actions == 1, mixed.canonical_actions
    assert mixed.deviations == {"missing_prefix": 1}, mixed.deviations
    assert mixed.malformed_lines == 0, mixed

    strict = parse_student_output(step.raw, tolerant_grammar=False)
    assert strict.parse_failed and strict.malformed_lines == 1, strict
    strict_ok = parse_student_output(
        "<think>t</think>\naction: click(<ref>(10, 20)</ref>, button=left)", tolerant_grammar=False)
    assert strict_ok.canonical_actions == 1 and not strict_ok.parse_failed, strict_ok
    logger.info("live response shapes OK (bare prefix, bare coords, missing parens, strict mode)")


def check_canonical_form_rate():
    """The metric must read 100% on canonical output and be correctly fractional on mixed output."""
    canonical_only = [EpisodeRecordFactory(actions_parsed=4, actions_canonical=4, deviations={})]
    metrics = compute_metrics(canonical_only)
    assert metrics["canonical_form_rate"] == 1.0, metrics
    assert metrics["action_form_deviations"] == {}, metrics

    mixed = [EpisodeRecordFactory(actions_parsed=8, actions_canonical=2,
                                  deviations={"missing_prefix": 6, "bare_coords": 4})]
    metrics = compute_metrics(mixed)
    assert metrics["canonical_form_rate"] == 0.25, metrics
    assert metrics["action_form_deviations"] == {"missing_prefix": 6, "bare_coords": 4}, metrics
    assert metrics["actions_canonical"] == 2 and metrics["actions_parsed"] == 8, metrics
    logger.info("canonical_form_rate OK (1.0 canonical, 0.25 mixed, deviations aggregated)")


def EpisodeRecordFactory(actions_parsed, actions_canonical, deviations):
    return EpisodeRecord(task_id="t", instruction="i", start_url="https://x/", domain="d",
                         outcome="completed", model_steps=4, actions_parsed=actions_parsed,
                         actions_canonical=actions_canonical, form_deviations=deviations)


def check_parser_tolerance():
    """Multi-action steps, code fences and a missing think block all still parse."""
    parsed = parse_student_output(
        "<think>I need to search.</think>\n"
        'action: click(<ref>(100, 200)</ref>, button=left)\n'
        'action: type("laptop")\n'
        'action: keypress(["Enter"])')
    assert parsed.think == "I need to search.", parsed.think
    assert [a["type"] for a in parsed.actions] == ["click", "type", "keypress"], parsed.actions
    assert parsed.canonical_actions == 3 and parsed.deviations == {}, parsed
    assert parsed.malformed_lines == 0 and not parsed.parse_failed

    garbage = parse_student_output("I will click the search box for you.")
    assert garbage.parse_failed and garbage.actions == [], garbage
    assert garbage.malformed_lines == 0, "prose without a call is not a malformed action"

    fenced = parse_student_output("```\naction: wait()\n```")
    assert [a["type"] for a in fenced.actions] == ["wait"], fenced.actions

    no_think = parse_student_output("action: done()")
    assert no_think.think == "" and no_think.actions == [{"type": "done"}]
    logger.info("parser tolerance OK")


def check_coordinate_scaling():
    """normalized_1000 -> viewport pixels for every coordinate-bearing action, drag endpoints included."""
    assert scale_action({"type": "click", "x": 353, "y": 968, "button": "left"}, "normalized_1000") \
        == {"type": "click", "x": 452, "y": 697, "button": "left"}
    # The eval scales with train_qwen's own mapping, and the EXECUTED pixel action reads back in model space.
    assert (452, 697) == to_viewport_coords(353, 968)
    executed = scale_action({"type": "click", "x": 353, "y": 968, "button": "left"}, "normalized_1000")
    assert history_entries_for_step({"actions": [executed]}) == ["click(353, 968)"]
    executed_drag = scale_action({"type": "drag", "path": [{"x": 100, "y": 200}, {"x": 900, "y": 800}]},
                                 "normalized_1000")
    assert history_entries_for_step({"actions": [executed_drag]}) == ["drag(100,200 -> 900,800)"]
    assert outside_coord_space(1100, 500, "normalized_1000") and not outside_coord_space(1000, 1000, "normalized_1000")
    assert outside_coord_space(1000, 1000, "viewport_abs") and not outside_coord_space(1279, 719, "viewport_abs")
    assert scale_action({"type": "double_click", "x": 500, "y": 500}, "normalized_1000") \
        == {"type": "double_click", "x": 640, "y": 360}
    assert scale_action({"type": "move", "x": 0, "y": 999}, "normalized_1000") \
        == {"type": "move", "x": 0, "y": 719}
    scrolled = scale_action(
        {"type": "scroll", "x": 500, "y": 500, "scroll_x": 0, "scroll_y": 500}, "normalized_1000")
    assert scrolled == {"type": "scroll", "x": 640, "y": 360, "scroll_x": 0, "scroll_y": 500}, scrolled
    dragged = scale_action(
        {"type": "drag", "path": [{"x": 100, "y": 200}, {"x": 900, "y": 800}]}, "normalized_1000")
    assert dragged == {"type": "drag", "path": [{"x": 128, "y": 144}, {"x": 1152, "y": 576}]}, dragged

    for action in ROUND_TRIP_ACTIONS:
        assert scale_action(action, "viewport_abs") == action, f"viewport_abs altered {action}"
        scaled = scale_action(action, "normalized_1000")
        for x, y in action_coordinates(scaled):
            assert not outside_viewport(x, y), f"scaled {action} landed off-viewport at ({x}, {y})"

    assert action_coordinates({"type": "type", "text": "x"}) == []
    assert outside_viewport(353, 968) and not outside_viewport(452, 697)
    logger.info("coordinate scaling OK (normalized_1000 -> 1280x720, drag endpoints included)")


def check_prefilled_think():
    """A prefilled `<think>` leaves only the closing tag; the reasoning must still be captured."""
    parsed = parse_student_output(
        "The search bar is visible at the bottom.\n</think>\n\naction: click(<ref>(353, 968)</ref>)")
    assert parsed.think == "The search bar is visible at the bottom.", parsed.think
    assert parsed.actions == [{"type": "click", "x": 353, "y": 968, "button": "left"}], parsed.actions
    logger.info("prefilled <think> handling OK")


def check_driver_mapping():
    """Every action type the student can emit maps onto a driver /act payload."""
    for action in ROUND_TRIP_ACTIONS:
        assert action["type"] in ACTION_TO_DRIVER, f"no driver mapping for {action['type']}"
    payload = driver_payload({"type": "click", "x": 5, "y": 6, "button": "right"}, "because")
    assert payload == {"kind": "click_xy", "x": 5, "y": 6, "button": "right", "thought": "because"}, payload
    scroll = driver_payload({"type": "scroll", "x": 1, "y": 2, "scroll_x": 3, "scroll_y": 4}, "t")
    assert scroll == {"kind": "scroll_xy", "x": 1, "y": 2, "dx": 3, "dy": 4, "thought": "t"}, scroll
    typed = driver_payload({"type": "type", "text": "x"}, "")
    assert typed["kind"] == "type_text" and typed["thought"], "driver requires a non-empty thought"
    nav = driver_payload({"type": "navigate", "url": "https://x.test/a"}, "t")
    assert nav == {"kind": "navigate", "url": "https://x.test/a", "thought": "t"}, nav
    assert driver_payload({"type": "handoff", "reason": "login"}, "t") is None
    assert driver_payload({"type": "confirm", "risk_category": "purchase"}, "t") is None
    logger.info("driver mapping OK")


def _png(size=(4, 4), color=(255, 255, 255), block=None) -> bytes:
    image = Image.new("RGB", size, color)
    if block:
        image.paste((0, 0, 0), block)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _flatten_openai(content: list) -> str:
    """The ms-swift `<image>` flattening of an OpenAI content list, so it can be diffed against the golden text."""
    return "".join("<image>" if item["type"] == "image_url" else item["text"] for item in content)


def _decode_images(content: list) -> list:
    images = []
    for item in content:
        if item["type"] != "image_url":
            continue
        url = item["image_url"]["url"]
        assert url.startswith("data:image/png;base64,"), url[:40]
        images.append(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))
    return images


def check_prompt_format():
    """What the eval sends must be the sft3 layout the golden fixture pins, images inlined as PNG data URLs."""
    golden = json.loads(PROMPT_GOLDEN.read_text())
    assert golden["constants"]["PROMPT_FORMAT_VERSION"] == PROMPT_FORMAT_VERSION == "sft3"
    assert golden["constants"]["SYSTEM_PROMPT"] == SYSTEM_PROMPT
    cur, prev = _png(), _png(block=(0, 0, 2, 2))

    first = build_student_messages(GOLDEN_INSTRUCTION, [], cur)
    assert first[0] == {"role": "system", "content": SYSTEM_PROMPT}, first[0]
    assert _flatten_openai(first[1]["content"]) == golden["observations"]["no_history_step1"]["text"]
    assert [im.size for im in _decode_images(first[1]["content"])] == [(4, 4)]

    # Built as the episode loop builds it (executed PIXEL actions + observed effect), so the golden proves the round trip.
    same_url, new_url = "https://x.test/", "https://x.test/projects/new"
    click = {"type": "click", "x": 640, "y": 360, "button": "left"}
    typed = {"type": "type", "text": "hello"}
    history = (history_entries_for_step({"actions": [click]}, observed_effect(cur, cur, same_url, same_url))
               + history_entries_for_step({"actions": [click, typed]}, observed_effect(cur, prev, same_url, new_url))
               + history_entries_for_step({"actions": [click]}, observed_effect(cur, cur, same_url, new_url)))
    assert history == ["click(500, 500) [no change]", "click(500, 500)", 'type("hello") -> /projects/new',
                       "click(500, 500) [no change] -> /projects/new"], history
    later = build_student_messages(GOLDEN_INSTRUCTION, history, cur, prev)
    assert _flatten_openai(later[1]["content"]) == golden["observations"]["annotated_stepn"]["text"]
    assert [im.size for im in _decode_images(later[1]["content"])] == [PREV_IMAGE_SIZE, (4, 4)]

    windowed = [f"click({i}, {i})" for i in range(HISTORY_WINDOW + 3)]
    text = _flatten_openai(build_student_messages(GOLDEN_INSTRUCTION, windowed, cur, prev)[1]["content"])
    assert text == golden["observations"]["windowed_23_stepn"]["text"]
    logger.info("prompt format OK (sft3 golden: one image on step 1, prev+cur after, annotated history)")


def check_v2_observation():
    """An sft-v1/v2 checkpoint (and any GRPO LoRA over one) trained on ONE screenshot.

    Sending it the sft3 two-image observation measures the prompt rather than the model, so the
    format is selectable per arm exactly as the QA bench's `cua_observation` is.
    """
    cur, prev = _png(), _png(block=(0, 0, 2, 2))
    history = ["click(500, 500) [no change]"]

    v2 = build_student_messages(GOLDEN_INSTRUCTION, history, cur, prev, observation="v2")
    assert [im.size for im in _decode_images(v2[1]["content"])] == [(4, 4)], "v2 must send exactly one image"
    assert "Previous screenshot" not in _flatten_openai(v2[1]["content"])

    v3 = build_student_messages(GOLDEN_INSTRUCTION, history, cur, prev, observation="v3")
    assert [im.size for im in _decode_images(v3[1]["content"])] == [PREV_IMAGE_SIZE, (4, 4)]

    # The history line, and therefore its effect annotations, is identical in both formats.
    assert history[0] in _flatten_openai(v2[1]["content"])
    logger.info("v2/v3 observation switch OK (one image vs prev+cur, same history)")


def check_png_forced():
    """Effect annotations compare screenshots pixel-wise, so a lossy format is a config error, not a silent mis-annotation."""
    assert EvalConfig().screenshot_format == "png"
    assert EvalConfig().coord_space == "normalized_1000", "stock Qwen and an sft3 checkpoint both answer normalized"
    try:
        EvalConfig(screenshot_format="jpeg")
    except ValueError as e:
        assert "png" in str(e), e
    else:
        raise AssertionError("EvalConfig(screenshot_format='jpeg') must be refused")
    logger.info("png forced OK")


def check_block_detection():
    assert detect_block("https://shop.example/x", "Please complete the CAPTCHA to continue") == "captcha"
    assert detect_block("https://x/y", "Checking your browser before accessing") == "cloudflare"
    assert detect_block("https://x/y", "Pardon Our Interruption") == "waf_block"
    assert detect_block("https://x/y", "## Interactive elements\n[1] button \"Add to cart\"") is None

    # Every wording below was scored as a MODEL failure in the 2026-08-31 runs.
    for snapshot, expected in (
            ("This page is not available in your area", "geo_block"),
            ("Your request is coming from a country which we do not currently support", "geo_block"),
            ("ERROR: The request could not be satisfied. Request blocked.", "waf_block"),
            ("Generated by cloudfront (CloudFront)", "waf_block"),
            ("Access temporarily restricted", "waf_block"),
            ("Performing security verification...", "cloudflare"),
            ("Just a moment...", "cloudflare"),
            ("Verifying you are human. This may take a few seconds.", "human_verification"),
            ("We're sorry, there seems to be a problem with our website", "site_error"),
            ("This site is temporarily unavailable", "site_error"),
    ):
        assert detect_block("https://x/y", snapshot) == expected, (snapshot, detect_block("https://x/y", snapshot))

    # Ordinary page copy that must NOT drop an episode out of the success denominator.
    for snapshot in (
            "This item is temporarily unavailable in your size",
            "Sorry, this product is not available in this color",
            "Country: United States. Region blocked list: none",
            "Read our article: how bot detection works",
    ):
        assert detect_block("https://x/y", snapshot) is None, snapshot
    logger.info("block detection OK")


def check_browser_backends():
    """Every egress backend must satisfy the surface the driver calls, or it fails mid-run."""
    surface = ["read_file", "write_file", "start_background", "wait_for_port", "http_json",
               "exec_command", "copy_out", "put_bytes", "start", "stop", "remaining_seconds"]
    for name, factory in BROWSER_BACKENDS.items():
        cfg = EvalConfig(browser_backend=name, browser_host="example.test:2222")
        built = factory(cfg)
        missing = [m for m in surface if not hasattr(built, m)]
        assert not missing, f"{name} backend is missing {missing}"
        assert isinstance(built, ShellSandbox), f"{name} backend is not a ShellSandbox"

    # Egress is a parity key: a Lviv run and a US-residential run see different sites, so they
    # must not silently compare as the same harness.
    source = TASK_SOURCES["online_mind2web"]
    local = harness_fingerprint(EvalConfig(browser_backend="docker", egress_note="UA/Lviv"), source, {})
    rented = harness_fingerprint(
        EvalConfig(browser_backend="vast_residential", egress_note="US/NYC"), source, {})
    assert local["egress"] != rented["egress"], "egress must differ in the fingerprint"
    assert local["browser_backend"] != rented["browser_backend"]
    logger.info(f"browser backends OK ({', '.join(sorted(BROWSER_BACKENDS))}); egress is a parity key")


def check_report_rendering():
    """Both renderers must run on a synthetic payload, or a crash surfaces only after a full run."""
    records = [
        EpisodeRecord(task_id="a", instruction="i", start_url="https://a/", domain="shop",
                      outcome="completed", verified_success=True, model_steps=8, done_emitted=True,
                      emitted_coords=8, actions_parsed=10, actions_canonical=6,
                      form_deviations={"missing_prefix": 4}),
        EpisodeRecord(task_id="b", instruction="i", start_url="https://b/", domain="travel",
                      outcome="site_blocked", block_reason="captcha", model_steps=2),
        EpisodeRecord(task_id="c", instruction="i", start_url="https://c/", domain="shop",
                      outcome="parse_failure_abort", model_steps=3, parse_failure_steps=3),
    ]
    cfg = EvalConfig(label="before", model="qwen", endpoint="http://x/v1")
    results = {
        "run_id": "before_x", "label": "before", "started_at": "t0", "finished_at": "t1",
        "config": asdict(cfg), "task_provenance": {},
        "harness": harness_fingerprint(cfg, TASK_SOURCES["online_mind2web"], {}),
        "aborted_reason": "", "episodes": [asdict(r) for r in records],
        "metrics": compute_metrics(records),
    }
    metrics = results["metrics"]
    assert metrics["episodes"] == 3 and metrics["scored_episodes"] == 2, metrics
    assert metrics["verified_success_rate"] == 0.5, metrics
    assert metrics["site_blocked_rate"] == round(1 / 3, 4), metrics
    assert metrics["parse_failure_rate"] == round(3 / 13, 4), metrics
    report = render_report(results)
    assert "verified task success: 50.0%" in report, report[:400]
    assert "site-blocked 33.3%" in report, report[:400]
    assert "canonical action form | 60.0% (6/10 actions)" in report, report
    assert "form deviations | missing_prefix 4" in report, report
    assert metrics["canonical_form_rate"] == 0.6, metrics
    assert results["harness"]["action_grammar"] == "tolerant", results["harness"]
    assert "action_grammar" not in EXPECTED_TO_DIFFER_KEYS, \
        "grammar tolerance is a harness property — a parity key, not an expected difference"
    assert "action_grammar" not in PARITY_EXEMPT_KEYS, PARITY_EXEMPT_KEYS
    # The observation layout is a parity key: a run prompted in sft2 is not comparable to an sft3 run.
    harness = results["harness"]
    assert harness["prompt_format"] == PROMPT_FORMAT_VERSION, harness
    assert harness["prev_image_size"] == f"{PREV_IMAGE_SIZE[0]}x{PREV_IMAGE_SIZE[1]}", harness
    assert harness["effect_annotations"] is True and harness["type_mode"] == "replace", harness
    for key in ("prompt_format", "prev_image_size", "effect_annotations", "type_mode"):
        assert key not in PARITY_EXEMPT_KEYS and key not in EXPECTED_TO_DIFFER_KEYS, key
    assert f"| prompt format | `{PROMPT_FORMAT_VERSION}` |" in report, report
    assert "| previous-screenshot size |" in report and "| effect annotations |" in report, report
    assert "| type mode |" in report, report
    after = json.loads(json.dumps(results))
    after["label"] = "after"
    after["metrics"]["verified_success_rate"] = 0.75
    comparison = render_comparison([results, after])
    assert "+25.0 pp" in comparison, comparison
    assert "All compared runs share" in comparison, comparison

    # A stock-vs-sft-v1/v2 pair differs in coord_space BY DESIGN; that must read as an annotated
    # expected difference, never as a parity violation that discredits the delta.
    results["harness"]["coord_space"] = "viewport_abs"
    comparison = render_comparison([results, after])
    assert "NOT the same harness" not in comparison, comparison
    assert "Expected model-side differences" in comparison, comparison
    assert "coord_space" in comparison, comparison

    after["harness"]["temperature"] = 0.7
    comparison = render_comparison([results, after])
    assert "NOT the same harness" in comparison, comparison
    assert "temperature" in comparison, comparison

    after["harness"]["temperature"] = results["harness"]["temperature"]
    after["harness"]["prompt_format"] = "sft2"
    comparison = render_comparison([results, after])
    assert "NOT the same harness" in comparison and "prompt_format" in comparison, comparison
    logger.info("report + comparison rendering OK (coord_space annotated, real drift flagged)")


def check_coord_space_diagnosis():
    """The run-level 'is this even measuring the model?' banner."""
    suspect = coord_space_diagnosis(
        {"coord_space": "viewport_abs"},
        {"emitted_coords": 36, "emitted_out_of_viewport": 36, "emitted_out_of_viewport_rate": 1.0})
    assert "COORDINATE SPACE SUSPECT" in suspect and "--coord-space normalized_1000" in suspect, suspect
    # The inverse: an sft-v1/v2 checkpoint answering in pixels while scored as normalized_1000.
    inverse = coord_space_diagnosis(
        {"coord_space": "normalized_1000"},
        {"emitted_coords": 36, "emitted_out_of_viewport": 36, "emitted_out_of_viewport_rate": 1.0})
    assert "COORDINATE SPACE SUSPECT" in inverse and "--coord-space viewport_abs" in inverse, inverse
    assert "sft-v1/v2" in inverse, inverse
    assert not coord_space_diagnosis(
        {"coord_space": "viewport_abs"},
        {"emitted_coords": 40, "emitted_out_of_viewport": 1, "emitted_out_of_viewport_rate": 0.025})
    assert not coord_space_diagnosis(
        {"coord_space": "normalized_1000"},
        {"emitted_coords": 40, "emitted_out_of_viewport": 1, "emitted_out_of_viewport_rate": 0.025})
    logger.info("coordinate-space diagnosis OK (both directions)")


def _write_mock_trajectory(path: Path, start_url: str) -> Path:
    trajectory = {
        "task_id": "smoke", "instruction": SMOKE_INSTRUCTION, "start_url": start_url,
        "steps": [{"step": i + 1, "screenshot": "", "thought": thought, "actions": [action]}
                  for i, (action, thought) in enumerate(REPLAY_STEPS)],
        "total_steps": len(REPLAY_STEPS), "success": True,
    }
    path.write_text(json.dumps(trajectory, indent=2))
    return path


SMOKE_INSTRUCTION = "Greet yourself as Smokey on this page, ending with the greeting visible on screen."


class FixedActionStudent:
    """Emits the same response every step — used to drive an episode into the stuck abort."""

    def __init__(self, response: str):
        self.response = response

    def complete(self, messages: list) -> str:
        return self.response


def _to_normalized_1000(action: dict) -> dict:
    to_norm = lambda x, y: (round(x / 1280 * 1000), round(y / 720 * 1000))
    if action.get("type") == "drag":
        return {**action, "path": [dict(zip(("x", "y"), to_norm(p["x"], p["y"])))
                                   for p in action["path"]]}
    if "x" not in action:
        return action
    x, y = to_norm(action["x"], action["y"])
    return {**action, "x": x, "y": y}


class RecordingStudent:
    """Keeps every outgoing message list so the observation layout the runner built can be asserted on."""

    def __init__(self, inner):
        self.inner = inner
        self.seen: list = []

    def complete(self, messages: list) -> str:
        self.seen.append(messages)
        return self.inner.complete(messages)


# Model-space (normalized_1000) actions chosen for their OBSERVABLE effect on the static test page.
EFFECT_STEPS = [
    ({"type": "wait"}, "Let the page settle first."),
    ({"type": "scroll", "x": 500, "y": 500, "scroll_x": 0, "scroll_y": 400}, "Scroll down to see more."),
    ({"type": "navigate", "url": "http://127.0.0.1:3000/index.html"}, "Open the page by its file name."),
    ({"type": "wait"}, "Give it a moment."),
]
EFFECT_HISTORY = ["wait [no change]", "scroll(500, 500)",
                  "navigate(http://127.0.0.1:3000/index.html) -> /index.html", "wait [no change]"]


async def _run_effect_episode(browser, tmp_dir: Path, start_url: str):
    """History entries come from EXECUTED actions with the effect the next screenshot showed; two images from step 2."""
    trajectory = {
        "task_id": "effect", "instruction": SMOKE_INSTRUCTION, "start_url": start_url,
        "steps": [{"step": i + 1, "screenshot": "", "thought": thought, "actions": [action]}
                  for i, (action, thought) in enumerate(EFFECT_STEPS)],
        "total_steps": len(EFFECT_STEPS), "success": True,
    }
    cfg = EvalConfig(task_source="held_out", label="effect", student="mock",
                     max_steps=len(EFFECT_STEPS) + 3, judge=False, resume=False, coord_space="normalized_1000")
    task = EvalTask(task_id="effect", instruction=SMOKE_INSTRUCTION, start_url=start_url,
                    domain="smoke", reference_length=len(EFFECT_STEPS), source="smoke")
    student = RecordingStudent(ReplayStudent(trajectory))
    record = await run_episode(browser, student, task, cfg, tmp_dir / "episodes" / "task_effect", cfg.max_steps)
    logger.info(f"effect episode: outcome={record.outcome} executed={record.actions_executed} "
                f"driver_errors={record.driver_errors}")
    assert record.outcome == "completed", f"expected completed, got {record.outcome}: {record.error}"
    assert record.actions_executed == len(EFFECT_STEPS), record.actions_executed
    assert len(student.seen) == len(EFFECT_STEPS) + 1, len(student.seen)

    for step_index, messages in enumerate(student.seen):
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}, messages[0]
        content = messages[1]["content"]
        sizes = [im.size for im in _decode_images(content)]
        expected_sizes = [(1280, 720)] if step_index == 0 else [PREV_IMAGE_SIZE, (1280, 720)]
        assert sizes == expected_sizes, f"step {step_index + 1}: image sizes {sizes}"
        text = _flatten_openai(content)
        assert text.count("<image>") == len(expected_sizes), text
        assert ("Previous screenshot" in text) == (step_index > 0), text
        expected = EFFECT_HISTORY[:step_index]
        history_line = f"Previous actions: [{', '.join(expected)}]" if expected else ""
        assert history_line in text, f"step {step_index + 1}: expected {history_line!r} in {text!r}"
    for item in student.seen[-1][1]["content"]:
        shown = item["text"] if item["type"] == "text" else item["image_url"]["url"][:48] + "..."
        logger.info(f"step {len(student.seen)} outgoing {item['type']}: {shown!r}")
    logger.info("effect episode OK (no-op step annotated [no change], path change annotated, prev+cur images)")


async def _run_stuck_episode(browser, tmp_dir: Path, start_url: str):
    """A model answering in the wrong space must be abandoned in a few steps, not a full cap."""
    cfg = EvalConfig(task_source="held_out", label="stuck", student="mock", max_steps=30,
                     judge=False, resume=False, coord_space="viewport_abs",
                     max_unproductive_steps=5)
    task = EvalTask(task_id="stuck", instruction=SMOKE_INSTRUCTION, start_url=start_url,
                    domain="smoke", reference_length=9, source="smoke")
    student = FixedActionStudent(
        "The search bar is visible at the bottom of the screen.\n</think>\n\n"
        "action: click(<ref>(353, 968)</ref>)")
    record = await run_episode(browser, student, task, cfg,
                               tmp_dir / "episodes" / "task_stuck", cfg.max_steps)
    logger.info(f"stuck episode: outcome={record.outcome} steps={record.model_steps} "
                f"out_of_bounds={record.out_of_bounds}")
    assert record.outcome == "stuck", f"expected stuck, got {record.outcome}"
    assert record.model_steps == 5, f"aborted after {record.model_steps} steps, expected 5"
    assert record.actions_executed == 0 and record.out_of_bounds == 5, record
    assert record.think_steps == 5, "prefilled <think> reasoning was not captured"
    assert "--coord-space normalized_1000" in record.coord_space_warning, record.coord_space_warning


async def _run_normalized_episode(browser, tmp_dir: Path, start_url: str) -> Path:
    """The SAME trajectory expressed in normalized_1000 must execute identically."""
    mock_path = tmp_dir / "mock_normalized.json"
    mock_path.write_text(json.dumps({
        "task_id": "normalized", "instruction": SMOKE_INSTRUCTION, "start_url": start_url,
        "steps": [{"step": i + 1, "screenshot": "", "thought": thought,
                   "actions": [_to_normalized_1000(action)]}
                  for i, (action, thought) in enumerate(REPLAY_STEPS)],
        "total_steps": len(REPLAY_STEPS), "success": True,
    }, indent=2))
    cfg = EvalConfig(task_source="held_out", label="normalized", student="mock",
                     mock_trajectory=str(mock_path), max_steps=len(REPLAY_STEPS) + 3,
                     judge=False, resume=False, coord_space="normalized_1000")
    task = EvalTask(task_id="normalized", instruction=SMOKE_INSTRUCTION, start_url=start_url,
                    domain="smoke", reference_length=len(REPLAY_STEPS), source="smoke")
    episode_dir = tmp_dir / "episodes" / "task_normalized"
    record = await run_episode(browser, ReplayStudent(json.loads(mock_path.read_text())), task,
                               cfg, episode_dir, cfg.max_steps)
    logger.info(f"normalized episode: outcome={record.outcome} executed={record.actions_executed} "
                f"driver_errors={record.driver_errors}")
    assert record.outcome == "completed", f"expected completed, got {record.outcome}: {record.error}"
    assert record.actions_executed == len(REPLAY_STEPS), record.actions_executed
    assert record.driver_errors == 0 and record.out_of_bounds == 0, record
    assert not record.coord_space_warning, record.coord_space_warning
    return episode_dir


async def run_episode_smoke(judge: bool) -> bool:
    """Layer 2 (+3): drive the scripted trajectory through the real runner, driver and recorder."""
    with tempfile.TemporaryDirectory(prefix="student_eval_smoke_") as tmp:
        tmp_dir = Path(tmp)
        start_url = "http://127.0.0.1:3000/"
        mock_path = _write_mock_trajectory(tmp_dir / "mock_trajectory.json", start_url)
        # REPLAY_STEPS are pixel coordinates read off the page layout, so this replay is a viewport_abs student.
        cfg = EvalConfig(
            task_source="held_out", label="smoke", student="mock", coord_space="viewport_abs",
            mock_trajectory=str(mock_path), max_steps=len(REPLAY_STEPS) + 3,
            judge=judge, judge_mid_screenshots=2, episode_timeout_s=600, resume=False,
        )
        task = EvalTask(task_id="smoke", instruction=SMOKE_INSTRUCTION, start_url=start_url,
                        website="127.0.0.1:3000", domain="smoke",
                        reference_length=len(REPLAY_STEPS), source="smoke")

        browser = EvalBrowser(cfg, "student_eval_smoke")
        try:
            await browser.ensure()
            browser.sandbox.exec_command("mkdir -p /workspace/site")
            browser.sandbox.write_file("/workspace/site/index.html", TEST_PAGE)
            browser.sandbox.start_background(
                "cd /workspace/site && python3 -m http.server 3000", "/workspace/site.log")
            assert browser.sandbox.wait_for_port(3000, 30), "test page server did not start"

            episode_dir = tmp_dir / "episodes" / "task_smoke"
            student = ReplayStudent(json.loads(mock_path.read_text()))
            record = await run_episode(browser, student, task, cfg, episode_dir,
                                       cfg.max_steps)
            normalized_dir = await _run_normalized_episode(browser, tmp_dir, start_url)
            await _run_effect_episode(browser, tmp_dir, start_url)
            await _run_stuck_episode(browser, tmp_dir, start_url)
        finally:
            await browser.stop()

        normalized = json.loads((normalized_dir / "trajectory.json").read_text())
        for step in normalized["steps"]:
            for executed in step["actions"]:
                if "x" in executed:
                    assert not outside_viewport(executed["x"], executed["y"]), \
                        f"trajectory.json must record EXECUTED pixel coords, got {executed}"

        logger.info(f"episode outcome={record.outcome} steps={record.model_steps} "
                    f"executed={record.actions_executed} driver_errors={record.driver_errors}")
        assert record.outcome == "completed", f"expected completed, got {record.outcome}: {record.error}"
        assert record.done_emitted, "the replay student never emitted done()"
        assert record.parse_failure_steps == 0, f"{record.parse_failure_steps} parse failures on recorded actions"
        assert record.malformed_lines == 0, f"{record.malformed_lines} malformed action lines"
        assert record.driver_errors == 0, f"driver refused {record.driver_errors} actions: {record.error}"
        assert record.actions_executed == len(REPLAY_STEPS), \
            f"executed {record.actions_executed} of {len(REPLAY_STEPS)} actions"
        # A replay executed in the wrong coordinate space still "completes"; only the page outcome exposes it.
        assert "hello Smokey" in record.final_snapshot, \
            f"the greeting never landed — a mis-scaled replay? final page text: {record.final_snapshot[:300]!r}"

        traj = json.loads((episode_dir / "trajectory.json").read_text())
        assert traj["total_steps"] == len(REPLAY_STEPS) + 1, traj["total_steps"]
        recorded_kinds = {a["type"] for s in traj["steps"] for a in s["actions"]}
        assert {"click", "type", "keypress", "scroll", "move", "drag", "wait", "done"} <= recorded_kinds, \
            f"missing recorded action kinds: {recorded_kinds}"
        for step in traj["steps"]:
            assert (episode_dir / step["screenshot"]).exists(), f"missing {step['screenshot']}"
            assert step["thought"].strip(), f"step {step['step']} recorded no thought"
            for action in step["actions"]:
                if "x" in action:
                    assert 0 <= action["x"] < 1280 and 0 <= action["y"] < 720, f"bad coords: {action}"
        assert (episode_dir / "final_screenshot.png").exists(), "no final screenshot for the judge"

        if not judge:
            logger.info("episode smoke OK (judging skipped)")
            return True

        await judge_episode(record, cfg)
        logger.info(f"judge: success={record.verified_success} efficiency={record.efficiency} "
                    f"reasoning={record.judge_reasoning}")
        assert record.verified_success, \
            f"the replayed episode did not verify: {record.judge_reasoning or 'no verdict'}"
        logger.info("episode + judge smoke OK")
        return True


PARSER_CHECKS = [
    check_parser_round_trip,
    check_grammar_matrix,
    check_line_discipline,
    check_live_response_shapes,
    check_canonical_form_rate,
    check_parser_tolerance,
    check_prefilled_think,
    check_coordinate_scaling,
    check_driver_mapping,
    check_prompt_format,
    check_v2_observation,
    check_png_forced,
    check_block_detection,
    check_browser_backends,
    check_coord_space_diagnosis,
    check_report_rendering,
]


def main():
    parser = argparse.ArgumentParser(description="Self-test for the student eval harness")
    parser.add_argument("--parser-only", action="store_true",
                        help="run only the pure checks (no Docker, no Claude)")
    parser.add_argument("--no-judge", action="store_true",
                        help="run the replay episode but skip verification (no Claude calls)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for check in PARSER_CHECKS:
        check()
    if args.parser_only:
        logger.info("Student eval PARSER checks PASSED")
        return 0

    ok = asyncio.run(run_episode_smoke(judge=not args.no_judge))
    logger.info("Student eval smoke test PASSED" if ok else "Student eval smoke test FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        logger.error(f"Student eval smoke test FAILED: {e}")
        sys.exit(1)
