"""Self-test for the student eval harness: pure checks, a mock replay episode, and a real judge verdict."""
import asyncio
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from repo_pipeline.student_eval import (
    EXPECTED_TO_DIFFER_KEYS,
    PARITY_EXEMPT_KEYS,
    ACTION_TO_DRIVER,
    EvalBrowser,
    EvalConfig,
    ReplayStudent,
    action_coordinates,
    build_user_text,
    coord_space_diagnosis,
    detect_block,
    driver_payload,
    judge_episode,
    outside_viewport,
    classify_action_line,
    parse_action_line,
    parse_student_output,
    render_comparison,
    render_report,
    run_episode,
    scale_action,
)
from repo_pipeline.eval_task_sources import EvalTask
from train_qwen import HISTORY_WINDOW, format_action_for_training

logger = logging.getLogger(__name__)

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
]


def check_parser_round_trip():
    """Every action type survives format -> parse unchanged, with and without the `action:` prefix."""
    seen = set()
    for action in ROUND_TRIP_ACTIONS:
        line = format_action_for_training(action)
        assert line.startswith("action: "), line
        bare = line[len("action: "):]

        parsed, kind = classify_action_line(line)
        assert parsed == action, f"round-trip mismatch for {line!r}: {parsed} != {action}"
        assert kind == "prefixed", f"{line!r} classified {kind}"

        parsed_bare, kind_bare = classify_action_line(bare)
        assert parsed_bare == action, f"bare round-trip mismatch for {bare!r}: {parsed_bare}"
        assert kind_bare == "bare", f"{bare!r} classified {kind_bare}"

        assert classify_action_line(f"  {bare}  ")[1] == "bare", "surrounding whitespace must be ok"
        assert classify_action_line(f"- {bare}")[1] == "bare", "a bullet must be ok"
        seen.add(action["type"])
    required = {"click", "double_click", "move", "type", "keypress", "scroll", "drag", "wait", "done"}
    assert required <= seen, f"round-trip missed action types: {required - seen}"
    logger.info(f"parser round-trip OK for {len(ROUND_TRIP_ACTIONS)} actions over {len(seen)} types, "
                f"prefixed and bare")


def check_bare_action_discipline():
    """Dropping the `action:` prefix is tolerated; everything else about the grammar stays strict."""
    for prose in [
        "The next logical step is to type a query into the search bar.",
        "I should click the search bar and then wait for results.",
        "I will scroll down to find the done button.",
    ]:
        assert classify_action_line(prose) == (None, "ignored"), f"narration picked up: {prose!r}"

    for near_miss in [
        'I should type("UK visa requirements") now.',
        "Next I will click(<ref>(100, 200)</ref>) on the field.",
        "action: I will search for visa requirements",
        "action: click(100, 200)",
    ]:
        action, kind = classify_action_line(near_miss)
        assert (action, kind) == (None, "malformed"), f"{near_miss!r} -> {kind}, expected malformed"

    # The exact shape the live baseline emitted: a complete, prefix-less action after </think>.
    step = parse_student_output(
        "I have clicked into the search bar. The next step is to type a query.\n</think>\n\n"
        'type("UK visa requirements for American citizens working in healthcare")')
    assert step.actions == [
        {"type": "type", "text": "UK visa requirements for American citizens working in healthcare"}
    ], step.actions
    assert step.bare_actions == 1 and step.malformed_lines == 0 and not step.parse_failed, step
    assert step.think.startswith("I have clicked into the search bar"), step.think

    strict = parse_student_output(step.raw, accept_bare_actions=False)
    assert strict.parse_failed and strict.malformed_lines == 1, strict
    assert strict.bare_actions == 0, strict

    mixed = parse_student_output(
        "<think>t</think>\naction: click(<ref>(10, 20)</ref>, button=left)\nwait()\nnonsense(")
    assert [a["type"] for a in mixed.actions] == ["click", "wait"], mixed.actions
    assert mixed.bare_actions == 1 and mixed.malformed_lines == 0, mixed
    logger.info("bare-action discipline OK (prefix optional, prose rejected, strict mode available)")


def check_parser_tolerance():
    """Structural tolerance only: prefixes, fences and bullets are stripped, action content never rewritten."""
    parsed = parse_student_output(
        "<think>I need to search.</think>\n"
        'action: click(<ref>(100, 200)</ref>, button=left)\n'
        'action: type("laptop")\n'
        'action: keypress(["Enter"])'
    )
    assert parsed.think == "I need to search.", parsed.think
    assert [a["type"] for a in parsed.actions] == ["click", "type", "keypress"], parsed.actions
    assert parsed.malformed_lines == 0 and not parsed.parse_failed

    garbage = parse_student_output("I will click the search box for you.")
    assert garbage.parse_failed and garbage.actions == [], garbage
    assert garbage.malformed_lines == 0, "prose without an action: line is not a malformed action"

    broken = parse_student_output("<think>t</think>\naction: click(100, 200)\naction: done()")
    assert broken.malformed_lines == 1, broken
    assert [a["type"] for a in broken.actions] == ["done"], broken.actions
    assert not broken.parse_failed, "a partially-parsed step keeps its good actions"

    fenced = parse_student_output("```\naction: wait()\n```")
    assert [a["type"] for a in fenced.actions] == ["wait"], fenced.actions

    empty_think = parse_student_output("action: done()")
    assert empty_think.think == "" and empty_think.actions == [{"type": "done"}]
    logger.info("parser tolerance OK")


def check_coordinate_scaling():
    """normalized_1000 -> viewport pixels for every coordinate-bearing action, drag endpoints included."""
    assert scale_action({"type": "click", "x": 353, "y": 968, "button": "left"}, "normalized_1000") \
        == {"type": "click", "x": 452, "y": 697, "button": "left"}
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
    logger.info("driver mapping OK")


def check_prompt_format():
    """The user text must match train_qwen's per-step example, history window and omission prefix included."""
    assert build_user_text("Buy milk", []) == "Task: Buy milk\nWhat action should you take?"
    history = [f"click({i}, {i})" for i in range(HISTORY_WINDOW + 3)]
    text = build_user_text("Buy milk", history)
    assert "... 3 earlier actions, " in text, text
    assert text.count("click(") == HISTORY_WINDOW, text
    assert text.endswith("\nWhat action should you take?"), text
    logger.info("prompt format OK")


def check_block_detection():
    assert detect_block("https://shop.example/x", "Please complete the CAPTCHA to continue") == "captcha"
    assert detect_block("https://x/y", "Checking your browser before accessing") == "cloudflare"
    assert detect_block("https://x/y", "Pardon Our Interruption") == "waf_block"
    assert detect_block("https://x/y", "## Interactive elements\n[1] button \"Add to cart\"") is None
    logger.info("block detection OK")


def check_report_rendering():
    """Both renderers must run on a synthetic payload, or a crash surfaces only after a full run."""
    from repo_pipeline.student_eval import EpisodeRecord, compute_metrics, harness_fingerprint
    from repo_pipeline.eval_task_sources import TASK_SOURCES
    from dataclasses import asdict

    records = [
        EpisodeRecord(task_id="a", instruction="i", start_url="https://a/", domain="shop",
                      outcome="completed", verified_success=True, model_steps=8, done_emitted=True,
                      emitted_coords=8, actions_parsed=10, actions_missing_prefix=4),
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
    assert "prefix adherence | 60.0% (4/10 actions omitted the prefix)" in report, report
    assert metrics["prefix_adherence_rate"] == 0.6, metrics
    assert results["harness"]["accepts_bare_actions"] is True, results["harness"]
    assert "accepts_bare_actions" not in EXPECTED_TO_DIFFER_KEYS, \
        "prefix tolerance is a harness property — it must be a parity key, not an expected difference"
    assert "accepts_bare_actions" not in PARITY_EXEMPT_KEYS, PARITY_EXEMPT_KEYS
    after = json.loads(json.dumps(results))
    after["label"] = "after"
    after["metrics"]["verified_success_rate"] = 0.75
    comparison = render_comparison([results, after])
    assert "+25.0 pp" in comparison, comparison
    assert "All compared runs share" in comparison, comparison

    # A stock-vs-finetuned pair differs in coord_space BY DESIGN; that must read as an annotated
    # expected difference, never as a parity violation that discredits the delta.
    results["harness"]["coord_space"] = "normalized_1000"
    comparison = render_comparison([results, after])
    assert "NOT the same harness" not in comparison, comparison
    assert "Expected model-side differences" in comparison, comparison
    assert "coord_space" in comparison, comparison

    after["harness"]["temperature"] = 0.7
    comparison = render_comparison([results, after])
    assert "NOT the same harness" in comparison, comparison
    assert "temperature" in comparison, comparison
    logger.info("report + comparison rendering OK (coord_space annotated, real drift flagged)")


def check_coord_space_diagnosis():
    """The run-level 'is this even measuring the model?' banner."""
    suspect = coord_space_diagnosis(
        {"coord_space": "viewport_abs"},
        {"emitted_coords": 36, "emitted_out_of_viewport": 36, "emitted_out_of_viewport_rate": 1.0})
    assert "COORDINATE SPACE SUSPECT" in suspect and "--coord-space normalized_1000" in suspect, suspect
    assert not coord_space_diagnosis(
        {"coord_space": "normalized_1000"},
        {"emitted_coords": 36, "emitted_out_of_viewport": 36, "emitted_out_of_viewport_rate": 1.0})
    assert not coord_space_diagnosis(
        {"coord_space": "viewport_abs"},
        {"emitted_coords": 40, "emitted_out_of_viewport": 1, "emitted_out_of_viewport_rate": 0.025})
    logger.info("coordinate-space diagnosis OK")


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
        cfg = EvalConfig(
            task_source="held_out", label="smoke", student="mock",
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
    check_parser_tolerance,
    check_bare_action_discipline,
    check_prefilled_think,
    check_coordinate_scaling,
    check_driver_mapping,
    check_prompt_format,
    check_block_detection,
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
