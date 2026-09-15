"""Scripted (no-LLM, no-network) check of the instruction style pass: qa_verify selection, cap, idempotency and writes."""

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

from cua.pipeline import instruction_style_pass
from cua.pipeline.instruction_style_pass import STYLE_MODES, find_targets, parse_json_instruction, run_pass

logger = logging.getLogger(__name__)

QA_MODE = STYLE_MODES["qa_verify"]
GOAL_MODE = STYLE_MODES["goal"]


def _step(index: int, thought: str, actions: list[dict]) -> dict:
    return {"step": index, "screenshot": f"step_{index:03d}_screenshot.png", "thought": thought, "actions": actions}


RECHECK_STEPS = [
    _step(
        1,
        "I open the notification settings to find the Email toggle.",
        [{"type": "click", "x": 100, "y": 200, "button": "left"}],
    ),
    _step(
        2, "I uncheck Email notifications and save the form.", [{"type": "click", "x": 120, "y": 300, "button": "left"}]
    ),
    _step(
        3,
        "I'll reload the settings page to confirm the change persisted.",
        [{"type": "navigate", "url": "http://app.test/settings"}],
    ),
    _step(4, "The settings page now shows Email notifications unchecked, so the change persisted.", [{"type": "done"}]),
]
PLAIN_STEPS = [
    _step(1, "I open the projects list.", [{"type": "click", "x": 100, "y": 200, "button": "left"}]),
    _step(2, "I create the project named Roadmap.", [{"type": "click", "x": 130, "y": 260, "button": "left"}]),
    _step(3, "I've finished the task.", [{"type": "done"}]),
]

# Twelve trainable dirs: cap 25% -> 3, one already tagged -> budget 2 of the three untagged re-check candidates.
CORPUS = [
    ("task_recheck_a", RECHECK_STEPS, True, {}),
    ("task_recheck_b", RECHECK_STEPS, True, {}),
    ("task_recheck_c", RECHECK_STEPS, True, {}),
    ("task_recheck_wall", RECHECK_STEPS, True, {"wall_demo": True}),
    ("task_recheck_tagged", RECHECK_STEPS, True, {"instruction_style": "qa_verify"}),
    ("task_recheck_unverified", RECHECK_STEPS, False, {}),
] + [(f"task_plain_{i}", PLAIN_STEPS, True, {}) for i in range(7)]
EXPECTED_SELECTION = ["task_recheck_a", "task_recheck_b"]
ORIGINAL = "Turn off email notifications in the settings and make sure the change sticks."
PROCEDURE = "Open the settings page, click Notifications, scroll to Email and then select off."
REWRITE = "Unchecking Email notifications in Settings persists after save and reload. Verify the box stays unchecked when the page is reopened."


def _write_corpus(root: Path) -> None:
    for name, steps, success, meta in CORPUS:
        traj_dir = root / "repo_smoke" / name
        traj_dir.mkdir(parents=True)
        instruction = PROCEDURE if name == "task_plain_0" else ORIGINAL
        trajectory = {
            "task_id": name,
            "instruction": instruction,
            "start_url": "http://app.test/",
            "steps": steps,
            "total_steps": len(steps),
            "success": success,
        }
        (traj_dir / "trajectory.json").write_text(json.dumps(trajectory))
        (traj_dir / "metadata.json").write_text(
            json.dumps({"capture_source": "agent", "instruction": instruction, **meta})
        )
        for step in steps:
            (traj_dir / step["screenshot"]).write_bytes(b"")


def _stub_agent(prompts: list[str], reply: str):
    async def run_agent(prompt, options, **kwargs):
        prompts.append(prompt)
        return reply

    return run_agent


def _names(dirs: list[Path]) -> list[str]:
    return [d.name for d in dirs]


def _read(traj_dir: Path) -> tuple[dict, dict]:
    return (
        json.loads((traj_dir / "trajectory.json").read_text()),
        json.loads((traj_dir / "metadata.json").read_text()),
    )


def check_parse(failures: list[str]) -> None:
    cases = [
        ('{"instruction": "X. Verify Y."}', "X. Verify Y."),
        ('Sure:\n{"instruction": "X. Verify Y.",}\nDone.', "X. Verify Y."),
        ('{"instruction": ""}', None),
        ("no json here", None),
        ('{"title": "X"}', None),
    ]
    for text, want in cases:
        got = parse_json_instruction(text)
        if got != want:
            failures.append(f"parse {text!r}: expected {want!r}, got {got!r}")


def check_selection(root: Path, failures: list[str]) -> None:
    selected = _names(find_targets(root, QA_MODE))
    if selected != EXPECTED_SELECTION:
        failures.append(f"qa_verify selection: expected {EXPECTED_SELECTION}, got {selected}")
    goal_flagged = _names(find_targets(root, GOAL_MODE))
    if goal_flagged != ["task_plain_0"]:
        failures.append(f"goal selection: expected ['task_plain_0'], got {goal_flagged}")


def check_dry_run(root: Path, failures: list[str]) -> None:
    prompts: list[str] = []
    instruction_style_pass.run_agent = _stub_agent(prompts, json.dumps({"instruction": REWRITE}))
    outcomes = asyncio.run(run_pass(root, QA_MODE, workers=2, limit=None, apply=False, report_dir=root))
    if outcomes.get("rewritten") != 2:
        failures.append(f"dry run outcomes: expected 2 rewritten, got {outcomes}")
    for prompt in prompts:
        for needle in (ORIGINAL, "reload the settings page", "now shows Email notifications unchecked"):
            if needle not in prompt:
                failures.append(f"prompt lacks {needle!r}")
        if not 0 <= prompt.find("HARD CONSTRAINT") < prompt.find("Current instruction:"):
            failures.append("the run-incidental rule must be a hard constraint stated before the run details")
    for name in EXPECTED_SELECTION:
        trajectory, meta = _read(root / "repo_smoke" / name)
        if trajectory["instruction"] != ORIGINAL or meta.get("instruction_style"):
            failures.append(f"dry run wrote {name}")
    if _names(find_targets(root, QA_MODE)) != EXPECTED_SELECTION:
        failures.append("dry run changed the selection")


def check_apply(root: Path, failures: list[str]) -> None:
    prompts: list[str] = []
    instruction_style_pass.run_agent = _stub_agent(prompts, f'{{"instruction": "{REWRITE}"}}')
    outcomes = asyncio.run(run_pass(root, QA_MODE, workers=2, limit=None, apply=True, report_dir=root))
    if outcomes.get("rewritten") != 2:
        failures.append(f"apply outcomes: expected 2 rewritten, got {outcomes}")
    for name in EXPECTED_SELECTION:
        trajectory, meta = _read(root / "repo_smoke" / name)
        want = {
            "trajectory.instruction": (trajectory["instruction"], REWRITE),
            "metadata.instruction": (meta.get("instruction"), REWRITE),
            "metadata.instruction_pre_qa_voice": (meta.get("instruction_pre_qa_voice"), ORIGINAL),
            "metadata.instruction_style": (meta.get("instruction_style"), "qa_verify"),
        }
        for field, (got, expected) in want.items():
            if got != expected:
                failures.append(f"{name} {field}: expected {expected!r}, got {got!r}")
    for name in ("task_recheck_c", "task_recheck_wall", "task_plain_1"):
        trajectory, meta = _read(root / "repo_smoke" / name)
        if trajectory["instruction"] != ORIGINAL or meta.get("instruction_pre_qa_voice"):
            failures.append(f"apply touched unselected {name}")
    if find_targets(root, QA_MODE):
        failures.append(f"re-run after apply must select nothing (cap reached): {_names(find_targets(root, QA_MODE))}")
    instruction_style_pass.run_agent = _stub_agent(prompts, "not json at all")
    prompts.clear()
    outcomes = asyncio.run(run_pass(root, QA_MODE, workers=2, limit=None, apply=True, report_dir=root))
    if prompts or outcomes:
        failures.append(f"second apply must be a no-op, got {outcomes} with {len(prompts)} calls")


def check_unparseable_never_writes(root: Path, failures: list[str]) -> None:
    traj_dir = root / "repo_smoke" / "task_recheck_c"
    (traj_dir / "metadata.json").write_text(json.dumps({"capture_source": "agent", "instruction": ORIGINAL}))
    for name in ("task_recheck_a", "task_recheck_b"):
        _, meta = _read(root / "repo_smoke" / name)
        meta.pop("instruction_style")
        (root / "repo_smoke" / name / "metadata.json").write_text(json.dumps(meta))
    for reply, status in (("I cannot help with that.", "unparseable"), ("", "empty")):
        instruction_style_pass.run_agent = _stub_agent([], reply)
        outcomes = asyncio.run(run_pass(root, QA_MODE, workers=1, limit=1, apply=True, report_dir=root))
        trajectory, meta = _read(root / "repo_smoke" / "task_recheck_a")
        if outcomes != {status: 1} or meta.get("instruction_style") or trajectory["instruction"] != REWRITE:
            failures.append(f"{status} reply must not write: {outcomes} style={meta.get('instruction_style')!r}")


def run_smoke() -> bool:
    failures: list[str] = []
    original_run_agent = instruction_style_pass.run_agent
    check_parse(failures)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_corpus(root)
            check_selection(root, failures)
            check_dry_run(root, failures)
            check_apply(root, failures)
            check_unparseable_never_writes(root, failures)
    finally:
        instruction_style_pass.run_agent = original_run_agent
    for failure in failures:
        logger.error(f"  {failure}")
    return not failures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ok = run_smoke()
    logger.info(f"Instruction style smoke test {'PASSED' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)
