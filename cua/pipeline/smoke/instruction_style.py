"""Scripted (no-LLM, no-network) check of the instruction style pass: selection, cap, idempotency and writes."""

import asyncio
import json
import sys
from pathlib import Path

from cua.pipeline.corpus import instruction_style as instruction_style_pass
from cua.pipeline.corpus.instruction_style import STYLE_MODES, InstructionStylePass, parse_json_instruction
from cua.pipeline.smoke._harness import run_checks, with_tempdir

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
REWRITE = (
    "Unchecking Email notifications in Settings persists after save and reload. "
    "Verify the box stays unchecked when the page is reopened."
)


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


def _selected(root: Path, mode) -> list[str]:
    """The mode's selection as directory names, which is the shape these checks compare against."""
    return [recorded.path.name for recorded in InstructionStylePass(mode, False, root).candidates(root)]


def _run_pass(root: Path, mode, reply: str, workers: int, limit, apply: bool) -> tuple[dict, list[str]]:
    """The mode over the corpus against a stubbed teacher; returns the outcomes and every prompt it sent."""
    prompts: list[str] = []
    original_run_agent = instruction_style_pass.run_agent
    instruction_style_pass.run_agent = _stub_agent(prompts, reply)
    try:
        outcomes = asyncio.run(InstructionStylePass(mode, apply, root).run(root, limit=limit, workers=workers))
    finally:
        instruction_style_pass.run_agent = original_run_agent
    return outcomes, prompts


def _read(traj_dir: Path) -> tuple[dict, dict]:
    return (
        json.loads((traj_dir / "trajectory.json").read_text()),
        json.loads((traj_dir / "metadata.json").read_text()),
    )


def check_parse() -> list[str]:
    failures: list[str] = []
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
    return failures


@with_tempdir
def check_selection(root: Path) -> list[str]:
    failures: list[str] = []
    _write_corpus(root)
    selected = _selected(root, QA_MODE)
    if selected != EXPECTED_SELECTION:
        failures.append(f"qa_verify selection: expected {EXPECTED_SELECTION}, got {selected}")
    goal_flagged = _selected(root, GOAL_MODE)
    if goal_flagged != ["task_plain_0"]:
        failures.append(f"goal selection: expected ['task_plain_0'], got {goal_flagged}")
    return failures


@with_tempdir
def check_dry_run(root: Path) -> list[str]:
    failures: list[str] = []
    _write_corpus(root)
    reply = json.dumps({"instruction": REWRITE})
    outcomes, prompts = _run_pass(root, QA_MODE, reply, workers=2, limit=None, apply=False)
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
    if _selected(root, QA_MODE) != EXPECTED_SELECTION:
        failures.append("dry run changed the selection")
    return failures


@with_tempdir
def check_apply(root: Path) -> list[str]:
    failures: list[str] = []
    _write_corpus(root)
    outcomes, _ = _run_pass(root, QA_MODE, f'{{"instruction": "{REWRITE}"}}', workers=2, limit=None, apply=True)
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
    if _selected(root, QA_MODE):
        failures.append(f"re-run after apply must select nothing (cap reached): {_selected(root, QA_MODE)}")
    outcomes, prompts = _run_pass(root, QA_MODE, "not json at all", workers=2, limit=None, apply=True)
    if prompts or outcomes:
        failures.append(f"second apply must be a no-op, got {outcomes} with {len(prompts)} calls")
    return failures


@with_tempdir
def check_unparseable_never_writes(root: Path) -> list[str]:
    """Runs the apply pass first, because what must never be written is a SECOND rewrite over an already styled dir."""
    failures: list[str] = []
    _write_corpus(root)
    _run_pass(root, QA_MODE, f'{{"instruction": "{REWRITE}"}}', workers=2, limit=None, apply=True)
    traj_dir = root / "repo_smoke" / "task_recheck_c"
    (traj_dir / "metadata.json").write_text(json.dumps({"capture_source": "agent", "instruction": ORIGINAL}))
    for name in ("task_recheck_a", "task_recheck_b"):
        _, meta = _read(root / "repo_smoke" / name)
        meta.pop("instruction_style")
        (root / "repo_smoke" / name / "metadata.json").write_text(json.dumps(meta))
    for reply, status in (("I cannot help with that.", "unparseable"), ("", "empty")):
        outcomes, _ = _run_pass(root, QA_MODE, reply, workers=1, limit=1, apply=True)
        trajectory, meta = _read(root / "repo_smoke" / "task_recheck_a")
        if outcomes != {status: 1} or meta.get("instruction_style") or trajectory["instruction"] != REWRITE:
            failures.append(f"{status} reply must not write: {outcomes} style={meta.get('instruction_style')!r}")
    return failures


CHECKS = [check_parse, check_selection, check_dry_run, check_apply, check_unparseable_never_writes]


if __name__ == "__main__":
    sys.exit(run_checks(CHECKS, "Instruction style smoke test"))
