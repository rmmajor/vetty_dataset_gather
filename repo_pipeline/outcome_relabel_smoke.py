"""Scripted (no-LLM, no-network) check of the outcome relabel pass: windows, fallback, apply writes, resume skip, prompt content."""
import asyncio
import base64
import json
import logging
import sys
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from repo_pipeline import outcome_relabel
from repo_pipeline.sdk_runner import UsageLimitReached
from train_qwen import effect_annotation, step_effects

logger = logging.getLogger(__name__)

START_URL = "http://app.test/"


def _png(color: tuple) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 36), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _step(index: int, thought: str, actions: list[dict]) -> dict:
    return {"step": index, "screenshot": f"step_{index:03d}_screenshot.png", "thought": thought, "actions": actions}


def make_trajectory_dir(work_dir: Path, name: str, n_steps: int, success: bool = True) -> Path:
    """n_steps clicks ending in done(); step 1 navigates to /settings, steps 2->3 leave the screen unchanged."""
    traj_dir = work_dir / "repo_smoke" / name
    traj_dir.mkdir(parents=True)
    steps = []
    meta_steps = {}
    for i in range(1, n_steps + 1):
        actions = [{"type": "done"}] if i == n_steps else [{"type": "click", "x": 100 + i * 9, "y": 200 + i * 5, "button": "left"}]
        steps.append(_step(i, f"old thought {i}", actions))
        color = (255, 255, 255) if i % 2 == 0 or i == 3 else (0, 0, 0)
        (traj_dir / steps[-1]["screenshot"]).write_bytes(_png(color))
        meta_steps[str(i)] = {"url": START_URL if i == 1 else START_URL + "settings"}
    trajectory = {"task_id": name, "instruction": "Turn on dark mode in settings", "start_url": START_URL,
                  "steps": steps, "total_steps": n_steps, "success": success}
    (traj_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    (traj_dir / "metadata.json").write_text(json.dumps({"capture_source": "agent", "thought_source": "teacher"}, indent=2))
    (traj_dir / "capture_meta.json").write_text(json.dumps({"recorder": "browser_driver", "steps": meta_steps}, indent=2))
    return traj_dir


def _prompt_text(blocks: list[dict]) -> str:
    return blocks[-1]["text"]


def _target_numbers(blocks: list[dict]) -> list[int]:
    return [int(line.split()[1]) for line in _prompt_text(blocks).splitlines()
            if line.startswith("Step ") and "CONTEXT ONLY" not in line]


class FakeLLM:
    """Records every window prompt and answers with `NEW k` per target step, or a wrong-length array for chosen calls."""

    def __init__(self, bad_calls: tuple = ()):
        self.calls: list[list[dict]] = []
        self.bad_calls = bad_calls

    async def __call__(self, blocks: list[dict], full_name: str) -> str:
        self.calls.append(blocks)
        if len(self.calls) in self.bad_calls:
            return json.dumps(["wrong length"])
        return "Here you go:\n" + json.dumps([f"NEW {k}" for k in _target_numbers(blocks)])


def check_windows_and_context(work_dir: Path) -> list[str]:
    failures = []
    traj_dir = make_trajectory_dir(work_dir, "task_windows", 14)
    fake = FakeLLM()
    result = asyncio.run(outcome_relabel.relabel_trajectory(traj_dir, call_llm=fake, apply=False))
    if len(fake.calls) != 2:
        failures.append(f"14 steps should take 2 windows, got {len(fake.calls)} calls")
    if outcome_relabel.step_windows(14) != [(None, list(range(12))), (11, [12, 13])]:
        failures.append(f"unexpected windows {outcome_relabel.step_windows(14)}")
    if _target_numbers(fake.calls[0]) != list(range(1, 13)) or _target_numbers(fake.calls[1]) != [13, 14]:
        failures.append(f"targets per window {[_target_numbers(c) for c in fake.calls]} != [1..12, 13..14]")
    second = _prompt_text(fake.calls[1])
    if "Step 12 " not in second or "CONTEXT ONLY" not in second or "NEW 12" not in second:
        failures.append("second window should carry step 12 as context with its already rewritten thought")
    images = [sum(1 for b in c if b["type"] == "image") for c in fake.calls]
    if images != [12, 3]:
        failures.append(f"images per window {images} != [12, 3]")
    labels = [b["text"] for b in fake.calls[1] if b["type"] == "text"][:-1]
    if labels != ["Screen at step 12", "Screen at step 13", "Screen at step 14"]:
        failures.append(f"unexpected image labels in window 2: {labels}")
    rewritten = [new for _, old, new in result["changes"] if new != old]
    if len(rewritten) != 14 or rewritten != [f"NEW {k}" for k in range(1, 15)]:
        failures.append(f"expected all 14 thoughts rewritten exactly once, got {rewritten}")
    if result["windows"] != 2 or result["unchanged_windows"] != 0 or result["status"] != "relabeled":
        failures.append(f"unexpected result summary {result['status']} {result['windows']} {result['unchanged_windows']}")
    return failures


def check_length_mismatch_fallback(work_dir: Path) -> list[str]:
    failures = []
    traj_dir = make_trajectory_dir(work_dir, "task_mismatch", 14)
    fake = FakeLLM(bad_calls=(2,))
    result = asyncio.run(outcome_relabel.relabel_trajectory(traj_dir, call_llm=fake, apply=True))
    written = json.loads((traj_dir / "trajectory.json").read_text())
    thoughts = [s["thought"] for s in written["steps"]]
    if thoughts[:12] != [f"NEW {k}" for k in range(1, 13)] or thoughts[12:] != ["old thought 13", "old thought 14"]:
        failures.append(f"window 2 should stay unchanged on a length mismatch, got {thoughts}")
    if result["unchanged_windows"] != 1 or result["status"] != "relabeled":
        failures.append(f"mismatch should count one unchanged window, got {result}")
    provenance = json.loads((traj_dir / "metadata.json").read_text()).get("outcome_relabel", {})
    if provenance.get("unchanged_windows") != 1 or provenance.get("windows") != 2:
        failures.append(f"provenance should record the unchanged window: {provenance}")
    all_bad = FakeLLM(bad_calls=(1,))
    unchanged_dir = make_trajectory_dir(work_dir, "task_allbad", 3)
    result = asyncio.run(outcome_relabel.relabel_trajectory(unchanged_dir, call_llm=all_bad, apply=True))
    meta = json.loads((unchanged_dir / "metadata.json").read_text())
    if result["status"] != "unchanged" or meta.get("thought_source") == outcome_relabel.THOUGHT_SOURCE:
        failures.append("a trajectory whose every window failed must not be marked relabeled")
    if (unchanged_dir / outcome_relabel.BACKUP_NAME).exists():
        failures.append("no backup should be written when nothing changed")
    return failures


def check_apply_and_resume(work_dir: Path) -> list[str]:
    failures = []
    traj_dir = make_trajectory_dir(work_dir, "task_apply", 3)
    original = (traj_dir / "trajectory.json").read_bytes()
    dry = asyncio.run(outcome_relabel.relabel_trajectory(traj_dir, call_llm=FakeLLM(), apply=False))
    if (traj_dir / "trajectory.json").read_bytes() != original or (traj_dir / outcome_relabel.BACKUP_NAME).exists():
        failures.append("dry run must write nothing")
    if dry["status"] != "relabeled" or [new for _, _, new in dry["changes"]] != ["NEW 1", "NEW 2", "NEW 3"]:
        failures.append(f"dry run should still report the rewrites: {dry}")
    if outcome_relabel.find_targets(work_dir) != [traj_dir]:
        failures.append(f"trainable unrelabeled dir should be a target: {outcome_relabel.find_targets(work_dir)}")

    asyncio.run(outcome_relabel.relabel_trajectory(traj_dir, call_llm=FakeLLM(), apply=True))
    if (traj_dir / outcome_relabel.BACKUP_NAME).read_bytes() != original:
        failures.append("backup must be a byte copy of the original trajectory.json")
    written = json.loads((traj_dir / "trajectory.json").read_text())
    if [s["thought"] for s in written["steps"]] != ["NEW 1", "NEW 2", "NEW 3"]:
        failures.append(f"apply must replace every thought: {[s['thought'] for s in written['steps']]}")
    meta = json.loads((traj_dir / "metadata.json").read_text())
    provenance = meta.get("outcome_relabel", {})
    if meta.get("thought_source") != outcome_relabel.THOUGHT_SOURCE or meta.get("capture_source") != "agent":
        failures.append(f"apply must set thought_source and keep the rest of metadata: {meta}")
    if set(provenance) != {"date", "model", "windows", "unchanged_windows"} or provenance["windows"] != 1:
        failures.append(f"unexpected provenance {provenance}")
    if outcome_relabel.find_targets(work_dir) != []:
        failures.append("a relabeled dir must be skipped on the next run")

    (traj_dir / "trajectory.json").write_text(json.dumps({**written, "instruction": "changed"}))
    (traj_dir / "metadata.json").write_text(json.dumps({"capture_source": "agent", "thought_source": "teacher"}))
    asyncio.run(outcome_relabel.relabel_trajectory(traj_dir, call_llm=FakeLLM(), apply=True))
    if (traj_dir / outcome_relabel.BACKUP_NAME).read_bytes() != original:
        failures.append("an existing backup must never be overwritten")
    untrainable = make_trajectory_dir(work_dir, "task_unverified", 3, success=False)
    if untrainable in outcome_relabel.find_targets(work_dir):
        failures.append("unverified trajectories must not be targets")
    return failures


def check_prompt_content(work_dir: Path) -> list[str]:
    failures = []
    traj_dir = make_trajectory_dir(work_dir, "task_prompt", 4)
    trajectory = json.loads((traj_dir / "trajectory.json").read_text())
    capture_meta = json.loads((traj_dir / "capture_meta.json").read_text())
    effects = step_effects(traj_dir, trajectory["steps"], capture_meta)
    fake = FakeLLM()
    asyncio.run(outcome_relabel.relabel_trajectory(traj_dir, call_llm=fake, apply=False))
    blocks = fake.calls[0]
    text = _prompt_text(blocks)
    path_note = effect_annotation(effects[0])
    no_change_note = effect_annotation(effects[1])
    if path_note != " -> /settings" or no_change_note != " [no change]":
        failures.append(f"synthetic effects did not produce both annotations: {path_note!r} {no_change_note!r}")
    for note in (path_note, no_change_note):
        if note not in text:
            failures.append(f"prompt must show the harness annotation {note!r}")
    for needle in ("Turn on dark mode in settings", START_URL, "old thought 1", "old thought 4", "done", "Step 1 ",
                   "Output ONLY a JSON array of 4 strings"):
        if needle not in text:
            failures.append(f"prompt missing {needle!r}")
    if outcome_relabel.NO_PREVIOUS_ACTION not in text:
        failures.append("step 1 must be marked as having no previous action")
    for block in blocks:
        if block["type"] == "image":
            source = block["source"]
            if source["type"] != "base64" or source["media_type"] != "image/png":
                failures.append(f"unexpected image block {source.keys()}")
            width, height = Image.open(BytesIO(base64.b64decode(source["data"]))).size
            if (width, height) != outcome_relabel.IMAGE_SIZE:
                failures.append(f"screens must be downscaled to {outcome_relabel.IMAGE_SIZE}, got {(width, height)}")
    if outcome_relabel.OUTCOME_PROMPT.split("{goal}")[0] not in text:
        failures.append("prompt must open with the module's OUTCOME_PROMPT")
    return failures


def check_parse_and_estimate(work_dir: Path) -> list[str]:
    failures = []
    if outcome_relabel.parse_thought_array('noise ["a", "b"] trailing', 2) != ["a", "b"]:
        failures.append("tolerant parse should take the first [ to the last ]")
    for bad in ('["a"]', '["a", 2]', '["a", ""]', "no array here", '["a", "b", "c"]'):
        if outcome_relabel.parse_thought_array(bad, 2) is not None:
            failures.append(f"{bad!r} should be rejected as a structural mismatch")
    make_trajectory_dir(work_dir, "task_est_a", 14)
    make_trajectory_dir(work_dir, "task_est_b", 3)
    estimate = outcome_relabel.estimate_cost(outcome_relabel.find_targets(work_dir))
    if (estimate["trajectories"], estimate["steps"], estimate["windows"], estimate["images"]) != (2, 17, 3, 18):
        failures.append(f"unexpected estimate counts {estimate}")
    if estimate["usd"] <= 0:
        failures.append("estimate must price the run")
    return failures


class RaisingLLM(FakeLLM):
    def __init__(self, error: Exception, on_call: int):
        super().__init__()
        self.error = error
        self.on_call = on_call

    async def __call__(self, blocks: list[dict], full_name: str) -> str:
        if len(self.calls) + 1 == self.on_call:
            self.calls.append(blocks)
            raise self.error
        return await super().__call__(blocks, full_name)


def check_window_errors(work_dir: Path) -> list[str]:
    failures = []
    traj_dir = make_trajectory_dir(work_dir, "task_sdk_error", 14)
    result = asyncio.run(outcome_relabel.relabel_trajectory(
        traj_dir, call_llm=RaisingLLM(RuntimeError("Claude Code returned an error result"), on_call=2), apply=True))
    thoughts = [s["thought"] for s in json.loads((traj_dir / "trajectory.json").read_text())["steps"]]
    if result["status"] != "relabeled" or result["unchanged_windows"] != 1 or thoughts[12:] != ["old thought 13", "old thought 14"]:
        failures.append(f"an SDK error on one window must leave only that window unchanged: {result['status']} {thoughts}")
    limit_dir = make_trajectory_dir(work_dir, "task_limit", 3)
    try:
        asyncio.run(outcome_relabel.relabel_trajectory(limit_dir, call_llm=RaisingLLM(UsageLimitReached("429"), on_call=1)))
        failures.append("UsageLimitReached must propagate so the pass stops and resumes later")
    except UsageLimitReached:
        pass
    return failures


def check_default_call_options(work_dir: Path) -> list[str]:
    failures = []
    seen = {}

    async def fake_run_agent(prompt_stream, options, **kwargs):
        seen["messages"] = [m async for m in prompt_stream]
        seen["options"] = options
        seen["kwargs"] = kwargs
        return json.dumps(["NEW 1", "NEW 2", "NEW 3"])

    traj_dir = make_trajectory_dir(work_dir, "task_options", 3)
    original_run_agent = outcome_relabel.run_agent
    outcome_relabel.run_agent = fake_run_agent
    try:
        result = asyncio.run(outcome_relabel.relabel_trajectory(traj_dir))
    finally:
        outcome_relabel.run_agent = original_run_agent
    options = seen["options"]
    if options.max_turns != 1 or options.tools != [] or options.model != outcome_relabel.MODEL_FOR_GROUNDED_GENERATION:
        failures.append(f"default call must be one tool-free turn on the grounded-generation model: {options}")
    if options.permission_mode != "bypassPermissions" or options.max_budget_usd != outcome_relabel.OUTCOME_BUDGET_USD:
        failures.append(f"unexpected permission/budget options: {options}")
    content = seen["messages"][0]["message"]["content"]
    if len(seen["messages"]) != 1 or content[0]["type"] != "text" or content[1]["type"] != "image" or content[-1]["type"] != "text":
        failures.append("the prompt stream must be one user message of labelled image blocks followed by the prompt text")
    if seen["kwargs"].get("purpose") != "outcome_relabel" or seen["kwargs"].get("raise_on_error") is not False:
        failures.append(f"unexpected run_agent telemetry kwargs {seen['kwargs']}")
    if [new for _, _, new in result["changes"]] != ["NEW 1", "NEW 2", "NEW 3"]:
        failures.append(f"default call path should feed the parsed thoughts through: {result}")
    return failures


CHECKS = [check_windows_and_context, check_length_mismatch_fallback, check_apply_and_resume,
          check_prompt_content, check_parse_and_estimate, check_window_errors, check_default_call_options]


def run_smoke() -> bool:
    failures = []
    for check in CHECKS:
        with tempfile.TemporaryDirectory() as tmp:
            found = check(Path(tmp))
        logger.info(f"[{'FAIL' if found else 'pass'}] {check.__name__}")
        failures.extend(f"{check.__name__}: {f}" for f in found)
    for failure in failures:
        logger.error(f"  {failure}")
    return not failures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ok = run_smoke()
    logger.info(f"Outcome-relabel smoke test {'PASSED' if ok else 'FAILED'} ({len(CHECKS)} checks)")
    sys.exit(0 if ok else 1)
