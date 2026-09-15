"""Outcome-checked thought relabel: every thought opens with what the previous action did, judged from consecutive screens.

Trainable trajectories are rewritten in windows of at most WINDOW_STEPS steps (one vision call per window; each later
window carries the previous window's last step as context only). Resumable: metadata.thought_source flips to
`teacher_outcome` on apply and such dirs are skipped on the next run. Dry-run by default.

  python -m cua.pipeline.outcome_relabel trajectories/ [--dry-run|--apply] [--limit N] [--workers 3] [--sample N]
                                          [--max-steps N] [--show N] [--report-path FILE] [--estimate]
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from cua.pipeline.agent_runner import run_agent
from cua.pipeline.corpus_audit import is_trainable
from cua.pipeline.llm_config import MODEL_FOR_GROUNDED_GENERATION
from cua.pipeline.sdk_runner import UsageLimitReached
from cua.sft.legacy_train_qwen import (
    PREV_IMAGE_SIZE,
    downscale_prev,
    format_action_summary,
    history_entries_for_step,
    step_effects,
    to_model_action,
)

logger = logging.getLogger(__name__)

THOUGHT_SOURCE = "teacher_outcome"
BACKUP_NAME = "trajectory_pre_outcome.json"
IMAGE_SIZE = PREV_IMAGE_SIZE
NO_PREVIOUS_ACTION = "none (this is the starting screen)"
WINDOW_STEPS = int(os.environ.get("OUTCOME_WINDOW_STEPS", "12"))
OUTCOME_BUDGET_USD = float(os.environ.get("OUTCOME_BUDGET_USD", "1.00"))
# Estimate knobs: image tokens per downscaled screen on the judge model, list prices per 1M tokens, SDK per-call floor.
IMAGE_TOKENS = int(os.environ.get("OUTCOME_IMAGE_TOKENS", "350"))
TEXT_TOKENS_PER_STEP = 130
PROMPT_TOKENS = 450
OUTPUT_TOKENS_PER_STEP = 70
PRICE_IN_PER_M = float(os.environ.get("OUTCOME_PRICE_IN_PER_M", "2.0"))
PRICE_OUT_PER_M = float(os.environ.get("OUTCOME_PRICE_OUT_PER_M", "10.0"))
CALL_OVERHEAD_USD = float(os.environ.get("OUTCOME_CALL_OVERHEAD_USD", "0.08"))

OUTCOME_PROMPT = """Above are the screens an agent saw while carrying out a web task, one per step, labelled
"Screen at step k" (shown at reduced size; where text is too small to read, trust the step's
current thought for what it says). Below, each step lists the action that came BEFORE the screen
(with the note the model will be given about it), the action taken next, and the thought written
at the time. Rewrite each step's thought into training supervision for a computer-use model: the
first-person, present-tense inner monologue of someone who looks at the screen, checks what the
last move did, and decides the next move.

Rules for every rewritten thought:
- 1-3 short sentences, first person, present tense.
- The FIRST sentence states the outcome of the previous action, judged by comparing the previous
  screen with the current one — e.g. "Clicking Save worked: a 'Settings saved' banner is visible."
  or "Nothing changed after that click; the menu is still closed." For step 1 (no previous action)
  describe what the starting screen shows instead.
- The bracketed note after a previous action is what the model will be told: [no change] means the
  screen stayed the same, -> /path means the page moved to that path. Agree with it and say what it
  means on screen.
- Then say what to do next and why, tied to the goal.
- Keep the original's observations and any visible UI text; you are adding the outcome check and
  polishing voice, not inventing content.
- Remove ALL traces of tooling: never mention refs, snapshots, screenshots, tools, selectors,
  elements by number, coordinates, "the page state", or being an AI or agent under instruction.
  Say "the screen" or "the page".
- For a step whose action is done(): cite each visible condition of the goal that the current screen
  satisfies — concrete on-screen evidence, one per condition. Never "the goal is accomplished"
  without evidence.
- For a step whose action is handoff(...) or confirm(...), or that ends without reaching the goal:
  state what is seen and why stopping (or asking) here is right.
- A step marked CONTEXT ONLY is already final: use it to understand the situation, but do not
  output a thought for it.

Goal of the task: {goal}
Starting URL: {start_url}

Steps:
{steps_block}

Output ONLY a JSON array of {n} strings — the rewritten thought for each step that is not context-only,
in step order. No markdown fence, no prose."""


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def needs_relabel(traj_dir: Path) -> bool:
    return _load(traj_dir / "metadata.json").get("thought_source") != THOUGHT_SOURCE


def find_targets(corpus_dir: Path, max_steps: int | None = None) -> list[Path]:
    """Trainable, not-yet-relabeled trajectory dirs in sorted order; --max-steps bounds their length."""
    targets = []
    for traj_dir in sorted(corpus_dir.glob("**/task_*")):
        if not (traj_dir / "trajectory.json").exists() or not needs_relabel(traj_dir) or not is_trainable(traj_dir):
            continue
        if max_steps is not None and len(_load(traj_dir / "trajectory.json").get("steps", [])) > max_steps:
            continue
        targets.append(traj_dir)
    return targets


def step_windows(n_steps: int, size: int = WINDOW_STEPS) -> list[tuple[int | None, list[int]]]:
    """(context index or None, target indices) per window; the context is the previous window's last target."""
    windows = []
    start = 0
    while start < n_steps:
        targets = list(range(start, min(start + size, n_steps)))
        windows.append((start - 1 if start else None, targets))
        start += size
    return windows


def _actions_summary(step: dict) -> str:
    actions = step.get("actions") or ([step["action"]] if "action" in step else [])
    return ", ".join(format_action_summary(to_model_action(a)) for a in actions) or "(none)"


def render_steps_block(steps: list[dict], effects: list, context_index: int | None, target_indices: list[int]) -> str:
    lines = []
    for index in ([context_index] if context_index is not None else []) + target_indices:
        step = steps[index]
        previous = (
            ", ".join(history_entries_for_step(steps[index - 1], effects[index - 1])) if index else NO_PREVIOUS_ACTION
        )
        header = f'Step {step["step"]} (see "Screen at step {step["step"]}")'
        if index == context_index:
            header += " — CONTEXT ONLY, thought already final, do not output it"
        lines.append(header)
        lines.append(f"- previous action: {previous}")
        lines.append(f"- action taken next: {_actions_summary(step)}")
        lines.append(f"- current thought: {(step.get('thought') or '').strip() or '(empty)'}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_window_blocks(
    traj_dir: Path, trajectory: dict, effects: list, context_index: int | None, target_indices: list[int]
) -> list[dict]:
    """Content blocks for one window: labelled downscaled screens, then the prompt with the steps block."""
    steps = trajectory["steps"]
    blocks = []
    for index in ([context_index] if context_index is not None else []) + target_indices:
        step = steps[index]
        png = downscale_prev((traj_dir / step["screenshot"]).read_bytes(), IMAGE_SIZE)
        blocks.append({"type": "text", "text": f"Screen at step {step['step']}"})
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode("utf-8")},
            }
        )
    blocks.append(
        {
            "type": "text",
            "text": OUTCOME_PROMPT.format(
                goal=(trajectory.get("instruction") or "(goal not recorded)").strip(),
                start_url=trajectory.get("start_url") or "(not recorded)",
                steps_block=render_steps_block(steps, effects, context_index, target_indices),
                n=len(target_indices),
            ),
        }
    )
    return blocks


def parse_thought_array(text: str, n: int) -> list[str] | None:
    """The JSON array between the first [ and the last ]; None unless it holds exactly n non-empty strings."""
    start, end = text.find("["), text.rfind("]") + 1
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or len(parsed) != n or not all(isinstance(t, str) and t.strip() for t in parsed):
        return None
    return [t.strip() for t in parsed]


async def llm_call(blocks: list[dict], full_name: str) -> str:
    async def prompt_stream():
        yield {
            "type": "user",
            "message": {"role": "user", "content": blocks},
            "parent_tool_use_id": None,
            "session_id": "",
        }

    return await run_agent(
        prompt_stream(),
        # tools=[] removes every built-in tool: a tool call would consume the single turn and the SDK raises error_max_turns.
        ClaudeAgentOptions(
            model=MODEL_FOR_GROUNDED_GENERATION,
            max_turns=1,
            max_budget_usd=OUTCOME_BUDGET_USD,
            permission_mode="bypassPermissions",
            tools=[],
        ),
        full_name=full_name,
        stage=7,
        purpose="outcome_relabel",
        raise_on_error=False,
    )


def write_relabel(
    traj_dir: Path, trajectory: dict, original_bytes: bytes, windows: int, unchanged_windows: int
) -> None:
    """Back up the original once, replace the thoughts where convert reads them, and record provenance."""
    backup = traj_dir / BACKUP_NAME
    if not backup.exists():
        backup.write_bytes(original_bytes)
    (traj_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    metadata_path = traj_dir / "metadata.json"
    metadata = _load(metadata_path)
    metadata["thought_source"] = THOUGHT_SOURCE
    metadata["outcome_relabel"] = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL_FOR_GROUNDED_GENERATION,
        "windows": windows,
        "unchanged_windows": unchanged_windows,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


async def relabel_trajectory(traj_dir: Path, call_llm=llm_call, apply: bool = False) -> dict:
    """Rewrite one trajectory's thoughts window by window; returns status, window counts and (step, old, new) changes."""
    trajectory_path = traj_dir / "trajectory.json"
    original_bytes = trajectory_path.read_bytes()
    trajectory = json.loads(original_bytes)
    steps = trajectory.get("steps", [])
    if not steps:
        return {"status": "empty", "windows": 0, "unchanged_windows": 0, "changes": []}
    effects = step_effects(traj_dir, steps, _load(traj_dir / "capture_meta.json"))
    windows = step_windows(len(steps))
    unchanged_windows = 0
    changes = []
    for number, (context_index, target_indices) in enumerate(windows, start=1):
        blocks = build_window_blocks(traj_dir, trajectory, effects, context_index, target_indices)
        # Windows run in order so each later window sees the already rewritten context thought.
        try:
            text = await call_llm(blocks, f"outcome_relabel:{traj_dir.name}:w{number}")
        except UsageLimitReached:
            raise
        except Exception as e:
            logger.warning(
                f"  {traj_dir.name}: window {number}/{len(windows)} call failed ({e}); leaving its thoughts unchanged"
            )
            text = ""
        rewritten = parse_thought_array(text or "", len(target_indices))
        if rewritten is None:
            logger.warning(
                f"  {traj_dir.name}: window {number}/{len(windows)} returned no usable "
                f"{len(target_indices)}-thought array; leaving its thoughts unchanged"
            )
            unchanged_windows += 1
            changes.extend(
                (steps[i]["step"], steps[i].get("thought", ""), steps[i].get("thought", "")) for i in target_indices
            )
            continue
        for index, thought in zip(target_indices, rewritten, strict=False):
            changes.append((steps[index]["step"], steps[index].get("thought", ""), thought))
            steps[index]["thought"] = thought
    if unchanged_windows == len(windows):
        return {
            "status": "unchanged",
            "windows": len(windows),
            "unchanged_windows": unchanged_windows,
            "changes": changes,
        }
    if apply:
        write_relabel(traj_dir, trajectory, original_bytes, len(windows), unchanged_windows)
    return {"status": "relabeled", "windows": len(windows), "unchanged_windows": unchanged_windows, "changes": changes}


def estimate_cost(targets: list[Path]) -> dict:
    """Window/image/token counts for the given targets priced at the module's list-price knobs."""
    trajectories = steps = windows = images = 0
    for traj_dir in targets:
        n_steps = len(_load(traj_dir / "trajectory.json").get("steps", []))
        if not n_steps:
            continue
        trajectories += 1
        steps += n_steps
        for context_index, target_indices in step_windows(n_steps):
            windows += 1
            images += len(target_indices) + (context_index is not None)
    input_tokens = images * IMAGE_TOKENS + steps * TEXT_TOKENS_PER_STEP + windows * PROMPT_TOKENS
    output_tokens = steps * OUTPUT_TOKENS_PER_STEP
    usd = input_tokens / 1e6 * PRICE_IN_PER_M + output_tokens / 1e6 * PRICE_OUT_PER_M + windows * CALL_OVERHEAD_USD
    return {
        "trajectories": trajectories,
        "steps": steps,
        "windows": windows,
        "images": images,
        "image_tokens": images * IMAGE_TOKENS,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": usd,
    }


def print_changes(corpus_dir: Path, traj_dir: Path, result: dict, show: int) -> None:
    print(
        f"\n{traj_dir.relative_to(corpus_dir)} [{result['status']}] "
        f"{len(result['changes'])} steps, {result['windows']} windows, {result['unchanged_windows']} unchanged"
    )
    for step_number, old, new in result["changes"][:show]:
        print(f"step {step_number}: {old}\n     → {new}")


def render_report(corpus_dir: Path, rows: list, apply: bool) -> str:
    lines = [
        f"# Outcome relabel report ({'applied' if apply else 'dry run'})",
        "",
        "| trajectory | step | old | new |",
        "| --- | --- | --- | --- |",
    ]
    for traj_dir, result in sorted(rows, key=lambda r: str(r[0])):
        for step_number, old, new in result["changes"]:
            old, new = (s.replace("|", "/").replace("\n", " ") for s in (old, new))
            lines.append(f"| {traj_dir.relative_to(corpus_dir)} | {step_number} | {old} | {new} |")
    return "\n".join(lines) + "\n"


async def run_pass(corpus_dir: Path, targets: list[Path], workers: int, apply: bool, show: int) -> tuple[dict, list]:
    semaphore = asyncio.Semaphore(workers)
    outcomes: dict = {}
    rows: list = []

    async def guarded(traj_dir: Path):
        async with semaphore:
            try:
                result = await relabel_trajectory(traj_dir, apply=apply)
            except UsageLimitReached:
                raise
            except Exception as e:
                logger.warning(f"  {traj_dir.name}: relabel failed: {e}")
                result = {"status": "failed", "windows": 0, "unchanged_windows": 0, "changes": []}
        outcomes[result["status"]] = outcomes.get(result["status"], 0) + 1
        rows.append((traj_dir, result))
        logger.info(
            f"  {traj_dir.name}: {result['status']} ({result['unchanged_windows']}/{result['windows']} windows unchanged)"
        )
        print_changes(corpus_dir, traj_dir, result, show)

    try:
        await asyncio.gather(*(guarded(d) for d in targets))
    except UsageLimitReached:
        logger.warning("usage limit reached — unrelabeled targets stay selectable; re-run to resume")
    return outcomes, rows


def select_targets(args: argparse.Namespace) -> list[Path]:
    targets = find_targets(args.corpus, args.max_steps)
    if args.sample:
        targets = sorted(random.Random(args.seed).sample(targets, min(args.sample, len(targets))))
    if args.limit:
        targets = targets[: args.limit]
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("trajectories/"))
    parser.add_argument("--apply", action="store_true", help="Write the rewrites (default is a dry run)")
    parser.add_argument("--dry-run", action="store_true", help="Print old -> new and write nothing (default)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None, help="Random subset of the targets (before --limit)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None, help="Only trajectories with at most N steps")
    parser.add_argument("--show", type=int, default=5, help="Steps printed per trajectory as OLD -> NEW")
    parser.add_argument("--report-path", type=Path, default=None, help="Also write a Markdown old/new table here")
    parser.add_argument("--estimate", action="store_true", help="Print the full-run cost estimate and exit")
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    targets = select_targets(args)
    if args.estimate:
        estimate = estimate_cost(targets)
        print(
            f"targets: {estimate['trajectories']} trajectories, {estimate['steps']} steps, {estimate['windows']} windows "
            f"({WINDOW_STEPS} steps each), {estimate['images']} images"
        )
        print(
            f"tokens: ~{estimate['input_tokens']:,} in ({estimate['image_tokens']:,} image @ {IMAGE_TOKENS}/image) "
            f"+ ~{estimate['output_tokens']:,} out"
        )
        print(
            f"estimate: ~${estimate['usd']:.2f} at ${PRICE_IN_PER_M}/M in, ${PRICE_OUT_PER_M}/M out, "
            f"${CALL_OVERHEAD_USD} SDK floor per call ({estimate['windows']} calls)"
        )
        return

    logger.info(f"{len(targets)} trajectories to relabel under {args.corpus} ({'apply' if args.apply else 'dry run'})")
    outcomes, rows = asyncio.run(run_pass(args.corpus, targets, args.workers, args.apply, args.show))
    if args.report_path and rows:
        args.report_path.write_text(render_report(args.corpus, rows, args.apply))
        logger.info(f"report: {args.report_path}")
    print(f"\n{'applied' if args.apply else 'dry run'}: {outcomes}")
    sys.exit(1 if outcomes.get("failed") and not outcomes.get("relabeled") else 0)


if __name__ == "__main__":
    main()
