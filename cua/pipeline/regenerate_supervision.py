"""CLI: wipe and regenerate `<think>` blocks and `instruction` text for recorded trajectories.

Two flavours, picked by `--dry-run`:
  --dry-run            Generate in memory, print an old-vs-new Markdown report, write nothing.
  (no --dry-run)       Bulk-write: overwrite the affected fields in `trajectory.json` in place.

`--instructions` / `--thoughts` narrow what gets regenerated (default: both).
`--sample N` picks N trajectories spanning Path A and Path B for review.

Path A trajectories (mind2web, in `trajectories/task_*/`) skip the instruction generator —
their instructions come from the GPT recorder with a different provenance and would not
benefit from the test-source-based prompt. They still get new thoughts.

Run from the repo root:
  .venv-pipeline/bin/python -m cua.pipeline.regenerate_supervision --dry-run --sample 5
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from cua.pipeline.stage6_generate import _load_capture_meta, _load_test_source, generate_instruction
from cua.pipeline.think_generator import (
    generate_think_for_step,
    render_action_nl,
    render_grounding_hint,
    render_history,
)

logger = logging.getLogger(__name__)


def find_trajectory_dirs(root: Path) -> list[Path]:
    """Collect every trajectory dir under `root` that has a trajectory.json. Walks both
    layouts: Path B nested (`repo_*/task_*`) and Path A flat (`task_*`)."""
    dirs: list[Path] = []
    for repo_dir in sorted(root.glob("repo_*")):
        if not repo_dir.is_dir():
            continue
        for traj_dir in sorted(repo_dir.glob("task_*")):
            if traj_dir.is_dir() and (traj_dir / "trajectory.json").exists():
                dirs.append(traj_dir)
    for traj_dir in sorted(root.glob("task_*")):
        if traj_dir.is_dir() and (traj_dir / "trajectory.json").exists():
            dirs.append(traj_dir)
    return dirs


def is_path_b(traj_dir: Path) -> bool:
    """Path B trajectories live under a `repo_*` parent (one per source repo). Path A
    trajectories sit directly under the trajectories root."""
    return traj_dir.parent.name.startswith("repo_")


def sample_trajectories(dirs: list[Path], n: int) -> list[Path]:
    """Pick N trajectories that span both paths so the smoke-test diff is representative.
    Heuristic: alternate between Path B (where the new instruction prompt has the biggest
    impact) and Path A (which only exercises the thought generator)."""
    if n <= 0 or not dirs:
        return []
    if n >= len(dirs):
        return list(dirs)
    path_b = [d for d in dirs if is_path_b(d)]
    path_a = [d for d in dirs if not is_path_b(d)]
    out: list[Path] = []
    i = j = 0
    while len(out) < n and (i < len(path_b) or j < len(path_a)):
        if i < len(path_b) and len(out) < n:
            out.append(path_b[i])
            i += 1
        if j < len(path_a) and len(out) < n:
            out.append(path_a[j])
            j += 1
    return out


def repo_dict_from_metadata(metadata: dict) -> dict:
    """Construct the minimal `repo` dict the instruction generator expects from whatever
    metadata.json carries. The registry DB has richer fields, but for regeneration of
    already-recorded trajectories metadata.json is the only durable source on disk."""
    full_name = metadata.get("source_repo") or ""
    return {
        "full_name": full_name,
        "name": full_name.split("/")[-1] if "/" in full_name else full_name,
        "description": (metadata.get("domain") or "").strip(),
    }


async def regenerate_one(
    traj_dir: Path,
    do_instructions: bool,
    do_thoughts: bool,
) -> tuple[dict, dict]:
    """Run the new generators against one trajectory entirely in memory.

    Returns (summary, new_trajectory): `summary` describes what changed for the report;
    `new_trajectory` is the post-regeneration trajectory dict ready to be written back to
    disk (caller decides whether to persist, based on --dry-run).
    """
    trajectory_path = traj_dir / "trajectory.json"
    with open(trajectory_path) as f:
        trajectory = json.load(f)

    metadata: dict = {}
    metadata_path = traj_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    capture_meta = _load_capture_meta(traj_dir)
    test_source = _load_test_source(traj_dir)
    path_b = is_path_b(traj_dir)

    new_trajectory = json.loads(json.dumps(trajectory))

    old_thoughts = [(s.get("step"), (s.get("thought") or "").strip()) for s in trajectory.get("steps", [])]
    summary: dict = {
        "traj_dir": str(traj_dir),
        "task_id": trajectory.get("task_id", ""),
        "path_kind": "B" if path_b else "A",
        "old_instruction": trajectory.get("instruction", ""),
        "new_instruction": trajectory.get("instruction", ""),
        "old_thoughts": old_thoughts,
        "new_thoughts": old_thoughts,
        "test_source_available": bool(test_source),
        "instruction_regenerated": False,
        "thoughts_regenerated": False,
    }

    if do_instructions and path_b:
        repo = repo_dict_from_metadata(metadata)
        new_instruction = await generate_instruction(
            new_trajectory,
            capture_meta,
            repo=repo,
            test_source=test_source,
        )
        new_trajectory["instruction"] = new_instruction
        summary["new_instruction"] = new_instruction
        summary["instruction_regenerated"] = True

    if do_thoughts:
        for step in new_trajectory.get("steps", []):
            step["thought"] = ""
        await _regenerate_thoughts_in_memory(new_trajectory, capture_meta, traj_dir)
        new_thoughts = [(s.get("step"), (s.get("thought") or "").strip()) for s in new_trajectory.get("steps", [])]
        summary["new_thoughts"] = new_thoughts
        summary["thoughts_regenerated"] = True

    return summary, new_trajectory


async def _regenerate_thoughts_in_memory(trajectory: dict, capture_meta: dict, traj_dir: Path) -> None:
    """Walk steps in order and fill `step['thought']` using the new generator. Mirrors the
    sequential history-aware loop in `generate_thinks_for_trajectory` so the prompt for
    step N sees the freshly-written thoughts for steps 1..N-1."""
    steps = trajectory.get("steps", [])
    if not steps:
        return

    goal = (trajectory.get("instruction") or "").strip() or "(goal not recorded)"
    start_url = (trajectory.get("start_url") or "").strip() or "(unknown)"
    steps_meta = capture_meta.get("steps", {}) if isinstance(capture_meta, dict) else {}

    action_nl_by_step: dict[int, str] = {}
    grounding_by_step: dict[int, tuple] = {}
    for step in steps:
        n = step.get("step")
        if not isinstance(n, int):
            continue
        actions = step.get("actions", []) or []
        meta = steps_meta.get(str(n), {}) if isinstance(steps_meta, dict) else {}
        role = step.get("role") or meta.get("role") or ""
        name = step.get("name") or meta.get("name") or ""
        bbox = step.get("bbox") or meta.get("bbox")
        action_nl_by_step[n] = render_action_nl(actions[0], role=role, name=name) if actions else "(no action)"
        grounding_by_step[n] = (role, name, bbox)

    ordered = sorted(steps, key=lambda s: s.get("step", 0))
    for step in ordered:
        n = step.get("step")
        if not isinstance(n, int) or not step.get("actions"):
            continue
        screenshot_path = traj_dir / step["screenshot"]
        if not screenshot_path.exists():
            continue
        history_block = render_history(steps, n, action_nl_by_step)
        role, name, bbox = grounding_by_step.get(n, ("", "", None))
        grounding_hint = render_grounding_hint(role, name, bbox)
        thought = await generate_think_for_step(
            screenshot_path=screenshot_path,
            goal=goal,
            start_url=start_url,
            history_block=history_block,
            about_to_action=action_nl_by_step.get(n, "(no action)"),
            grounding_hint=grounding_hint,
        )
        step["thought"] = thought


def render_report(summaries: list[dict]) -> str:
    """Build a Markdown old-vs-new report for stdout (and optionally a file). One section
    per trajectory: instruction diff at the top, then a per-step table of old vs new thought."""
    lines: list[str] = []
    lines.append("# Supervision regeneration smoke report")
    lines.append("")
    lines.append(f"Trajectories sampled: **{len(summaries)}**")
    lines.append("")

    for s in summaries:
        lines.append(f"## `{s['task_id']}`  *(Path {s['path_kind']})*")
        lines.append("")
        lines.append(f"- Dir: `{s['traj_dir']}`")
        lines.append(f"- Test source available: **{s['test_source_available']}**")
        lines.append(f"- Instruction regenerated: **{s['instruction_regenerated']}**")
        lines.append(f"- Thoughts regenerated: **{s['thoughts_regenerated']}**")
        lines.append("")

        if s["instruction_regenerated"]:
            lines.append("### Instruction")
            lines.append("")
            lines.append("| | text |")
            lines.append("|---|---|")
            lines.append(f"| **OLD** | {_md_cell(s['old_instruction'])} |")
            lines.append(f"| **NEW** | {_md_cell(s['new_instruction'])} |")
            lines.append("")

        if s["thoughts_regenerated"]:
            lines.append("### Thoughts (per step)")
            lines.append("")
            lines.append("| step | old | new |")
            lines.append("|---|---|---|")
            old_map = {n: t for n, t in s["old_thoughts"]}
            for step_num, new_t in s["new_thoughts"]:
                old_t = old_map.get(step_num, "")
                lines.append(f"| {step_num} | {_md_cell(old_t)} | {_md_cell(new_t)} |")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def _md_cell(text: str) -> str:
    """Squeeze a thought/instruction into one table cell — collapse newlines and escape pipes."""
    text = (text or "").strip()
    if not text:
        return "*(empty)*"
    text = text.replace("\n", " ").replace("|", "\\|")
    if len(text) > 400:
        text = text[:400] + "…"
    return text


async def amain(args: argparse.Namespace) -> int:
    root = Path(args.path).resolve()
    if not root.exists():
        logger.error(f"trajectories root not found: {root}")
        return 2

    all_dirs = find_trajectory_dirs(root)
    if not all_dirs:
        logger.error(f"no trajectories under {root}")
        return 2

    chosen = sample_trajectories(all_dirs, args.sample) if args.sample else all_dirs
    logger.info(
        f"Found {len(all_dirs)} trajectories under {root}; processing {len(chosen)} "
        f"(dry-run={args.dry_run}, instructions={args.do_instructions}, thoughts={args.do_thoughts})"
    )

    summaries: list[dict] = []
    for traj_dir in chosen:
        logger.info(f"  → {traj_dir.relative_to(root)}")
        summary, new_trajectory = await regenerate_one(
            traj_dir,
            do_instructions=args.do_instructions,
            do_thoughts=args.do_thoughts,
        )
        summaries.append(summary)

        if not args.dry_run:
            with open(traj_dir / "trajectory.json", "w") as f:
                json.dump(new_trajectory, f, indent=2)

    report = render_report(summaries)
    if args.dry_run or args.report_path:
        print(report)
    if args.report_path:
        Path(args.report_path).write_text(report)
        logger.info(f"Wrote report to {args.report_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="trajectories", help="trajectories root (default: trajectories/)")
    parser.add_argument(
        "--dry-run", action="store_true", help="generate in memory and print Markdown report, no writes"
    )
    parser.add_argument("--sample", type=int, default=0, help="limit to N sampled trajectories (0 = all)")
    parser.add_argument(
        "--instructions",
        dest="do_instructions",
        action="store_true",
        help="regenerate the trajectory.instruction field",
    )
    parser.add_argument(
        "--thoughts", dest="do_thoughts", action="store_true", help="regenerate every steps[].thought field"
    )
    parser.add_argument("--report-path", default="", help="also write the Markdown report to this file")
    parser.add_argument("--verbose", action="store_true", help="enable INFO logging")
    args = parser.parse_args()

    if not args.do_instructions and not args.do_thoughts:
        args.do_instructions = True
        args.do_thoughts = True

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
