"""Rewrite procedure-flavored instructions in the trainable corpus into terse user goals.

Instructions must read as GOALS ("Sign me up for X"), never step-by-step tours ("Open X, filter Y,
sort Z, then…") — the proposer rule that later trajectories already follow. This pass finds the
residue, rewrites each with an LLM while preserving every concrete parameter, and marks
metadata.instruction_style="goal_rewrite_v1" so a re-run skips done work. The trajectory is already
verifier-approved; the rewrite must keep the same outcome, only drop the route — review the
dry-run report before applying.

  python -m repo_pipeline.instruction_style_pass trajectories/ --dry-run
  python -m repo_pipeline.instruction_style_pass trajectories/ --apply [--limit N]
"""
import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from repo_pipeline.agent_runner import run_agent
from repo_pipeline.corpus_audit import is_trainable
from repo_pipeline.llm_config import MODEL_FOR_GROUNDED_GENERATION
from repo_pipeline.relabel_recoverable import NOT_GOAL_SHAPED
from repo_pipeline.sdk_runner import UsageLimitReached

logger = logging.getLogger(__name__)

REWRITE_BUDGET_USD = 0.10

_STARTS_WITH_ROUTE = re.compile(r"^(open|go to|navigate|visit|click|scroll)\b", re.IGNORECASE)
# Route verbs mid-instruction; content specs ("a post with five items, each…") must NOT fire.
_ROUTE_VERBS = re.compile(
    r"\b(click|clicking|navigate|navigating|scroll|scrolling|go to|open the|advancing through|"
    r"proceed(ing)? (to|through)|then (type|select|press|choose))\b", re.IGNORECASE)

REWRITE_PROMPT = """Rewrite this browser-task instruction as the terse GOAL a real user would give.
Keep EVERY concrete parameter (names, titles, values, filters, quantities, dates) and the same
final outcome; drop only the route — no "open/click/scroll/navigate", no step lists, no UI tour.
One sentence when possible, two at most.

Instruction: {instruction}

Reply with ONLY the rewritten instruction."""


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def is_procedure_flavored(instruction: str) -> bool:
    text = (instruction or "").strip()
    if _STARTS_WITH_ROUTE.match(text):
        return True
    return len(_ROUTE_VERBS.findall(text)) >= 2


def find_targets(corpus_dir: Path) -> list[Path]:
    targets = []
    for traj_dir in sorted(corpus_dir.glob("**/task_*")):
        if not (traj_dir / "trajectory.json").exists() or not is_trainable(traj_dir):
            continue
        meta = _load(traj_dir / "metadata.json")
        if meta.get("instruction_style") == "goal_rewrite_v1":
            continue
        if is_procedure_flavored(_load(traj_dir / "trajectory.json").get("instruction", "")):
            targets.append(traj_dir)
    return targets


async def rewrite_one(traj_dir: Path) -> tuple[str, str, str]:
    """(status, old, new) — status is rewritten | rejected | unchanged."""
    traj_file = traj_dir / "trajectory.json"
    traj = _load(traj_file)
    old = traj.get("instruction", "")
    text = await run_agent(
        REWRITE_PROMPT.format(instruction=old),
        ClaudeAgentOptions(model=MODEL_FOR_GROUNDED_GENERATION, max_turns=1,
                           max_budget_usd=REWRITE_BUDGET_USD, permission_mode="bypassPermissions"),
        full_name=f"style:{traj_dir.name}",
        stage=7,
        purpose="instruction_style",
        raise_on_error=False,
    )
    new = (text or "").strip().strip('"')
    if not new or len(new) > 300 or "\n" in new or NOT_GOAL_SHAPED.search(new):
        return "rejected", old, new
    if new == old:
        return "unchanged", old, new
    traj["instruction"] = new
    traj_file.write_text(json.dumps(traj, indent=2))
    meta_file = traj_dir / "metadata.json"
    meta = _load(meta_file)
    meta.update({"instruction": new, "original_instruction": meta.get("original_instruction") or old,
                 "instruction_style": "goal_rewrite_v1"})
    meta_file.write_text(json.dumps(meta, indent=2))
    return "rewritten", old, new


async def run_apply(corpus_dir: Path, workers: int, limit: int | None) -> dict:
    targets = find_targets(corpus_dir)
    if limit:
        targets = targets[:limit]
    logger.info(f"{len(targets)} procedure-flavored instructions to rewrite")
    semaphore = asyncio.Semaphore(workers)
    outcomes: dict = {}
    rows: list = []

    async def guarded(traj_dir):
        async with semaphore:
            status, old, new = await rewrite_one(traj_dir)
        outcomes[status] = outcomes.get(status, 0) + 1
        rows.append((traj_dir, status, old, new))
        logger.info(f"  {traj_dir.name}: {status}")

    try:
        await asyncio.gather(*(guarded(d) for d in targets))
    except UsageLimitReached:
        logger.warning("usage limit reached — unmarked targets remain selectable; re-run to resume")

    lines = ["# Instruction style rewrite report", "", "| trajectory | status | old | new |",
             "| --- | --- | --- | --- |"]
    for traj_dir, status, old, new in sorted(rows, key=lambda r: str(r[0])):
        old, new = (s.replace("|", "/").replace("\n", " ")[:150] for s in (old, new))
        lines.append(f"| {traj_dir.relative_to(corpus_dir)} | {status} | {old} | {new} |")
    Path("instruction_style_report.md").write_text("\n".join(lines) + "\n")
    logger.info(f"report: instruction_style_report.md ({len(rows)} rows)")
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("trajectories/"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List flagged instructions and exit (default)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.apply:
        print(asyncio.run(run_apply(args.corpus, args.workers, args.limit)))
    else:
        targets = find_targets(args.corpus)
        for traj_dir in targets:
            instr = _load(traj_dir / "trajectory.json").get("instruction", "")
            print(f"{traj_dir.relative_to(args.corpus)}: {instr[:140]}")
        print(f"\n{len(targets)} procedure-flavored instructions flagged")
    sys.exit(0)


if __name__ == "__main__":
    main()
