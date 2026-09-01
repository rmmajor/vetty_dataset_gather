"""Audited hindsight-relabel pass over recoverable failed/partial agent captures.

Capture-time hindsight-reclassification stays forbidden (agent_capture.verify_trajectory); this is
the separate, reviewable pass the corpus round agreed on: propose a goal-shaped instruction for what
the run ACTUALLY accomplished, re-judge the trajectory against that NEW instruction with the very
same verifier, and keep it only on a fresh success verdict. Every write is provenance-marked
(metadata.relabeled_from="hindsight_v2", original_instruction kept) and rejected attempts restore
the original files byte-for-byte.

  python -m repo_pipeline.relabel_recoverable trajectories/ --dry-run   # review candidates first
  python -m repo_pipeline.relabel_recoverable trajectories/ --judge-orphans   # verdict-less dirs
  python -m repo_pipeline.relabel_recoverable trajectories/ --apply [--limit N]
"""
import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from repo_pipeline.agent_capture import RELABEL_MIN_STEPS, _step_summary, verify_trajectory
from repo_pipeline.agent_runner import run_agent
from repo_pipeline.llm_config import MODEL_FOR_GROUNDED_GENERATION
from repo_pipeline.sdk_runner import UsageLimitReached

logger = logging.getLogger(__name__)

PROPOSE_BUDGET_USD = 0.10

# The June failure mode: "compress a PDF" relabeled to "navigate to the compress page". A proposed
# instruction that reads as a browsing procedure instead of a user goal is rejected outright.
NOT_GOAL_SHAPED = re.compile(
    r"^(navigate|go to|open|visit|click|scroll|attempt|try to|load)\b|screenshot|\bstep\b", re.IGNORECASE)

PROPOSE_PROMPT = """A browser agent attempted a task on {domain} but only got part of the way. \
Below is what it did. Name the user goal that the run ACTUALLY accomplished end to end, as one \
terse instruction a real user would give (imperative, outcome-focused, e.g. "Create a project \
board named Roadmap"). Never phrase it as navigation or a procedure ("go to", "click", "open the \
page"), never mention screenshots or steps, and never include more than the run visibly finished.

Original task (context only — the run did NOT finish this): {instruction}

Actions performed ({n_steps} steps):
{step_summary}

Agent's own closing report: {agent_summary}

If the run finished nothing a user would ask for, reply exactly NONE.
Reply with ONLY the instruction text (or NONE)."""


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def find_candidates(corpus_dir: Path) -> list[dict]:
    candidates = []
    for traj_dir in sorted(corpus_dir.glob("**/task_*")):
        if not (traj_dir / "trajectory.json").exists():
            continue
        traj = _load(traj_dir / "trajectory.json")
        meta = _load(traj_dir / "metadata.json")
        cmeta = _load(traj_dir / "capture_meta.json")
        if traj.get("success"):
            continue
        if meta.get("excluded_reason") or cmeta.get("excluded_reason"):
            continue
        if (meta.get("capture_source") or "agent") != "agent":
            continue
        if meta.get("relabeled_from") or meta.get("relabel_attempted") or meta.get("wall_demo"):
            continue
        if cmeta.get("agent_outcome") not in ("partial", "completed"):
            continue
        if ((cmeta.get("verifier") or {}).get("degenerate")):
            continue
        steps = [s for s in traj.get("steps", []) if s.get("actions")]
        if len(steps) < RELABEL_MIN_STEPS:
            continue
        candidates.append({"dir": traj_dir, "trajectory": traj, "metadata": meta, "capture_meta": cmeta})
    return candidates


def find_orphans(corpus_dir: Path) -> list[Path]:
    """Agent captures that were never judged at all (no capture_meta.verifier)."""
    orphans = []
    for traj_dir in sorted(corpus_dir.glob("**/task_*")):
        if not (traj_dir / "trajectory.json").exists():
            continue
        meta = _load(traj_dir / "metadata.json")
        cmeta = _load(traj_dir / "capture_meta.json")
        if meta.get("excluded_reason") or cmeta.get("excluded_reason"):
            continue
        if (meta.get("capture_source") or "agent") != "agent":
            continue
        if "verifier" not in cmeta and cmeta.get("recorder") == "browser_driver":
            orphans.append(traj_dir)
    return orphans


async def propose_instruction(candidate: dict) -> tuple[str | None, str]:
    """(proposal, reject_reason) — a goal-shaped instruction for what the run accomplished, or None."""
    traj = candidate["trajectory"]
    meta = candidate["metadata"]
    cmeta = candidate["capture_meta"]
    prompt = PROPOSE_PROMPT.format(
        domain=meta.get("website") or meta.get("domain") or "a website",
        instruction=traj.get("instruction", ""),
        n_steps=len(traj.get("steps", [])),
        step_summary=_step_summary(traj),
        agent_summary=(cmeta.get("agent_summary") or "(none)")[:800],
    )
    text = await run_agent(
        prompt,
        ClaudeAgentOptions(model=MODEL_FOR_GROUNDED_GENERATION, max_turns=1,
                           max_budget_usd=PROPOSE_BUDGET_USD, permission_mode="bypassPermissions"),
        full_name=f"relabel:{candidate['dir'].name}",
        stage=7,
        purpose="relabel_propose",
        raise_on_error=False,
    )
    proposal = (text or "").strip().strip('"')
    if not proposal or proposal.upper() == "NONE":
        return None, "proposer found nothing accomplished"
    if len(proposal) > 220 or "\n" in proposal:
        return None, f"proposal not terse: {proposal[:80]}…"
    if NOT_GOAL_SHAPED.search(proposal):
        return None, f"not goal-shaped: {proposal[:80]}"
    return proposal, ""


def _append_done_step(traj_dir: Path, proposal: str):
    """Terminal stop supervision for a relabeled run, mirroring convert's synthesized-done shape."""
    traj_file = traj_dir / "trajectory.json"
    traj = _load(traj_file)
    steps = traj.get("steps", [])
    if any(a.get("type") == "done" for s in steps for a in s.get("actions", [])):
        return
    screenshot = "final_screenshot.png"
    if not (traj_dir / screenshot).exists():
        if not steps:
            return
        screenshot = steps[-1].get("screenshot", "")
    steps.append({
        "step": (steps[-1]["step"] + 1) if steps else 1,
        "screenshot": screenshot,
        "thought": f"The goal is accomplished: {proposal.rstrip('.')}. I stop here.",
        "actions": [{"type": "done"}],
    })
    traj["total_steps"] = len(steps)
    traj_file.write_text(json.dumps(traj, indent=2))


async def apply_one(candidate: dict, report_rows: list) -> str:
    traj_dir = candidate["dir"]
    traj_file = traj_dir / "trajectory.json"
    cmeta_file = traj_dir / "capture_meta.json"
    meta_file = traj_dir / "metadata.json"
    snapshots = {p: p.read_bytes() for p in (traj_file, cmeta_file, meta_file) if p.exists()}
    original_instruction = candidate["trajectory"].get("instruction", "")

    proposal, reject_reason = await propose_instruction(candidate)
    if proposal is None:
        meta = _load(meta_file)
        meta["relabel_attempted"] = f"hindsight_v2_rejected: {reject_reason[:120]}"
        meta_file.write_text(json.dumps(meta, indent=2))
        report_rows.append((traj_dir, "no_proposal", reject_reason))
        return "no_proposal"

    traj = _load(traj_file)
    traj["instruction"] = proposal
    traj_file.write_text(json.dumps(traj, indent=2))
    try:
        verdict = await verify_trajectory(traj_dir, full_name=f"relabel:{traj_dir.name}")
    except UsageLimitReached:
        for path, blob in snapshots.items():
            path.write_bytes(blob)
        raise

    if not verdict.get("final_success"):
        for path, blob in snapshots.items():
            path.write_bytes(blob)
        meta = _load(meta_file)
        meta["relabel_attempted"] = "hindsight_v2_rejected: verifier declined the relabeled goal"
        meta_file.write_text(json.dumps(meta, indent=2))
        report_rows.append((traj_dir, "verifier_rejected", proposal))
        return "verifier_rejected"

    _append_done_step(traj_dir, proposal)
    cmeta = _load(cmeta_file)
    verifier = cmeta.get("verifier") or {}
    meta = _load(meta_file)
    meta.update({
        "instruction": proposal,
        "original_instruction": original_instruction,
        "relabeled": True,
        "relabeled_from": "hindsight_v2",
        "success": True,
        "efficiency": verifier.get("efficiency"),
        "efficiency_ratio": verifier.get("efficiency_ratio"),
        # The thoughts still pursue the ORIGINAL goal (naming sub-goals the new instruction drops);
        # marking them raw sends them through the polish pass, which rewrites against the new goal.
        "thought_source": "teacher_raw",
    })
    meta_file.write_text(json.dumps(meta, indent=2))
    report_rows.append((traj_dir, "relabeled", proposal))
    logger.info(f"  {traj_dir.name}: relabeled -> {proposal!r}")
    return "relabeled"


async def run_apply(corpus_dir: Path, workers: int, limit: int | None) -> dict:
    candidates = find_candidates(corpus_dir)
    if limit:
        candidates = candidates[:limit]
    logger.info(f"{len(candidates)} relabel candidates")
    semaphore = asyncio.Semaphore(workers)
    outcomes: dict = {}
    report_rows: list = []

    async def guarded(candidate):
        async with semaphore:
            outcome = await apply_one(candidate, report_rows)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    try:
        await asyncio.gather(*(guarded(c) for c in candidates))
    except UsageLimitReached:
        logger.warning("usage limit reached — untouched candidates remain selectable; re-run to resume")
    _write_report(corpus_dir, report_rows, "relabel_report.md")
    return outcomes


async def run_judge_orphans(corpus_dir: Path, workers: int, limit: int | None) -> dict:
    orphans = find_orphans(corpus_dir)
    if limit:
        orphans = orphans[:limit]
    logger.info(f"{len(orphans)} never-judged agent captures")
    semaphore = asyncio.Semaphore(workers)
    outcomes: dict = {}

    async def judge(traj_dir):
        async with semaphore:
            verdict = await verify_trajectory(traj_dir, full_name=f"orphan:{traj_dir.name}")
        key = "success" if verdict.get("final_success") else "failed"
        outcomes[key] = outcomes.get(key, 0) + 1
        logger.info(f"  {traj_dir.name}: {key}")

    try:
        await asyncio.gather(*(judge(d) for d in orphans))
    except UsageLimitReached:
        logger.warning("usage limit reached — remaining orphans stay verdict-less; re-run to resume")
    return outcomes


def _write_report(corpus_dir: Path, rows: list, name: str):
    lines = ["# Hindsight relabel report", "", "| trajectory | outcome | detail |", "| --- | --- | --- |"]
    for traj_dir, outcome, detail in sorted(rows, key=lambda r: str(r[0])):
        detail = str(detail).replace("|", "/")[:160]
        lines.append(f"| {traj_dir.relative_to(corpus_dir)} | {outcome} | {detail} |")
    path = Path(name)
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"report: {path} ({len(rows)} rows)")


def print_dry_run(corpus_dir: Path):
    candidates = find_candidates(corpus_dir)
    print(f"# {len(candidates)} relabel candidates (no writes)\n")
    print("| trajectory | outcome | steps | original instruction |")
    print("| --- | --- | --- | --- |")
    for c in candidates:
        instr = c["trajectory"].get("instruction", "").replace("|", "/")[:100]
        print(f"| {c['dir'].relative_to(corpus_dir)} | {c['capture_meta'].get('agent_outcome')} "
              f"| {len(c['trajectory'].get('steps', []))} | {instr} |")
    print(f"\norphans (never judged): {len(find_orphans(corpus_dir))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("trajectories/"))
    parser.add_argument("--dry-run", action="store_true", help="List candidates, write nothing")
    parser.add_argument("--apply", action="store_true", help="Propose + re-verify + write")
    parser.add_argument("--judge-orphans", action="store_true", help="Run the verifier on never-judged dirs")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.judge_orphans:
        print(asyncio.run(run_judge_orphans(args.corpus, args.workers, args.limit)))
    elif args.apply:
        print(asyncio.run(run_apply(args.corpus, args.workers, args.limit)))
    else:
        print_dry_run(args.corpus)
        if not args.dry_run:
            print("\n(pass --apply to execute, --judge-orphans for the verdict-less dirs)")
    sys.exit(0)


if __name__ == "__main__":
    main()
