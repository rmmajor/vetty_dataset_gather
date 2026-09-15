"""Audited hindsight-relabel pass over recoverable failed/partial agent captures."""

import asyncio
import logging
import re
import sys
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from cua.pipeline.agent_runner import run_agent
from cua.pipeline.corpus.cli import add_shared, build_parser
from cua.pipeline.corpus.pass_base import CorpusPass, Outcome
from cua.pipeline.llm_config import MODEL_FOR_GROUNDED_GENERATION
from cua.pipeline.sdk_runner import UsageLimitReached
from cua.pipeline.teacher import step_summary, verify_trajectory
from cua.settings import CAPTURE
from cua.trajectory.io import CAPTURE_META_FILE, METADATA_FILE, TRAJECTORY_FILE, TrajectoryDir

logger = logging.getLogger(__name__)

PROPOSE_BUDGET_USD = 0.10
RELABEL_REPORT = Path("relabel_report.md")

# The June failure mode: a proposal that reads as a browsing procedure instead of a user goal ("navigate to the compress page").
NOT_GOAL_SHAPED = re.compile(
    r"^(navigate|go to|open|visit|click|scroll|attempt|try to|load)\b|screenshot|\bstep\b", re.IGNORECASE
)

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


async def propose_instruction(recorded: TrajectoryDir) -> tuple[str | None, str]:
    """(proposal, reject_reason) — a goal-shaped instruction for what the run accomplished, or None."""
    traj = recorded.trajectory
    meta = recorded.metadata
    cmeta = recorded.capture_meta
    prompt = PROPOSE_PROMPT.format(
        domain=meta.get("website") or meta.get("domain") or "a website",
        instruction=traj.get("instruction", ""),
        n_steps=len(traj.get("steps", [])),
        step_summary=step_summary(traj),
        agent_summary=(cmeta.get("agent_summary") or "(none)")[:800],
    )
    text = await run_agent(
        prompt,
        ClaudeAgentOptions(
            model=MODEL_FOR_GROUNDED_GENERATION,
            max_turns=1,
            max_budget_usd=PROPOSE_BUDGET_USD,
            permission_mode="bypassPermissions",
        ),
        full_name=f"relabel:{recorded.path.name}",
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
    recorded = TrajectoryDir.load(traj_dir)
    traj = recorded.trajectory
    steps = traj.get("steps", [])
    if any(a.get("type") == "done" for s in steps for a in s.get("actions", [])):
        return
    screenshot = "final_screenshot.png"
    if not (traj_dir / screenshot).exists():
        if not steps:
            return
        screenshot = steps[-1].get("screenshot", "")
    steps.append(
        {
            "step": (steps[-1]["step"] + 1) if steps else 1,
            "screenshot": screenshot,
            "thought": f"The goal is accomplished: {proposal.rstrip('.')}. I stop here.",
            "actions": [{"type": "done"}],
        }
    )
    traj["total_steps"] = len(steps)
    recorded.save_trajectory()


class RelabelPass(CorpusPass):
    """Propose a goal for what a partial run finished, re-judge it with the SAME verifier, and keep it only on a fresh success."""

    name = "relabel"
    workers = 2
    report_path = RELABEL_REPORT

    def select(self, recorded: TrajectoryDir) -> bool:
        traj, meta, cmeta = recorded.trajectory, recorded.metadata, recorded.capture_meta
        if traj.get("success"):
            return False
        if meta.get("excluded_reason") or cmeta.get("excluded_reason"):
            return False
        if (meta.get("capture_source") or "agent") != "agent":
            return False
        if meta.get("relabeled_from") or meta.get("relabel_attempted") or meta.get("wall_demo"):
            return False
        if cmeta.get("agent_outcome") not in ("partial", "completed"):
            return False
        if (cmeta.get("verifier") or {}).get("degenerate"):
            return False
        return len([s for s in traj.get("steps", []) if s.get("actions")]) >= CAPTURE.relabel_min_steps

    async def apply(self, recorded: TrajectoryDir) -> Outcome:
        traj_dir = recorded.path
        snapshots = {
            path: path.read_bytes()
            for path in (traj_dir / TRAJECTORY_FILE, traj_dir / CAPTURE_META_FILE, traj_dir / METADATA_FILE)
            if path.exists()
        }
        original_instruction = recorded.trajectory.get("instruction", "")

        proposal, reject_reason = await propose_instruction(recorded)
        if proposal is None:
            rejected = TrajectoryDir.load(traj_dir)
            rejected.metadata["relabel_attempted"] = f"hindsight_v2_rejected: {reject_reason[:120]}"
            rejected.save_metadata()
            return Outcome(changed=True, note="no_proposal", detail=reject_reason)

        proposed = TrajectoryDir.load(traj_dir)
        proposed.trajectory["instruction"] = proposal
        proposed.save_trajectory()
        try:
            verdict = await verify_trajectory(traj_dir, full_name=f"relabel:{traj_dir.name}")
        except UsageLimitReached:
            for path, blob in snapshots.items():
                path.write_bytes(blob)
            raise

        if not verdict.get("final_success"):
            for path, blob in snapshots.items():
                path.write_bytes(blob)
            # Loaded only after the restore, so the rejection lands on the ORIGINAL metadata, not the proposed one.
            declined = TrajectoryDir.load(traj_dir)
            declined.metadata["relabel_attempted"] = "hindsight_v2_rejected: verifier declined the relabeled goal"
            declined.save_metadata()
            return Outcome(changed=True, note="verifier_rejected", detail=proposal)

        _append_done_step(traj_dir, proposal)
        accepted = TrajectoryDir.load(traj_dir)
        verifier = accepted.capture_meta.get("verifier") or {}
        accepted.metadata.update(
            {
                "instruction": proposal,
                "original_instruction": original_instruction,
                "relabeled": True,
                "relabeled_from": "hindsight_v2",
                "success": True,
                "efficiency": verifier.get("efficiency"),
                "efficiency_ratio": verifier.get("efficiency_ratio"),
                # The thoughts still pursue the ORIGINAL goal, so marking them raw sends them through the polish pass.
                "thought_source": "teacher_raw",
            }
        )
        accepted.save_metadata()
        logger.info(f"  {traj_dir.name}: relabeled -> {proposal!r}")
        return Outcome(changed=True, note="relabeled", detail=proposal)

    def summary(self, corpus: Path, candidates: list) -> str:
        return f"{len(candidates)} relabel candidates"

    def describe(self, corpus: Path, recorded: TrajectoryDir) -> str:
        instruction = recorded.trajectory.get("instruction", "").replace("|", "/")[:100]
        return (
            f"| {recorded.path.relative_to(corpus)} | {recorded.capture_meta.get('agent_outcome')} "
            f"| {len(recorded.trajectory.get('steps', []))} | {instruction} |"
        )

    def print_dry_run(self, corpus: Path, candidates: list) -> None:
        print(f"# {len(candidates)} relabel candidates (no writes)\n")
        print("| trajectory | outcome | steps | original instruction |")
        print("| --- | --- | --- | --- |")
        for recorded in candidates:
            print(self.describe(corpus, recorded))
        print(f"\norphans (never judged): {len(JudgeOrphansPass().candidates(corpus))}")

    def report(self, corpus: Path, outcomes: list) -> str:
        lines = ["# Hindsight relabel report", "", "| trajectory | outcome | detail |", "| --- | --- | --- |"]
        for outcome in sorted(outcomes, key=lambda o: str(o.path)):
            detail = str(outcome.detail).replace("|", "/")[:160]
            lines.append(f"| {outcome.path.relative_to(corpus)} | {outcome.note} | {detail} |")
        return "\n".join(lines) + "\n"


class JudgeOrphansPass(CorpusPass):
    """Agent captures that were never judged at all (no capture_meta.verifier) get a first verdict."""

    name = "judge_orphans"
    workers = 2

    def select(self, recorded: TrajectoryDir) -> bool:
        meta, cmeta = recorded.metadata, recorded.capture_meta
        if meta.get("excluded_reason") or cmeta.get("excluded_reason"):
            return False
        if (meta.get("capture_source") or "agent") != "agent":
            return False
        return "verifier" not in cmeta and cmeta.get("recorder") == "browser_driver"

    async def apply(self, recorded: TrajectoryDir) -> Outcome:
        verdict = await verify_trajectory(recorded.path, full_name=f"orphan:{recorded.path.name}")
        judged = bool(verdict.get("final_success"))
        return Outcome(changed=True, note="success" if judged else "failed")

    def summary(self, corpus: Path, candidates: list) -> str:
        return f"{len(candidates)} never-judged agent captures"

    def observe(self, corpus: Path, recorded: TrajectoryDir, outcome: Outcome) -> None:
        logger.info(f"  {recorded.path.name}: {outcome.note}")


def main():
    parser = build_parser(
        __doc__,
        "corpus",
        "dry_run",
        "apply",
        dry_run={"help": "List candidates, write nothing"},
        apply={"help": "Propose + re-verify + write"},
    )
    parser.add_argument("--judge-orphans", action="store_true", help="Run the verifier on never-judged dirs")
    add_shared(parser, "workers", "limit", workers={"default": 2})
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.judge_orphans:
        print(asyncio.run(JudgeOrphansPass().run(args.corpus, limit=args.limit, workers=args.workers)))
    elif args.apply:
        print(asyncio.run(RelabelPass().run(args.corpus, limit=args.limit, workers=args.workers)))
    else:
        asyncio.run(RelabelPass().run(args.corpus, apply=False))
        if not args.dry_run:
            print("\n(pass --apply to execute, --judge-orphans for the verdict-less dirs)")
    sys.exit(0)


if __name__ == "__main__":
    main()
