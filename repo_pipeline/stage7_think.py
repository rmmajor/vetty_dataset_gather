"""Stage 7 — supervision finalize.

Decoupled from capture (stage 6): captures land fast and robustly; this stage finishes the
per-step `<think>` supervision afterwards over the captured corpus. Two paths, by provenance:

  * agent-captured trajectories arrive with the teacher's raw, tool-aware reasoning in every
    step — those get POLISHED (one text-only call per trajectory) into clean student-voice
    thoughts, flipping metadata.thought_source from `teacher_raw` to `teacher`;
  * replay trajectories arrive with empty thoughts — those get GENERATED step-by-step from
    the screenshots (the original vision path).

A captured repo is marked `completed` once every trajectory is finalized; if some remain (a
transient SDK stall, a rate limit), the repo simply stays `captured` and is picked up again
next run — it is never rejected, because the trajectories themselves are already good.
"""

import asyncio
import json
import logging
from pathlib import Path

from repo_pipeline.logging_utils import repo_logger
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.schemas import RepoStatus
from repo_pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from repo_pipeline.think_generator import (
    generate_thinks_for_trajectory,
    needs_polish,
    polish_thoughts_for_trajectory,
)

logger = logging.getLogger(__name__)


def thoughts_complete(repo_traj_dir: Path) -> bool:
    """True when every generatable step (has an action and an on-disk screenshot) in every
    trajectory of this repo has a thought AND no trajectory still carries raw (unpolished)
    teacher reasoning. Steps with a missing screenshot can't be generated, so they don't block
    completion."""
    task_dirs = list(repo_traj_dir.glob("task_*")) if repo_traj_dir.exists() else []
    if not task_dirs:
        return False
    for task_dir in task_dirs:
        trajectory_file = task_dir / "trajectory.json"
        if not trajectory_file.exists():
            continue
        if needs_polish(task_dir):
            return False
        try:
            steps = json.loads(trajectory_file.read_text()).get("steps", [])
        except (json.JSONDecodeError, OSError):
            return False
        for step in steps:
            if not step.get("actions"):
                continue
            if not (task_dir / step.get("screenshot", "")).exists():
                continue
            if not (step.get("thought") or "").strip():
                return False
    return True


async def think_for_repo(registry: RepoRegistry, repo: dict):
    """Fill think blocks for one captured repo, then mark it completed if all thoughts are present."""
    repo_name = repo["full_name"]
    if repo.get("current_stage", 0) >= 7:
        return
    rlog = repo_logger(7, repo_name)

    traj_dir = Path(repo.get("trajectory_dir") or "")
    if not traj_dir.exists():
        rlog.warning(f"trajectory_dir missing ({traj_dir}); cannot generate thoughts")
        return

    task_dirs = sorted(traj_dir.glob("task_*"))
    rlog.info(f"finalizing supervision across {len(task_dirs)} trajectories")
    for task_dir in task_dirs:
        if needs_polish(task_dir):
            # Teacher trajectory: raw reasoning is already present — one text-only polish call.
            await polish_thoughts_for_trajectory(task_dir, registry=registry, full_name=repo_name)
        else:
            # Replay trajectory: thoughts are empty — generate from screenshots through one
            # persistent session (see think_generator), sequential within a trajectory.
            await generate_thinks_for_trajectory(task_dir, registry=registry, full_name=repo_name)

    if thoughts_complete(traj_dir):
        registry.update_status(repo_name, RepoStatus.COMPLETED.value, stage=7)
        rlog.info(f"all think blocks complete -> completed ({repo.get('trajectory_count', 0)} trajectories)")
    else:
        rlog.info("think blocks still incomplete -> staying captured (will resume next run)")


async def run_think(
    registry: RepoRegistry, trajectory_dir=None, limit: int = None, workers: int = 2, budget_remaining: float = None
):
    repos = registry.get_repos_for_stage(7, limit=limit)
    logger.info(f"Think generation for {len(repos)} captured repos with {workers} workers")

    semaphore = asyncio.Semaphore(workers)
    completed = 0
    still_captured = 0
    trajectories = 0

    async def process(repo):
        nonlocal completed, still_captured, trajectories
        if rate_limit_tracker.limit_reached():
            return
        async with semaphore:
            try:
                await think_for_repo(registry, repo)
            except UsageLimitReached:
                return  # quota hit — leave captured, resume next run
            except Exception as e:
                logger.error(f"Error generating thoughts for {repo['full_name']}: {e}")
        after = registry.get_repo(repo["full_name"])
        if after and after["status"] == RepoStatus.COMPLETED.value:
            completed += 1
            trajectories += after.get("trajectory_count", 0) or 0
        else:
            still_captured += 1

    await asyncio.gather(*[process(repo) for repo in repos])

    logger.info(
        f"[stage 7] complete — {completed} repos completed, {still_captured} still captured, "
        f"{trajectories} trajectories finished"
    )
    return {
        "processed": len(repos),
        "passed": completed,
        "rejected": 0,
        "still_captured": still_captured,
        "trajectories": trajectories,
    }
