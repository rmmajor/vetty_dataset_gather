import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from repo_pipeline import schemas
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.schemas import (
    NON_QUALITY_CATEGORIES,
    STAGE_BEHAVIOR_CUTOFFS,
    STAGE_TO_REJECTED_STATUS,
)
from repo_pipeline.sdk_runner import cost_tracker, rate_limit_tracker

logger = logging.getLogger(__name__)


def requeue_stale(registry: RepoRegistry, dry_run: bool = False) -> int:
    """Re-queue rejections produced by superseded code: for each stage with a known behavior
    cutoff, reset rejections recorded before that cutoff back to the stage's input status."""
    verb = "would re-queue" if dry_run else "re-queued"
    total = 0
    for stage, cutoff in sorted(STAGE_BEHAVIOR_CUTOFFS.items()):
        n = registry.reset_rejected(stage=stage, before=cutoff, dry_run=dry_run)
        logger.info(f"  stage {stage}: {verb} {n} stale rejections (before {cutoff})")
        total += n
    logger.info(f"Stale re-queue: {verb} {total} repos total{' (dry-run)' if dry_run else ''}")
    return total


def requeue_infra(registry: RepoRegistry, dry_run: bool = False) -> int:
    """Re-queue infra/flaky rejections (clone/SDK/timeout/max-turns/...) across all stages,
    leaving genuine quality rejections in place."""
    verb = "would re-queue" if dry_run else "re-queued"
    n = registry.reset_rejected(category_in=NON_QUALITY_CATEGORIES, dry_run=dry_run)
    logger.info(f"Infra re-queue: {verb} {n} infra/flaky rejections{' (dry-run)' if dry_run else ''}")
    return n


def requeue_rescore(registry: RepoRegistry, stage: int, min_score: float, dry_run: bool = False) -> int:
    """Re-queue rejections a lowered gate would now accept: reset stage-N rejects whose stored gate
    score is >= min_score back to that stage's input status, so re-running stage N re-admits them."""
    verb = "would re-queue" if dry_run else "re-queued"
    n = registry.reset_rejected(stage=stage, min_score=min_score, dry_run=dry_run)
    logger.info(f"Rescore re-queue: {verb} {n} stage-{stage} rejections with score >= {min_score}"
                f"{' (dry-run)' if dry_run else ''}. Re-run --stage {stage} to re-admit them.")
    return n


def setup_logging(data_dir: Path, verbose: bool = False):
    log_dir = data_dir / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    from datetime import datetime
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_file


def print_status(registry: RepoRegistry, stages: list[int] = None):
    """Print the pipeline registry status.

    ``stages``: when set (e.g. by --stage N), an extra "current stage" rejection
    section is printed for each stage in the list, on top of the all-stages
    aggregate — so a stage-3 run shows both the all-time top reasons and only the
    reasons added at stage 3.
    """
    stats = registry.get_stats()
    pending = registry.get_pending_counts()
    domain_dist = registry.get_domain_distribution()
    rejection_reasons = registry.get_rejection_reasons(limit=10)
    total_traj = registry.count_total_trajectories()

    print("\n=== Pipeline Registry Status ===\n")

    print("Status counts:")
    # The 7-stage funnel — always shown (even at 0) so the full pipeline is visible at a glance.
    progress_statuses = [
        "discovered", "filtered", "scored", "analyzed", "validated", "captured", "completed",
    ]
    # Rejection buckets — only shown when non-empty.
    rejection_statuses = [
        "rejected_filter", "rejected_score", "rejected_analyze", "rejected_trial", "rejected_generate",
    ]
    for status in progress_statuses:
        print(f"  {status:25s} {stats.get(status, 0):>6,}")
    for status in rejection_statuses:
        count = stats.get(status, 0)
        if count > 0:
            print(f"  {status:25s} {count:>6,}")

    total = sum(stats.values())
    print(f"  {'TOTAL':25s} {total:>6,}")
    print(f"\n  Total trajectories generated: {total_traj:,}")

    print("\nPending for next run:")
    for stage in [3, 4, 5, 6, 7]:
        count = pending.get(stage, 0)
        print(f"  Stage {stage}: {count:,} repos")

    if domain_dist:
        print("\nDomain distribution (non-rejected repos):")
        for domain, count in domain_dist.items():
            print(f"  {domain:20s} {count:>6,}")

    category_dist = registry.get_failure_category_distribution()
    if category_dist:
        print("\nRejection categories (quality vs infra):")
        for cat, count in category_dist:
            print(f"  {cat:20s} {count:>6,}")

    if rejection_reasons:
        print("\nTop rejection reasons (all stages, all time):")
        for i, (reason, count) in enumerate(rejection_reasons, 1):
            print(f"  {i}. \"{reason}\" ({count:,})")

    if stages:
        for stage in stages:
            rejected_status = STAGE_TO_REJECTED_STATUS.get(stage)
            if rejected_status is None:
                continue
            per_stage = registry.get_rejection_reasons(limit=5, status=rejected_status.value)
            if not per_stage:
                continue
            print(f"\nTop rejection reasons (stage {stage} — {rejected_status.value}):")
            for i, (reason, count) in enumerate(per_stage, 1):
                print(f"  {i}. \"{reason}\" ({count:,})")

    print()


def parse_stage_range(stage_str: str) -> list[int]:
    if "-" in stage_str:
        start, end = stage_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(stage_str)]


async def run_stage(
    stage: int,
    registry: RepoRegistry,
    github_token: str,
    trajectory_dir: Path,
    limit: int = None,
    workers: int = 5,
    budget_remaining: float = None,
    star_floor: int = 5,
) -> dict:
    if stage == 1:
        from repo_pipeline.stage1_discover import run_discovery
        return run_discovery(registry, github_token, star_floor=star_floor)

    elif stage == 2:
        from repo_pipeline.stage2_filter import run_filter
        return run_filter(registry, limit=limit)

    elif stage == 3:
        from repo_pipeline.stage3_shallow_score import run_shallow_score
        return await run_shallow_score(registry, limit=limit, workers=workers, budget_remaining=budget_remaining)

    elif stage == 4:
        from repo_pipeline.stage4_deep_analyze import run_deep_analyze
        return await run_deep_analyze(registry, limit=limit, workers=min(workers, 3), budget_remaining=budget_remaining)

    elif stage == 5:
        from repo_pipeline.stage5_trial_run import run_trial
        return await run_trial(registry, limit=limit, workers=min(workers, 2), budget_remaining=budget_remaining)

    elif stage == 6:
        from repo_pipeline.stage6_generate import run_generate
        # Each stage-6 worker holds a sandbox with a booted app + headless Chromium (~1.2GB
        # real). Default 2 fits alongside the other stages in an ~8GB Docker VM; raise via
        # env when the VM has headroom or other lanes are idle.
        stage6_cap = int(os.environ.get("STAGE6_WORKERS", "2"))
        return await run_generate(registry, trajectory_dir, limit=limit, workers=min(workers, stage6_cap), budget_remaining=budget_remaining)

    elif stage == 7:
        from repo_pipeline.stage7_think import run_think
        return await run_think(registry, trajectory_dir, limit=limit, workers=min(workers, 3), budget_remaining=budget_remaining)

    else:
        raise ValueError(f"Unknown stage: {stage}")


# Per-repo LLM cost estimates used only for --budget pacing (not metered spend).
# Stage 5 is code-driven (~$0 LLM). Stage 6 depends on the capture mode: agent capture runs
# the task proposer + ~8 multi-turn teacher tasks + per-task verification per repo; replay
# only does the 1-per-trajectory instruction + task-type calls.
STAGE_COST_ESTIMATES = {
    3: 0.15,   # Sonnet shallow scoring
    4: 0.20,   # Sonnet static deep-analysis
    5: 0.02,   # code-driven boot gate / trial — negligible LLM
    7: 0.50,   # finalize: polish (agent, text-only) / per-step think-with-image (replay)
}
STAGE6_COST_BY_MODE = {"agent": 6.00, "replay": 0.20}


def stage_cost(stage: int) -> float:
    if stage == 6:
        return STAGE6_COST_BY_MODE.get(schemas.CAPTURE_MODE, STAGE6_COST_BY_MODE["agent"])
    return STAGE_COST_ESTIMATES.get(stage, 0.10)


async def drain_funnel(
    registry: RepoRegistry,
    github_token: str,
    trajectory_dir: Path,
    budget: float = 50.0,
    workers: int = 5,
    limit: int = None,
    loop: bool = False,
):
    """Run stages 3->6 over pending repos, capping each stage by --budget affordability and
    stopping when the budget is spent OR the Claude subscription usage limit is reached.

    ``loop=False`` (overnight): one forward pass — each stage feeds the next within the pass.
    ``loop=True`` (until-limit): repeat passes until the usage limit / budget stops it, or a
    full pass processes 0 repos (the funnel is drained). The funnel strictly drains because
    stages 1-2 are not looped, so no new ``filtered`` repos are seeded mid-run.

    Both modes honour ``rate_limit_tracker.limit_reached()``: a hard 429 (or, when a threshold
    is set, crossing it) stops the run instead of letting repos be rejected.
    """
    start_time = time.time()
    # Budget is now paced by ACTUAL metered spend (every ResultMessage's total_cost_usd lands
    # in cost_tracker), not by per-repo estimates — those drifted badly enough to stop a run
    # at "$30 estimated" while the real session window sat at 10%. Estimates below only
    # pre-size batches, never charge the budget.
    cost_baseline = cost_tracker.total_usd
    stage_results = {}
    stage_errors: list[str] = []

    def budget_used() -> float:
        return cost_tracker.total_usd - cost_baseline

    def should_stop() -> bool:
        if budget_used() >= budget:
            logger.warning(f"Budget exhausted (${budget_used():.2f}/${budget:.2f} metered), stopping")
            return True
        if rate_limit_tracker.limit_reached():
            util = rate_limit_tracker.max_utilization()
            util_str = f"{util*100:.0f}%" if util is not None else "hard 429"
            logger.warning(f"Claude usage limit reached ({util_str}), stopping")
            return True
        return False

    while True:
        pass_processed = 0

        for stage in [3, 4, 5, 6, 7]:
            if should_stop():
                break

            remaining_budget = budget - budget_used()
            logger.info(f"--- Stage {stage} --- (budget remaining: ${remaining_budget:.2f} of ${budget:.2f}, metered)")

            pending = registry.get_pending_counts()
            pending_count = pending.get(stage, 0)
            if pending_count == 0:
                logger.info(f"  No repos pending for stage {stage}, skipping")
                continue

            cost_per_repo = stage_cost(stage)
            max_affordable = int(remaining_budget / cost_per_repo) if cost_per_repo > 0 else pending_count
            effective_limit = min(pending_count, max_affordable)
            if limit:
                effective_limit = min(effective_limit, limit)

            logger.info(f"  Processing up to {effective_limit} repos (of {pending_count} pending)")

            try:
                result = await run_stage(
                    stage,
                    registry,
                    github_token,
                    trajectory_dir,
                    limit=effective_limit,
                    workers=workers,
                    budget_remaining=remaining_budget,
                )
                processed = result.get("processed", 0)
                pass_processed += processed

                # Accumulate processed across passes; keep the latest registry-derived
                # passed/rejected/trajectories (those are cumulative totals already).
                prev_processed = stage_results.get(stage, {}).get("processed", 0)
                stage_results[stage] = {**result, "processed": prev_processed + processed}

            except Exception as e:
                logger.error(f"  Stage {stage} failed: {e}")
                stage_results.setdefault(stage, {"processed": 0, "passed": 0, "rejected": 0})
                stage_results[stage]["error"] = str(e)
                # A crashed stage starves everything downstream of it. Record it loudly: a
                # stage-4 NameError once ran a whole night looking like "0 repos pending".
                err = f"stage {stage}: {type(e).__name__}: {str(e)[:200]}"
                if err not in stage_errors:
                    stage_errors.append(err)

        if not loop:
            break
        if should_stop():
            break
        if pass_processed == 0:
            logger.info("Funnel drained (a full pass processed 0 repos), stopping")
            break

    run_duration = time.time() - start_time

    if loop and rate_limit_tracker.threshold < 1.0 and rate_limit_tracker.observed_event_count == 0:
        logger.warning(
            f"--until-limit {rate_limit_tracker.threshold * 100:.0f}%% was requested but zero "
            f"RateLimitEvents were observed; the run effectively used 'until hard 429 / funnel "
            f"drained' instead. Headless OAuth may not emit rate_limit_event messages."
        )

    from repo_pipeline.notifier import send_report
    try:
        await send_report(
            registry, run_duration, budget_used(), budget, stage_results,
            rate_limit_info=rate_limit_tracker.latest(),
        )
    except Exception as e:
        logger.error(f"Failed to send Telegram report: {e}")

    if stage_errors:
        logger.warning("RUN DEGRADED — one or more stages crashed (downstream stages were starved):")
        for err in stage_errors:
            logger.warning(f"  !! {err}")

    logger.info(f"Run complete in {run_duration/3600:.1f}h, metered spend: ${budget_used():.2f}/${budget:.2f} "
                f"across {cost_tracker.call_count} calls")
    return stage_results


async def run_overnight(
    registry: RepoRegistry,
    github_token: str,
    trajectory_dir: Path,
    budget: float = 50.0,
    workers: int = 5,
    limit: int = None,
):
    return await drain_funnel(
        registry, github_token, trajectory_dir,
        budget=budget, workers=workers, limit=limit, loop=False,
    )


async def run_until_limit(
    registry: RepoRegistry,
    github_token: str,
    trajectory_dir: Path,
    threshold_pct: float = 100.0,
    budget: float = 50.0,
    workers: int = 5,
    limit: int = None,
):
    rate_limit_tracker.threshold = threshold_pct / 100.0
    logger.info(
        f"Running until Claude usage reaches {threshold_pct:.0f}% "
        f"(or budget ${budget:.2f} spent, or funnel drained)"
    )
    return await drain_funnel(
        registry, github_token, trajectory_dir,
        budget=budget, workers=workers, limit=limit, loop=True,
    )


async def async_main(args):
    if args.capture_mode:
        schemas.CAPTURE_MODE = args.capture_mode

    data_dir = Path(args.data_dir)
    trajectory_dir = Path(args.traj_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / "registry.db"
    registry = RepoRegistry(str(db_path))

    github_token = os.environ.get("GITHUB_TOKEN", "")

    try:
        if args.status:
            print_status(registry)
            return

        if args.requeue_stale:
            requeue_stale(registry, dry_run=args.dry_run)
            print_status(registry)
            return

        if args.requeue_infra:
            requeue_infra(registry, dry_run=args.dry_run)
            print_status(registry)
            return

        if args.requeue_stage is not None:
            if args.min_score is None:
                logger.error("--requeue-stage requires --min-score")
                sys.exit(1)
            requeue_rescore(registry, args.requeue_stage, args.min_score, dry_run=args.dry_run)
            print_status(registry)
            return

        if args.until_limit is not None:
            pct = args.until_limit
            if not (0 < pct <= 100):
                logger.error(f"--until-limit must be in (0, 100], got {pct}")
                sys.exit(1)
            await run_until_limit(
                registry,
                github_token,
                trajectory_dir,
                threshold_pct=pct,
                budget=args.budget,
                workers=args.workers,
                limit=args.limit,
            )
            return

        if args.overnight:
            await run_overnight(
                registry,
                github_token,
                trajectory_dir,
                budget=args.budget,
                workers=args.workers,
                limit=args.limit,
            )
            return

        if args.stage:
            stages = parse_stage_range(args.stage)
            cost_baseline = cost_tracker.total_usd

            for stage in stages:
                if stage == 1 and not github_token:
                    logger.error("GITHUB_TOKEN environment variable required for stage 1")
                    sys.exit(1)

                logger.info(f"=== Running Stage {stage} ===")
                result = await run_stage(
                    stage,
                    registry,
                    github_token,
                    trajectory_dir,
                    limit=args.limit,
                    workers=args.workers,
                    budget_remaining=args.budget - (cost_tracker.total_usd - cost_baseline),
                    star_floor=args.star_floor,
                )
                logger.info(f"Stage {stage} result: {result} "
                            f"(metered spend so far: ${cost_tracker.total_usd - cost_baseline:.2f})")

            print_status(registry, stages=stages)
            return

        logger.error("Specify --stage, --overnight, --until-limit, or --status")
        sys.exit(1)

    finally:
        registry.close()


def main():
    parser = argparse.ArgumentParser(
        description="QA Repo Trajectory Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m repo_pipeline.orchestrator --stage 1        # Discover repos
  python -m repo_pipeline.orchestrator --stage 2        # Filter repos
  python -m repo_pipeline.orchestrator --stage 3-7      # Run Claude stages (capture + think)
  python -m repo_pipeline.orchestrator --stage 7        # Fill think blocks for captured repos
  python -m repo_pipeline.orchestrator --overnight       # Full overnight run
  python -m repo_pipeline.orchestrator --until-limit 75  # Drain funnel until 75% usage
  python -m repo_pipeline.orchestrator --status          # Print stats
        """,
    )
    parser.add_argument("--stage", type=str, help="Run specific stage(s): N or N-M (e.g., 3 or 3-6)")
    parser.add_argument("--capture-mode", type=str, choices=["agent", "replay"], default=None,
                        help="Stage 4-6 behavior: 'agent' (teacher acts in the booted app; default) or "
                             "'replay' (legacy instrumented test suites). Also settable via PIPELINE_CAPTURE_MODE.")
    parser.add_argument("--overnight", action="store_true", help="Run stages 3-6 on all pending repos, notify via Telegram")
    parser.add_argument("--until-limit", type=float, nargs="?", const=100.0, default=None, metavar="PCT", help="Loop stages 3-6 until Claude usage reaches PCT%% of the subscription limit (bare flag = 100). Honors --budget too.")
    parser.add_argument("--status", action="store_true", help="Print registry stats and exit")
    parser.add_argument("--requeue-stale", action="store_true", help="Re-queue rejections from superseded code (per STAGE_BEHAVIOR_CUTOFFS) for a retry")
    parser.add_argument("--requeue-infra", action="store_true", help="Re-queue infra/flaky rejections (clone/SDK/timeout/max-turns) across stages for a retry")
    parser.add_argument("--requeue-stage", type=int, default=None, help="Re-queue rejections at this stage whose stored gate score >= --min-score (after lowering a threshold)")
    parser.add_argument("--min-score", type=float, default=None, help="Score floor for --requeue-stage (e.g. --requeue-stage 4 --min-score 4)")
    parser.add_argument("--dry-run", action="store_true", help="With --requeue-*: report counts without modifying the registry")
    parser.add_argument("--limit", type=int, default=None, help="Max repos per stage in this run")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers for LLM stages (default: 5)")
    parser.add_argument("--star-floor", type=int, default=5, help="Stage 1: exhaustively scrape repos down to this star count (default: 5)")
    parser.add_argument("--budget", type=float, default=50.0, help="Max total LLM budget for this run in USD (default: 50.0)")
    parser.add_argument("--data-dir", type=str, default="pipeline_data/", help="Pipeline data directory")
    parser.add_argument("--traj-dir", type=str, default="trajectories/", help="Trajectory output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    setup_logging(data_dir, verbose=args.verbose)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
