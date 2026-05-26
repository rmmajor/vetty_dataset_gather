import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from repo_pipeline.registry import RepoRegistry

logger = logging.getLogger(__name__)


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


def print_status(registry: RepoRegistry):
    stats = registry.get_stats()
    pending = registry.get_pending_counts()
    domain_dist = registry.get_domain_distribution()
    rejection_reasons = registry.get_rejection_reasons(limit=10)
    total_traj = registry.count_total_trajectories()

    print("\n=== Pipeline Registry Status ===\n")

    print("Status counts:")
    status_order = [
        "discovered", "filtered", "scored", "analyzed", "validated", "completed",
        "rejected_filter", "rejected_score", "rejected_analyze", "rejected_trial", "rejected_generate",
    ]
    for status in status_order:
        count = stats.get(status, 0)
        if count > 0:
            print(f"  {status:25s} {count:>6,}")

    total = sum(stats.values())
    print(f"  {'TOTAL':25s} {total:>6,}")
    print(f"\n  Total trajectories generated: {total_traj:,}")

    print("\nPending for next run:")
    for stage in [3, 4, 5, 6]:
        count = pending.get(stage, 0)
        print(f"  Stage {stage}: {count:,} repos")

    if domain_dist:
        print("\nDomain distribution (non-rejected repos):")
        for domain, count in domain_dist.items():
            print(f"  {domain:20s} {count:>6,}")

    if rejection_reasons:
        print("\nTop rejection reasons:")
        for i, (reason, count) in enumerate(rejection_reasons, 1):
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
        return await run_generate(registry, trajectory_dir, limit=limit, workers=min(workers, 2), budget_remaining=budget_remaining)

    else:
        raise ValueError(f"Unknown stage: {stage}")


STAGE_COST_ESTIMATES = {
    3: 0.15,
    4: 0.75,
    5: 1.00,
    6: 1.50,
}


async def run_overnight(
    registry: RepoRegistry,
    github_token: str,
    trajectory_dir: Path,
    budget: float = 50.0,
    workers: int = 5,
    limit: int = None,
):
    start_time = time.time()
    budget_used = 0.0
    stage_results = {}

    for stage in [3, 4, 5, 6]:
        if budget_used >= budget:
            logger.warning(f"Budget exhausted (${budget_used:.2f}/${budget:.2f}), stopping at stage {stage}")
            break

        remaining_budget = budget - budget_used
        logger.info(f"--- Stage {stage} --- (budget remaining: ${remaining_budget:.2f})")

        pending = registry.get_pending_counts()
        pending_count = pending.get(stage, 0)
        if pending_count == 0:
            logger.info(f"  No repos pending for stage {stage}, skipping")
            continue

        cost_per_repo = STAGE_COST_ESTIMATES.get(stage, 0.10)
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
            stage_results[stage] = result

            processed = result.get("processed", 0)
            budget_used += processed * cost_per_repo

        except Exception as e:
            logger.error(f"  Stage {stage} failed: {e}")
            stage_results[stage] = {"processed": 0, "passed": 0, "rejected": 0, "error": str(e)}

    run_duration = time.time() - start_time

    from repo_pipeline.notifier import send_report
    try:
        await send_report(registry, run_duration, budget_used, budget, stage_results)
    except Exception as e:
        logger.error(f"Failed to send Telegram report: {e}")

    logger.info(f"Overnight run complete in {run_duration/3600:.1f}h, budget used: ${budget_used:.2f}/${budget:.2f}")
    return stage_results


async def async_main(args):
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
            budget_remaining = args.budget

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
                    budget_remaining=budget_remaining,
                    star_floor=args.star_floor,
                )
                logger.info(f"Stage {stage} result: {result}")

                processed = result.get("processed", 0)
                cost_per = STAGE_COST_ESTIMATES.get(stage, 0)
                budget_remaining -= processed * cost_per

            print_status(registry)
            return

        logger.error("Specify --stage, --overnight, or --status")
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
  python -m repo_pipeline.orchestrator --stage 3-6      # Run Claude stages
  python -m repo_pipeline.orchestrator --overnight       # Full overnight run
  python -m repo_pipeline.orchestrator --status          # Print stats
        """,
    )
    parser.add_argument("--stage", type=str, help="Run specific stage(s): N or N-M (e.g., 3 or 3-6)")
    parser.add_argument("--overnight", action="store_true", help="Run stages 3-6 on all pending repos, notify via Telegram")
    parser.add_argument("--status", action="store_true", help="Print registry stats and exit")
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
