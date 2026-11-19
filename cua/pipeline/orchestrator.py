import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from cua.pipeline import schemas
from cua.pipeline.notifier import send_report
from cua.pipeline.registry import RepoRegistry
from cua.pipeline.sandbox import install_sigterm_handler, safe_parallel_sandboxes
from cua.pipeline.schemas import (
    NON_QUALITY_CATEGORIES,
    STAGE_BEHAVIOR_CUTOFFS,
    STAGE_TO_REJECTED_STATUS,
    DeepAnalysis,
    FailureCategory,
    RepoStatus,
)
from cua.pipeline.sdk_runner import UsageLimitReached, cost_tracker, rate_limit_tracker
from cua.pipeline.stage1_discover import run_discovery
from cua.pipeline.stage2_filter import run_filter
from cua.pipeline.stage3_shallow_score import run_shallow_score
from cua.pipeline.stage4_deep_analyze import fit_veto, judge_fit, run_deep_analyze
from cua.pipeline.stage5_trial_run import run_trial
from cua.pipeline.stage6_generate import run_generate
from cua.pipeline.stage7_think import run_think
from cua.settings import GITHUB, PIPELINE

logger = logging.getLogger(__name__)


def requeue_stale(registry: RepoRegistry, dry_run: bool = False) -> int:
    """Re-queue rejections produced by superseded code, per stage with a known behavior cutoff."""
    verb = "would re-queue" if dry_run else "re-queued"
    total = 0
    for stage, cutoff in sorted(STAGE_BEHAVIOR_CUTOFFS.items()):
        n = registry.reset_rejected(stage=stage, before=cutoff, dry_run=dry_run)
        logger.info(f"  stage {stage}: {verb} {n} stale rejections (before {cutoff})")
        total += n
    logger.info(f"Stale re-queue: {verb} {total} repos total{' (dry-run)' if dry_run else ''}")
    return total


def requeue_infra(registry: RepoRegistry, dry_run: bool = False) -> int:
    """Re-queue infra and flaky rejections across all stages, leaving genuine quality rejections in place."""
    verb = "would re-queue" if dry_run else "re-queued"
    n = registry.reset_rejected(category_in=NON_QUALITY_CATEGORIES, dry_run=dry_run)
    logger.info(f"Infra re-queue: {verb} {n} infra/flaky rejections{' (dry-run)' if dry_run else ''}")
    return n


def requeue_rescore(registry: RepoRegistry, stage: int, min_score: float, dry_run: bool = False) -> int:
    """Re-queue the stage-N rejects whose stored gate score a lowered `min_score` would now accept."""
    verb = "would re-queue" if dry_run else "re-queued"
    n = registry.reset_rejected(stage=stage, min_score=min_score, dry_run=dry_run)
    logger.info(
        f"Rescore re-queue: {verb} {n} stage-{stage} rejections with score >= {min_score}"
        f"{' (dry-run)' if dry_run else ''}. Re-run --stage {stage} to re-admit them."
    )
    return n


async def prune_unfit(registry: RepoRegistry, dry_run: bool = False) -> int:
    """Retroactively reject repos past stage 4 that today's fit veto refuses, re-judged from their STORED analysis."""
    candidates = registry.get_repos_by_status([RepoStatus.ANALYZED.value, RepoStatus.VALIDATED.value])
    verb = "would prune" if dry_run else "pruned"
    logger.info(f"Prune unfit: re-judging {len(candidates)} repos past stage 4")
    pruned = 0
    for repo in candidates:
        raw = repo.get("deep_analysis")
        if not raw:
            logger.info(f"  SKIP  {repo['full_name']} ({repo['status']}): no stored analysis")
            continue
        analysis = DeepAnalysis(**(json.loads(raw) if isinstance(raw, str) else raw))
        shallow_raw = repo.get("shallow_scores")
        shallow = (json.loads(shallow_raw) if isinstance(shallow_raw, str) else shallow_raw) or {}

        # Deterministic backstops first (free); only spend a verdict call if they keep the repo.
        reason = fit_veto(analysis, shallow)
        if reason is None:
            try:
                analysis.good_fit, analysis.veto_reason = await judge_fit(registry, repo, analysis, shallow)
            except UsageLimitReached:
                logger.warning("  usage limit reached — stopping prune; re-run to finish the rest")
                break
            reason = fit_veto(analysis, shallow)

        if reason:
            logger.info(f"  PRUNE {repo['full_name']} ({repo['status']}): {reason}")
            if not dry_run:
                registry.reject(
                    repo["full_name"],
                    stage=4,
                    reason=f"fit veto: {reason}",
                    category=FailureCategory.QUALITY_GATE,
                    deep_analysis=analysis.model_dump(),
                )
            pruned += 1
        else:
            logger.info(f"  KEEP  {repo['full_name']} ({repo['status']})")

    logger.info(f"Prune unfit: {verb} {pruned}/{len(candidates)} repos{' (dry-run)' if dry_run else ''}")
    return pruned


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

    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_file


# The 7-stage funnel — always shown (even at 0) so the full pipeline is visible at a glance.
PROGRESS_STATUSES = [
    "discovered",
    "filtered",
    "scored",
    "analyzed",
    "validated",
    "captured",
    "completed",
]
# Rejection buckets — only shown when non-empty.
REJECTION_STATUSES = [
    "rejected_filter",
    "rejected_score",
    "rejected_analyze",
    "rejected_trial",
    "rejected_generate",
]


def print_status_counts(registry: RepoRegistry) -> None:
    """The funnel's status counts plus the trajectory totals behind `completed` and the not-yet-finalized `captured`."""
    stats = registry.get_stats()
    total_traj = registry.count_total_trajectories()
    captured_traj = registry.count_total_trajectories(RepoStatus.CAPTURED.value)

    print("Status counts:")
    for status in PROGRESS_STATUSES:
        print(f"  {status:25s} {stats.get(status, 0):>6,}")
    for status in REJECTION_STATUSES:
        count = stats.get(status, 0)
        if count > 0:
            print(f"  {status:25s} {count:>6,}")

    print(f"  {'TOTAL':25s} {sum(stats.values()):>6,}")
    print(f"\n  Total trajectories generated (completed): {total_traj:,}")
    if captured_traj:
        print(f"  Captured but pending stage-7 finalize:    {captured_traj:,}")


def print_status_distributions(registry: RepoRegistry) -> None:
    """What is queued for the next drain, and how the live backlog and the rejections are spread."""
    pending = registry.get_pending_counts()
    print("\nPending for next run:")
    for stage in DRAIN_STAGES:
        print(f"  Stage {stage}: {pending.get(stage, 0):,} repos")

    domain_dist = registry.get_domain_distribution()
    if domain_dist:
        print("\nDomain distribution (non-rejected repos):")
        for domain, count in domain_dist.items():
            print(f"  {domain:20s} {count:>6,}")

    category_dist = registry.get_failure_category_distribution()
    if category_dist:
        print("\nRejection categories (quality vs infra):")
        for cat, count in category_dist:
            print(f"  {cat:20s} {count:>6,}")


def print_rejection_reasons(registry: RepoRegistry, stages: list[int] = None) -> None:
    """Top rejection reasons across all stages, plus one section per requested stage."""
    rejection_reasons = registry.get_rejection_reasons(limit=10)
    if rejection_reasons:
        print("\nTop rejection reasons (all stages, all time):")
        for i, (reason, count) in enumerate(rejection_reasons, 1):
            print(f'  {i}. "{reason}" ({count:,})')

    for stage in stages or []:
        rejected_status = STAGE_TO_REJECTED_STATUS.get(stage)
        if rejected_status is None:
            continue
        per_stage = registry.get_rejection_reasons(limit=5, status=rejected_status.value)
        if not per_stage:
            continue
        print(f"\nTop rejection reasons (stage {stage} — {rejected_status.value}):")
        for i, (reason, count) in enumerate(per_stage, 1):
            print(f'  {i}. "{reason}" ({count:,})')


def print_status(registry: RepoRegistry, stages: list[int] = None):
    """Print the pipeline registry status, with a per-stage rejection section for each requested stage."""
    print("\n=== Pipeline Registry Status ===\n")
    print_status_counts(registry)
    print_status_distributions(registry)
    print_rejection_reasons(registry, stages)
    print()


def parse_stage_range(stage_str: str) -> list[int]:
    if "-" in stage_str:
        start, end = stage_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(stage_str)]


@dataclass(frozen=True)
class StageRequest:
    """Everything a stage entry point can need; ``workers`` is already capped and clamped for that stage."""

    registry: RepoRegistry
    github_token: str
    trajectory_dir: Path
    limit: int | None
    workers: int
    budget_remaining: float | None
    star_floor: int


@dataclass(frozen=True)
class StageSpec:
    """One funnel stage: its entry point plus the worker, drain and budget-pacing policy applied to it."""

    number: int
    name: str
    run: Callable[[StageRequest], Awaitable[dict]]
    workers_cap: int | None = None
    sandboxed: bool = False
    drained: bool = False
    cost_estimate: float | None = None


async def _discover(request: StageRequest) -> dict:
    return run_discovery(request.registry, request.github_token, star_floor=request.star_floor)


async def _filter(request: StageRequest) -> dict:
    return run_filter(request.registry, limit=request.limit)


async def _shallow_score(request: StageRequest) -> dict:
    return await run_shallow_score(
        request.registry, limit=request.limit, workers=request.workers, budget_remaining=request.budget_remaining
    )


async def _deep_analyze(request: StageRequest) -> dict:
    return await run_deep_analyze(
        request.registry, limit=request.limit, workers=request.workers, budget_remaining=request.budget_remaining
    )


async def _trial_run(request: StageRequest) -> dict:
    return await run_trial(
        request.registry, limit=request.limit, workers=request.workers, budget_remaining=request.budget_remaining
    )


async def _capture(request: StageRequest) -> dict:
    return await run_generate(
        request.registry,
        request.trajectory_dir,
        limit=request.limit,
        workers=request.workers,
        budget_remaining=request.budget_remaining,
    )


async def _finalize(request: StageRequest) -> dict:
    return await run_think(
        request.registry,
        request.trajectory_dir,
        limit=request.limit,
        workers=request.workers,
        budget_remaining=request.budget_remaining,
    )


DEFAULT_STAGE_COST = 0.10
# Stage 6 costs whatever the capture mode costs, so its estimate is resolved per call instead of stored here.
STAGE6_COST_BY_MODE = {"agent": 6.00, "replay": 0.20}

# Adding a stage is an entry here plus its STAGE_TO_* statuses in schemas.py — never a new branch in run_stage.
STAGES = {
    spec.number: spec
    for spec in (
        StageSpec(1, "discover", _discover, cost_estimate=DEFAULT_STAGE_COST),
        StageSpec(2, "filter", _filter, cost_estimate=DEFAULT_STAGE_COST),
        # Sonnet shallow scoring.
        StageSpec(3, "shallow_score", _shallow_score, drained=True, cost_estimate=0.15),
        # Static analysis only (clone + read files, no install or app boot), so its containers are light.
        StageSpec(
            4,
            "deep_analyze",
            _deep_analyze,
            workers_cap=PIPELINE.stage4_workers,
            sandboxed=True,
            drained=True,
            cost_estimate=0.20,
        ),
        # Boot gate = clone + install + boot + driver probe (~1-2GB/worker), and code-driven, so ~$0 LLM.
        StageSpec(
            5,
            "trial_run",
            _trial_run,
            workers_cap=PIPELINE.stage5_workers,
            sandboxed=True,
            drained=True,
            cost_estimate=0.02,
        ),
        # Each worker holds a sandbox with a booted app plus headless Chromium (~1.2GB real).
        StageSpec(6, "capture", _capture, workers_cap=PIPELINE.stage6_workers, sandboxed=True, drained=True),
        # Polish (agent, text-only) / per-step think-with-image (replay).
        StageSpec(7, "finalize", _finalize, workers_cap=3, drained=True, cost_estimate=0.50),
    )
}

DRAIN_STAGES = tuple(number for number, spec in sorted(STAGES.items()) if spec.drained)
# Kept as a name because README.md and docs/architecture.md cite it; the specs are the source of the figures.
STAGE_COST_ESTIMATES = {
    number: spec.cost_estimate
    for number, spec in STAGES.items()
    if spec.cost_estimate not in (None, DEFAULT_STAGE_COST)
}


def stage_workers(spec: StageSpec, workers: int) -> int:
    """``--workers`` capped by the stage's own ceiling, then clamped against Docker VM memory where that applies."""
    capped = workers if spec.workers_cap is None else min(workers, spec.workers_cap)
    return safe_parallel_sandboxes(capped) if spec.sandboxed else capped


def stage_cost(stage: int) -> float:
    """Per-repo LLM estimate used only to pre-size a --budget batch, never to charge it."""
    spec = STAGES.get(stage)
    if spec is None:
        return DEFAULT_STAGE_COST
    if spec.cost_estimate is None:
        return STAGE6_COST_BY_MODE.get(schemas.CAPTURE_MODE, STAGE6_COST_BY_MODE["agent"])
    return spec.cost_estimate


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
    spec = STAGES.get(stage)
    if spec is None:
        raise ValueError(f"Unknown stage: {stage}")
    return await spec.run(
        StageRequest(
            registry=registry,
            github_token=github_token,
            trajectory_dir=trajectory_dir,
            limit=limit,
            workers=stage_workers(spec, workers),
            budget_remaining=budget_remaining,
            star_floor=star_floor,
        )
    )


def budget_used(baseline: float) -> float:
    """Metered spend since the drain started; every ResultMessage's total_cost_usd lands in cost_tracker."""
    return cost_tracker.total_usd - baseline


def should_continue(baseline: float, budget: float) -> bool:
    """False once --budget is covered by metered spend or the usage limit is reached, so repos keep their status."""
    used = budget_used(baseline)
    if used >= budget:
        logger.warning(f"Budget exhausted (${used:.2f}/${budget:.2f} metered), stopping")
        return False
    if rate_limit_tracker.limit_reached():
        util = rate_limit_tracker.max_utilization()
        util_str = f"{util * 100:.0f}%" if util is not None else "hard 429"
        logger.warning(f"Claude usage limit reached ({util_str}), stopping")
        return False
    return True


def affordable_limit(stage: int, pending_count: int, remaining_budget: float, limit: int = None) -> int:
    """How many repos this stage may take: what is pending, what the budget pre-sizes, and any explicit --limit."""
    cost_per_repo = stage_cost(stage)
    max_affordable = int(remaining_budget / cost_per_repo) if cost_per_repo > 0 else pending_count
    effective_limit = min(pending_count, max_affordable)
    return min(effective_limit, limit) if limit else effective_limit


def record_stage_error(stage: int, error: Exception, stage_results: dict, stage_errors: list[str]) -> None:
    """Record a crashed stage loudly: it starves everything downstream while reading as "0 repos pending"."""
    logger.error(f"  Stage {stage} failed: {error}")
    stage_results.setdefault(stage, {"processed": 0, "passed": 0, "rejected": 0})
    stage_results[stage]["error"] = str(error)
    err = f"stage {stage}: {type(error).__name__}: {str(error)[:200]}"
    if err not in stage_errors:
        stage_errors.append(err)


async def run_pass(
    registry: RepoRegistry,
    github_token: str,
    trajectory_dir: Path,
    *,
    baseline: float,
    budget: float,
    workers: int,
    limit: int,
    stage_results: dict,
    stage_errors: list[str],
) -> int:
    """One forward pass over the drain stages, returning how many repos it processed."""
    pass_processed = 0
    for stage in DRAIN_STAGES:
        if not should_continue(baseline, budget):
            break

        remaining_budget = budget - budget_used(baseline)
        logger.info(f"--- Stage {stage} --- (budget remaining: ${remaining_budget:.2f} of ${budget:.2f}, metered)")

        pending_count = registry.get_pending_counts().get(stage, 0)
        if pending_count == 0:
            logger.info(f"  No repos pending for stage {stage}, skipping")
            continue

        effective_limit = affordable_limit(stage, pending_count, remaining_budget, limit)
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
            # Accumulated across passes; the registry-derived passed/rejected/trajectories are cumulative already.
            prev_processed = stage_results.get(stage, {}).get("processed", 0)
            stage_results[stage] = {**result, "processed": prev_processed + processed}
        except Exception as e:
            record_stage_error(stage, e, stage_results, stage_errors)
    return pass_processed


async def close_out_drain(
    registry: RepoRegistry,
    run_duration: float,
    spent: float,
    budget: float,
    stage_results: dict,
    stage_errors: list[str],
    loop: bool,
) -> None:
    """Close out a drain: the threshold caveat, the Telegram report, the crashed-stage roll-up and the spend line."""
    if loop and rate_limit_tracker.threshold < 1.0 and rate_limit_tracker.observed_event_count == 0:
        logger.warning(
            f"--until-limit {rate_limit_tracker.threshold * 100:.0f}%% was requested but zero "
            f"RateLimitEvents were observed; the run effectively used 'until hard 429 / funnel "
            f"drained' instead. Headless OAuth may not emit rate_limit_event messages."
        )

    try:
        await send_report(
            registry,
            run_duration,
            spent,
            budget,
            stage_results,
            rate_limit_info=rate_limit_tracker.latest(),
        )
    except Exception as e:
        logger.error(f"Failed to send Telegram report: {e}")

    if stage_errors:
        logger.warning("RUN DEGRADED — one or more stages crashed (downstream stages were starved):")
        for err in stage_errors:
            logger.warning(f"  !! {err}")

    logger.info(
        f"Run complete in {run_duration / 3600:.1f}h, metered spend: ${spent:.2f}/${budget:.2f} "
        f"across {cost_tracker.call_count} calls"
    )


async def drain_funnel(
    registry: RepoRegistry,
    github_token: str,
    trajectory_dir: Path,
    budget: float = 50.0,
    workers: int = 5,
    limit: int = None,
    loop: bool = False,
):
    """Run the drain stages over pending repos, stopping on --budget or the usage limit; ``loop`` repeats to drain."""
    start_time = time.time()
    # Estimates only pre-size batches: the budget is paced by ACTUAL metered spend, which guesses drifted from.
    baseline = cost_tracker.total_usd
    stage_results = {}
    stage_errors: list[str] = []

    while True:
        pass_processed = await run_pass(
            registry,
            github_token,
            trajectory_dir,
            baseline=baseline,
            budget=budget,
            workers=workers,
            limit=limit,
            stage_results=stage_results,
            stage_errors=stage_errors,
        )
        if not loop or not should_continue(baseline, budget):
            break
        if pass_processed == 0:
            logger.info("Funnel drained (a full pass processed 0 repos), stopping")
            break

    await close_out_drain(
        registry,
        time.time() - start_time,
        budget_used(baseline),
        budget,
        stage_results,
        stage_errors,
        loop,
    )
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
        registry,
        github_token,
        trajectory_dir,
        budget=budget,
        workers=workers,
        limit=limit,
        loop=False,
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
        f"Running until Claude usage reaches {threshold_pct:.0f}% (or budget ${budget:.2f} spent, or funnel drained)"
    )
    return await drain_funnel(
        registry,
        github_token,
        trajectory_dir,
        budget=budget,
        workers=workers,
        limit=limit,
        loop=True,
    )


async def run_maintenance(args, registry: RepoRegistry) -> bool:
    """The registry-only commands: True once one of them has handled the invocation."""
    if args.status:
        print_status(registry)
        return True

    if args.requeue_stale:
        requeue_stale(registry, dry_run=args.dry_run)
    elif args.requeue_infra:
        requeue_infra(registry, dry_run=args.dry_run)
    elif args.requeue_stage is not None:
        if args.min_score is None:
            logger.error("--requeue-stage requires --min-score")
            sys.exit(1)
        requeue_rescore(registry, args.requeue_stage, args.min_score, dry_run=args.dry_run)
    elif args.prune_unfit:
        await prune_unfit(registry, dry_run=args.dry_run)
    else:
        return False

    print_status(registry)
    return True


async def run_stage_range(args, registry: RepoRegistry, github_token: str, trajectory_dir: Path) -> None:
    """Run each stage of --stage N or N-M in order, pacing --budget by the spend metered since the first one."""
    stages = parse_stage_range(args.stage)
    baseline = cost_tracker.total_usd

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
            budget_remaining=args.budget - budget_used(baseline),
            star_floor=args.star_floor,
        )
        logger.info(f"Stage {stage} result: {result} (metered spend so far: ${budget_used(baseline):.2f})")

    print_status(registry, stages=stages)


async def run_requested_stages(args, registry: RepoRegistry, github_token: str, trajectory_dir: Path) -> None:
    """The two drain modes and the explicit --stage run; an invocation naming none of them is an error."""
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
        await run_stage_range(args, registry, github_token, trajectory_dir)
        return

    logger.error("Specify --stage, --overnight, --until-limit, or --status")
    sys.exit(1)


async def async_main(args):
    if args.capture_mode:
        schemas.CAPTURE_MODE = args.capture_mode

    data_dir = Path(args.data_dir)
    trajectory_dir = Path(args.traj_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    registry = RepoRegistry(str(data_dir / "registry.db"))
    github_token = GITHUB.token

    try:
        if await run_maintenance(args, registry):
            return
        await run_requested_stages(args, registry, github_token, trajectory_dir)
    finally:
        registry.close()


def main():
    parser = argparse.ArgumentParser(
        description="QA Repo Trajectory Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cua.pipeline.orchestrator --stage 1        # Discover repos
  python -m cua.pipeline.orchestrator --stage 2        # Filter repos
  python -m cua.pipeline.orchestrator --stage 3-7      # Run Claude stages (capture + think)
  python -m cua.pipeline.orchestrator --stage 7        # Fill think blocks for captured repos
  python -m cua.pipeline.orchestrator --overnight       # Full overnight run
  python -m cua.pipeline.orchestrator --until-limit 75  # Drain funnel until 75% usage
  python -m cua.pipeline.orchestrator --status          # Print stats
        """,
    )
    parser.add_argument("--stage", type=str, help="Run specific stage(s): N or N-M (e.g., 3 or 3-6)")
    parser.add_argument(
        "--capture-mode",
        type=str,
        choices=["agent", "replay"],
        default=None,
        help="Stage 4-6 behavior: 'agent' (teacher acts in the booted app; default) or "
        "'replay' (legacy instrumented test suites). Also settable via PIPELINE_CAPTURE_MODE.",
    )
    parser.add_argument(
        "--overnight", action="store_true", help="Run stages 3-6 on all pending repos, notify via Telegram"
    )
    parser.add_argument(
        "--until-limit",
        type=float,
        nargs="?",
        const=100.0,
        default=None,
        metavar="PCT",
        help="Loop stages 3-6 until Claude usage reaches PCT%% of the subscription limit (bare flag = 100). "
        "Honors --budget too.",
    )
    parser.add_argument("--status", action="store_true", help="Print registry stats and exit")
    parser.add_argument(
        "--requeue-stale",
        action="store_true",
        help="Re-queue rejections from superseded code (per STAGE_BEHAVIOR_CUTOFFS) for a retry",
    )
    parser.add_argument(
        "--requeue-infra",
        action="store_true",
        help="Re-queue infra/flaky rejections (clone/SDK/timeout/max-turns) across stages for a retry",
    )
    parser.add_argument(
        "--requeue-stage",
        type=int,
        default=None,
        help="Re-queue rejections at this stage whose stored gate score >= --min-score (after lowering a threshold)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Score floor for --requeue-stage (e.g. --requeue-stage 4 --min-score 4)",
    )
    parser.add_argument(
        "--prune-unfit",
        action="store_true",
        help="Re-judge repos already past stage 4 and reject obvious bad-fits (toys, boilerplates, login-walled, "
        "test-suite-only) so no more stage-5/6 compute is wasted. Reversible via --requeue-stage 4 --min-score 0. "
        "Honors --dry-run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="With --requeue-*: report counts without modifying the registry"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max repos per stage in this run")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers for LLM stages (default: 5)")
    parser.add_argument(
        "--star-floor",
        type=int,
        default=5,
        help="Stage 1: exhaustively scrape repos down to this star count (default: 5)",
    )
    parser.add_argument(
        "--budget", type=float, default=50.0, help="Max total LLM budget for this run in USD (default: 50.0)"
    )
    parser.add_argument("--data-dir", type=str, default="pipeline_data/", help="Pipeline data directory")
    parser.add_argument("--traj-dir", type=str, default="trajectories/", help="Trajectory output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    setup_logging(Path(args.data_dir), verbose=args.verbose)
    install_sigterm_handler()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
