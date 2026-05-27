import asyncio
import json
import logging
import re

from repo_pipeline.frameworks import resolve_profile
from repo_pipeline.instrumentation import REPO_DIR, RESULTS_DIR, setup_repo_instrumentation
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox
from repo_pipeline.schemas import RepoStatus, TrialResults

logger = logging.getLogger(__name__)

# How many test files to try (a POC trial — stop early once we capture real coordinates).
MAX_TRIAL_FILES = 3
TEST_RUN_TIMEOUT = 240

_PASS_RE = re.compile(r"(\d+)\s+passed")
_FAIL_RE = re.compile(r"(\d+)\s+failed")


async def _run(fn, *args):
    """Run a blocking DockerSandbox call off the event loop so concurrent workers overlap."""
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def _collect_metrics(sandbox) -> TrialResults:
    """Compute trial metrics from the trajectories captured under /workspace/results."""
    traj_files = sandbox.list_files("trajectory.json", RESULTS_DIR)
    screenshots = sandbox.list_files("*.png", RESULTS_DIR)
    valid_bboxes = 0
    total_actions = 0
    action_types: set[str] = set()
    for tf in traj_files:
        try:
            data = json.loads(sandbox.read_file(tf))
        except Exception:
            continue
        for step in data.get("steps", []):
            for action in step.get("actions", []):
                total_actions += 1
                action_types.add(action.get("type", ""))
                if action.get("x", 0) > 0 or action.get("y", 0) > 0:
                    valid_bboxes += 1
    return TrialResults(
        action_diversity_score=min(len(action_types) / 5.0, 1.0),
        domain_relevance_score=0.8,  # repo already cleared the stage-3 app-vs-library gate
        trajectory_count_estimate=len(traj_files),
        setup_complexity_score=0.2,  # install + instrumentation succeeded in code
        valid_bounding_boxes=valid_bboxes,
        total_actions_captured=total_actions,
        screenshots_captured=len(screenshots),
    )


async def trial_single_repo(registry: RepoRegistry, repo: dict):
    repo_name = repo["full_name"]

    if repo.get("current_stage", 0) >= 5:
        return

    logger.info(f"Trial run: {repo_name}")

    deep_analysis = {}
    if repo.get("deep_analysis"):
        try:
            deep_analysis = json.loads(repo["deep_analysis"]) if isinstance(repo["deep_analysis"], str) else repo["deep_analysis"]
        except (json.JSONDecodeError, TypeError):
            pass

    profile = resolve_profile(deep_analysis.get("test_framework"), deep_analysis.get("framework_language"))
    test_name_prefix = repo_name.replace("/", "_")

    sandbox = DockerSandbox(image=profile.sandbox_image, platform=profile.sandbox_platform)
    try:
        await _run(sandbox.start, repo_name)

        _, stderr, exit_code = await _run(sandbox.clone_repo, repo_name, repo.get("default_branch"))
        if exit_code != 0:
            registry.reject(repo_name, stage=5, reason=f"clone failed: {stderr[:200]}")
            return

        await _run(setup_repo_instrumentation, sandbox, profile)

        # Locate test files via the profile's globs (skip vendored deps).
        test_files: list[str] = []
        for glob in profile.test_globs:
            for path in await _run(sandbox.list_files, glob, REPO_DIR):
                if "/node_modules/" not in path and path not in test_files:
                    test_files.append(path)
        if not test_files:
            registry.reject(repo_name, stage=5, reason="no test files found")
            return

        # Run up to MAX_TRIAL_FILES instrumented test files in code; stop once we capture coords.
        passed = failed = 0
        for test_file in test_files[:MAX_TRIAL_FILES]:
            rel = test_file.replace(REPO_DIR + "/", "")
            cmd = profile.test_command_single.format(
                repo_name=repo_name, test_name_prefix=test_name_prefix
            ).replace("<test_file>", rel)
            out, err, _ = await _run(sandbox.exec_command, cmd, TEST_RUN_TIMEOUT)
            combined = out + err
            if (m := _PASS_RE.search(combined)):
                passed += int(m.group(1))
            if (m := _FAIL_RE.search(combined)):
                failed += int(m.group(1))
            interim = await _run(_collect_metrics, sandbox)
            if interim.valid_bounding_boxes > 0:
                break

        metrics = await _run(_collect_metrics, sandbox)
        total = passed + failed
        # pass_rate informs ranking only; the gate is capture-based (a test that performs actions
        # then fails an assertion still produced a valid trajectory).
        metrics.pass_rate = (passed / total) if total else (1.0 if metrics.valid_bounding_boxes else 0.0)
        metrics.notes = f"code-driven trial: {len(test_files)} test files found, ran up to {MAX_TRIAL_FILES}"

        if metrics.valid_bounding_boxes < 1 or metrics.screenshots_captured < 1:
            registry.reject(
                repo_name,
                stage=5,
                reason=(
                    f"no usable capture (bboxes={metrics.valid_bounding_boxes}, "
                    f"screenshots={metrics.screenshots_captured}, traj={metrics.trajectory_count_estimate})"
                ),
            )
            return

        weighted = metrics.weighted_score
        logger.info(
            f"  {repo_name}: weighted={weighted:.2f} bboxes={metrics.valid_bounding_boxes} "
            f"actions={metrics.total_actions_captured} traj={metrics.trajectory_count_estimate} pass_rate={metrics.pass_rate:.0%}"
        )
        registry.update_status(
            repo_name,
            RepoStatus.VALIDATED.value,
            stage=5,
            trial_results=metrics.model_dump(),
            trial_score=weighted,
        )

    except Exception as e:
        logger.error(f"Error in trial run for {repo_name}: {e}")
        registry.reject(repo_name, stage=5, reason=f"trial error: {str(e)[:200]}")
    finally:
        await _run(sandbox.stop)


async def run_trial(
    registry: RepoRegistry,
    limit: int = None,
    workers: int = 2,
    budget_remaining: float = None,
):
    repos = registry.get_repos_for_stage(5, limit=limit)
    logger.info(f"Trial running {len(repos)} repos with {workers} workers")

    semaphore = asyncio.Semaphore(workers)

    async def process_with_semaphore(repo):
        async with semaphore:
            await trial_single_repo(registry, repo)

    await asyncio.gather(*[process_with_semaphore(repo) for repo in repos])

    stats = registry.get_stats()
    passed = stats.get(RepoStatus.VALIDATED.value, 0)
    rejected = stats.get(RepoStatus.REJECTED_TRIAL.value, 0)
    logger.info(f"Trial runs complete: {passed} validated, {rejected} rejected")
    return {"processed": len(repos), "passed": passed, "rejected": rejected}
