import asyncio
import json
import logging
import os
import re

from cua.pipeline import app_runner, driver_session, schemas
from cua.pipeline._processing import run_repo
from cua.pipeline.frameworks import resolve_profile
from cua.pipeline.instrumentation import REPO_DIR, RESULTS_DIR, setup_repo_instrumentation
from cua.pipeline.logging_utils import StageProgress, repo_logger
from cua.pipeline.registry import RepoRegistry
from cua.pipeline.sandbox import DockerSandbox
from cua.pipeline.schemas import (
    FailureCategory,
    QualityReject,
    RepoStatus,
    RetryableFailure,
    TrialResults,
)
from cua.pipeline.sdk_runner import rate_limit_tracker

logger = logging.getLogger(__name__)

# How many test files to try (a POC trial — stop early once we capture real coordinates).
MAX_TRIAL_FILES = 3
TEST_RUN_TIMEOUT = 240

# Fewer interactive elements than this on the booted homepage means there is nothing for an
# agent to meaningfully do (blank page, build splash, bare error page).
MIN_INTERACTIVE_ELEMENTS = int(os.environ.get("BOOT_GATE_MIN_ELEMENTS", "5"))

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


async def boot_gate_single_repo(registry: RepoRegistry, repo: dict):
    """Agent-mode stage 5: gate on what agent capture actually needs — the repo's app boots in
    the sandbox and renders a non-trivial UI through the browser driver. No instrumentation is
    injected and no tests run, so the entire legacy `no_capture` failure class (instrumentation
    hooks not firing) cannot occur here. Repos whose tests only target public URLs are rejected:
    agent capture drives locally booted apps (real-site trajectories are Path A's job)."""
    repo_name = repo["full_name"]

    if repo.get("current_stage", 0) >= 5:
        return

    rlog = repo_logger(5, repo_name)
    rlog.info("boot gate")

    deep_analysis = {}
    if repo.get("deep_analysis"):
        try:
            deep_analysis = (
                json.loads(repo["deep_analysis"]) if isinstance(repo["deep_analysis"], str) else repo["deep_analysis"]
            )
        except (json.JSONDecodeError, TypeError):
            pass

    profile = resolve_profile(deep_analysis.get("test_framework"), deep_analysis.get("framework_language"))

    # Always the default Playwright image: agent capture never runs the repo's tests, and the
    # browser driver needs the /ms-playwright browsers that only this image ships.
    sandbox = DockerSandbox()
    try:
        await _run(sandbox.start, repo_name)

        _, stderr, exit_code = await _run(sandbox.clone_repo, repo_name, repo.get("default_branch"))
        if exit_code != 0:
            raise RetryableFailure(FailureCategory.CLONE_FAILED, f"clone failed: {stderr[:200]}")

        # start_app owns the dependency install now: it picks the repo's own package manager
        # (npm/pnpm/yarn) in the app's actual dir, which a blind root `npm install` got wrong.
        app_url = await _run(driver_session.start_app, sandbox, profile)
        if app_url is None:
            app_log, _, _ = await _run(sandbox.exec_command, f"tail -20 {driver_session.APP_LOG} 2>/dev/null")
            detail = app_log.strip() or "no recognizable start script / no bootable Node app"
            raise QualityReject(
                FailureCategory.NO_BOOT,
                f"app did not boot (public_url_tests={bool(deep_analysis.get('tests_public_url'))}); {detail[-300:]}",
            )

        await _run(driver_session.install_driver, sandbox)
        await _run(driver_session.start_driver, sandbox)
        client = driver_session.DriverClient(sandbox)
        probe = await _run(client.session, "boot_probe", app_url)
        n_elements = driver_session.count_interactive_elements(probe.get("snapshot", ""))

        if n_elements < MIN_INTERACTIVE_ELEMENTS:
            raise QualityReject(
                FailureCategory.QUALITY_GATE,
                f"booted UI too trivial: {n_elements} interactive elements "
                f"(< {MIN_INTERACTIVE_ELEMENTS}); snapshot head: {probe.get('snapshot', '')[:200]}",
            )

        try:
            affordances = await _run(client.affordances)
        except Exception:
            affordances = {}

        metrics = TrialResults(
            pass_rate=1.0,
            action_diversity_score=min(n_elements / 30.0, 1.0),
            domain_relevance_score=0.8,
            trajectory_count_estimate=0,
            setup_complexity_score=0.2,
            valid_bounding_boxes=n_elements,
            screenshots_captured=1,
            notes=f"boot gate: app served at {app_url}, {n_elements} interactive elements on homepage",
            affordances=affordances,
        )
        rich = {k: v for k, v in affordances.items() if v}
        rlog.info(f"boot OK at {app_url}: {n_elements} interactive elements; affordances: {rich or 'basic'}")
        registry.update_status(
            repo_name,
            RepoStatus.VALIDATED.value,
            stage=5,
            trial_results=metrics.model_dump(),
            trial_score=metrics.weighted_score,
        )
    finally:
        await _run(sandbox.stop)


async def trial_single_repo(registry: RepoRegistry, repo: dict):
    """POC trial: run a few instrumented tests and gate on capture. Raises QualityReject for
    deterministic quality failures (no tests / no usable capture) and RetryableFailure for infra
    (clone failure); run_repo turns infra into retries, anything else into a categorized rejection."""
    repo_name = repo["full_name"]

    if repo.get("current_stage", 0) >= 5:
        return

    rlog = repo_logger(5, repo_name)
    rlog.info("trial run")

    deep_analysis = {}
    if repo.get("deep_analysis"):
        try:
            deep_analysis = (
                json.loads(repo["deep_analysis"]) if isinstance(repo["deep_analysis"], str) else repo["deep_analysis"]
            )
        except (json.JSONDecodeError, TypeError):
            pass

    profile = resolve_profile(deep_analysis.get("test_framework"), deep_analysis.get("framework_language"))
    test_name_prefix = repo_name.replace("/", "_")

    sandbox = DockerSandbox(image=profile.sandbox_image, platform=profile.sandbox_platform)
    try:
        await _run(sandbox.start, repo_name)

        _, stderr, exit_code = await _run(sandbox.clone_repo, repo_name, repo.get("default_branch"))
        if exit_code != 0:
            raise RetryableFailure(FailureCategory.CLONE_FAILED, f"clone failed: {stderr[:200]}")

        await _run(setup_repo_instrumentation, sandbox, profile)

        # Start the repo's own app (if it bundles one) so its E2E tests have something to drive —
        # the single biggest no_capture cause. None when there's no app / Playwright self-manages.
        app_spec = await _run(app_runner.detect_app_start, sandbox, profile)
        await _run(app_runner.build_app, sandbox, app_spec)

        # Locate test files via the profile's globs (skip vendored deps).
        test_files: list[str] = []
        for glob in profile.test_globs:
            for path in await _run(sandbox.list_files, glob, REPO_DIR):
                if "/node_modules/" not in path and path not in test_files:
                    test_files.append(path)
        if not test_files:
            raise QualityReject(FailureCategory.NO_TESTS, "no test files found")

        # Run up to MAX_TRIAL_FILES instrumented test files in code; stop once we capture coords.
        passed = failed = 0
        trial_outputs: list[str] = []
        for test_file in test_files[:MAX_TRIAL_FILES]:
            rel = test_file.replace(REPO_DIR + "/", "")
            cmd = profile.test_command_single.format(
                repo_name=repo_name,
                test_name_prefix=test_name_prefix,
                framework=profile.test_framework,
                framework_language=profile.language,
            ).replace("<test_file>", rel)
            cmd = app_runner.wrap_test_command(cmd, app_spec)
            out, err, _ = await _run(sandbox.exec_command, cmd, TEST_RUN_TIMEOUT)
            combined = out + err
            trial_outputs.append(f"$ {rel}\n{combined}")
            if m := _PASS_RE.search(combined):
                passed += int(m.group(1))
            if m := _FAIL_RE.search(combined):
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
        weighted = metrics.weighted_score

        if metrics.valid_bounding_boxes < 1 or metrics.screenshots_captured < 1:
            # Persist the tail of the test output so "why pass_rate=0" (app-needed vs install
            # failed vs instrumentation broke) is diagnosable after the container is gone — the
            # feedback loop for tuning app-start and selection. The container is ephemeral.
            tail = "\n".join("\n".join(trial_outputs).splitlines()[-40:])[-4000:]
            metrics.notes += f"\n--- test output tail ---\n{tail}"
            raise QualityReject(
                FailureCategory.NO_CAPTURE,
                (
                    f"no usable capture (bboxes={metrics.valid_bounding_boxes}, "
                    f"screenshots={metrics.screenshots_captured}, traj={metrics.trajectory_count_estimate})"
                ),
                trial_results=metrics.model_dump(),
                trial_score=weighted,
            )

        rlog.info(
            f"weighted={weighted:.2f} bboxes={metrics.valid_bounding_boxes} "
            f"actions={metrics.total_actions_captured} traj={metrics.trajectory_count_estimate} pass_rate={metrics.pass_rate:.0%}"
        )
        registry.update_status(
            repo_name,
            RepoStatus.VALIDATED.value,
            stage=5,
            trial_results=metrics.model_dump(),
            trial_score=weighted,
        )
    finally:
        await _run(sandbox.stop)


async def run_trial(
    registry: RepoRegistry,
    limit: int = None,
    workers: int = 2,
    budget_remaining: float = None,
    trial_mode: str = None,
):
    trial_mode = trial_mode or schemas.CAPTURE_MODE
    core_fn = boot_gate_single_repo if trial_mode == "agent" else trial_single_repo
    repos = registry.get_repos_for_stage(5, limit=limit)
    logger.info(f"Trial running {len(repos)} repos with {workers} workers (mode: {trial_mode})")

    semaphore = asyncio.Semaphore(workers)
    progress = StageProgress(5, len(repos))

    async def process_with_semaphore(repo):
        if rate_limit_tracker.limit_reached():
            return
        async with semaphore:
            await run_repo(5, repo, registry, core_fn, progress)

    await asyncio.gather(*[process_with_semaphore(repo) for repo in repos])

    return progress.summary()
