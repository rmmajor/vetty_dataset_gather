"""Run orchestration: load the task set, drive one episode per task, judge it, and write results.json + report.md."""

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from cua.eval.student.backends import EvalBrowser, build_student
from cua.eval.student.config import EVAL_RUNS_DIR, UNSCORED_OUTCOMES, EpisodeRecord, EvalConfig, step_cap_for
from cua.eval.student.episode import run_episode
from cua.eval.student.fingerprint import harness_fingerprint, step_cap_policy
from cua.eval.student.metrics import compute_metrics
from cua.eval.student.report import render_report
from cua.eval.task_sources import (
    TASK_SOURCES,
    EvalTask,
    TaskLoadConfig,
    TaskSourceUnavailable,
    episode_dir_name,
    load_tasks,
    load_tasks_from_file,
)
from cua.pipeline.sandbox import safe_parallel_sandboxes
from cua.pipeline.sdk_runner import UsageLimitReached
from cua.pipeline.teacher import VERIFIER_UNAVAILABLE_REASON, verify_trajectory
from cua.trajectory import VIEWPORT

logger = logging.getLogger(__name__)


def _write_episode_metadata(episode_dir: Path, record: EpisodeRecord, cfg: EvalConfig):
    """Write metadata whose `excluded_reason` keeps an eval episode untrainable even if it is copied into the corpus."""
    (episode_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task_id": record.task_id,
                "instruction": record.instruction,
                "start_url": record.start_url,
                "website": record.website,
                "domain": record.domain,
                "total_steps": record.driver_steps,
                "success": record.verified_success,
                "capture_source": "student_eval",
                "excluded_reason": "eval_episode",
                "eval_label": cfg.label,
                "eval_model": cfg.model,
                "viewport": {"width": VIEWPORT[0], "height": VIEWPORT[1]},
            },
            indent=2,
        )
    )


async def judge_episode(record: EpisodeRecord, cfg: EvalConfig):
    """Score one recorded episode with the repo's existing verifier (the WebJudge analogue)."""
    episode_dir = Path(record.episode_dir)
    if not (episode_dir / "trajectory.json").exists():
        record.error = record.error or "no trajectory.json recorded — nothing to judge"
        return
    verdict = await verify_trajectory(
        episode_dir,
        full_name=f"eval:{cfg.label}:{record.task_id}",
        mid_screenshots=cfg.judge_mid_screenshots,
    )
    # An unavailable verifier defers as a usage limit does, rather than charge the model for an unmeasured episode.
    if verdict.get("reasoning") == VERIFIER_UNAVAILABLE_REASON:
        record.judge_deferred = True
        record.judge_reasoning = VERIFIER_UNAVAILABLE_REASON
        return
    # `final_success` carries the capture gate's loop override, which the raw `success` key would score as a win.
    record.judged_success = bool(verdict.get("claimed_success", verdict.get("success")))
    record.verified_success = bool(verdict.get("final_success", verdict.get("success")))
    record.efficiency = verdict.get("efficiency") or ""
    record.judge_reasoning = (verdict.get("reasoning") or "")[:400]


def _make_run_id(cfg: EvalConfig) -> str:
    if cfg.run_id:
        return cfg.run_id
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{cfg.label}_{cfg.task_source}_{stamp}"


def _load_existing_episode(episode_dir: Path) -> EpisodeRecord | None:
    record_file = episode_dir / "episode.json"
    if not record_file.exists():
        return None
    try:
        return EpisodeRecord(**json.loads(record_file.read_text()))
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def _persist_episode(episode_dir: Path, record: EpisodeRecord, cfg: EvalConfig):
    if not episode_dir.exists():
        return
    _write_episode_metadata(episode_dir, record, cfg)
    (episode_dir / "episode.json").write_text(json.dumps(asdict(record), indent=2))


async def _resume_episode(episode_dir: Path, task: EvalTask, cfg: EvalConfig) -> EpisodeRecord | None:
    """The recorded episode when this task is already settled on disk, re-judging one that only owes a verdict."""
    existing = _load_existing_episode(episode_dir)
    if existing is None:
        return None
    if not (existing.judge_deferred and cfg.judge):
        logger.info(f"[{task.task_id}] already evaluated ({existing.outcome}) — skipping")
        return existing
    # The browser work is already on disk; only the verdict is owed.
    logger.info(f"[{task.task_id}] re-judging a deferred episode")
    existing.judge_deferred = False
    await judge_episode(existing, cfg)
    _persist_episode(episode_dir, existing, cfg)
    return existing


async def _judge_fresh_episode(record: EpisodeRecord, cfg: EvalConfig, episode_dir: Path, task: EvalTask):
    """Attach a verdict to a just-recorded episode, persisting first when the usage limit cuts the run short."""
    try:
        if cfg.judge and record.outcome not in UNSCORED_OUTCOMES:
            await judge_episode(record, cfg)
        elif not cfg.judge and record.outcome not in UNSCORED_OUTCOMES:
            # --no-judge OWES a verdict: without it the episode reads as a judged failure and a resume skips it forever.
            record.judge_deferred = True
    except UsageLimitReached:
        # The episode ran and only the verdict is owed, so a resume must re-judge from disk.
        record.error = record.error or "judge skipped: Claude usage limit reached"
        record.judge_deferred = True
        _persist_episode(episode_dir, record, cfg)
        raise
    except Exception as e:
        logger.warning(f"[{task.task_id}] judging failed: {e}")
        record.error = record.error or f"judge failed: {str(e)[:200]}"


async def _process_task(
    task: EvalTask,
    cfg: EvalConfig,
    browser: EvalBrowser,
    run_dir: Path,
    source_default_cap: int,
    semaphore: asyncio.Semaphore,
) -> EpisodeRecord:
    episode_dir = run_dir / "episodes" / episode_dir_name(task)
    max_steps = step_cap_for(task, cfg, source_default_cap)
    async with semaphore:
        if cfg.resume:
            resumed = await _resume_episode(episode_dir, task, cfg)
            if resumed is not None:
                return resumed

        # A fresh client per episode: the replay student walks a recorded trajectory statefully.
        record = await run_episode(browser, build_student(cfg), task, cfg, episode_dir, max_steps)
        await _judge_fresh_episode(record, cfg, episode_dir, task)
        _persist_episode(episode_dir, record, cfg)
        logger.info(
            f"[{task.task_id}] {record.outcome} | verified={record.verified_success} | "
            f"{record.model_steps} steps | {record.duration_s}s"
        )
        return record


def _load_task_set(cfg: EvalConfig):
    """(source, tasks, provenance) from the registry loader, or straight from a pinned --tasks-file."""
    if cfg.tasks_file:
        tasks = load_tasks_from_file(Path(cfg.tasks_file), cfg.limit)
        return TASK_SOURCES[cfg.task_source], tasks, {"tasks_file": cfg.tasks_file, "rows_available": len(tasks)}
    load_cfg = TaskLoadConfig(
        limit=cfg.limit,
        revision=cfg.dataset_revision,
        cache_dir=EVAL_RUNS_DIR / ".task_cache",
        trajectories_dir=Path(cfg.trajectories_dir),
        stratify=cfg.stratify,
    )
    return load_tasks(cfg.task_source, load_cfg)


def _open_run_dir(run_id: str) -> Path:
    run_dir = EVAL_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(run_dir / "run.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    return run_dir


async def _gather_episodes(coros) -> tuple:
    """(records, aborted_reason): a usage limit stops the run but never discards the episodes already recorded."""
    records: list[EpisodeRecord] = []
    aborted_reason = ""
    for result in await asyncio.gather(*coros, return_exceptions=True):
        if isinstance(result, UsageLimitReached):
            aborted_reason = f"Claude usage limit reached during judging ({result})"
            logger.error(aborted_reason)
        elif isinstance(result, BaseException):
            logger.exception(f"episode raised: {result}")
        else:
            records.append(result)
    return records, aborted_reason


async def run_eval(cfg: EvalConfig) -> dict:
    source, tasks, provenance = _load_task_set(cfg)
    run_id = _make_run_id(cfg)
    run_dir = _open_run_dir(run_id)

    logger.info(
        f"run {run_id}: {len(tasks)} tasks from {source.name} "
        f"(levels: {provenance.get('level_distribution', {})}), "
        f"step cap {step_cap_policy(cfg, source)}, coord_space={cfg.coord_space}, "
        f"student={cfg.student} model={cfg.model or '(n/a)'}"
    )

    workers = safe_parallel_sandboxes(max(1, cfg.parallel))
    if workers != cfg.parallel:
        logger.info(f"sandbox parallelism {cfg.parallel} -> {workers}")
    browsers = [EvalBrowser(cfg, f"{run_id}_w{i}") for i in range(workers)]
    semaphores = [asyncio.Semaphore(1) for _ in browsers]

    started_at = datetime.now(UTC).isoformat()
    try:
        coros = [
            _process_task(task, cfg, browsers[i % workers], run_dir, source.default_step_cap, semaphores[i % workers])
            for i, task in enumerate(tasks)
        ]
        records, aborted_reason = await _gather_episodes(coros)
    finally:
        for browser in browsers:
            await browser.stop()

    results = {
        "run_id": run_id,
        "label": cfg.label,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "config": asdict(cfg) | {"api_key": "<redacted>" if cfg.api_key else ""},
        "harness": harness_fingerprint(cfg, source, provenance),
        "task_provenance": provenance,
        "aborted_reason": aborted_reason,
        "episodes": [asdict(r) for r in records],
        "metrics": compute_metrics(records),
    }
    _write_run_files(run_dir, results)
    return results


def _write_run_files(run_dir: Path, results: dict):
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    (run_dir / "report.md").write_text(render_report(results))
    logger.info(f"wrote {run_dir / 'report.md'} and {run_dir / 'results.json'}")


def _records_for_judging(run_dir: Path) -> tuple:
    """(results summary, records): episode.json on disk wins, and a killed run with no results.json still judges."""
    # results.json is an end-of-run snapshot: stale about what is still owed, or absent when the run was killed.
    on_disk = {}
    for episode_file in sorted((run_dir / "episodes").glob("*/episode.json")):
        existing = _load_existing_episode(episode_file.parent)
        if existing is not None:
            on_disk[existing.task_id] = existing
    try:
        results = load_run(run_dir)
        return results, [on_disk.get(episode["task_id"], EpisodeRecord(**episode)) for episode in results["episodes"]]
    except (FileNotFoundError, TaskSourceUnavailable):
        logger.warning(f"{run_dir.name}: no results.json — rebuilding from {len(on_disk)} episode(s) on disk")
        return {
            "run_id": run_dir.name,
            "label": run_dir.name,
            "config": {},
            "harness": {},
            "task_provenance": {},
            "aborted_reason": "run interrupted; summary rebuilt from episodes",
        }, list(on_disk.values())


async def judge_run(run_dir: Path, cfg: EvalConfig) -> dict:
    """Score the episodes a --no-judge run deferred, without re-driving the browser."""
    results, records = _records_for_judging(run_dir)
    owed = [record for record in records if record.judge_deferred]
    logger.info(f"{run_dir.name}: {len(owed)} episode(s) owe a verdict of {len(records)} recorded")

    judged = 0
    for record in owed:
        episode_dir = Path(record.episode_dir) if record.episode_dir else run_dir / "episodes" / record.task_id
        try:
            record.judge_deferred = False
            await judge_episode(record, cfg)
        except UsageLimitReached:
            # The verdict is still owed; leave the flag set so a later pass picks it up.
            record.judge_deferred = True
            _persist_episode(episode_dir, record, cfg)
            logger.warning(f"usage limit reached after {judged} verdict(s) — rerun to continue")
            break
        _persist_episode(episode_dir, record, cfg)
        judged += 1
        logger.info(
            f"[{record.task_id}] verified={record.verified_success} "
            f"deferred={record.judge_deferred} ({judged}/{len(owed)})"
        )

    results["episodes"] = [asdict(record) for record in records]
    results["metrics"] = compute_metrics(records)
    results["finished_at"] = datetime.now(UTC).isoformat()
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    (run_dir / "report.md").write_text(render_report(results))
    logger.info(f"judged {judged}; rewrote {run_dir / 'report.md'}")
    return results


def load_run(run_dir: str | Path) -> dict:
    path = Path(run_dir)
    results_file = path if path.suffix == ".json" else path / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"{results_file} not found — is {run_dir} an eval run directory?")
    return json.loads(results_file.read_text())
