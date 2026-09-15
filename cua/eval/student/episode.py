"""One task, start to finish: model coordinates into driver calls, and the two abort rules that end a wandering run."""

import base64
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from cua.eval.blocks import detect_block
from cua.eval.student.backends import EvalBrowser, StudentClient, StudentError
from cua.eval.student.config import EpisodeRecord, EvalConfig
from cua.eval.student.grammar import HANDOFF_VERBS, parse_student_output
from cua.eval.task_sources import EvalTask, episode_dir_name
from cua.pipeline.driver_session import DriverError, run_blocking
from cua.pipeline.instrumentation import RESULTS_DIR
from cua.prompt.format import history_entries_for_step
from cua.student.executor import (
    COORD_SPACE_RANGE,
    COORD_SPACE_SUSPECT,
    action_coordinates,
    driver_payload,
    outside_coord_space,
    scale_action,
)
from cua.student.observation import build_student_messages, observed_effect

logger = logging.getLogger(__name__)

SNAPSHOT_RECORD_CHARS = 4000


def _note_emitted_coordinates(record: EpisodeRecord, actions: list, cfg: EvalConfig):
    """Count emitted coordinates outside the declared space's range and name the likely coordinate-space fix once."""
    hint, other_space = COORD_SPACE_SUSPECT[cfg.coord_space]
    max_x, max_y = COORD_SPACE_RANGE[cfg.coord_space]
    for action in actions:
        for x, y in action_coordinates(action):
            record.emitted_coords += 1
            if not outside_coord_space(x, y, cfg.coord_space):
                continue
            record.emitted_out_of_viewport += 1
            if record.coord_space_warning:
                continue
            record.coord_space_warning = (
                f"emitted ({x}, {y}) is outside 0-{max_x} x 0-{max_y} under coord_space={cfg.coord_space}; "
                f"{hint} — re-run with --coord-space {other_space}"
            )
            logger.warning(f"[{record.task_id}] COORDINATE SPACE: {record.coord_space_warning}")


def _new_record(task: EvalTask, episode_dir: Path, max_steps: int) -> EpisodeRecord:
    return EpisodeRecord(
        task_id=task.task_id,
        instruction=task.instruction,
        start_url=task.start_url,
        website=task.website,
        domain=task.domain,
        split_tag=task.split_tag,
        level=task.level,
        reference_length=task.reference_length,
        step_cap=max_steps,
        episode_dir=str(episode_dir),
    )


async def _start_session(client, task: EvalTask, cfg: EvalConfig, record: EpisodeRecord):
    """The opened driver session, or None after recording the episode as unreachable."""
    try:
        return await run_blocking(
            lambda: client.session(
                task.task_id,
                task.start_url,
                task.instruction,
                {"expected_actions": task.reference_length, "eval_label": cfg.label, "eval_source": task.source},
                type_mode=cfg.type_mode,
            )
        )
    except (DriverError, ConnectionError) as e:
        record.outcome = "unreachable"
        record.error = str(e)[:300]
        return None


def _step_messages(task, cfg, action_history, executed_actions, prev_png, prev_url, cur_png, cur_url) -> list:
    """This step's chat messages, after folding the previous step's observed effect into the history."""
    if prev_png is not None:
        # The last step enters the history only now: its effect is what THIS screenshot/url show.
        action_history.extend(
            history_entries_for_step(
                {"actions": executed_actions}, observed_effect(prev_png, cur_png, prev_url, cur_url)
            )
        )
    return build_student_messages(task.instruction, action_history, cur_png, prev_png, observation=cfg.observation)


async def _ask_student(client, student, task, cfg, action_history, executed_actions, prev_png, prev_url, cur_url):
    """(screenshot, raw model text, error) for one step; a non-empty error aborts the episode as harness_error."""
    try:
        cur_png = base64.b64decode(await run_blocking(client.screenshot_b64, cfg.screenshot_format))
    except (DriverError, ConnectionError) as e:
        return None, "", f"screenshot failed: {str(e)[:250]}"
    messages = _step_messages(task, cfg, action_history, executed_actions, prev_png, prev_url, cur_png, cur_url)
    try:
        return cur_png, await run_blocking(student.complete, messages), ""
    except StudentError as e:
        return cur_png, "", f"student: {str(e)[:250]}"


def _record_step_parse(record: EpisodeRecord, parsed, raw: str, step: int):
    """Fold one model turn's grammar counters into the episode record."""
    record.model_steps = step
    record.raw_last_response = (raw or "")[-1500:]
    record.malformed_lines += parsed.malformed_lines
    record.actions_parsed += len(parsed.actions)
    record.actions_canonical += parsed.canonical_actions
    for deviation, count in parsed.deviations.items():
        record.form_deviations[deviation] = record.form_deviations.get(deviation, 0) + count
    if parsed.think:
        record.think_steps += 1


def _abort_outcome(cfg: EvalConfig, parse_failures: int, unproductive: int) -> str:
    """The outcome the two abort rules have reached, or "" while the episode may continue."""
    if parse_failures >= cfg.max_parse_failures:
        return "parse_failure_abort"
    if unproductive >= cfg.max_unproductive_steps:
        return "stuck"
    return ""


def _record_block(record: EpisodeRecord, url: str, snapshot: str, step: int) -> bool:
    """Record a bot wall as the episode's outcome; True when the episode must stop there."""
    blocked = detect_block(url, snapshot)
    if not blocked:
        return False
    record.outcome = "site_blocked"
    record.block_reason = blocked
    record.block_step = step
    return True


class _Dispatch(NamedTuple):
    terminated: bool
    handed_off: bool
    executed: int
    url: str
    snapshot: str


async def _dispatch_actions(client, record, parsed, cfg, step, executed_actions, url, snapshot) -> _Dispatch:
    """Execute one step's parsed actions through the driver, stopping at the first terminal one."""
    executed_count = 0
    for action in parsed.actions:
        if action["type"] in HANDOFF_VERBS:
            record.handoff_kind = action["type"]
            record.handoff_reason = action.get("reason") or action.get("risk_category", "")
            return _Dispatch(True, True, executed_count, url, snapshot)
        executed = scale_action(action, cfg.coord_space)
        payload = driver_payload(executed, parsed.think)
        if payload is None:
            continue
        if action["type"] == "done":
            record.done_emitted = True
            try:
                await run_blocking(client.act, payload)
            except (DriverError, ConnectionError) as e:
                record.driver_errors += 1
                record.error = record.error or f"done failed: {str(e)[:200]}"
            return _Dispatch(True, False, executed_count, url, snapshot)
        try:
            result = await run_blocking(client.act, payload)
        except (DriverError, ConnectionError) as e:
            message = str(e)
            record.driver_errors += 1
            if "outside the" in message:
                record.out_of_bounds += 1
            logger.info(f"[{record.task_id}] step {step}: driver refused {action['type']}: {message[:160]}")
            continue
        record.actions_executed += 1
        executed_count += 1
        if action["type"] == "navigate":
            record.navigations += 1
        # History is derived from the EXECUTED pixel action, so its coordinates round-trip as in training.
        executed_actions.append(executed)
        url = result.get("url", url)
        snapshot = result.get("snapshot", snapshot)
    return _Dispatch(False, False, executed_count, url, snapshot)


def _log_stuck(record: EpisodeRecord, unproductive: int):
    logger.warning(
        f"[{record.task_id}] aborting: {unproductive} consecutive "
        f"steps executed no action ({record.driver_errors} driver refusals, "
        f"{record.out_of_bounds} out of viewport)"
    )


async def _close_session(client, sandbox, record: EpisodeRecord, episode_dir: Path, container_dir: str):
    """Close the driver session and copy the recorded trajectory out of the container for the judge."""
    summary = f"{record.actions_executed} actions executed; outcome {record.outcome}"
    try:
        finish = await run_blocking(client.finish, record.outcome, summary)
        record.driver_steps = finish.get("steps", 0)
    except Exception as e:
        logger.warning(f"[{record.task_id}] driver finish failed: {e}")
    episode_dir.mkdir(parents=True, exist_ok=True)
    try:
        await run_blocking(sandbox.copy_out, container_dir, str(episode_dir))
        await run_blocking(sandbox.exec_command, f"rm -rf {container_dir}")
    except Exception as e:
        logger.warning(f"[{record.task_id}] copy_out failed: {e}")


@dataclass
class _Cursor:
    """What one step carries into the next: the last observation, the pending history and the two abort counters."""

    url: str
    snapshot: str = ""
    history: list = field(default_factory=list)
    executed: list = field(default_factory=list)
    prev_png: bytes | None = None
    prev_url: str = ""
    parse_failures: int = 0
    unproductive: int = 0


async def _run_step(client, student, task, cfg, record: EpisodeRecord, cursor: _Cursor, step: int) -> str:
    """Observe, ask, parse, dispatch. Returns the outcome that ends the episode, or "" to take another step."""
    cur_url = cursor.url
    cur_png, raw, failure = await _ask_student(
        client, student, task, cfg, cursor.history, cursor.executed, cursor.prev_png, cursor.prev_url, cur_url
    )
    if failure:
        record.error = failure
        return "harness_error"
    cursor.prev_png, cursor.prev_url, cursor.executed = cur_png, cur_url, []

    parsed = parse_student_output(raw, cfg.tolerant_grammar)
    _record_step_parse(record, parsed, raw, step)
    if parsed.parse_failed:
        record.parse_failure_steps += 1
        cursor.parse_failures += 1
        cursor.unproductive += 1
        logger.warning(f"[{task.task_id}] step {step}: no parseable action in {(raw or '')[:200]!r}")
        return _abort_outcome(cfg, cursor.parse_failures, cursor.unproductive)
    cursor.parse_failures = 0

    _note_emitted_coordinates(record, parsed.actions, cfg)
    dispatched = await _dispatch_actions(
        client, record, parsed, cfg, step, cursor.executed, cursor.url, cursor.snapshot
    )
    cursor.url, cursor.snapshot = dispatched.url, dispatched.snapshot
    if dispatched.terminated:
        return "handoff" if dispatched.handed_off else "completed"
    cursor.unproductive = 0 if dispatched.executed else cursor.unproductive + 1
    if cursor.unproductive >= cfg.max_unproductive_steps:
        _log_stuck(record, cursor.unproductive)
        return "stuck"
    return record.outcome if _record_block(record, cursor.url, cursor.snapshot, step) else ""


async def run_episode(
    browser: EvalBrowser, student: StudentClient, task: EvalTask, cfg: EvalConfig, episode_dir: Path, max_steps: int
) -> EpisodeRecord:
    """Run one task start to finish, recording its trajectory into `episode_dir` for the judge."""
    record = _new_record(task, episode_dir, max_steps)
    started = time.time()
    cursor = _Cursor(url=task.start_url)
    container_dir = f"{RESULTS_DIR}/{episode_dir_name(task)}"
    session_open = False

    await browser.ensure()
    client = browser.client

    try:
        session = await _start_session(client, task, cfg, record)
        if session is None:
            return record
        session_open = True
        cursor.url = session.get("url", task.start_url)
        cursor.snapshot = session.get("snapshot", "")
        if _record_block(record, cursor.url, cursor.snapshot, 0):
            return record

        for step in range(1, max_steps + 1):
            if time.time() - started > cfg.episode_timeout_s:
                record.outcome = "timeout"
                break
            outcome = await _run_step(client, student, task, cfg, record, cursor, step)
            if outcome:
                record.outcome = outcome
                break
        else:
            record.outcome = "step_cap"
    finally:
        record.final_url = cursor.url or ""
        record.final_snapshot = (cursor.snapshot or "")[:SNAPSHOT_RECORD_CHARS]
        if session_open:
            await _close_session(client, browser.sandbox, record, episode_dir, container_dir)
        record.duration_s = round(time.time() - started, 1)

    return record
