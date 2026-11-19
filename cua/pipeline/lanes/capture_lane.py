"""The one capture loop every lane runs: boot a driver, run the teacher per task, copy out, verify, finalize."""

import json
import logging
from pathlib import Path
from urllib.parse import urlparse

from cua.pipeline.driver_session import run_blocking
from cua.pipeline.instrumentation import RESULTS_DIR
from cua.pipeline.teacher import run_teacher_task, verify_trajectory
from cua.pipeline.think_generator import needs_polish, polish_thoughts_for_trajectory
from cua.settings import CAPTURE
from cua.trajectory import VIEWPORT
from cua.trajectory.io import CAPTURE_META_FILE, METADATA_FILE, TRAJECTORY_FILE, write_json
from cua.trajectory.schema import metadata_from, safe_task_id

logger = logging.getLogger(__name__)

# Host-only sidecar: every copy_out overwrites capture_meta.json, so a retry counter kept there resets.
ATTEMPTS_FILE = "capture_attempts.json"
# The burner address is masked at TYPE time, so the teacher naming it in a thought is not a masking failure.
LEAK_EXEMPT_KEYS = frozenset({"email"})


class SecretLeak(Exception):
    """A masked credential reached the recording literally, so driver-side substitution failed."""


def is_valid_trajectory(traj_dir: Path) -> bool:
    """Whether a recording has at least one real-coordinate action and every screenshot it names."""
    trajectory_file = traj_dir / TRAJECTORY_FILE
    if not trajectory_file.exists():
        return False
    try:
        traj = json.loads(trajectory_file.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    steps = traj.get("steps", [])
    if not steps:
        return False
    has_valid_coords = any(
        any(a.get("x", 0) > 0 or a.get("y", 0) > 0 for a in step.get("actions", [])) for step in steps
    )
    screenshots_exist = all((traj_dir / step.get("screenshot", "")).exists() for step in steps)
    return has_valid_coords and screenshots_exist


def capture_attempts(traj_dir: Path) -> int:
    """How many finalized capture attempts this trajectory has already cost."""
    try:
        return int(json.loads((traj_dir / ATTEMPTS_FILE).read_text()).get("attempts", 0))
    except (OSError, ValueError, TypeError):
        return 0


def record_capture_attempt(traj_dir: Path, outcome: str) -> int:
    """Count one finalized capture attempt against this trajectory's retry bound."""
    attempts = capture_attempts(traj_dir) + 1
    (traj_dir / ATTEMPTS_FILE).write_text(json.dumps({"attempts": attempts, "last_outcome": outcome}, indent=2))
    return attempts


def capture_verdict(traj_dir: Path) -> str | None:
    """A recorded trajectory's settled verdict, or None when the verifier never judged it."""
    try:
        if "verifier" not in json.loads((traj_dir / CAPTURE_META_FILE).read_text()):
            return None
        succeeded = json.loads((traj_dir / TRAJECTORY_FILE).read_text()).get("success")
    except (OSError, ValueError):
        return None
    return "success" if succeeded else "failed"


def is_capture_exhausted(traj_dir: Path) -> bool:
    """A recorded trajectory that failed verification and has no attempts left."""
    return capture_verdict(traj_dir) == "failed" and capture_attempts(traj_dir) >= CAPTURE.max_attempts


def is_capture_settled(traj_dir: Path) -> bool:
    """Whether task-level resume must leave this recorded trajectory alone."""
    if not is_valid_trajectory(traj_dir):
        return False
    # A None verdict is an interrupted run's orphan awaiting finalization, not a failure to re-capture.
    return capture_verdict(traj_dir) != "failed" or is_capture_exhausted(traj_dir)


def needs_verification(traj_dir: Path) -> bool:
    """A recording an interrupted run finalized but never verified, which resume must finish rather than skip."""
    if not is_valid_trajectory(traj_dir):
        return False
    try:
        return "verifier" not in json.loads((traj_dir / CAPTURE_META_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return False


def secret_leak_check(traj_dir: Path, secrets: dict | None) -> None:
    """Raise unless every masked secret outside LEAK_EXEMPT_KEYS is absent from the recording."""
    if not secrets:
        return
    blob = (traj_dir / TRAJECTORY_FILE).read_text()
    leaked = sorted(
        key for key, value in secrets.items() if key not in LEAK_EXEMPT_KEYS and value and str(value) in blob
    )
    if leaked:
        raise SecretLeak(f"{', '.join(leaked)} appear(s) literally in {traj_dir.name}/{TRAJECTORY_FILE}")


async def copy_out_site(sandbox, results_dir: str, site_dir: Path) -> None:
    """Pull the container's recordings into the host's site/repo dir, creating it on first use."""
    site_dir.mkdir(parents=True, exist_ok=True)
    await run_blocking(sandbox.copy_out, results_dir, str(site_dir))


async def record_capture(state: dict, task: dict, *, sandbox, site_dir: Path, label: str) -> dict:
    """Copy one finished task's recording out of the container and log what the teacher did."""
    # Per task, not per site, so a container timeout mid-task costs only that task and not the whole site.
    await copy_out_site(sandbox, RESULTS_DIR, site_dir)
    logger.info(f"[{label}] task '{task['id']}': {state['actions']} actions, outcome={state['agent_outcome']}")
    return state


async def capture_task(client, task: dict, start_url: str, *, sandbox, site_dir: Path, label: str, **teacher) -> dict:
    """Run the teacher on one task and copy what it recorded out immediately."""
    state = await run_teacher_task(client, task, start_url, **teacher)
    return await record_capture(state, task, sandbox=sandbox, site_dir=site_dir, label=label)


async def verify_and_finalize(traj_dir: Path, site: dict, task: dict, state: dict, registry) -> dict:
    """Judge one recording against its goal and write the metadata.json that makes it a corpus entry."""
    verdict = await verify_trajectory(traj_dir, state=state, registry=registry, full_name=f"site:{site['name']}")
    traj = json.loads((traj_dir / TRAJECTORY_FILE).read_text())
    capture_meta = json.loads((traj_dir / CAPTURE_META_FILE).read_text())
    verifier = capture_meta.get("verifier", {})
    task_meta = capture_meta.get("task_meta") or {}
    # The trajectory's terminal risk class: the last consequential action the agent paused on.
    risks = [
        a.get("risk_category", "") for s in traj["steps"] for a in s.get("actions", []) if a.get("type") == "confirm"
    ]
    metadata = metadata_from(
        task_id=traj["task_id"],
        instruction=traj["instruction"],
        start_url=traj["start_url"],
        total_steps=traj["total_steps"],
        success=traj["success"],
        website=urlparse(site["start_url"]).netloc,
        domain=site.get("domain", "public_web"),
        framework="agent",
        capture_source="agent",
        instruction_source=task.get("instruction_source", "curated"),
        thought_source="teacher_raw",
        task_type="other",
        difficulty=task.get("difficulty", "medium"),
        efficiency=verifier.get("efficiency"),
        efficiency_ratio=verifier.get("efficiency_ratio"),
        relabeled=verifier.get("relabeled", False),
        viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]},
        source_repo=f"public:{site['name']}",
        autonomy=task_meta.get("autonomy", "autonomous"),
        auth_mode=task_meta.get("auth_mode", "none"),
        task_style=task_meta.get("task_style", ""),
        risk_category=risks[-1] if risks else "",
        # Replay manifest: what an eval/deploy harness must reconstruct for these actions to be valid.
        preconditions={
            "requires_session": task_meta.get("auth_mode") == "pre_session",
            "session_ref": site.get("storage_state_ref", ""),
            "start_url": traj["start_url"],
            "env": task_meta.get("env", "live"),
        },
    )
    write_json(traj_dir / METADATA_FILE, metadata)

    attempts = record_capture_attempt(traj_dir, "success" if traj["success"] else "failed")
    if not traj["success"] and attempts >= CAPTURE.max_attempts:
        logger.warning(
            f"[{site['name']}] task '{task['id']}' has now failed {attempts}x — resume will "
            f"stop re-capturing it (raise CAPTURE_MAX_ATTEMPTS to keep trying)"
        )

    if traj["success"] and needs_polish(traj_dir):
        await polish_thoughts_for_trajectory(traj_dir, registry=registry, full_name=f"site:{site['name']}")
    return verdict


async def finalize_tasks(
    tasks: list,
    site: dict,
    site_dir: Path,
    states: dict,
    registry,
    *,
    secrets: dict | None = None,
    require_valid: bool = False,
) -> dict[str, dict]:
    """Verify + finalize host-side, returning `{task_id: {"verdict": ..., "leaked": bool}}` per recorded task."""
    results: dict[str, dict] = {}
    for task in tasks:
        traj_dir = site_dir / f"task_{safe_task_id(task['id'])}"
        recorded = is_valid_trajectory(traj_dir) if require_valid else traj_dir.exists()
        if not recorded:
            continue
        # `states.get`, not `[...]`: an orphan from an interrupted run has no state, so the verifier reads capture_meta.
        verdict = await verify_and_finalize(traj_dir, site, task, states.get(task["id"]), registry)
        leaked = False
        if secrets:
            try:
                secret_leak_check(traj_dir, secrets)
            except SecretLeak as e:
                leaked = True
                logger.error(f"[{site['name']}] SECRET LEAK in '{task['id']}': {e}")
        results[task["id"]] = {"verdict": verdict, "leaked": leaked}
        logger.info(
            f"[{site['name']}] task '{task['id']}': verified={verdict.get('final_success')} "
            f"({verdict.get('efficiency')})" + (f" leaked={leaked}" if secrets else "")
        )
    return results
