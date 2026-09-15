"""Terminal success for a rollout dir: the programmatic check first, the Claude verifier as fallback or audit."""

import asyncio
import json
import logging
import threading
import time
from pathlib import Path

from cua.pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from cua.pipeline.teacher import verify_trajectory
from cua.settings import RL

logger = logging.getLogger(__name__)


class AsyncBridge:
    """One event loop thread for every Claude SDK call so rate_limit_tracker's single-loop assumption holds."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.semaphore = asyncio.Semaphore(RL.verifier_concurrency)
        self.thread = threading.Thread(target=self.loop.run_forever, name="rl-async-bridge", daemon=True)
        self.thread.start()

    def run(self, coro, timeout=None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


def cached_verdict(traj_dir):
    meta_file = Path(traj_dir) / "capture_meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    return meta.get("verifier")


async def _verify(traj_dir, task, bridge, verify):
    async with bridge.semaphore:
        await verify(traj_dir, autonomy=task.get("lane", "autonomous"), full_name=f"rl:{task.get('task_id', '')}")


def _run_verifier(traj_dir, task, bridge, verify, sleep):
    # The tracker's rejected flag is sticky, so a stale limit would defer every verdict for the rest of the run.
    deadline = time.monotonic() + RL.verify_wait_s
    while True:
        if rate_limit_tracker.limit_reached():
            rate_limit_tracker.reset()
        try:
            bridge.run(_verify(traj_dir, task, bridge, verify), timeout=RL.verify_timeout_s)
            return True
        except UsageLimitReached as e:
            if time.monotonic() >= deadline:
                logger.warning(f"{traj_dir.name}: verdict deferred — {e}")
                return False
            logger.warning(f"{traj_dir.name}: usage limit — retrying the verifier in {RL.verify_retry_s:.0f}s ({e})")
            sleep(RL.verify_retry_s)


def judge_success(traj_dir, task, check_result, mode, bridge, verify=verify_trajectory, sleep=time.sleep):
    traj_dir = Path(traj_dir)
    if check_result is not None and mode != "always":
        return bool(check_result["passed"]), "success_check", "final"
    if mode == "never":
        return (bool(check_result["passed"]), "success_check", "final") if check_result else (None, "none", "final")
    if cached_verdict(traj_dir) is None and not _run_verifier(traj_dir, task, bridge, verify, sleep):
        return None, "verifier", "deferred"
    traj = json.loads((traj_dir / "trajectory.json").read_text())
    return bool(traj.get("success")), "verifier", "final"
