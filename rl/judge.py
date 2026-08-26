"""Terminal success for a rollout dir: programmatic check first, Claude verifier as fallback/audit — all SDK calls on one background loop."""
import asyncio
import json
import os
import threading
from pathlib import Path

from repo_pipeline.agent_capture import verify_trajectory
from repo_pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker

VERIFIER_CONCURRENCY = int(os.environ.get("RL_VERIFIER_CONCURRENCY", "2"))
VERIFY_TIMEOUT_S = int(os.environ.get("RL_VERIFY_TIMEOUT_S", "2700"))


class AsyncBridge:
    """One event loop thread for every Claude SDK call so rate_limit_tracker's single-loop assumption holds."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.semaphore = asyncio.Semaphore(VERIFIER_CONCURRENCY)
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


def judge_success(traj_dir, task, check_result, mode, bridge, verify=verify_trajectory):
    traj_dir = Path(traj_dir)
    if check_result is not None and mode != "always":
        return bool(check_result["passed"]), "success_check", "final"
    if mode == "never":
        return (bool(check_result["passed"]), "success_check", "final") if check_result else (None, "none", "final")
    if cached_verdict(traj_dir) is None:
        if rate_limit_tracker.limit_reached():
            return None, "verifier", "deferred"
        try:
            bridge.run(_verify(traj_dir, task, bridge, verify), timeout=VERIFY_TIMEOUT_S)
        except UsageLimitReached:
            return None, "verifier", "deferred"
    traj = json.loads((traj_dir / "trajectory.json").read_text())
    return bool(traj.get("success")), "verifier", "final"
