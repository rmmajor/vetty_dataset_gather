import json
import tempfile
from pathlib import Path

from repo_pipeline.agent_capture import verdict_from_parsed
from repo_pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from rl.judge import VERIFY_RETRY_S, AsyncBridge, judge_success


def test_verifier_retries_through_a_stale_usage_limit():
    attempts, sleeps = [], []

    async def flaky_verify(traj_dir, autonomy, full_name):
        attempts.append(1)
        if len(attempts) == 1:
            raise UsageLimitReached("HTTP 429")
        t = json.loads((traj_dir / "trajectory.json").read_text())
        t["success"] = True
        (traj_dir / "trajectory.json").write_text(json.dumps(t))
        (traj_dir / "capture_meta.json").write_text(json.dumps({"verifier": {"success": True}}))

    rate_limit_tracker.mark_rejected()
    bridge = AsyncBridge()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = judge_success(
                _traj_dir(tmp), {"task_id": "t"}, None, "fallback", bridge, verify=flaky_verify, sleep=sleeps.append
            )
    finally:
        bridge.close()
    assert result == (True, "verifier", "final") and len(attempts) == 2 and sleeps == [VERIFY_RETRY_S]
    assert rate_limit_tracker.limit_reached() is False


def test_verdict_success_requires_every_criterion_met():
    verdict = verdict_from_parsed(
        {
            "criteria": ["sorted by most helpful", "5-star filter on"],
            "criteria_met": [True, False],
            "success": True,
            "efficiency": "meandering",
            "reasoning": "not filtered for 5-star",
        }
    )
    assert verdict["success"] is False and verdict["claimed_success"] is True and verdict["contradiction"] is True
    assert verdict["efficiency"] == "meandering" and verdict["criteria_met"] == [True, False]
    assert verdict_from_parsed({"criteria_met": [True, True], "success": True})["success"] is True
    assert verdict_from_parsed({"success": True})["contradiction"] is False
    assert verdict_from_parsed({"success": False, "criteria_met": [True]})["success"] is False


def _traj_dir(tmp, success=False, verifier=None):
    d = Path(tmp)
    (d / "trajectory.json").write_text(
        json.dumps(
            {"task_id": "t", "instruction": "i", "start_url": "u", "steps": [], "total_steps": 0, "success": success}
        )
    )
    (d / "capture_meta.json").write_text(json.dumps({"verifier": verifier} if verifier else {}))
    return d


def test_check_wins_in_fallback_mode_without_verifier_call():
    calls = []

    async def fake_verify(traj_dir, autonomy, full_name):
        calls.append(traj_dir)

    with tempfile.TemporaryDirectory() as tmp:
        result = judge_success(
            _traj_dir(tmp),
            {"task_id": "t", "lane": "autonomous"},
            {"passed": True},
            "fallback",
            None,
            verify=fake_verify,
        )
    assert result == (True, "success_check", "final") and calls == []


def test_never_mode_returns_none_without_check():
    with tempfile.TemporaryDirectory() as tmp:
        assert judge_success(_traj_dir(tmp), {"task_id": "t"}, None, "never", None) == (None, "none", "final")


def test_verifier_runs_on_bridge_and_reads_trajectory_success():
    seen = {}

    async def fake_verify(traj_dir, autonomy, full_name):
        seen["autonomy"] = autonomy
        t = json.loads((traj_dir / "trajectory.json").read_text())
        t["success"] = True
        (traj_dir / "trajectory.json").write_text(json.dumps(t))
        (traj_dir / "capture_meta.json").write_text(json.dumps({"verifier": {"success": True}}))

    bridge = AsyncBridge()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = judge_success(
                _traj_dir(tmp),
                {"task_id": "t", "lane": "guarded"},
                {"passed": False},
                "always",
                bridge,
                verify=fake_verify,
            )
    finally:
        bridge.close()
    assert result == (True, "verifier", "final") and seen["autonomy"] == "guarded"


def test_cached_verdict_skips_verifier():
    async def boom(traj_dir, autonomy, full_name):
        raise AssertionError("must not run")

    with tempfile.TemporaryDirectory() as tmp:
        d = _traj_dir(tmp, success=False, verifier={"success": False})
        assert judge_success(d, {"task_id": "t"}, None, "fallback", None, verify=boom) == (False, "verifier", "final")
