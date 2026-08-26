import json
import tempfile
from pathlib import Path

from rl.policies import MockReplayPolicy, ScriptedPolicy, make_policy
from rl.runner import summarize


def test_mock_replay_emits_recorded_steps_then_done():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "trajectory.json").write_text(json.dumps({"steps": [
            {"thought": "a", "actions": [{"type": "click", "x": 1, "y": 2, "button": "left"}]},
            {"thought": "", "actions": [], "event": {"kind": "approval", "decision": "approve"}},
            {"thought": "b", "actions": [{"type": "done"}]}]}))
        policy = MockReplayPolicy(tmp)
    assert policy.act({}) == "<think>a</think>\naction: click(<ref>(1, 2)</ref>, button=left)"
    assert policy.act({}) == "<think>b</think>\naction: done()"
    assert policy.act({}).endswith("action: done()")


def test_scripted_repeats_last_text():
    policy = ScriptedPolicy(["x", "y"])
    assert [policy.act({}) for _ in range(3)] == ["x", "y", "y"]


def test_make_policy_registry():
    assert isinstance(make_policy({"kind": "scripted", "texts": ["z"]}), ScriptedPolicy)


def test_summarize_counts():
    records = [{"reward": 1.0, "success": True, "terminal_reason": "done", "turns": [{"clean": True}], "steps": 3, "reward_status": "final"},
               {"reward": None, "success": None, "terminal_reason": "step_cap", "turns": [{"clean": False}], "steps": 5, "reward_status": "deferred"}]
    s = summarize(records)
    assert s == {"episodes": 2, "final": 1, "deferred": 1, "success_rate": 0.5, "mean_reward": 1.0, "done_rate": 0.5, "parse_failure_rate": 0.5, "mean_steps_success": 3.0}
