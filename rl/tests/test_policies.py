import json
import tempfile
from pathlib import Path

import ssl
import urllib.error

from rl.policies import MockReplayPolicy, OpenAICompatiblePolicy, ScriptedPolicy, make_policy
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


class FakeChatResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "<think>t</think>\naction: done()"}}]}).encode()


def _policy(outcomes, slept):
    def opener(request, timeout=None):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    ticks = iter(range(0, 100000, 7))
    return OpenAICompatiblePolicy("http://x/v1", "m", opener=opener, backoff_s=0, sleep=slept.append,
                                  wait_s=20, clock=lambda: next(ticks))


def test_policy_retries_a_dropped_connection_then_succeeds():
    slept = []
    policy = _policy([ssl.SSLEOFError("EOF"), FakeChatResponse()], slept)
    assert policy.act({"system": "s", "text": "t", "image_b64": "", "step_index": 0}).endswith("done()")
    assert len(slept) == 1


def test_policy_gives_up_when_the_wait_budget_runs_out_and_never_retries_4xx():
    slept = []
    policy = _policy([ssl.SSLEOFError("EOF")] * 10, slept)
    try:
        policy.act({"system": "s", "text": "t", "image_b64": "", "step_index": 0})
        raise AssertionError("must raise once the budget is spent")
    except ConnectionError as e:
        assert "unreachable after" in str(e)
    fatal = _policy([urllib.error.HTTPError("http://x", 400, "bad", {}, None)], [])
    try:
        fatal.act({"system": "s", "text": "t", "image_b64": "", "step_index": 0})
        raise AssertionError("4xx must not be retried")
    except urllib.error.HTTPError:
        pass

