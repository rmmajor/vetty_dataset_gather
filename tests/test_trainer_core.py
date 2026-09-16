import base64
import io
import json
import os
import tempfile
import urllib.error
from pathlib import Path

from cua.rl.actions import encode_step
from cua.rl.trainer.core import (
    ENV_CLIENTS,
    EpisodeRecorder,
    HttpEnvClient,
    MissingRewardError,
    MockEnvClient,
    action_format_score,
    env_client_for,
    episode_records,
    observation_to_user_message,
    require_reward,
)


class FakeResponse:
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _http_error(code, body):
    return urllib.error.HTTPError("http://x/reset", code, "err", {}, io.BytesIO(json.dumps(body).encode()))


def test_registry_and_scheme_dispatch():
    assert set(ENV_CLIENTS) == {"http", "mock"}
    assert isinstance(env_client_for("mock://"), MockEnvClient)
    assert isinstance(env_client_for("http://127.0.0.1:8900"), HttpEnvClient)


def test_http_client_reset_waits_through_5xx_and_fails_fast_on_4xx():
    outcomes = [
        _http_error(524, {}),
        _http_error(503, {"retry_after": 0}),
        FakeResponse({"episode_id": "e", "observation": {}}),
    ]
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, json.loads(request.data)))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = HttpEnvClient("http://primary, http://fallback/", backoff_s=0, reset_wait_s=5, opener=opener)
    assert client.reset("t", 0, "r", 0, "")["episode_id"] == "e"
    assert len(calls) == 3 and calls[0][1]["request_id"] == calls[2][1]["request_id"]
    assert [url.split("/")[2] for url, _ in calls] == ["primary", "fallback", "primary"]

    def not_found(request, timeout):
        raise _http_error(404, {"error": "unknown episode"})

    try:
        HttpEnvClient("http://x", backoff_s=0, opener=not_found).step("e", "x", seq=1)
        raise AssertionError("4xx must raise")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_mock_env_round_trip_rewards_parseable_actions():
    env = MockEnvClient(steps=2)
    first = env.reset("t1", 0, "run", 0, "tag")
    obs = first["observation"]
    assert obs["text"].startswith("Task: ") and base64.b64decode(obs["image_b64"])[:8] == b"\x89PNG\r\n\x1a\n"
    step = env.step(first["episode_id"], encode_step("go", [{"type": "wait"}]))
    assert step["done"] is False and step["observation"]["step_index"] == 1
    last = env.step(first["episode_id"], encode_step("stop", [{"type": "done"}]))
    assert last["done"] is True and last["reward"] == 1.0 and last["observation"] is None
    bad = MockEnvClient(steps=1)
    eid = bad.reset("t1", 0, "run", 0, "tag")["episode_id"]
    assert bad.step(eid, "nonsense")["reward"] == 0.0


def test_user_message_uses_image_placeholder():
    msg = observation_to_user_message({"text": "Task: x\nWhat action should you take?", "image_b64": ""})
    assert msg == {"role": "user", "content": "<image>Task: x\nWhat action should you take?"}


def test_episode_records_one_per_turn_with_broadcast_reward():
    turns = [
        {
            "user": {"role": "user", "content": "<image>u0"},
            "assistant": "a0",
            "token_ids": [1, 2],
            "image": "/tmp/0.png",
        },
        {"user": {"role": "user", "content": "<image>u1"}, "assistant": "a1", "token_ids": [3], "image": "/tmp/1.png"},
    ]
    records = episode_records("SYS", turns, 0.8, "uuid-1")
    assert len(records) == 2
    assert records[1]["messages"] == [
        {"role": "system", "content": "SYS"},
        turns[1]["user"],
        {"role": "assistant", "content": "a1"},
    ]
    assert records[1]["images"] == ["/tmp/1.png"] and records[1]["response_token_ids"] == [3]
    assert records[0]["rollout_infos"] == {
        "total_reward": 0.8,
        "num_turns": 2,
        "turn": 0,
        "trajectory_uuid": "uuid-1",
        "images": ["/tmp/0.png"],
        "env_error": None,
    }


def test_recorder_writes_png_and_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        recorder = EpisodeRecorder(tmp)
        png = MockEnvClient().reset("t", 0, "r", 0, "")["observation"]["image_b64"]
        path = recorder.save_png("u", 3, png)
        assert Path(path).name == "u_003.png" and Path(path).read_bytes()[:4] == b"\x89PNG"
        recorder.log({"trajectory_uuid": "u", "total_reward": 1.0})
        assert json.loads((Path(tmp) / "episodes.jsonl").read_text().splitlines()[0])["total_reward"] == 1.0


def test_action_format_score_levels():
    assert action_format_score("<think>t</think>\naction: done()") == 1.0
    assert action_format_score("<think>t</think>\naction: done()\ntrailing prose") == 0.5
    assert action_format_score("no actions here") == 0.0


def test_require_reward_refuses_none_and_keeps_zero():
    assert require_reward(0.0, "u") == 0.0 and require_reward(1.9, "u") == 1.9
    try:
        require_reward(None, "u-missing")
    except MissingRewardError as e:
        assert "u-missing" in str(e)
    else:
        raise AssertionError("None reward must raise")


def test_mock_env_no_reward_knob_simulates_verdictless_service():
    # This test sets the mock-env switch MockEnvClient reads at construction
    os.environ["RL_MOCK_NO_REWARD"] = "1"
    try:
        client = MockEnvClient(steps=1)
        episode = client.reset("t", 0, "run", 0, "")["episode_id"]
        result = client.step(episode, "<think>t</think>\naction: done()")
        assert result["done"] and result["reward"] is None
        assert client.close(episode)["reward"] is None
    finally:
        # Paired with the set above so the toggle never leaks into another test
        del os.environ["RL_MOCK_NO_REWARD"]
