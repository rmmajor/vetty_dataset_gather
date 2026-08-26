import base64
import json
import tempfile
from pathlib import Path

from rl.actions import encode_step
from rl.trainer.core import (ENV_CLIENTS, EpisodeRecorder, HttpEnvClient, MockEnvClient, action_format_score,
                             env_client_for, episode_records, observation_to_user_message)


def test_registry_and_scheme_dispatch():
    assert set(ENV_CLIENTS) == {"http", "mock"}
    assert isinstance(env_client_for("mock://"), MockEnvClient)
    assert isinstance(env_client_for("http://127.0.0.1:8900"), HttpEnvClient)


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
    turns = [{"user": {"role": "user", "content": "<image>u0"}, "assistant": "a0", "token_ids": [1, 2], "image": "/tmp/0.png"},
             {"user": {"role": "user", "content": "<image>u1"}, "assistant": "a1", "token_ids": [3], "image": "/tmp/1.png"}]
    records = episode_records("SYS", turns, 0.8, "uuid-1")
    assert len(records) == 2
    assert records[1]["messages"] == [{"role": "system", "content": "SYS"}, turns[1]["user"], {"role": "assistant", "content": "a1"}]
    assert records[1]["images"] == ["/tmp/1.png"] and records[1]["response_token_ids"] == [3]
    assert records[0]["rollout_infos"] == {"total_reward": 0.8, "num_turns": 2, "turn": 0, "trajectory_uuid": "uuid-1", "images": ["/tmp/0.png"]}


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
