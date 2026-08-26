"""ms-swift-free half of the GRPO plugin: env clients, message/record shaping, format reward — unit-tested on the Mac."""
import base64
import io
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from rl.actions import decode_step
from rl.observation import build_observation

logger = logging.getLogger(__name__)


class HttpEnvClient:
    """Talks to rl/service.py (through the reverse SSH tunnel on the box); retries so a tunnel blip is not an episode failure."""

    def __init__(self, base_url, token="", timeout_s=600, retries=6, backoff_s=5, **_):
        self.base_url, self.token, self.timeout_s, self.retries, self.backoff_s = base_url.rstrip("/"), token, timeout_s, retries, backoff_s

    def _request(self, method, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-RL-Token"] = self.token
        data = json.dumps(body).encode() if body is not None else None
        last = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as e:
                if e.code != 503:
                    raise
                last = e
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last = e
            time.sleep(self.backoff_s * (attempt + 1))
        raise ConnectionError(f"env service unreachable after {self.retries} attempts: {last}")

    def reset(self, task_id, sample_idx, run_id, iteration, policy_tag):
        return self._request("POST", "/reset", {"task_id": task_id, "sample_k": sample_idx, "run_id": run_id, "iter": iteration, "policy_tag": policy_tag})

    def step(self, episode_id, response_text):
        return self._request("POST", "/step", {"episode_id": episode_id, "response_text": response_text})

    def close(self, episode_id, reason="external"):
        return self._request("POST", "/close", {"episode_id": episode_id, "reason": reason})

    def tasks(self, split="train"):
        return self._request("GET", f"/tasks?split={split}")["tasks"]


def _blank_png_b64(size):
    buffer = io.BytesIO()
    Image.new("RGB", size, (250, 250, 250)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class MockEnvClient:
    """Offline stand-in for the env service: fixed PNG, reward 1.0 iff the final response decodes, done after `steps`."""

    def __init__(self, steps=3, image_size=(1280, 720), **_):
        self.steps, self.png = steps, _blank_png_b64(image_size)
        self.episodes = {}

    def reset(self, task_id, sample_idx, run_id, iteration, policy_tag):
        episode_id = f"mock-{task_id}-{sample_idx}-{len(self.episodes)}"
        self.episodes[episode_id] = {"task_id": task_id, "history": [], "n": 0}
        return {"episode_id": episode_id, "observation": build_observation(f"mock task {task_id}", [], self.png, 0)}

    def step(self, episode_id, response_text):
        state = self.episodes[episode_id]
        decoded = decode_step(response_text)
        state["n"] += 1
        state["history"].append("mock_action")
        done = state["n"] >= self.steps or any(a["type"] == "done" for a in decoded.actions)
        if done:
            return {"observation": None, "reward": 1.0 if decoded.ok else 0.0, "done": True,
                    "info": {"terminal_reason": "done" if decoded.ok else "parse_abort"}}
        observation = build_observation(f"mock task {state['task_id']}", state["history"], self.png, state["n"])
        return {"observation": observation, "reward": None, "done": False, "info": {"parse": "ok" if decoded.ok else "failure"}}

    def close(self, episode_id, reason="external"):
        return {"reward": 0.0, "info": {"terminal_reason": reason}}

    def tasks(self, split="train"):
        return [{"task_id": "mock__task", "split": split}]


ENV_CLIENTS = {"http": HttpEnvClient, "mock": MockEnvClient}


def env_client_for(url, **kwargs):
    if url.startswith("mock://"):
        return ENV_CLIENTS["mock"](**kwargs)
    return ENV_CLIENTS["http"](url, **kwargs)


def observation_to_user_message(observation):
    """ms-swift multimodal convention: an <image> placeholder in the text, the file in `images`."""
    return {"role": "user", "content": "<image>" + observation["text"]}


def episode_records(system, turns, total_reward, trajectory_uuid):
    records = []
    for index, turn in enumerate(turns):
        records.append({
            "messages": [{"role": "system", "content": system}, turn["user"], {"role": "assistant", "content": turn["assistant"]}],
            "images": [turn["image"]],
            "response_token_ids": turn.get("token_ids"),
            "rollout_infos": {"total_reward": total_reward, "num_turns": len(turns), "turn": index,
                              "trajectory_uuid": trajectory_uuid, "images": [turn["image"]]},
        })
    return records


class EpisodeRecorder:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        (self.run_dir / "obs").mkdir(parents=True, exist_ok=True)

    def save_png(self, trajectory_uuid, turn, image_b64):
        path = self.run_dir / "obs" / f"{trajectory_uuid}_{turn:03d}.png"
        path.write_bytes(base64.b64decode(image_b64))
        return str(path)

    def log(self, record):
        with (self.run_dir / "episodes.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")


def action_format_score(text):
    decoded = decode_step(text)
    if not decoded.ok:
        return 0.0
    return 1.0 if not decoded.errors else 0.5
