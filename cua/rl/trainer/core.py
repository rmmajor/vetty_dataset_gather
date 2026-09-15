"""ms-swift-free half of the GRPO plugin: env clients, message and record shaping, and the format reward."""

import base64
import gzip
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

from cua.rl.actions import decode_step
from cua.rl.observation import build_observation
from cua.settings import RL

logger = logging.getLogger(__name__)


def _retry_after(error):
    try:
        return float(json.loads(error.read()).get("retry_after") or 0) or None
    except Exception:
        return None


class HttpEnvClient:
    """Talks to cua/rl/service.py through the ingress, where every 5xx is retried and every 4xx is fatal."""

    def __init__(
        self,
        base_url,
        token="",
        timeout_s=600,
        retries=6,
        backoff_s=5,
        reset_wait_s=RL.reset_wait_s,
        step_wait_s=RL.step_wait_s,
        opener=urllib.request.urlopen,
        **_,
    ):
        self.step_wait_s = step_wait_s
        # A comma-separated list is an ordered set of ingresses to one service, so a dead tunnel cannot end the run.
        self.base_urls = [u.strip().rstrip("/") for u in base_url.split(",") if u.strip()]
        self.token, self.timeout_s = token, timeout_s
        self.retries, self.backoff_s, self.reset_wait_s, self.opener = retries, backoff_s, reset_wait_s, opener

    def _request(self, method, path, body=None, wait_s=None):
        headers = {"Content-Type": "application/json", "Accept-Encoding": "gzip"}
        if self.token:
            headers["X-RL-Token"] = self.token
        data = json.dumps(body).encode() if body is not None else None
        deadline = time.monotonic() + wait_s if wait_s else None
        last, attempt = None, 0
        while True:
            try:
                request = urllib.request.Request(
                    f"{self.base_urls[attempt % len(self.base_urls)]}{path}", data=data, headers=headers, method=method
                )
                with self.opener(request, timeout=self.timeout_s) as response:
                    raw = response.read()
                    if (getattr(response, "headers", None) or {}).get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    return json.loads(raw)
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    raise
                last, delay = e, _retry_after(e)
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last, delay = e, None
            attempt += 1
            if (deadline is None and attempt >= self.retries) or (
                deadline is not None and time.monotonic() >= deadline
            ):
                break
            time.sleep(delay or self.backoff_s * min(attempt, 6))
        raise ConnectionError(f"env service unreachable after {attempt} attempts: {last}")

    def reset(self, task_id, sample_idx, run_id, iteration, policy_tag):
        # A reset waits far longer than a step, and request_id lands a cut-off retry on the same episode.
        body = {
            "task_id": task_id,
            "sample_k": sample_idx,
            "run_id": run_id,
            "iter": iteration,
            "policy_tag": policy_tag,
            "request_id": uuid.uuid4().hex,
        }
        return self._request("POST", "/reset", body, wait_s=self.reset_wait_s)

    def step(self, episode_id, response_text, seq=None):
        return self._request(
            "POST",
            "/step",
            {"episode_id": episode_id, "response_text": response_text, "seq": seq},
            wait_s=self.step_wait_s,
        )

    def close(self, episode_id, reason="external"):
        return self._request("POST", "/close", {"episode_id": episode_id, "reason": reason}, wait_s=self.step_wait_s)

    def tasks(self, split="train"):
        return self._request("GET", f"/tasks?split={split}")["tasks"]


def blank_png_b64(size):
    buffer = io.BytesIO()
    Image.new("RGB", size, (250, 250, 250)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class MockEnvClient:
    """Offline stand-in for the env service: a fixed PNG, and reward 1.0 only when the final response decodes."""

    def __init__(self, steps=3, image_size=(1280, 720), **_):
        self.steps, self.png = steps, blank_png_b64(image_size)
        # env-read: the GRPO smoke toggles this to simulate a verdictless service the run must refuse
        self.no_reward = os.environ.get("RL_MOCK_NO_REWARD") == "1"
        self.episodes = {}

    def reset(self, task_id, sample_idx, run_id, iteration, policy_tag):
        episode_id = f"mock-{task_id}-{sample_idx}-{len(self.episodes)}"
        self.episodes[episode_id] = {"task_id": task_id, "history": [], "n": 0}
        return {"episode_id": episode_id, "observation": build_observation(f"mock task {task_id}", [], self.png, 0)}

    def step(self, episode_id, response_text, seq=None):
        state = self.episodes[episode_id]
        decoded = decode_step(response_text)
        state["n"] += 1
        state["history"].append("mock_action")
        done = state["n"] >= self.steps or any(a["type"] == "done" for a in decoded.actions)
        if done:
            return {
                "observation": None,
                "reward": None if self.no_reward else (1.0 if decoded.ok else 0.0),
                "done": True,
                "info": {"terminal_reason": "done" if decoded.ok else "parse_abort"},
            }
        observation = build_observation(f"mock task {state['task_id']}", state["history"], self.png, state["n"])
        return {
            "observation": observation,
            "reward": None,
            "done": False,
            "info": {"parse": "ok" if decoded.ok else "failure"},
        }

    def close(self, episode_id, reason="external"):
        return {"reward": None if self.no_reward else 0.0, "info": {"terminal_reason": reason}}

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


class MissingRewardError(RuntimeError):
    pass


def require_reward(reward, trajectory_uuid):
    # None means a verdictless episode was about to train as a fabricated 0.0 (the grpo-v2 failure); refuse loudly.
    if reward is None:
        raise MissingRewardError(f"episode {trajectory_uuid} finished without a reward")
    return float(reward)


def episode_records(system, turns, total_reward, trajectory_uuid, env_error=None):
    records = []
    for index, turn in enumerate(turns):
        records.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    turn["user"],
                    {"role": "assistant", "content": turn["assistant"]},
                ],
                "images": [turn["image"]],
                "response_token_ids": turn.get("token_ids"),
                "rollout_infos": {
                    "total_reward": total_reward,
                    "num_turns": len(turns),
                    "turn": index,
                    "trajectory_uuid": trajectory_uuid,
                    "images": [turn["image"]],
                    "env_error": env_error,
                },
            }
        )
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
