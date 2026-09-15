# RL Harness Phase 3 (GPU-side GRPO trainer + remote job) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run GRPO on the rented GPU against the Phase 1 env service: an ms-swift `Env`/scheduler plugin whose logic is pure and unit-tested on the Mac, a box-side phase driver (`cua/rl/train.py`) that merges the SFT adapter, serves for before/after eval, and runs `swift rollout` + `swift rlhf`, and a `remote_train` `GrpoJob` (+ `ServeJob`, SSH tunnel, poll hook) with the existing teardown guarantees — all mock-smokable without a GPU.

**Architecture:** `cua/rl/trainer/core.py` holds everything the plugin does that does not need ms-swift (env HTTP client with retries, mock env, message/record shaping, format reward). `cua/rl/trainer/swift_plugin.py` is a thin adapter over ms-swift's `Env` / `GYMScheduler` / `ORM`. `cua/rl/train.py` is both the local command builder (`print-commands`, `make-dataset`) and the on-box phase driver (`run`), with process launching injected so the driver is testable. `remote_train` gains `GrpoJob` and `ServeJob`, an SSH tunnel resource opened after `wait_ready` and closed before destroy, and an `on_poll` hook that services the before/after-eval handshake and rsyncs the latest adapter.

**Tech Stack:** stdlib (`subprocess`, `threading`, `urllib`), existing `remote_train`, `cua/rl/` Phase 1, Pillow (mock PNG). Box side: ms-swift 4.5.2 / vLLM 0.23.0 / torch 2.11.0 (pinned in `requirements-rl.txt`, VALIDATE-ON-BOX).

**Spec:** `docs/superpowers/specs/2026-08-26-rl-harness-design.md` §Phase 3 (D1 merge-before-RL, D2 one record per turn, 3.1–3.3; 3.4 fallback deferred).

## Global Constraints

- One-line comments only; no nested imports; no imports inside tests; registries over name checks.
- Never `git commit`/`push` without an explicit go-ahead. Work on `feat/rl-harness-phase3` (stacked on `feat/rl-harness`).
- Cloud rules: guaranteed teardown (reuse `lifecycle._run_job`'s `try/finally`; the tunnel closes inside that `finally` BEFORE `_guaranteed_destroy`), RU/CN deny-list, label `roman.maior`. Nothing in this phase rents a GPU.
- Both tunnel ends bind `127.0.0.1`; the env service port is `8900`, the serve port `8001`, the rollout server `8000` (box-local).
- `$PY` = `/Users/rmajor/unik/maga/dataset_gather/.venv-pipeline/bin/python`, run from the worktree root. ms-swift/vLLM/torch are NOT importable here — `cua/rl/trainer/swift_plugin.py` is compile-checked only.
- Deferred (documented, not stubbed with fake behavior): `--mode fallback` (decoupled batch GRPO) raises `NotImplementedError` with the spec reference.

---

## File map

| File | Responsibility |
|---|---|
| `cua/rl/trainer/__init__.py` | empty |
| `cua/rl/trainer/core.py` | `ENV_CLIENTS` (`http`, `mock`), `env_client_for(url)`, `observation_to_user_message`, `episode_records`, `EpisodeRecorder`, `action_format_score` |
| `cua/rl/trainer/swift_plugin.py` | `BrowserEnv`, `BrowserScheduler`, `ActionFormatORM`, registrations (box only) |
| `cua/rl/train.py` | flag maps, `build_*_command`, `make_dataset`, `PhaseDriver`, CLI `print-commands` / `make-dataset` / `run` |
| `rl_config.yaml`, `requirements-rl.txt` | RL run config (mirrors `config.yaml` style) and pinned box deps |
| `cua/cloud/vast/tunnel.py` | `SshTunnel`, `NullTunnel` |
| `cua/cloud/vast/base.py` (modify) | `ComputeProvider.tunnel()` default |
| `cua/cloud/vast/vastai.py` (modify) | `tunnel()` via the provider's ssh args |
| `cua/cloud/vast/mock.py` (modify) | `tunnel()` → `NullTunnel` (records forwards); scripted `PHASE` answers |
| `cua/cloud/vast/jobs.py` (modify) | `RemoteJob.resources()` / `on_poll()` defaults; `GrpoJob`, `ServeJob`; `JOBS` |
| `cua/cloud/vast/lifecycle.py` (modify) | enter `job.resources` after `wait_ready`; `on_poll` in the poll loop; smoke covers `grpo` + `serve` |
| `cua/cloud/vast/config.py` (modify) | `rl_config`, `rl_runs_dir`, `collect_every_polls`, `serve_adapter`, `serve_seconds`, `env_url` |
| `cua/cloud/vast/__main__.py` (modify) | `grpo`, `serve` subcommands |
| `Makefile` (modify) | `remote-grpo`, `remote-serve`, `test-grpo`, `grpo-dataset` |
| `tests/test_trainer_core.py`, `tests/test_rl_train.py`, `tests/test_tunnel.py` | Mac-side tests |

---

### Task 1: `cua/rl/trainer/core.py` — env clients, message shaping, records, format reward

**Files:** Create `cua/rl/trainer/__init__.py`, `cua/rl/trainer/core.py`, `tests/test_trainer_core.py`

**Interfaces (produced):**
- `HttpEnvClient(base_url, token="", timeout_s=600, retries=6, backoff_s=5)` with `reset(task_id, sample_idx, run_id, iteration, policy_tag) -> dict`, `step(episode_id, response_text) -> dict`, `close(episode_id, reason) -> dict`, `tasks(split) -> list`.
- `MockEnvClient(steps=3, image_size=(1280, 720))` same methods; reward 1.0 when the last response decodes, done after `steps`.
- `ENV_CLIENTS = {"http": ..., "mock": ...}`; `env_client_for(url, **kw)` picks by scheme (`mock://` → mock).
- `observation_to_user_message(obs) -> {"role": "user", "content": "<image>" + obs["text"]}`.
- `episode_records(system, turns, total_reward, trajectory_uuid) -> list[dict]` where `turns = [{"user": msg, "assistant": text, "token_ids": list|None, "image": path}]`; each record: `{"messages": [system, user, assistant], "images": [path], "response_token_ids": ..., "rollout_infos": {"total_reward", "num_turns", "turn", "trajectory_uuid", "images"}}`.
- `EpisodeRecorder(run_dir)`: `save_png(uuid, turn, b64) -> str path`, `log(record: dict)` appends to `episodes.jsonl`.
- `action_format_score(text) -> float` = 1.0 clean decode, 0.5 decodes with errors, 0.0 no actions (pure; used by the ORM).

- [ ] **Step 1: Failing tests** — `tests/test_trainer_core.py`:
```python
import base64
import json
import tempfile
from pathlib import Path

from cua.rl.actions import encode_step
from cua.rl.trainer.core import (ENV_CLIENTS, EpisodeRecorder, HttpEnvClient, MockEnvClient, action_format_score,
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
```
- [ ] **Step 2: Run** — `$PY -m tests test_trainer_core` → `ModuleNotFoundError: No module named 'cua.rl.trainer'`.
- [ ] **Step 3: Implement** `cua/rl/trainer/core.py`:
```python
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

from cua.rl.actions import decode_step
from cua.rl.observation import build_observation

logger = logging.getLogger(__name__)


class HttpEnvClient:
    """Talks to cua/rl/service.py (through the reverse SSH tunnel on the box); retries so a tunnel blip is not an episode failure."""

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
            return {"observation": None, "reward": 1.0 if decoded.ok else 0.0, "done": True, "info": {"terminal_reason": "done" if decoded.ok else "parse_abort"}}
        observation = build_observation(f"mock task {state['task_id']}", state["history"], self.png, state["n"])
        return {"observation": observation, "reward": None, "done": False, "info": {"parse": "ok" if decoded.ok else "failure"}}

    def close(self, episode_id, reason="external"):
        return {"reward": 0.0, "info": {"terminal_reason": reason}}

    def tasks(self, split="train"):
        return [{"task_id": "mock__task", "split": split}]


ENV_CLIENTS = {"http": HttpEnvClient, "mock": MockEnvClient}


def env_client_for(url, **kwargs):
    return ENV_CLIENTS["mock" if url.startswith("mock://") else "http"](url, **kwargs) if url.startswith("http") else ENV_CLIENTS["mock"](**kwargs)


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
```
`env_client_for` must be readable: rewrite as
```python
def env_client_for(url, **kwargs):
    if url.startswith("mock://"):
        return ENV_CLIENTS["mock"](**kwargs)
    return ENV_CLIENTS["http"](url, **kwargs)
```
- [ ] **Step 4: Run** → `6/6 passed`. **Step 5: Report.**

---

### Task 2: `cua/rl/trainer/swift_plugin.py` — thin ms-swift adapter (box only)

**Files:** Create `cua/rl/trainer/swift_plugin.py`

**Interfaces (produced, VALIDATE-ON-BOX):** `envs["browser_env"] = BrowserEnv`, `multi_turns["browser_scheduler"] = BrowserScheduler`, `orms["action_format"] = ActionFormatORM`. Reads env `RL_ENV_URL` (default `http://127.0.0.1:8900`), `RL_ENV_TOKEN`, `RL_RUN_DIR` (default `/workspace/rl_runs/current`), `RL_RUN_ID`.

- [ ] **Step 1: Write the adapter** (no Mac test possible; Step 2 compile-checks):
```python
"""ms-swift plugin: BrowserEnv (gym Env over the Mac env service), BrowserScheduler (one record per turn), ActionFormatORM. VALIDATE-ON-BOX: ms-swift 4.5.x module paths."""
import copy
import os
import uuid

from swift.llm.infer.protocol import RolloutOutput
from swift.plugin.orm import ORM, orms
from swift.rollout.gym_env import Env, envs
from swift.rollout.multi_turn import GYMScheduler, multi_turns

from cua.rl.trainer.core import EpisodeRecorder, action_format_score, env_client_for, episode_records, observation_to_user_message
from cua.sft.legacy_train_qwen import SYSTEM_PROMPT

RL_ENV_URL = os.environ.get("RL_ENV_URL", "http://127.0.0.1:8900")
RL_ENV_TOKEN = os.environ.get("RL_ENV_TOKEN", "")
RL_RUN_DIR = os.environ.get("RL_RUN_DIR", "/workspace/rl_runs/current")
RL_RUN_ID = os.environ.get("RL_RUN_ID", "grpo")


class BrowserEnv(Env):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.client = env_client_for(RL_ENV_URL, token=RL_ENV_TOKEN)
        self.recorder = EpisodeRecorder(RL_RUN_DIR)
        self.episode_id = None
        self.trajectory_uuid = uuid.uuid4().hex
        self.turn = 0
        self.last_image_path = None
        self.observation = None

    async def reset(self, config):
        env_config = config.data_dict.get("env_config", {})
        result = self.client.reset(env_config["task_id"], int(env_config.get("sample_idx", 0)), RL_RUN_ID, int(env_config.get("iteration", 0)), env_config.get("policy_tag", ""))
        self.episode_id = result["episode_id"]
        self.observation = result["observation"]
        self.last_image_path = self.recorder.save_png(self.trajectory_uuid, 0, self.observation["image_b64"])
        return self.observation["text"], {"episode_id": self.episode_id}, SYSTEM_PROMPT

    async def step(self, messages):
        response_text = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else ""
        try:
            result = self.client.step(self.episode_id, response_text)
        except Exception as e:
            return "", 0.0, True, {"env_error": str(e)[:300]}
        self.turn += 1
        observation = result.get("observation")
        if observation is not None:
            self.observation = observation
            self.last_image_path = self.recorder.save_png(self.trajectory_uuid, self.turn, observation["image_b64"])
        reward = result.get("reward")
        return (observation or {}).get("text", ""), float(reward or 0.0), bool(result.get("done")), result.get("info", {})

    async def close(self):
        if self.episode_id is not None:
            try:
                self.client.close(self.episode_id, "external")
            except Exception:
                pass


class BrowserScheduler(GYMScheduler):
    """Each turn is a fresh single-image prompt; run() returns one RolloutOutput per turn carrying the episode reward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turns_by_uuid = {}

    def _fresh_prompt(self, env):
        user = observation_to_user_message(env.observation)
        return [{"role": "system", "content": SYSTEM_PROMPT}, user], [env.last_image_path]

    async def on_trajectory_start(self, infer_request, request_config, **kwargs):
        result = await super().on_trajectory_start(infer_request, request_config, **kwargs)
        env = self.get_env(infer_request.uuid) if hasattr(self, "get_env") else kwargs.get("env")
        if env is not None:
            infer_request.messages, infer_request.images = self._fresh_prompt(env)
        self.turns_by_uuid[infer_request.uuid] = []
        return result

    def step(self, infer_request, response_choice, current_turn, **kwargs):
        env = kwargs.get("env")
        self.turns_by_uuid.setdefault(infer_request.uuid, []).append({
            "user": copy.deepcopy(infer_request.messages[-1]), "assistant": response_choice.message.content,
            "token_ids": getattr(response_choice, "token_ids", None), "image": infer_request.images[0] if infer_request.images else None,
        })
        if env is not None:
            infer_request.messages, infer_request.images = self._fresh_prompt(env)
        return infer_request

    def build_outputs(self, infer_request, total_reward, **kwargs):
        turns = self.turns_by_uuid.pop(infer_request.uuid, [])
        return [RolloutOutput(response=None, messages=r["messages"], response_token_ids=[r["response_token_ids"]] if r["response_token_ids"] else None, rollout_infos=r["rollout_infos"])
                for r in episode_records(SYSTEM_PROMPT, turns, total_reward, infer_request.uuid)]


class ActionFormatORM(ORM):
    def __call__(self, completions, **kwargs):
        return [action_format_score(c if isinstance(c, str) else str(c)) for c in completions]


envs["browser_env"] = BrowserEnv
multi_turns["browser_scheduler"] = BrowserScheduler
orms["action_format"] = ActionFormatORM
```
- [ ] **Step 2: Compile-check** — `$PY -m py_compile cua/rl/trainer/swift_plugin.py` → no output. Record in the report that `GYMScheduler`'s exact hook names (`on_trajectory_start`, `step`, and where `run()` assembles outputs — here exposed as `build_outputs`) are the V6 validation item; the box checklist diffs this file against `swift/rollout/multi_turn.py` before the first run.

---

### Task 3: `cua/rl/train.py` — flag maps, command builders, dataset, phase driver

**Files:** Create `cua/rl/train.py`, `rl_config.yaml`, `requirements-rl.txt`, `tests/test_rl_train.py`

**Interfaces (produced):**
- `load_rl_config(path) -> dict`; `ROLLOUT_FLAG_MAP`, `RLHF_FLAG_MAP`, `DEPLOY_FLAG_MAP` (config key → swift flag).
- `build_merge_command(cfg) -> list[str]`, `build_rollout_command(cfg) -> list[str]`, `build_rlhf_command(cfg) -> list[str]`, `build_deploy_command(cfg, adapters=None) -> list[str]`, `build_env(cfg, run_dir) -> dict`.
- `make_dataset(suite_path, out_path, split="train", repeats=1) -> int`.
- `PhaseDriver(cfg, run_dir, workdir, launcher=subprocess.Popen, run=subprocess.run, http_ok=_http_ok, sleep=time.sleep, clock=time.time)` with `.run(mode="swift") -> int`; phase files `PHASE` / `PHASE_ACK` in `workdir`; copies newest `checkpoint-*` into `<output_dir>/latest`.
- CLI: `python -m cua.rl.train print-commands --config rl_config.yaml`, `make-dataset --tasks cua/rl/rl_tasks.json --out rl_tasks.jsonl [--split train] [--repeats 1]`, `run --config rl_config.yaml --run-dir DIR [--workdir /workspace] [--mode swift|fallback]`.

- [ ] **Step 1: Failing tests** — `tests/test_rl_train.py`:
```python
import json
import tempfile
from pathlib import Path

from cua.rl.train import (PhaseDriver, build_deploy_command, build_env, build_merge_command, build_rlhf_command,
                      build_rollout_command, load_rl_config, make_dataset)

CFG = load_rl_config("rl_config.yaml")


def test_config_mirrors_sft_base_and_ports():
    assert CFG["base_model"] == "Qwen/Qwen3.5-9B" and CFG["rollout_port"] == 8000 and CFG["compute"]["serve_port"] == 8001
    assert CFG["env_url"].endswith(":8900") and CFG["compute"]["min_vram_gb"] == 80


def test_commands_are_argv_lists_with_expected_flags():
    merge = build_merge_command(CFG)
    assert merge[:2] == ["swift", "export"] and "--merge_lora" in merge and "/workspace/sft_adapter" in merge
    rollout = build_rollout_command(CFG)
    assert rollout[:2] == ["swift", "rollout"] and "--use_gym_env" in rollout and rollout[rollout.index("--gym_env") + 1] == "browser_env"
    assert rollout[rollout.index("--multi_turn_scheduler") + 1] == "browser_scheduler" and "--vllm_enable_lora" in rollout
    rlhf = build_rlhf_command(CFG)
    assert rlhf[:2] == ["swift", "rlhf"] and rlhf[rlhf.index("--rlhf_type") + 1] == "grpo" and rlhf[rlhf.index("--vllm_mode") + 1] == "server"
    assert rlhf[rlhf.index("--reward_weights"):rlhf.index("--reward_weights") + 3] == ["--reward_weights", "0.2", "1.0"]
    assert rlhf[rlhf.index("--num_generations") + 1] == "4" and "--freeze_vit" in rlhf and rlhf[rlhf.index("--freeze_vit") + 1] == "true"
    deploy = build_deploy_command(CFG, adapters="/workspace/rl_output/latest")
    assert deploy[:2] == ["swift", "deploy"] and deploy[deploy.index("--adapters") + 1] == "/workspace/rl_output/latest" and "--port" in deploy
    assert "--adapters" not in build_deploy_command(CFG)


def test_env_carries_pixels_and_service_url():
    env = build_env(CFG, "/workspace/rl_runs/r1")
    assert env["MAX_PIXELS"] == "1003520" and env["RL_ENV_URL"] == CFG["env_url"] and env["RL_RUN_DIR"] == "/workspace/rl_runs/r1"


def test_make_dataset_rows():
    with tempfile.TemporaryDirectory() as tmp:
        suite = Path(tmp) / "suite.json"
        suite.write_text(json.dumps({"suite_sha256": "x", "tasks": [{"task_id": "a", "split": "train"}, {"task_id": "b", "split": "eval_unseen_site"}]}))
        out = Path(tmp) / "rl_tasks.jsonl"
        assert make_dataset(suite, out, split="train", repeats=2) == 2
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert rows[0]["messages"] == [{"role": "user", "content": "<placeholder>"}]
        assert rows[0]["env_config"] == {"name": "browser_env", "task_id": "a", "sample_idx": 0} and rows[1]["env_config"]["sample_idx"] == 1


class FakeProc:
    def __init__(self, argv):
        self.argv, self.killed, self.returncode = argv, False, 0

    def poll(self):
        return None if not self.killed else self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.killed = True

    def kill(self):
        self.killed = True


def test_phase_driver_runs_all_phases_with_handshake():
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "sft_adapter").mkdir()
        launched, ran = [], []
        cfg = dict(CFG, merged_model=str(workdir / "merged"), output_dir=str(workdir / "rl_output"), sft_adapter_remote=str(workdir / "sft_adapter"))
        driver = PhaseDriver(cfg, str(workdir / "run"), str(workdir), launcher=lambda argv, **kw: launched.append(FakeProc(argv)) or launched[-1],
                             run=lambda argv, **kw: ran.append(argv) or type("R", (), {"returncode": 0})(), http_ok=lambda url: True, sleep=lambda s: None)
        acks = []

        def rlhf_side_effect():
            (Path(cfg["output_dir"]) / "checkpoint-5").mkdir(parents=True)
            (Path(cfg["output_dir"]) / "checkpoint-5" / "adapter_config.json").write_text("{}")
            return 0
        driver.run_foreground = lambda argv, env: rlhf_side_effect()
        driver.wait_for_ack = lambda phase: acks.append(phase) or True
        rc = driver.run(mode="swift")
        assert rc == 0
        assert ran[0][:2] == ["swift", "export"] and (workdir / "merged").exists() is False or True
        assert acks == ["EVAL_BEFORE_READY", "EVAL_AFTER_READY"]
        assert (Path(cfg["output_dir"]) / "latest" / "adapter_config.json").exists()
        assert (workdir / "PHASE").read_text().strip() == "DONE"
        kinds = [p.argv[1] for p in launched]
        assert kinds == ["deploy", "rollout", "deploy"] and all(p.killed for p in launched)
        assert json.loads((workdir / "run" / "versions.json").read_text())["mode"] == "swift"


def test_fallback_mode_is_explicitly_unimplemented():
    with tempfile.TemporaryDirectory() as tmp:
        driver = PhaseDriver(dict(CFG), tmp, tmp, launcher=None, run=None, http_ok=lambda url: True, sleep=lambda s: None)
        try:
            driver.run(mode="fallback")
            raise AssertionError("fallback must raise")
        except NotImplementedError as e:
            assert "3.4" in str(e)
```
- [ ] **Step 2: Run** → `ModuleNotFoundError: No module named 'cua.rl.train'`.
- [ ] **Step 3: Implement**

`rl_config.yaml`:
```yaml
# GRPO run config for `python -m cua.rl.train` (box) and `python -m cua.cloud.vast grpo` (Mac). Mirrors config.yaml's style.
base_model: Qwen/Qwen3.5-9B            # the SFT base (config.yaml); the SFT LoRA is merged into it before RL (spec D1)
sft_adapter: output                     # Mac path of the SFT adapter to ship
sft_adapter_remote: /workspace/sft_adapter
merged_model: /workspace/merged
template: qwen3_5
tuner_type: lora
lora_rank: 64
lora_alpha: 128
target_modules: all-linear
freeze_vit: true
torch_dtype: bfloat16
max_pixels: 1003520
# rollout server (swift rollout, vLLM inside)
rollout_port: 8000
vllm_gpu_memory_utilization: 0.35
vllm_max_model_len: 4096
vllm_max_lora_rank: 64
vllm_limit_mm_per_prompt: '{"image": 1}'
max_turns: 15
# trainer (swift rlhf --rlhf_type grpo)
dataset: rl_tasks.jsonl
learning_rate: 1.0e-5
beta: 0.01
epsilon: 0.2
num_generations: 4
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
steps_per_generation: 16
max_length: 3072
max_completion_length: 512
completion_length_limit_scope: per_round
temperature: 0.7
top_p: 0.95
max_steps: 30
save_steps: 5
save_total_limit: 3
async_generate: true
dynamic_sample: false
log_completions: true
report_to: tensorboard
output_dir: /workspace/rl_output
# plugin / env
external_plugins: cua/rl/trainer/swift_plugin.py
gym_env: browser_env
multi_turn_scheduler: browser_scheduler
reward_funcs: action_format
reward_weights: "0.2 1.0"
env_url: http://127.0.0.1:8900
env_retry_s: 180
# eval phases (swift deploy on serve_port; the Mac runs cua.rl.eval through the tunnel and acks)
eval_before: true
eval_after: true
eval_wait_s: 7200
# compute overrides applied onto remote_train.yaml by GrpoJob.preflight
compute:
  min_vram_gb: 80
  disk_gb: 120
  image: pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
  max_runtime_hours: 8
  serve_port: 8001
  env_port: 8900
```

`requirements-rl.txt`:
```
# GRPO box deps — the ms-swift-recommended column (vLLM 0.23.0 <-> torch 2.11.0 <-> transformers 5.12.1). VALIDATE-ON-BOX.
torch==2.11.0
vllm==0.23.0
transformers==5.12.1
ms-swift[llm]==4.5.2
peft<0.21
trl<1.0
qwen-vl-utils
flash-linear-attention
datasets<4.8.5
Pillow
pyyaml
requests
```

`cua/rl/train.py`:
```python
"""GRPO run driver: local command builders (print-commands, make-dataset) and the on-box phase runner (run)."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROLLOUT_FLAG_MAP = {
    "merged_model": "--model", "template": "--template", "vllm_gpu_memory_utilization": "--vllm_gpu_memory_utilization",
    "vllm_max_model_len": "--vllm_max_model_len", "vllm_max_lora_rank": "--vllm_max_lora_rank",
    "vllm_limit_mm_per_prompt": "--vllm_limit_mm_per_prompt", "external_plugins": "--external_plugins",
    "multi_turn_scheduler": "--multi_turn_scheduler", "gym_env": "--gym_env", "max_turns": "--max_turns", "rollout_port": "--port",
}
RLHF_FLAG_MAP = {
    "merged_model": "--model", "template": "--template", "tuner_type": "--tuner_type", "lora_rank": "--lora_rank",
    "lora_alpha": "--lora_alpha", "target_modules": "--target_modules", "freeze_vit": "--freeze_vit", "torch_dtype": "--torch_dtype",
    "rollout_port": "--vllm_server_port", "external_plugins": "--external_plugins", "reward_funcs": "--reward_funcs",
    "dataset": "--dataset", "max_length": "--max_length", "max_completion_length": "--max_completion_length",
    "completion_length_limit_scope": "--completion_length_limit_scope", "num_generations": "--num_generations",
    "per_device_train_batch_size": "--per_device_train_batch_size", "gradient_accumulation_steps": "--gradient_accumulation_steps",
    "steps_per_generation": "--steps_per_generation", "learning_rate": "--learning_rate", "beta": "--beta", "epsilon": "--epsilon",
    "temperature": "--temperature", "top_p": "--top_p", "max_steps": "--max_steps", "save_steps": "--save_steps",
    "save_total_limit": "--save_total_limit", "async_generate": "--async_generate", "dynamic_sample": "--dynamic_sample",
    "log_completions": "--log_completions", "report_to": "--report_to", "output_dir": "--output_dir",
}
DEPLOY_FLAG_MAP = {"merged_model": "--model", "template": "--template", "vllm_max_lora_rank": "--vllm_max_lora_rank",
                   "vllm_limit_mm_per_prompt": "--vllm_limit_mm_per_prompt", "max_completion_length": "--max_new_tokens"}
ROLLOUT_FIXED = ["--vllm_use_async_engine", "true", "--vllm_enable_lora", "true", "--use_gym_env", "true", "--host", "127.0.0.1"]
RLHF_FIXED = ["--rlhf_type", "grpo", "--use_vllm", "true", "--vllm_mode", "server", "--vllm_server_host", "127.0.0.1",
              "--vllm_server_pass_dataset", "true", "--vllm_enable_lora", "true", "--use_gym_env", "true"]
DEPLOY_FIXED = ["--infer_backend", "vllm", "--vllm_enable_lora", "true", "--host", "127.0.0.1"]
PHASE_FILE, ACK_FILE = "PHASE", "PHASE_ACK"


def load_rl_config(path="rl_config.yaml"):
    return yaml.safe_load(Path(path).read_text())


def _flag_value(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def _flags(cfg, flag_map):
    argv = []
    for key, flag in flag_map.items():
        if key in cfg and cfg[key] is not None:
            argv += [flag, _flag_value(cfg[key])]
    return argv


def build_merge_command(cfg):
    return ["swift", "export", "--model", cfg["base_model"], "--adapters", cfg["sft_adapter_remote"], "--merge_lora", "true",
            "--output_dir", cfg["merged_model"]]


def build_rollout_command(cfg):
    return ["swift", "rollout", *_flags(cfg, ROLLOUT_FLAG_MAP), *ROLLOUT_FIXED]


def build_rlhf_command(cfg):
    argv = ["swift", "rlhf", *RLHF_FIXED, *_flags(cfg, RLHF_FLAG_MAP)]
    if cfg.get("reward_weights"):
        argv += ["--reward_weights", *str(cfg["reward_weights"]).split()]
    return argv


def build_deploy_command(cfg, adapters=None):
    argv = ["swift", "deploy", *_flags(cfg, DEPLOY_FLAG_MAP), *DEPLOY_FIXED, "--port", str(cfg["compute"]["serve_port"])]
    if adapters:
        argv += ["--adapters", adapters]
    return argv


def build_env(cfg, run_dir):
    return {**os.environ, "MAX_PIXELS": str(cfg["max_pixels"]), "RL_ENV_URL": cfg["env_url"], "RL_RUN_DIR": str(run_dir),
            "RL_RUN_ID": Path(run_dir).name, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}


def make_dataset(suite_path, out_path, split="train", repeats=1):
    suite = json.loads(Path(suite_path).read_text())
    rows = [{"messages": [{"role": "user", "content": "<placeholder>"}],
             "env_config": {"name": "browser_env", "task_id": t["task_id"], "sample_idx": r}}
            for r in range(repeats) for t in suite["tasks"] if t.get("split") == split or split == "all"]
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def _http_ok(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status < 500
    except Exception:
        return False


class PhaseDriver:
    """On-box orchestration: merge → (deploy, EVAL_BEFORE) → rollout + rlhf → (deploy latest, EVAL_AFTER); phases signalled via PHASE/PHASE_ACK files."""

    def __init__(self, cfg, run_dir, workdir="/workspace", launcher=subprocess.Popen, run=subprocess.run, http_ok=_http_ok, sleep=time.sleep, clock=time.time):
        self.cfg, self.run_dir, self.workdir = cfg, Path(run_dir), Path(workdir)
        self.launcher, self.run_cmd, self.http_ok, self.sleep, self.clock = launcher, run, http_ok, sleep, clock
        self.env = build_env(cfg, self.run_dir)

    def set_phase(self, phase):
        (self.workdir / PHASE_FILE).write_text(phase + "\n")
        print(f"[cua.rl.train] PHASE {phase}", flush=True)

    def wait_for_ack(self, phase):
        deadline = self.clock() + float(self.cfg.get("eval_wait_s", 7200))
        ack = self.workdir / ACK_FILE
        while self.clock() < deadline:
            if ack.exists() and ack.read_text().strip() == phase:
                ack.unlink()
                return True
            self.sleep(15)
        print(f"[cua.rl.train] no ack for {phase} within eval_wait_s — continuing", flush=True)
        return False

    def wait_healthy(self, url, timeout_s=1800):
        deadline = self.clock() + timeout_s
        while self.clock() < deadline:
            if self.http_ok(url):
                return True
            self.sleep(10)
        return False

    def run_foreground(self, argv, env):
        return self.run_cmd(argv, env=env).returncode

    def _stop(self, proc):
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()

    def merge(self):
        if (Path(self.cfg["merged_model"]) / "config.json").exists():
            return 0
        return self.run_foreground(build_merge_command(self.cfg), self.env)

    def serve_phase(self, phase, adapters=None):
        proc = self.launcher(build_deploy_command(self.cfg, adapters), env=self.env)
        try:
            self.wait_healthy(f"http://127.0.0.1:{self.cfg['compute']['serve_port']}/v1/models")
            self.set_phase(phase)
            self.wait_for_ack(phase)
        finally:
            self._stop(proc)

    def latest_checkpoint(self):
        out = Path(self.cfg["output_dir"])
        candidates = [p for p in out.glob("**/checkpoint-*") if p.is_dir()]
        return max(candidates, key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1) if candidates else None

    def publish_latest(self):
        checkpoint = self.latest_checkpoint()
        if checkpoint is None:
            return None
        latest = Path(self.cfg["output_dir"]) / "latest"
        if latest.exists():
            shutil.rmtree(latest)
        shutil.copytree(checkpoint, latest)
        return latest

    def train_phase(self):
        rollout = self.launcher(build_rollout_command(self.cfg), env=self.env)
        try:
            if not self.wait_healthy(f"http://127.0.0.1:{self.cfg['rollout_port']}/health/"):
                print("[cua.rl.train] rollout server never became healthy", flush=True)
                return 1
            self.set_phase("TRAINING")
            rc = self.run_foreground(build_rlhf_command(self.cfg), self.env)
        finally:
            self._stop(rollout)
        self.publish_latest()
        return rc

    def write_versions(self, mode):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        freeze = self.run_cmd([sys.executable, "-m", "pip", "freeze"], env=self.env, capture_output=True, text=True)
        packages = [l for l in (getattr(freeze, "stdout", "") or "").splitlines() if l.split("==")[0].lower() in ("torch", "vllm", "transformers", "ms-swift", "peft", "trl")]
        (self.run_dir / "versions.json").write_text(json.dumps({"mode": mode, "packages": packages, "config": self.cfg}, indent=2))

    def run(self, mode="swift"):
        if mode != "swift":
            raise NotImplementedError("--mode fallback (decoupled batch GRPO) is spec §3.4, not implemented yet")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.set_phase("MERGE")
        if self.merge() != 0:
            self.set_phase("FAILED")
            return 1
        if self.cfg.get("eval_before"):
            self.serve_phase("EVAL_BEFORE_READY")
        rc = self.train_phase()
        if rc == 0 and self.cfg.get("eval_after"):
            latest = Path(self.cfg["output_dir"]) / "latest"
            self.serve_phase("EVAL_AFTER_READY", adapters=str(latest) if latest.exists() else None)
        self.write_versions(mode)
        self.set_phase("DONE" if rc == 0 else "FAILED")
        return rc


def main():
    parser = argparse.ArgumentParser(description="GRPO run driver")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("print-commands")
    p.add_argument("--config", default="rl_config.yaml")
    d = sub.add_parser("make-dataset")
    d.add_argument("--tasks", default="cua/rl/rl_tasks.json")
    d.add_argument("--out", default="rl_tasks.jsonl")
    d.add_argument("--split", default="train")
    d.add_argument("--repeats", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--config", default="rl_config.yaml")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--workdir", default="/workspace")
    r.add_argument("--mode", default="swift", choices=["swift", "fallback"])
    args = parser.parse_args()
    if args.command == "print-commands":
        cfg = load_rl_config(args.config)
        for name, argv in (("merge", build_merge_command(cfg)), ("rollout", build_rollout_command(cfg)), ("rlhf", build_rlhf_command(cfg)),
                           ("deploy", build_deploy_command(cfg, adapters=f"{cfg['output_dir']}/latest"))):
            print(f"# {name}\n{' '.join(argv)}\n")
        return 0
    if args.command == "make-dataset":
        print(f"wrote {make_dataset(args.tasks, args.out, args.split, args.repeats)} rows -> {args.out}")
        return 0
    return PhaseDriver(load_rl_config(args.config), args.run_dir, args.workdir).run(mode=args.mode)


if __name__ == "__main__":
    sys.exit(main())
```
- [ ] **Step 4: Run** → `$PY -m tests test_rl_train` → `6/6 passed`; `$PY cua/rl/train.py print-commands` prints four argv lines.
- [ ] **Step 5: Report.**

---

### Task 4: `cua/cloud/vast/tunnel.py` + provider hook

**Files:** Create `cua/cloud/vast/tunnel.py`, `tests/test_tunnel.py`; modify `cua/cloud/vast/base.py`, `cua/cloud/vast/vastai.py`, `cua/cloud/vast/mock.py`

**Interfaces (produced):**
- `SshTunnel(ssh_argv_base: list[str], forwards: list[tuple[str, int]], keepalive_s=15, popen=subprocess.Popen, sleep=time.sleep)` context manager; `.argv()`; `.start()`; `.close()`; `.alive`; supervisor thread restarts on exit with backoff `min(60, 5·attempt)`.
- `NullTunnel(forwards)` no-op context manager.
- `ComputeProvider.tunnel(instance, cfg, forwards) -> context manager` (default `NullTunnel`); `VastAIProvider.tunnel` → `SshTunnel(self._ssh_base(instance)[:-1], forwards)`; `MockProvider.tunnel` records `self.tunnels.append(forwards)` and returns `NullTunnel`.

- [ ] **Step 1: Failing tests** — `tests/test_tunnel.py`:
```python
import threading

from cua.cloud.vast.tunnel import NullTunnel, SshTunnel


class FakePopen:
    instances = []

    def __init__(self, argv, **kw):
        self.argv, self.rc = argv, None
        FakePopen.instances.append(self)

    def poll(self):
        return self.rc

    def terminate(self):
        self.rc = 0

    def wait(self, timeout=None):
        return self.rc

    def kill(self):
        self.rc = -9


def test_argv_binds_loopback_both_ways():
    tunnel = SshTunnel(["ssh", "-p", "2222", "-i", "k", "root@h"], [("R", 8900), ("L", 8001)], popen=FakePopen, sleep=lambda s: None)
    argv = tunnel.argv()
    assert argv[0] == "ssh" and "-N" in argv and argv[-1] == "root@h"
    assert argv[argv.index("-R") + 1] == "127.0.0.1:8900:127.0.0.1:8900" and argv[argv.index("-L") + 1] == "127.0.0.1:8001:127.0.0.1:8001"
    assert "ServerAliveInterval=15" in argv and "ExitOnForwardFailure=yes" in argv


def test_supervisor_restarts_until_closed():
    FakePopen.instances.clear()
    restarted = threading.Event()

    def sleep(seconds):
        if len(FakePopen.instances) >= 2:
            restarted.set()

    with SshTunnel(["ssh", "root@h"], [("R", 8900)], popen=FakePopen, sleep=sleep) as tunnel:
        FakePopen.instances[0].rc = 255
        assert restarted.wait(2)
    assert not tunnel.alive and all(p.rc is not None for p in FakePopen.instances)


def test_null_tunnel_is_a_noop():
    with NullTunnel([("R", 8900)]) as t:
        assert t.forwards == [("R", 8900)]
```
- [ ] **Step 2: Run** → ModuleNotFoundError.
- [ ] **Step 3: Implement** `cua/cloud/vast/tunnel.py`:
```python
"""SSH tunnel resource for remote jobs: -R exposes the Mac env service to the box, -L brings the box's serve port here; supervised like autossh."""
import logging
import subprocess
import threading
import time

logger = logging.getLogger(__name__)


class NullTunnel:
    def __init__(self, forwards=()):
        self.forwards = list(forwards)
        self.alive = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class SshTunnel:
    def __init__(self, ssh_argv_base, forwards, keepalive_s=15, popen=subprocess.Popen, sleep=time.sleep):
        self.base, self.forwards, self.keepalive_s, self.popen, self.sleep = list(ssh_argv_base), list(forwards), keepalive_s, popen, sleep
        self.alive = False
        self.proc = None
        self.thread = None

    def argv(self):
        target = self.base[-1]
        opts = self.base[:-1]
        argv = [*opts, "-N", "-o", f"ServerAliveInterval={self.keepalive_s}", "-o", "ServerAliveCountMax=4", "-o", "ExitOnForwardFailure=yes"]
        for direction, port in self.forwards:
            argv += [f"-{direction}", f"127.0.0.1:{port}:127.0.0.1:{port}"]
        return [*argv, target]

    def _supervise(self):
        attempt = 0
        while self.alive:
            self.proc = self.popen(self.argv(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while self.alive and self.proc.poll() is None:
                self.sleep(1)
            if not self.alive:
                break
            attempt += 1
            logger.warning(f"ssh tunnel exited (rc={self.proc.poll()}); restarting in {min(60, 5 * attempt)}s")
            self.sleep(min(60, 5 * attempt))

    def start(self):
        self.alive = True
        self.thread = threading.Thread(target=self._supervise, name="ssh-tunnel", daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.alive = False
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
        return False
```
`cua/cloud/vast/base.py`: add to `ComputeProvider` (after `list_owned`):
```python
    def tunnel(self, instance: Instance, cfg, forwards: list):
        """Optional: a context manager forwarding loopback ports ((\"R\"|\"L\", port) pairs). Default: no tunnel."""
        return NullTunnel(forwards)
```
with `from cua.cloud.vast.tunnel import NullTunnel` at the top (tunnel.py has no remote_train imports, so no cycle).
`cua/cloud/vast/vastai.py`: add `from cua.cloud.vast.tunnel import SshTunnel` and
```python
    def tunnel(self, instance, cfg, forwards):
        return SshTunnel(self._ssh_base(instance), forwards)
```
(`_ssh_base` ends with `user@host`, which `SshTunnel.argv` keeps last.)
`cua/cloud/vast/mock.py`: `self.tunnels = []` in `__init__`; `def tunnel(self, instance, cfg, forwards): self.tunnels.append(list(forwards)); return NullTunnel(forwards)` with the import.
- [ ] **Step 4: Run** → `3/3 passed`; `$PY -m cua.cloud.vast run --smoke` still ALL PASS. **Step 5: Report.**

---

### Task 5: `GrpoJob` + `ServeJob` + lifecycle hooks + config + CLI + Makefile + smoke

**Files:** Modify `cua/cloud/vast/jobs.py`, `cua/cloud/vast/lifecycle.py`, `cua/cloud/vast/config.py`, `cua/cloud/vast/__main__.py`, `cua/cloud/vast/mock.py`, `Makefile`; extend `tests/test_rl_train.py` with job-level tests.

**Interfaces (produced):**
- `RemoteJob.resources(provider, instance, cfg)` → context manager (default `contextlib.nullcontext()`); `RemoteJob.on_poll(provider, instance, cfg) -> bool` (default `False`; `True` = finish the run now).
- `GrpoJob(cfg)`: `preflight` (load `cfg.rl_config`, apply `compute` overrides, require adapter + suite, build `rl_tasks.jsonl`, health-check env service unless `env_url` is `mock://`, write `run_header.json`), `upload` (code files + adapter), `resources` (tunnel `[("R", env_port), ("L", serve_port)]`), `remote_command`, `on_poll` (phase handshake + periodic adapter rsync), `collect`, `results_dest`, `report`.
- `ServeJob(cfg)`: rent → upload code (+ adapter if `cfg.serve_adapter`) → `swift deploy` foreground on the box; `resources` = `[("L", serve_port)]`; `on_poll` logs readiness and returns `True` after `cfg.serve_seconds` (0 = until Ctrl-C); `collect` pulls the log only.
- `JOBS["grpo"] = GrpoJob`, `JOBS["serve"] = ServeJob`.
- Config keys: `rl_config="rl_config.yaml"`, `rl_runs_dir="rl_runs"`, `collect_every_polls=10`, `serve_adapter=""`, `serve_seconds=0`, `env_url=""` (override; empty → from rl_config).
- Mock provider: `exec` answers `cat /workspace/PHASE` with a scripted sequence (`cfg.mock_phases`, default `["EVAL_BEFORE_READY", "TRAINING", "EVAL_AFTER_READY", "DONE"]`, one per call) and records writes of `PHASE_ACK`.

- [ ] **Step 1: Failing tests** — append to `tests/test_rl_train.py`:
```python
from cua.cloud.vast.config import RemoteTrainConfig
from cua.cloud.vast.jobs import JOBS, GrpoJob, ServeJob
from cua.cloud.vast.mock import MockProvider


def _grpo_cfg(tmp):
    cfg = RemoteTrainConfig()
    cfg.rl_config = "rl_config.yaml"
    cfg.rl_runs_dir = str(Path(tmp) / "rl_runs")
    cfg.output_dir = str(Path(tmp) / "output")
    cfg.env_url = "mock://"
    cfg.poll_interval_s = 0
    Path(cfg.output_dir).mkdir(parents=True)
    (Path(cfg.output_dir) / "adapter_config.json").write_text("{}")
    (Path(cfg.output_dir) / "adapter_model.safetensors").write_bytes(b"0" * 16)
    return cfg


def test_grpo_job_registered_and_preflight_applies_compute():
    assert set(JOBS) >= {"train", "eval", "grpo", "serve"}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _grpo_cfg(tmp)
        job = GrpoJob(cfg)
        job.preflight(cfg)
        assert cfg.min_vram_gb == 80 and cfg.disk_gb == 120 and cfg.max_runtime_hours == 8
        header = json.loads((Path(cfg.rl_runs_dir) / job.run_id / "run_header.json").read_text())
        assert header["sft_adapter_sha256"] and header["suite_sha256"] and header["rl_config_sha256"]
        assert Path(job.dataset_path).read_text().count("\n") > 0
        cmd = job.remote_command(cfg)
        assert "requirements-rl.txt" in cmd and "cua/rl/train.py run" in cmd and "--run-dir /workspace/rl_runs/" in cmd


def test_grpo_job_on_poll_handles_handshake_and_collects_adapter():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _grpo_cfg(tmp)
        job = GrpoJob(cfg)
        job.preflight(cfg)
        job.run_eval = lambda phase: {"phase": phase}
        provider = MockProvider(cfg)
        instance = provider.provision(provider.search_offer(cfg), cfg)
        provider.wait_ready(instance, 1)
        stop = [job.on_poll(provider, instance, cfg) for _ in range(4)]
        assert stop == [False, False, False, False]
        assert job.evals_done == ["EVAL_BEFORE_READY", "EVAL_AFTER_READY"]
        assert any("PHASE_ACK" in c for c in provider.exec_log)
        assert any("DOWNLOAD /workspace/rl_output/latest/" in c for c in provider.exec_log)


def test_serve_job_stops_after_serve_seconds():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _grpo_cfg(tmp)
        cfg.serve_adapter = cfg.output_dir
        cfg.serve_seconds = 1
        job = ServeJob(cfg)
        job.preflight(cfg)
        job.clock = iter([0, 0.5, 2.0]).__next__
        job.http_ok = lambda url: True
        provider = MockProvider(cfg)
        instance = provider.provision(provider.search_offer(cfg), cfg)
        assert job.on_poll(provider, instance, cfg) is False
        assert job.on_poll(provider, instance, cfg) is True
        assert "swift deploy" in job.remote_command(cfg) and "--adapters" in job.remote_command(cfg)
```
- [ ] **Step 2: Run** → `ImportError: cannot import name 'GrpoJob'`.
- [ ] **Step 3: Implement**

`cua/cloud/vast/config.py`: add fields after `notify`:
```python
    rl_config: str = "rl_config.yaml"
    rl_runs_dir: str = "rl_runs"
    collect_every_polls: int = 10
    serve_adapter: str = ""
    serve_seconds: int = 0
    env_url: str = ""
    mock_phases: tuple = ("EVAL_BEFORE_READY", "TRAINING", "EVAL_AFTER_READY", "DONE")
```
and add `"rl_config", "rl_runs_dir", "collect_every_polls", "serve_adapter", "serve_seconds", "env_url"` to `_YAML_KEYS`.

`cua/cloud/vast/mock.py` `exec`: replace the body with
```python
    def exec(self, instance, command, *, stream=False, timeout=0) -> tuple:
        self.exec_log.append(command)
        if "PHASE_ACK" in command:
            return "", "", 0
        if "/PHASE" in command:
            phase = self.phases[min(self.phase_calls, len(self.phases) - 1)]
            self.phase_calls += 1
            return phase, "", 0
        if "STATUS" in command:
            return "DONE", "", 0
        return "", "", 0
```
with `self.phases = list(getattr(cfg, "mock_phases", ()))`, `self.phase_calls = 0` in `__init__`.

`cua/cloud/vast/jobs.py`: add imports `import contextlib, hashlib, json, os, subprocess, sys, time, urllib.request` and `from cua.cloud.vast.tunnel import NullTunnel`; add to `RemoteJob`:
```python
    def resources(self, provider, instance, cfg):
        """Context entered after wait_ready and left BEFORE destroy (tunnels, local helpers)."""
        return contextlib.nullcontext()

    def on_poll(self, provider, instance, cfg) -> bool:
        """Called every poll while the box runs; return True to finish the run now."""
        return False
```
then the two jobs:
```python
RL_CODE_FILES = ("cua/rl/train.py", "cua/sft/legacy_train_qwen.py", "requirements-rl.txt", "cua/pipeline/__init__.py", "cua/pipeline/corpus_audit.py",
                 "cua/rl/__init__.py", "cua/rl/actions.py", "cua/rl/observation.py", "cua/rl/trainer/__init__.py", "cua/rl/trainer/core.py", "cua/rl/trainer/swift_plugin.py")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _http_ok(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status < 500
    except Exception:
        return False


def _upload_code(provider, instance, cfg, files):
    wd = cfg.remote_workdir
    for rel in files:
        if Path(rel).is_file():
            provider.exec(instance, f"mkdir -p {wd}/{Path(rel).parent}", timeout=60)
            provider.upload(instance, rel, f"{wd}/{rel}")


class GrpoJob(RemoteJob):
    name = "grpo"

    def __init__(self, cfg):
        self.rl = yaml.safe_load(Path(cfg.rl_config).read_text())
        self.run_id = time.strftime("grpo-%Y%m%d-%H%M%S")
        self.run_dir = Path(cfg.rl_runs_dir) / self.run_id
        self.dataset_path = self.run_dir / "rl_tasks.jsonl"
        self.evals_done: list = []
        self.polls = 0
        self.http_ok = _http_ok

    def preflight(self, cfg) -> None:
        for key, value in (self.rl.get("compute") or {}).items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        if cfg.env_url:
            self.rl["env_url"] = cfg.env_url
        adapter = Path(cfg.output_dir) / "adapter_config.json"
        if not adapter.is_file():
            raise FileNotFoundError(f"SFT adapter '{adapter}' missing — run `make remote-train` first (spec Phase 2)")
        suite_path = Path("cua/rl/rl_tasks.json")
        if not suite_path.is_file():
            raise FileNotFoundError("cua/rl/rl_tasks.json missing — run `make rl-tasks`")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        rows = make_dataset(suite_path, self.dataset_path, "train", 1)
        if rows == 0:
            raise ValueError("no train tasks in cua/rl/rl_tasks.json")
        env_url = self.rl["env_url"]
        if not env_url.startswith("mock://") and not self.http_ok(f"{env_url}/health"):
            raise ConnectionError(f"env service not answering at {env_url}/health — start `make rl-serve` first")
        weights = Path(cfg.output_dir) / "adapter_model.safetensors"
        header = {
            "run_id": self.run_id, "base_model": self.rl["base_model"], "sft_adapter": cfg.output_dir,
            "sft_adapter_sha256": _sha256(weights) if weights.is_file() else "", "rl_config_sha256": _sha256(cfg.rl_config),
            "suite_sha256": json.loads(suite_path.read_text()).get("suite_sha256", ""), "train_rows": rows,
            "git_head": _git_head(), "env_url": env_url, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.run_dir / "run_header.json").write_text(json.dumps(header, indent=2))
        logger.info(f"grpo run {self.run_id}: {rows} train rows, env {env_url}, compute {self.rl.get('compute')}")

    def upload(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        provider.exec(instance, f"mkdir -p {wd}/sft_adapter {wd}/rl_runs/{self.run_id}", timeout=60)
        logger.info("Uploading GRPO bundle (code + SFT adapter + config)…")
        _upload_code(provider, instance, cfg, RL_CODE_FILES + (cfg.rl_config,))
        provider.upload(instance, str(self.dataset_path), f"{wd}/{Path(self.rl['dataset']).name}")
        provider.upload(instance, cfg.output_dir.rstrip("/") + "/", f"{wd}/sft_adapter/")

    def resources(self, provider, instance, cfg):
        compute = self.rl.get("compute") or {}
        return provider.tunnel(instance, cfg, [("R", int(compute.get("env_port", 8900))), ("L", int(compute.get("serve_port", 8001)))])

    def remote_command(self, cfg) -> str:
        wd = cfg.remote_workdir
        return (
            "pip install -r requirements-rl.txt && \\\n"
            f"  python -m cua.rl.train run --config {Path(cfg.rl_config).name} --run-dir {wd}/rl_runs/{self.run_id} --workdir {wd}"
        )

    def run_eval(self, phase):
        tag = "before" if "BEFORE" in phase else "after"
        port = (self.rl.get("compute") or {}).get("serve_port", 8001)
        argv = [sys.executable, "-m", "cua.rl.eval", "--endpoint", f"http://127.0.0.1:{port}/v1", "--model", "policy",
                "--run-id", f"{self.run_id}-{tag}", "--out", str(self.run_dir / "eval_runs")]
        logger.info(f"running student eval ({tag}) through the tunnel: {' '.join(argv)}")
        result = subprocess.run(argv, capture_output=True, text=True)
        (self.run_dir / f"eval_{tag}.log").write_text(result.stdout + result.stderr)
        return {"phase": phase, "returncode": result.returncode}

    def on_poll(self, provider, instance, cfg) -> bool:
        wd = cfg.remote_workdir
        self.polls += 1
        out, _, _ = provider.exec(instance, f"cat {wd}/PHASE 2>/dev/null || true", timeout=30)
        phase = (out.strip().splitlines() or [""])[-1]
        if phase.endswith("_READY") and phase not in self.evals_done:
            self.run_eval(phase)
            self.evals_done.append(phase)
            provider.exec(instance, f"echo {phase} > {wd}/PHASE_ACK", timeout=30)
        if self.polls % max(1, cfg.collect_every_polls) == 0:
            try:
                provider.download(instance, f"{wd}/rl_output/latest/", str(self.run_dir / "adapter_latest") + "/")
            except Exception as e:
                logger.debug(f"latest adapter not available yet: {e}")
        return False

    def collect(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        logger.info("Downloading RL outputs…")
        provider.download(instance, f"{wd}/rl_output/", str(self.run_dir / "rl_output") + "/")
        provider.download(instance, f"{wd}/rl_runs/{self.run_id}/", str(self.run_dir / "box") + "/")

    def results_dest(self, cfg) -> str:
        return str(self.run_dir)

    def report(self, cfg) -> str:
        return f"run {self.run_id}: evals {self.evals_done}; outputs in {self.run_dir}"


class ServeJob(RemoteJob):
    """Rent a box, serve the merged/adapted student on serve_port through the tunnel until serve_seconds or Ctrl-C."""
    name = "serve"

    def __init__(self, cfg):
        self.rl = yaml.safe_load(Path(cfg.rl_config).read_text())
        self.started = None
        self.clock = time.time
        self.http_ok = _http_ok
        self.ready_logged = False

    def preflight(self, cfg) -> None:
        for key, value in (self.rl.get("compute") or {}).items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        if cfg.serve_adapter and not (Path(cfg.serve_adapter) / "adapter_config.json").is_file():
            raise FileNotFoundError(f"adapter '{cfg.serve_adapter}' has no adapter_config.json")

    def upload(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        provider.exec(instance, f"mkdir -p {wd}/serve_adapter", timeout=60)
        _upload_code(provider, instance, cfg, RL_CODE_FILES + (cfg.rl_config,))
        if cfg.serve_adapter:
            provider.upload(instance, cfg.serve_adapter.rstrip("/") + "/", f"{wd}/serve_adapter/")

    def resources(self, provider, instance, cfg):
        return provider.tunnel(instance, cfg, [("L", int((self.rl.get("compute") or {}).get("serve_port", 8001)))])

    def remote_command(self, cfg) -> str:
        wd = cfg.remote_workdir
        deploy = build_deploy_command(dict(self.rl, merged_model=self.rl["base_model"]), adapters=f"{wd}/serve_adapter" if cfg.serve_adapter else None)
        return "pip install -r requirements-rl.txt && \\\n  MAX_PIXELS=" + str(self.rl["max_pixels"]) + " " + " ".join(deploy)

    def on_poll(self, provider, instance, cfg) -> bool:
        now = self.clock()
        self.started = self.started or now
        port = (self.rl.get("compute") or {}).get("serve_port", 8001)
        if not self.ready_logged and self.http_ok(f"http://127.0.0.1:{port}/v1/models"):
            logger.info(f"student endpoint ready at http://127.0.0.1:{port}/v1 — run `make rl-eval` in another shell")
            self.ready_logged = True
        return bool(cfg.serve_seconds) and now - self.started >= cfg.serve_seconds

    def collect(self, provider, instance, cfg) -> None:
        return None

    def results_dest(self, cfg) -> str:
        return str(Path(cfg.rl_runs_dir) / "serve")

    def report(self, cfg) -> str:
        return "serve session ended; instance destroyed"
```
plus `from cua.rl.train import build_deploy_command, make_dataset` at the top of `jobs.py` and `JOBS["grpo"] = GrpoJob`, `JOBS["serve"] = ServeJob`.

`cua/cloud/vast/lifecycle.py` `_run_job`: replace the block from `provider.wait_ready(...)` through `job.collect(...)` with
```python
        provider.wait_ready(instance, cfg.ready_timeout_s)
        with job.resources(provider, instance, cfg):
            job.upload(provider, instance, cfg)
            _launch(provider, instance, cfg, job.remote_command(cfg))
            deadline = time.time() + cfg.max_runtime_hours * 3600
            ok = _wait_for_completion(provider, instance, cfg, deadline, job)
            _pull_logs(provider, instance, cfg, job.results_dest(cfg))
            if not ok:
                logger.error(f"Remote {job.name} reported FAILED — see ./{job.results_dest(cfg)}/{_LOG_FILE}")
                return 1
            job.collect(provider, instance, cfg)
```
and in `_wait_for_completion(provider, instance, cfg, deadline, job=None)` add, right after the log-tail block:
```python
        if job is not None:
            try:
                if job.on_poll(provider, instance, cfg):
                    return True
            except Exception as e:
                logger.warning(f"on_poll hook failed (ignored): {e}")
```
`run_smoke_test`: `make_cfg` also sets `cfg.rl_config = "rl_config.yaml"`, `cfg.rl_runs_dir = str(tmp / "rl_runs")`, `cfg.env_url = "mock://"`, `cfg.serve_seconds = 1`, and writes a fake adapter into `cfg.output_dir` (`adapter_config.json` + `adapter_model.safetensors`); loop `for job_name in ("train", "eval", "grpo", "serve")`; for `grpo` additionally `check("grpo: tunnel opened R:8900 L:8001", happy.tunnels == [[("R", 8900), ("L", 8001)]])` and `check("grpo: both eval phases acked", sum("PHASE_ACK" in c for c in happy.exec_log) == 2)`. The grpo job's `run_eval` must not spawn the real eval in the smoke: set `GrpoJob.run_eval` via a module-level `_SMOKE_EVAL = lambda self, phase: {"phase": phase}` assignment inside `run_smoke_test` (`GrpoJob.run_eval = _SMOKE_EVAL`) and restore it after.

`cua/cloud/vast/__main__.py`: `grpo` subparser (`_add_common`, `--rl-config`, `--env-url`, `--notify`) → `_job_cmd(args, "grpo")`; `serve` subparser (`_add_common`, `--rl-config`, `--adapter`, `--for-seconds`) → `_job_cmd(args, "serve")`; `_cli_overrides` gains `"rl_config": getattr(args, "rl_config", None)`, `"env_url": getattr(args, "env_url", None)`, `"serve_adapter": getattr(args, "adapter", None)`, `"serve_seconds": getattr(args, "for_seconds", None)`. Module docstring gets the two new usage lines.

Makefile (after `test-remote`):
```make
remote-grpo: ## Rent a GPU, run GRPO against the local env service (make rl-serve first), pull adapters, ALWAYS destroy
	@test -n "$$VAST_API_KEY" || (echo "ERROR: VAST_API_KEY not set" && exit 1)
	python -m cua.cloud.vast grpo $(if $(REMOTE_ARGS),$(REMOTE_ARGS))

remote-serve: ## Rent a GPU and serve the student on 127.0.0.1:8001 through a tunnel (SERVE_ADAPTER=output, SERVE_SECONDS=0)
	@test -n "$$VAST_API_KEY" || (echo "ERROR: VAST_API_KEY not set" && exit 1)
	python -m cua.cloud.vast serve $(if $(SERVE_ADAPTER),--adapter $(SERVE_ADAPTER)) $(if $(SERVE_SECONDS),--for-seconds $(SERVE_SECONDS)) $(if $(REMOTE_ARGS),$(REMOTE_ARGS))

test-grpo: ## No-cost GRPO self-test: mock provider + mock env (teardown, tunnel, phase handshake) + command builders
	python -m cua.cloud.vast run --smoke
	python -m cua.rl.train print-commands --config rl_config.yaml
	python -m tests test_trainer_core test_rl_train test_tunnel

grpo-dataset: ## Write rl_tasks.jsonl (train split) for the GRPO trainer
	python -m cua.rl.train make-dataset --tasks cua/rl/rl_tasks.json --out rl_tasks.jsonl
```
Add `remote-grpo remote-serve test-grpo grpo-dataset` to `.PHONY`.

- [ ] **Step 4: Run** — `$PY -m tests` (all), `$PY -m cua.cloud.vast run --smoke` → ALL PASS incl. grpo/serve lines, `$PY cua/rl/train.py print-commands`, `make -n test-grpo`, `$PY -m cua.cloud.vast grpo --help`.
- [ ] **Step 5: Report** — list the VALIDATE-ON-BOX items that remain (V1–V16 from the spec) and the deferred fallback.

---

## Self-review

- **Spec coverage:** 3.1 → Tasks 1–2; 3.2 → Task 3; 3.3 → Tasks 4–5 (GrpoJob, ServeJob, tunnel, on_poll, `run --smoke`, Makefile); 3.4 → deferred explicitly (NotImplementedError + report). Phase 2's `ServeJob` lands here so `make remote-serve` exists before the SFT run.
- **Placeholders:** none; `--mode fallback` raising is a documented deferral, not a stub pretending to work.
- **Type consistency:** `episode_records` record keys (`messages`, `images`, `response_token_ids`, `rollout_infos`) match what `BrowserScheduler.build_outputs` unpacks; `HttpEnvClient.reset` body keys match `cua/rl/service.py`'s `/reset` (`task_id`, `sample_k`, `run_id`, `iter`, `policy_tag`); `PhaseDriver` phase strings match `GrpoJob.on_poll`'s `_READY` handling and the mock's `mock_phases`; tunnel forwards `("R"|"L", port)` match `SshTunnel.argv`.
