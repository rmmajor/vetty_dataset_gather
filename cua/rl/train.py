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
    "merged_model": "--model",
    "template": "--template",
    "response_prefix": "--response_prefix",
    "vllm_gpu_memory_utilization": "--vllm_gpu_memory_utilization",
    "vllm_max_model_len": "--vllm_max_model_len",
    "vllm_max_lora_rank": "--vllm_max_lora_rank",
    "vllm_limit_mm_per_prompt": "--vllm_limit_mm_per_prompt",
    "external_plugins": "--external_plugins",
    "multi_turn_scheduler": "--multi_turn_scheduler",
    "gym_env": "--gym_env",
    "max_turns": "--max_turns",
    "rollout_port": "--port",
}
RLHF_FLAG_MAP = {
    "merged_model": "--model",
    "template": "--template",
    "response_prefix": "--response_prefix",
    "tuner_type": "--tuner_type",
    "lora_rank": "--lora_rank",
    "lora_alpha": "--lora_alpha",
    "target_modules": "--target_modules",
    "freeze_vit": "--freeze_vit",
    "torch_dtype": "--torch_dtype",
    "rollout_port": "--vllm_server_port",
    "external_plugins": "--external_plugins",
    "reward_funcs": "--reward_funcs",
    "dataset": "--dataset",
    "max_length": "--max_length",
    "max_completion_length": "--max_completion_length",
    "completion_length_limit_scope": "--completion_length_limit_scope",
    "num_generations": "--num_generations",
    "per_device_train_batch_size": "--per_device_train_batch_size",
    "gradient_accumulation_steps": "--gradient_accumulation_steps",
    "steps_per_generation": "--steps_per_generation",
    "learning_rate": "--learning_rate",
    "beta": "--beta",
    "epsilon": "--epsilon",
    "temperature": "--temperature",
    "top_p": "--top_p",
    "max_steps": "--max_steps",
    "save_steps": "--save_steps",
    "save_total_limit": "--save_total_limit",
    "async_generate": "--async_generate",
    "dynamic_sample": "--dynamic_sample",
    "max_resample_times": "--max_resample_times",
    "epsilon_high": "--epsilon_high",
    "log_completions": "--log_completions",
    "report_to": "--report_to",
    "output_dir": "--output_dir",
}
DEPLOY_FLAG_MAP = {
    "merged_model": "--model",
    "template": "--template",
    "response_prefix": "--response_prefix",
    "vllm_max_lora_rank": "--vllm_max_lora_rank",
    "vllm_limit_mm_per_prompt": "--vllm_limit_mm_per_prompt",
    "max_completion_length": "--max_new_tokens",
}
ROLLOUT_FIXED = [
    "--vllm_use_async_engine",
    "true",
    "--vllm_enable_lora",
    "true",
    "--use_gym_env",
    "true",
    "--host",
    "127.0.0.1",
]
RLHF_FIXED = [
    "--rlhf_type",
    "grpo",
    "--use_vllm",
    "true",
    "--vllm_mode",
    "server",
    "--vllm_server_host",
    "127.0.0.1",
    "--vllm_server_pass_dataset",
    "true",
    "--vllm_enable_lora",
    "true",
    "--use_gym_env",
    "true",
]
DEPLOY_FIXED = ["--infer_backend", "vllm", "--host", "127.0.0.1"]
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
    # --exist_ok: a merge killed mid-write leaves a dir without config.json, which our own guard lets
    # through and swift export then refuses — a failed attempt would poison every retry.
    return [
        "swift",
        "export",
        "--model",
        cfg["base_model"],
        "--adapters",
        cfg["sft_adapter_remote"],
        "--merge_lora",
        "true",
        "--exist_ok",
        "true",
        "--output_dir",
        cfg["merged_model"],
    ]


def build_rollout_command(cfg):
    return ["swift", "rollout", *_flags(cfg, ROLLOUT_FLAG_MAP), *ROLLOUT_FIXED]


def build_rlhf_command(cfg):
    argv = ["swift", "rlhf", *RLHF_FIXED, *_flags(cfg, RLHF_FLAG_MAP)]
    if cfg.get("reward_weights"):
        argv += ["--reward_weights", *str(cfg["reward_weights"]).split()]
    return argv


def build_deploy_command(cfg, adapters=None):
    argv = [
        "swift",
        "deploy",
        *_flags(cfg, DEPLOY_FLAG_MAP),
        *DEPLOY_FIXED,
        "--port",
        str(cfg["compute"]["serve_port"]),
    ]
    if adapters:
        # A dict names each arm for a multi-adapter endpoint; a string stays the bare path the RL lane passes.
        argv += [
            "--adapters",
            *([f"{n}={p}" for n, p in adapters.items()] if isinstance(adapters, dict) else [adapters]),
        ]
    return argv


def build_vllm_serve_command(cfg, adapters=None):
    """Plain vLLM OpenAI server for the vllm/vllm-openai image: base model as `cua-student`, the adapter as `cua-student-lora`."""
    model = cfg["merged_model"] if cfg.get("serve_merged_base") else cfg["base_model"]
    argv = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        "cua-student",
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg["compute"]["serve_port"]),
        "--max-model-len",
        str(cfg.get("vllm_max_model_len", 8192)),
        "--gpu-memory-utilization",
        str(cfg.get("vllm_gpu_memory_utilization", 0.9)),
        "--limit-mm-per-prompt",
        str(cfg.get("vllm_limit_mm_per_prompt", '{"image": 1}')),
    ]
    if adapters:
        # A dict serves several checkpoints from one rental, each under its own name; a string keeps the canonical single-adapter name.
        named = adapters if isinstance(adapters, dict) else {"cua-student-lora": adapters}
        argv += [
            "--enable-lora",
            "--lora-modules",
            *[f"{name}={path}" for name, path in named.items()],
            "--max-lora-rank",
            str(cfg.get("vllm_max_lora_rank", 64)),
        ]
    return argv


def build_env(cfg, run_dir, workdir=None):
    env = {
        **os.environ,
        "MAX_PIXELS": str(cfg["max_pixels"]),
        "RL_ENV_URL": cfg["env_url"],
        "RL_RUN_DIR": str(run_dir),
        "RL_RUN_ID": Path(run_dir).name,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }
    if cfg.get("env_token"):
        env["RL_ENV_TOKEN"] = str(cfg["env_token"])
    if workdir:
        # swift loads --external_plugins as a top-level module, so the repo root must be importable for `rl.*` / cua.sft.legacy_train_qwen.
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(workdir), os.environ.get("PYTHONPATH", "")) if p)
    return env


def derive_run_config(cfg, overrides):
    """A run-specific copy of rl_config: `compute`-free overrides win, nested `compute` keys merge."""
    derived = {k: v for k, v in cfg.items()}
    derived["compute"] = dict(cfg.get("compute") or {})
    for key, value in overrides.items():
        if key == "compute":
            derived["compute"].update(value or {})
        elif value is not None:
            derived[key] = value
    return derived


def load_task_ids(path):
    """A frontier_selection.json ({"selected": [...]}) or a plain JSON list of task ids."""
    data = json.loads(Path(path).read_text())
    return list(data["selected"] if isinstance(data, dict) else data)


def make_dataset(suite_path, out_path, split="train", repeats=1, max_expected_actions=None, task_ids=None):
    suite = json.loads(Path(suite_path).read_text())
    tasks = [
        t
        for t in suite["tasks"]
        if (t.get("split") == split or split == "all")
        and (max_expected_actions is None or (t.get("expected_actions") or 0) <= max_expected_actions)
        and (task_ids is None or t["task_id"] in set(task_ids))
    ]
    # The trainer groups GRPO advantages by the row's messages, so the placeholder must be unique per task.
    rows = [
        {
            "messages": [{"role": "user", "content": f"<placeholder:{t['task_id']}>"}],
            "env_config": {"name": "browser_env", "task_id": t["task_id"], "sample_idx": r},
        }
        for r in range(repeats)
        for t in tasks
    ]
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

    def __init__(
        self,
        cfg,
        run_dir,
        workdir="/workspace",
        launcher=subprocess.Popen,
        run=subprocess.run,
        http_ok=_http_ok,
        sleep=time.sleep,
        clock=time.time,
    ):
        self.cfg, self.run_dir, self.workdir = cfg, Path(run_dir), Path(workdir)
        self.launcher, self.run_cmd, self.http_ok, self.sleep, self.clock = launcher, run, http_ok, sleep, clock
        self.env = build_env(cfg, self.run_dir, self.workdir)

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
        proc = self.launcher(build_deploy_command(self.cfg, adapters), env=self.device_env("rollout"))
        try:
            self.wait_healthy(f"http://127.0.0.1:{self.cfg['compute']['serve_port']}/v1/models")
            self.set_phase(phase)
            self.wait_for_ack(phase)
        finally:
            self._stop(proc)

    def latest_checkpoint(self):
        out = Path(self.cfg["output_dir"])
        candidates = [p for p in out.glob("**/checkpoint-*") if p.is_dir()]
        if not candidates:
            return None
        return max(candidates, key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)

    def publish_latest(self):
        checkpoint = self.latest_checkpoint()
        if checkpoint is None:
            return None
        latest = Path(self.cfg["output_dir"]) / "latest"
        if latest.exists():
            shutil.rmtree(latest)
        shutil.copytree(checkpoint, latest)
        return latest

    def device_env(self, role):
        compute = self.cfg.get("compute") or {}
        return dict(
            self.env,
            CUDA_VISIBLE_DEVICES=str(compute.get(f"{role}_devices", self.env.get("CUDA_VISIBLE_DEVICES", "0"))),
        )

    def train_phase(self):
        rollout = self.launcher(build_rollout_command(self.cfg), env=self.device_env("rollout"))
        try:
            if not self.wait_healthy(f"http://127.0.0.1:{self.cfg['rollout_port']}/health/"):
                print("[cua.rl.train] rollout server never became healthy", flush=True)
                return 1
            self.set_phase("TRAINING")
            rc = self.run_foreground(build_rlhf_command(self.cfg), self.device_env("trainer"))
        finally:
            self._stop(rollout)
        self.publish_latest()
        return rc

    def write_versions(self, mode):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        freeze = self.run_cmd([sys.executable, "-m", "pip", "freeze"], env=self.env, capture_output=True, text=True)
        packages = [
            line
            for line in (getattr(freeze, "stdout", "") or "").splitlines()
            if line.split("==")[0].lower() in ("torch", "vllm", "transformers", "ms-swift", "peft", "trl")
        ]
        (self.run_dir / "versions.json").write_text(
            json.dumps({"mode": mode, "packages": packages, "config": self.cfg}, indent=2)
        )

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
    d.add_argument("--max-expected-actions", type=int)
    d.add_argument("--task-ids-file", help="frontier_selection.json or a JSON list; only these tasks enter the dataset")
    r = sub.add_parser("run")
    r.add_argument("--config", default="rl_config.yaml")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--workdir", default="/workspace")
    r.add_argument("--mode", default="swift", choices=["swift", "fallback"])
    args = parser.parse_args()
    if args.command == "print-commands":
        cfg = load_rl_config(args.config)
        for name, argv in (
            ("merge", build_merge_command(cfg)),
            ("rollout", build_rollout_command(cfg)),
            ("rlhf", build_rlhf_command(cfg)),
            ("deploy", build_deploy_command(cfg, adapters=f"{cfg['output_dir']}/latest")),
        ):
            print(f"# {name}\n{' '.join(argv)}\n")
        return 0
    if args.command == "make-dataset":
        task_ids = load_task_ids(args.task_ids_file) if args.task_ids_file else None
        print(
            f"wrote {make_dataset(args.tasks, args.out, args.split, args.repeats, args.max_expected_actions, task_ids)} rows -> {args.out}"
        )
        return 0
    return PhaseDriver(load_rl_config(args.config), args.run_dir, args.workdir).run(mode=args.mode)


if __name__ == "__main__":
    sys.exit(main())
