"""Run the GRPO phase driver (cua/rl/train.py run) on a Modal GPU against the Mac env service exposed through a public ingress."""

import os
import pathlib
import tomllib

import modal

from cua.settings import CLOUD, RL

APP_NAME = "cua-grpo"

OUT_MOUNT = "/outputs"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
CODE_ROOT = "/workspace"

app = modal.App(APP_NAME)

outputs = modal.Volume.from_name("cua-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("cua-hf-cache", create_if_missing=True)

_PYPROJECT_FILE = pathlib.Path("pyproject.toml")
_CONSTRAINTS_FILE = pathlib.Path("constraints/rl-modal.txt")


def _rl_requirements():
    """Everything `pip install -e ".[rl]" -c constraints/rl-modal.txt` installs: core deps, the rl extra, the resolved GPU column."""
    project = tomllib.loads(_PYPROJECT_FILE.read_text())["project"]
    pins = [line.strip() for line in _CONSTRAINTS_FILE.read_text().splitlines() if line.strip() and line[0] != "#"]
    return list(project["dependencies"]) + list(project["optional-dependencies"]["rl"]) + pins


# Module-level reads re-run inside the container, where these files are absent and the image is already built.
_RL_REQUIREMENTS = _rl_requirements() if _PYPROJECT_FILE.is_file() and _CONSTRAINTS_FILE.is_file() else []

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(*_RL_REQUIREMENTS, "accelerate>=1.0", "huggingface_hub[hf_transfer]")
    # Baked at deploy time so the container's summary names the GPU actually rented, not the module default.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CUA_GRPO_GPU": CLOUD.grpo_gpu, "PYTHONPATH": CODE_ROOT})
    .add_local_file("rl_config.yaml", f"{CODE_ROOT}/rl_config.yaml")
    .add_local_file("pyproject.toml", f"{CODE_ROOT}/pyproject.toml")
    .add_local_dir("constraints", f"{CODE_ROOT}/constraints")
    # Reached through PYTHONPATH, not an editable install: add_local_* must close an image, so no pip step can follow it.
    .add_local_dir("cua", f"{CODE_ROOT}/cua", ignore=["**/__pycache__", "**/*.pyc"])
)


@app.function(
    image=image,
    gpu=CLOUD.grpo_gpu,
    volumes={OUT_MOUNT: outputs, HF_CACHE_MOUNT: hf_cache},
    timeout=CLOUD.grpo_timeout,
)
def grpo(
    run_name: str,
    env_url: str,
    env_token: str = "",
    sft_adapter: str = "sft-v1/adapter",
    max_steps: int = 0,
    num_generations: int = 0,
    max_turns: int = 0,
    smoke: bool = False,
    tasks_file: str = "",
    task_ids_file: str = "",
    no_reward_smoke: bool = False,
):
    import json
    import shutil
    import subprocess
    import sys
    import threading
    from pathlib import Path

    import yaml

    from cua.rl.train import derive_run_config, load_rl_config, load_task_ids, make_dataset

    if no_reward_smoke:
        # env-read: inherited by cua.rl.train's subprocesses, so the mock env's verdictless closes kill the run instead of training zeros
        os.environ["RL_MOCK_NO_REWARD"] = "1"

    adapter_src = Path(OUT_MOUNT) / sft_adapter
    if not (adapter_src / "adapter_config.json").is_file():
        raise RuntimeError(
            f"{adapter_src} has no adapter_config.json — `modal volume ls cua-outputs` to see what is there"
        )
    run_dir = Path(OUT_MOUNT) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(CODE_ROOT)
    sft_adapter_local = workdir / "sft_adapter"
    if sft_adapter_local.exists():
        shutil.rmtree(sft_adapter_local)
    shutil.copytree(adapter_src, sft_adapter_local)
    base_cfg = load_rl_config(str(workdir / "rl_config.yaml"))
    suite_path = workdir / (tasks_file or "cua/rl/rl_tasks.json")
    task_ids = load_task_ids(workdir / task_ids_file) if task_ids_file else None
    rows = make_dataset(
        suite_path, workdir / "rl_tasks.jsonl", "train", 1, base_cfg.get("train_max_expected_actions"), task_ids
    )
    if rows == 0:
        raise RuntimeError(f"{suite_path.name} has no train tasks — run `make rl-tasks` before deploying")
    print(f"{rows} train rows from {suite_path.name}", flush=True)

    overrides = {
        "env_url": env_url,
        "env_token": env_token or None,
        "sft_adapter_remote": str(sft_adapter_local),
        # Merged weights live on the HF-cache volume so the 3-minute merge is paid once per SFT adapter.
        "merged_model": f"{HF_CACHE_MOUNT}/merged/{sft_adapter.replace('/', '__')}",
        "output_dir": str(run_dir / "rl_output"),
        "dataset": str(workdir / "rl_tasks.jsonl"),
        "eval_before": False,
        "eval_after": False,
        "max_steps": max_steps or None,
        "num_generations": num_generations or None,
        "max_turns": max_turns or None,
    }
    if smoke:
        overrides.update(
            {"max_steps": max_steps or 1, "num_generations": num_generations or 2, "max_turns": max_turns or 3}
        )
    cfg = derive_run_config(base_cfg, overrides)
    run_config = run_dir / "rl_config.yaml"
    run_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    stop = threading.Event()

    def commit_periodically():
        while not stop.wait(CLOUD.grpo_commit_every_s):
            outputs.commit()

    committer = threading.Thread(target=commit_periodically, daemon=True)
    committer.start()
    cmd = [
        sys.executable,
        "-m",
        "cua.rl.train",
        "run",
        "--config",
        str(run_config),
        "--run-dir",
        str(run_dir),
        "--workdir",
        str(workdir),
    ]
    print("+", " ".join(cmd), flush=True)
    try:
        result = subprocess.run(cmd, cwd=CODE_ROOT)
    finally:
        stop.set()
        hf_cache.commit()
        outputs.commit()

    latest = run_dir / "rl_output" / "latest"
    adapter = run_dir / "adapter"
    if latest.is_dir():
        if adapter.exists():
            shutil.rmtree(adapter)
        shutil.copytree(latest, adapter)
    summary = {
        "run": run_name,
        "returncode": result.returncode,
        "adapter": str(adapter) if adapter.exists() else "",
        "env_url": env_url,
        "gpu": CLOUD.grpo_gpu,
        "smoke": smoke,
        "max_steps": cfg["max_steps"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    outputs.commit()
    print(json.dumps(summary, indent=2), flush=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"cua/rl/train.py run exited {result.returncode} — see {run_dir}/versions.json and the Modal logs"
        )
    return summary


@app.local_entrypoint()
def main(
    run_name: str = "grpo-v1",
    env_url: str = "",
    env_token: str = "",
    sft_adapter: str = "sft-v1/adapter",
    max_steps: int = 0,
    num_generations: int = 0,
    max_turns: int = 0,
    smoke: bool = False,
    tasks_file: str = "",
    task_ids_file: str = "",
    no_reward_smoke: bool = False,
):
    if smoke and not env_url:
        env_url = "mock://"
    if not env_url:
        raise SystemExit("--env-url is required: the public URL of `make rl-expose` (or mock:// with --smoke)")
    if not smoke and env_url.startswith("mock://"):
        raise SystemExit("mock:// env is for --smoke only")
    if no_reward_smoke and not smoke:
        raise SystemExit("--no-reward-smoke needs --smoke")
    result = grpo.remote(
        run_name=run_name,
        env_url=env_url,
        env_token=env_token or RL.env_token,
        sft_adapter=sft_adapter,
        max_steps=max_steps,
        num_generations=num_generations,
        max_turns=max_turns,
        smoke=smoke,
        tasks_file=tasks_file,
        task_ids_file=task_ids_file,
        no_reward_smoke=no_reward_smoke,
    )
    print(f"\nrun {result['run']} finished rc={result['returncode']}; adapter: {result['adapter'] or '(none)'}")
    if result["adapter"]:
        print(f"serve it with:  CUA_ADAPTER_NAME={result['run']}/adapter modal deploy cua/cloud/modal/serve.py")
