"""Run the GRPO phase driver (rl_train.py run) on a Modal GPU against the Mac env service exposed through a public ingress."""
import os
import pathlib

import modal

APP_NAME = "cua-grpo"
# Two GPUs: ms-swift server mode syncs weights over NCCL, which cannot host the trainer and the rollout engine on one device.
GPU = os.environ.get("CUA_GRPO_GPU", "A100-80GB:2")
GRPO_TIMEOUT = int(os.environ.get("CUA_GRPO_TIMEOUT", str(9 * 3600)))
COMMIT_EVERY_S = int(os.environ.get("CUA_GRPO_COMMIT_EVERY_S", "600"))

OUT_MOUNT = "/outputs"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
CODE_ROOT = "/workspace"

app = modal.App(APP_NAME)

outputs = modal.Volume.from_name("cua-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("cua-hf-cache", create_if_missing=True)

_RL_REQUIREMENTS_FILE = pathlib.Path("requirements-rl.txt")
# Module-level reads re-run inside the container, where this file is absent and the image is already built.
_RL_REQUIREMENTS = ([line.strip() for line in _RL_REQUIREMENTS_FILE.read_text().splitlines() if line.strip() and not line.startswith("#")]
                    if _RL_REQUIREMENTS_FILE.is_file() else [])

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(*_RL_REQUIREMENTS, "accelerate>=1.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file("rl_train.py", f"{CODE_ROOT}/rl_train.py")
    .add_local_file("train_qwen.py", f"{CODE_ROOT}/train_qwen.py")
    .add_local_file("rl_config.yaml", f"{CODE_ROOT}/rl_config.yaml")
    .add_local_dir("repo_pipeline", f"{CODE_ROOT}/repo_pipeline", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir("rl", f"{CODE_ROOT}/rl", ignore=["**/__pycache__", "**/*.pyc", "tests/**"])
)


@app.function(image=image, gpu=GPU, volumes={OUT_MOUNT: outputs, HF_CACHE_MOUNT: hf_cache}, timeout=GRPO_TIMEOUT)
def grpo(run_name: str, env_url: str, env_token: str = "", sft_adapter: str = "sft-v1/adapter", max_steps: int = 0,
         num_generations: int = 0, max_turns: int = 0, smoke: bool = False):
    import json
    import shutil
    import subprocess
    import sys
    import threading
    from pathlib import Path

    import yaml

    sys.path.insert(0, CODE_ROOT)
    from rl_train import derive_run_config, load_rl_config, make_dataset

    adapter_src = Path(OUT_MOUNT) / sft_adapter
    if not (adapter_src / "adapter_config.json").is_file():
        raise RuntimeError(f"{adapter_src} has no adapter_config.json — `modal volume ls cua-outputs` to see what is there")
    run_dir = Path(OUT_MOUNT) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(CODE_ROOT)
    sft_adapter_local = workdir / "sft_adapter"
    if sft_adapter_local.exists():
        shutil.rmtree(sft_adapter_local)
    shutil.copytree(adapter_src, sft_adapter_local)
    base_cfg = load_rl_config(str(workdir / "rl_config.yaml"))
    rows = make_dataset(workdir / "rl" / "rl_tasks.json", workdir / "rl_tasks.jsonl", "train", 1, base_cfg.get("train_max_expected_actions"))
    if rows == 0:
        raise RuntimeError("rl/rl_tasks.json has no train tasks — run `make rl-tasks` before deploying")
    print(f"{rows} train rows from rl/rl_tasks.json", flush=True)

    overrides = {
        "env_url": env_url, "env_token": env_token or None, "sft_adapter_remote": str(sft_adapter_local),
        # Merged weights live on the HF-cache volume so the 3-minute merge is paid once per SFT adapter.
        "merged_model": f"{HF_CACHE_MOUNT}/merged/{sft_adapter.replace('/', '__')}",
        "output_dir": str(run_dir / "rl_output"), "dataset": str(workdir / "rl_tasks.jsonl"),
        "eval_before": False, "eval_after": False,
        "max_steps": max_steps or None, "num_generations": num_generations or None, "max_turns": max_turns or None,
    }
    if smoke:
        overrides.update({"max_steps": max_steps or 1, "num_generations": num_generations or 2, "max_turns": max_turns or 3})
    cfg = derive_run_config(base_cfg, overrides)
    run_config = run_dir / "rl_config.yaml"
    run_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    stop = threading.Event()

    def commit_periodically():
        while not stop.wait(COMMIT_EVERY_S):
            outputs.commit()

    committer = threading.Thread(target=commit_periodically, daemon=True)
    committer.start()
    cmd = [sys.executable, "rl_train.py", "run", "--config", str(run_config), "--run-dir", str(run_dir), "--workdir", str(workdir)]
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
    summary = {"run": run_name, "returncode": result.returncode, "adapter": str(adapter) if adapter.exists() else "",
               "env_url": env_url, "gpu": GPU, "smoke": smoke, "max_steps": cfg["max_steps"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    outputs.commit()
    print(json.dumps(summary, indent=2), flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"rl_train.py run exited {result.returncode} — see {run_dir}/versions.json and the Modal logs")
    return summary


@app.local_entrypoint()
def main(run_name: str = "grpo-v1", env_url: str = "", env_token: str = "", sft_adapter: str = "sft-v1/adapter",
         max_steps: int = 0, num_generations: int = 0, max_turns: int = 0, smoke: bool = False):
    if smoke and not env_url:
        env_url = "mock://"
    if not env_url:
        raise SystemExit("--env-url is required: the public URL of `make rl-expose` (or mock:// with --smoke)")
    if not smoke and env_url.startswith("mock://"):
        raise SystemExit("mock:// env is for --smoke only")
    result = grpo.remote(run_name=run_name, env_url=env_url, env_token=env_token or os.environ.get("RL_ENV_TOKEN", ""),
                         sft_adapter=sft_adapter, max_steps=max_steps, num_generations=num_generations, max_turns=max_turns, smoke=smoke)
    print(f"\nrun {result['run']} finished rc={result['returncode']}; adapter: {result['adapter'] or '(none)'}")
    if result["adapter"]:
        print(f"serve it with:  CUA_ADAPTER_NAME={result['run']}/adapter modal deploy modal_serve.py")
