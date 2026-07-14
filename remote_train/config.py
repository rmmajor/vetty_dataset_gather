"""Remote-training configuration: one dataclass plus a layered loader.

Precedence, lowest to highest: dataclass defaults < remote_train.yaml < env vars < CLI
flags. This mirrors `train_qwen.py:train_model`'s "load YAML, then apply non-None
overrides" pattern. Secrets are never read from the YAML — only from env
(`VAST_API_KEY`, `HF_TOKEN`), like `GITHUB_TOKEN`/`OPENAI_API_KEY` elsewhere in the repo.
"""
import os
from dataclasses import dataclass, field

import yaml

DEFAULT_EXCLUDE_COUNTRIES = ("RU", "CN")


@dataclass
class RemoteTrainConfig:
    provider: str = "vastai"

    gpu_name: str = ""
    min_vram_gb: int = 48
    # bfloat16 (config.yaml torch_dtype) needs Ampere sm_80+; Turing cards (compute_cap 750,
    # e.g. Quadro RTX 8000) are cheap but cannot run the workload. 800 = A100/A6000-class floor.
    min_compute_cap: int = 800
    num_gpus: int = 1
    max_price_per_hour: float = 2.0
    min_reliability: float = 0.90
    exclude_countries: tuple = DEFAULT_EXCLUDE_COUNTRIES

    instance_label: str = "roman.maior"
    image: str = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel"
    disk_gb: int = 80

    max_runtime_hours: float = 48.0
    ready_timeout_s: int = 900
    poll_interval_s: int = 30
    ssh_key_path: str = "~/.ssh/id_rsa"
    remote_workdir: str = "/workspace"

    trajectories_dir: str = "trajectories"
    train_config: str = "config.yaml"
    output_dir: str = "output"
    val_split: float = 0.1

    benchmark: str = "screenspot"
    eval_models: tuple = ()
    eval_limit: int = 0  # 0 = full dataset
    eval_results_dir: str = "eval_results"

    push_hf_repo: str = ""
    notify: bool = False

    api_key: str = field(default="", repr=False)
    hf_token: str = field(default="", repr=False)


_YAML_KEYS = {
    "provider", "gpu_name", "min_vram_gb", "min_compute_cap", "num_gpus", "max_price_per_hour",
    "min_reliability",
    "exclude_countries", "instance_label", "image", "disk_gb", "max_runtime_hours",
    "ready_timeout_s", "poll_interval_s", "ssh_key_path", "remote_workdir", "trajectories_dir",
    "train_config", "output_dir", "val_split", "benchmark", "eval_models", "eval_limit",
    "eval_results_dir", "push_hf_repo", "notify",
}


def load_config(path: str, cli_overrides: dict | None = None) -> RemoteTrainConfig:
    cfg = RemoteTrainConfig()

    if path and os.path.isfile(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            if key in _YAML_KEYS and value is not None:
                setattr(cfg, key, value)

    cfg.api_key = os.environ.get("VAST_API_KEY", cfg.api_key)
    cfg.hf_token = os.environ.get("HF_TOKEN", cfg.hf_token)
    if os.environ.get("REMOTE_TRAIN_EXCLUDE_COUNTRIES"):
        cfg.exclude_countries = tuple(
            c.strip().upper() for c in os.environ["REMOTE_TRAIN_EXCLUDE_COUNTRIES"].split(",") if c.strip()
        )
    if os.environ.get("REMOTE_TRAIN_INSTANCE_LABEL"):
        cfg.instance_label = os.environ["REMOTE_TRAIN_INSTANCE_LABEL"]

    for key, value in (cli_overrides or {}).items():
        if value is not None:
            setattr(cfg, key, value)

    cfg.exclude_countries = tuple(str(c).strip().upper() for c in cfg.exclude_countries)
    return cfg
