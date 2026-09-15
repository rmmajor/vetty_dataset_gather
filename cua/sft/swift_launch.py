"""YAML recipe -> `swift` argv, and the SFT launch that runs it."""

import os
import subprocess
import sys
from pathlib import Path

import yaml

SFT_FLAG_MAP = {
    "model": "--model",
    "train_type": "--tuner_type",
    "dataset": "--dataset",
    "val_dataset": "--val_dataset",
    "eval_strategy": "--eval_strategy",
    "packing": "--packing",
    "attn_impl": "--attn_impl",
    "torch_dtype": "--torch_dtype",
    "num_train_epochs": "--num_train_epochs",
    "per_device_train_batch_size": "--per_device_train_batch_size",
    "learning_rate": "--learning_rate",
    "lora_rank": "--lora_rank",
    "lora_alpha": "--lora_alpha",
    "target_modules": "--target_modules",
    "gradient_accumulation_steps": "--gradient_accumulation_steps",
    "max_length": "--max_length",
    "warmup_ratio": "--warmup_ratio",
    "freeze_vit": "--freeze_vit",
    "output_dir": "--output_dir",
}
# Config keys that are environment, not flags (train_model exports them); everything else unmapped is forwarded as --key value.
SFT_ENV_KEYS = ("max_pixels",)
MAX_PIXELS_ENV = "MAX_PIXELS"


def _cli_value(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def yaml_to_cli_args(flag_map, config, passthrough=False):
    """Config -> argv: passthrough forwards every key in config order (unmapped ones verbatim), else only mapped non-None keys."""
    keys = list(config) if passthrough else [key for key in flag_map if config.get(key) is not None]
    return [arg for key in keys for arg in (flag_map.get(key, f"--{key}"), _cli_value(config[key]))]


def build_sft_command(config, dataset=None, val_dataset=None, passthrough_args=None):
    """The swift sft argv for a config: mapped keys via SFT_FLAG_MAP, unmapped keys forwarded verbatim, then passthrough."""
    merged = {key: value for key, value in config.items() if key not in SFT_ENV_KEYS}
    if dataset:
        merged["dataset"] = dataset
    if val_dataset:
        merged["val_dataset"] = val_dataset
    return ["swift", "sft", *yaml_to_cli_args(SFT_FLAG_MAP, merged, passthrough=True), *(passthrough_args or [])]


def _resolve_val_dataset(config):
    """convert writes sft_val.jsonl beside the train file; without picking it up no eval_loss was ever computed."""
    if config.get("val_dataset") or not config.get("dataset"):
        return
    sibling = Path(config["dataset"]).parent / "sft_val.jsonl"
    if sibling.is_file():
        config["val_dataset"] = str(sibling)
        print(f"Validation split found: {sibling}")
        return
    # eval_strategy without a val set makes the trainer raise at startup, so drop it instead of dying.
    dropped = config.pop("eval_strategy", None)
    print(
        f"No validation split at {sibling} — training will report no eval_loss"
        + (f" (eval_strategy={dropped} dropped)" if dropped else "")
    )


def train_model(
    config_path,
    dataset_override=None,
    epochs_override=None,
    lr_override=None,
    batch_size_override=None,
    passthrough_args=None,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if dataset_override:
        config["dataset"] = dataset_override
    if epochs_override:
        config["num_train_epochs"] = epochs_override
    if lr_override:
        config["learning_rate"] = lr_override
    if batch_size_override:
        config["per_device_train_batch_size"] = batch_size_override
    _resolve_val_dataset(config)

    cmd = build_sft_command(config, passthrough_args=passthrough_args)
    env = os.environ.copy()  # env-read: the `swift sft` child inherits this process's environment
    max_pixels = config.get("max_pixels")
    if max_pixels:
        env[MAX_PIXELS_ENV] = str(max_pixels)

    print("Running command:")
    print("  " + " ".join(cmd))
    if max_pixels:
        print(f"  {MAX_PIXELS_ENV}={max_pixels}")

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)
