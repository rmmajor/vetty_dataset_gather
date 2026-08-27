"""Run the unchanged train_qwen.py convert+train on a Modal GPU, leaving the adapter on a volume the serve app mounts."""
import os

import modal

APP_NAME = "cua-train"
GPU = os.environ.get("CUA_TRAIN_GPU", "A100-80GB")
RUN_NAME = os.environ.get("CUA_RUN_NAME", "sft-v1")
TRAIN_TIMEOUT = int(os.environ.get("CUA_TRAIN_TIMEOUT", str(10 * 3600)))

DATA_MOUNT = "/data"
OUT_MOUNT = "/outputs"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
CODE_ROOT = "/workspace"

app = modal.App(APP_NAME)

trajectories = modal.Volume.from_name("cua-trajectories", create_if_missing=True)
outputs = modal.Volume.from_name("cua-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("cua-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "torch==2.6.0",
        # qwen_vl_utils imports torchvision; 0.21.0 is the release paired with torch 2.6.0.
        "torchvision==0.21.0",
        # 4.x renamed train_type -> tuner_type; train_qwen.py's flag map targets that CLI.
        "ms-swift[llm]>=4.5,<5",
        # datasets<2.19 calls pa.PyExtensionType, which pyarrow>=16 removed; swift dies on import.
        "datasets>=3.0",
        "transformers>=4.57",
        # swift needs this for Qwen-VL image preprocessing; ms-swift[llm] does not pull it.
        "qwen_vl_utils>=0.0.14",
        "peft>=0.14",
        "accelerate>=1.0",
        "pyyaml",
        "Pillow",
        "tqdm",
        "huggingface_hub[hf_transfer]",
    )
    # Xet backend ignores the hub/ cache this volume already holds and re-pulls 18GB; keep the old path.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file("train_qwen.py", f"{CODE_ROOT}/train_qwen.py")
    .add_local_file("config.yaml", f"{CODE_ROOT}/config.yaml")
    .add_local_dir("repo_pipeline", f"{CODE_ROOT}/repo_pipeline", ignore=["**/__pycache__", "**/*.pyc"])
)


@app.function(
    image=image,
    gpu=GPU,
    volumes={DATA_MOUNT: trajectories, OUT_MOUNT: outputs, HF_CACHE_MOUNT: hf_cache},
    timeout=TRAIN_TIMEOUT,
)
def train(val_split: float = 0.1, data_subdir: str = "trajectories", max_examples: int = 0,
          run_name: str = "", extra_convert_args: list = None, extra_train_args: list = None):
    import json
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    traj_dir = Path(DATA_MOUNT) / data_subdir
    if not traj_dir.is_dir() or not any(traj_dir.iterdir()):
        raise RuntimeError(
            f"{traj_dir} is empty — upload the corpus first:\n"
            f"  modal volume put cua-trajectories ./trajectories /{data_subdir}"
        )

    # RUN_NAME is read at import, which re-runs in the container where the env var is absent.
    run_name = run_name or RUN_NAME
    run_dir = Path(OUT_MOUNT) / run_name
    swift_out = run_dir / "swift"
    dataset_dir = run_dir / "training_data"
    run_dir.mkdir(parents=True, exist_ok=True)

    convert = [
        sys.executable, "train_qwen.py", "convert",
        "--input", str(traj_dir),
        "--output", str(dataset_dir),
        "--val-split", str(val_split),
        "--exclude-waste-steps", "--exclude-meandering",
    ] + (extra_convert_args or [])
    print("+", " ".join(convert), flush=True)
    subprocess.run(convert, cwd=CODE_ROOT, check=True)

    dataset_file = dataset_dir / "sft_dataset.jsonl"
    if max_examples:
        kept = dataset_file.read_text().splitlines()[:max_examples]
        dataset_file = dataset_dir / "sft_smoke.jsonl"
        dataset_file.write_text("\n".join(kept) + "\n")
        print(f"smoke mode: training on {len(kept)} examples", flush=True)

    config = yaml.safe_load(Path(CODE_ROOT, "config.yaml").read_text())
    config["output_dir"] = str(swift_out)
    run_config = run_dir / "config.yaml"
    run_config.write_text(yaml.safe_dump(config, sort_keys=False))

    train_cmd = [
        sys.executable, "train_qwen.py", "train",
        "--config", str(run_config),
        "--dataset", str(dataset_file),
    ] + (extra_train_args or [])
    print("+", " ".join(train_cmd), flush=True)
    subprocess.run(train_cmd, cwd=CODE_ROOT, check=True)

    # swift nests the adapter under output_dir/<run>/checkpoint-N; the serve app needs one stable path.
    checkpoints = sorted(swift_out.rglob("adapter_config.json"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"training finished but no adapter_config.json under {swift_out}")
    final = checkpoints[-1].parent
    adapter = run_dir / "adapter"
    if adapter.exists():
        shutil.rmtree(adapter)
    shutil.copytree(final, adapter)

    summary = {"run": run_name, "adapter": str(adapter), "source_checkpoint": str(final), "gpu": GPU}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # The weight cache is only worth having if it survives the run that paid for it.
    hf_cache.commit()
    outputs.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(val_split: float = 0.1, data_subdir: str = "trajectories", max_examples: int = 0,
         epochs: int = 0, run_name: str = "", train_args: str = ""):
    extra_train_args = ["--epochs", str(epochs)] if epochs else []
    extra_train_args += train_args.split()
    extra_train_args = extra_train_args or None
    result = train.remote(val_split=val_split, data_subdir=data_subdir, max_examples=max_examples,
                          run_name=run_name, extra_train_args=extra_train_args)
    print(f"\nadapter ready on volume cua-outputs at {result['adapter']}")
    print(f"serve it with:  CUA_ADAPTER_NAME={result['run']}/adapter modal serve modal_serve.py")
