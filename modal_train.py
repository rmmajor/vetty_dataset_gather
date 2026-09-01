"""Run the unchanged train_qwen.py convert+train on a Modal GPU, leaving the adapter on a volume the serve app mounts."""
# Variants are configs, not forks: `modal run modal_train.py --config config_sft_v2_conservative.yaml --run-name sft-v2`.
import os
import pathlib

import modal

APP_NAME = "cua-train"
GPU = os.environ.get("CUA_TRAIN_GPU", "A100-80GB")
RUN_NAME = os.environ.get("CUA_RUN_NAME", "sft-v1")
TRAIN_TIMEOUT = int(os.environ.get("CUA_TRAIN_TIMEOUT", str(10 * 3600)))
# Without periodic commits the per-epoch checkpoints only reach the volume when the whole run ends,
# so a crash costs every epoch and an earlier epoch cannot be served while training continues.
COMMIT_EVERY_S = int(os.environ.get("CUA_TRAIN_COMMIT_EVERY_S", "600"))

DATA_MOUNT = "/data"
OUT_MOUNT = "/outputs"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
CODE_ROOT = "/workspace"
# Every config*.yaml ships in the image so --config picks one per run without a rebuild.
CONFIG_DIR = pathlib.Path(__file__).parent
CONFIGS = sorted(path.name for path in CONFIG_DIR.glob("config*.yaml"))
DEFAULT_CONFIG = "config.yaml"

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
    # USE_HF sends ms-swift to HuggingFace instead of its ModelScope default, whose cache is not mounted.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "USE_HF": "1"})
    .add_local_file("train_qwen.py", f"{CODE_ROOT}/train_qwen.py")
    .add_local_file("corpus_health.json", f"{CODE_ROOT}/corpus_health.json")
    .add_local_dir("repo_pipeline", f"{CODE_ROOT}/repo_pipeline", ignore=["**/__pycache__", "**/*.pyc"])
)
for _name in CONFIGS:
    image = image.add_local_file(str(CONFIG_DIR / _name), f"{CODE_ROOT}/{_name}")


@app.function(
    image=image,
    gpu=GPU,
    volumes={DATA_MOUNT: trajectories, OUT_MOUNT: outputs, HF_CACHE_MOUNT: hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=TRAIN_TIMEOUT,
)
def train(val_split: float = 0.1, data_subdir: str = "trajectories", max_examples: int = 0,
          run_name: str = "", config: str = DEFAULT_CONFIG,
          extra_convert_args: list = None, extra_train_args: list = None):
    import json
    import shutil
    import subprocess
    import sys
    import threading
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

    audit = [sys.executable, "-m", "repo_pipeline.corpus_audit", str(traj_dir)]
    print("+", " ".join(audit), flush=True)
    subprocess.run(audit, cwd=CODE_ROOT, check=True)

    convert = [
        sys.executable, "train_qwen.py", "convert",
        "--input", str(traj_dir),
        "--output", str(dataset_dir),
        "--val-split", str(val_split),
        "--exclude-waste-steps", "--meandering-mode", "subsample",
    ] + (extra_convert_args or [])
    print("+", " ".join(convert), flush=True)
    subprocess.run(convert, cwd=CODE_ROOT, check=True)

    dataset_file = dataset_dir / "sft_dataset.jsonl"
    if max_examples:
        kept = dataset_file.read_text().splitlines()[:max_examples]
        dataset_file = dataset_dir / "sft_smoke.jsonl"
        dataset_file.write_text("\n".join(kept) + "\n")
        print(f"smoke mode: training on {len(kept)} examples", flush=True)

    config_path = Path(CODE_ROOT, config)
    if not config_path.is_file():
        raise RuntimeError(f"config {config!r} is not in the image; shipped: {CONFIGS}")
    print(f"training config: {config}", flush=True)
    config = yaml.safe_load(config_path.read_text())
    config["output_dir"] = str(swift_out)
    run_config = run_dir / "config.yaml"
    run_config.write_text(yaml.safe_dump(config, sort_keys=False))

    train_cmd = [
        sys.executable, "train_qwen.py", "train",
        "--config", str(run_config),
        "--dataset", str(dataset_file),
    ] + (extra_train_args or [])
    print("+", " ".join(train_cmd), flush=True)

    stop = threading.Event()

    def commit_periodically():
        while not stop.wait(COMMIT_EVERY_S):
            outputs.commit()

    committer = threading.Thread(target=commit_periodically, daemon=True)
    committer.start()
    try:
        subprocess.run(train_cmd, cwd=CODE_ROOT, check=True)
    finally:
        stop.set()
        outputs.commit()

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
         epochs: int = 0, run_name: str = "", config: str = DEFAULT_CONFIG, train_args: str = ""):
    extra_train_args = ["--epochs", str(epochs)] if epochs else []
    extra_train_args += train_args.split()
    extra_train_args = extra_train_args or None
    result = train.remote(val_split=val_split, data_subdir=data_subdir, max_examples=max_examples,
                          run_name=run_name, config=config, extra_train_args=extra_train_args)
    print(f"\nadapter ready on volume cua-outputs at {result['adapter']}")
    print(f"serve it with:  CUA_ADAPTER_NAME={result['run']}/adapter modal serve modal_serve.py")
