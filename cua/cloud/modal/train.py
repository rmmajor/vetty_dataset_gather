"""Run the unchanged cua/sft/legacy_train_qwen.py convert+train on a Modal GPU, leaving the adapter on a volume the serve app mounts."""

# Variants are configs, not forks: `modal run cua/cloud/modal/train.py --config config_sft_v2_conservative.yaml --run-name sft-v2`.
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
CONFIG_DIR = pathlib.Path(__file__).resolve().parents[3]
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
        # 4.x renamed train_type -> tuner_type; cua/sft/legacy_train_qwen.py's flag map targets that CLI.
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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "USE_HF": "1", "PYTHONPATH": CODE_ROOT})
    .add_local_file("corpus_health.json", f"{CODE_ROOT}/corpus_health.json")
    .add_local_file("pyproject.toml", f"{CODE_ROOT}/pyproject.toml")
    .add_local_dir("constraints", f"{CODE_ROOT}/constraints")
    # Reached through PYTHONPATH, not an editable install: add_local_* must close an image, so no pip step can follow it.
    .add_local_dir("cua", f"{CODE_ROOT}/cua", ignore=["**/__pycache__", "**/*.pyc"])
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
def train(
    val_split: float = 0.1,
    data_subdir: str = "trajectories",
    max_examples: int = 0,
    run_name: str = "",
    config: str = DEFAULT_CONFIG,
    extra_convert_args: list = None,
    extra_train_args: list = None,
    skip_audit: bool = False,
):
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

    audit = [sys.executable, "-m", "cua.pipeline.corpus_audit", str(traj_dir)]
    if skip_audit:
        print("skip_audit: corpus gates NOT enforced (timing smoke only)", flush=True)
    else:
        print("+", " ".join(audit), flush=True)
        subprocess.run(audit, cwd=CODE_ROOT, check=True)

    convert = [
        sys.executable,
        "-m",
        "cua.sft",
        "convert",
        "--input",
        str(traj_dir),
        "--output",
        str(dataset_dir),
        "--val-split",
        str(val_split),
        "--exclude-waste-steps",
        "--meandering-mode",
        "subsample",
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
        sys.executable,
        "-m",
        "cua.sft",
        "train",
        "--config",
        str(run_config),
        "--dataset",
        str(dataset_file),
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


@app.function(
    image=image,
    volumes={HF_CACHE_MOUNT: hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=1800,
    cpu=4,
    memory=16384,
)
def encode_check(model: str = "Qwen/Qwen3.5-9B") -> dict:
    """CPU-only: tokenize one two-image sft3 step and one grounding example through ms-swift's template and report the image-pad and label counts."""
    import json

    from PIL import Image

    os.environ.setdefault("MAX_PIXELS", "1003520")
    from swift.model import get_model_processor
    from swift.template import get_template

    from cua.sft import legacy_train_qwen as tq

    tmp = pathlib.Path("/tmp/encode_check")
    tmp.mkdir(exist_ok=True)
    cur, prev = tmp / "cur.png", tmp / "prev.png"
    Image.new("RGB", (1280, 720), (240, 240, 240)).save(cur)
    prev.write_bytes(tq.downscale_prev(cur.read_bytes()))
    history = [
        f"click({i * 10 + 5}, {i * 7 + 3}) [no change] -> /settings/notifications/preferences" for i in range(23)
    ]
    step = tq.build_step_messages(
        tq.SYSTEM_PROMPT,
        tq.build_step_observation(
            "Toggling Email notifications persists after reload. Verify the box stays unchecked.",
            history,
            str(cur),
            str(prev),
        ),
    )
    step.append(
        {
            "role": "assistant",
            "content": "<think>Clicking Save worked: a banner is visible. I reload to confirm.</think>\naction: click(<ref>(500, 500)</ref>, button=left)",
        }
    )
    grounding = tq.build_step_messages(
        tq.GROUNDING_SYSTEM_PROMPT, tq.build_grounding_observation('the "Save" button', str(cur))
    )
    grounding.append({"role": "assistant", "content": "action: click(<ref>(500, 500)</ref>, button=left)"})
    loaded = get_model_processor(model, load_model=False)
    processor = loaded[1] if isinstance(loaded, tuple) else loaded
    template = get_template(processor, max_length=4096, max_pixels=int(os.environ["MAX_PIXELS"]))
    template.set_mode("train")
    pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    report = {"model": model, "template": template.template_meta.template_type, "max_pixels": os.environ["MAX_PIXELS"]}
    for name, messages in (("two_image_step", step), ("grounding", grounding)):
        encoded = template.encode({"messages": messages})
        ids, labels = list(encoded["input_ids"]), list(encoded.get("labels") or [])
        supervised = [t for t in labels if t != -100]
        report[name] = {
            "input_tokens": len(ids),
            "image_pad_tokens": ids.count(pad_id),
            "label_tokens": len(supervised),
            "labels_text": processor.tokenizer.decode(supervised)[:300],
        }
    print(json.dumps(report, indent=2), flush=True)
    hf_cache.commit()
    return report


@app.local_entrypoint()
def main(
    val_split: float = 0.1,
    data_subdir: str = "trajectories",
    max_examples: int = 0,
    epochs: int = 0,
    run_name: str = "",
    config: str = DEFAULT_CONFIG,
    train_args: str = "",
    convert_args: str = "",
    skip_audit: bool = False,
):
    extra_train_args = ["--epochs", str(epochs)] if epochs else []
    extra_train_args += train_args.split()
    extra_train_args = extra_train_args or None
    result = train.remote(
        val_split=val_split,
        data_subdir=data_subdir,
        max_examples=max_examples,
        run_name=run_name,
        config=config,
        extra_train_args=extra_train_args,
        extra_convert_args=convert_args.split() or None,
        skip_audit=skip_audit,
    )
    print(f"\nadapter ready on volume cua-outputs at {result['adapter']}")
    print(f"serve it with:  CUA_ADAPTER_NAME={result['run']}/adapter modal serve cua/cloud/modal/serve.py")
