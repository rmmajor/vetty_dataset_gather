"""Run the unchanged `python -m cua.sft` convert and train on a Modal GPU, leaving the adapter on a volume."""

# Variants are configs, not forks: `modal run cua/cloud/modal/train.py --config config_sft_v2_conservative.yaml`.
import json
import os
import pathlib
import shutil
import subprocess
import sys
from pathlib import Path

import modal
import yaml
from PIL import Image

from cua.cloud.modal.common import (
    CODE_ROOT,
    DATA_MOUNT,
    HF_CACHE_MOUNT,
    OUT_MOUNT,
    REPO_ROOT,
    commit_periodically,
    configs,
    hf_cache,
    outputs,
    trajectories,
    with_package,
)
from cua.prompt.effects import downscale_prev
from cua.prompt.format import (
    GROUNDING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_grounding_observation,
    build_step_messages,
    build_step_observation,
)
from cua.settings import CLOUD, MAX_PIXELS_DEFAULT
from cua.sft.recipe import convert_command, train_command

APP_NAME = "cua-train"

CONFIGS = configs()
DEFAULT_CONFIG = "config.yaml"

app = modal.App(APP_NAME)

# Pinned to CUDA 12.4.1 + torch 2.6.0: the SFT stack ms-swift 4.x was proven on.
image = with_package(
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "torch==2.6.0",
        # qwen_vl_utils imports torchvision; 0.21.0 is the release paired with torch 2.6.0.
        "torchvision==0.21.0",
        # 4.x renamed train_type -> tuner_type; cua.sft.swift_launch's flag map targets that CLI.
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
    ),
    # USE_HF sends ms-swift to HuggingFace instead of its ModelScope default, whose cache is not mounted.
    env={"USE_HF": "1"},
    files=("corpus_health.json", *(REPO_ROOT / name for name in CONFIGS)),
)


@app.function(
    image=image,
    gpu=CLOUD.train_gpu,
    volumes={DATA_MOUNT: trajectories, OUT_MOUNT: outputs, HF_CACHE_MOUNT: hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=CLOUD.train_timeout,
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
    traj_dir = Path(DATA_MOUNT) / data_subdir
    if not traj_dir.is_dir() or not any(traj_dir.iterdir()):
        raise RuntimeError(
            f"{traj_dir} is empty — upload the corpus first:\n"
            f"  modal volume put cua-trajectories ./trajectories /{data_subdir}"
        )

    run_name = run_name or CLOUD.run_name
    run_dir = Path(OUT_MOUNT) / run_name
    swift_out = run_dir / "swift"
    dataset_dir = run_dir / "training_data"
    run_dir.mkdir(parents=True, exist_ok=True)

    audit = [sys.executable, "-m", "cua.pipeline.corpus.audit", str(traj_dir)]
    if skip_audit:
        print("skip_audit: corpus gates NOT enforced (timing smoke only)", flush=True)
    else:
        print("+", " ".join(audit), flush=True)
        subprocess.run(audit, cwd=CODE_ROOT, check=True)

    convert = convert_command(sys.executable, traj_dir, dataset_dir, val_split, extra_convert_args or [])
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

    train_cmd = train_command(sys.executable, run_config, dataset_file, extra_train_args or [])
    print("+", " ".join(train_cmd), flush=True)

    stop = commit_periodically(outputs, CLOUD.train_commit_every_s)
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

    summary = {"run": run_name, "adapter": str(adapter), "source_checkpoint": str(final), "gpu": CLOUD.train_gpu}
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
    """CPU-only: tokenize one two-image step and one grounding example, and report their image-pad and label counts."""
    # ms-swift's template reads MAX_PIXELS from the env, so it is written here and read back
    max_pixels = os.environ.setdefault("MAX_PIXELS", str(MAX_PIXELS_DEFAULT))
    # ms-swift is installed only in the image, and reads MAX_PIXELS as the line above sets it
    from swift.model import get_model_processor
    from swift.template import get_template

    tmp = pathlib.Path("/tmp/encode_check")
    tmp.mkdir(exist_ok=True)
    cur, prev = tmp / "cur.png", tmp / "prev.png"
    Image.new("RGB", (1280, 720), (240, 240, 240)).save(cur)
    prev.write_bytes(downscale_prev(cur.read_bytes()))
    history = [
        f"click({i * 10 + 5}, {i * 7 + 3}) [no change] -> /settings/notifications/preferences" for i in range(23)
    ]
    step = build_step_messages(
        SYSTEM_PROMPT,
        build_step_observation(
            "Toggling Email notifications persists after reload. Verify the box stays unchecked.",
            history,
            str(cur),
            str(prev),
        ),
    )
    step.append(
        {
            "role": "assistant",
            "content": "<think>Clicking Save worked: a banner is visible. I reload to confirm.</think>\n"
            "action: click(<ref>(500, 500)</ref>, button=left)",
        }
    )
    grounding = build_step_messages(GROUNDING_SYSTEM_PROMPT, build_grounding_observation('the "Save" button', str(cur)))
    grounding.append({"role": "assistant", "content": "action: click(<ref>(500, 500)</ref>, button=left)"})
    loaded = get_model_processor(model, load_model=False)
    processor = loaded[1] if isinstance(loaded, tuple) else loaded
    template = get_template(processor, max_length=4096, max_pixels=int(max_pixels))
    template.set_mode("train")
    pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    report = {"model": model, "template": template.template_meta.template_type, "max_pixels": max_pixels}
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
