"""Merge a LoRA adapter into its base on Modal, so a second adapter can be served on top of it."""
# A GRPO adapter is trained over MERGED SFT weights and cannot be stacked on the raw base; this
# writes the merged checkpoint to the shared HF cache once, and modal_serve then serves it by path.
import os
import pathlib
import subprocess

import modal

BASE_MODEL = os.environ.get("CUA_MERGE_BASE", "Qwen/Qwen3.5-9B")
ADAPTER_NAME = os.environ.get("CUA_MERGE_ADAPTER", "sft-v2/adapter")
MERGED_NAME = os.environ.get("CUA_MERGE_OUT", ADAPTER_NAME.replace("/", "__"))
GPU = os.environ.get("CUA_MERGE_GPU", "L40S")

ADAPTERS_MOUNT = "/adapters"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

hf_cache = modal.Volume.from_name("cua-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("cua-outputs", create_if_missing=True)

app = modal.App("cua-merge")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # qwen_vl_utils is a hard requirement of ms-swift's Qwen-VL template, not an optional extra.
    .uv_pip_install("ms-swift==4.5.2", "torch", "torchvision", "transformers", "peft",
                    "qwen_vl_utils>=0.0.14", "huggingface_hub[hf_transfer]", "accelerate")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu=GPU, volumes={HF_CACHE_MOUNT: hf_cache, ADAPTERS_MOUNT: outputs},
              secrets=[modal.Secret.from_name("huggingface")], timeout=10800)
def merge():
    out = pathlib.Path(HF_CACHE_MOUNT) / "merged" / MERGED_NAME
    if (out / "config.json").exists():
        print(f"already merged at {out}", flush=True)
        return str(out)
    adapter = pathlib.Path(ADAPTERS_MOUNT) / ADAPTER_NAME
    if not (adapter / "adapter_config.json").is_file():
        raise RuntimeError(f"no adapter_config.json under {adapter}")
    cmd = ["swift", "export", "--model", BASE_MODEL, "--adapters", str(adapter),
           "--merge_lora", "true", "--exist_ok", "true", "--output_dir", str(out)]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    hf_cache.commit()
    print(f"merged -> {out}", flush=True)
    return str(out)


@app.local_entrypoint()
def main():
    print(merge.remote())
