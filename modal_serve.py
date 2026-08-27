"""Serve the CUA student on a Modal GPU as an OpenAI-compatible endpoint, with or without a LoRA adapter."""
# A second lane is env vars, not a fork: `CUA_SERVE_APP=... CUA_EVAL_SECRET=... modal deploy modal_serve.py`.
import os
import pathlib

import modal

APP_NAME = os.environ.get("CUA_SERVE_APP", "cua-serve")
EVAL_SECRET = os.environ.get("CUA_EVAL_SECRET", "cua-eval")
MODEL = os.environ.get("CUA_SERVE_MODEL", "Qwen/Qwen3.5-9B")
SERVED_NAME = os.environ.get("CUA_SERVED_NAME", "cua-student")
ADAPTER_NAME = os.environ.get("CUA_ADAPTER_NAME", "")
GPU = os.environ.get("CUA_SERVE_GPU", "L40S")
MAX_MODEL_LEN = int(os.environ.get("CUA_MAX_MODEL_LEN", "8192"))
GPU_MEM_UTIL = os.environ.get("CUA_GPU_MEM_UTIL", "0.90")
MAX_LORA_RANK = int(os.environ.get("CUA_MAX_LORA_RANK", "64"))

ADAPTERS_MOUNT = "/adapters"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
VLLM_CACHE_MOUNT = "/root/.cache/vllm"

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("cua-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("cua-vllm-cache", create_if_missing=True)
outputs = modal.Volume.from_name("cua-outputs", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.28.0", "huggingface_hub[hf_transfer]")
    # Baked at deploy time: module-level env reads re-run in the container, where these are unset.
    # APP_NAME and EVAL_SECRET are read again when the container re-imports this module, so they
    # are baked too — otherwise a renamed lane would resolve differently there than at deploy.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_LOG_STATS_INTERVAL": "10",
          "CUA_ADAPTER_NAME": ADAPTER_NAME, "CUA_SERVED_NAME": SERVED_NAME,
          "CUA_SERVE_APP": APP_NAME, "CUA_EVAL_SECRET": EVAL_SECRET})
)


warm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(image=warm_image, volumes={HF_CACHE_MOUNT: hf_cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600, cpu=4)
def warm_cache(model: str = MODEL):
    """Populate the weight cache on CPU so the GPU container starts against a warm volume."""
    import huggingface_hub

    path = huggingface_hub.snapshot_download(
        model,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HF_READ_TOKEN"),
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py", "*.jinja"],
    )
    hf_cache.commit()
    total = sum(f.stat().st_size for f in pathlib.Path(path).rglob("*") if f.is_file())
    print(f"warmed {model} -> {path} ({total / 1e9:.1f} GB)")
    return {"model": model, "path": path, "bytes": total}


@app.local_entrypoint()
def warm(model: str = MODEL):
    print(warm_cache.remote(model))


@app.server(
    image=image,
    gpu=GPU,
    volumes={HF_CACHE_MOUNT: hf_cache, VLLM_CACHE_MOUNT: vllm_cache, ADAPTERS_MOUNT: outputs},
    secrets=[modal.Secret.from_name(EVAL_SECRET), modal.Secret.from_name("huggingface")],
    port=8000,
    # Gaps between requests are seconds within an episode, so 5 min never evicts mid-run but caps idle billing.
    scaledown_window=300,
    startup_timeout=1800,
    target_concurrency=16,
    max_containers=1,
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        # vLLM reads VLLM_API_KEY from the env; passing --api-key would print the secret into Modal's logs.
        if not os.environ.get("VLLM_API_KEY"):
            raise RuntimeError("VLLM_API_KEY missing — the cua-eval secret is not attached")
        cmd = [
            "vllm", "serve", MODEL,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", GPU_MEM_UTIL,
        ]
        if ADAPTER_NAME:
            adapter_path = f"{ADAPTERS_MOUNT}/{ADAPTER_NAME}"
            if not os.path.isdir(adapter_path):
                raise RuntimeError(
                    f"adapter {ADAPTER_NAME!r} not found at {adapter_path} on the cua-outputs volume; "
                    f"run `modal volume ls cua-outputs` to see what is there"
                )
            cmd += [
                "--enable-lora",
                "--lora-modules", f"{SERVED_NAME}-lora={adapter_path}",
                "--max-lora-rank", str(MAX_LORA_RANK),
            ]
        print("launching:", " ".join(cmd), flush=True)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
        self.process.wait(timeout=60)
