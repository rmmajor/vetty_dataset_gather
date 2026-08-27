"""Serve the CUA student on a Modal GPU as an OpenAI-compatible endpoint, with or without a LoRA adapter."""
import os

import modal

APP_NAME = "cua-serve"
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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_LOG_STATS_INTERVAL": "10",
          "CUA_ADAPTER_NAME": ADAPTER_NAME, "CUA_SERVED_NAME": SERVED_NAME})
)


@app.server(
    image=image,
    gpu=GPU,
    volumes={HF_CACHE_MOUNT: hf_cache, VLLM_CACHE_MOUNT: vllm_cache, ADAPTERS_MOUNT: outputs},
    secrets=[modal.Secret.from_name("cua-eval")],
    port=8000,
    # Gaps between requests are seconds within an episode, so 5 min never evicts mid-run but caps idle billing.
    scaledown_window=300,
    startup_timeout=900,
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
