"""vLLM serve exposes every adapter it is given, so one rental can score several checkpoints."""

from rl_train import build_vllm_serve_command

CFG = {"base_model": "Qwen/Qwen3.5-9B", "compute": {"serve_port": 8001}}


def _modules(cmd):
    return cmd[cmd.index("--lora-modules") + 1 : cmd.index("--max-lora-rank")]


def test_single_adapter_keeps_the_canonical_served_name():
    cmd = build_vllm_serve_command(CFG, "/workspace/sft_adapter")
    assert _modules(cmd) == ["cua-student-lora=/workspace/sft_adapter"]


def test_named_adapters_each_get_their_own_served_name():
    cmd = build_vllm_serve_command(CFG, {"sft-v2": "/w/a", "sft-v3": "/w/b"})
    assert _modules(cmd) == ["sft-v2=/w/a", "sft-v3=/w/b"]
    assert "--enable-lora" in cmd


def test_no_adapter_serves_the_base_alone():
    cmd = build_vllm_serve_command(CFG)
    assert "--enable-lora" not in cmd and "--lora-modules" not in cmd
