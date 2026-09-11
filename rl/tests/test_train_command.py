"""Every config key reaches swift sft: mapped keys through the flag map, the rest forwarded verbatim."""

from train_qwen import build_sft_command


def _pair(cmd, flag):
    index = cmd.index(flag)
    return cmd[index : index + 2]


def test_unmapped_config_keys_are_forwarded_as_flags():
    config = {
        "model": "Qwen/Qwen3.5-9B",
        "num_train_epochs": 2,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "max_pixels": 1003520,
        "freeze_vit": True,
    }
    cmd = build_sft_command(config, "ds.jsonl", None, [])
    assert cmd[:2] == ["swift", "sft"]
    assert _pair(cmd, "--model") == ["--model", "Qwen/Qwen3.5-9B"]
    assert _pair(cmd, "--num_train_epochs") == ["--num_train_epochs", "2"]
    assert _pair(cmd, "--save_strategy") == ["--save_strategy", "epoch"]
    assert _pair(cmd, "--save_total_limit") == ["--save_total_limit", "2"]
    assert _pair(cmd, "--freeze_vit") == ["--freeze_vit", "true"]
    assert "--max_pixels" not in cmd


def test_passthrough_args_follow_config_flags():
    cmd = build_sft_command({"model": "m"}, "ds.jsonl", "val.jsonl", ["--packing", "true"])
    assert cmd[-2:] == ["--packing", "true"]
    assert _pair(cmd, "--val_dataset") == ["--val_dataset", "val.jsonl"]
