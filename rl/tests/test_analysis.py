import json
import math
import struct
import tempfile
from pathlib import Path

from rl.analysis import load_rollouts, lora_norms, per_task_stats, rollout_report, select_frontier

RECORDS = [
    {"task_id": "a", "success": True, "reward": 1.9, "terminal_reason": "done", "reward_status": "final"},
    {"task_id": "a", "success": False, "reward": 0.0, "terminal_reason": "step_cap", "reward_status": "final"},
    {"task_id": "b", "success": False, "reward": -0.1, "terminal_reason": "loop_guard", "reward_status": "final"},
    {"task_id": "b", "success": False, "reward": -0.1, "terminal_reason": "loop_guard", "reward_status": "final"},
    {"task_id": "c", "success": False, "reward": None, "terminal_reason": "done", "reward_status": "deferred"},
    {"task_id": "c", "success": True, "reward": 1.0, "terminal_reason": "done", "reward_status": "final"},
]


def test_harness_failures_leave_the_pass_rate_denominator():
    crashed = dict(RECORDS[0], task_id="d", terminal_reason="abandoned", success=False, reward=0.2)
    solved = dict(RECORDS[0], task_id="d", terminal_reason="done", success=True, reward=1.9)
    stats = per_task_stats([crashed, solved])["d"]
    assert stats["k"] == 1 and stats["harness_failures"] == 1 and stats["pass_rate"] == 1.0
    only_crashes = per_task_stats([crashed])["d"]
    assert only_crashes["pass_rate"] is None and select_frontier([crashed], 0.0, 1.0) == []


def test_a_walled_site_is_not_scored_as_a_policy_failure():
    walled = dict(RECORDS[0], task_id="e", terminal_reason="step_cap", success=False, reward=-0.2, blocked="cloudflare")
    solved = dict(RECORDS[0], task_id="e", terminal_reason="done", success=True, reward=1.9)
    stats = per_task_stats([walled, solved])["e"]
    assert stats["k"] == 1 and stats["harness_failures"] == 1 and stats["pass_rate"] == 1.0


def test_per_task_stats_flags_zero_variance_and_missing():
    stats = per_task_stats(RECORDS)
    assert stats["a"]["pass_rate"] == 0.5 and not stats["a"]["zero_variance"]
    assert stats["b"]["pass_rate"] == 0.0 and stats["b"]["zero_variance"]
    assert stats["c"]["missing_rewards"] == 1 and not stats["c"]["zero_variance"]


def test_rollout_report_counts():
    report = rollout_report(RECORDS)
    assert report["episodes"] == 6 and report["missing_rewards"] == 1
    assert report["zero_variance_tasks"] == ["b"] and report["zero_variance_fraction"] == round(1 / 3, 4)
    assert report["terminal_reasons"]["loop_guard"] == 2 and report["reward_status"]["deferred"] == 1
    assert report["reward_histogram"]["None"] == 1


def test_select_frontier_keeps_only_the_band():
    assert select_frontier(RECORDS, 0.2, 0.8) == ["a", "c"]
    assert select_frontier(RECORDS, 0.6, 1.0) == []


def test_load_rollouts_reads_jsonl_from_dir():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
        assert len(load_rollouts(tmp)) == 6


def _write_safetensors(path, tensors):
    header, blob, offset = {}, b"", 0
    for name, values in tensors.items():
        data = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": [len(values)], "data_offsets": [offset, offset + len(data)]}
        blob += data
        offset += len(data)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)


def test_lora_norms_on_constructed_fixture():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adapter_model.safetensors"
        _write_safetensors(
            path, {"m.lora_A.weight": [3.0, 4.0], "m.lora_B.weight": [1.0, 2.0, 2.0], "m.base.weight": [9.0]}
        )
        norms = lora_norms(path)
        assert math.isclose(norms["A"], 5.0) and math.isclose(norms["B"], 3.0)
        assert norms["tensors"] == {"A": 1, "B": 1}


def test_load_rollouts_falls_back_to_episode_files_of_a_partial_run():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "iter_000" / "site__task" / "k00" / "task_site__task__i000__k00"
        run.mkdir(parents=True)
        (run / "episode.json").write_text(json.dumps(RECORDS[0]))
        assert len(load_rollouts(Path(tmp))) == 1
        (Path(tmp) / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
        assert len(load_rollouts(Path(tmp))) == len(RECORDS)


def test_zero_variance_needs_two_samples():
    single = per_task_stats([RECORDS[0]])
    assert not single["a"]["zero_variance"]
    assert per_task_stats(RECORDS)["b"]["zero_variance"]
