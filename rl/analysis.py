"""Run diagnostics: adapter LoRA norms (is the run alive?), rollout reward structure, frontier selection."""
import argparse
import collections
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

# A GRPO adapter whose LoRA-B norm stays under this share of the SFT adapter's after ~10 steps learned nothing — kill the run.
DEAD_RUN_B_RATIO = 0.10
# Endings that measure the harness rather than the policy; counting them as failures would push a task out of the band for free.
HARNESS_REASONS = {"harness_error", "abandoned", "skipped", "step_error"}


def _tensor(raw, dtype):
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    raise ValueError(f"unsupported safetensors dtype {dtype}")


def lora_norms(path):
    """Frobenius norms of all lora_A / lora_B tensors, parsed straight from the safetensors layout."""
    sums = {"A": 0.0, "B": 0.0}
    counts = {"A": 0, "B": 0}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        for name, meta in header.items():
            which = "B" if "lora_B" in name else ("A" if "lora_A" in name else None)
            if name == "__metadata__" or which is None:
                continue
            start, end = meta["data_offsets"]
            f.seek(base + start)
            values = _tensor(f.read(end - start), meta["dtype"]).astype(np.float64)
            sums[which] += float((values * values).sum())
            counts[which] += 1
    return {"A": math.sqrt(sums["A"]), "B": math.sqrt(sums["B"]), "tensors": counts}


def load_rollouts(path):
    """rollouts.jsonl when the run finished, else the episode.json files it already wrote, so a run stopped early is still analysable."""
    path = Path(path)
    if path.is_file():
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    manifest = path / "rollouts.jsonl"
    if manifest.is_file():
        return [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    records = [json.loads(p.read_text()) for p in sorted(path.glob("**/task_*/episode.json"))]
    if not records:
        raise SystemExit(f"no rollouts.jsonl and no episode.json under {path}")
    return records


def per_task_stats(records):
    groups = collections.defaultdict(list)
    for r in records:
        groups[r["task_id"]].append(r)
    stats = {}
    for task_id, rows in sorted(groups.items()):
        scored = [r for r in rows if r.get("terminal_reason") not in HARNESS_REASONS and not r.get("blocked")]
        rewards = [r.get("reward") for r in scored]
        defined = [x for x in rewards if x is not None]
        successes = sum(1 for r in scored if r.get("success"))
        stats[task_id] = {
            "k": len(scored),
            "harness_failures": len(rows) - len(scored),
            "successes": successes,
            "pass_rate": round(successes / len(scored), 4) if scored else None,
            # Undefined until a task has two samples to differ; a partial sweep would otherwise read as all-degenerate.
            "zero_variance": len(scored) > 1 and len(set(defined)) <= 1 and len(defined) == len(rewards),
            "missing_rewards": sum(1 for x in rewards if x is None),
            "rewards": [None if x is None else round(x, 3) for x in rewards],
        }
    return stats


def rollout_report(records):
    rewards = [r.get("reward") for r in records]
    histogram = collections.Counter("None" if x is None else f"{round(x, 1):+.1f}" for x in rewards)
    tasks = per_task_stats(records)
    zero_variance = [t for t, s in tasks.items() if s["zero_variance"]]
    return {
        "episodes": len(records),
        "reward_histogram": dict(sorted(histogram.items())),
        "terminal_reasons": dict(collections.Counter(r.get("terminal_reason") for r in records)),
        "reward_status": dict(collections.Counter(r.get("reward_status") for r in records)),
        "missing_rewards": sum(1 for x in rewards if x is None),
        "zero_variance_tasks": zero_variance,
        "zero_variance_fraction": round(len(zero_variance) / len(tasks), 4) if tasks else 0.0,
        "per_task": tasks,
    }


def select_frontier(records, low, high, min_scored=2):
    """Tasks whose measured pass rate sits in the band; a task with too few scored samples is not measured."""
    tasks = per_task_stats(records)
    return sorted(t for t, s in tasks.items()
                  if s["pass_rate"] is not None and s["k"] >= min_scored and low <= s["pass_rate"] <= high)


def _cmd_bnorm(args):
    norms = lora_norms(args.adapter)
    out = {"adapter": args.adapter, "B_frobenius": round(norms["B"], 4), "A_frobenius": round(norms["A"], 2), "tensors": norms["tensors"]}
    if args.ref:
        ref = lora_norms(args.ref)
        ratio = norms["B"] / ref["B"] if ref["B"] else float("inf")
        out.update({"ref_B_frobenius": round(ref["B"], 4), "B_ratio": round(ratio, 4), "dead_run": ratio < DEAD_RUN_B_RATIO})
    print(json.dumps(out, indent=2))
    return 1 if out.get("dead_run") else 0


def _cmd_rollouts(args):
    print(json.dumps(rollout_report(load_rollouts(args.rollouts)), indent=2))
    return 0


def _cmd_select(args):
    records = load_rollouts(args.rollouts)
    selected = select_frontier(records, args.min, args.max)
    summary_path = Path(args.rollouts if Path(args.rollouts).is_dir() else Path(args.rollouts).parent) / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    payload = {
        "selected": selected, "band": [args.min, args.max], "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policy": summary.get("policy"), "suite_sha256": summary.get("suite_sha256"), "run_id": summary.get("run_id"),
        "per_task": per_task_stats(records),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(selected)} tasks in [{args.min}, {args.max}] -> {args.out}")
    return 0 if selected else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="RL run diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bnorm", help="LoRA A/B Frobenius norms of an adapter; --ref flags a dead run")
    b.add_argument("adapter")
    b.add_argument("--ref", help="reference adapter (the SFT LoRA) for the B-norm ratio")
    r = sub.add_parser("rollouts", help="reward histogram, per-task pass@k, zero-variance groups")
    r.add_argument("rollouts", help="rollouts.jsonl or the iter dir containing it")
    s = sub.add_parser("select-frontier", help="write frontier_selection.json with tasks inside the pass-rate band")
    s.add_argument("--rollouts", required=True)
    s.add_argument("--min", type=float, default=0.2)
    s.add_argument("--max", type=float, default=0.8)
    s.add_argument("--out", default="rl/frontier_selection.json")
    args = parser.parse_args(argv)
    return {"bnorm": _cmd_bnorm, "rollouts": _cmd_rollouts, "select-frontier": _cmd_select}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
