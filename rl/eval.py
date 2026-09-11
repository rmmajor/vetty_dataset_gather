"""E3 eval: run the student on held-out tasks (temperature 0, k=1) and report per-split metrics with baseline deltas."""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from repo_pipeline.sandbox import install_sigterm_handler
from rl.episode import EpisodeConfig
from rl.judge import AsyncBridge
from rl.policies import make_policy
from rl.rollout import add_common_args, build_pool, policy_spec, selected_tasks
from rl.runner import make_run_context, run_batch, write_rollouts

logger = logging.getLogger(__name__)
METRIC_KEYS = (
    "verified_success",
    "check_pass_rate",
    "parse_failure_rate",
    "parse_abort_rate",
    "done_rate",
    "mean_steps_success",
    "loop_guard_rate",
)


def _rate(items, predicate):
    return round(sum(1 for i in items if predicate(i)) / len(items), 4) if items else 0.0


def metrics_by_split(records):
    out = {}
    for split in sorted({r.get("split") for r in records}):
        rs = [r for r in records if r.get("split") == split]
        turns = [t for r in rs for t in r.get("turns", [])]
        checked = [r for r in rs if r.get("check")]
        successes = [r for r in rs if r.get("success")]
        out[split] = {
            "n": len(rs),
            "verified_success": _rate(rs, lambda r: bool(r.get("success"))),
            "check_pass_rate": _rate(checked, lambda r: r["check"]["passed"]),
            "parse_failure_rate": _rate(turns, lambda t: not t.get("clean")),
            "parse_abort_rate": _rate(rs, lambda r: r.get("terminal_reason") == "parse_abort"),
            "done_rate": _rate(rs, lambda r: r.get("terminal_reason") == "done"),
            "mean_steps_success": round(sum(r["steps"] for r in successes) / len(successes), 2) if successes else 0.0,
            "loop_guard_rate": _rate(rs, lambda r: r.get("terminal_reason") == "loop_guard"),
        }
    return out


def per_domain(records):
    out = {}
    for site in sorted({r.get("site") for r in records}):
        rs = [r for r in records if r.get("site") == site]
        out[site] = {"n": len(rs), "success_rate": _rate(rs, lambda r: bool(r.get("success")))}
    return out


def render_report(header, metrics, domains, baseline=None):
    lines = ["# Student eval report", ""]
    lines += [f"- {k}: {v}" for k, v in header.items()]
    lines += ["", "| split | n | " + " | ".join(METRIC_KEYS) + " |", "|---|---|" + "---|" * len(METRIC_KEYS)]
    for split, m in metrics.items():
        cells = []
        for key in METRIC_KEYS:
            base = ((baseline or {}).get("metrics") or {}).get(split, {}).get(key)
            cells.append(f"{m[key]} ({m[key] - base:+.2f})" if isinstance(base, (int, float)) else str(m[key]))
        lines.append(f"| {split} | {m['n']} | " + " | ".join(cells) + " |")
    lines += ["", "| site | n | success_rate |", "|---|---|---|"]
    lines += [f"| {site} | {d['n']} | {d['success_rate']} |" for site, d in domains.items()]
    return "\n".join(lines) + "\n"


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Student eval on held-out tasks")
    add_common_args(parser)
    parser.set_defaults(split="eval", temperature=0.0, max_steps=40, verifier_mode="always")
    parser.add_argument("--baseline", default=None, help="report.json of a previous eval to diff against")
    parser.add_argument("--checkpoint-sha", default="")
    parser.add_argument("--out", default="eval_runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    install_sigterm_handler()
    run_id = args.run_id or time.strftime("eval-%Y%m%d-%H%M%S")
    suite, tasks = selected_tasks(args)
    spec = policy_spec(args)
    probe = make_policy(spec)
    header = {
        "run_id": run_id,
        "model": getattr(probe, "served_model_id", lambda: args.model)(),
        "checkpoint_sha": args.checkpoint_sha,
        "git_head": git_head(),
        "suite_sha256": suite["suite_sha256"],
        "tasks": len(tasks),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    bridge, pool = AsyncBridge(), build_pool(args)
    cfg = EpisodeConfig(max_steps=args.max_steps, verifier_mode=args.verifier_mode)
    run_ctx = make_run_context(run_id, args.out, 0, policy_tag=args.model, bridge=bridge)
    try:
        pool.start()
        records = run_batch(
            lambda: make_policy(spec),
            tasks,
            1,
            pool,
            run_ctx,
            cfg,
            on_record=lambda r: logger.info(
                f"{r['episode_id']}: {r.get('terminal_reason')} success={r.get('success')}"
            ),
        )
    finally:
        pool.stop()
        bridge.close()
    out_dir = Path(args.out) / run_id
    write_rollouts(records, out_dir / "rollouts.jsonl")
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    metrics, domains = metrics_by_split(records), per_domain(records)
    (out_dir / "report.json").write_text(
        json.dumps({"header": header, "metrics": metrics, "domains": domains}, indent=2)
    )
    report = render_report(header, metrics, domains, baseline)
    (out_dir / "report.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
