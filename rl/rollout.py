"""Rollout CLI: run k samples per task with any policy, write rl_runs/<run_id>/iter_NNN/rollouts.jsonl."""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

from repo_pipeline.sandbox import install_sigterm_handler
from rl.backends import BACKENDS, WorkerPool
from rl.episode import EpisodeConfig
from rl.judge import AsyncBridge
from rl.policies import make_policy
from rl.runner import make_run_context, run_batch, summarize, write_rollouts
from rl.tasks import SUITE_PATH, load_suite, select_tasks

logger = logging.getLogger(__name__)


def add_common_args(parser):
    parser.add_argument("--policy", default="openai", choices=["openai", "mock", "scripted"])
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="policy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--replay-dir", default=None, help="trajectory dir for --policy mock")
    parser.add_argument("--tasks", default=str(SUITE_PATH))
    parser.add_argument("--split", default="train")
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--drivers-per-sandbox", type=int, default=1)
    parser.add_argument("--backend", default="public_site", choices=sorted(BACKENDS))
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--verifier-mode", default="fallback", choices=["always", "fallback", "never"])
    parser.add_argument("--run-id", default=None)


def policy_spec(args):
    if args.policy == "openai":
        return {"kind": "openai", "endpoint": args.endpoint, "model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens}
    if args.policy == "mock":
        return {"kind": "mock", "traj_dir": args.replay_dir}
    return {"kind": "scripted", "texts": ["action: done()"]}


def build_pool(args):
    backend_cls = BACKENDS[args.backend]
    backend = backend_cls("<html><body>static</body></html>") if args.backend == "static_page" else backend_cls()
    return WorkerPool(backend, args.workers, args.drivers_per_sandbox)


def selected_tasks(args):
    suite = load_suite(args.tasks)
    tasks = select_tasks(suite, split=args.split, task_ids=args.task_ids.split(",") if args.task_ids else None)
    return suite, tasks[: args.limit] if args.limit else tasks


def main():
    parser = argparse.ArgumentParser(description="RL rollouts")
    add_common_args(parser)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--out", default="rl_runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    install_sigterm_handler()
    run_id = args.run_id or time.strftime("rollout-%Y%m%d-%H%M%S")
    suite, tasks = selected_tasks(args)
    spec = policy_spec(args)
    bridge = AsyncBridge()
    pool = build_pool(args)
    cfg = EpisodeConfig(max_steps=args.max_steps, verifier_mode=args.verifier_mode)
    run_ctx = make_run_context(run_id, args.out, args.iter, policy_tag=args.model, bridge=bridge)
    out_dir = Path(args.out) / run_id / f"iter_{args.iter:03d}"
    try:
        pool.start()
        records = run_batch(lambda: make_policy(spec), tasks, args.k, pool, run_ctx, cfg,
                            on_record=lambda r: logger.info(f"{r['episode_id']}: {r.get('terminal_reason')} reward={r.get('reward')}"))
    finally:
        pool.stop()
        bridge.close()
    write_rollouts(records, out_dir / "rollouts.jsonl")
    summary = {"run_id": run_id, "suite_sha256": suite["suite_sha256"], "policy": spec, **summarize(records)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
