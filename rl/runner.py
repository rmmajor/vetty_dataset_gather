"""Run episodes in-process over a WorkerPool — shared by rollout, eval and smoke."""
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rl.episode import Episode, RunContext
from rl.rewards import load_reward_config

logger = logging.getLogger(__name__)


def make_run_context(run_id, out_root, iteration=0, policy_tag="", bridge=None):
    return RunContext(run_id=run_id, out_root=Path(out_root), iteration=iteration, policy_tag=policy_tag,
                      reward_config=load_reward_config(), bridge=bridge)


def require_oracles(tasks, verifier_mode):
    # With the judge off, a task without a success_check would finish verdictless — refuse before any episode runs.
    if verifier_mode != "never":
        return
    items = tasks.values() if isinstance(tasks, dict) else tasks
    missing = sorted(t["task_id"] for t in items if not t.get("success_check"))
    if missing:
        raise SystemExit(f"--verifier-mode never requires a success_check on every task; missing: {', '.join(missing)}")


def run_episode(policy, task, pool, sample_k, run_ctx, cfg):
    worker, slot = pool.acquire()
    try:
        episode = None
        try:
            episode = Episode(task, worker, slot, sample_k, run_ctx, cfg)
            observation = episode.reset()
        except Exception as e:
            # A precondition refusal skips a task, a dead sandbox is a harness error, and neither may kill a batch of hundreds.
            reason = "skipped" if isinstance(e, RuntimeError) else "harness_error"
            episode_id = getattr(episode, "episode_id", f"{task['task_id']}__i{run_ctx.iteration:03d}__k{sample_k:02d}")
            logger.warning(f"{episode_id}: episode did not start ({reason}): {str(e)[:200]}")
            return {"episode_id": episode_id, "task_id": task["task_id"], "site": task.get("site"), "split": task.get("split"),
                    "k": sample_k, "iter": run_ctx.iteration, "skipped": str(e)[:300], "reward": None, "success": None,
                    "terminal_reason": reason, "turns": [], "steps": 0, "reward_status": "skipped"}
        done = False
        try:
            while not done:
                observation, _, done, _ = episode.step(policy.act(observation))
        except Exception as e:
            logger.exception(f"{episode.episode_id}: policy/episode error: {e}")
            episode.close("abandoned")
        return episode.record
    finally:
        pool.release(worker, slot)


def run_batch(policy_factory, tasks, k, pool, run_ctx, cfg, on_record=None):
    # Sample-major: a run stopped early yields the same sample count per task instead of full coverage of a prefix.
    jobs = [(task, sample) for sample in range(k) for task in tasks]

    def one(job):
        task, sample = job
        record = run_episode(policy_factory(), task, pool, sample, run_ctx, cfg)
        if on_record:
            on_record(record)
        return record

    with ThreadPoolExecutor(max_workers=max(1, pool.slot_count)) as executor:
        return list(executor.map(one, jobs))


def write_rollouts(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            slim = {k: v for k, v in r.items() if k != "turns"}
            slim["turns"] = [{"clean": t["clean"], "errors": t["errors"]} for t in r.get("turns", [])]
            f.write(json.dumps(slim) + "\n")


def summarize(records):
    final = [r for r in records if r.get("reward_status") == "final"]
    successes = [r for r in records if r.get("success")]
    turns = [t for r in records for t in r.get("turns", [])]
    return {
        "episodes": len(records), "final": len(final), "deferred": sum(1 for r in records if r.get("reward_status") == "deferred"),
        "success_rate": round(len(successes) / len(records), 4) if records else 0.0,
        "mean_reward": round(sum(r["reward"] for r in final) / len(final), 4) if final else 0.0,
        "done_rate": round(sum(1 for r in records if r.get("terminal_reason") == "done") / len(records), 4) if records else 0.0,
        "parse_failure_rate": round(sum(1 for t in turns if not t.get("clean")) / len(turns), 4) if turns else 0.0,
        "mean_steps_success": round(sum(r["steps"] for r in successes) / len(successes), 2) if successes else 0.0,
    }
