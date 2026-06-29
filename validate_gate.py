"""Empirical proof for the bootable-app gate. Runs it (no LLM, one gh probe/repo) over:
  1. proven-good repos (status=completed, trajectory_count>0) -> PRECISION FLOOR: all must ACCEPT,
  2. a random sample of LLM-rejected backlog (rejected_score/rejected_analyze) -> RECALL: how much
     junk the gate would have killed at stage 2, before any LLM spend,
then projects the saved shallow_score/deep_analyze dollars.
"""
import concurrent.futures
import glob
import json
import random
import sqlite3
import sys

from repo_pipeline.bootable_gate import ProbeError, gate_repo

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SCORE_COST = 184.19 / 4181   # $/repo measured from the agent_calls ledger
ANALYZE_COST = 91.50 / 525

con = sqlite3.connect("pipeline_data/registry.db")
con.row_factory = sqlite3.Row


def capture_source(full_name):
    metas = glob.glob(f"trajectories/repo_{full_name.replace('/', '_')}/task_*/metadata.json")
    for m in metas:
        try:
            return json.load(open(m)).get("capture_source")
        except Exception:
            continue
    return None


completed = [r["full_name"] for r in con.execute(
    "SELECT full_name FROM repos WHERE status='completed' AND trajectory_count>0").fetchall()]
# The gate is an AGENT-mode intake filter: agent-captured repos MUST pass it; legacy replay test
# suites (no capture_source) are exactly what the agent-mode pivot stops admitting.
proven = [c for c in completed if capture_source(c) == "agent"]
legacy = [c for c in completed if capture_source(c) != "agent"]
rej_score = [r["full_name"] for r in con.execute(
    "SELECT full_name FROM repos WHERE status='rejected_score'").fetchall()]
rej_analyze = [r["full_name"] for r in con.execute(
    "SELECT full_name FROM repos WHERE status='rejected_analyze'").fetchall()]

random.seed(1781409272)
rej_sample = random.sample(rej_score, min(SAMPLE, len(rej_score))) + \
             random.sample(rej_analyze, min(SAMPLE // 3, len(rej_analyze)))


def run(name):
    try:
        accept, reason = gate_repo(name)
        return name, accept, reason, None
    except ProbeError as e:
        return name, None, None, str(e)[:80]


def evaluate(names, label):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run, names))
    ok = [r for r in results if r[1] is True]
    rej = [r for r in results if r[1] is False]
    err = [r for r in results if r[1] is None]
    print(f"\n=== {label}: {len(names)} repos ===")
    print(f"  ACCEPT={len(ok)}  REJECT={len(rej)}  probe_error={len(err)}")
    return results, ok, rej, err


# 1. Precision floor — AGENT-mode proven-good must ALL accept
pr_results, pr_ok, pr_rej, pr_err = evaluate(proven, "AGENT-MODE PROVEN-GOOD (must all ACCEPT)")
if pr_rej:
    print("  !! FALSE NEGATIVES (gate would wrongly reject agent-mode proven-good repos):")
    for name, _, reason, _ in pr_rej:
        print(f"     {name}: {reason}")
else:
    print("  PRECISION FLOOR PASS: zero false negatives on agent-mode proven-good")
for name, _, _, e in pr_err:
    print(f"     probe error: {name}: {e}")

# 1b. Legacy replay proven-good — rejection here is INTENDED (agent-mode pivot), shown for honesty
lg_results, lg_ok, lg_rej, lg_err = evaluate(legacy, "LEGACY REPLAY proven-good (rejection is intended)")
print(f"  legacy accepted (also bootable apps): {[r[0] for r in lg_ok]}")
print(f"  legacy rejected (test suites, no bootable app — intended exclusion): {[r[0] for r in lg_rej]}")

# 2. Recall on junk backlog
bk_results, bk_ok, bk_rej, bk_err = evaluate(rej_sample, "LLM-REJECTED BACKLOG SAMPLE (should mostly REJECT)")
scored_n = sum(1 for r in bk_results if r[0] in set(rej_score))
graded = [r for r in bk_results if r[1] is not None]
reject_rate = len(bk_rej) / len(graded) if graded else 0
print(f"  gate REJECT rate on junk: {reject_rate:.0%}  ({len(bk_rej)}/{len(graded)} gradable)")
print("  sample reject reasons:")
from collections import Counter
for reason, n in Counter(r[2] for r in bk_rej).most_common(6):
    print(f"     {n:>3}  {reason}")
print("  sample of what the gate would still ACCEPT (passes to LLM):")
for name, _, reason, _ in bk_ok[:8]:
    print(f"     {name}: {reason}")

# 3. Projected savings
print("\n=== PROJECTED SAVINGS (applying gate before LLM stages) ===")
saved_score = reject_rate * len(rej_score) * SCORE_COST
print(f"  Of {len(rej_score):,} rejected_score repos, gate would pre-reject ~{reject_rate:.0%} "
      f"= ~{int(reject_rate*len(rej_score)):,} repos")
print(f"  shallow_score $ avoided on those: ~${saved_score:,.0f}  (@ ${SCORE_COST:.3f}/repo)")
print(f"  + every future discovery batch is pre-filtered, so the flood never reaches the LLM again.")
print(f"  gate cost: ~$0 (no LLM; 1 gh API call/repo, ~0.5s)")
