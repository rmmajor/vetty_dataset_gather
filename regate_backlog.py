"""Re-gate the existing backlog through the no-LLM bootable-app gate. Targets `filtered` (pending
stage 3) and `scored` (pending stage 4) — the repos that would otherwise burn LLM scoring/analysis.

Default: DRY-RUN over a sample, projecting the accept/reject split + LLM dollars saved.
  python regate_backlog.py [sample_per_bucket]
Apply for real (probes every backlog repo, rejects non-bootable -> NOT_BOOTABLE):
  python regate_backlog.py --apply
"""
import concurrent.futures
import random
import sqlite3
import subprocess
import sys
import time
from collections import Counter

from repo_pipeline.bootable_gate import ProbeError, gate_repo, is_hard_reject
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.schemas import FailureCategory

# Conservative (option B): only HARD rejects drop; ambiguous "no framework" repos pass to the LLM.
AGGRESSIVE = "--aggressive" in sys.argv

SCORE_COST = 184.19 / 4181
ANALYZE_COST = 91.50 / 525
APPLY = "--apply" in sys.argv
nums = [a for a in sys.argv[1:] if a.isdigit()]
SAMPLE = int(nums[0]) if nums else 400

con = sqlite3.connect("pipeline_data/registry.db")
con.row_factory = sqlite3.Row
filtered = [r["full_name"] for r in con.execute("SELECT full_name FROM repos WHERE status='filtered'")]
scored = [r["full_name"] for r in con.execute("SELECT full_name FROM repos WHERE status='scored'")]
con.close()


def probe(name):
    try:
        accept, reason = gate_repo(name)
        return name, accept, reason
    except ProbeError as e:
        return name, None, str(e)[:120]


def sweep(names, label, workers=8):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(probe, names))
    acc = [r for r in results if r[1] is True]
    rej = [r for r in results if r[1] is False]
    err = [r for r in results if r[1] is None]
    hard = [r for r in rej if is_hard_reject(r[2]) or AGGRESSIVE]
    soft = [r for r in rej if not (is_hard_reject(r[2]) or AGGRESSIVE)]
    graded = len(acc) + len(rej)
    rate = len(hard) / graded if graded else 0   # only HARD rejects are dropped
    print(f"\n=== {label}: {len(names)} probed ===")
    print(f"  KEEP={len(acc) + len(soft)} (accept {len(acc)} + soft-defer {len(soft)})  "
          f"HARD-REJECT={len(hard)}  probe_error={len(err)}  -> drop rate {rate:.0%}")
    for reason, n in Counter(r[2] for r in rej).most_common(6):
        tag = "DROP" if (is_hard_reject(reason) or AGGRESSIVE) else "keep"
        print(f"     {n:>4}  [{tag}] {reason}")
    return results, rate


if not APPLY:
    random.seed(1781409272)
    f_s = random.sample(filtered, min(SAMPLE, len(filtered)))
    s_s = random.sample(scored, min(SAMPLE, len(scored)))
    _, f_rate = sweep(f_s, f"FILTERED sample (pending stage 3)")
    _, s_rate = sweep(s_s, f"SCORED sample (pending stage 4)")
    print("\n=== PROJECTED BACKLOG SAVINGS (full apply) ===")
    f_saved = f_rate * len(filtered) * SCORE_COST
    s_saved = s_rate * len(scored) * ANALYZE_COST
    print(f"  filtered: ~{f_rate:.0%} of {len(filtered):,} pre-rejected -> ~${f_saved:,.0f} shallow_score avoided")
    print(f"  scored:   ~{s_rate:.0%} of {len(scored):,} pre-rejected -> ~${s_saved:,.0f} deep_analyze avoided")
    print(f"  total projected LLM $ avoided on existing backlog: ~${f_saved + s_saved:,.0f}")
    print(f"\n  DRY RUN — re-run with --apply to reject the non-bootable backlog for real.")
    sys.exit(0)

# Apply: probe every backlog repo (budget-aware), reject the non-bootable ones.
# The GraphQL probe costs ~1 point/repo against a 5000/hr budget; 11k repos spans multiple windows.
# Process in chunks at low concurrency (dodge secondary rate limits) and sleep until reset when the
# budget runs low, committing rejects per chunk so progress survives and re-runs are idempotent.
CHUNK = 150
APPLY_WORKERS = 4


def graphql_budget():
    try:
        out = subprocess.run(["gh", "api", "rate_limit", "--jq",
                              ".resources.graphql.remaining,.resources.graphql.reset"],
                             capture_output=True, text=True, timeout=30)
        rem, reset = out.stdout.split()
        return int(rem), int(reset)
    except Exception:
        return None, None


def wait_for_budget(need):
    rem, reset = graphql_budget()
    if rem is not None and rem < need:
        wait = max(reset - int(time.time()) + 5, 5)
        print(f"  graphql budget {rem} < {need}; sleeping {wait}s until reset", flush=True)
        time.sleep(wait)


registry = RepoRegistry()
total_rej = total_err = 0
for bucket_names, stage in ((filtered, 2), (scored, 3)):
    print(f"\n=== APPLY stage-{stage}: {len(bucket_names)} repos (chunks of {CHUNK}) ===", flush=True)
    for i in range(0, len(bucket_names), CHUNK):
        chunk = bucket_names[i:i + CHUNK]
        wait_for_budget(len(chunk) + 50)
        with concurrent.futures.ThreadPoolExecutor(max_workers=APPLY_WORKERS) as ex:
            results = list(ex.map(probe, chunk))
        for name, accept, reason in results:
            if accept is False and (is_hard_reject(reason) or AGGRESSIVE):
                registry.reject(name, stage=stage, reason=f"not a bootable app (re-gate): {reason}",
                                category=FailureCategory.NOT_BOOTABLE)
                total_rej += 1
            elif accept is None:
                total_err += 1
        print(f"  stage-{stage}: {min(i + CHUNK, len(bucket_names))}/{len(bucket_names)} probed; "
              f"{total_rej} rejected, {total_err} probe-errors", flush=True)
print(f"\nAPPLIED: {total_rej} repos rejected NOT_BOOTABLE, {total_err} probe errors left in place.", flush=True)
