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
import sys
from collections import Counter

from repo_pipeline.bootable_gate import ProbeError, gate_repo
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.schemas import FailureCategory

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
    graded = len(acc) + len(rej)
    rate = len(rej) / graded if graded else 0
    print(f"\n=== {label}: {len(names)} probed ===")
    print(f"  ACCEPT={len(acc)}  REJECT={len(rej)}  probe_error={len(err)}  -> reject rate {rate:.0%}")
    for reason, n in Counter(r[2] for r in rej).most_common(6):
        print(f"     {n:>4}  {reason}")
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

# Apply: probe every backlog repo, reject the non-bootable ones.
registry = RepoRegistry()
total_rej = total_err = 0
for bucket_names, stage in ((filtered, 2), (scored, 3)):
    results, _ = sweep(bucket_names, f"APPLY stage-{stage} input ({len(bucket_names)} repos)", workers=8)
    for name, accept, reason in results:
        if accept is False:
            registry.reject(name, stage=stage, reason=f"not a bootable app (re-gate): {reason}",
                            category=FailureCategory.NOT_BOOTABLE)
            total_rej += 1
        elif accept is None:
            total_err += 1
print(f"\nAPPLIED: {total_rej} repos rejected NOT_BOOTABLE, {total_err} probe errors left in place.")
