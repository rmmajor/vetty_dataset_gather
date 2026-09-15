"""Two offline passes over the no-LLM bootable-app gate: `validate` measures its precision/recall, `regate` re-gates the existing backlog."""

import argparse
import concurrent.futures
import glob
import json
import random
import sqlite3
import subprocess
import sys
import time
from collections import Counter

from cua.pipeline.bootable_gate import ProbeError, gate_repo, is_hard_reject
from cua.pipeline.registry import RepoRegistry
from cua.pipeline.schemas import FailureCategory

DEFAULT_DB = "pipeline_data/registry.db"
DEFAULT_TRAJECTORIES = "trajectories"
# $/repo measured from the agent_calls ledger.
SCORE_COST = 184.19 / 4181
ANALYZE_COST = 91.50 / 525
# Fixed so a re-run samples the same repos and the two passes are comparable.
SAMPLE_SEED = 1781409272
PROBE_ERROR_CHARS = 120
PROBE_WORKERS = 8
# The GraphQL probe costs ~1 point/repo against a 5000/hr budget, so an apply over 11k repos spans several windows.
APPLY_CHUNK = 150
APPLY_WORKERS = 4


def open_readonly(db_path: str) -> sqlite3.Connection:
    """A `mode=ro` connection, so a report cannot write: RepoRegistry's constructor would (WAL pragma, schema, migrations)."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def names_with_status(connection: sqlite3.Connection, status: str) -> list[str]:
    return [row["full_name"] for row in connection.execute("SELECT full_name FROM repos WHERE status=?", (status,))]


def probe(name: str) -> tuple[str, bool | None, str | None]:
    """Gate one repo; a probe error reports accept=None and carries its message in the reason slot, so it is counted apart from an accept or a reject."""
    try:
        accept, reason = gate_repo(name)
        return name, accept, reason
    except ProbeError as e:
        # One truncation for both subcommands: regate's 120 chars, not validate's 80.
        return name, None, str(e)[:PROBE_ERROR_CHARS]


def probe_all(names, workers: int = PROBE_WORKERS) -> list[tuple]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(probe, names))


def capture_source(full_name: str, trajectories: str) -> str | None:
    metas = glob.glob(f"{trajectories}/repo_{full_name.replace('/', '_')}/task_*/metadata.json")
    for m in metas:
        try:
            return json.load(open(m)).get("capture_source")
        except Exception:
            continue
    return None


def evaluate(names, label: str):
    results = probe_all(names)
    ok = [r for r in results if r[1] is True]
    rej = [r for r in results if r[1] is False]
    err = [r for r in results if r[1] is None]
    print(f"\n=== {label}: {len(names)} repos ===")
    print(f"  ACCEPT={len(ok)}  REJECT={len(rej)}  probe_error={len(err)}")
    return results, ok, rej, err


def report_precision_floor(proven, legacy) -> None:
    """Agent-mode proven-good repos must ALL accept; legacy replay suites rejecting here is the agent-mode pivot working, shown for honesty."""
    _, _, pr_rej, pr_err = evaluate(proven, "AGENT-MODE PROVEN-GOOD (must all ACCEPT)")
    if pr_rej:
        print("  !! FALSE NEGATIVES (gate would wrongly reject agent-mode proven-good repos):")
        for name, _, reason in pr_rej:
            print(f"     {name}: {reason}")
    else:
        print("  PRECISION FLOOR PASS: zero false negatives on agent-mode proven-good")
    for name, _, error in pr_err:
        print(f"     probe error: {name}: {error}")

    _, lg_ok, lg_rej, _ = evaluate(legacy, "LEGACY REPLAY proven-good (rejection is intended)")
    print(f"  legacy accepted (also bootable apps): {[r[0] for r in lg_ok]}")
    print(f"  legacy rejected (test suites, no bootable app — intended exclusion): {[r[0] for r in lg_rej]}")


def report_recall(rej_sample, rej_score) -> None:
    """How much of the LLM-rejected backlog the gate would have killed at stage 2, and what that saves."""
    bk_results, bk_ok, bk_rej, _ = evaluate(rej_sample, "LLM-REJECTED BACKLOG SAMPLE (should mostly REJECT)")
    graded = [r for r in bk_results if r[1] is not None]
    reject_rate = len(bk_rej) / len(graded) if graded else 0
    print(f"  gate REJECT rate on junk: {reject_rate:.0%}  ({len(bk_rej)}/{len(graded)} gradable)")
    print("  sample reject reasons:")
    for reason, n in Counter(r[2] for r in bk_rej).most_common(6):
        print(f"     {n:>3}  {reason}")
    print("  sample of what the gate would still ACCEPT (passes to LLM):")
    for name, _, reason in bk_ok[:8]:
        print(f"     {name}: {reason}")

    print("\n=== PROJECTED SAVINGS (applying gate before LLM stages) ===")
    saved_score = reject_rate * len(rej_score) * SCORE_COST
    print(
        f"  Of {len(rej_score):,} rejected_score repos, gate would pre-reject ~{reject_rate:.0%} "
        f"= ~{int(reject_rate * len(rej_score)):,} repos"
    )
    print(f"  shallow_score $ avoided on those: ~${saved_score:,.0f}  (@ ${SCORE_COST:.3f}/repo)")
    print("  + every future discovery batch is pre-filtered, so the flood never reaches the LLM again.")
    print("  gate cost: ~$0 (no LLM; 1 gh API call/repo, ~0.5s)")


def command_validate(args) -> int:
    """Empirical proof for the gate: proven-good repos set the precision floor, a sample of LLM-rejected backlog measures recall, then the saved LLM dollars are projected."""
    connection = open_readonly(args.db)
    completed = [
        row["full_name"]
        for row in connection.execute("SELECT full_name FROM repos WHERE status='completed' AND trajectory_count>0")
    ]
    rej_score = names_with_status(connection, "rejected_score")
    rej_analyze = names_with_status(connection, "rejected_analyze")
    connection.close()

    # The gate is an AGENT-mode intake filter, so agent-captured repos must pass it; legacy replay suites are what the pivot stops admitting.
    proven = [c for c in completed if capture_source(c, args.trajectories) == "agent"]
    legacy = [c for c in completed if capture_source(c, args.trajectories) != "agent"]

    random.seed(SAMPLE_SEED)
    rej_sample = random.sample(rej_score, min(args.sample, len(rej_score))) + random.sample(
        rej_analyze, min(args.sample // 3, len(rej_analyze))
    )

    report_precision_floor(proven, legacy)
    report_recall(rej_sample, rej_score)
    return 0


def drops(reason: str, aggressive: bool) -> bool:
    """Conservative by default: only HARD rejects drop, so ambiguous "no framework" repos still pass to the LLM."""
    return is_hard_reject(reason) or aggressive


def sweep(names, label: str, aggressive: bool, workers: int = PROBE_WORKERS):
    results = probe_all(names, workers)
    acc = [r for r in results if r[1] is True]
    rej = [r for r in results if r[1] is False]
    err = [r for r in results if r[1] is None]
    hard = [r for r in rej if drops(r[2], aggressive)]
    soft = [r for r in rej if not drops(r[2], aggressive)]
    graded = len(acc) + len(rej)
    rate = len(hard) / graded if graded else 0
    print(f"\n=== {label}: {len(names)} probed ===")
    print(
        f"  KEEP={len(acc) + len(soft)} (accept {len(acc)} + soft-defer {len(soft)})  "
        f"HARD-REJECT={len(hard)}  probe_error={len(err)}  -> drop rate {rate:.0%}"
    )
    for reason, n in Counter(r[2] for r in rej).most_common(6):
        tag = "DROP" if drops(reason, aggressive) else "keep"
        print(f"     {n:>4}  [{tag}] {reason}")
    return results, rate


def report_projection(filtered, scored, args) -> int:
    """Dry run: sample each backlog bucket and project the LLM dollars a full apply would avoid."""
    random.seed(SAMPLE_SEED)
    f_s = random.sample(filtered, min(args.sample, len(filtered)))
    s_s = random.sample(scored, min(args.sample, len(scored)))
    _, f_rate = sweep(f_s, "FILTERED sample (pending stage 3)", args.aggressive)
    _, s_rate = sweep(s_s, "SCORED sample (pending stage 4)", args.aggressive)
    print("\n=== PROJECTED BACKLOG SAVINGS (full apply) ===")
    f_saved = f_rate * len(filtered) * SCORE_COST
    s_saved = s_rate * len(scored) * ANALYZE_COST
    print(f"  filtered: ~{f_rate:.0%} of {len(filtered):,} pre-rejected -> ~${f_saved:,.0f} shallow_score avoided")
    print(f"  scored:   ~{s_rate:.0%} of {len(scored):,} pre-rejected -> ~${s_saved:,.0f} deep_analyze avoided")
    print(f"  total projected LLM $ avoided on existing backlog: ~${f_saved + s_saved:,.0f}")
    print("\n  DRY RUN — re-run with --apply to reject the non-bootable backlog for real.")
    return 0


def graphql_budget() -> tuple[int | None, int | None]:
    try:
        out = subprocess.run(
            ["gh", "api", "rate_limit", "--jq", ".resources.graphql.remaining,.resources.graphql.reset"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        rem, reset = out.stdout.split()
        return int(rem), int(reset)
    except Exception:
        return None, None


def wait_for_budget(need: int) -> None:
    rem, reset = graphql_budget()
    if rem is not None and rem < need:
        wait = max(reset - int(time.time()) + 5, 5)
        print(f"  graphql budget {rem} < {need}; sleeping {wait}s until reset", flush=True)
        time.sleep(wait)


def apply_bucket(registry: RepoRegistry, names, stage: int, aggressive: bool, counts: dict) -> None:
    """Probe one backlog bucket in chunks at low concurrency, committing rejects per chunk so progress survives and re-runs are idempotent."""
    print(f"\n=== APPLY stage-{stage}: {len(names)} repos (chunks of {APPLY_CHUNK}) ===", flush=True)
    for i in range(0, len(names), APPLY_CHUNK):
        chunk = names[i : i + APPLY_CHUNK]
        wait_for_budget(len(chunk) + 50)
        for name, accept, reason in probe_all(chunk, APPLY_WORKERS):
            if accept is False and drops(reason, aggressive):
                registry.reject(
                    name,
                    stage=stage,
                    reason=f"not a bootable app (re-gate): {reason}",
                    category=FailureCategory.NOT_BOOTABLE,
                )
                counts["rejected"] += 1
            elif accept is None:
                counts["errors"] += 1
        print(
            f"  stage-{stage}: {min(i + APPLY_CHUNK, len(names))}/{len(names)} probed; "
            f"{counts['rejected']} rejected, {counts['errors']} probe-errors",
            flush=True,
        )


def command_regate(args) -> int:
    """Re-gate `filtered` (pending stage 3) and `scored` (pending stage 4) — the repos that would otherwise burn LLM scoring/analysis."""
    connection = open_readonly(args.db)
    filtered = names_with_status(connection, "filtered")
    scored = names_with_status(connection, "scored")
    connection.close()

    if not args.apply:
        return report_projection(filtered, scored, args)

    registry = RepoRegistry(args.db)
    counts = {"rejected": 0, "errors": 0}
    for names, stage in ((filtered, 2), (scored, 3)):
        apply_bucket(registry, names, stage, args.aggressive, counts)
    print(
        f"\nAPPLIED: {counts['rejected']} repos rejected NOT_BOOTABLE, {counts['errors']} probe errors left in place.",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure and apply the no-LLM bootable-app gate")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"Registry database (default: {DEFAULT_DB})")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Measure the gate's precision floor and recall against the backlog")
    validate.add_argument("sample", nargs="?", type=int, default=150, help="Rejected repos sampled (default: 150)")
    validate.add_argument(
        "--trajectories",
        default=DEFAULT_TRAJECTORIES,
        help=f"Corpus scanned for each repo's capture_source (default: {DEFAULT_TRAJECTORIES})",
    )
    validate.set_defaults(func=command_validate)

    regate = sub.add_parser("regate", help="Re-gate the backlog pending stages 3 and 4")
    regate.add_argument("sample", nargs="?", type=int, default=400, help="Repos sampled per bucket (default: 400)")
    regate.add_argument("--apply", action="store_true", help="Probe every backlog repo and reject the non-bootable")
    regate.add_argument("--aggressive", action="store_true", help="Also drop soft rejects, not only hard ones")
    regate.set_defaults(func=command_regate)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
