"""One-time backfill: recover the structured data the old reject() discarded.

For every already-rejected repo, parse its free-text filter_reason into:
  * a failure_category, so the dashboard's quality-vs-infra split and `--requeue-infra` work on
    historical rejections (not just ones produced going forward), and
  * the scores behind the rejection — the stage-3 gate dims (app / run / interact, which are
    exactly what the gate is tuned on) and the stage-4 deep_score + test-count / self-contained —
    so the dashboard's score-distribution charts have data immediately.

The other five stage-3 dimensions were never stored in the reason string, so they stay sparse
until repos are re-scored under the new reject() that persists full scores. Parse-only, no LLM,
idempotent: re-running rewrites the same values and never overwrites scores that already exist.
"""

import argparse
import json
import re
from pathlib import Path

from repo_pipeline.registry import RepoRegistry
from repo_pipeline.schemas import FailureCategory

GATE_RE = re.compile(r"app=(\d+),\s*run=(\d+),\s*interact=(\d+)")
DEEP_RE = re.compile(r"deep score (\d+)/\d+ \(tests: (\d+), self_contained: (\w+)\)")


def categorize(status: str, reason: str):
    r = (reason or "").lower()
    if status == "rejected_filter":
        if "empty repo" in r:
            return FailureCategory.EMPTY_REPO
        if "too large" in r:
            return FailureCategory.TOO_LARGE
        if "fork of" in r or "duplicate" in r:
            return FailureCategory.DEDUP
    if "clone failed" in r:
        return FailureCategory.CLONE_FAILED
    if "claude code returned an error" in r:
        return FailureCategory.SDK_ERROR
    if "control request timeout" in r or "timeout" in r:
        return FailureCategory.CONTROL_TIMEOUT
    if "failed to parse" in r:
        return FailureCategory.PARSE_FAILED
    if "copy results failed" in r:
        return FailureCategory.SANDBOX_ERROR
    if "no test files" in r:
        return FailureCategory.NO_TESTS
    if "no usable capture" in r or "no valid trajectories" in r:
        return FailureCategory.NO_CAPTURE
    # quality gates: stage-3 app/run/interact gate, stage-4 deep score, the old stage-5 pass-rate gate
    if "not a runnable" in r or "deep score" in r or "low pass rate" in r:
        return FailureCategory.QUALITY_GATE
    return None


def backfill(registry: RepoRegistry, dry_run: bool = False) -> dict:
    rows = registry.conn.execute(
        "SELECT full_name, status, filter_reason, shallow_scores, deep_score "
        "FROM repos WHERE status LIKE 'rejected_%'"
    ).fetchall()

    counts = {"categorized": 0, "shallow_scores": 0, "deep_score": 0}
    for row in rows:
        name, status, reason = row["full_name"], row["status"], row["filter_reason"]
        updates = {}

        category = categorize(status, reason)
        if category:
            updates["failure_category"] = category.value
            counts["categorized"] += 1

        if status == "rejected_score" and not row["shallow_scores"]:
            m = GATE_RE.search(reason or "")
            if m:
                updates["shallow_scores"] = json.dumps({
                    "application_vs_library": int(m.group(1)),
                    "runnability": int(m.group(2)),
                    "interaction_richness": int(m.group(3)),
                })
                counts["shallow_scores"] += 1

        if status == "rejected_analyze" and row["deep_score"] is None:
            m = DEEP_RE.search(reason or "")
            if m:
                updates["deep_score"] = int(m.group(1))
                updates["deep_analysis"] = json.dumps({
                    "test_case_count": int(m.group(2)),
                    "is_self_contained": m.group(3) == "True",
                })
                counts["deep_score"] += 1

        if updates and not dry_run:
            assignments = ", ".join(f"{k} = ?" for k in updates)
            registry.conn.execute(
                f"UPDATE repos SET {assignments} WHERE full_name = ?",
                list(updates.values()) + [name],
            )

    if not dry_run:
        registry.conn.commit()
    return counts


def main():
    parser = argparse.ArgumentParser(description="Backfill failure_category + scores from rejection reason strings")
    parser.add_argument("--data-dir", default="pipeline_data/", help="Pipeline data directory")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing")
    args = parser.parse_args()

    registry = RepoRegistry(str(Path(args.data_dir) / "registry.db"))
    try:
        counts = backfill(registry, dry_run=args.dry_run)
    finally:
        registry.close()

    tag = " (dry-run)" if args.dry_run else ""
    print(
        f"Backfill{tag}: categorized {counts['categorized']}, "
        f"shallow_scores {counts['shallow_scores']}, deep_score {counts['deep_score']}"
    )


if __name__ == "__main__":
    main()
