"""Measures how much of the corpus demonstrates act -> persist -> re-check, and whether done() follows evidence."""

import json
import re
import statistics
from pathlib import Path

from cua.pipeline.corpus.audit import (
    SAVE_TEXT,
    VERIFY_TEXT,
    done_after_verify,
    done_step_index,
    navigate_after_save,
)
from cua.pipeline.corpus.cli import build_parser
from cua.trajectory.io import TrajectoryDir, iter_trajectory_dirs

PAGINATION_TEXT = re.compile(r"\b(page \d|next page|pagination|paginat|load more|show more)\b", re.I)
RELOGIN_MIN_GAP = 5
LONG_ROUTE_STEPS = 25


def scan(corpus: Path) -> dict:
    lengths = []
    counts = dict(
        total=0,
        verify_any=0,
        verify_after_save=0,
        mid_navigate=0,
        navigate_after_save=0,
        pagination=0,
        relogin=0,
        long_route=0,
        done_total=0,
        done_after_verify=0,
        wall_demos=0,
    )
    for traj_dir in iter_trajectory_dirs(corpus):
        recorded = TrajectoryDir.load(traj_dir)
        if not recorded.is_trainable():
            continue
        traj = recorded.trajectory
        meta = recorded.metadata
        steps = traj.get("steps", [])
        if not steps:
            continue
        counts["total"] += 1
        counts["wall_demos"] += bool(meta.get("wall_demo"))
        lengths.append(len(steps))
        counts["long_route"] += len(steps) >= LONG_ROUTE_STEPS

        thoughts = [(step.get("thought") or "") for step in steps]
        types = [[action.get("type") for action in step.get("actions", [])] for step in steps]
        counts["verify_any"] += any(VERIFY_TEXT.search(t) for t in thoughts)
        counts["mid_navigate"] += any("navigate" in t for t in types[1:])
        counts["navigate_after_save"] += navigate_after_save(steps)
        counts["pagination"] += bool(PAGINATION_TEXT.search(" ".join(thoughts) + " " + (traj.get("instruction") or "")))

        save_at = next((i for i, t in enumerate(thoughts) if SAVE_TEXT.search(t)), None)
        if save_at is not None and any(VERIFY_TEXT.search(t) for t in thoughts[save_at + 1 :]):
            counts["verify_after_save"] += 1

        password_at = [
            i
            for i, step in enumerate(steps)
            if any(
                action.get("type") == "type" and "secret:password" in str(action.get("text", ""))
                for action in step.get("actions", [])
            )
        ]
        if len(password_at) >= 2 and password_at[-1] - password_at[0] >= RELOGIN_MIN_GAP:
            counts["relogin"] += 1

        counts["done_total"] += done_step_index(steps) is not None
        counts["done_after_verify"] += done_after_verify(steps)
    lengths.sort()
    counts["median_steps"] = statistics.median(lengths) if lengths else 0
    counts["p90_steps"] = lengths[int(0.9 * len(lengths))] if lengths else 0
    counts["max_steps"] = lengths[-1] if lengths else 0
    return counts


def main() -> None:
    parser = build_parser(__doc__, "corpus")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    counts = scan(args.corpus)
    if args.json:
        print(json.dumps(counts, indent=2))
        return

    total = max(counts["total"], 1)
    done_total = max(counts["done_total"], 1)
    print(f"trainable trajectories scanned: {counts['total']}")
    print(f"  steps: median {counts['median_steps']:.0f}  p90 {counts['p90_steps']}  max {counts['max_steps']}")
    for key, label, denom in (
        ("verify_any", "any verification language", total),
        ("verify_after_save", "verification AFTER a save/submit", total),
        ("mid_navigate", "mid-run navigate (re-open to check)", total),
        ("navigate_after_save", "navigate AFTER a save/submit", total),
        ("long_route", f">= {LONG_ROUTE_STEPS} steps", total),
        ("pagination", "pagination language", total),
        ("relogin", "two password entries far apart", total),
    ):
        print(f"  {label:38s} {counts[key]:5d}  ({100 * counts[key] / denom:.1f}%)")
    print(
        f"done steps: {counts['done_total']} ({counts['wall_demos']} wall demos), "
        f"preceded by a verification thought: {counts['done_after_verify']} "
        f"({100 * counts['done_after_verify'] / done_total:.1f}%)"
    )


if __name__ == "__main__":
    main()
