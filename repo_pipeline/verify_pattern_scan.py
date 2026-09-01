"""Measures how much of the corpus demonstrates act -> persist -> re-check, and whether done() follows evidence."""

import argparse
import json
import re
import statistics
from pathlib import Path

VERIFY_TEXT = re.compile(
    r"\b(verif|confirm(ing|ed)?|double.?check|re-?open|reopen|reload|refresh|check(ing)? (that|it|the)|"
    r"make sure|ensure it|persisted|saved correctly|still there|appears in the list)\b", re.I)
SAVE_TEXT = re.compile(r"\b(save|saved|saving|submit|apply|confirm)\b", re.I)
PAGINATION_TEXT = re.compile(r"\b(page \d|next page|pagination|paginat|load more|show more)\b", re.I)
RELOGIN_MIN_GAP = 5
LONG_ROUTE_STEPS = 25


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def scan(corpus: Path) -> dict:
    lengths = []
    counts = dict(total=0, verify_any=0, verify_after_save=0, mid_navigate=0, pagination=0,
                  relogin=0, long_route=0, done_total=0, done_after_verify=0, wall_demos=0)
    for traj_dir in sorted(corpus.glob("**/task_*")):
        traj = _load(traj_dir / "trajectory.json")
        if not traj.get("success"):
            continue
        meta = _load(traj_dir / "metadata.json")
        if meta.get("excluded_reason"):
            continue
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
        counts["pagination"] += bool(PAGINATION_TEXT.search(
            " ".join(thoughts) + " " + (traj.get("instruction") or "")))

        save_at = next((i for i, t in enumerate(thoughts) if SAVE_TEXT.search(t)), None)
        if save_at is not None and any(VERIFY_TEXT.search(t) for t in thoughts[save_at + 1:]):
            counts["verify_after_save"] += 1

        password_at = [i for i, step in enumerate(steps)
                       if any(action.get("type") == "type" and "secret:password" in str(action.get("text", ""))
                              for action in step.get("actions", []))]
        if len(password_at) >= 2 and password_at[-1] - password_at[0] >= RELOGIN_MIN_GAP:
            counts["relogin"] += 1

        for i, step_types in enumerate(types):
            if "done" in step_types:
                counts["done_total"] += 1
                if i > 0 and VERIFY_TEXT.search(thoughts[i - 1]):
                    counts["done_after_verify"] += 1
                break
    lengths.sort()
    counts["median_steps"] = statistics.median(lengths) if lengths else 0
    counts["p90_steps"] = lengths[int(0.9 * len(lengths))] if lengths else 0
    counts["max_steps"] = lengths[-1] if lengths else 0
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", default="trajectories", type=Path)
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
            ("long_route", f">= {LONG_ROUTE_STEPS} steps", total),
            ("pagination", "pagination language", total),
            ("relogin", "two password entries far apart", total),
    ):
        print(f"  {label:38s} {counts[key]:5d}  ({100 * counts[key] / denom:.1f}%)")
    print(f"done steps: {counts['done_total']} ({counts['wall_demos']} wall demos), "
          f"preceded by a verification thought: {counts['done_after_verify']} "
          f"({100 * counts['done_after_verify'] / done_total:.1f}%)")


if __name__ == "__main__":
    main()
