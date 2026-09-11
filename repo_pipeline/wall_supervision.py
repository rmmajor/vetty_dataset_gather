"""Terminal supervision for blocked runs: turn recorded bot-wall failures into wall demos.

The corpus has no examples of what to DO on a captcha/WAF/403 page, and sft-v1 answers those pages
with broken format. The raw material already exists — failed/partial runs that hit a wall — but they
end mid-flail with no stop action and are gated out as unverified. This pass trims such a run to its
pre-wall prefix plus the step that recognizes the wall, appends the intended terminal behavior
(handoff(reason=captcha) for interactive checks, done() with an honest report for hard blocks), and
marks it success=true + metadata.wall_demo=true. The untrimmed original is kept as
trajectory_prewall.json; the old verdict stays in capture_meta untouched.

  python -m repo_pipeline.wall_supervision trajectories/ --dry-run
  python -m repo_pipeline.wall_supervision trajectories/ --apply [--max-examples 150]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from repo_pipeline.student_eval import BLOCK_SIGNATURES

logger = logging.getLogger(__name__)

# Interactive checks a human could pass get a handoff; hard blocks get an honest done() report.
HANDOFF_KINDS = {"captcha", "human_verification", "cloudflare"}
# A wall that only appears deep into a run leaves too much unverified prefix to train as success.
WALL_MAX_PREFIX = 8
# A trimmed demo is purposeful by construction: arrive, recognize the wall, stop.
WALL_DEMO_EFFICIENCY = "acceptable"

DONE_THOUGHT = (
    "The page is blocking automated access ({kind}) and the task cannot proceed here. "
    "I stop and report the site as blocked rather than keep clicking."
)
HANDOFF_THOUGHT = (
    "The page presents a {kind} check that I must not attempt myself. I hand control back for a human to clear it."
)


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _match_block(text: str) -> str | None:
    for name, pattern in BLOCK_SIGNATURES:
        if pattern.search(text or ""):
            return name
    return None


def analyze(traj_dir: Path) -> dict | None:
    """Wall-demo plan for one dir, or None when it is not a blocked-run candidate."""
    traj = _load(traj_dir / "trajectory.json")
    meta = _load(traj_dir / "metadata.json")
    cmeta = _load(traj_dir / "capture_meta.json")
    if traj.get("success") or meta.get("wall_demo") or meta.get("relabeled_from"):
        return None
    if meta.get("excluded_reason") or cmeta.get("excluded_reason"):
        return None
    if (meta.get("capture_source") or "agent") != "agent":
        return None
    if cmeta.get("agent_outcome") not in ("failed", "partial"):
        return None
    steps = traj.get("steps", [])
    if not steps:
        return None

    onset = kind = None
    for i, step in enumerate(steps):
        kind = _match_block(step.get("thought", ""))
        if kind:
            onset = i
            break
    if onset is None:
        kind = _match_block(
            (cmeta.get("agent_summary") or "") + " " + ((cmeta.get("verifier") or {}).get("reasoning") or "")
        )
        if kind is None:
            return None
    if onset is not None and onset > WALL_MAX_PREFIX:
        return {"dir": traj_dir, "skip": f"wall at step {onset + 1} — prefix too long"}

    kept = steps[: onset + 1] if onset is not None else steps[:2]
    verb = "handoff" if kind in HANDOFF_KINDS else "done"
    return {
        "dir": traj_dir,
        "kind": kind,
        "onset": onset,
        "kept": len(kept),
        "verb": verb,
        "domain": meta.get("website") or meta.get("domain") or "unknown",
        "examples": len(kept) + 1,
        "trajectory": traj,
        "metadata": meta,
    }


def find_candidates(corpus_dir: Path) -> tuple[list[dict], list[dict]]:
    plans, skipped = [], []
    for traj_dir in sorted(corpus_dir.glob("**/task_*")):
        if not (traj_dir / "trajectory.json").exists():
            continue
        plan = analyze(traj_dir)
        if plan is None:
            continue
        (skipped if "skip" in plan else plans).append(plan)
    return plans, skipped


def select_within_cap(plans: list[dict], max_examples: int) -> list[dict]:
    """Round-robin across domains so no single site dominates the wall-demo mix."""
    by_domain: dict = {}
    for plan in plans:
        by_domain.setdefault(plan["domain"], []).append(plan)
    selected, budget = [], max_examples
    queues = sorted(by_domain.values(), key=lambda q: q[0]["domain"])
    while budget > 0 and any(queues):
        for queue in queues:
            if not queue:
                continue
            plan = queue.pop(0)
            if plan["examples"] > budget:
                continue
            selected.append(plan)
            budget -= plan["examples"]
            if budget <= 0:
                break
    return selected


def apply_plan(plan: dict):
    traj_dir = plan["dir"]
    traj_file = traj_dir / "trajectory.json"
    backup = traj_dir / "trajectory_prewall.json"
    if not backup.exists():
        backup.write_bytes(traj_file.read_bytes())

    traj = plan["trajectory"]
    steps = traj["steps"][: plan["kept"]]
    screenshot = "final_screenshot.png"
    if not (traj_dir / screenshot).exists():
        screenshot = steps[-1].get("screenshot", "")
    template = HANDOFF_THOUGHT if plan["verb"] == "handoff" else DONE_THOUGHT
    action = {"type": "handoff", "reason": "captcha"} if plan["verb"] == "handoff" else {"type": "done"}
    steps.append(
        {
            "step": (steps[-1]["step"] + 1) if steps else 1,
            "screenshot": screenshot,
            "thought": template.format(kind=plan["kind"].replace("_", " ")),
            "actions": [action],
        }
    )
    traj["steps"] = steps
    traj["total_steps"] = len(steps)
    traj["success"] = True
    traj_file.write_text(json.dumps(traj, indent=2))

    # The old verdict graded the UNTRIMMED run; carrying its efficiency over would let a 4-step
    # wall demo be subsampled as "meandering" and inflate the corpus meandering share.
    cmeta_file = traj_dir / "capture_meta.json"
    cmeta = _load(cmeta_file)
    if cmeta.get("verifier"):
        cmeta["verifier"] = {
            **cmeta["verifier"],
            "efficiency": WALL_DEMO_EFFICIENCY,
            "efficiency_before_trim": cmeta["verifier"].get("efficiency"),
        }
        cmeta_file.write_text(json.dumps(cmeta, indent=2))

    meta_file = traj_dir / "metadata.json"
    meta = _load(meta_file)
    meta.update(
        {
            "wall_demo": True,
            "wall_kind": plan["kind"],
            "success": True,
            "total_steps": len(steps),
            "efficiency": WALL_DEMO_EFFICIENCY,
            "efficiency_ratio": None,
        }
    )
    meta_file.write_text(json.dumps(meta, indent=2))
    logger.info(f"  {traj_dir.name}: wall demo ({plan['kind']}, {plan['verb']}, {len(steps)} steps)")


def render(plans: list[dict], skipped: list[dict], corpus_dir: Path) -> str:
    lines = ["| trajectory | kind | onset | kept | terminal | examples |", "| --- | --- | --- | --- | --- | --- |"]
    for p in plans:
        onset = "-" if p["onset"] is None else p["onset"] + 1
        lines.append(
            f"| {p['dir'].relative_to(corpus_dir)} | {p['kind']} | {onset} "
            f"| {p['kept']} | {p['verb']} | {p['examples']} |"
        )
    for p in skipped:
        lines.append(f"| {p['dir'].relative_to(corpus_dir)} | skipped | | | {p['skip']} | |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("trajectories/"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-examples", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true", help="List the plans and exit (default)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    plans, skipped = find_candidates(args.corpus)
    selected = select_within_cap(plans, args.max_examples)
    print(
        f"{len(plans)} wall candidates, {len(skipped)} skipped, "
        f"{len(selected)} selected under --max-examples {args.max_examples} "
        f"({sum(p['examples'] for p in selected)} examples)\n"
    )
    print(render(selected, skipped, args.corpus))

    if args.apply:
        for plan in selected:
            apply_plan(plan)
        print(f"\napplied {len(selected)} wall demos")
    sys.exit(0)


if __name__ == "__main__":
    main()
