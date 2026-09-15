"""Terminal supervision for blocked runs: turn recorded bot-wall failures into wall demos."""

import asyncio
import logging
import sys
from pathlib import Path

from cua.eval.student.blocks import BLOCK_SIGNATURES
from cua.pipeline.corpus.cli import build_parser
from cua.pipeline.corpus.pass_base import CorpusPass, Outcome
from cua.trajectory.io import TRAJECTORY_FILE, TrajectoryDir, iter_trajectory_dirs

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


def _match_block(text: str) -> str | None:
    for name, pattern in BLOCK_SIGNATURES:
        if pattern.search(text or ""):
            return name
    return None


def analyze(traj_dir: Path) -> dict | None:
    """Wall-demo plan for one dir, or None when it is not a blocked-run candidate."""
    recorded = TrajectoryDir.load(traj_dir)
    traj = recorded.trajectory
    meta = recorded.metadata
    cmeta = recorded.capture_meta
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
    recorded = TrajectoryDir.load(traj_dir)
    backup = traj_dir / "trajectory_prewall.json"
    if not backup.exists():
        backup.write_bytes((traj_dir / TRAJECTORY_FILE).read_bytes())

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
    recorded.save_trajectory(traj)

    # The old verdict graded the UNTRIMMED run, so carrying its efficiency over would subsample a 4-step demo as meandering.
    cmeta = recorded.capture_meta
    if cmeta.get("verifier"):
        cmeta["verifier"] = {
            **cmeta["verifier"],
            "efficiency": WALL_DEMO_EFFICIENCY,
            "efficiency_before_trim": cmeta["verifier"].get("efficiency"),
        }
        recorded.save_capture_meta()

    recorded.metadata.update(
        {
            "wall_demo": True,
            "wall_kind": plan["kind"],
            "success": True,
            "total_steps": len(steps),
            "efficiency": WALL_DEMO_EFFICIENCY,
            "efficiency_ratio": None,
        }
    )
    recorded.save_metadata()
    logger.info(f"  {traj_dir.name}: wall demo ({plan['kind']}, {plan['verb']}, {len(steps)} steps)")


class WallSupervisionPass(CorpusPass):
    """Trim a blocked run to its pre-wall prefix, append the intended terminal behaviour, and mark it a wall demo."""

    name = "wall_supervision"
    workers = 1

    def __init__(self, max_examples: int):
        self.max_examples = max_examples
        self.plans: list[dict] = []
        self.skipped: list[dict] = []
        self.plan_by_dir: dict = {}

    def select(self, recorded: TrajectoryDir) -> bool:
        plan = analyze(recorded.path)
        return plan is not None and "skip" not in plan

    def candidates(self, corpus: Path, limit: int | None = None) -> list[TrajectoryDir]:
        """Wall candidates in the order the domain round-robin picked them, which is also the order the table prints."""
        found = {recorded.path: recorded for recorded in super().candidates(corpus, limit)}
        self.plans, self.skipped, self.plan_by_dir = [], [], {}
        for traj_dir in iter_trajectory_dirs(corpus):
            plan = analyze(traj_dir)
            if plan is None:
                continue
            if "skip" in plan:
                self.skipped.append(plan)
                continue
            self.plans.append(plan)
            self.plan_by_dir[traj_dir] = plan
        picked = select_within_cap(self.plans, self.max_examples)
        return [found[plan["dir"]] for plan in picked if plan["dir"] in found]

    async def apply(self, recorded: TrajectoryDir) -> Outcome:
        plan = self.plan_by_dir[recorded.path]
        apply_plan(plan)
        return Outcome(changed=True, note="wall_demo", detail=plan["kind"])

    def summary(self, corpus: Path, candidates: list) -> str:
        examples = sum(self.plan_by_dir[recorded.path]["examples"] for recorded in candidates)
        return (
            f"{len(self.plans)} wall candidates, {len(self.skipped)} skipped, "
            f"{len(candidates)} selected under --max-examples {self.max_examples} ({examples} examples)\n"
        )

    def describe(self, corpus: Path, recorded: TrajectoryDir) -> str:
        plan = self.plan_by_dir[recorded.path]
        onset = "-" if plan["onset"] is None else plan["onset"] + 1
        return (
            f"| {plan['dir'].relative_to(corpus)} | {plan['kind']} | {onset} "
            f"| {plan['kept']} | {plan['verb']} | {plan['examples']} |"
        )

    def print_dry_run(self, corpus: Path, candidates: list) -> None:
        print(self.summary(corpus, candidates))
        print(self.render(corpus, candidates))

    def render(self, corpus: Path, candidates: list) -> str:
        lines = ["| trajectory | kind | onset | kept | terminal | examples |", "| --- | --- | --- | --- | --- | --- |"]
        lines.extend(self.describe(corpus, recorded) for recorded in candidates)
        for plan in self.skipped:
            lines.append(f"| {plan['dir'].relative_to(corpus)} | skipped | | | {plan['skip']} | |")
        return "\n".join(lines)


def main():
    parser = build_parser(__doc__, "corpus", "apply")
    parser.add_argument("--max-examples", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true", help="List the plans and exit (default)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # This pass prints its plan table in both modes, so the listing runs before any apply rather than instead of it.
    corpus_pass = WallSupervisionPass(args.max_examples)
    selected = corpus_pass.candidates(args.corpus)
    corpus_pass.print_dry_run(args.corpus, selected)
    if args.apply:
        asyncio.run(corpus_pass.run(args.corpus, candidates=selected))
        print(f"\napplied {len(selected)} wall demos")
    sys.exit(0)


if __name__ == "__main__":
    main()
