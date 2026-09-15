"""Scripted (no-LLM, no-Docker, no-network) check of the corpus audit's outcome metrics and health gates."""

import json
import logging
import sys
import tempfile
from pathlib import Path

from cua.pipeline import corpus_audit

logger = logging.getLogger(__name__)


def _step(index: int, thought: str, actions: list[dict]) -> dict:
    return {"step": index, "screenshot": f"step_{index:03d}_screenshot.png", "thought": thought, "actions": actions}


def _click(index: int, thought: str) -> dict:
    return _step(index, thought, [{"type": "click", "x": 100 + index, "y": 200, "button": "left"}])


def _done(index: int, thought: str) -> dict:
    return _step(index, thought, [{"type": "done"}])


RECHECK_STEPS = [
    _click(1, "I open the notification settings to find the Email toggle."),
    _click(2, "I uncheck Email notifications and save the form."),
    _step(
        3,
        "I'll reload the settings page to confirm the change persisted.",
        [{"type": "navigate", "url": "http://app.test/settings"}],
    ),
    _done(4, "The settings page now shows Email notifications unchecked, so the change persisted."),
]
NO_EVIDENCE_STEPS = [
    _click(1, "I open the projects list."),
    _click(2, "I create the project named Roadmap."),
    _done(3, "I've finished the task."),
]
EVIDENCE_ONLY_STEPS = [
    _click(1, "I add the item to the cart."),
    _done(2, "The cart badge displays 1 item, which is the goal."),
]

# Every case is data: the recorded steps, the metadata the corpus carries, and whether the dir trains.
CASES = [
    {"name": "task_recheck", "steps": RECHECK_STEPS, "success": True, "meta": {"thought_source": "teacher_outcome"}},
    {"name": "task_no_evidence", "steps": NO_EVIDENCE_STEPS, "success": True, "meta": {"thought_source": "teacher"}},
    {
        "name": "task_unverified",
        "steps": EVIDENCE_ONLY_STEPS,
        "success": False,
        "meta": {"thought_source": "teacher_outcome"},
    },
    {
        "name": "task_qa_voice",
        "steps": EVIDENCE_ONLY_STEPS,
        "success": True,
        "meta": {"instruction_style": "qa_verify"},
    },
    {"name": "task_qa_captured", "steps": NO_EVIDENCE_STEPS, "success": True, "meta": {"task_style": "qa_verify"}},
]
EXPECTED_METRICS = {
    "trainable_trajectories": 4,
    "done_steps": 4,
    "done_with_evidence": 2,
    "done_after_recheck": 1,
    "done_evidence_share": 0.5,
    "done_after_recheck_share": 0.25,
    "qa_verify_trajectories": 2,
    "outcome_relabeled": 1,
    "outcome_relabeled_share": 0.25,
}


def _write_corpus(root: Path) -> None:
    for case in CASES:
        traj_dir = root / "repo_smoke" / case["name"]
        traj_dir.mkdir(parents=True)
        trajectory = {
            "task_id": case["name"],
            "instruction": "Do the thing",
            "start_url": "http://app.test/",
            "steps": case["steps"],
            "total_steps": len(case["steps"]),
            "success": case["success"],
        }
        (traj_dir / "trajectory.json").write_text(json.dumps(trajectory))
        (traj_dir / "metadata.json").write_text(json.dumps({"capture_source": "agent", **case["meta"]}))
        for step in case["steps"]:
            (traj_dir / step["screenshot"]).write_bytes(b"")


def check_predicates(failures: list[str]) -> None:
    checks = [
        ("done_after_verify on a re-check run", corpus_audit.done_after_verify(RECHECK_STEPS), True),
        ("done_after_verify without a verify thought", corpus_audit.done_after_verify(NO_EVIDENCE_STEPS), False),
        ("navigate_after_save on a re-check run", corpus_audit.navigate_after_save(RECHECK_STEPS), True),
        ("navigate_after_save without a navigate", corpus_audit.navigate_after_save(NO_EVIDENCE_STEPS), False),
        ("demonstrates_recheck union", corpus_audit.demonstrates_recheck(RECHECK_STEPS), True),
        ("done_cites_evidence with 'displays'", corpus_audit.done_cites_evidence(EVIDENCE_ONLY_STEPS), True),
        ("done_cites_evidence on a bare finish", corpus_audit.done_cites_evidence(NO_EVIDENCE_STEPS), False),
        ("done_cites_evidence with no done step", corpus_audit.done_cites_evidence(RECHECK_STEPS[:2]), False),
    ]
    for name, got, want in checks:
        if got != want:
            failures.append(f"{name}: expected {want!r}, got {got!r}")


def check_metrics(root: Path, failures: list[str]) -> None:
    trainable = [d for d in sorted(root.glob("**/task_*")) if corpus_audit.is_trainable(d)]
    metrics = corpus_audit.corpus_metrics(trainable)
    for key, want in EXPECTED_METRICS.items():
        if metrics.get(key) != want:
            failures.append(f"metric {key}: expected {want!r}, got {metrics.get(key)!r}")


def check_partial_stats(failures: list[str]) -> None:
    thresholds = {
        "min_trainable_trajectories": 500,
        "min_done_coverage": 0.95,
        "max_teacher_raw_trajectories": 0,
        "max_meandering_example_share": 0.25,
        "max_group_example_share": 0.15,
        "min_example_action_share": {"click": 0.2},
        "min_done_evidence_share": 0.8,
        "max_grounding_example_share": 0.25,
        "min_outcome_relabeled_share": 0.9,
    }
    only_audit = corpus_audit.corpus_health_violations(
        {"done_evidence_share": 0.5, "done_after_recheck_share": 0.1}, thresholds
    )
    if len(only_audit) != 1 or "outcome-relabel" not in only_audit[0]:
        failures.append(f"partial stats: expected one outcome-relabel violation, got {only_audit!r}")
    provenance = corpus_audit.corpus_health_violations({"outcome_relabeled_share": 0.0}, thresholds)
    if len(provenance) != 1 or "make outcome-relabel APPLY=1" not in provenance[0]:
        failures.append(f"provenance stat: expected one violation naming the make target, got {provenance!r}")
    grounding = corpus_audit.corpus_health_violations({"grounding_example_share": 0.3}, thresholds)
    if len(grounding) != 1 or "grounding" not in grounding[0]:
        failures.append(f"grounding stat: expected one violation, got {grounding!r}")
    if corpus_audit.corpus_health_violations({}, thresholds):
        failures.append("empty stats must produce no violations")
    passing = corpus_audit.corpus_health_violations(
        {"done_evidence_share": 0.9, "grounding_example_share": 0.1, "outcome_relabeled_share": 0.95}, thresholds
    )
    if passing:
        failures.append(f"passing stats produced violations: {passing!r}")


def check_exit_codes(root: Path, failures: list[str]) -> None:
    strict = root / "strict.json"
    strict.write_text(json.dumps({"min_done_evidence_share": 0.8}))
    provenance = root / "provenance.json"
    provenance.write_text(json.dumps({"min_outcome_relabeled_share": 0.9}))
    lenient = root / "lenient.json"
    lenient.write_text(json.dumps({"min_done_evidence_share": 0.1, "min_outcome_relabeled_share": 0.2}))
    corpus = root
    cases = [
        ("gate violation exits 1", [str(corpus), "--health-config", str(strict)], 1),
        ("provenance gate violation exits 1", [str(corpus), "--health-config", str(provenance)], 1),
        ("gate pass exits 0", [str(corpus), "--health-config", str(lenient)], 0),
        (
            "--skip-health-gate reports but exits 0",
            [str(corpus), "--health-config", str(strict), "--skip-health-gate"],
            0,
        ),
        ("missing thresholds file skips the gate", [str(corpus), "--health-config", str(root / "absent.json")], 0),
    ]
    for name, argv, want in cases:
        got = corpus_audit.main(argv)
        if got != want:
            failures.append(f"{name}: expected exit {want}, got {got}")


def run_smoke() -> bool:
    failures: list[str] = []
    check_predicates(failures)
    check_partial_stats(failures)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_corpus(root)
        check_metrics(root, failures)
        check_exit_codes(root, failures)
    for failure in failures:
        logger.error(f"  {failure}")
    return not failures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ok = run_smoke()
    logger.info(f"Corpus audit smoke test {'PASSED' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)
