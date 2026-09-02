"""Mechanical quality gate over the TRAINABLE corpus.

Checks every trajectory that `train_qwen.py convert` would train on (excluded/unverified/
replay-gated ones are reported but not gated) for recording defects that no LLM judge needs:

  * stuck loops — identical-action runs >= LOOP_RUN_LIMIT (the Booking.com failure class);
  * empty type actions ({type:"type", text:""} — broken recording);
  * coordinates outside the 1280x720 viewport;
  * steps with missing screenshots or empty thoughts;
  * verified agent trajectories lacking a terminal done step.

Meandering goal-completions (efficiency:meandering + success:true) are a VALID, TAGGED state,
not a defect — reported as a [warn] and excludable from training via `convert --exclude-meandering`.

Exit code 1 when any trainable trajectory fails or a corpus_health.json gate is violated — wire into CI / run before every training
export: `python -m repo_pipeline.corpus_audit trajectories/` (or `make audit-corpus`).
"""
# Stdlib-only on purpose: remote_train ships this file alone, so verify_pattern_scan imports the predicates from here.
import argparse
import json
import re
import sys
from pathlib import Path

LOOP_RUN_LIMIT = 3
VIEWPORT = (1280, 720)
HEALTH_CONFIG_NAME = "corpus_health.json"
QA_VERIFY_STYLE = "qa_verify"
OUTCOME_RELABEL_SOURCE = "teacher_outcome"

VERIFY_TEXT = re.compile(
    r"\b(verif|confirm(ing|ed)?|double.?check|re-?open|reopen|reload|refresh|check(ing)? (that|it|the)|"
    r"make sure|ensure it|persisted|saved correctly|still there|appears in the list)\b", re.I)
SAVE_TEXT = re.compile(r"\b(save|saved|saving|submit|apply|confirm)\b", re.I)
# Generous on purpose: a done thought that names anything observed on screen counts as evidence.
DONE_EVIDENCE = re.compile(
    r"\b(shows?|showing|visible|display(s|ing|ed)?|lists?|reads|now (has|contains|includes|in)|I can see|can see|"
    r"confirm(s|ing|ed)?|appears|is present|remains?|still (there|listed|checked|unchecked)|matches)\b", re.I)

# Guarded-mode control/observation actions are deliberate decisions, not page manipulation; they
# must never be read as a stuck loop. A step that is ONLY these yields an empty signature, which the
# run-length check treats as a break.
_CONTROL_TYPES = {"confirm", "handoff", "approval_event"}


def _action_signature(step: dict) -> tuple:
    """Full action identity, INCLUDING scroll deltas and drag paths — progressive scrolls
    share (type, x, y) but are distinct actions, not a stuck loop. Control/observation actions
    (confirm/handoff/approval_event) are excluded so repeated pauses never count as a loop."""
    return tuple(
        (
            a.get("type"), a.get("x"), a.get("y"), a.get("text"), tuple(a.get("keys") or []),
            a.get("scroll_x"), a.get("scroll_y"),
            json.dumps(a.get("path")) if a.get("path") else None,
            a.get("url"),
        )
        for a in step.get("actions", [])
        if a.get("type") not in _CONTROL_TYPES
    )


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def max_identical_run(steps) -> int:
    max_run, run, prev = 0, 0, None
    for step in steps:
        sig = _action_signature(step)
        run = run + 1 if sig == prev and sig else 1
        prev = sig
        max_run = max(max_run, run)
    return max_run


def _thoughts(steps) -> list[str]:
    return [(step.get("thought") or "") for step in steps]


def _has_action(step: dict, action_type: str) -> bool:
    return any(action.get("type") == action_type for action in step.get("actions", []))


def done_step_index(steps) -> int | None:
    """Index of the last step carrying a done action, or None."""
    return next((i for i in range(len(steps) - 1, -1, -1) if _has_action(steps[i], "done")), None)


def done_after_verify(steps) -> bool:
    """The done step is preceded by a thought in verification language."""
    done_at = done_step_index(steps)
    return done_at is not None and done_at > 0 and bool(VERIFY_TEXT.search(_thoughts(steps)[done_at - 1]))


def navigate_after_save(steps) -> bool:
    """A mid-run navigate re-opens a page after an earlier save/submit thought."""
    thoughts = _thoughts(steps)
    save_at = next((i for i, thought in enumerate(thoughts) if SAVE_TEXT.search(thought)), None)
    if save_at is None:
        return False
    return any(_has_action(step, "navigate") for step in steps[save_at + 1:])


def demonstrates_recheck(steps) -> bool:
    """The run visibly re-checks its own result: done after a verify thought, or a re-open after a save."""
    return done_after_verify(steps) or navigate_after_save(steps)


def done_cites_evidence(steps) -> bool:
    """The done thought names something observed on screen rather than just declaring completion."""
    done_at = done_step_index(steps)
    return done_at is not None and bool(DONE_EVIDENCE.search(_thoughts(steps)[done_at]))


def in_qa_verify_voice(metadata: dict) -> bool:
    return QA_VERIFY_STYLE in (metadata.get("task_style"), metadata.get("instruction_style"))


def trainability(trajectory, metadata, capture_meta, dir_name, include_replay=False):
    """Skip reason (excluded|replay_gated|unverified|stuck_loop) or None — the ONE gate convert and this audit share."""
    if metadata.get("excluded_reason") or capture_meta.get("excluded_reason"):
        return "excluded"
    source = metadata.get("capture_source") or ("path_a" if "mind2web" in dir_name else "replay")
    if source == "replay" and not include_replay:
        return "replay_gated"
    if source == "agent" and not trajectory.get("success", True):
        return "unverified"
    if max_identical_run(trajectory.get("steps", [])) >= LOOP_RUN_LIMIT:
        return "stuck_loop"
    return None


# (threshold key, stat key, breach test, message) — a check runs only when the caller supplied its stat.
_HEALTH_CHECKS = (
    ("min_trainable_trajectories", "trainable", lambda actual, limit: actual < limit,
     lambda stats, actual, limit: f"trainable trajectories {actual} < {limit}"),
    ("min_done_coverage", "done_coverage", lambda actual, limit: actual < limit,
     lambda stats, actual, limit: f"done() coverage {actual:.1%} < {limit:.0%} of trainable trajectories"),
    ("max_teacher_raw_trajectories", "teacher_raw", lambda actual, limit: actual > limit,
     lambda stats, actual, limit: f"{actual} trajectories still thought_source=teacher_raw (max {limit}) — run the stage-7 polish"),
    ("max_meandering_example_share", "meandering_example_share", lambda actual, limit: actual > limit,
     lambda stats, actual, limit: f"meandering trajectories supervise {actual:.1%} of examples (max {limit:.0%})"),
    ("max_group_example_share", "max_group_example_share", lambda actual, limit: actual > limit,
     lambda stats, actual, limit: f"group {stats.get('max_group')!r} holds {actual:.1%} of examples (max {limit:.0%})"),
    ("min_done_evidence_share", "done_evidence_share", lambda actual, limit: actual < limit,
     lambda stats, actual, limit: f"done thoughts citing on-screen evidence {actual:.1%} < {limit:.0%} — expected on an "
                                  f"unrelabeled corpus; run the outcome-relabel pass so done() states what the screen shows"),
    ("max_grounding_example_share", "grounding_example_share", lambda actual, limit: actual > limit,
     lambda stats, actual, limit: f"grounding examples are {actual:.1%} of the dataset (max {limit:.0%})"),
    ("min_outcome_relabeled_share", "outcome_relabeled_share", lambda actual, limit: actual < limit,
     lambda stats, actual, limit: f"only {actual:.1%} of trainable trajectories carry thought_source={OUTCOME_RELABEL_SOURCE} "
                                  f"(min {limit:.0%}) — run `make outcome-relabel APPLY=1`"),
)


def corpus_health_violations(stats: dict, thresholds: dict) -> list[str]:
    """Threshold breaches over whichever corpus-level stats the caller computed; empty list means releasable."""
    violations = []
    for threshold_key, stat_key, breached, message in _HEALTH_CHECKS:
        limit = thresholds.get(threshold_key)
        if limit is None or stat_key not in stats:
            continue
        if breached(stats[stat_key], limit):
            violations.append(message(stats, stats[stat_key], limit))
    if "example_action_share" in stats:
        for action_type, floor in (thresholds.get("min_example_action_share") or {}).items():
            actual = stats["example_action_share"].get(action_type, 0.0)
            if actual < floor:
                violations.append(f"action {action_type!r} is {actual:.1%} of supervision targets (min {floor:.0%})")
    return violations


def corpus_metrics(trajectory_dirs) -> dict:
    """Outcome metrics over trainable trajectory dirs: does done() follow a re-check and cite evidence?"""
    counts = {"trainable_trajectories": 0, "done_steps": 0, "done_with_evidence": 0, "done_after_recheck": 0,
              "qa_verify_trajectories": 0, "outcome_relabeled": 0}
    for traj_dir in trajectory_dirs:
        steps = _load_json(traj_dir / "trajectory.json").get("steps", [])
        metadata = _load_json(traj_dir / "metadata.json")
        counts["trainable_trajectories"] += 1
        counts["qa_verify_trajectories"] += in_qa_verify_voice(metadata)
        counts["outcome_relabeled"] += metadata.get("thought_source") == OUTCOME_RELABEL_SOURCE
        if done_step_index(steps) is None:
            continue
        counts["done_steps"] += 1
        counts["done_with_evidence"] += done_cites_evidence(steps)
        counts["done_after_recheck"] += done_after_verify(steps)
    done_steps = counts["done_steps"] or 1
    trainable = counts["trainable_trajectories"] or 1
    return {**counts, "done_evidence_share": counts["done_with_evidence"] / done_steps,
            "done_after_recheck_share": counts["done_after_recheck"] / done_steps,
            "outcome_relabeled_share": counts["outcome_relabeled"] / trainable}


def audit_trajectory(traj_dir: Path) -> list[str]:
    """Defects for one trajectory; empty list means clean."""
    trajectory = _load_json(traj_dir / "trajectory.json")
    capture_meta = _load_json(traj_dir / "capture_meta.json")
    steps = trajectory.get("steps", [])
    defects = []

    if not steps:
        return ["no steps"]

    max_run = max_identical_run(steps)
    if max_run >= LOOP_RUN_LIMIT:
        defects.append(f"stuck loop: {max_run} consecutive identical actions")

    for step in steps:
        for action in step.get("actions", []):
            if action.get("type") == "type" and not (action.get("text") or "").strip():
                defects.append(f"step {step.get('step')}: empty type action")
            for key in ("x", "y"):
                if key in action:
                    limit = VIEWPORT[0] if key == "x" else VIEWPORT[1]
                    if not (0 <= action[key] <= limit):
                        defects.append(f"step {step.get('step')}: {key}={action[key]} outside viewport")
        if not (traj_dir / step.get("screenshot", "")).exists():
            defects.append(f"step {step.get('step')}: missing screenshot")
        # Watchdog approval-event steps are observations (event + empty actions) with no thought by
        # design; every ACTION step must still carry one.
        if step.get("actions") and not (step.get("thought") or "").strip():
            defects.append(f"step {step.get('step')}: empty thought")

    verifier = capture_meta.get("verifier") or {}
    if verifier.get("efficiency") == "meandering" and trajectory.get("success"):
        defects.append("tagged meandering (success:true) — kept; drop at convert with --exclude-meandering")

    is_agent = capture_meta.get("recorder") == "browser_driver"
    if is_agent and trajectory.get("success"):
        has_done = any(a.get("type") == "done" for s in steps for a in s.get("actions", []))
        if not has_done and capture_meta.get("agent_outcome") == "completed":
            defects.append("completed agent trajectory has no terminal done step (pre-done-era; convert synthesizes)")

    return defects


def is_trainable(traj_dir: Path) -> bool:
    """Same gate as train_qwen.py convert (default, no --include-replay), via the shared `trainability`."""
    trajectory = _load_json(traj_dir / "trajectory.json")
    metadata = _load_json(traj_dir / "metadata.json")
    capture_meta = _load_json(traj_dir / "capture_meta.json")
    return trainability(trajectory, metadata, capture_meta, traj_dir.name) is None


def default_health_config() -> Path:
    """corpus_health.json from the cwd (how Modal and make run the audit), else the repo root's copy."""
    local = Path(HEALTH_CONFIG_NAME)
    return local if local.is_file() else Path(__file__).resolve().parent.parent / HEALTH_CONFIG_NAME


def run_audit(corpus_dir: Path, health_config: Path | None = None, skip_health_gate: bool = False) -> int:
    trainable_failures = 0
    trainable_dirs = []
    skipped = 0
    for traj_dir in sorted(corpus_dir.glob("**/task_*")):
        if not (traj_dir / "trajectory.json").exists():
            continue
        if not is_trainable(traj_dir):
            skipped += 1
            continue
        trainable_dirs.append(traj_dir)
        defects = audit_trajectory(traj_dir)
        hard = [d for d in defects if "pre-done-era" not in d and "tagged meandering" not in d]
        if defects:
            marker = "FAIL" if hard else "warn"
            print(f"[{marker}] {traj_dir.relative_to(corpus_dir)}")
            for d in defects:
                print(f"       - {d}")
        if hard:
            trainable_failures += 1

    trainable = len(trainable_dirs)
    print(f"\naudit: {trainable} trainable trajectories, {trainable_failures} failing, {skipped} not trainable (excluded/replay/unverified)")
    metrics = corpus_metrics(trainable_dirs)
    print(f"outcome: done thoughts citing on-screen evidence {metrics['done_with_evidence']}/{metrics['done_steps']} "
          f"({metrics['done_evidence_share']:.1%}); done after a re-check {metrics['done_after_recheck']}/{metrics['done_steps']} "
          f"({metrics['done_after_recheck_share']:.1%}); qa_verify-voice trajectories {metrics['qa_verify_trajectories']}; "
          f"outcome-relabeled (thought_source={OUTCOME_RELABEL_SOURCE}) {metrics['outcome_relabeled']}/{trainable} "
          f"({metrics['outcome_relabeled_share']:.1%})")

    health_path = health_config or default_health_config()
    violations = corpus_health_violations(metrics, _load_json(health_path)) if health_path.is_file() else []
    if not health_path.is_file():
        print(f"corpus health gate: SKIPPED — no thresholds file at {health_path}")
    elif violations:
        print(f"CORPUS HEALTH GATE {'(bypassed with --skip-health-gate)' if skip_health_gate else 'FAILED'} "
              f"— {len(violations)} violation(s) against {health_path}:")
        for violation in violations:
            print(f"  - {violation}")
    else:
        print(f"corpus health gate: PASS ({health_path})")
    return 1 if trainable_failures or (violations and not skip_health_gate) else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("trajectories/"))
    parser.add_argument("--health-config", type=Path, default=None,
                        help=f"Thresholds JSON (default: ./{HEALTH_CONFIG_NAME}, else the repo root's; missing file skips the gate)")
    parser.add_argument("--skip-health-gate", action="store_true",
                        help="Report health violations but exit 0 on them (experiments only)")
    args = parser.parse_args(argv)
    return run_audit(args.corpus, args.health_config, args.skip_health_gate)


if __name__ == "__main__":
    sys.exit(main())
