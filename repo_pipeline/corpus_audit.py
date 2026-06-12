"""Mechanical quality gate over the TRAINABLE corpus.

Checks every trajectory that `train_qwen.py convert` would train on (excluded/unverified/
replay-gated ones are reported but not gated) for recording defects that no LLM judge needs:

  * stuck loops — identical-action runs >= LOOP_RUN_LIMIT (the Booking.com failure class);
  * empty type actions ({type:"type", text:""} — broken recording);
  * coordinates outside the 1280x720 viewport;
  * steps with missing screenshots or empty thoughts;
  * verified agent trajectories lacking a terminal done step;
  * verifier-graded meandering that still carries success:true.

Exit code 1 when any trainable trajectory fails — wire into CI / run before every training
export: `python -m repo_pipeline.corpus_audit trajectories/` (or `make audit-corpus`).
"""
import json
import sys
from pathlib import Path

LOOP_RUN_LIMIT = 3
VIEWPORT = (1280, 720)


def _action_signature(step: dict) -> tuple:
    """Full action identity, INCLUDING scroll deltas and drag paths — progressive scrolls
    share (type, x, y) but are distinct actions, not a stuck loop."""
    return tuple(
        (
            a.get("type"), a.get("x"), a.get("y"), a.get("text"), tuple(a.get("keys") or []),
            a.get("scroll_x"), a.get("scroll_y"),
            json.dumps(a.get("path")) if a.get("path") else None,
        )
        for a in step.get("actions", [])
    )


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def audit_trajectory(traj_dir: Path) -> list[str]:
    """Defects for one trajectory; empty list means clean."""
    trajectory = _load_json(traj_dir / "trajectory.json")
    capture_meta = _load_json(traj_dir / "capture_meta.json")
    steps = trajectory.get("steps", [])
    defects = []

    if not steps:
        return ["no steps"]

    max_run, run, prev = 0, 0, None
    for step in steps:
        sig = _action_signature(step)
        run = run + 1 if sig == prev and sig else 1
        prev = sig
        max_run = max(max_run, run)
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
        if not (step.get("thought") or "").strip():
            defects.append(f"step {step.get('step')}: empty thought")

    verifier = capture_meta.get("verifier") or {}
    if verifier.get("efficiency") == "meandering" and trajectory.get("success"):
        defects.append("verifier graded meandering but success:true")

    is_agent = capture_meta.get("recorder") == "browser_driver"
    if is_agent and trajectory.get("success"):
        has_done = any(a.get("type") == "done" for s in steps for a in s.get("actions", []))
        if not has_done and capture_meta.get("agent_outcome") == "completed":
            defects.append("completed agent trajectory has no terminal done step (pre-done-era; convert synthesizes)")

    return defects


def is_trainable(traj_dir: Path) -> bool:
    """Mirrors train_qwen.py convert gating (default, no --include-replay)."""
    trajectory = _load_json(traj_dir / "trajectory.json")
    metadata = _load_json(traj_dir / "metadata.json")
    if metadata.get("excluded_reason"):
        return False
    source = metadata.get("capture_source") or ("path_a" if "mind2web" in traj_dir.name else "replay")
    if source == "replay":
        return False
    if source == "agent" and not trajectory.get("success", True):
        return False
    return True


def run_audit(corpus_dir: Path) -> int:
    trainable_failures = 0
    trainable = 0
    skipped = 0
    for traj_dir in sorted(corpus_dir.glob("**/task_*")):
        if not (traj_dir / "trajectory.json").exists():
            continue
        if not is_trainable(traj_dir):
            skipped += 1
            continue
        trainable += 1
        defects = audit_trajectory(traj_dir)
        hard = [d for d in defects if "pre-done-era" not in d]
        if defects:
            marker = "FAIL" if hard else "warn"
            print(f"[{marker}] {traj_dir.relative_to(corpus_dir)}")
            for d in defects:
                print(f"       - {d}")
        if hard:
            trainable_failures += 1

    print(f"\naudit: {trainable} trainable trajectories, {trainable_failures} failing, {skipped} not trainable (excluded/replay/unverified)")
    return 1 if trainable_failures else 0


if __name__ == "__main__":
    corpus = Path(sys.argv[1] if len(sys.argv) > 1 else "trajectories/")
    sys.exit(run_audit(corpus))
