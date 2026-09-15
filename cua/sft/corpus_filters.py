"""Which recorded steps make bad supervision targets — dropped as targets only, always kept in the text history."""

from cua.settings import CORPUS

# A single-click step landing this close to a click in the previous lookback steps is a retry/backtrack, not work.
WASTE_CLICK_RADIUS_PX = CORPUS.waste_click_radius_px
WASTE_LOOKBACK_STEPS = CORPUS.waste_lookback_steps

MEANDER_KEEP_HEAD = CORPUS.meander_keep_head
MEANDER_KEEP_TAIL = CORPUS.meander_keep_tail
# 6 lands the meandering EXAMPLE share at roughly their TRAJECTORY share (~23%): represented, not dominant.
MEANDER_MID_KEEP_EVERY = CORPUS.meander_mid_keep_every

# Past this, a repeated key is an ArrowDown chain whose post-chain screenshot no longer matches per-press supervision.
KEYPRESS_MAX_REPEAT = CORPUS.keypress_max_repeat


def meander_drop_indices(steps, head=None, tail=None, keep_every=None):
    """Mid-trajectory step indices to drop as targets in a meandering trajectory (kept in history)."""
    head = MEANDER_KEEP_HEAD if head is None else head
    tail = MEANDER_KEEP_TAIL if tail is None else tail
    keep_every = MEANDER_MID_KEEP_EVERY if keep_every is None else keep_every
    drop = set()
    for i in range(head, max(head, len(steps) - tail)):
        if (i - head) % keep_every == 0:
            continue
        if any(a.get("type") == "done" for a in steps[i].get("actions", [])):
            continue
        drop.add(i)
    return drop


def absurd_keypress_indices(steps):
    """Indices of steps whose keypress repeats one key more than KEYPRESS_MAX_REPEAT times."""
    absurd = set()
    for i, step in enumerate(steps):
        for action in step.get("actions", []):
            keys = action.get("keys") or []
            if action.get("type") == "keypress" and len(keys) > KEYPRESS_MAX_REPEAT and len(set(keys)) == 1:
                absurd.add(i)
    return absurd


def waste_step_indices(steps, radius_px=WASTE_CLICK_RADIUS_PX, lookback=WASTE_LOOKBACK_STEPS):
    """Indices of PURE single-click steps that merely retry a recent click; a step's own click is recorded after it is checked."""
    waste = set()
    recent = []
    for i, step in enumerate(steps):
        actions = step.get("actions") or ([step["action"]] if "action" in step else [])
        clicks = [a for a in actions if a.get("type") == "click" and a.get("x") is not None]
        if len(actions) == 1 and len(clicks) == 1:
            x, y = clicks[0]["x"], clicks[0]["y"]
            if any(i - j <= lookback and abs(x - px) <= radius_px and abs(y - py) <= radius_px for px, py, j in recent):
                waste.add(i)
        for a in clicks:
            recent.append((a["x"], a["y"], i))
    return waste


def maybe_synthesize_done_step(trajectory, trajectory_dir, capture_meta):
    """Terminal supervision for agent trajectories recorded before the driver emitted `done` steps; without one the model never learns to STOP."""
    steps = trajectory.get("steps", [])
    if any(a.get("type") == "done" for s in steps for a in s.get("actions", [])):
        return
    if capture_meta.get("agent_outcome") != "completed":
        return
    summary = (capture_meta.get("agent_summary") or "").strip()
    final_screenshot = trajectory_dir / "final_screenshot.png"
    if not summary or not final_screenshot.exists():
        return
    steps.append(
        {
            "step": (steps[-1]["step"] + 1) if steps else 1,
            "screenshot": "final_screenshot.png",
            "thought": summary,
            "actions": [{"type": "done"}],
        }
    )
