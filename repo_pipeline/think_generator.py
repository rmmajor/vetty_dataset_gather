"""Generate `<think>` blocks for each step in a recorded trajectory.

The goal of these blocks is to teach Qwen3.5 the inner monologue of an agent executing a
GUI task. Each thought is the agent's first-person, pre-action reasoning toward the task
goal, grounded in the screenshot it sees. Concretely, for step N the model receives:

- The current screenshot (PNG, vision-grounded).
- The trajectory's goal (`trajectory.instruction`) and starting URL — passed in the system
  preamble so the goal is always salient.
- The thoughts and actions already produced for steps 1..N-1, so the chain stays consistent
  across steps and the agent voice can refer back to prior progress.
- A natural-language description of the action it is *about to* perform.
- A private grounding hint (role/name/bbox of the target) that the prompt explicitly
  forbids from leaking into the output — these are mechanical details, not part of the
  agent's monologue.

Generation is sequential within a single trajectory because the prompt for step N depends
on the (already-written) thought for step N-1. Concurrency happens at the trajectory level
in the orchestrator, not inside this function.
"""
import base64
import json
import logging
import os
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from repo_pipeline.llm_config import MODEL_FOR_GROUNDED_GENERATION
from repo_pipeline.sdk_runner import rate_limit_tracker, run_query, UsageLimitReached

logger = logging.getLogger(__name__)

# Each Claude Agent SDK `query()` carries a fixed overhead (~$0.08 for Sonnet 4.6 — see
# memory pathb-e2e-and-sdk-budget-floor). Defaults sit comfortably above that floor; the
# per-call envelope for one think block is dominated by SDK overhead, not by tokens.
THINK_BUDGET_USD = float(os.environ.get("THINK_BUDGET_USD", "0.15"))

THINK_SYSTEM_TEMPLATE = """You are the agent currently executing a task in a web browser.
Given the current screenshot and the next action you are about to take, write the inner
monologue (1-3 short sentences) that explains why this action moves you toward the goal.

Voice and format:
- First person, present tense, as if narrating your own next move.
- Anchor in what is VISIBLE on the screen — refer to UI elements by their visible text,
  not by role / selector / bounding box / coordinates.
- Tie the action to progress toward the goal: WHAT you observe about current progress,
  then WHY this action is the next step.
- Never ask for more context. Never say you are an AI. Never narrate as an annotator
  ("this click makes sense because...").
- If the action obviously will not work, write the thought as the agent would in the
  moment — do not editorialize.

Goal: {goal}
Starting URL: {start_url}

Good examples (style only):
- "The cart still shows 0 items, so I need to add the iPhone first. Clicking 'Add to
  cart' next to its listing."
- "The search returned no results for 'wireless mouse'. I'll broaden the query —
  clicking the search bar to retype 'mouse'."
- "I'm on the login screen. I'll enter my email first, so I'm clicking the email field."

Bad examples (do NOT produce):
- "I need more context to write this." (never ask, just write)
- "This click selects an element with role=button name='Submit'." (never quote selectors)
- "The action taken makes sense because..." (never narrate as an annotator)

Output only the thought text. No preamble, no quotes, no markdown, no JSON."""


THINK_USER_TEMPLATE = """Steps so far in this trajectory:
{history_block}

Now about to: {about_to_action}
(Grounding hint — do not quote in your output: {grounding_hint})

Write the thought."""


def render_action_nl(action: dict, role: str = "", name: str = "") -> str:
    """Render one action as a short English clause for the prompt. Coordinates never enter
    the prompt — the grounding hint carries the precise target separately."""
    action_type = action.get("type", "unknown")

    if name:
        target_phrase = f" '{name}'"
    elif role:
        target_phrase = f" the {role}"
    else:
        target_phrase = ""

    if action_type == "click":
        return f"click{target_phrase}" if target_phrase else "click (no labeled target)"
    if action_type == "double_click":
        return f"double-click{target_phrase}" if target_phrase else "double-click (no labeled target)"
    if action_type == "type":
        text = action.get("text", "")
        display = text[:60] + "…" if len(text) > 60 else text
        into = f" into{target_phrase}" if target_phrase else ""
        return f"type '{display}'{into}"
    if action_type == "keypress":
        keys = action.get("keys", []) or []
        return f"press {'+'.join(keys)}" if keys else "press a key"
    if action_type == "scroll":
        sy = action.get("scroll_y", 0) or 0
        direction = "down" if sy >= 0 else "up"
        return f"scroll {direction}"
    if action_type == "wait":
        return "wait"
    if action_type == "drag":
        return f"drag{target_phrase}" if target_phrase else "drag"

    return action_type or "act"


def render_grounding_hint(role: str, name: str, bbox: dict | None) -> str:
    """Format role/name/bbox into a single private hint line. The prompt explicitly forbids
    the model from quoting these strings — they exist only so the model can confirm which
    on-screen element it is acting on when the screenshot is ambiguous."""
    parts = []
    if role:
        parts.append(f"role={role}")
    if name:
        safe_name = name.replace('"', "'")
        parts.append(f'name="{safe_name}"')
    if isinstance(bbox, dict) and bbox:
        coords: list[int] = []
        for k in ("x", "y", "width", "height"):
            v = bbox.get(k)
            if not isinstance(v, (int, float)):
                coords = []
                break
            coords.append(int(v))
        if coords:
            parts.append(f"bbox=({coords[0]},{coords[1]},{coords[2]},{coords[3]})")
    return ", ".join(parts) or "(no grounding metadata)"


def render_history(
    steps: list[dict],
    current_step_num: int,
    action_nl_by_step: dict[int, str],
) -> str:
    """Build the per-trajectory history block: every step strictly before `current_step_num`,
    rendered as `N. [thought] → action`. Uses the thoughts already written in memory, so the
    caller must process steps sequentially in step-number order."""
    lines = []
    for step in steps:
        n = step.get("step")
        if not isinstance(n, int) or n >= current_step_num:
            continue
        thought = (step.get("thought") or "").strip() or "(no thought recorded)"
        action_nl = action_nl_by_step.get(n, "(no action)")
        lines.append(f"{n}. [{thought}] → {action_nl}")
    if not lines:
        return "(this is the first step)"
    return "\n".join(lines)


async def generate_think_for_step(
    screenshot_path: Path,
    goal: str,
    start_url: str,
    history_block: str,
    about_to_action: str,
    grounding_hint: str,
) -> str:
    """Single SDK call: send (system preamble + per-step user message + screenshot image)
    and return the model's thought verbatim. No regex post-filters: if the model produces
    something that doesn't look like a thought, the response is to *tighten the prompt*,
    not to silently retry. Genuine SDK failures (timeout, budget) return "" — the dashboard's
    observability metrics surface the gap so a prompt revision can be scheduled."""
    if not screenshot_path.exists():
        return ""

    b64_screenshot = base64.b64encode(screenshot_path.read_bytes()).decode("utf-8")

    system_text = THINK_SYSTEM_TEMPLATE.format(goal=goal, start_url=start_url)
    user_text = THINK_USER_TEMPLATE.format(
        history_block=history_block,
        about_to_action=about_to_action,
        grounding_hint=grounding_hint,
    )
    combined_text = system_text + "\n\n---\n\n" + user_text

    async def prompt_stream():
        yield {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": combined_text},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_screenshot}},
                ],
            },
            "parent_tool_use_id": None,
            "session_id": "",
        }

    try:
        result_text = await run_query(
            prompt_stream(),
            ClaudeAgentOptions(
                model=MODEL_FOR_GROUNDED_GENERATION,
                max_turns=1,
                max_budget_usd=THINK_BUDGET_USD,
                permission_mode="bypassPermissions",
            ),
            label="think",
        )
    except UsageLimitReached:
        raise
    except Exception as e:
        logger.warning(f"Think generation failed: {e}")
        return ""

    return result_text.strip()


async def generate_thinks_for_trajectory(
    trajectory_dir: Path,
    workers: int = 1,
) -> bool:
    """Generate a `<think>` block for every step in one trajectory, in step order.

    `workers` is accepted for backwards compatibility with stage 6 call sites but ignored —
    generation is strictly sequential within a trajectory because each prompt depends on
    the thoughts already produced for earlier steps. Parallelism, when wanted, happens at
    the trajectory level in the orchestrator."""
    del workers

    trajectory_path = trajectory_dir / "trajectory.json"
    if not trajectory_path.exists():
        return False

    with open(trajectory_path) as f:
        trajectory = json.load(f)

    steps = trajectory.get("steps", [])
    if not steps:
        return False

    if all(step.get("thought", "").strip() for step in steps):
        logger.debug(f"  {trajectory_dir.name}: already has all think blocks, skipping")
        return True

    goal = (trajectory.get("instruction") or "").strip() or "(goal not recorded)"
    start_url = (trajectory.get("start_url") or "").strip() or "(unknown)"

    capture_meta = {}
    capture_meta_path = trajectory_dir / "capture_meta.json"
    if capture_meta_path.exists():
        try:
            with open(capture_meta_path) as f:
                capture_meta = json.load(f)
        except Exception:
            capture_meta = {}
    steps_meta = capture_meta.get("steps", {}) if isinstance(capture_meta, dict) else {}

    action_nl_by_step: dict[int, str] = {}
    grounding_by_step: dict[int, tuple[str, str, dict | None]] = {}
    for step in steps:
        n = step.get("step")
        if not isinstance(n, int):
            continue
        actions = step.get("actions", []) or []
        meta = steps_meta.get(str(n), {}) if isinstance(steps_meta, dict) else {}
        role = step.get("role") or meta.get("role") or ""
        name = step.get("name") or meta.get("name") or ""
        bbox = step.get("bbox") or meta.get("bbox")
        action_nl_by_step[n] = render_action_nl(actions[0], role=role, name=name) if actions else "(no action)"
        grounding_by_step[n] = (role, name, bbox)

    ordered_steps = sorted(steps, key=lambda s: s.get("step", 0))
    for step in ordered_steps:
        n = step.get("step")
        if not isinstance(n, int):
            continue
        if (step.get("thought") or "").strip():
            continue
        if not step.get("actions"):
            continue

        screenshot_path = trajectory_dir / step["screenshot"]
        if not screenshot_path.exists():
            logger.debug(f"  {trajectory_dir.name}: screenshot missing for step {n}, skipping")
            continue

        history_block = render_history(steps, n, action_nl_by_step)
        about_to = action_nl_by_step.get(n, "(no action)")
        role, name, bbox = grounding_by_step.get(n, ("", "", None))
        grounding_hint = render_grounding_hint(role, name, bbox)

        thought = await generate_think_for_step(
            screenshot_path=screenshot_path,
            goal=goal,
            start_url=start_url,
            history_block=history_block,
            about_to_action=about_to,
            grounding_hint=grounding_hint,
        )
        step["thought"] = thought

    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2)

    filled = sum(1 for s in steps if (s.get("thought") or "").strip())
    logger.info(f"  {trajectory_dir.name}: generated {filled}/{len(steps)} think blocks")
    return True


async def generate_thinks_for_all(
    trajectories_dir: Path,
    workers: int = 1,
    budget_remaining: float | None = None,
):
    """Walk every `task_*/` subdir under `trajectories_dir` and fill missing thoughts."""
    del workers

    trajectory_dirs = sorted(trajectories_dir.glob("task_*/"))
    logger.info(f"Generating think blocks for {len(trajectory_dirs)} trajectories")

    cost_estimate = 0.0
    processed = 0

    for traj_dir in trajectory_dirs:
        if rate_limit_tracker.limit_reached():
            logger.warning("Usage limit reached, stopping think generation")
            break
        if budget_remaining is not None and cost_estimate > budget_remaining:
            logger.warning("Budget limit reached for think generation")
            break

        trajectory_path = traj_dir / "trajectory.json"
        if not trajectory_path.exists():
            continue

        with open(trajectory_path) as f:
            traj = json.load(f)

        step_count = len(traj.get("steps", []))
        estimated_cost = step_count * 0.008

        try:
            await generate_thinks_for_trajectory(traj_dir)
        except UsageLimitReached:
            logger.warning("Usage limit reached, stopping think generation")
            break
        cost_estimate += estimated_cost
        processed += 1

    logger.info(f"Think generation complete: {processed} trajectories, ~${cost_estimate:.2f} estimated")
    return {"processed": processed, "estimated_cost": cost_estimate}
