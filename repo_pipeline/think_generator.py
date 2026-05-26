import asyncio
import base64
import json
import logging
import os
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

logger = logging.getLogger(__name__)

THINK_BUDGET_USD = float(os.environ.get("THINK_BUDGET_USD", "0.05"))

THINK_PROMPT = """You are annotating a web browsing trajectory for training data.

Given this screenshot and the action that was taken, write a brief reasoning explanation (1-3 sentences) for WHY this action makes sense.

Action taken: {action_description}
Previous actions: {previous_actions}

Write ONLY the reasoning, no action. Be specific about what you see in the screenshot."""


def format_action_description(action: dict) -> str:
    action_type = action.get("type", "unknown")

    if action_type == "click":
        x, y = action.get("x", 0), action.get("y", 0)
        button = action.get("button", "left")
        return f'click({x}, {y}) with {button} button'
    elif action_type == "double_click":
        x, y = action.get("x", 0), action.get("y", 0)
        return f'double_click({x}, {y})'
    elif action_type == "type":
        text = action.get("text", "")
        display_text = text[:50] + "..." if len(text) > 50 else text
        return f'type("{display_text}")'
    elif action_type == "keypress":
        keys = action.get("keys", [])
        return f'keypress({"+".join(keys)})'
    elif action_type == "scroll":
        x, y = action.get("x", 0), action.get("y", 0)
        sy = action.get("scroll_y", 0)
        return f'scroll at ({x}, {y}), scrollY={sy}'
    elif action_type == "wait":
        return "wait()"
    elif action_type == "drag":
        path = action.get("path", [])
        if len(path) >= 2:
            return f'drag from ({path[0]["x"]}, {path[0]["y"]}) to ({path[-1]["x"]}, {path[-1]["y"]})'
        return "drag()"

    return f"{action_type}()"


def format_previous_actions(steps: list[dict], current_step: int) -> str:
    previous = []
    for step in steps:
        if step["step"] >= current_step:
            break
        for action in step.get("actions", []):
            previous.append(format_action_description(action))

    if not previous:
        return "(this is the first action)"

    if len(previous) > 5:
        return "[" + ", ".join(previous[-5:]) + f"] (showing last 5 of {len(previous)})"

    return "[" + ", ".join(previous) + "]"


async def generate_think_for_step(
    screenshot_path: Path,
    action: dict,
    previous_actions_str: str,
) -> str:
    if not screenshot_path.exists():
        return ""

    screenshot_bytes = screenshot_path.read_bytes()
    b64_screenshot = base64.b64encode(screenshot_bytes).decode("utf-8")

    action_desc = format_action_description(action)

    prompt = f"""Look at this screenshot image (base64 encoded PNG below) and explain why the following action makes sense.

Screenshot (base64): {b64_screenshot[:100]}... [full image provided]

{THINK_PROMPT.format(
    action_description=action_desc,
    previous_actions=previous_actions_str,
)}"""

    # Use a simpler prompt that works with the Agent SDK
    simple_prompt = (
        f"You are annotating a web browsing trajectory. "
        f"The action taken was: {action_desc}. "
        f"Previous actions: {previous_actions_str}. "
        f"Write a brief reasoning explanation (1-3 sentences) for why this action makes sense "
        f"in the context of web browsing automation. "
        f"Write ONLY the reasoning, be specific and concise."
    )

    result_text = ""
    try:
        async for message in query(
            prompt=simple_prompt,
            options=ClaudeAgentOptions(
                model="claude-haiku-4-5-20251001",
                max_turns=1,
                max_budget_usd=THINK_BUDGET_USD,
                permission_mode="bypassPermissions",
            ),
        ):
            if isinstance(message, ResultMessage) and message.subtype == "success":
                result_text = message.result or ""
    except Exception as e:
        logger.warning(f"Think generation failed: {e}")
        return ""

    return result_text.strip()


async def generate_thinks_for_trajectory(
    trajectory_dir: Path,
    workers: int = 10,
) -> bool:
    trajectory_path = trajectory_dir / "trajectory.json"
    if not trajectory_path.exists():
        return False

    with open(trajectory_path) as f:
        trajectory = json.load(f)

    steps = trajectory.get("steps", [])
    if not steps:
        return False

    already_has_thinks = all(step.get("thought", "").strip() for step in steps)
    if already_has_thinks:
        logger.debug(f"  {trajectory_dir.name}: already has all think blocks, skipping")
        return True

    semaphore = asyncio.Semaphore(workers)

    async def process_step(step):
        if step.get("thought", "").strip():
            return

        actions = step.get("actions", [])
        if not actions:
            return

        screenshot_path = trajectory_dir / step["screenshot"]
        previous_str = format_previous_actions(steps, step["step"])

        async with semaphore:
            thought = await generate_think_for_step(
                screenshot_path,
                actions[0],
                previous_str,
            )
            step["thought"] = thought

    await asyncio.gather(*[process_step(step) for step in steps])

    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2)

    filled = sum(1 for s in steps if s.get("thought", "").strip())
    logger.info(f"  {trajectory_dir.name}: generated {filled}/{len(steps)} think blocks")
    return True


async def generate_thinks_for_all(
    trajectories_dir: Path,
    workers: int = 10,
    budget_remaining: float = None,
):
    trajectory_dirs = sorted(trajectories_dir.glob("task_*/"))
    logger.info(f"Generating think blocks for {len(trajectory_dirs)} trajectories")

    cost_estimate = 0.0
    processed = 0

    for traj_dir in trajectory_dirs:
        if budget_remaining is not None and cost_estimate > budget_remaining:
            logger.warning("Budget limit reached for think generation")
            break

        trajectory_path = traj_dir / "trajectory.json"
        if not trajectory_path.exists():
            continue

        with open(trajectory_path) as f:
            traj = json.load(f)

        step_count = len(traj.get("steps", []))
        estimated_cost = step_count * 0.005

        await generate_thinks_for_trajectory(traj_dir, workers=workers)
        cost_estimate += estimated_cost
        processed += 1

    logger.info(f"Think generation complete: {processed} trajectories, ~${cost_estimate:.2f} estimated")
    return {"processed": processed, "estimated_cost": cost_estimate}
