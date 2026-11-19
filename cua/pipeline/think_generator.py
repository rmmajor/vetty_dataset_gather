"""Generate `<think>` blocks for a recorded trajectory's steps, and polish raw teacher thoughts into student voice."""

import base64
import json
import logging
from pathlib import Path

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, RateLimitEvent, ResultMessage

from cua.pipeline.agent_runner import run_agent
from cua.pipeline.llm_config import MODEL_FOR_GROUNDED_GENERATION
from cua.pipeline.sdk_runner import UsageLimitReached, cost_tracker, harden_options, rate_limit_tracker
from cua.settings import PIPELINE

logger = logging.getLogger(__name__)

POLISH_PROMPT = """Below are the step-by-step inner thoughts an agent wrote WHILE executing a web
task, paired with the action each thought preceded. Rewrite each thought into clean training
supervision for a computer-use model: the first-person, present-tense inner monologue of an
agent looking at the screen and deciding its next move.

Rules for every rewritten thought:
- 1-3 short sentences, first person, present tense.
- Keep the original's reasoning and observations — you are polishing voice, not inventing
  content. If the original references visible UI text, keep those references.
- Remove ALL traces of tooling: never mention refs, snapshots, screenshots, tools, selectors,
  elements by number, "the page state", coordinates, or being an AI/agent under instruction.
- Tie the thought to progress toward the goal: what I observe, why this action is next.
- If an original thought is empty or unusable, write a plausible thought from its action and
  the surrounding context.

Goal of the task: {goal}

Steps (original thought -> action):
{steps_block}

Output ONLY a JSON array of {n} strings — the rewritten thought for each step, in order. No
markdown fence, no prose."""


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
    """Render one action as a short English clause for the prompt, naming the target but never its coordinates."""
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
    if action_type == "move":
        return f"hover over{target_phrase}" if target_phrase else "hover"
    if action_type == "done":
        return "declare the task complete"

    return action_type or "act"


def render_grounding_hint(role: str, name: str, bbox: dict | None) -> str:
    """Format role/name/bbox into one private hint line, which the prompt forbids the model from quoting back."""
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
    """Every step before `current_step_num` as `N. [thought] → action`, read from the thoughts already written."""
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
    registry=None,
    full_name: str = "",
) -> str:
    """One SDK call returning the model's thought verbatim, or "" when the call itself failed."""
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
        result_text = await run_agent(
            prompt_stream(),
            ClaudeAgentOptions(
                model=MODEL_FOR_GROUNDED_GENERATION,
                max_turns=1,
                max_budget_usd=PIPELINE.think_budget_usd,
                permission_mode="bypassPermissions",
            ),
            registry=registry,
            full_name=full_name,
            stage=7,
            purpose="think",
            raise_on_error=False,
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
    registry=None,
    full_name: str = "",
) -> bool:
    """Generate a `<think>` block for every step of one trajectory, in step order."""
    # Each prompt depends on the thoughts already written, so `workers` cannot apply inside one trajectory.
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

    # One session per trajectory: the fixed preamble is then cache-READ on turns 2+ instead of re-created every call.
    options = ClaudeAgentOptions(
        model=MODEL_FOR_GROUNDED_GENERATION,
        max_turns=1,
        max_budget_usd=PIPELINE.think_session_budget_usd,
        permission_mode="bypassPermissions",
    )
    harden_options(options)

    shown: set[int] = set()  # steps whose context the model has already seen this session
    first_turn = True
    try:
        async with ClaudeSDKClient(options=options) as client:
            for step in ordered_steps:
                n = step.get("step")
                if not isinstance(n, int) or not step.get("actions"):
                    continue
                if (step.get("thought") or "").strip():
                    continue  # already done (resume) — seeded into context below, not regenerated
                screenshot_path = trajectory_dir / step["screenshot"]
                if not screenshot_path.exists():
                    logger.debug(f"  {trajectory_dir.name}: screenshot missing for step {n}, skipping")
                    continue

                # Seeds only the finished steps this session has not been shown, so a resumed run stays cache-cheap.
                seed_lines = []
                for prior in ordered_steps:
                    m = prior.get("step")
                    if not isinstance(m, int) or m >= n or m in shown:
                        continue
                    if (prior.get("thought") or "").strip():
                        seed_lines.append(
                            f"{m}. [{prior['thought'].strip()}] → {action_nl_by_step.get(m, '(no action)')}"
                        )
                        shown.add(m)

                about_to = action_nl_by_step.get(n, "(no action)")
                role, name, bbox = grounding_by_step.get(n, ("", "", None))
                grounding_hint = render_grounding_hint(role, name, bbox)

                blocks = []
                if first_turn:
                    # Full instructions + goal once; the session carries them for later turns.
                    blocks.append(THINK_SYSTEM_TEMPLATE.format(goal=goal, start_url=start_url))
                if seed_lines:
                    blocks.append("Steps so far in this trajectory:\n" + "\n".join(seed_lines))
                reminder = (
                    "Write the thought."
                    if first_turn
                    else "Write the next thought (1-3 sentences, first person, anchored in what is "
                    "visible on screen; never quote selectors, roles, or coordinates)."
                )
                blocks.append(
                    f"Now about to: {about_to}\n"
                    f"(Grounding hint — do not quote in your output: {grounding_hint})\n\n{reminder}"
                )
                user_text = "\n\n---\n\n".join(blocks)
                first_turn = False

                step["thought"] = await _session_think_turn(
                    client,
                    user_text,
                    screenshot_path,
                    registry=registry,
                    full_name=full_name,
                    model=options.model,
                )
                shown.add(n)
                # Crash-safe: persist after each step so a mid-trajectory stop still resumes cleanly.
                with open(trajectory_path, "w") as f:
                    json.dump(trajectory, f, indent=2)
    except UsageLimitReached:
        # Persist progress, then propagate so the run aborts without rejecting the repo.
        with open(trajectory_path, "w") as f:
            json.dump(trajectory, f, indent=2)
        raise
    except Exception as e:
        logger.warning(f"  {trajectory_dir.name}: think session ended early: {e}")
        with open(trajectory_path, "w") as f:
            json.dump(trajectory, f, indent=2)

    filled = sum(1 for s in steps if (s.get("thought") or "").strip())
    logger.info(f"  {trajectory_dir.name}: generated {filled}/{len(steps)} think blocks")
    return True


async def _session_think_turn(
    client, user_text: str, screenshot_path: Path, *, registry, full_name: str, model: str
) -> str:
    """One turn of the persistent session; "" on any error, so a bad turn costs one step and not the trajectory."""
    b64 = base64.b64encode(screenshot_path.read_bytes()).decode("utf-8")
    message = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            ],
        },
        "parent_tool_use_id": None,
        "session_id": "default",
    }

    async def _stream(m=message):
        yield m

    if rate_limit_tracker.limit_reached():
        raise UsageLimitReached("usage limit reached before think turn")

    thought = ""
    result_msg = None
    try:
        await client.query(_stream())
        async for msg in client.receive_response():
            if isinstance(msg, RateLimitEvent):
                rate_limit_tracker.observe(msg.rate_limit_info)
                continue
            if isinstance(msg, AssistantMessage):
                for block in getattr(msg, "content", None) or []:
                    if hasattr(block, "text"):
                        thought = block.text
            if isinstance(msg, ResultMessage):
                result_msg = msg
                if msg.subtype == "success" and not msg.is_error:
                    thought = msg.result or thought
                elif getattr(msg, "api_error_status", None) == 429:
                    rate_limit_tracker.mark_rejected()
                    raise UsageLimitReached("HTTP 429 from think turn")
    except UsageLimitReached:
        raise
    except Exception as e:
        logger.warning(f"Think turn failed: {e}")
        return ""

    if result_msg is not None:
        cost_tracker.add(getattr(result_msg, "total_cost_usd", None))
        if registry is not None:
            registry.record_agent_call(
                full_name,
                7,
                "think",
                model,
                result_msg.num_turns,
                result_msg.total_cost_usd,
                result_msg.usage,
                result_msg.subtype,
                result_msg.is_error,
                result_msg.duration_ms,
            )
    return thought.strip()


def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def needs_polish(trajectory_dir: Path) -> bool:
    """True while a trajectory's thoughts are still the teacher's raw, tool-aware reasoning."""
    return _read_json_file(trajectory_dir / "metadata.json").get("thought_source") == "teacher_raw"


async def polish_thoughts_for_trajectory(
    trajectory_dir: Path,
    registry=None,
    full_name: str = "",
) -> bool:
    """Rewrite one trajectory's raw teacher thoughts into student voice, flipping thought_source on success."""
    trajectory_path = trajectory_dir / "trajectory.json"
    metadata_path = trajectory_dir / "metadata.json"
    if not trajectory_path.exists():
        return False

    trajectory = _read_json_file(trajectory_path)
    steps = [s for s in trajectory.get("steps", []) if s.get("actions")]
    if not steps:
        return False

    lines = []
    for step in steps:
        action_nl = render_action_nl(step["actions"][0], role=step.get("role") or "", name=step.get("name") or "")
        raw = (step.get("thought") or "").strip() or "(empty)"
        lines.append(f"{step['step']}. [{raw}] -> {action_nl}")

    prompt = POLISH_PROMPT.format(
        goal=(trajectory.get("instruction") or "(goal not recorded)").strip(),
        steps_block="\n".join(lines),
        n=len(steps),
    )

    try:
        result_text = await run_agent(
            prompt,
            ClaudeAgentOptions(
                model=MODEL_FOR_GROUNDED_GENERATION,
                max_turns=1,
                max_budget_usd=PIPELINE.polish_budget_usd,
                permission_mode="bypassPermissions",
            ),
            registry=registry,
            full_name=full_name,
            stage=7,
            purpose="polish",
            raise_on_error=False,
        )
    except UsageLimitReached:
        raise
    except Exception as e:
        logger.warning(f"  {trajectory_dir.name}: thought polish failed: {e}")
        return False

    start, end = result_text.find("["), result_text.rfind("]") + 1
    if start == -1 or end <= start:
        logger.warning(f"  {trajectory_dir.name}: polish output not a JSON array")
        return False
    try:
        rewritten = json.loads(result_text[start:end])
    except json.JSONDecodeError:
        logger.warning(f"  {trajectory_dir.name}: polish output unparseable")
        return False
    if len(rewritten) != len(steps) or not all(isinstance(t, str) and t.strip() for t in rewritten):
        logger.warning(f"  {trajectory_dir.name}: polish returned {len(rewritten)} thoughts for {len(steps)} steps")
        return False

    for step, thought in zip(steps, rewritten, strict=False):
        step["thought"] = thought.strip()
    trajectory_path.write_text(json.dumps(trajectory, indent=2))

    metadata = _read_json_file(metadata_path)
    metadata["thought_source"] = "teacher"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"  {trajectory_dir.name}: polished {len(steps)} thoughts")
    return True
