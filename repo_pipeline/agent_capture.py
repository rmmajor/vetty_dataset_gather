"""Teacher loop: Claude pursues one user task in a sandbox-booted app, recording a trajectory.

The teacher acts through MCP tools that wrap the in-container browser driver. Every action
tool REQUIRES a `thought` argument — the first-person, intent-level reasoning that becomes the
step's supervision signal (recorded by the driver alongside the screenshot + coordinates). This
makes per-step reasoning a structural property of the capture rather than something scraped
from the message stream after the fact.

After the run, an independent verifier judges the final screenshot against the goal. Verified
trajectories get success=true (the only ones `convert` trains on); coherent partial successes
are hindsight-relabeled — the instruction is rewritten to what was actually accomplished, which
turns an honest 12-step partial into a perfectly-supervised trajectory instead of waste.
"""
import asyncio
import json
import logging
import os
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from repo_pipeline.agent_runner import run_agent
from repo_pipeline.driver_session import DriverClient, DriverError
from repo_pipeline.llm_config import MODEL_FOR_REASONING, MODEL_FOR_TEACHER_AGENT
from repo_pipeline.sdk_runner import UsageLimitReached

logger = logging.getLogger(__name__)

MAX_ACTIONS_PER_TASK = int(os.environ.get("AGENT_MAX_ACTIONS", "60"))
AGENT_TASK_BUDGET_USD = float(os.environ.get("AGENT_TASK_BUDGET_USD", "2.50"))
AGENT_MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "150"))
VERIFY_BUDGET_USD = float(os.environ.get("VERIFY_BUDGET_USD", "0.30"))
# Hindsight relabeling only makes sense for trajectories with real substance.
RELABEL_MIN_STEPS = int(os.environ.get("AGENT_RELABEL_MIN_STEPS", "3"))

TEACHER_PROMPT = """You are a skilled computer user accomplishing a goal in a web application. \
Work through the app's UI exactly as a careful human would.

YOUR GOAL: {instruction}

The app is already open. Its current state:
URL: {url}

{snapshot}

How to work:
- Observe the page (the element list above; use browser_snapshot to refresh it, \
browser_screenshot when you need to SEE the layout). Then act with the browser_* action tools.
- Every action tool requires a `thought`: 1-3 short sentences of YOUR reasoning, first person, \
present tense, anchored in what is visible on screen. State what you observe about your \
progress and why this action is the next step toward the goal. These thoughts are recorded.
  - GOOD: "The project list is still empty, so I need to create one first. The 'New project' \
button in the toolbar should open the creation form."
  - BAD: "Clicking element 14." / "I will use the snapshot tool." — never mention refs, \
snapshots, tools, or selectors in a thought; speak only about the app as a user sees it.
- Prefer acting through the UI. Do not use browser_navigate to jump to deep URLs — a real \
user finds things by clicking; navigate only if the UI is genuinely stuck.
- Controls are often hidden until hover (row edit/delete buttons, dropdown menus) — hover \
over the element, then re-snapshot. Use drag for reordering and kanban moves. Use wait when \
the page is visibly loading. On canvas/drawing surfaces (no listed elements), take a \
screenshot and act with click_xy / drag_path at pixel coordinates.
- If an action is refused or the page doesn't react as expected, re-snapshot, reconsider, and \
try a DIFFERENT approach rather than repeating the same action.
- Distinguish YOUR mistakes (recover and continue) from a BROKEN APP (error pages, missing \
content, dead backend). If the app itself cannot support the goal, confirm the blockage once \
— one re-snapshot or one alternate route — then call task_complete with outcome='failed' \
immediately. Grinding against a broken app produces worthless recordings.
- You have at most {max_actions} actions. Be purposeful, not hasty: completing the goal \
properly matters more than speed.
- You MUST end by calling task_complete with an honest outcome — when the goal is \
accomplished, or when you are certain it cannot be. Never finish with only a text message; \
the run does not count as ended until task_complete is called. Lying about completion is \
worse than reporting failure.

Begin now."""

VERIFY_PROMPT = """You are auditing whether a browser-automation run actually accomplished its goal, \
and whether it did so EFFICIENTLY. Be skeptical: agents overclaim. Judge ONLY from the evidence below.

Goal given to the agent: {instruction}

Actions performed ({n_steps} steps; the task designer expected roughly {expected_actions}):
{step_summary}

Agent's own outcome claim: {agent_outcome} — {agent_summary}

Attached images, in order: {image_labels}.

Answer in JSON only (no fence, no prose):
{{"success": <true|false>,
  "efficiency": "<efficient|acceptable|meandering>",
  "accomplished": "<if success=false but the run coherently completed a meaningful sub-goal, \
state that sub-goal as a standalone user instruction in imperative voice; else null>",
  "reasoning": "<one or two sentences>"}}

Rules:
- success=true only if the final state visibly reflects the goal being met.
- efficiency: 'efficient' = purposeful progress throughout (mid-run screenshots show \
distinct forward states); 'acceptable' = minor detours or one recovered mistake; \
'meandering' = repeated actions without effect, aimless wandering, or step count far beyond \
what the goal needs. Meandering trajectories are bad training data even when they succeed.
- "accomplished" must describe something the FINAL screenshot actually evidences, must cover \
the trajectory from its first step, and must be a goal a real user could have asked for. \
If the run is a meandering mess, use null.
- Never reward an agent that just clicked around without effect."""


def build_driver_mcp_server(client: DriverClient, state: dict, server_name: str = "browser"):
    """In-process MCP server exposing the in-container driver to the teacher.

    `state` is the per-task mutable record shared with the caller: action count, completion
    flag, the agent's claimed outcome. Tool handlers run the (blocking, docker-exec-backed)
    driver calls in an executor so concurrent stage workers overlap.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    async def _run(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)

    def _text(payload: str, is_error: bool = False):
        return {"content": [{"type": "text", "text": payload}], "is_error": is_error}

    async def _act(payload: dict):
        if state["completed"]:
            return _text("task already completed — no further actions.", is_error=True)
        if state["actions"] >= MAX_ACTIONS_PER_TASK:
            return _text(
                "action budget exhausted. Call task_complete now with an honest outcome.",
                is_error=True,
            )
        try:
            result = await _run(client.act, payload)
        except DriverError as e:
            return _text(f"action refused: {e}", is_error=True)
        except Exception as e:
            return _text(f"driver failure: {str(e)[:300]}", is_error=True)
        state["actions"] += 1
        remaining = MAX_ACTIONS_PER_TASK - state["actions"]
        return _text(
            f"done (step {result['step']}, {remaining} actions left)\nURL: {result['url']}\n\n{result['snapshot']}"
        )

    THOUGHT_SCHEMA = {
        "type": "string",
        "description": (
            "Your reasoning for this action: 1-3 first-person sentences anchored in the "
            "visible UI. Recorded as training supervision — never mention refs, tools, "
            "snapshots, or selectors."
        ),
    }

    def action_schema(extra: dict, required: list[str]) -> dict:
        return {
            "type": "object",
            "properties": {"thought": THOUGHT_SCHEMA, **extra},
            "required": ["thought", *required],
        }

    @tool(
        "click",
        "Click an element by its snapshot ref. Optional: button='right', dblclick=true.",
        action_schema(
            {
                "ref": {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right"]},
                "dblclick": {"type": "boolean"},
            },
            ["ref"],
        ),
    )
    async def click_tool(args):
        return await _act({
            "kind": "click", "ref": args["ref"], "thought": args.get("thought", ""),
            "button": args.get("button", "left"), "dblclick": bool(args.get("dblclick", False)),
        })

    @tool(
        "type",
        "Click a field and type text into it (replaces existing content). "
        "Set press_enter=true to submit with Enter afterwards.",
        action_schema(
            {
                "ref": {"type": "integer"},
                "text": {"type": "string"},
                "press_enter": {"type": "boolean"},
            },
            ["ref", "text"],
        ),
    )
    async def type_tool(args):
        return await _act({
            "kind": "type", "ref": args["ref"], "text": args.get("text", ""),
            "thought": args.get("thought", ""), "press_enter": bool(args.get("press_enter", False)),
        })

    @tool(
        "press",
        "Press a key or chord, e.g. keys=['Enter'] or ['Control','a'].",
        action_schema({"keys": {"type": "array", "items": {"type": "string"}}}, ["keys"]),
    )
    async def press_tool(args):
        return await _act({"kind": "press", "keys": args.get("keys", []), "thought": args.get("thought", "")})

    @tool(
        "scroll",
        "Scroll the page vertically by dy pixels (positive = down, negative = up).",
        action_schema({"dy": {"type": "integer"}}, ["dy"]),
    )
    async def scroll_tool(args):
        return await _act({"kind": "scroll", "dy": args.get("dy", 500), "thought": args.get("thought", "")})

    @tool(
        "drag",
        "Drag one element onto another (kanban cards, sliders, reordering). Falls back to "
        "HTML5 drag events automatically when raw mouse dragging has no effect.",
        action_schema({"ref": {"type": "integer"}, "to_ref": {"type": "integer"}}, ["ref", "to_ref"]),
    )
    async def drag_tool(args):
        return await _act({
            "kind": "drag", "ref": args["ref"], "to_ref": args["to_ref"], "thought": args.get("thought", ""),
        })

    @tool(
        "hover",
        "Move the mouse over an element without clicking — opens hover menus and reveals "
        "hidden row controls (edit/delete buttons). Re-snapshot afterwards to see what appeared.",
        action_schema({"ref": {"type": "integer"}}, ["ref"]),
    )
    async def hover_tool(args):
        return await _act({"kind": "hover", "ref": args["ref"], "thought": args.get("thought", "")})

    @tool(
        "click_xy",
        "Click at exact pixel coordinates (1280x720 viewport). ONLY for surfaces with no "
        "referenceable elements — canvases, maps, drawing areas. Read coordinates from a "
        "screenshot first. For normal elements always use click with a ref.",
        action_schema(
            {
                "x": {"type": "integer"}, "y": {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right"]},
                "dblclick": {"type": "boolean"},
            },
            ["x", "y"],
        ),
    )
    async def click_xy_tool(args):
        return await _act({
            "kind": "click_xy", "x": args["x"], "y": args["y"], "thought": args.get("thought", ""),
            "button": args.get("button", "left"), "dblclick": bool(args.get("dblclick", False)),
        })

    @tool(
        "drag_path",
        "Press the mouse down and drag through a sequence of pixel coordinates. ONLY for "
        "canvas surfaces (drawing, lasso-select, panning a map). For element-to-element "
        "drags use drag with refs.",
        action_schema(
            {
                "path": {
                    "type": "array", "minItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                        "required": ["x", "y"],
                    },
                },
            },
            ["path"],
        ),
    )
    async def drag_path_tool(args):
        return await _act({"kind": "drag_path", "path": args.get("path", []), "thought": args.get("thought", "")})

    @tool(
        "upload",
        "Attach a file to a file input (generates a small file in the environment). Optional "
        "filename and text content. Not recorded as a user action — use only when a task "
        "needs a file to proceed.",
        {
            "type": "object",
            "properties": {
                "ref": {"type": "integer"},
                "filename": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["ref"],
        },
    )
    async def upload_tool(args):
        return await _act({
            "kind": "upload", "ref": args["ref"],
            "filename": args.get("filename", "upload.txt"), "content": args.get("content"),
        })

    @tool(
        "select",
        "Choose an option in a native dropdown (<select>) by its value.",
        action_schema({"ref": {"type": "integer"}, "value": {"type": "string"}}, ["ref", "value"]),
    )
    async def select_tool(args):
        return await _act({
            "kind": "select", "ref": args["ref"], "value": args.get("value", ""),
            "thought": args.get("thought", ""),
        })

    @tool("snapshot", "Refresh the interactive-element view of the current page.", {})
    async def snapshot_tool(args):
        try:
            result = await _run(client.snapshot)
            return _text(f"URL: {result['url']}\nTitle: {result['title']}\n\n{result['snapshot']}")
        except Exception as e:
            return _text(f"snapshot failed: {str(e)[:300]}", is_error=True)

    @tool("screenshot", "See the current page as an image (use sparingly; the snapshot is usually enough).", {})
    async def screenshot_tool(args):
        try:
            b64 = await _run(client.screenshot_b64)
            return {"content": [{"type": "image", "data": b64, "mimeType": "image/png"}]}
        except Exception as e:
            return _text(f"screenshot failed: {str(e)[:300]}", is_error=True)

    @tool(
        "wait",
        "Deliberately wait up to 10 seconds for the page to settle (loading spinners, async "
        "updates, animations). This is a recorded action — knowing WHEN to wait is part of "
        "the skill being captured.",
        action_schema({"seconds": {"type": "number"}}, ["seconds"]),
    )
    async def wait_tool(args):
        return await _act({
            "kind": "wait", "seconds": float(args.get("seconds", 1)), "thought": args.get("thought", ""),
        })

    @tool(
        "navigate",
        "Load a URL directly. LAST RESORT — real users click through the UI; mid-task jumps "
        "degrade the recording.",
        {"url": str},
    )
    async def navigate_tool(args):
        try:
            result = await _run(client.navigate, args.get("url", ""))
            return _text(f"URL: {result['url']}\n\n{result['snapshot']}")
        except Exception as e:
            return _text(f"navigate failed: {str(e)[:300]}", is_error=True)

    @tool(
        "task_complete",
        "End the task. outcome: 'completed' | 'partial' | 'failed'. summary: what was actually "
        "achieved, honestly. closing_thought (required when outcome=completed): 1-3 first-person "
        "present-tense sentences stating what you SEE on screen that confirms the goal is met — "
        "this is recorded as the final step's reasoning.",
        {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": ["completed", "partial", "failed"]},
                "summary": {"type": "string"},
                "closing_thought": {"type": "string"},
            },
            "required": ["outcome", "summary"],
        },
    )
    async def complete_tool(args):
        outcome = args.get("outcome", "")
        # A confirmed completion records a terminal `done` step (screenshot of the final state
        # + the agent's goal-is-met observation) — the supervision that teaches WHEN to stop.
        if outcome == "completed" and not state["completed"]:
            thought = (args.get("closing_thought") or "").strip() or (args.get("summary") or "").strip()
            try:
                await _run(client.act, {"kind": "done", "thought": thought})
                state["actions"] += 1
            except Exception as e:
                logger.warning(f"done-step recording failed: {e}")
        state["completed"] = True
        state["agent_outcome"] = outcome
        state["agent_summary"] = args.get("summary", "")
        return _text("task ended. Stop here — give a one-line closing message.")

    tools = [
        click_tool, type_tool, press_tool, scroll_tool, drag_tool, hover_tool,
        click_xy_tool, drag_path_tool, upload_tool, select_tool,
        snapshot_tool, screenshot_tool, wait_tool, navigate_tool, complete_tool,
    ]
    server = create_sdk_mcp_server(name=server_name, version="1.0.0", tools=tools)
    allowed = [
        f"mcp__{server_name}__{name}"
        for name in (
            "click", "type", "press", "scroll", "drag", "hover",
            "click_xy", "drag_path", "upload", "select",
            "snapshot", "screenshot", "wait", "navigate", "task_complete",
        )
    ]
    return server, allowed


async def run_teacher_task(
    client: DriverClient,
    task: dict,
    app_url: str,
    registry=None,
    full_name: str = "",
) -> dict:
    """Run one task end-to-end: open a driver session, let the teacher act, ensure the
    trajectory is finalized in-container. Returns the per-task state record."""
    state = {"actions": 0, "completed": False, "agent_outcome": "", "agent_summary": ""}

    loop = asyncio.get_running_loop()
    task_meta = {"difficulty": task.get("difficulty", ""), "expected_actions": task.get("expected_actions", 0)}
    session = await loop.run_in_executor(
        None, lambda: client.session(task["id"], app_url, task["instruction"], task_meta=task_meta)
    )

    server, allowed = build_driver_mcp_server(client, state)
    prompt = TEACHER_PROMPT.format(
        instruction=task["instruction"],
        url=session["url"],
        snapshot=session["snapshot"],
        max_actions=MAX_ACTIONS_PER_TASK,
    )

    final_text = ""
    try:
        final_text = await run_agent(
            prompt,
            ClaudeAgentOptions(
                model=MODEL_FOR_TEACHER_AGENT,
                max_turns=AGENT_MAX_TURNS,
                max_budget_usd=AGENT_TASK_BUDGET_USD,
                permission_mode="bypassPermissions",
                mcp_servers={"browser": server},
                allowed_tools=allowed,
                disallowed_tools=["Bash"],
            ),
            registry=registry,
            full_name=full_name,
            stage=6,
            purpose="teacher_task",
            raise_on_error=False,
        )
    finally:
        # Always finalize so the trajectory + final screenshot land on disk, even when the
        # teacher ran out of turns/budget without calling task_complete. A teacher that ends
        # with a closing message but no task_complete call still leaves the verifier its own
        # account of the run.
        if not state["completed"]:
            state["agent_outcome"] = state["agent_outcome"] or "unreported"
            state["agent_summary"] = state["agent_summary"] or (final_text or "").strip()[:400]
        try:
            await loop.run_in_executor(
                None, client.finish, state["agent_outcome"], state["agent_summary"]
            )
        except Exception as e:
            logger.warning(f"driver finish failed for {task['id']}: {e}")

    return state


def _step_summary(traj: dict, limit: int = 40) -> str:
    lines = []
    for step in traj.get("steps", [])[:limit]:
        kinds = []
        for a in step.get("actions", []):
            if a.get("type") == "type":
                kinds.append(f"type \"{(a.get('text') or '')[:40]}\"")
            elif a.get("type") == "keypress":
                kinds.append(f"press {'+'.join(a.get('keys') or [])}")
            else:
                kinds.append(a.get("type", "?"))
        target = f" on {step.get('role') or ''} \"{(step.get('name') or '')[:50]}\"".rstrip()
        lines.append(f"{step['step']}. {', '.join(kinds)}{target if step.get('name') else ''}")
    if len(traj.get("steps", [])) > limit:
        lines.append(f"... and {len(traj['steps']) - limit} more steps")
    return "\n".join(lines) or "(no steps recorded)"


def _verify_prompt_stream(text: str, screenshot_paths: list[Path]):
    import base64

    content = [{"type": "text", "text": text}]
    for path in screenshot_paths:
        if not path.exists():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode()
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})

    async def stream():
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": None,
            "session_id": "",
        }

    return stream()


def _verification_screenshots(traj_dir: Path, steps: list) -> tuple[list[Path], str]:
    """Final screenshot plus 2 sampled mid-trajectory frames. A final-screenshot-only verifier
    is process-blind: a run that wandered into the right end state trains wandering as
    positive. Mid frames let the judge see whether progress was actually being made."""
    paths = []
    labels = []
    ui_steps = [s for s in steps if s.get("actions") and s["actions"][0].get("type") != "done"]
    if len(ui_steps) >= 6:
        for fraction, label in ((1 / 3, "mid-run (1/3)"), (2 / 3, "mid-run (2/3)")):
            step = ui_steps[int(len(ui_steps) * fraction)]
            p = traj_dir / step.get("screenshot", "")
            if p.exists():
                paths.append(p)
                labels.append(f"{label} — before step {step.get('step')}")
    final = traj_dir / "final_screenshot.png"
    if not final.exists() and steps:
        final = traj_dir / steps[-1].get("screenshot", "")
    paths.append(final)
    labels.append("FINAL state")
    return paths, "; ".join(labels)


async def verify_trajectory(
    traj_dir: Path,
    state: dict | None = None,
    registry=None,
    full_name: str = "",
) -> dict:
    """Judge one captured trajectory on the host (after copy_out) and apply the verdict:
    success=true on a confirmed goal, hindsight-relabel on a coherent partial, success=false
    otherwise. Returns the verdict dict; writes trajectory.json + capture_meta.json in place."""
    trajectory_file = traj_dir / "trajectory.json"
    traj = json.loads(trajectory_file.read_text())
    meta_file = traj_dir / "capture_meta.json"
    capture_meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    agent_outcome = (state or {}).get("agent_outcome") or capture_meta.get("agent_outcome") or "(none)"
    agent_summary = (state or {}).get("agent_summary") or capture_meta.get("agent_summary") or "(none)"

    steps = traj.get("steps", [])
    screenshots, image_labels = _verification_screenshots(traj_dir, steps)
    expected_actions = (capture_meta.get("task_meta") or {}).get("expected_actions") or "(unspecified)"

    prompt = VERIFY_PROMPT.format(
        instruction=traj.get("instruction", ""),
        n_steps=len(steps),
        expected_actions=expected_actions,
        step_summary=_step_summary(traj),
        agent_outcome=agent_outcome,
        agent_summary=agent_summary,
        image_labels=image_labels,
    )

    verdict = {"success": False, "efficiency": None, "accomplished": None, "reasoning": "verifier unavailable"}
    try:
        result_text = await run_agent(
            _verify_prompt_stream(prompt, screenshots),
            ClaudeAgentOptions(
                model=MODEL_FOR_REASONING,
                max_turns=1,
                max_budget_usd=VERIFY_BUDGET_USD,
                permission_mode="bypassPermissions",
            ),
            registry=registry,
            full_name=full_name,
            stage=6,
            purpose="verify",
            raise_on_error=False,
        )
        start, end = result_text.find("{"), result_text.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(result_text[start:end])
            verdict = {
                "success": bool(parsed.get("success", False)),
                "efficiency": parsed.get("efficiency"),
                "accomplished": parsed.get("accomplished"),
                "reasoning": str(parsed.get("reasoning", ""))[:500],
            }
    except UsageLimitReached:
        raise
    except Exception as e:
        logger.warning(f"verification failed for {traj_dir.name}: {e}")

    relabeled = False
    accomplished = (verdict.get("accomplished") or "").strip() if not verdict["success"] else ""
    # A meandering run is bad supervision even when the goal was technically met — the model
    # would learn the wandering, not the goal.
    if verdict["success"] and verdict.get("efficiency") != "meandering":
        traj["success"] = True
    elif accomplished and len(traj.get("steps", [])) >= RELABEL_MIN_STEPS and verdict.get("efficiency") != "meandering":
        traj["instruction"] = accomplished
        traj["success"] = True
        relabeled = True
    else:
        traj["success"] = False

    ui_steps = sum(1 for s in steps if s.get("actions") and s["actions"][0].get("type") != "done")
    expected = (capture_meta.get("task_meta") or {}).get("expected_actions") or 0
    capture_meta["verifier"] = {
        **verdict,
        "relabeled": relabeled,
        "ui_steps": ui_steps,
        "expected_actions": expected,
        "efficiency_ratio": round(ui_steps / expected, 2) if expected else None,
    }
    trajectory_file.write_text(json.dumps(traj, indent=2))
    meta_file.write_text(json.dumps(capture_meta, indent=2))
    return {**verdict, "relabeled": relabeled, "final_success": traj["success"]}
