"""The teacher's browser tools: one registry entry per tool, assembled into the in-process MCP server the driver sits behind."""

import json
import logging
from collections.abc import Awaitable, Callable
from functools import partial
from typing import NamedTuple

from cua.pipeline import operator_pii
from cua.pipeline.driver_session import DriverClient, DriverError, run_blocking
from cua.settings import CAPTURE

logger = logging.getLogger(__name__)

THOUGHT_SCHEMA = {
    "type": "string",
    "description": (
        "Your reasoning for this action: 1-3 first-person sentences anchored in the "
        "visible UI. Recorded as training supervision — never mention refs, tools, "
        "snapshots, or selectors."
    ),
}


class ToolContext(NamedTuple):
    """The per-task driver client and mutable state record every handler acts through."""

    client: DriverClient
    state: dict


class ToolSpec(NamedTuple):
    """One teacher tool: the name and description the model sees, its argument schema, and its handler."""

    name: str
    description: str
    schema: dict
    handler: Callable[[ToolContext, dict], Awaitable[dict]]


def action_schema(extra: dict, required: list[str]) -> dict:
    """An action tool's schema: the mandatory `thought` plus this tool's own arguments."""
    return {
        "type": "object",
        "properties": {"thought": THOUGHT_SCHEMA, **extra},
        "required": ["thought", *required],
    }


def text_result(payload: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": payload}], "is_error": is_error}


async def dispatch_action(ctx: ToolContext, payload: dict) -> dict:
    """Run one recorded driver action, refusing it when the task has ended, the budget is spent, or it carries operator identity."""
    state = ctx.state
    if state["completed"]:
        return text_result("task already completed — no further actions.", is_error=True)
    if state["actions"] >= CAPTURE.max_actions_per_task:
        return text_result(
            "action budget exhausted. Call task_complete now with an honest outcome.",
            is_error=True,
        )
    # Refuse BEFORE the driver runs: an operator-identity leak must never reach the recording.
    leaks = operator_pii.scan_text(json.dumps(payload, default=str))
    if leaks:
        return text_result(
            f"refused: that text contains a real person's identity ({', '.join(leaks)}). You are a "
            "disposable QA persona — use your persona's name and the <secret:...> placeholders, or "
            "an obviously fictional value. Never a real name, email address, or phone number.",
            is_error=True,
        )
    try:
        result = await run_blocking(ctx.client.act, payload)
    except DriverError as e:
        return text_result(f"action refused: {e}", is_error=True)
    except Exception as e:
        return text_result(f"driver failure: {str(e)[:300]}", is_error=True)
    state["actions"] += 1
    remaining = CAPTURE.max_actions_per_task - state["actions"]
    return text_result(
        f"done (step {result['step']}, {remaining} actions left)\nURL: {result['url']}\n\n{result['snapshot']}"
    )


async def handle_click(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(
        ctx,
        {
            "kind": "click",
            "ref": args["ref"],
            "thought": args.get("thought", ""),
            "button": args.get("button", "left"),
            "dblclick": bool(args.get("dblclick", False)),
        },
    )


async def handle_type(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(
        ctx,
        {
            "kind": "type",
            "ref": args["ref"],
            "text": args.get("text", ""),
            "thought": args.get("thought", ""),
            "press_enter": bool(args.get("press_enter", False)),
        },
    )


async def handle_press(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(
        ctx, {"kind": "press", "keys": args.get("keys", []), "thought": args.get("thought", "")}
    )


async def handle_scroll(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(ctx, {"kind": "scroll", "dy": args.get("dy", 500), "thought": args.get("thought", "")})


async def handle_drag(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(
        ctx, {"kind": "drag", "ref": args["ref"], "to_ref": args["to_ref"], "thought": args.get("thought", "")}
    )


async def handle_hover(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(ctx, {"kind": "hover", "ref": args["ref"], "thought": args.get("thought", "")})


async def handle_click_xy(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(
        ctx,
        {
            "kind": "click_xy",
            "x": args["x"],
            "y": args["y"],
            "thought": args.get("thought", ""),
            "button": args.get("button", "left"),
            "dblclick": bool(args.get("dblclick", False)),
        },
    )


async def handle_drag_path(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(
        ctx, {"kind": "drag_path", "path": args.get("path", []), "thought": args.get("thought", "")}
    )


async def handle_upload(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(
        ctx,
        {
            "kind": "upload",
            "ref": args["ref"],
            "filename": args.get("filename", "upload.txt"),
            "content": args.get("content"),
        },
    )


async def handle_select(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(
        ctx, {"kind": "select", "ref": args["ref"], "value": args.get("value", ""), "thought": args.get("thought", "")}
    )


async def handle_snapshot(ctx: ToolContext, args: dict) -> dict:
    try:
        result = await run_blocking(ctx.client.snapshot)
        return text_result(f"URL: {result['url']}\nTitle: {result['title']}\n\n{result['snapshot']}")
    except Exception as e:
        return text_result(f"snapshot failed: {str(e)[:300]}", is_error=True)


async def handle_screenshot(ctx: ToolContext, args: dict) -> dict:
    try:
        b64 = await run_blocking(ctx.client.screenshot_b64)
        return {"content": [{"type": "image", "data": b64, "mimeType": "image/jpeg"}]}
    except Exception as e:
        return text_result(f"screenshot failed: {str(e)[:300]}", is_error=True)


async def handle_wait(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(
        ctx, {"kind": "wait", "seconds": float(args.get("seconds", 1)), "thought": args.get("thought", "")}
    )


async def handle_navigate(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(
        ctx, {"kind": "navigate", "url": args.get("url", ""), "thought": args.get("thought", "")}
    )


async def _record_done_step(ctx: ToolContext, args: dict) -> None:
    """The terminal `done` step — final screenshot plus the agent's goal-is-met observation — which is the supervision that teaches WHEN to stop."""
    thought = (args.get("closing_thought") or "").strip() or (args.get("summary") or "").strip()
    try:
        await run_blocking(ctx.client.act, {"kind": "done", "thought": thought})
        ctx.state["actions"] += 1
        ctx.state["done_recorded"] = True
    except Exception as e:
        logger.warning(f"done-step recording failed: {e}")


async def handle_task_complete(ctx: ToolContext, args: dict) -> dict:
    outcome = args.get("outcome", "")
    # Guarded on done-recording, not `completed`: a watchdog DENY already set `completed` and that trajectory still needs its terminal stop supervision.
    if outcome == "completed" and not ctx.state.get("done_recorded"):
        await _record_done_step(ctx, args)
    ctx.state["completed"] = True
    ctx.state["agent_outcome"] = outcome
    ctx.state["agent_summary"] = args.get("summary", "")
    return text_result("task ended. Stop here — give a one-line closing message.")


async def handle_handoff(ctx: ToolContext, args: dict) -> dict:
    return await dispatch_action(
        ctx, {"kind": "handoff", "reason": args.get("reason", "login"), "thought": args.get("thought", "")}
    )


async def _request_approval(ctx: ToolContext, proposed: dict, risk: str) -> dict:
    """The watchdog's verdict on a proposed consequential action, judged against the live screen."""
    try:
        screen_b64 = await run_blocking(ctx.client.screenshot_b64)
    except Exception:
        screen_b64 = None
    state = ctx.state
    return await state["approval_policy"].get_approval(
        goal=state.get("instruction", ""),
        proposed_action=proposed,
        risk_category=risk,
        screenshot_b64=screen_b64,
        media_type="image/jpeg",
        autonomy=state.get("autonomy", "guarded"),
        auth_mode=state.get("auth_mode", "none"),
        env=state.get("env", "live"),
        registry=state.get("registry"),
        full_name=state.get("full_name", ""),
    )


async def _record_approval_event(ctx: ToolContext, verdict: dict) -> None:
    """The verdict as an observation step: no SFT target of its own, but threaded into the student's history so it learns to execute only after approval."""
    try:
        await run_blocking(
            ctx.client.act,
            {
                "kind": "approval_event",
                "event": {
                    "kind": "approval",
                    "decision": verdict["decision"],
                    "rationale": verdict.get("rationale", ""),
                    "by": "watchdog",
                },
            },
        )
    except Exception as e:
        logger.warning(f"approval_event recording failed: {e}")


async def handle_confirm(ctx: ToolContext, args: dict) -> dict:
    proposed = args.get("proposed_action") or {}
    risk = args.get("risk_category", "")
    result = await dispatch_action(
        ctx,
        {
            "kind": "confirm",
            "proposed_action": proposed,
            "risk_category": risk,
            "thought": args.get("thought", ""),
        },
    )
    # A refused confirm recorded no step, so a verdict would have nothing to attach to: never spend a watchdog call or emit an orphan event step.
    if result.get("is_error") or ctx.state["completed"]:
        return result
    verdict = await _request_approval(ctx, proposed, risk)
    await _record_approval_event(ctx, verdict)
    if verdict["decision"] == "approve":
        return text_result(
            f"Approved: {verdict.get('rationale', '')}. Now perform the proposed action with the matching tool."
        )
    # Deny is terminal: completing here (the agent paused correctly) is the success.
    ctx.state["completed"] = True
    return text_result(
        f"Denied: {verdict.get('rationale', '')}. Do NOT perform the action. "
        "Call task_complete with outcome='completed' — pausing here is the correct outcome."
    )


INPUT_TOOLS = (
    ToolSpec(
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
        handle_click,
    ),
    ToolSpec(
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
        handle_type,
    ),
    ToolSpec(
        "press",
        "Press a key or chord, e.g. keys=['Enter'] or ['Control','a'].",
        action_schema({"keys": {"type": "array", "items": {"type": "string"}}}, ["keys"]),
        handle_press,
    ),
    ToolSpec(
        "scroll",
        "Scroll the page vertically by dy pixels (positive = down, negative = up).",
        action_schema({"dy": {"type": "integer"}}, ["dy"]),
        handle_scroll,
    ),
    ToolSpec(
        "drag",
        "Drag one element onto another (kanban cards, sliders, reordering). Falls back to "
        "HTML5 drag events automatically when raw mouse dragging has no effect.",
        action_schema({"ref": {"type": "integer"}, "to_ref": {"type": "integer"}}, ["ref", "to_ref"]),
        handle_drag,
    ),
    ToolSpec(
        "hover",
        "Move the mouse over an element without clicking — opens hover menus and reveals "
        "hidden row controls (edit/delete buttons). Re-snapshot afterwards to see what appeared.",
        action_schema({"ref": {"type": "integer"}}, ["ref"]),
        handle_hover,
    ),
    ToolSpec(
        "click_xy",
        "Click at exact pixel coordinates (1280x720 viewport). ONLY for surfaces with no "
        "referenceable elements — canvases, maps, drawing areas. Read coordinates from a "
        "screenshot first. For normal elements always use click with a ref.",
        action_schema(
            {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right"]},
                "dblclick": {"type": "boolean"},
            },
            ["x", "y"],
        ),
        handle_click_xy,
    ),
    ToolSpec(
        "drag_path",
        "Press the mouse down and drag through a sequence of pixel coordinates. ONLY for "
        "canvas surfaces (drawing, lasso-select, panning a map). For element-to-element "
        "drags use drag with refs.",
        action_schema(
            {
                "path": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                        "required": ["x", "y"],
                    },
                },
            },
            ["path"],
        ),
        handle_drag_path,
    ),
    ToolSpec(
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
        handle_upload,
    ),
    ToolSpec(
        "select",
        "Choose an option in a native dropdown (<select>) by its value.",
        action_schema({"ref": {"type": "integer"}, "value": {"type": "string"}}, ["ref", "value"]),
        handle_select,
    ),
)

OBSERVATION_TOOLS = (
    ToolSpec(
        "snapshot",
        "Refresh the interactive-element view of the current page.",
        {},
        handle_snapshot,
    ),
    ToolSpec(
        "screenshot",
        "See the current page as an image (use sparingly; the snapshot is usually enough).",
        {},
        handle_screenshot,
    ),
)

NAVIGATION_TOOLS = (
    ToolSpec(
        "wait",
        "Deliberately wait up to 10 seconds for the page to settle (loading spinners, async "
        "updates, animations). This is a recorded action — knowing WHEN to wait is part of "
        "the skill being captured.",
        action_schema({"seconds": {"type": "number"}}, ["seconds"]),
        handle_wait,
    ),
    ToolSpec(
        "navigate",
        "Go to a URL (recorded as a step). Use it to reach a DIFFERENT site you legitimately "
        "need — e.g. opening your email inbox to read a verification or password-reset message — "
        "then act there normally. Do NOT use it to skip the app's own UI by jumping to deep links.",
        action_schema({"url": {"type": "string"}}, ["url"]),
        handle_navigate,
    ),
)

CONTROL_TOOLS = (
    ToolSpec(
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
        handle_task_complete,
    ),
)

GUARDED_CONTROL_TOOLS = (
    ToolSpec(
        "handoff",
        "Guarded mode only. Yield control for something you must NOT do yourself: logging in, "
        "entering a password or payment details, or solving a CAPTCHA / 2FA. reason: one of "
        "login|payment|captcha|2fa|consent. After the handoff, assume control returns and continue "
        "toward the goal.",
        action_schema({"reason": {"type": "string"}}, ["reason"]),
        handle_handoff,
    ),
    ToolSpec(
        "confirm",
        "Guarded mode only. When you are about to perform a CONSEQUENTIAL, hard-to-undo action "
        "(place an order, pay, send a message to a real person, publish, delete data, accept "
        "terms), DO NOT do it — call this to propose it and wait for a verdict. proposed_action: "
        'the action you would take, e.g. {"type":"click","ref":12}. risk_category: one of '
        "purchase|payment|send|publish|delete|consent|account_create|other. If approved, perform "
        "the action next; if denied, stop — pausing is the correct, successful outcome.",
        action_schema(
            {"proposed_action": {"type": "object"}, "risk_category": {"type": "string"}},
            ["proposed_action", "risk_category"],
        ),
        handle_confirm,
    ),
)


AUTONOMOUS_TOOLS = (*INPUT_TOOLS, *OBSERVATION_TOOLS, *NAVIGATION_TOOLS, *CONTROL_TOOLS)
# Guarded mode adds the pause-and-ask / takeover behaviors; every other autonomy drives straight through, so repo captures stay byte-for-byte unchanged.
TOOLS_BY_AUTONOMY = {
    "autonomous": AUTONOMOUS_TOOLS,
    "guarded": (*AUTONOMOUS_TOOLS, *GUARDED_CONTROL_TOOLS),
}


def build_driver_mcp_server(
    client: DriverClient, state: dict, server_name: str = "browser", autonomy: str = "autonomous"
):
    """In-process MCP server exposing the in-container driver to the teacher, with `state` the per-task record the handlers read and mutate."""
    # lazy: the server factory and the @tool wrapper are needed only once a server is actually built.
    from claude_agent_sdk import create_sdk_mcp_server, tool

    ctx = ToolContext(client, state)
    specs = TOOLS_BY_AUTONOMY.get(autonomy, AUTONOMOUS_TOOLS)
    tools = [tool(spec.name, spec.description, spec.schema)(partial(spec.handler, ctx)) for spec in specs]
    server = create_sdk_mcp_server(name=server_name, version="1.0.0", tools=tools)
    allowed = [f"mcp__{server_name}__{spec.name}" for spec in specs]
    return server, allowed
