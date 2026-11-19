"""One teacher task end to end: open a driver session, let the teacher act, finalize the recording in-container."""

import logging

from claude_agent_sdk import ClaudeAgentOptions

from cua.pipeline.agent_runner import run_agent
from cua.pipeline.driver_session import DriverClient, run_blocking
from cua.pipeline.llm_config import MODEL_FOR_TEACHER_AGENT
from cua.pipeline.persona import default_persona, persona_secrets
from cua.pipeline.safety_watchdog import get_policy
from cua.pipeline.teacher.policy import TEACHER_PROMPT, render_policy_block
from cua.pipeline.teacher.tools import build_driver_mcp_server
from cua.settings import CAPTURE

logger = logging.getLogger(__name__)


def _task_state(task: dict, registry, full_name: str, autonomy: str, auth_mode: str, env: str) -> dict:
    """The per-task record the tool handlers mutate and the verifier reads back."""
    return {
        "actions": 0,
        "completed": False,
        "agent_outcome": "",
        "agent_summary": "",
        "instruction": task["instruction"],
        "autonomy": autonomy,
        "auth_mode": auth_mode,
        "env": env,
        "registry": registry,
        "full_name": full_name,
        "approval_policy": get_policy(),
    }


def _task_meta(task: dict, autonomy: str, auth_mode: str, env: str, persona: dict, task_style: str) -> dict:
    """What the driver records alongside the trajectory, and what the verifier later reads its lane and budget from."""
    return {
        "difficulty": task.get("difficulty", ""),
        "expected_actions": task.get("expected_actions", 0),
        "autonomy": autonomy,
        "auth_mode": auth_mode,
        "env": env,
        "persona": persona["username"],
        "task_style": task_style,
    }


def _teacher_prompt(task: dict, session: dict, policy_block: str) -> str:
    return TEACHER_PROMPT.format(
        instruction=task["instruction"],
        url=session["url"],
        snapshot=session["snapshot"],
        max_actions=CAPTURE.max_actions_per_task,
        policy_block=policy_block,
    )


def _teacher_options(server, allowed: list[str]) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model=MODEL_FOR_TEACHER_AGENT,
        max_turns=CAPTURE.max_turns,
        max_budget_usd=CAPTURE.task_budget_usd,
        permission_mode="bypassPermissions",
        mcp_servers={"browser": server},
        allowed_tools=allowed,
        disallowed_tools=["Bash"],
    )


async def _finalize_recording(client: DriverClient, task: dict, state: dict, final_text: str, capture_session: bool):
    """Land the trajectory and final screenshot on disk even when the teacher never called task_complete."""
    if not state["completed"]:
        state["agent_outcome"] = state["agent_outcome"] or "unreported"
        state["agent_summary"] = state["agent_summary"] or (final_text or "").strip()[:400]
    # Captured BEFORE finish closes the context, so a register or login trajectory's session is reusable.
    if capture_session:
        try:
            state["storage_state"] = await run_blocking(client.storage_state)
        except Exception as e:
            logger.warning(f"session capture failed for {task['id']}: {e}")
            state["storage_state"] = None
    try:
        await run_blocking(client.finish, state["agent_outcome"], state["agent_summary"])
    except Exception as e:
        logger.warning(f"driver finish failed for {task['id']}: {e}")


async def run_teacher_task(
    client: DriverClient,
    task: dict,
    app_url: str,
    registry=None,
    full_name: str = "",
    autonomy: str = "autonomous",
    auth_mode: str = "none",
    env: str = "live",
    secrets: dict | None = None,
    auth_config: dict | None = None,
    mail_config: dict | None = None,
    webmail_url: str = "",
    webmail_preauthed: bool = True,
    capture_session: bool = False,
    persona: dict | None = None,
    task_style: str = "",
) -> dict:
    """Run one task end-to-end and return its state record, in the mode `autonomy`, `auth_mode` and `env` pick."""
    persona = persona or default_persona()
    state = _task_state(task, registry, full_name, autonomy, auth_mode, env)
    # A lane's provisioned credential always wins over the persona's fallback under the same key.
    session_secrets = {**persona_secrets(persona), **(secrets or {})}
    session = await run_blocking(
        lambda: client.session(
            task["id"],
            app_url,
            task["instruction"],
            task_meta=_task_meta(task, autonomy, auth_mode, env, persona, task_style),
            auth_config=auth_config,
            secrets=session_secrets,
            mail_config=mail_config,
        )
    )
    server, allowed = build_driver_mcp_server(client, state, autonomy=autonomy)
    policy_block = render_policy_block(
        autonomy,
        auth_mode,
        list((secrets or {}).keys()),
        webmail_url=((mail_config or {}).get("url") or webmail_url),
        webmail_preauthed=webmail_preauthed,
        persona=persona,
        task_style=task_style,
    )
    prompt = _teacher_prompt(task, session, policy_block)
    final_text = ""
    try:
        final_text = await run_agent(
            prompt,
            _teacher_options(server, allowed),
            registry=registry,
            full_name=full_name,
            stage=6,
            purpose="teacher_task",
            raise_on_error=False,
        )
    finally:
        await _finalize_recording(client, task, state, final_text, capture_session)
    return state
