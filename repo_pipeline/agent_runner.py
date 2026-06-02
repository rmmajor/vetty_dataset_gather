"""Single entry point for every Claude Agent SDK call in the pipeline.

Before this, each stage hand-rolled the same `async for message in query(...)` loop and threw
away the `ResultMessage` telemetry (turns, cost, tokens). `run_agent` centralizes that loop so:

  * every call records one `agent_calls` row (turns / cost / tokens / subtype) for the dashboard,
  * a non-success result is classified (`error_max_turns` -> MAX_TURNS, else SDK_ERROR) and, when
    `raise_on_error` is set, surfaced as a `RetryableFailure` so the retry wrapper can recover it
    instead of mislabeling it a quality rejection.

Stages whose agent call is critical (scoring, deep analysis) set `raise_on_error=True`. Stages
with a graceful per-item fallback (instruction / think generation) leave it False and use whatever
text came back.
"""

import logging
import os
import shutil

from claude_agent_sdk import AssistantMessage, ResultMessage, query

from repo_pipeline.schemas import FailureCategory, RetryableFailure

logger = logging.getLogger(__name__)

# The SDK always prefers its bundled CLI, which can lag the system install and has wedged on
# subprocess spawn. Pin to the (newer, session-proven) system `claude` when available; override
# with CLAUDE_CLI_PATH. None -> fall back to the SDK's default (bundled) discovery.
CLAUDE_CLI_PATH = os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude")


async def run_agent(
    prompt,
    options,
    *,
    registry=None,
    full_name: str = "",
    stage: int = 0,
    purpose: str = "",
    capture_assistant_fallback: bool = False,
    raise_on_error: bool = False,
) -> str:
    """Run one agent query, record telemetry, and return the model's text result.

    ``capture_assistant_fallback`` keeps the last assistant message so a result that arrives in the
    final turn (rather than as a tool-free ``ResultMessage.result``) is still recovered.
    """
    result_text = ""
    last_assistant_text = ""
    result_msg = None

    if CLAUDE_CLI_PATH and not getattr(options, "cli_path", None):
        options.cli_path = CLAUDE_CLI_PATH

    async for message in query(prompt=prompt, options=options):
        if capture_assistant_fallback and isinstance(message, AssistantMessage):
            for block in getattr(message, "content", None) or []:
                if hasattr(block, "text"):
                    last_assistant_text = block.text
        if isinstance(message, ResultMessage):
            result_msg = message
            if message.subtype == "success":
                result_text = message.result or ""

    if result_msg is not None and registry is not None:
        registry.record_agent_call(
            full_name, stage, purpose, getattr(options, "model", ""),
            result_msg.num_turns, result_msg.total_cost_usd, result_msg.usage,
            result_msg.subtype, result_msg.is_error, result_msg.duration_ms,
        )

    if not result_text and last_assistant_text:
        result_text = last_assistant_text

    if result_msg is not None and result_msg.subtype != "success":
        logger.warning(f"agent[{purpose}] {full_name} ended with: {result_msg.subtype}")
        if raise_on_error and not result_text:
            category = (
                FailureCategory.MAX_TURNS
                if result_msg.subtype == "error_max_turns"
                else FailureCategory.SDK_ERROR
            )
            raise RetryableFailure(category, f"agent ended with {result_msg.subtype}")

    return result_text
