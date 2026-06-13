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

from claude_agent_sdk import AssistantMessage, RateLimitEvent, ResultMessage, query

from repo_pipeline.schemas import FailureCategory, RetryableFailure
from repo_pipeline.sdk_runner import (
    CLAUDE_CLI_PATH,
    UsageLimitReached,
    cost_tracker,
    harden_options,
    rate_limit_tracker,
)

logger = logging.getLogger(__name__)


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
    # Subscription rate limit: never reject a repo for quota. Bail before the call if the shared
    # tracker already saw the limit, and turn a hard 429 into UsageLimitReached (which the retry
    # wrapper re-raises, so the repo stays in its input status and resumes next run).
    if rate_limit_tracker.limit_reached():
        raise UsageLimitReached(f"usage limit reached before {purpose or 'agent call'}")

    result_text = ""
    last_assistant_text = ""
    result_msg = None

    harden_options(options)

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, RateLimitEvent):
                rate_limit_tracker.observe(message.rate_limit_info)
                continue
            if capture_assistant_fallback and isinstance(message, AssistantMessage):
                for block in getattr(message, "content", None) or []:
                    if hasattr(block, "text"):
                        last_assistant_text = block.text
            if isinstance(message, ResultMessage):
                result_msg = message
                cost_tracker.add(getattr(message, "total_cost_usd", None))
                if message.subtype == "success" and not message.is_error:
                    result_text = message.result or ""
    except UsageLimitReached:
        raise
    except Exception as e:
        # Budget is INFORMATIONAL ONLY — it must NEVER reject a repo or discard a trajectory.
        # The SDK surfaces a per-call budget cap as an exception ("...Reached maximum budget").
        # Treat it as a soft stop: return whatever was produced (the teacher's partial
        # trajectory is already finalized by its own finally block). Runaway protection comes
        # from max_turns + the per-task action cap, not from dollars.
        if "maximum budget" in str(e).lower():
            logger.info(f"agent[{purpose}] {full_name} reached its per-call budget cap "
                        f"(informational — not treated as a failure)")
            return result_text
        raise

    if result_msg is not None and registry is not None:
        registry.record_agent_call(
            full_name, stage, purpose, getattr(options, "model", ""),
            result_msg.num_turns, result_msg.total_cost_usd, result_msg.usage,
            result_msg.subtype, result_msg.is_error, result_msg.duration_ms,
        )

    if not result_text and last_assistant_text:
        result_text = last_assistant_text

    if result_msg is not None and not (result_msg.subtype == "success" and not result_msg.is_error):
        status = getattr(result_msg, "api_error_status", None)
        logger.warning(f"agent[{purpose}] {full_name} ended with: {result_msg.subtype} api_error_status={status}")
        if status == 429:
            rate_limit_tracker.mark_rejected()
            raise UsageLimitReached(f"HTTP 429 from agent ({purpose})")
        if raise_on_error and not result_text:
            category = (
                FailureCategory.MAX_TURNS
                if result_msg.subtype == "error_max_turns"
                else FailureCategory.SDK_ERROR
            )
            raise RetryableFailure(category, f"agent ended with {result_msg.subtype}")

    return result_text
