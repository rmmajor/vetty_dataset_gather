"""The one entry point for a Claude Agent SDK call that is also metered: one `agent_calls` ledger row per call."""

import asyncio
import logging

from claude_agent_sdk import AssistantMessage, RateLimitEvent, ResultMessage, query

from cua.pipeline.schemas import FailureCategory, RetryableFailure
from cua.pipeline.sdk_runner import (
    UsageLimitReached,
    cost_tracker,
    harden_options,
    rate_limit_tracker,
)
from cua.settings import SDK

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
    timeout_s: float = SDK.query_timeout_s,
) -> str:
    """Run one agent query, record its telemetry row, and return the model's text result."""
    # Bailing before the call keeps a quota-exhausted stage from rejecting repos it never actually tried.
    if rate_limit_tracker.limit_reached():
        raise UsageLimitReached(f"usage limit reached before {purpose or 'agent call'}")

    result_text = ""
    last_assistant_text = ""
    result_msg = None

    harden_options(options)

    try:
        async with asyncio.timeout(timeout_s):
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
    except TimeoutError:
        # A wedged CLI subprocess is a soft stop: no ResultMessage arrived, so there is no telemetry row to record.
        logger.warning(f"agent[{purpose}] {full_name} exceeded {timeout_s}s — abandoning the call")
        return result_text or last_assistant_text
    except Exception as e:
        # Budget is informational only: runaway protection is max_turns and the per-task action cap, never dollars.
        if "maximum budget" in str(e).lower():
            logger.info(
                f"agent[{purpose}] {full_name} reached its per-call budget cap "
                f"(informational — not treated as a failure)"
            )
            return result_text
        raise

    if result_msg is not None and registry is not None:
        registry.record_agent_call(
            full_name,
            stage,
            purpose,
            getattr(options, "model", ""),
            result_msg.num_turns,
            result_msg.total_cost_usd,
            result_msg.usage,
            result_msg.subtype,
            result_msg.is_error,
            result_msg.duration_ms,
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
                FailureCategory.MAX_TURNS if result_msg.subtype == "error_max_turns" else FailureCategory.SDK_ERROR
            )
            raise RetryableFailure(category, f"agent ended with {result_msg.subtype}")

    return result_text
