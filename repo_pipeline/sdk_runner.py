"""Shared Claude Agent SDK `query()` wrapper + subscription rate-limit tracking.

Every `query()` call in the pipeline goes through `run_query`, so there is a single
place that observes subscription usage. The SDK (claude-agent-sdk >= 0.2.x) emits a
`RateLimitEvent` carrying `RateLimitInfo` into the message stream whenever usage state
changes, and sets `ResultMessage.api_error_status` to 429 on a hard rate-limit. We turn
both into a process-global signal (`rate_limit_tracker`) that the orchestrator consults
between stages, and we raise `UsageLimitReached` so a quota-exhausted call NEVER causes a
repo to be rejected — the repo stays in its input status and is retried on the next run.
"""
import logging
import os
import shutil

from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    query,
)

logger = logging.getLogger(__name__)

# The SDK always prefers its bundled CLI, which can lag the system install and has wedged on
# subprocess spawn. Pin to the (newer, session-proven) system `claude` when available; override
# with CLAUDE_CLI_PATH. None -> fall back to the SDK's default (bundled) discovery.
CLAUDE_CLI_PATH = os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude")


def harden_options(options) -> None:
    """Apply pipeline-wide option defaults in place: pin the system CLI, and strip filesystem
    settings from headless calls. Without `setting_sources=[]` the spawned CLI loads the
    OPERATOR'S user-level plugins/skills into every call's system prompt — measured at +20k
    cache-creation tokens per call after a plugin install (bare `claude -p pong` = $0.19),
    which silently tripled the fixed cost of every query and blew through per-call budgets.
    Pipeline prompts are self-contained; they must not inherit the operator's environment."""
    if CLAUDE_CLI_PATH and not getattr(options, "cli_path", None):
        options.cli_path = CLAUDE_CLI_PATH
    if getattr(options, "setting_sources", None) is None:
        options.setting_sources = []


class UsageLimitReached(Exception):
    """The Claude subscription usage limit has been reached.

    Callers MUST let this propagate without rejecting the repo (it is not a content
    or infrastructure failure — the work simply could not run). It is distinct from
    the generic exceptions each stage catches-and-rejects.
    """


class RateLimitTracker:
    """Process-global view of Claude subscription rate-limit state.

    Utilization is tracked per `rate_limit_type` because the CLI interleaves events
    for different windows (e.g. `five_hour` and `seven_day`); a single "latest" slot
    would flip-flop between windows with different utilizations. `limit_reached` fires
    when ANY tracked window crosses `threshold`, or a hard 429 marked the limit rejected.

    All methods are synchronous with no `await`, so they run atomically under the single
    asyncio event loop — concurrent workers call them without locking.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.observed_event_count = 0
        self._by_type: dict[str, RateLimitInfo] = {}
        self._rejected = False

    def observe(self, info: RateLimitInfo) -> None:
        self.observed_event_count += 1
        self._by_type[info.rate_limit_type or "_unknown"] = info
        if info.status == "rejected":
            self._rejected = True

    def mark_rejected(self) -> None:
        self._rejected = True

    def limit_reached(self) -> bool:
        if self._rejected:
            return True
        return any(
            info.utilization is not None and info.utilization >= self.threshold
            for info in self._by_type.values()
        )

    def max_utilization(self) -> float | None:
        utilizations = [
            info.utilization for info in self._by_type.values() if info.utilization is not None
        ]
        return max(utilizations) if utilizations else None

    def latest(self) -> RateLimitInfo | None:
        """The most report-worthy window: highest utilization seen, else any window, else None."""
        if not self._by_type:
            return None
        with_util = [info for info in self._by_type.values() if info.utilization is not None]
        if with_util:
            return max(with_util, key=lambda info: info.utilization)
        return next(iter(self._by_type.values()))

    def reset(self) -> None:
        self.observed_event_count = 0
        self._by_type.clear()
        self._rejected = False


rate_limit_tracker = RateLimitTracker()


class CostTracker:
    """Process-global accumulator of ACTUAL spend, fed by every recorded ResultMessage's
    total_cost_usd. The orchestrator's --budget used to be paced by per-repo guesses
    (STAGE_COST_ESTIMATES), which drifted badly from reality — a 200-repo stage-3 pass
    'spent' $30 of estimate while the subscription window sat at 10% — and stopped runs
    that had barely started. Budget checks now read this; estimates only pre-size batches.

    Synchronous, no awaits — atomic under the single asyncio event loop."""

    def __init__(self):
        self.total_usd = 0.0
        self.call_count = 0

    def add(self, cost_usd) -> None:
        if cost_usd:
            self.total_usd += float(cost_usd)
            self.call_count += 1

    def reset(self) -> None:
        self.total_usd = 0.0
        self.call_count = 0


cost_tracker = CostTracker()


async def run_query(
    prompt,
    options,
    *,
    label: str = "",
    fallback_to_assistant_text: bool = False,
) -> str:
    """Run one SDK `query()` and return its success result text.

    Besides extracting the result, this observes subscription usage: every
    `RateLimitEvent` updates `rate_limit_tracker`, and a hard 429 on the terminal
    `ResultMessage` marks the limit rejected. When the limit is (or becomes) reached
    it raises `UsageLimitReached` so the caller aborts WITHOUT rejecting the repo.

    Non-rate-limit failures (timeout, budget exhaustion, 5xx, max-turns) are NOT raised
    — they leave the result empty and the caller's existing fallback/parse logic handles
    them exactly as before. `fallback_to_assistant_text=True` returns the last assistant
    text block when the success result was empty (stage 3's scoring path relies on this).
    """
    if rate_limit_tracker.limit_reached():
        raise UsageLimitReached(f"usage limit reached before query ({label})")

    harden_options(options)

    result_text = ""
    last_assistant_text = ""

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, RateLimitEvent):
            rate_limit_tracker.observe(message.rate_limit_info)
            continue

        if fallback_to_assistant_text and isinstance(message, AssistantMessage) and hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    last_assistant_text = block.text

        if isinstance(message, ResultMessage):
            cost_tracker.add(getattr(message, "total_cost_usd", None))
            if message.subtype == "success" and not message.is_error:
                result_text = message.result or ""
            else:
                status = getattr(message, "api_error_status", None)
                logger.warning(
                    f"query ({label}) ended with subtype={message.subtype} api_error_status={status}"
                )
                if status == 429:
                    rate_limit_tracker.mark_rejected()
                    raise UsageLimitReached(f"HTTP 429 from query ({label})")

    if not result_text and fallback_to_assistant_text and last_assistant_text:
        result_text = last_assistant_text

    return result_text
