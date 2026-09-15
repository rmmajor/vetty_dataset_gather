"""Per-repo stage runner with retry-on-infra.

Stages no longer reject directly. Their `core_fn(registry, repo)` does the work and signals the
outcome by what it raises:

  * returns normally  -> passed (it set the repo's passed status itself; may return a trajectory count)
  * raises QualityReject -> the repo genuinely failed a gate; terminal, never retried
  * raises RetryableFailure (or any other Exception) -> infra / flakiness; retried up to MAX_RETRIES,
    then rejected with the infra category so it stays distinguishable and re-queueable.

Centralizing this here means every stage gets identical retry, single-point registry writes, and
inline reject/retry logging for free.
"""

import asyncio
import logging
import os

from cua.pipeline.logging_utils import repo_logger
from cua.pipeline.schemas import (
    MAX_RETRIES,
    RETRYABLE_CATEGORIES,
    FailureCategory,
    QualityReject,
    RetryableFailure,
)
from cua.pipeline.sdk_runner import UsageLimitReached

logger = logging.getLogger(__name__)

# Linear backoff between attempts (base * attempt seconds). Set PIPELINE_RETRY_BACKOFF=0 to disable
# the wait (used in fast verification runs).
RETRY_BACKOFF_BASE = float(os.environ.get("PIPELINE_RETRY_BACKOFF", "5"))


async def run_repo(stage: int, repo: dict, registry, core_fn, progress=None) -> str:
    name = repo["full_name"]
    rlog = repo_logger(stage, name)
    if progress:
        progress.start()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            trajectories = await core_fn(registry, repo)
            if progress:
                progress.passed_repo(trajectories or 0)
            return "passed"
        except QualityReject as e:
            registry.reject(name, stage, e.reason, category=e.category, **e.extra)
            rlog.info(f"REJECT [{e.category.value}] {e.reason}")
            if progress:
                progress.rejected_repo(infra=False)
            return "rejected"
        except UsageLimitReached:
            # Subscription quota hit — never reject; abort so the repo resumes next run.
            raise
        except RetryableFailure as e:
            category, reason, extra = e.category, e.reason, e.extra
        except Exception as e:  # unclassified crash -> treat as a retryable sandbox error
            category, reason, extra = FailureCategory.SANDBOX_ERROR, str(e)[:200], {}

        # Reached only for infra/flaky failures (the success and quality paths return above).
        retryable = category in RETRYABLE_CATEGORIES
        if retryable and attempt < MAX_RETRIES:
            registry.record_retry(name, stage, category, reason, attempt)
            rlog.warning(f"RETRY {attempt}/{MAX_RETRIES} [{category.value}] {reason}")
            if progress:
                progress.retry()
            if RETRY_BACKOFF_BASE:
                await asyncio.sleep(RETRY_BACKOFF_BASE * attempt)
            continue

        final_reason = f"{reason} (after {attempt} attempts)" if retryable else reason
        registry.reject(name, stage, final_reason, category=category, retry_count=attempt, **extra)
        rlog.warning(f"REJECT(infra) [{category.value}] {final_reason}")
        if progress:
            progress.rejected_repo(infra=True)
        return "rejected_infra"
