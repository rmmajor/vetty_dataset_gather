"""Per-repo stage runner: a stage signals its outcome by what it raises, and this applies the retry policy."""

import asyncio
import logging

from cua.pipeline.logging_utils import repo_logger
from cua.pipeline.schemas import (
    RETRYABLE_CATEGORIES,
    FailureCategory,
    QualityReject,
    RetryableFailure,
)
from cua.pipeline.sdk_runner import UsageLimitReached
from cua.settings import PIPELINE

logger = logging.getLogger(__name__)


async def run_repo(stage: int, repo: dict, registry, core_fn, progress=None) -> str:
    name = repo["full_name"]
    rlog = repo_logger(stage, name)
    if progress:
        progress.start()

    for attempt in range(1, PIPELINE.max_retries + 1):
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
        if retryable and attempt < PIPELINE.max_retries:
            registry.record_retry(name, stage, category, reason, attempt)
            rlog.warning(f"RETRY {attempt}/{PIPELINE.max_retries} [{category.value}] {reason}")
            if progress:
                progress.retry()
            if PIPELINE.retry_backoff:
                await asyncio.sleep(PIPELINE.retry_backoff * attempt)
            continue

        final_reason = f"{reason} (after {attempt} attempts)" if retryable else reason
        registry.reject(name, stage, final_reason, category=category, retry_count=attempt, **extra)
        rlog.warning(f"REJECT(infra) [{category.value}] {final_reason}")
        if progress:
            progress.rejected_repo(infra=True)
        return "rejected_infra"
