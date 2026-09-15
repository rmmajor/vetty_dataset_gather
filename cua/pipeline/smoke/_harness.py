"""The one assertion style every smoke shares: a check returns its failures, the harness prints and scores them."""

import functools
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def with_tempdir(check):
    """Give a check a fresh work directory it never has to create or clean up."""

    @functools.wraps(check)
    def wrapped() -> list[str]:
        with tempfile.TemporaryDirectory(prefix=f"{check.__name__}_") as tmp:
            return check(Path(tmp))

    return wrapped


def run_checks(checks: list, title: str = "smoke") -> int:
    """Run every check, report ok/FAIL per check plus the failures, and return the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    failures: list[str] = []
    for check in checks:
        try:
            found = check()
        except Exception as e:
            logger.exception(f"{check.__name__} raised: {e!r}")
            found = [f"raised {e!r}"]
        logger.info(f"[{'FAIL' if found else 'ok'}] {check.__name__}")
        failures.extend(f"{check.__name__}: {failure}" for failure in found)
    for failure in failures:
        logger.error(f"  {failure}")
    logger.info(f"{title} {'PASSED' if not failures else 'FAILED'} ({len(checks)} checks, {len(failures)} failures)")
    return 1 if failures else 0
