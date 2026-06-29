"""Crash-safe record of the instance a run owns, so teardown survives a hard process death.

The instance id is written here the instant it is created. If the orchestrator is killed
(-9, power loss, closed laptop) before `destroy`, the `cleanup` command reads this file —
and also scans the provider for any instance carrying the run label — and destroys the
orphans. This is the backstop behind the in-process try/finally + signal traps in
`lifecycle.py`. Path is overridable via `REMOTE_TRAIN_STATE_FILE`.
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILE = Path(os.environ.get("REMOTE_TRAIN_STATE_FILE", ".remote_train_state.json"))


def record_instance(instance) -> None:
    data = {"provider": instance.provider, "id": instance.id, "label": instance.label}
    try:
        STATE_FILE.write_text(json.dumps(data))
        logger.info(f"Recorded instance {instance.id} to {STATE_FILE} for crash-safe teardown")
    except OSError as e:
        logger.error(f"Could not write state file {STATE_FILE}: {e}")


def read_instance() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"State file {STATE_FILE} unreadable: {e}")
        return None


def clear_instance() -> None:
    try:
        STATE_FILE.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"Could not remove state file {STATE_FILE}: {e}")
