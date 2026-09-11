"""Crash-safe record of the instances this machine owns, so teardown survives a hard process death.

An instance is recorded the moment it is created, stamped with the pid of the run that owns it.
If that orchestrator is killed (-9, power loss, closed laptop) before `destroy`, the record stays
behind and `cleanup` tears the box down. The pid is what separates an ORPHAN (owner process gone)
from a box a CONCURRENT run is actively using — several runs share one state file whenever they
share a working directory, and destroying a live sibling's GPU is as bad as destroying a
stranger's. Records are therefore a list, never a single slot that later runs overwrite.

Path is overridable via `REMOTE_TRAIN_STATE_FILE`.
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILE = Path(os.environ.get("REMOTE_TRAIN_STATE_FILE", ".remote_train_state.json"))


def record_instance(instance) -> None:
    records = [r for r in read_instances() if str(r.get("id")) != str(instance.id)]
    records.append(
        {
            "provider": instance.provider,
            "id": instance.id,
            "label": instance.label,
            "pid": os.getpid(),
            "started_at": time.time(),
        }
    )
    _write(records)
    logger.info(f"Recorded instance {instance.id} to {STATE_FILE} for crash-safe teardown")


def read_instances() -> list:
    if not STATE_FILE.exists():
        return []
    try:
        data = json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"State file {STATE_FILE} unreadable: {e}")
        return []
    if isinstance(data, dict):
        return [data] if data.get("id") else []  # legacy single-slot file
    return [r for r in data if isinstance(r, dict) and r.get("id")]


def owner_alive(record) -> bool:
    """Is the run that recorded this instance still running? `pid: null` is the released sentinel; an UNKNOWN owner counts as alive because a leaked box is reclaimable and a destroyed live job is not."""
    if "pid" in record and record["pid"] is None:
        return False
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return True
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as e:
        logger.warning(f"Could not probe owner pid {pid} ({e}) — assuming the run is alive")
        return True
    return True


def mark_abandoned(instance_id) -> None:
    """Release ownership of a recorded instance so `cleanup` reclaims it: writes the `pid: null` sentinel."""
    records = read_instances()
    for r in records:
        if str(r.get("id")) == str(instance_id):
            r["pid"] = None
    _write(records)
    logger.warning(f"Instance {instance_id} released — `python -m remote_train cleanup` will destroy it")


def clear_instance(instance_id=None) -> None:
    """Forget one instance, or (with no id) every instance this process owns."""
    keep = []
    for r in read_instances():
        if instance_id is not None:
            drop = str(r.get("id")) == str(instance_id)
        else:
            drop = r.get("pid") == os.getpid()
        if not drop:
            keep.append(r)
    if keep:
        _write(keep)
        return
    try:
        STATE_FILE.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"Could not remove state file {STATE_FILE}: {e}")


def _write(records) -> None:
    try:
        STATE_FILE.write_text(json.dumps(records))
    except OSError as e:
        logger.error(f"Could not write state file {STATE_FILE}: {e}")


BAD_OFFER_FILE = Path(os.environ.get("REMOTE_TRAIN_BAD_OFFERS", ".remote_train_bad_offers.json"))
BAD_OFFER_TTL_S = 6 * 3600


def record_bad_offer(offer_id, reason) -> None:
    """Remember an offer that took our money and never booted, so a retry picks a different host."""
    entries = {k: v for k, v in _read_bad_offers().items()}
    entries[str(offer_id)] = {"at": time.time(), "reason": str(reason)[:200]}
    try:
        BAD_OFFER_FILE.write_text(json.dumps(entries))
        logger.warning(f"offer {offer_id} blocklisted for {BAD_OFFER_TTL_S // 3600}h: {reason}")
    except OSError as e:
        logger.warning(f"could not record bad offer {offer_id}: {e}")


def bad_offer_ids() -> set:
    return {k for k, v in _read_bad_offers().items() if time.time() - v.get("at", 0) < BAD_OFFER_TTL_S}


def _read_bad_offers() -> dict:
    if not BAD_OFFER_FILE.exists():
        return {}
    try:
        data = json.loads(BAD_OFFER_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}
