"""Persistent per-site provisioning status for the login-gated lanes.

Records whether a site's burner account could be created autonomously. The autonomous lane
(`capture_logingated`) marks `registered` on success and `needs_human` when registration is blocked
(captcha / phone / disposable-email rejection); the human-in-the-loop provisioning lane consumes the
`needs_human` queue, and future autonomous runs skip those sites so they are not re-failed for cost.

Generic site-name -> status store (status is just a string), so adding a status or a consumer is a
data change, not a code change.
"""
import json
from pathlib import Path

STATUS_PATH = Path("pipeline_data/provisioning_status.json")

NEEDS_HUMAN = "needs_human"
REGISTERED = "registered"


def load() -> dict:
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def get_status(site_name: str) -> str | None:
    return (load().get(site_name) or {}).get("status")


def sites_with_status(status: str) -> list[str]:
    return [name for name, entry in load().items() if entry.get("status") == status]


def clear(site_name: str) -> bool:
    """Remove a site's entry (e.g. a needs_human mark that turned out to be a false negative, or a
    site a human has since provisioned). Returns True if an entry was removed."""
    data = load()
    if site_name not in data:
        return False
    del data[site_name]
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return True


def mark(site_name: str, status: str, reason: str = "", provisioning: str = ""):
    """Upsert a site's status (increments an attempt counter). Synchronous, called from the asyncio
    event-loop thread so concurrent capture coroutines do not truly overlap this read-modify-write."""
    data = load()
    entry = data.get(site_name, {})
    entry["status"] = status
    entry["reason"] = reason
    if provisioning:
        entry["provisioning"] = provisioning
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    data[site_name] = entry
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
