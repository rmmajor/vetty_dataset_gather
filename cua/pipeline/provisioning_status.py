"""Persistent site-name to provisioning-status store, which routes a site between the autonomous and human lanes."""

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
    """Remove a site's entry, returning True when there was one."""
    data = load()
    if site_name not in data:
        return False
    del data[site_name]
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return True


def mark(site_name: str, status: str, reason: str = "", provisioning: str = ""):
    """Upsert a site's status and bump its attempt counter."""
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
