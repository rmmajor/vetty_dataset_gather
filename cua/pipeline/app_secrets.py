"""Boot secrets for a sandboxed app, plus the fail-closed registries of known test accounts and burner identities."""

import json
import logging
from pathlib import Path

from cua.settings import PIPELINE

logger = logging.getLogger(__name__)

# Fixed, not random: long enough for a minimum-key-length assert, and stable so a captured session survives a resume.
_DUMMY_SECRET = "qa_test_secret_0123456789abcdef0123456789abcdef"

# A superset is harmless here: an app simply ignores the names it does not read.
_BOOT_ENV_SECRETS = {
    "JWT_SECRET": _DUMMY_SECRET,
    "JWT_REFRESH_SECRET": _DUMMY_SECRET,
    "ACCESS_TOKEN_SECRET": _DUMMY_SECRET,
    "REFRESH_TOKEN_SECRET": _DUMMY_SECRET,
    "AUTH_SECRET": _DUMMY_SECRET,
    "NEXTAUTH_SECRET": _DUMMY_SECRET,
    "SESSION_SECRET": _DUMMY_SECRET,
    "COOKIE_SECRET": _DUMMY_SECRET,
    "TOKEN_SECRET": _DUMMY_SECRET,
    "SECRET": _DUMMY_SECRET,
    "APP_SECRET": _DUMMY_SECRET,
    "JWT_EXPIRES_IN": "1d",
    "AUTH_TRUST_HOST": "true",
    # Absolute links, emailed verification URLs included, are built from this, and the app binds this port in-container.
    "NEXTAUTH_URL": "http://127.0.0.1:3000",
    "AUTH_URL": "http://127.0.0.1:3000",
}

_CRED_DIR = Path(PIPELINE.credentials_dir)
_KNOWN_ACCOUNTS_FILE = _CRED_DIR / "known_accounts.json"
# Real-site logins the capture is permitted to use, plus the saved sessions that start a run already signed in.
_BURNER_FILE = _CRED_DIR / "burner_accounts.json"
_STORAGE_STATE_DIR = _CRED_DIR / "storage_state"

# Ref convention for every provisioning lane (`<site>-<lane>-1`); dict order is the reuse preference.
_LANE_SLUG = {"generated": "auto", "human_provisioned": "human"}


def boot_env_secrets() -> dict:
    """A fresh copy of the dummy boot-secret env. Callers merge it into the app's start env."""
    return dict(_BOOT_ENV_SECRETS)


def known_account(full_name: str) -> dict | None:
    """The {email, password} test login registered for `full_name`, or None."""
    # Re-read every call, so an operator can add an entry between runs.
    if not _KNOWN_ACCOUNTS_FILE.exists():
        return None
    try:
        registry = json.loads(_KNOWN_ACCOUNTS_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"known_accounts file unreadable: {e}")
        return None
    entry = registry.get(full_name)
    if not isinstance(entry, dict) or not entry.get("email") or not entry.get("password"):
        return None
    return {"email": entry["email"], "password": entry["password"]}


def _load_burners() -> dict:
    if not _BURNER_FILE.exists():
        return {}
    try:
        return json.loads(_BURNER_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"burner_accounts file unreadable: {e}")
        return {}


def is_registered_burner(burner_id: str) -> bool:
    """FAIL-CLOSED: True only for an id mapping to a burner explicitly marked `registered`."""
    # Never relax this to "assume registered": capturing against a real site is permitted only on this evidence.
    entry = _load_burners().get(burner_id or "")
    return bool(isinstance(entry, dict) and entry.get("registered") and entry.get("email") and entry.get("password"))


def burner_account(burner_id: str) -> dict | None:
    """The {email, password, provisioning_mode} of a REGISTERED burner, or None, read as "do not proceed"."""
    if not is_registered_burner(burner_id):
        return None
    entry = _load_burners()[burner_id]
    return {
        "email": entry["email"],
        "password": entry["password"],
        "provisioning_mode": entry.get("provisioning_mode", "human_provisioned"),
    }


def register_burner(
    burner_id: str,
    email: str,
    password: str,
    provisioning_mode: str = "human_provisioned",
    first_name: str = "",
    last_name: str = "",
) -> None:
    """Register or update a burner in the gitignored store that `is_registered_burner` gates on."""
    # Disposable identities only: a real personal account must never be registered here.
    if not (burner_id and email and password):
        raise ValueError("register_burner requires a non-empty burner_id, email, and password")
    _CRED_DIR.mkdir(parents=True, exist_ok=True)
    registry = _load_burners()
    entry = {"email": email, "password": password, "provisioning_mode": provisioning_mode, "registered": True}
    if first_name or last_name:
        entry["first_name"], entry["last_name"] = first_name, last_name
    registry[burner_id] = entry
    _BURNER_FILE.write_text(json.dumps(registry, indent=2))
    logger.info(f"registered burner '{burner_id}' ({provisioning_mode})")


def _storage_state_path(ref: str) -> Path:
    return _STORAGE_STATE_DIR / f"{ref}.json"


def _carries_a_session(state) -> bool:
    return bool(isinstance(state, dict) and (state.get("cookies") or state.get("origins")))


def save_storage_state(ref: str, state: dict) -> None:
    """Persist a Playwright storage_state dump so a later pre_session run starts already signed in."""
    if not ref:
        raise ValueError("save_storage_state requires a non-empty ref")
    if not _carries_a_session(state):
        raise ValueError("refusing to save an empty storage_state (no cookies or origins)")
    _STORAGE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _storage_state_path(ref)
    path.write_text(json.dumps(state, indent=2))
    logger.info(f"saved storage_state '{ref}' ({len(json.dumps(state))} bytes)")


def load_storage_state(ref: str) -> dict | None:
    """A saved Playwright storage_state for pre_session login, or None so the caller skips the site."""
    if not ref:
        return None
    path = _storage_state_path(ref)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"storage_state {ref} unreadable: {e}")
        return None


def session_ref(site_name: str, provisioning_mode: str) -> str:
    """The burner id / storage_state ref naming `site_name`'s login in the lane that provisioned it."""
    if provisioning_mode not in _LANE_SLUG:
        raise ValueError(f"unknown provisioning_mode '{provisioning_mode}' (expected {sorted(_LANE_SLUG)})")
    return f"{site_name}-{_LANE_SLUG[provisioning_mode]}-1"


def provisioned_refs(site_name: str) -> list[str]:
    """Every lane's ref for `site_name`, in lookup-preference order."""
    return [session_ref(site_name, mode) for mode in _LANE_SLUG]


def save_provisioned_session(
    ref: str,
    email: str,
    password: str,
    state: dict,
    provisioning_mode: str = "human_provisioned",
    first_name: str = "",
    last_name: str = "",
) -> None:
    """Persist both halves of a provisioned login — every lane's write path, human or autonomous."""
    # State first: its empty-dump refusal then aborts before a burner is registered with no session.
    save_storage_state(ref, state)
    register_burner(
        ref, email, password, provisioning_mode=provisioning_mode, first_name=first_name, last_name=last_name
    )


def load_provisioned_session(ref: str) -> dict | None:
    """The {storage_state, secrets, provisioning_mode} for `ref`, or None unless BOTH halves exist."""
    burner = burner_account(ref)
    if burner is None:
        return None
    state = load_storage_state(ref)
    # Same emptiness bar as the save path, so a hollowed-out file resumes nothing instead of pretending.
    if not _carries_a_session(state):
        return None
    return {
        "storage_state": state,
        "provisioning_mode": burner["provisioning_mode"],
        "secrets": {"email": burner["email"], "password": burner["password"]},
    }


def find_provisioned_session(site_name: str) -> dict | None:
    """The site's already-provisioned login and its `ref` (own lane preferred), or None."""
    for ref in provisioned_refs(site_name):
        session = load_provisioned_session(ref)
        if session:
            return {"ref": ref, **session}
    return None


def unvetted_session_refs(site_name: str) -> list[str]:
    """Refs with a saved session no registered burner vouches for — unknown provenance, never reusable."""
    return [
        ref
        for ref in provisioned_refs(site_name)
        if _storage_state_path(ref).exists() and not is_registered_burner(ref)
    ]
