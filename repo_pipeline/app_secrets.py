"""Boot secrets and known test-account credentials for agent capture.

Two concerns, both generic (config/data driven, never a per-repo `if name ==` check):

  * `boot_env_secrets()` — dummy values for the auth secrets a typical auth app reads at startup
    (JWT_SECRET, *_SECRET, NEXTAUTH_URL ...). Without them most Next.js / Express auth apps throw
    during boot. In a disposable, single-run sandbox a fixed dummy value is safe — these never
    protect anything real. Merged into the app's environment alongside the mail config.

  * `known_account(full_name)` — an OPTIONAL, gitignored registry of pre-existing test logins for
    specific apps (demo apps routinely ship/advertise one, e.g. `admin@demo / admin`). When an app
    has a known account, capture signs in with it instead of registering a fresh generated address
    — which is the only practical path for apps whose mail leaves the sandbox (cloud email APIs,
    hardcoded providers) so a generated mailbox would never receive the verification message.
    Returned credentials are handed to the driver as masked `<secret:...>` values, never recorded.

The known-accounts file is the first slice of the Phase-5 burner store; it lives under
`credentials/` (gitignored) so real logins never enter version control.
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Fixed dummy secret for the disposable sandbox. Long enough for libraries that assert a minimum
# key length; identical across runs so a captured pre_session cookie stays valid on resume.
_DUMMY_SECRET = "qa_test_secret_0123456789abcdef0123456789abcdef"

# Common env var names auth apps read for signing/session secrets and the app's own base URL.
# A superset is harmless: an app ignores the names it does not use.
_BOOT_ENV_SECRETS = {
    "JWT_SECRET": _DUMMY_SECRET, "JWT_REFRESH_SECRET": _DUMMY_SECRET,
    "ACCESS_TOKEN_SECRET": _DUMMY_SECRET, "REFRESH_TOKEN_SECRET": _DUMMY_SECRET,
    "AUTH_SECRET": _DUMMY_SECRET, "NEXTAUTH_SECRET": _DUMMY_SECRET,
    "SESSION_SECRET": _DUMMY_SECRET, "COOKIE_SECRET": _DUMMY_SECRET,
    "TOKEN_SECRET": _DUMMY_SECRET, "SECRET": _DUMMY_SECRET, "APP_SECRET": _DUMMY_SECRET,
    "JWT_EXPIRES_IN": "1d", "AUTH_TRUST_HOST": "true",
    # Next.js / many SSR apps build absolute links (incl. emailed verification URLs) from this.
    # The app binds the dev-server default port inside the container, so localhost == the app.
    "NEXTAUTH_URL": "http://127.0.0.1:3000",
    "AUTH_URL": "http://127.0.0.1:3000",
}

_CRED_DIR = Path(os.environ.get("PIPELINE_CREDENTIALS_DIR", "credentials"))
_KNOWN_ACCOUNTS_FILE = _CRED_DIR / "known_accounts.json"
# Guarded/public-lane burner store (see [[burner-accounts-only-for-capture]]): real-site logins the
# capture is permitted to use, plus the saved browser sessions that let a run start already signed in.
_BURNER_FILE = _CRED_DIR / "burner_accounts.json"
_STORAGE_STATE_DIR = _CRED_DIR / "storage_state"

# Ref convention for every provisioning lane (`<site>-<lane>-1`); dict order is the reuse preference.
_LANE_SLUG = {"generated": "auto", "human_provisioned": "human"}


def boot_env_secrets() -> dict:
    """A fresh copy of the dummy boot-secret env. Callers merge it into the app's start env."""
    return dict(_BOOT_ENV_SECRETS)


def known_account(full_name: str) -> dict | None:
    """The {email, password} test login registered for `full_name`, or None. Reads the gitignored
    known-accounts file each call (cheap, and lets the operator add an entry between runs)."""
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
    """FAIL-CLOSED guard for the guarded/public lane: True only when the id maps to a burner that is
    explicitly marked `registered`. Anything missing, malformed, or not-yet-registered → False, so a
    guarded capture refuses to run against a real site without a vetted disposable identity. NEVER
    relax this to 'assume registered' — the whole safety story rests on it."""
    entry = _load_burners().get(burner_id or "")
    return bool(isinstance(entry, dict) and entry.get("registered")
                and entry.get("email") and entry.get("password"))


def burner_account(burner_id: str) -> dict | None:
    """The {email, password, provisioning_mode} for a REGISTERED burner, or None. Returns None for
    any unregistered/unknown id (callers must treat None as 'do not proceed')."""
    if not is_registered_burner(burner_id):
        return None
    entry = _load_burners()[burner_id]
    return {"email": entry["email"], "password": entry["password"],
            "provisioning_mode": entry.get("provisioning_mode", "human_provisioned")}


def register_burner(burner_id: str, email: str, password: str,
                    provisioning_mode: str = "human_provisioned",
                    first_name: str = "", last_name: str = "") -> None:
    """Register (or update) a burner in the gitignored burner store so the guarded/public lane's
    fail-closed `is_registered_burner` gate will permit it. Called by the human-provisioning helper
    after a person clears a signup wall; `provisioning_mode` records how the identity was created
    ('human_provisioned' | 'generated'). The generated disposable `first_name`/`last_name` are stored
    when present so a signup form's name fields stay consistent on reuse. NEVER register a real
    personal account here — the burner store is for disposable identities only
    ([[burner-accounts-only-for-capture]])."""
    if not (burner_id and email and password):
        raise ValueError("register_burner requires a non-empty burner_id, email, and password")
    _CRED_DIR.mkdir(parents=True, exist_ok=True)
    registry = _load_burners()
    entry = {"email": email, "password": password,
             "provisioning_mode": provisioning_mode, "registered": True}
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
    """Persist a Playwright storage_state dump (cookies + localStorage + IndexedDB) to
    `credentials/storage_state/<ref>.json` so a later pre_session run starts already signed in.
    Written by the human-provisioning helper right after a manual login; read back by
    `load_storage_state`. Refuses an empty dump so a mis-run (human never actually signed in) fails
    loudly instead of persisting a logged-out session."""
    if not ref:
        raise ValueError("save_storage_state requires a non-empty ref")
    if not _carries_a_session(state):
        raise ValueError("refusing to save an empty storage_state (no cookies or origins)")
    _STORAGE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _storage_state_path(ref)
    path.write_text(json.dumps(state, indent=2))
    logger.info(f"saved storage_state '{ref}' ({len(json.dumps(state))} bytes)")


def load_storage_state(ref: str) -> dict | None:
    """A saved Playwright storage_state (cookies + localStorage) for pre_session login on a real
    site, read from `credentials/storage_state/<ref>.json`. None if the ref is missing/unreadable —
    the caller fails closed (skips the site) rather than capture a logged-out session."""
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


def save_provisioned_session(ref: str, email: str, password: str, state: dict,
                             provisioning_mode: str = "human_provisioned",
                             first_name: str = "", last_name: str = "") -> None:
    """Persist both halves of a provisioned login — every lane's write path, human or autonomous."""
    # State first: its empty-dump refusal then aborts before a burner is registered with no session.
    save_storage_state(ref, state)
    register_burner(ref, email, password, provisioning_mode=provisioning_mode,
                    first_name=first_name, last_name=last_name)


def load_provisioned_session(ref: str) -> dict | None:
    """The {storage_state, secrets, provisioning_mode} for `ref`, or None unless BOTH halves exist."""
    burner = burner_account(ref)
    if burner is None:
        return None
    state = load_storage_state(ref)
    # Same emptiness bar as the save path, so a hollowed-out file resumes nothing instead of pretending.
    if not _carries_a_session(state):
        return None
    return {"storage_state": state, "provisioning_mode": burner["provisioning_mode"],
            "secrets": {"email": burner["email"], "password": burner["password"]}}


def find_provisioned_session(site_name: str) -> dict | None:
    """The site's already-provisioned login and its `ref` (own lane preferred), or None."""
    for ref in provisioned_refs(site_name):
        session = load_provisioned_session(ref)
        if session:
            return {"ref": ref, **session}
    return None


def unvetted_session_refs(site_name: str) -> list[str]:
    """Refs with a saved session no registered burner vouches for — unknown provenance, never reusable."""
    return [ref for ref in provisioned_refs(site_name)
            if _storage_state_path(ref).exists() and not is_registered_burner(ref)]
