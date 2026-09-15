"""Real disposable mailbox via mail.tm — the readable inbox for REAL-site auth flows.

GreenMail ([[mail-service]]) only catches mail from sandboxed apps whose SMTP we redirect. A real
public site sends its verification / reset mail to wherever the address resolves, so to capture
register / password-reset on real sites the agent needs a genuine throwaway mailbox with a web UI
it can open in the browser. mail.tm provides exactly that: an API to mint an address + a real web
inbox at https://mail.tm the agent signs into and reads like a human.

The mailbox is a disposable burner (random address + generated password), consistent with
[[burner-accounts-only-for-capture]] — never a real personal inbox. The API here is used to MINT
the box and (out-of-band, for the harness) confirm a message arrived; the agent still reads the
message visually in the browser, which is the skill being captured.
"""

import json
import logging
import secrets
import string
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

API = "https://api.mail.tm"
WEBMAIL_URL = "https://mail.tm/"
_UA = "Mozilla/5.0 (capture-pipeline mail.tm client)"


def _req(method: str, path: str, body: dict | None = None, token: str | None = None, tries: int = 5):
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(tries):
        req = urllib.request.Request(f"{API}{path}", data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _UA)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", "replace")
            return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            # mail.tm throttles bursts (account/token creation especially). Back off and retry on
            # 429/503; re-raise everything else (e.g. 422 address-taken, which create_mailbox handles).
            if e.code in (429, 503) and attempt < tries - 1:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                delay = float(retry_after) if (retry_after and str(retry_after).isdigit()) else 5 * (2**attempt)
                logger.warning(
                    f"mail.tm {e.code} on {method} {path} — backing off {delay:.0f}s (attempt {attempt + 1}/{tries})"
                )
                time.sleep(delay)
                continue
            raise
        except urllib.error.URLError as e:
            # Transient host network/DNS blip (getaddrinfo "nodename nor servname", connection
            # refused/reset/timeout) — these otherwise fail a whole site instantly during a brief
            # outage and burn the queue. HTTPError (a URLError subclass) is handled above; this is
            # the connection-level fallback. Back off and retry; give up only after `tries`.
            if attempt < tries - 1:
                delay = 3 * (2**attempt)
                logger.warning(
                    f"mail.tm network error on {method} {path} ({e.reason}) — retry in "
                    f"{delay:.0f}s (attempt {attempt + 1}/{tries})"
                )
                time.sleep(delay)
                continue
            raise


def _members(resp) -> list:
    """Collection responses come back either as a plain JSON array (Accept: application/json) or in
    hydra form ({'hydra:member': [...]}); normalise both to a list."""
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        return resp.get("hydra:member", [])
    return []


def _random_password(n: int = 14) -> str:
    alphabet = string.ascii_letters + string.digits
    return "Qa!" + "".join(secrets.choice(alphabet) for _ in range(n)) + "7"


def available_domain() -> str:
    domains = _members(_req("GET", "/domains?page=1"))
    active = [d["domain"] for d in domains if d.get("isActive") and not d.get("isPrivate")]
    if not active:
        raise RuntimeError("mail.tm has no active public domain right now")
    return active[0]


def create_mailbox(local_part: str | None = None) -> dict:
    """Mint a disposable mailbox. Returns {address, password, token} — `password` logs into both the
    mail.tm web inbox AND (commonly reused as) the site account, `token` is for out-of-band reads."""
    domain = available_domain()
    local = local_part or ("qa" + secrets.token_hex(6))
    address = f"{local}@{domain}"
    password = _random_password()
    # 422 here means the address was taken — retry with a fresh random local part.
    for attempt in range(3):
        try:
            _req("POST", "/accounts", {"address": address, "password": password})
            break
        except urllib.error.HTTPError as e:
            if e.code == 422 and attempt < 2:
                local = "qa" + secrets.token_hex(6)
                address = f"{local}@{domain}"
                continue
            raise
    token = _req("POST", "/token", {"address": address, "password": password})["token"]
    logger.info(f"minted disposable mailbox {address}")
    return {"address": address, "password": password, "token": token}


def list_messages(token: str) -> list[dict]:
    return _members(_req("GET", "/messages?page=1", token=token))


def get_message(token: str, message_id: str) -> dict:
    return _req("GET", f"/messages/{message_id}", token=token)


def wait_for_message(token: str, timeout: int = 90, match: str | None = None) -> dict | None:
    """Poll until a message arrives (out-of-band harness confirmation that the site actually sent
    mail to the box). `match` optionally requires a substring in the subject/intro. None on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for m in list_messages(token):
            blob = f"{m.get('subject', '')} {m.get('intro', '')}".lower()
            if match is None or match.lower() in blob:
                return m
        time.sleep(3)
    return None
