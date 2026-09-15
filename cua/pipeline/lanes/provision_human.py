"""Human-in-the-loop session provisioning for login-gated sites the autonomous lane can't reach.

Some sites guard their signup/login with an interactive wall (captcha / phone / headless-fingerprint
detection) that the headless sandbox agent cannot clear, so they end up in the `needs_human` queue
([[autonomous-login-gated-capture]]). This helper is the human on-ramp: it opens a HEADED browser on
the host, a person signs up / logs in and clears the wall ONCE by hand, and the resulting
authenticated session is saved for the existing capture pipeline to reuse.

Why the saved session then works from the headless sandbox: Docker Desktop containers egress via the
host's public IP, so the sandbox replays the session from the SAME IP the human logged in from — the
only thing the human cleared was the one-time interactive challenge; the durable session is portable.

The identity is a disposable mail.tm burner — NEVER a personal account
([[burner-accounts-only-for-capture]]) — minted ONCE per batch and reused on every site, so the
operator types (and then autofills) the same credentials throughout while each site still gets its
OWN storage_state under its OWN registered burner ref. Its inbox rides along in a second browser
window so a verification link is one click away. After a successful login the helper persists the
session + burner (via app_secrets), marks the site `registered`, and wires it into
`public_sites_human.json` (auth_mode=pre_session) so `cua.pipeline.lanes.public --config
public_sites_human.json` can run its tasks already signed in.

Two ways to drive the SAME per-site open/save (`open_site` / `save_site`): the terminal loop, and
`--serve`, a localhost-only JSON control API an assistant session drives with curl while the
operator only ever types into the browser window in front of them.

Run: python -m cua.pipeline.lanes.provision_human <site-name> | --sites a,b,c | --all [--config <cfg>]
     python -m cua.pipeline.lanes.provision_human --serve [--port 8778] [--config <cfg>]
"""

import argparse
import functools
import hashlib
import json
import logging
import secrets
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

from cua.pipeline import app_secrets, mailtm, provisioning_status
from cua.pipeline.sites.registry import HUMAN_CONFIG, find_site

logger = logging.getLogger(__name__)

PROVISIONING_MODE = "human_provisioned"
CONTROL_HOST = "127.0.0.1"
DEFAULT_CONTROL_PORT = 8778

_FIRST_NAMES = [
    "Emma",
    "Liam",
    "Olivia",
    "Noah",
    "Ava",
    "Ethan",
    "Sophia",
    "Mason",
    "Isabella",
    "Lucas",
    "Mia",
    "Oliver",
    "Amelia",
    "Elijah",
    "Charlotte",
    "James",
    "Harper",
    "Benjamin",
    "Evelyn",
    "Henry",
    "Chloe",
    "Daniel",
    "Grace",
    "Owen",
    "Nora",
]
_LAST_NAMES = [
    "Smith",
    "Johnson",
    "Miller",
    "Davis",
    "Garcia",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Martin",
    "Lee",
    "Walker",
    "Hall",
    "Allen",
    "Young",
    "King",
    "Wright",
    "Scott",
    "Bennett",
    "Reed",
    "Cooper",
    "Bailey",
    "Rivera",
    "Hughes",
]


def random_name() -> tuple[str, str]:
    """A plausible disposable first/last name for signup forms — never the operator's real identity."""
    return secrets.choice(_FIRST_NAMES), secrets.choice(_LAST_NAMES)


def mint_batch_identity() -> dict:
    """The ONE disposable burner every site in this batch is signed up with (see module docstring)."""
    mailbox = mailtm.create_mailbox()
    first, last = random_name()
    return {
        "email": mailbox["address"],
        "password": mailbox["password"],
        "token": mailbox["token"],
        "first_name": first,
        "last_name": last,
    }


def print_identity(identity: dict) -> None:
    """Show the batch burner where the operator needs it — right above the prompt for this site."""
    print("  Sign up with this DISPOSABLE burner (NEVER a personal account). Same one on every site,")
    print("  so after the first signup the browser autofills it:")
    print(f"    full name : {identity['first_name']} {identity['last_name']}")
    print(f"    first name: {identity['first_name']}")
    print(f"    last name : {identity['last_name']}")
    print(f"    email     : {identity['email']}")
    print(f"    password  : {identity['password']}")


def show_latest_mail(token: str) -> None:
    """Print the most recent message in the burner inbox so the human can read a verification
    link/code without leaving the terminal (they may also open mail.tm in the browser instead)."""
    messages = mailtm.list_messages(token)
    if not messages:
        print("  (no mail yet — wait ~10-20s after requesting it, then type 'mail' again)")
        return
    msg = mailtm.get_message(token, messages[0]["id"])
    sender = (msg.get("from") or {}).get("address", "?")
    body = (msg.get("text") or msg.get("intro") or "").strip()
    print(f"\n  ── latest email ──  from {sender}")
    print(f"  Subject: {msg.get('subject', '')}\n")
    print("  " + "\n  ".join(body.splitlines()[:40]))
    print("  ──────────────────\n")


def upsert_human_config(site: dict, ref: str, app_url: str) -> None:
    """Add/replace `site` in public_sites_human.json as a pre_session entry reusing the saved burner session."""
    # Its curated tasks ride along so the public lane can run them signed in with no further wiring.
    config = json.loads(HUMAN_CONFIG.read_text()) if HUMAN_CONFIG.exists() else {"sites": []}
    entry = {
        "name": site["name"],
        "start_url": app_url,
        "domain": site.get("domain", site["name"]),
        "autonomy": "autonomous",
        "auth_mode": "pre_session",
        "burner_account_id": ref,
        "storage_state_ref": ref,
        "tasks": site.get("tasks", []),
    }
    others = [s for s in config.get("sites", []) if s.get("name") != site["name"]]
    config["sites"] = others + [entry]
    HUMAN_CONFIG.write_text(json.dumps(config, indent=2))
    logger.info(f"wrote {site['name']} ({len(entry['tasks'])} tasks) to {HUMAN_CONFIG.name}")


def launch_realistic_chrome(playwright):
    """Launch a HEADED browser that does not trip anti-bot fingerprinting (Cloudflare Turnstile,
    reCAPTCHA). Playwright's default Chromium is the automation-branded 'Chrome for Testing' launched
    with `--enable-automation` + the AutomationControlled blink feature, which set navigator.webdriver
    and the tells Turnstile blocks even when a human solves the challenge. Prefer the real system
    Google Chrome and strip those flags. One browser is reused across a batch; each site gets a fresh
    isolated context (new_stealth_context). The person still solves the challenge — this only stops the
    browser itself from being fingerprinted as a bot."""
    launch = {
        "headless": False,
        "args": ["--disable-blink-features=AutomationControlled"],
        "ignore_default_args": ["--enable-automation"],
    }
    try:
        browser = playwright.chromium.launch(channel="chrome", **launch)
        logger.info("launched real system Google Chrome (channel=chrome)")
    except Exception as e:
        logger.warning(f"system Google Chrome unavailable ({e}); using bundled chromium with anti-automation flags")
        browser = playwright.chromium.launch(**launch)
    return browser


def new_stealth_context(browser):
    context = browser.new_context(no_viewport=True)
    context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return context


class Webmail:
    """The batch burner's mail.tm inbox, kept open beside the site window for one-click verification."""

    def __init__(self, browser, identity: dict):
        self.browser = browser
        self.identity = identity
        self.page = None

    def _sign_in_script(self) -> str:
        account = {"address": self.identity["email"], "password": self.identity["password"]}
        # mail.tm has no login form: the web app signs in from the account it finds in localStorage.
        return (
            f"localStorage.setItem('accounts', {json.dumps(json.dumps([account]))});"
            f"localStorage.setItem('active', {json.dumps(json.dumps(account))});"
        )

    def _shown_address(self) -> str:
        """The address the inbox is actually displaying, or '' when it cannot be read."""
        try:
            return (self.page.locator("input[type='email']").first.input_value(timeout=10000) or "").strip()
        except Exception:
            return ""

    def open(self):
        """Open (or reopen) the inbox window signed in as the burner; a failure never stops the batch."""
        try:
            # Own context: mail.tm's own cookies must never land in a site's saved storage_state.
            context = self.browser.new_context(no_viewport=True)
            context.add_init_script(self._sign_in_script())
            self.page = context.new_page()
            self.page.goto(mailtm.WEBMAIL_URL, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            logger.warning(f"could not open the inbox window ({e}); use the 'mail' command instead")
            self.page = None
            return None
        shown = self._shown_address()
        if shown == self.identity["email"]:
            logger.info(f"inbox window open at {mailtm.WEBMAIL_URL}, signed in as {shown}")
        else:
            logger.warning(
                f"inbox window is not showing the burner ({shown or 'address unreadable'}) — "
                f"sign in by hand with the printed credentials, or read mail with 'mail'"
            )
        return self.page

    def ensure(self):
        """The inbox page, opening the window if it is missing or the operator closed it."""
        if self.page is None or self.page.is_closed():
            self.open()
        return self.page

    def focus(self) -> None:
        """Bring the inbox to the front, reopening it if the operator closed the window."""
        page = self.ensure()
        if page is None:
            return
        try:
            page.bring_to_front()
        except Exception as e:
            logger.warning(f"inbox window unavailable ({e}); showing the terminal copy instead")


def storage_fingerprint(state: dict) -> dict:
    """Every cookie / localStorage / IndexedDB entry in a dump, valued so a rewrite counts as a change."""
    fingerprint = {}
    for cookie in state.get("cookies", []):
        key = f"cookie:{cookie.get('domain', '')}:{cookie.get('name', '')}"
        fingerprint[key] = hashlib.sha1(str(cookie.get("value", "")).encode()).hexdigest()
    for origin in state.get("origins", []):
        host = origin.get("origin", "")
        for item in origin.get("localStorage") or []:
            key = f"localStorage:{host}:{item.get('name', '')}"
            fingerprint[key] = hashlib.sha1(str(item.get("value", "")).encode()).hexdigest()
        for database in origin.get("indexedDB") or []:
            key = f"indexedDB:{host}:{database.get('name', '')}"
            fingerprint[key] = hashlib.sha1(json.dumps(database, sort_keys=True).encode()).hexdigest()
    return fingerprint


def changed_storage_keys(before: dict, after: dict) -> list:
    """The storage keys added or rewritten since the tab opened — a real login always shows up here."""
    return sorted(key for key, value in after.items() if before.get(key) != value)


def open_site(browser, site: dict, webmail: Webmail):
    """Open `site` in its OWN fresh isolated context beside the inbox; (context, page, opened-details)."""
    # Inbox first so the site window, created last, is the one in front of the operator.
    webmail.ensure()
    context = new_stealth_context(browser)
    page = context.new_page()
    navigation_error = ""
    try:
        page.goto(site["start_url"], wait_until="domcontentloaded", timeout=60000)
    except Exception as e:
        navigation_error = str(e).splitlines()[0]
        logger.warning(f"[{site['name']}] navigation issue ({navigation_error}); navigate manually in the window")
    # Pre-login baseline: on a real site a plain visit already sets cookies, so "dump is non-empty"
    # alone never proves anyone signed in — only a change against this does.
    baseline = storage_fingerprint(context.storage_state(indexed_db=True))
    return context, page, {"navigation_error": navigation_error, "baseline": baseline}


def save_site(site: dict, context, identity: dict, app_url: str, baseline: dict | None = None) -> dict:
    """Persist the site's OWN ref + session and wire it into the human config; ValueError if nobody signed in."""
    ref = app_secrets.session_ref(site["name"], PROVISIONING_MODE)
    state = context.storage_state(indexed_db=True)
    changed = changed_storage_keys(baseline or {}, storage_fingerprint(state))
    if baseline is not None and not changed:
        raise ValueError(
            "browser storage is unchanged since the tab opened (not one cookie, localStorage "
            "or IndexedDB entry differs), so no login happened"
        )
    # Raises if the dump carries no session at all, before any burner is registered.
    app_secrets.save_provisioned_session(
        ref,
        identity["email"],
        identity["password"],
        state,
        provisioning_mode=PROVISIONING_MODE,
        first_name=identity["first_name"],
        last_name=identity["last_name"],
    )
    provisioning_status.mark(site["name"], provisioning_status.REGISTERED, provisioning=site.get("provisioning", ""))
    upsert_human_config(site, ref, app_url)
    return {
        "ref": ref,
        "cookies": len(state.get("cookies", [])),
        "origins": len(state.get("origins", [])),
        "bytes": len(json.dumps(state)),
        "changed_keys": len(changed),
        "changed_sample": changed[:8],
    }


def provision_site(browser, site: dict, identity: dict, webmail: Webmail, position: tuple[int, int]) -> str:
    """Provision one site from the terminal with the batch burner; 'provisioned' | 'skipped' (SystemExit ends the batch)."""
    index, total = position
    print("\n" + "=" * 72)
    print(f"  [{index}/{total}] PROVISIONING: {site['name']}   →  {site['start_url']}")
    print_identity(identity)
    print("  Verification mail: the burner inbox is open in the other browser window; 'mail' brings")
    print(f"  it forward and also prints the latest message here ({mailtm.WEBMAIL_URL}).")
    print("=" * 72)

    context, page, opened = open_site(browser, site, webmail)
    print(
        "\n  Sign up / log in with the burner above and solve any captcha. When you are fully signed\n"
        "  in (the app's own screen is showing), return here.\n"
    )
    saved = None
    try:
        while True:
            cmd = input("  [Enter]=done | mail=open inbox | skip=skip this site | q=quit batch : ").strip().lower()
            if cmd == "":
                saved = save_site(site, context, identity, page.url or site["start_url"], opened["baseline"])
                break
            if cmd == "skip":
                logger.info(f"[{site['name']}] skipped by operator")
                return "skipped"
            if cmd == "q":
                raise SystemExit("batch stopped by operator")
            if cmd == "mail":
                webmail.focus()
                show_latest_mail(identity["token"])
    finally:
        context.close()

    print(
        f"\n  ✓ provisioned {site['name']}: session saved ({saved['ref']}), burner registered, "
        f"added to {HUMAN_CONFIG.name}"
    )
    return "provisioned"


class ControlError(Exception):
    """A control-API failure carrying the HTTP status and JSON body the assistant should see."""

    def __init__(self, status: int, message: str, **detail):
        super().__init__(message)
        self.status = status
        self.detail = {"ok": False, "error": message, **detail}


class ProvisionSession:
    """One headed browser + ONE shared burner for a whole run, with at most one site context open."""

    def __init__(self, browser, identity: dict, config_path: Path | None):
        self.browser = browser
        self.identity = identity
        self.config_path = config_path
        self.webmail = Webmail(browser, identity)
        self.site = None
        self.context = None
        self.page = None
        self.baseline = None
        self.provisioned = []
        self.skipped = []
        self.stopped = False

    def _public_identity(self) -> dict:
        return {key: self.identity[key] for key in ("email", "password", "first_name", "last_name")}

    def _current_url(self) -> str | None:
        try:
            return self.page.url if self.page is not None and not self.page.is_closed() else None
        except Exception:
            return None

    def _close_context(self) -> None:
        if self.context is not None:
            try:
                self.context.close()
            except Exception as e:
                logger.warning(f"closing the site context failed ({e})")
        self.site = self.context = self.page = self.baseline = None

    def _require_open(self, key: str | None) -> dict:
        """The open site, cross-checked against the name the caller believes is open."""
        if self.site is None:
            raise ControlError(409, "no site is open — POST /open first")
        if key and key not in (self.site["name"], self.site.get("start_url", "")):
            raise ControlError(409, f"'{key}' is not the open site ({self.site['name']})")
        return self.site

    def status(self, body: dict | None = None) -> dict:
        """Everything the assistant needs to narrate the loop: shared identity, open site, progress."""
        return {
            "ok": True,
            "identity": self._public_identity(),
            "current_site": self.site["name"] if self.site else None,
            "current_url": self._current_url(),
            "provisioned": list(self.provisioned),
            "skipped": list(self.skipped),
            "browser_alive": self.browser.is_connected(),
        }

    def open(self, key: str | None) -> dict:
        """Open a site in a fresh context; an unknown site or a failed navigation never ends the run."""
        if not key:
            raise ControlError(400, 'POST /open needs {"site": "<name-or-url>"}')
        site = find_site(key, self.config_path)
        if site is None:
            raise ControlError(404, f"no site matches '{key}' in the source config(s)")
        if not self.browser.is_connected():
            raise ControlError(409, "the browser is gone (window closed?) — restart --serve")
        self._close_context()
        try:
            self.context, self.page, opened = open_site(self.browser, site, self.webmail)
        except Exception as e:
            self._close_context()
            raise ControlError(502, f"could not open {site['name']}: {e}")
        self.site = site
        self.baseline = opened["baseline"]
        return {
            "ok": True,
            "site": site["name"],
            "start_url": site["start_url"],
            "identity": self._public_identity(),
            "navigation_error": opened["navigation_error"] or None,
            "provisioning": site.get("provisioning", ""),
            "tasks": len(site.get("tasks", [])),
        }

    def save(self, key: str | None) -> dict:
        """Persist the open site under its own ref; a logged-out dump fails loudly and keeps the tab open."""
        site = self._require_open(key)
        try:
            saved = save_site(
                site, self.context, self.identity, self._current_url() or site["start_url"], self.baseline
            )
        except ValueError as e:
            raise ControlError(422, f"{e} — {site['name']} is not signed in; log in, then POST /save again")
        self._close_context()
        self.provisioned.append(site["name"])
        return {"ok": True, "site": site["name"], **saved}

    def skip(self, key: str | None) -> dict:
        """Drop the open site without saving anything."""
        site = self._require_open(key)
        self._close_context()
        self.skipped.append(site["name"])
        logger.info(f"[{site['name']}] skipped")
        return {"ok": True, "site": site["name"], "skipped": True}

    def quit(self, body: dict | None = None) -> dict:
        """Close the browser and end the serve loop."""
        self._close_context()
        self.stopped = True
        self.close_browser()
        return {"ok": True, "stopped": True, "provisioned": list(self.provisioned), "skipped": list(self.skipped)}

    def close_browser(self) -> None:
        try:
            if self.browser.is_connected():
                self.browser.close()
        except Exception as e:
            logger.warning(f"closing the browser failed ({e})")


# Adding an endpoint is a dict entry, never a new branch in the handler.
_GET_ROUTES = {"/status": lambda session, body: session.status(body)}
_POST_ROUTES = {
    "/open": lambda session, body: session.open(body.get("site")),
    "/save": lambda session, body: session.save(body.get("site")),
    "/skip": lambda session, body: session.skip(body.get("site")),
    "/quit": lambda session, body: session.quit(body),
}


class ControlHandler(BaseHTTPRequestHandler):
    """Localhost-only JSON control API routing each request onto the shared ProvisionSession."""

    def __init__(self, session: ProvisionSession, *args, **kwargs):
        self.session = session
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        logger.info(f"control {fmt % args}")

    def _respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            parsed = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            raise ControlError(400, f"body is not JSON ({e})")
        return parsed if isinstance(parsed, dict) else {}

    def _dispatch(self, routes: dict) -> None:
        path = self.path.split("?")[0].rstrip("/") or "/status"
        route = routes.get(path)
        if route is None:
            self._respond(
                404,
                {
                    "ok": False,
                    "error": f"unknown endpoint {path}",
                    "endpoints": sorted(set(_GET_ROUTES) | set(_POST_ROUTES)),
                },
            )
            return
        # Nothing may escape: one bad request must never end a 37-site session.
        try:
            self._respond(200, route(self.session, self._body()))
        except ControlError as e:
            self._respond(e.status, e.detail)
        except Exception as e:
            logger.exception(f"control request {path} failed")
            self._respond(500, {"ok": False, "error": str(e)})

    def do_GET(self):
        self._dispatch(_GET_ROUTES)

    def do_POST(self):
        self._dispatch(_POST_ROUTES)


def print_control_help(base: str) -> None:
    """The exact curl calls that drive the loop, printed where the assistant can copy them."""
    print(f"  curl -s {base}/status")
    print(f"""  curl -s -X POST {base}/open -H 'Content-Type: application/json' -d '{{"site":"baserow"}}'""")
    print(f"""  curl -s -X POST {base}/save -H 'Content-Type: application/json' -d '{{"site":"baserow"}}'""")
    print(f"""  curl -s -X POST {base}/skip -H 'Content-Type: application/json' -d '{{"site":"baserow"}}'""")
    print(f"  curl -s -X POST {base}/quit")


def serve(port: int, config_path: Path | None) -> None:
    """Serve the control API until /quit, single-threaded so every Playwright call stays on this thread."""
    # Bind-test first: a taken port must fail before a browser window is on screen.
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((CONTROL_HOST, port))
    except OSError as e:
        raise SystemExit(f"cannot serve on {CONTROL_HOST}:{port} ({e}) — pass a free --port")
    finally:
        probe.close()

    identity = mint_batch_identity()
    with sync_playwright() as playwright:
        browser = launch_realistic_chrome(playwright)
        session = ProvisionSession(browser, identity, config_path)
        session.webmail.ensure()
        server = HTTPServer((CONTROL_HOST, port), functools.partial(ControlHandler, session))
        base = f"http://{CONTROL_HOST}:{port}"
        print("\n" + "=" * 72)
        print(f"  control API ready on {base} (localhost only) — the operator only types in the browser")
        print_identity(identity)
        print("  drive it with:")
        print_control_help(base)
        print("=" * 72, flush=True)
        try:
            while not session.stopped:
                server.handle_request()
        except KeyboardInterrupt:
            print("\n  interrupted")
        finally:
            server.server_close()
            session.close_browser()

    print(f"\n  done: {len(session.provisioned)} provisioned, {len(session.skipped)} skipped")
    for outcome, names in (("provisioned", session.provisioned), ("skipped", session.skipped)):
        if names:
            print(f"    {outcome}: {', '.join(names)}")
    print(f"  session burner: {identity['email']} / {identity['password']}")


def batch_targets(config_path: Path | None) -> list[dict]:
    """Every needs_human site not yet provisioned (registered), resolved to site dicts, name-sorted."""
    done = set(provisioning_status.sites_with_status(provisioning_status.REGISTERED))
    todo = sorted(n for n in provisioning_status.sites_with_status(provisioning_status.NEEDS_HUMAN) if n not in done)
    return [s for s in (find_site(n, config_path) for n in todo) if s]


def resolve_sites(args, config_path: Path | None) -> list[dict]:
    """The sites to provision — an explicit SITE / --sites list keeps the operator's order exactly."""
    if args.all:
        return batch_targets(config_path)
    keys = [k.strip() for k in args.sites.split(",")] if args.sites else [args.site]
    resolved = []
    for key in keys:
        site = find_site(key, config_path)
        if site is None:
            logger.warning(f"no site matches '{key}' in the source config(s) — skipping it")
            continue
        resolved.append(site)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-in-the-loop login provisioning (headed browser)")
    parser.add_argument("site", nargs="?", default=None, help="Site name or signup URL (omit with --all/--sites)")
    parser.add_argument("--sites", default=None, help="Comma-separated site names/URLs to provision in a loop")
    parser.add_argument("--all", action="store_true", help="Provision ALL needs_human sites in a loop")
    parser.add_argument("--config", default=None, help="Source config path (default: search both)")
    parser.add_argument(
        "--serve", action="store_true", help="Serve the localhost control API instead of prompting in the terminal"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_CONTROL_PORT,
        help=f"Control API port on {CONTROL_HOST} (default {DEFAULT_CONTROL_PORT})",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config_path = Path(args.config) if args.config else None

    if args.serve:
        serve(args.port, config_path)
        return
    if not (args.all or args.sites or args.site):
        parser.error("give a SITE / URL, or --sites a,b,c, or --all")
    sites = resolve_sites(args, config_path)
    if not sites:
        raise SystemExit("no matching sites to provision (check names against the source configs)")

    # Minted before the browser so a mail.tm outage fails now, not with windows already on screen.
    identity = mint_batch_identity()
    results = {"provisioned": [], "skipped": [], "failed": []}
    with sync_playwright() as playwright:
        browser = launch_realistic_chrome(playwright)
        webmail = Webmail(browser, identity)
        webmail.open()
        try:
            for i, site in enumerate(sites, 1):
                try:
                    outcome = provision_site(browser, site, identity, webmail, (i, len(sites)))
                except SystemExit as e:
                    print(f"\n{e}")
                    break
                except Exception as e:
                    logger.error(f"[{site['name']}] provisioning failed ({e}); moving on")
                    outcome = "failed"
                results[outcome].append(site["name"])
        finally:
            browser.close()

    print(
        f"\n  done: {len(results['provisioned'])} provisioned, {len(results['skipped'])} skipped, "
        f"{len(results['failed'])} failed"
    )
    for outcome, names in results.items():
        if names:
            print(f"    {outcome}: {', '.join(names)}")
    print(f"  batch burner: {identity['email']} / {identity['password']}")
    if results["provisioned"]:
        print("  Next: tell the assistant to run the batch gather over public_sites_human.json\n")


if __name__ == "__main__":
    main()
