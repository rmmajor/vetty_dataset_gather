"""Human-in-the-loop session provisioning for login-gated sites the autonomous lane can't reach.

Some sites guard their signup/login with an interactive wall (captcha / phone / headless-fingerprint
detection) that the headless sandbox agent cannot clear, so they end up in the `needs_human` queue
([[autonomous-login-gated-capture]]). This helper is the human on-ramp: it opens a HEADED browser on
the host, a person signs up / logs in and clears the wall ONCE by hand, and the resulting
authenticated session is saved for the existing capture pipeline to reuse.

Why the saved session then works from the headless sandbox: Docker Desktop containers egress via the
host's public IP, so the sandbox replays the session from the SAME IP the human logged in from — the
only thing the human cleared was the one-time interactive challenge; the durable session is portable.

The identity is a disposable mail.tm burner minted here — NEVER a personal account
([[burner-accounts-only-for-capture]]). After a successful login the helper persists the session +
burner (via app_secrets), marks the site `registered`, and wires it into `public_sites_human.json`
(auth_mode=pre_session) so `public_capture.py --config public_sites_human.json` can run its tasks
already signed in.

Run: python -m repo_pipeline.provision_human <site-name> [--config <source-config>]
"""
import argparse
import json
import logging
from pathlib import Path

from playwright.sync_api import sync_playwright

from repo_pipeline import app_secrets
from repo_pipeline import mailtm
from repo_pipeline import provisioning_status

logger = logging.getLogger(__name__)

SOURCE_CONFIGS = [
    Path(__file__).parent / "public_sites_actions_indie.json",
    Path(__file__).parent / "public_sites_actions_logingated.json",
]
HUMAN_CONFIG = Path(__file__).parent / "public_sites_human.json"


def find_site(name: str, config_path: Path | None = None) -> dict | None:
    """The site dict for `name`, searched in `config_path` (if given) else both source configs."""
    for cfg in ([config_path] if config_path else SOURCE_CONFIGS):
        if not cfg or not cfg.exists():
            continue
        for site in json.loads(cfg.read_text()).get("sites", []):
            if site.get("name") == name:
                return site
    return None


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
    """Add/replace `site` in public_sites_human.json as a pre_session entry that reuses the saved
    burner session, carrying the site's curated tasks so public_capture.py can run them signed in."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-in-the-loop login provisioning (headed browser)")
    parser.add_argument("site", help="Site name from a source config (indie / logingated)")
    parser.add_argument("--config", default=None, help="Source config path (default: search both)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    site = find_site(args.site, Path(args.config) if args.config else None)
    if not site:
        raise SystemExit(f"site '{args.site}' not found in source config(s)")

    ref = f"{site['name']}-human-1"
    mailbox = mailtm.create_mailbox()
    print("\n" + "=" * 72)
    print(f"  PROVISIONING: {site['name']}   →  {site['start_url']}")
    print("  Sign up with this DISPOSABLE burner (NEVER a personal account):")
    print(f"    email    : {mailbox['address']}")
    print(f"    password : {mailbox['password']}")
    print(f"  Read verification email here or at {mailtm.WEBMAIL_URL} (same email + password).")
    print("=" * 72)

    state = None
    app_url = site["start_url"]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(site["start_url"], wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            logger.warning(f"initial navigation issue ({e}); navigate manually in the window")
        print("\n  A browser window is open. Sign up / log in with the burner above and solve any\n"
              "  captcha. When you are fully signed in (the app's own screen is showing), return here.\n")
        try:
            while True:
                cmd = input("  [Enter]=done  |  mail=show latest email  |  q=abort : ").strip().lower()
                if cmd == "":
                    break
                if cmd == "q":
                    raise SystemExit("aborted by operator — nothing saved")
                if cmd == "mail":
                    show_latest_mail(mailbox["token"])
            app_url = page.url or site["start_url"]
            state = context.storage_state(indexed_db=True)
        finally:
            browser.close()

    app_secrets.save_storage_state(ref, state)  # raises if the dump is empty (never actually logged in)
    app_secrets.register_burner(ref, mailbox["address"], mailbox["password"])
    provisioning_status.mark(site["name"], provisioning_status.REGISTERED,
                             provisioning=site.get("provisioning", ""))
    upsert_human_config(site, ref, app_url)
    print(f"\n  ✓ provisioned {site['name']}: session saved ({ref}), burner registered, "
          f"added to {HUMAN_CONFIG.name}")
    print(f"  Next:  make capture-human SITES={site['name']}\n")


if __name__ == "__main__":
    main()
