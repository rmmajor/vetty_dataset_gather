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
import secrets
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

_FIRST_NAMES = ["Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason", "Isabella",
                "Lucas", "Mia", "Oliver", "Amelia", "Elijah", "Charlotte", "James", "Harper",
                "Benjamin", "Evelyn", "Henry", "Chloe", "Daniel", "Grace", "Owen", "Nora"]
_LAST_NAMES = ["Smith", "Johnson", "Miller", "Davis", "Garcia", "Wilson", "Anderson", "Thomas",
               "Taylor", "Moore", "Martin", "Lee", "Walker", "Hall", "Allen", "Young", "King",
               "Wright", "Scott", "Bennett", "Reed", "Cooper", "Bailey", "Rivera", "Hughes"]


def random_name() -> tuple[str, str]:
    """A plausible disposable first/last name for signup forms — never the operator's real identity."""
    return secrets.choice(_FIRST_NAMES), secrets.choice(_LAST_NAMES)


def find_site(key: str, config_path: Path | None = None) -> dict | None:
    """The site dict matching `key` — its `name` OR its `start_url` (trailing slash ignored), searched
    in `config_path` (if given) else both source configs. Accepting the URL too lets the operator paste
    either the site name or the signup URL straight from the worklist."""
    key_url = key.rstrip("/")
    for cfg in ([config_path] if config_path else SOURCE_CONFIGS):
        if not cfg or not cfg.exists():
            continue
        for site in json.loads(cfg.read_text()).get("sites", []):
            if site.get("name") == key or site.get("start_url", "").rstrip("/") == key_url:
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


def provision_site(browser, site: dict) -> str:
    """Provision one site interactively in a fresh context: mint a disposable burner, let the operator
    sign up / clear the wall by hand, then save the session + burner and wire it into the human config.
    Returns 'provisioned' or 'skipped'. Raises SystemExit to stop a whole batch."""
    ref = f"{site['name']}-human-1"
    mailbox = mailtm.create_mailbox()
    first, last = random_name()
    print("\n" + "=" * 72)
    print(f"  PROVISIONING: {site['name']}   →  {site['start_url']}")
    print("  Sign up with this DISPOSABLE burner (NEVER a personal account):")
    print(f"    full name : {first} {last}")
    print(f"    first name: {first}")
    print(f"    last name : {last}")
    print(f"    email     : {mailbox['address']}")
    print(f"    password  : {mailbox['password']}")
    print(f"  Read verification email here or at {mailtm.WEBMAIL_URL} (same email + password).")
    print("=" * 72)

    context = new_stealth_context(browser)
    page = context.new_page()
    try:
        page.goto(site["start_url"], wait_until="domcontentloaded", timeout=60000)
    except Exception as e:
        logger.warning(f"initial navigation issue ({e}); navigate manually in the window")
    print("\n  Sign up / log in with the burner above and solve any captcha. When you are fully signed\n"
          "  in (the app's own screen is showing), return here.\n")
    state = None
    app_url = site["start_url"]
    try:
        while True:
            cmd = input("  [Enter]=done | mail=show email | skip=skip this site | q=quit batch : ").strip().lower()
            if cmd == "":
                app_url = page.url or site["start_url"]
                state = context.storage_state(indexed_db=True)
                break
            if cmd == "skip":
                logger.info(f"[{site['name']}] skipped by operator")
                return "skipped"
            if cmd == "q":
                raise SystemExit("batch stopped by operator")
            if cmd == "mail":
                show_latest_mail(mailbox["token"])
    finally:
        context.close()

    app_secrets.save_storage_state(ref, state)  # raises if the dump is empty (never actually logged in)
    app_secrets.register_burner(ref, mailbox["address"], mailbox["password"],
                                first_name=first, last_name=last)
    provisioning_status.mark(site["name"], provisioning_status.REGISTERED,
                             provisioning=site.get("provisioning", ""))
    upsert_human_config(site, ref, app_url)
    print(f"\n  ✓ provisioned {site['name']}: session saved ({ref}), burner registered, added to {HUMAN_CONFIG.name}")
    return "provisioned"


def batch_targets(config_path: Path | None) -> list[dict]:
    """Every needs_human site not yet provisioned (registered), resolved to site dicts, name-sorted."""
    done = set(provisioning_status.sites_with_status(provisioning_status.REGISTERED))
    todo = sorted(n for n in provisioning_status.sites_with_status(provisioning_status.NEEDS_HUMAN) if n not in done)
    return [s for s in (find_site(n, config_path) for n in todo) if s]


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-in-the-loop login provisioning (headed browser)")
    parser.add_argument("site", nargs="?", default=None, help="Site name or signup URL (omit with --all/--sites)")
    parser.add_argument("--sites", default=None, help="Comma-separated site names/URLs to provision in a loop")
    parser.add_argument("--all", action="store_true", help="Provision ALL needs_human sites in a loop")
    parser.add_argument("--config", default=None, help="Source config path (default: search both)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config_path = Path(args.config) if args.config else None

    if args.all:
        sites = batch_targets(config_path)
    elif args.sites:
        sites = [find_site(k.strip(), config_path) for k in args.sites.split(",")]
    elif args.site:
        sites = [find_site(args.site, config_path)]
    else:
        parser.error("give a SITE / URL, or --sites a,b,c, or --all")
    sites = [s for s in sites if s]
    if not sites:
        raise SystemExit("no matching sites to provision (check names against the source configs)")

    batch = len(sites) > 1
    counts = {"provisioned": 0, "skipped": 0, "failed": 0}
    with sync_playwright() as playwright:
        browser = launch_realistic_chrome(playwright)
        try:
            for i, site in enumerate(sites, 1):
                if batch:
                    print(f"\n########## [{i}/{len(sites)}] {site['name']} ##########")
                try:
                    counts[provision_site(browser, site)] += 1
                except SystemExit as e:
                    print(f"\n{e}")
                    break
                except Exception as e:
                    logger.error(f"[{site['name']}] provisioning failed ({e}); moving on")
                    counts["failed"] += 1
        finally:
            browser.close()

    print(f"\n  done: {counts['provisioned']} provisioned, {counts['skipped']} skipped, {counts['failed']} failed")
    if counts["provisioned"]:
        print("  Next: tell the assistant to run the batch gather over public_sites_human.json\n")


if __name__ == "__main__":
    main()
