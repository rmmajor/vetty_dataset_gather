"""Cheap capturability pre-filter for login-gated sites.

The autonomous lane cannot capture sites guarded by an anti-bot WAF (Cloudflare challenge, Akamai /
Imperva / PerimeterX / DataDome) — the headless sandbox browser on a datacenter IP is blocked before
any signup/email step. Discovering that with a full teacher run is wasteful, so this HTTP probe marks
the unambiguously-blocked sites `needs_human` (the human-provisioning lane — they need a real browser
/ residential IP) BEFORE the capture wave, which already skips `needs_human`.

Same host egress IP as the sandbox, so IP-level blocks and full-page WAF challenges show here too.
Conservative by design: it only rejects on explicit WAF block / challenge signatures and leaves
anything uncertain (incl. transient network errors, and Turnstile-only-on-submit pages that still
render) for the more accurate capture attempt to judge.

Run: python -m repo_pipeline.probe_capturable [--provisioning email_verify] [--dry-run] [--workers N]
"""
import argparse
import concurrent.futures
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

from repo_pipeline import provisioning_status

logger = logging.getLogger(__name__)

CONFIG = Path(__file__).parent / "public_sites_actions_logingated.json"
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# Explicit full-page WAF block / challenge signatures (matched as lowercased substrings of the body).
BLOCK_SIGNATURES = [
    "just a moment...",
    "attention required! | cloudflare",
    "cf-browser-verification", "cf_chl_opt", "cf-challenge-running",
    "checking if the site connection is secure",
    "enable javascript and cookies to continue",
    "verifying you are human. this may take a few seconds",
    "performance & security by cloudflare",
    "pardon our interruption",              # PerimeterX / HUMAN
    "request unsuccessful. incapsula",       # Imperva
    "incident id:",                          # Imperva Incapsula block
    "powered by perimeterx",
    "access to this page has been denied",   # PerimeterX
    "captcha-delivery.com",                  # DataDome
    "you don't have permission to access",   # Akamai / generic deny
    "akamai reference",
]
WAF_SERVERS = ("cloudflare", "akamaighost", "akamai", "imperva")


def probe(start_url: str, timeout: int = 15) -> tuple[bool, str]:
    """(capturable, reason). Conservative: only an explicit WAF block / challenge → not capturable."""
    req = urllib.request.Request(start_url, headers={"User-Agent": _UA, "Accept": "text/html,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            server = (resp.headers.get("Server") or "").lower()
            body = resp.read(80_000).decode("utf-8", "replace").lower()
    except urllib.error.HTTPError as e:
        status = e.code
        server = (e.headers.get("Server") or "").lower() if e.headers else ""
        try:
            body = e.read(80_000).decode("utf-8", "replace").lower()
        except Exception:
            body = ""
    except Exception as e:
        # A transient network error is NOT a block — let the capture decide.
        return True, f"probe-network-error ({str(e)[:60]})"
    hit = next((s for s in BLOCK_SIGNATURES if s in body), None)
    if hit:
        return False, f"anti-bot WAF challenge page (signature: {hit!r})"
    if status in (403, 429, 503) and any(w in server for w in WAF_SERVERS):
        return False, f"WAF block HTTP {status} (server: {server!r})"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="HTTP capturability pre-filter (mark WAF-blocked sites needs_human)")
    parser.add_argument("--provisioning", type=str, default="email_verify",
                        help="Comma-separated provisioning classes to probe")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write needs_human marks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = Path(args.config) if args.config else CONFIG
    allow = set(args.provisioning.split(","))
    sites = [s for s in json.loads(cfg.read_text())["sites"] if s.get("provisioning") in allow]
    # Skip sites already classified (registered, or already in the human queue) — don't re-probe.
    already = (set(provisioning_status.sites_with_status(provisioning_status.NEEDS_HUMAN))
               | set(provisioning_status.sites_with_status(provisioning_status.REGISTERED)))
    todo = [s for s in sites if s["name"] not in already]
    logger.info(f"Probing {len(todo)} site(s) ({len(sites) - len(todo)} already classified) "
                f"[provisioning={args.provisioning}]")

    def work(s):
        cap, reason = probe(s["start_url"])
        return s, cap, reason

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(work, todo))

    blocked = 0
    for s, cap, reason in results:
        if cap:
            continue
        blocked += 1
        logger.info(f"[{s['name']}] BLOCKED — {reason}")
        if not args.dry_run:
            provisioning_status.mark(s["name"], provisioning_status.NEEDS_HUMAN,
                                     reason=f"{reason} — needs human lane (real browser/IP)",
                                     provisioning=s.get("provisioning", ""))
    capturable = len(todo) - blocked
    logger.info(f"DONE: {capturable} capturable, {blocked} WAF-blocked"
                f"{' (dry-run, not marked)' if args.dry_run else ' (marked needs_human)'}")


if __name__ == "__main__":
    main()
