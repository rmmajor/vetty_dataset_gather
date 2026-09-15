"""Autonomous login-gated capture — self-register a disposable burner, then run a site's
login-gated tasks with that one session.

Per site: mint a real disposable mailbox (mail.tm) and run the site's tasks easy-first. The first
task runs in `discovered` auth mode with the masked burner credentials + the webmail inbox, so the
teacher signs in (creating the account first when the site has none), then completes the task; its
authenticated session (storage_state) is captured and REUSED via `pre_session` for the remaining
tasks so there is no repeated login. Session + burner are PERSISTED (app_secrets), so a later run of
the same site skips registration entirely and starts every task signed in — a site is registered
once, not once per run. If the first task cannot establish a session — registration
blocked by captcha / phone / disposable-domain rejection — the site is abandoned after that single
attempt, so a gated-beyond-reach site costs one trajectory rather than many; those sites belong to
the human provisioning lane.

Only attempts sites whose `provisioning` is plausibly autonomous (default: open_signup, email_verify).
Credentials only ever reach the recording as `<secret:...>` placeholders (driver-side substitution),
consistent with the burner-only invariant.

Run: python -m cua.pipeline.lanes.logingated [--sites a,b] [--provisioning open_signup,email_verify]
     [--max-tasks-per-site N] [--max-sites N] [--parallel K]
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from cua.pipeline import app_secrets, mailtm, provisioning_status
from cua.pipeline.driver_session import boot_driver, run_blocking
from cua.pipeline.lanes.capture_lane import (
    finalize_tasks,
    is_capture_exhausted,
    is_capture_settled,
    needs_verification,
    record_capture,
)
from cua.pipeline.registry import RepoRegistry
from cua.pipeline.sandbox import DockerSandbox, install_sigterm_handler, safe_parallel_sandboxes
from cua.pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from cua.pipeline.sites.registry import LOGINGATED_CONFIG, load_sites
from cua.pipeline.teacher import run_teacher_task, session_authenticated
from cua.settings import CAPTURE
from cua.trajectory.schema import safe_task_id

logger = logging.getLogger(__name__)

AUTONOMOUS_PROVISIONING = {"open_signup", "email_verify"}
DIFFRANK = {"easy": 0, "medium": 1, "hard": 2}
# The first task per site also signs up, so budget extra expected actions or a goal-completing run is mis-graded "meandering".
REGISTRATION_ACTION_BUDGET = 16


def _site_identity(site: dict, pending: list, summary: dict):
    """The burner this site runs as — a persisted one, or a fresh mailbox — or None when it is fail-closed."""
    provisioned = app_secrets.find_provisioned_session(site["name"])
    unvetted_refs = app_secrets.unvetted_session_refs(site["name"]) if provisioned is None else []
    if unvetted_refs:
        # Unknown provenance (possibly a real account): never reuse it, never overwrite it either.
        logger.error(
            f"[{site['name']}] FAIL-CLOSED: storage_state {unvetted_refs} exists but no "
            f"registered burner — delete it to re-provision this site; skipping"
        )
        summary["fail_closed"] = True
        return None
    if provisioned:
        summary["resumed"] = True
        logger.info(
            f"[{site['name']}] reusing persisted burner session '{provisioned['ref']}' "
            f"({provisioned['provisioning_mode']}) — {len(pending)} task(s) run signed in"
        )
        return None, provisioned["storage_state"], provisioned["secrets"], provisioned["ref"]
    mailbox = mailtm.create_mailbox()
    # One masked secret reused for the site account and the mail.tm inbox login.
    secrets = {"email": mailbox["address"], "password": mailbox["password"]}
    logger.info(f"[{site['name']}] mailbox={mailbox['address']} (masked) — {len(pending)} task(s) to run")
    return mailbox, None, secrets, app_secrets.session_ref(site["name"], "generated")


async def _register_outcome(site: dict, task: dict, state: dict, site_dir: Path, registry) -> bool:
    """Whether the first task left a REUSABLE AUTHENTICATED SESSION, judged from the final screen."""
    # Not whether the task's goal completed: a slow email-verify login often establishes the session yet exhausts its budget first.
    session = state.get("storage_state")
    authed = bool(session) and await session_authenticated(
        site_dir / f"task_{safe_task_id(task['id'])}",
        site["name"],
        state.get("agent_summary", ""),
        registry=registry,
        full_name=f"site:{site['name']}",
    )
    logger.info(
        f"[{site['name']}] registration: storage_state="
        f"{'present' if session else 'EMPTY'}"
        f"({len(json.dumps(session)) if session else 0}B), authenticated={authed}"
    )
    return authed


def _persist_session(site: dict, ref: str, mailbox: dict, session: dict) -> None:
    """Persist the burner + its session so later runs of this site start signed in, best effort."""
    try:
        app_secrets.save_provisioned_session(
            ref, mailbox["address"], mailbox["password"], session, provisioning_mode="generated"
        )
        logger.info(
            f"[{site['name']}] persisted burner + session as '{ref}' — future "
            f"runs of this site start signed in, no re-registration"
        )
    except (ValueError, OSError) as e:
        # An unpersistable session must not cost this run its remaining tasks.
        logger.error(f"[{site['name']}] could not persist the burner session '{ref}': {e}")


def _mark_registration_outcome(site: dict, summary: dict, attempted: bool) -> None:
    """Persist the registration outcome so genuinely-blocked sites become a human-provisioning queue."""
    if summary["registered"]:
        provisioning_status.mark(
            site["name"], provisioning_status.REGISTERED, provisioning=site.get("provisioning", "")
        )
        return
    if not attempted:
        return
    # A site with prior autonomous success stays in this lane: a human would hit the same flaky email, so a transient failure is the wrong reason to route there.
    already_doable = (
        provisioning_status.get_status(site["name"]) == provisioning_status.REGISTERED or summary["verified"] > 0
    )
    if already_doable:
        logger.info(
            f"[{site['name']}] registration failed this run but the site has prior "
            f"autonomous success — keeping it in the autonomous lane (not needs_human)"
        )
        return
    provisioning_status.mark(
        site["name"],
        provisioning_status.NEEDS_HUMAN,
        reason="autonomous registration could not establish a session "
        "(captcha / phone / disposable-email block likely)",
        provisioning=site.get("provisioning", ""),
    )
    summary["needs_human"] = True


async def _run_pending(client, sandbox, site, site_dir, pending, identity, registry, summary, states, ran):
    """Run this site's pending tasks, the first one registering the burner, and stop at the first wall."""
    mailbox, captured_session, secrets, ref = identity
    attempted_registration = False
    for task in pending:
        if rate_limit_tracker.limit_reached():
            raise UsageLimitReached("usage limit reached during login-gated capture")
        is_registration = captured_session is None
        run_task = task
        auth = {
            "auth_mode": "pre_session",
            "auth_config": {"auth_mode": "pre_session", "storage_state": captured_session},
        }
        if is_registration:
            # The first task also registers/signs in (discovered mode); give it the signup budget.
            expected = int(task.get("expected_actions", 10)) + REGISTRATION_ACTION_BUDGET
            run_task = {**task, "expected_actions": expected}
            auth = {"auth_mode": "discovered", "webmail_url": mailtm.WEBMAIL_URL, "webmail_preauthed": False}
            auth["capture_session"] = True
        # Only the teacher call is guarded, so one task's failure never skips finalizing the tasks already captured here.
        try:
            state = await run_teacher_task(
                client,
                run_task,
                site["start_url"],
                registry=registry,
                full_name=f"site:{site['name']}",
                autonomy="autonomous",
                secrets=secrets,
                **auth,
            )
        except UsageLimitReached:
            raise
        except Exception as e:
            logger.error(f"[{site['name']}] task '{task['id']}' errored ({e}); stopping this site")
            break
        states[task["id"]] = state
        ran.append(task)
        summary["captured"] += 1
        await record_capture(state, task, sandbox=sandbox, site_dir=site_dir, label=site["name"])
        if not is_registration:
            continue
        attempted_registration = True
        if await _register_outcome(site, task, state, site_dir, registry):
            captured_session = state["storage_state"]
            summary["registered"] = True
            _persist_session(site, ref, mailbox, captured_session)
        else:
            logger.warning(
                f"[{site['name']}] no authenticated session after the first task — "
                f"site is gated beyond autonomous reach; skipping the remaining "
                f"{len(pending) - 1} task(s) (needs human provisioning)"
            )
            break
    return attempted_registration


async def capture_site(site: dict, base_dir: Path, registry, max_tasks: int | None) -> dict:
    """Register a burner on one login-gated site, then run its tasks on that single session."""
    site_dir = base_dir / f"site_{site['name']}"
    tasks = sorted(site.get("tasks", []), key=lambda t: DIFFRANK.get(t.get("difficulty"), 1))
    if max_tasks:
        tasks = tasks[:max_tasks]
    pending = [t for t in tasks if not is_capture_settled(site_dir / f"task_{safe_task_id(t['id'])}")]
    # Captured on a prior run but never verified: resume would skip these as "valid" so they would never become trainable.
    orphaned = [t for t in tasks if t not in pending and needs_verification(site_dir / f"task_{safe_task_id(t['id'])}")]
    summary = {
        "site": site["name"],
        "registered": False,
        "captured": 0,
        "verified": 0,
        "skipped": len(tasks) - len(pending),
    }
    if not pending and not orphaned:
        exhausted = [t for t in tasks if is_capture_exhausted(site_dir / f"task_{safe_task_id(t['id'])}")]
        logger.info(
            f"[{site['name']}] all {len(tasks)} task(s) settled "
            f"({len(tasks) - len(exhausted)} recorded, {len(exhausted)} out of retries)"
        )
        return summary
    if not pending:
        # Verification is host-side, so an orphan-only resume needs no sandbox and no mailbox.
        logger.info(
            f"[{site['name']}] finalizing {len(orphaned)} orphaned trajectory(ies) from a prior interrupted run"
        )
        results = await finalize_tasks(orphaned, site, site_dir, {}, registry)
        summary["verified"] = sum(1 for r in results.values() if r["verdict"].get("final_success"))
        return summary

    identity = _site_identity(site, pending, summary)
    if identity is None:
        return summary

    sandbox = DockerSandbox(container_timeout=CAPTURE.logingated_container_timeout)
    attempted_registration = False
    ran: list[dict] = []
    states: dict = {}
    try:
        await run_blocking(sandbox.start, f"public/{site['name']}")
        client = await boot_driver(sandbox)
        attempted_registration = await _run_pending(
            client, sandbox, site, site_dir, pending, identity, registry, summary, states, ran
        )
    finally:
        await run_blocking(sandbox.stop)

    # Finalizing here sets summary["verified"] first, so the registration outcome below can tell a flaky failure from a wall.
    results = await finalize_tasks(ran + orphaned, site, site_dir, states, registry, secrets=identity[2])
    summary["verified"] = sum(1 for r in results.values() if r["verdict"].get("final_success"))
    _mark_registration_outcome(site, summary, attempted_registration)
    return summary


def _sites_for_run(args) -> list[dict]:
    """The sites this run attempts: the config filtered by provisioning, minus the human-provisioning queue."""
    cfg = Path(args.config) if args.config else LOGINGATED_CONFIG
    only = [n.strip() for n in args.sites.split(",") if n.strip()] if args.sites else None
    sites = load_sites(cfg, only=only, allow_provisioning=set(args.provisioning.split(",")), include_skipped=True)
    # A site already queued for the human lane would just fail at the same wall, so it is not re-attempted.
    if not only and not args.retry_blocked:
        blocked = set(provisioning_status.sites_with_status(provisioning_status.NEEDS_HUMAN))
        before = len(sites)
        sites = [s for s in sites if s["name"] not in blocked]
        if before - len(sites):
            logger.info(
                f"Skipping {before - len(sites)} site(s) already marked needs_human "
                f"(human-provisioning queue; --retry-blocked to force a re-attempt)"
            )
    return sites[: args.max_sites] if args.max_sites else sites


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous login-gated capture (self-register burner)")
    parser.add_argument(
        "--sites", type=str, default=None, help="Comma-separated site names (overrides --provisioning filter)"
    )
    parser.add_argument(
        "--provisioning",
        type=str,
        default=",".join(sorted(AUTONOMOUS_PROVISIONING)),
        help="Comma-separated provisioning classes to attempt",
    )
    parser.add_argument("--max-tasks-per-site", type=int, default=None)
    parser.add_argument("--max-sites", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--traj-dir", type=str, default="trajectories/")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--retry-blocked", action="store_true", help="Re-attempt sites already marked needs_human (default: skip them)"
    )
    return parser


async def main():
    args = build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    install_sigterm_handler()
    try:
        sites = _sites_for_run(args)
    except ValueError as e:
        raise SystemExit(f"--sites: {e}")
    logger.info(f"Login-gated autonomous capture over {len(sites)} site(s): {', '.join(s['name'] for s in sites)}")

    registry = RepoRegistry("pipeline_data/registry.db")
    results: list[dict] = []
    parallel = safe_parallel_sandboxes(int(args.parallel))
    semaphore = asyncio.Semaphore(parallel)
    logger.info(f"Running {parallel} site(s) concurrently (requested {args.parallel})")

    async def run_site(site):
        async with semaphore:
            if rate_limit_tracker.limit_reached():
                return
            try:
                r = await capture_site(site, Path(args.traj_dir), registry, args.max_tasks_per_site)
                results.append(r)
                session = "reused" if r.get("resumed") else "registered" if r["registered"] else "none"
                logger.info(
                    f"[{r['site']}] done ({len(results)}/{len(sites)}): session={session}, "
                    f"{r['captured']} captured, {r['verified']} verified, {r['skipped']} skipped"
                )
            except UsageLimitReached:
                logger.warning(f"[{site['name']}] usage limit reached — captures so far are kept")
            except Exception as e:
                logger.error(f"[{site['name']}] failed: {e}")

    try:
        await asyncio.gather(*[run_site(s) for s in sites])
    finally:
        registry.close()

    registered = sum(1 for r in results if r.get("registered"))
    reused = sum(1 for r in results if r.get("resumed"))
    logger.info(
        f"DONE: {len(results)}/{len(sites)} sites attempted, {registered} newly registered, "
        f"{reused} run on a persisted session, {sum(r['captured'] for r in results)} captured, "
        f"{sum(r['verified'] for r in results)} verified"
    )


if __name__ == "__main__":
    asyncio.run(main())
