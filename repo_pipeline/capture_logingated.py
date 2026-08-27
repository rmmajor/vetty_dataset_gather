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

Run: python -m repo_pipeline.capture_logingated [--sites a,b] [--provisioning open_signup,email_verify]
     [--max-tasks-per-site N] [--max-sites N] [--parallel K]
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path

from repo_pipeline import app_secrets, driver_session, mailtm, provisioning_status
from repo_pipeline.agent_capture import run_teacher_task, session_authenticated
from repo_pipeline.instrumentation import RESULTS_DIR
from repo_pipeline.public_capture import _verify_and_finalize, resolve_named_sites
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox, install_sigterm_handler, safe_parallel_sandboxes
from repo_pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from repo_pipeline.stage6_generate import (_is_valid_trajectory, _safe_task_id, is_capture_exhausted,
                                           is_capture_settled)

logger = logging.getLogger(__name__)

CONFIG = Path(__file__).parent / "public_sites_actions_logingated.json"
CONTAINER_TIMEOUT = 5400
AUTONOMOUS_PROVISIONING = {"open_signup", "email_verify"}
DIFFRANK = {"easy": 0, "medium": 1, "hard": 2}
# The first task per site also performs the one-time signup. Budget extra expected actions for it so
# a goal-completing run is not mis-graded "meandering" just for the registration steps it had to do.
REGISTRATION_ACTION_BUDGET = 16


async def _run(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def load_sites(config_path: Path, allow: set[str],
               only: list[str] | set[str] | None = None) -> list[dict]:
    if only:
        # Searched across every config, not just this lane's: the worklist is split over several.
        return resolve_named_sites(list(only), config_path)
    sites = json.loads(config_path.read_text())["sites"]
    return [s for s in sites if s.get("provisioning") in allow]


def _needs_verification(traj_dir: Path) -> bool:
    """A captured trajectory that was never verified — capture_meta written without a 'verifier'
    verdict (an interrupt / usage-limit abort finalized capture but not verification). Resume must
    finalize these rather than skip them, else good captures never become trainable."""
    if not _is_valid_trajectory(traj_dir):
        return False
    try:
        return "verifier" not in json.loads((traj_dir / "capture_meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False


async def _finalize_tasks(tasks: list, site: dict, site_dir: Path, states: dict,
                          secrets: dict | None, summary: dict, registry):
    """Verify + finalize trajectories host-side. Used for tasks captured this run AND for orphans
    from a prior interrupted run — `states.get(id)` is None for orphans (verify_trajectory falls
    back to capture_meta), and `secrets` is None when there is no burner password to assert the
    no-leak invariant against (driver-side masking already guarantees only placeholders are
    recorded)."""
    for task in tasks:
        traj_dir = site_dir / f"task_{_safe_task_id(task['id'])}"
        if not traj_dir.exists():
            continue
        verdict = await _verify_and_finalize(traj_dir, site, task, states.get(task["id"]), registry)
        leaked = bool(secrets) and secrets["password"] in (traj_dir / "trajectory.json").read_text()
        if leaked:
            logger.error(f"[{site['name']}] SECRET LEAK in '{task['id']}': password appears literally")
        if verdict.get("final_success"):
            summary["verified"] += 1
        logger.info(f"[{site['name']}] task '{task['id']}': verified={verdict.get('final_success')} "
                    f"({verdict.get('efficiency')}) leaked={leaked}")


async def capture_site(site: dict, base_dir: Path, registry, max_tasks: int | None) -> dict:
    site_dir = base_dir / f"site_{site['name']}"
    tasks = sorted(site.get("tasks", []), key=lambda t: DIFFRANK.get(t.get("difficulty"), 1))
    if max_tasks:
        tasks = tasks[:max_tasks]
    pending = [t for t in tasks if not is_capture_settled(site_dir / f"task_{_safe_task_id(t['id'])}")]
    # Captured on a prior run but never verified (interrupt / usage-limit abort). Resume would skip
    # these as "valid" so they would never become trainable — finalize them here. Verification is
    # host-side, so an orphan-only resume needs no sandbox or mailbox.
    orphaned = [t for t in tasks if t not in pending
                and _needs_verification(site_dir / f"task_{_safe_task_id(t['id'])}")]
    summary = {"site": site["name"], "registered": False, "captured": 0, "verified": 0,
               "skipped": len(tasks) - len(pending)}
    if not pending and not orphaned:
        exhausted = [t for t in tasks
                     if is_capture_exhausted(site_dir / f"task_{_safe_task_id(t['id'])}")]
        logger.info(f"[{site['name']}] all {len(tasks)} task(s) settled "
                    f"({len(tasks) - len(exhausted)} recorded, {len(exhausted)} out of retries)")
        return summary
    if not pending:
        logger.info(f"[{site['name']}] finalizing {len(orphaned)} orphaned trajectory(ies) from a "
                    f"prior interrupted run")
        await _finalize_tasks(orphaned, site, site_dir, {}, None, summary, registry)
        return summary

    provisioned = app_secrets.find_provisioned_session(site["name"])
    unvetted_refs = app_secrets.unvetted_session_refs(site["name"]) if provisioned is None else []
    if unvetted_refs:
        # Unknown provenance (possibly a real account): never reuse it, never overwrite it either.
        logger.error(f"[{site['name']}] FAIL-CLOSED: storage_state {unvetted_refs} exists but no "
                     f"registered burner — delete it to re-provision this site; skipping")
        summary["fail_closed"] = True
        return summary

    if provisioned:
        # Registered on an earlier run: no signup and no new mailbox, just the persisted identity.
        mailbox, captured_session = None, provisioned["storage_state"]
        secrets, ref = provisioned["secrets"], provisioned["ref"]
        summary["resumed"] = True
        logger.info(f"[{site['name']}] reusing persisted burner session '{ref}' "
                    f"({provisioned['provisioning_mode']}) — {len(pending)} task(s) run signed in")
    else:
        mailbox, captured_session = mailtm.create_mailbox(), None
        # One masked secret reused for the site account and the mail.tm inbox login.
        secrets = {"email": mailbox["address"], "password": mailbox["password"]}
        ref = app_secrets.session_ref(site["name"], "generated")
        logger.info(f"[{site['name']}] mailbox={mailbox['address']} (masked) — {len(pending)} task(s) to run")

    sandbox = DockerSandbox(container_timeout=CONTAINER_TIMEOUT)
    attempted_registration = False
    ran: list[dict] = []
    states: dict = {}
    try:
        await _run(sandbox.start, f"public/{site['name']}")
        await _run(driver_session.install_driver, sandbox)
        await _run(driver_session.start_driver, sandbox)
        client = driver_session.DriverClient(sandbox)

        for task in pending:
            if rate_limit_tracker.limit_reached():
                raise UsageLimitReached("usage limit reached during login-gated capture")
            is_registration = captured_session is None
            try:
                if is_registration:
                    # First task also registers/signs in (discovered mode); give it the signup budget.
                    reg_task = {**task, "expected_actions": int(task.get("expected_actions", 10))
                                + REGISTRATION_ACTION_BUDGET}
                    state = await run_teacher_task(
                        client, reg_task, site["start_url"], registry=registry, full_name=f"site:{site['name']}",
                        autonomy="autonomous", auth_mode="discovered", secrets=secrets,
                        webmail_url=mailtm.WEBMAIL_URL, webmail_preauthed=False, capture_session=True,
                    )
                    attempted_registration = True
                else:
                    # Start signed in; the masked credentials ride along in case the session expired.
                    state = await run_teacher_task(
                        client, task, site["start_url"], registry=registry, full_name=f"site:{site['name']}",
                        autonomy="autonomous", auth_mode="pre_session", secrets=secrets,
                        auth_config={"auth_mode": "pre_session", "storage_state": captured_session},
                    )
            except UsageLimitReached:
                raise
            except Exception as e:
                # A single task's failure must not skip finalizing the tasks already captured this
                # site (an earlier bug: a later /session error aborted the whole site unverified).
                logger.error(f"[{site['name']}] task '{task['id']}' errored ({e}); stopping this site")
                break
            states[task["id"]] = state
            ran.append(task)
            summary["captured"] += 1
            site_dir.mkdir(parents=True, exist_ok=True)
            await _run(sandbox.copy_out, RESULTS_DIR, str(site_dir))
            logger.info(f"[{site['name']}] task '{task['id']}': {state['actions']} actions, "
                        f"outcome={state['agent_outcome']}")
            if is_registration:
                # Registration succeeds when it leaves a REUSABLE AUTHENTICATED SESSION, judged from
                # the final screen — not when the bundled task's goal completed. A slow email-verify
                # login often establishes the session yet exhausts its action budget before finishing
                # the task; gating on goal-completion discarded those sessions and mis-filed working
                # sites as needs_human.
                session = state.get("storage_state")
                traj_dir = site_dir / f"task_{_safe_task_id(task['id'])}"
                authed = bool(session) and await session_authenticated(
                    traj_dir, site["name"], state.get("agent_summary", ""),
                    registry=registry, full_name=f"site:{site['name']}",
                )
                logger.info(f"[{site['name']}] registration: storage_state="
                            f"{'present' if session else 'EMPTY'}"
                            f"({len(json.dumps(session)) if session else 0}B), authenticated={authed}")
                if authed:
                    captured_session = session
                    summary["registered"] = True
                    try:
                        app_secrets.save_provisioned_session(
                            ref, mailbox["address"], mailbox["password"], session,
                            provisioning_mode="generated",
                        )
                        logger.info(f"[{site['name']}] persisted burner + session as '{ref}' — future "
                                    f"runs of this site start signed in, no re-registration")
                    except (ValueError, OSError) as e:
                        # Best-effort: an unpersistable session must not cost this run its remaining tasks.
                        logger.error(f"[{site['name']}] could not persist the burner session '{ref}': {e}")
                else:
                    logger.warning(f"[{site['name']}] no authenticated session after the first task — "
                                   f"site is gated beyond autonomous reach; skipping the remaining "
                                   f"{len(pending) - 1} task(s) (needs human provisioning)")
                    break
    finally:
        await _run(sandbox.stop)

    # Verify + finalize FIRST (copy_out clobbers per-task host files mid-loop): freshly captured this
    # run, plus any orphans from a prior interrupted run. This sets summary["verified"], so the
    # registration outcome below can distinguish a flaky failure on a proven-doable site from a wall.
    await _finalize_tasks(ran + orphaned, site, site_dir, states, secrets, summary, registry)

    # Persist the registration outcome so genuinely-blocked sites become a human-provisioning queue.
    if summary["registered"]:
        provisioning_status.mark(site["name"], provisioning_status.REGISTERED,
                                 provisioning=site.get("provisioning", ""))
    elif attempted_registration:
        # A flaky failure (e.g. a verification email that did not arrive this run) must NOT downgrade
        # a site that has ALREADY proved autonomously doable — registered on a prior run, or having
        # produced a trainable trajectory this run. Those stay in the autonomous lane (retried next
        # run); only sites with no evidence of autonomous success go to the human-provisioning queue
        # (a human hits the same flaky email, so a transient failure is the wrong reason to route there).
        already_doable = (provisioning_status.get_status(site["name"]) == provisioning_status.REGISTERED
                          or summary["verified"] > 0)
        if already_doable:
            logger.info(f"[{site['name']}] registration failed this run but the site has prior "
                        f"autonomous success — keeping it in the autonomous lane (not needs_human)")
        else:
            provisioning_status.mark(site["name"], provisioning_status.NEEDS_HUMAN,
                                     reason="autonomous registration could not establish a session "
                                            "(captcha / phone / disposable-email block likely)",
                                     provisioning=site.get("provisioning", ""))
            summary["needs_human"] = True
    return summary


async def main():
    parser = argparse.ArgumentParser(description="Autonomous login-gated capture (self-register burner)")
    parser.add_argument("--sites", type=str, default=None, help="Comma-separated site names (overrides --provisioning filter)")
    parser.add_argument("--provisioning", type=str, default=",".join(sorted(AUTONOMOUS_PROVISIONING)),
                        help="Comma-separated provisioning classes to attempt")
    parser.add_argument("--max-tasks-per-site", type=int, default=None)
    parser.add_argument("--max-sites", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--traj-dir", type=str, default="trajectories/")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--retry-blocked", action="store_true",
                        help="Re-attempt sites already marked needs_human (default: skip them)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    install_sigterm_handler()
    cfg = Path(args.config) if args.config else CONFIG
    only = [n.strip() for n in args.sites.split(",") if n.strip()] if args.sites else None
    allow = set(args.provisioning.split(","))
    try:
        sites = load_sites(cfg, allow, only)
    except ValueError as e:
        raise SystemExit(f"--sites: {e}")
    # Sites already queued for the human lane are not re-attempted autonomously (they would just
    # fail again at the same wall) unless explicitly named or --retry-blocked is set.
    if not only and not args.retry_blocked:
        blocked = set(provisioning_status.sites_with_status(provisioning_status.NEEDS_HUMAN))
        before = len(sites)
        sites = [s for s in sites if s["name"] not in blocked]
        if before - len(sites):
            logger.info(f"Skipping {before - len(sites)} site(s) already marked needs_human "
                        f"(human-provisioning queue; --retry-blocked to force a re-attempt)")
    if args.max_sites:
        sites = sites[:args.max_sites]
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
                logger.info(f"[{r['site']}] done ({len(results)}/{len(sites)}): session={session}, "
                            f"{r['captured']} captured, {r['verified']} verified, {r['skipped']} skipped")
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
    logger.info(f"DONE: {len(results)}/{len(sites)} sites attempted, {registered} newly registered, "
                f"{reused} run on a persisted session, {sum(r['captured'] for r in results)} captured, "
                f"{sum(r['verified'] for r in results)} verified")


if __name__ == "__main__":
    asyncio.run(main())
