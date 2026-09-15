import asyncio
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from claude_agent_sdk import ClaudeAgentOptions

from cua.pipeline import app_runner, app_secrets, driver_session, mail_service, schemas
from cua.pipeline._processing import run_repo
from cua.pipeline.agent_capture import run_teacher_task, verify_trajectory
from cua.pipeline.agent_runner import run_agent
from cua.pipeline.frameworks import resolve_profile
from cua.pipeline.instrumentation import REPO_DIR, RESULTS_DIR, setup_repo_instrumentation
from cua.pipeline.llm_config import MODEL_FOR_CLASSIFICATION, MODEL_FOR_GROUNDED_GENERATION
from cua.pipeline.logging_utils import StageProgress, repo_logger
from cua.pipeline.registry import RepoRegistry
from cua.pipeline.sandbox import DockerSandbox
from cua.pipeline.schemas import FailureCategory, QualityReject, RepoStatus, RetryableFailure
from cua.pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from cua.pipeline.task_proposer import propose_tasks

logger = logging.getLogger(__name__)

# Each Claude Agent SDK `query()` carries a fixed overhead (~$0.08 for Sonnet-class models — see
# memory pathb-e2e-and-sdk-budget-floor). Defaults sit comfortably above that floor; tune
# down only after verifying a smaller cap doesn't trip `error_max_budget_usd`.
INSTRUCTION_BUDGET_USD = float(os.environ.get("INSTRUCTION_BUDGET_USD", "0.30"))
TASK_TYPE_BUDGET_USD = float(os.environ.get("TASK_TYPE_BUDGET_USD", "0.30"))
SUITE_RUN_TIMEOUT = 480  # seconds; partial capture on timeout is fine for trajectory mining

# One container hosts a whole repo's agent session (app boot + N teacher tasks), so it needs
# more than the 30-minute default. Teacher turns can stretch enormously under subscription
# rate-limit waits, hence the generous ceiling plus the per-task time reserve below.
AGENT_CONTAINER_TIMEOUT = int(os.environ.get("AGENT_CONTAINER_TIMEOUT", "10800"))
# Don't start another teacher task with less than this much container time left — a task that
# gets cut off mid-flight records an incoherent tail.
AGENT_TASK_TIME_RESERVE = int(os.environ.get("AGENT_TASK_TIME_RESERVE", "900"))
# At most this many verified trajectories are kept per repo — diversity guard so one prolific
# app cannot dominate the corpus the way Booking.com did.
AGENT_MAX_KEPT = int(os.environ.get("AGENT_MAX_KEPT_PER_REPO", "10"))
# Early-abort: stop a repo after this many CONSECUTIVE unproductive tasks (teacher reported
# failure, or the task aborted on an exception). A repo whose backend never came up or whose
# auth wall blocks every flow (e.g. TREK) otherwise grinds through its whole task list — minutes
# per doomed task. A productive task resets the counter.
AGENT_MAX_CONSECUTIVE_FAILS = int(os.environ.get("AGENT_MAX_CONSECUTIVE_FAILS", "3"))
# Bounds re-capture of failed trajectories so a permanently-blocked task cannot burn a sandbox forever.
CAPTURE_MAX_ATTEMPTS = int(os.environ.get("CAPTURE_MAX_ATTEMPTS", "2"))
ATTEMPTS_FILE = "capture_attempts.json"

INSTRUCTION_PROMPT = """You are reading an automated browser test for a web application.
Your job is to write a one-sentence INSTRUCTION describing what user-facing scenario the
test verifies — the goal or intent behind the test, NOT a play-by-play of the recorded
actions.

Voice and rules:
- Phrase the instruction as the user's goal or the tester's stated intent. Examples:
  "Test the login flow.", "Add an item to the cart and check out.",
  "Verify the search filter excludes out-of-stock items."
- Do NOT enumerate clicks, keystrokes, or specific element labels. The instruction must
  work even if the underlying UI is restyled.
- The test function NAME and its DOCSTRING are the strongest signals — use them first.
  The function body is secondary. The recorded action sequence is the weakest signal,
  used only when the name and source give no intent.
- Output one sentence, 4-15 words. No quotes, no markdown, no preamble.

Good vs. bad examples:
- Test `test_login_with_invalid_password`, actions [type 'a@b', type 'wrong', click Sign
  in] → GOOD: "Test that login fails gracefully with an invalid password."
  BAD: "Type an email, type a password, click sign in."
- Test `test_add_to_cart`, actions [click 'iPhone 12', click 'Add to cart', click 'Cart']
  → GOOD: "Add an item to the cart and verify it appears."
  BAD: "Click iPhone 12, click Add to cart, click Cart."
- Test `test_1` (uninformative), actions [type 'banana' into Search, click Search, click
  first result] → GOOD: "Search for a product and open its details page."
  BAD: "Type 'banana', click search, click result."

---

Repository: {repo_name} — {repo_description}
Test function: {test_name}

Test source:
{test_source}

Recorded actions (disambiguation only): {action_summary}

Write the instruction."""

TASK_TYPES = ["auth", "form-fill", "search", "CRUD", "drag-drop", "nav", "other"]

TASK_TYPE_PROMPT = """Classify this web task into EXACTLY ONE category.
Categories: auth, form-fill, search, CRUD, drag-drop, nav, other.

Instruction: {instruction}
Action sequence: {action_types}
Target elements (role: name): {elements}

Reply with ONLY the single category word."""


async def _run(fn, *args):
    """Run a blocking DockerSandbox call off the event loop so concurrent workers overlap."""
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def _is_valid_trajectory(traj_dir: Path) -> bool:
    """A captured trajectory is usable if it has at least one action with real coordinates and
    every referenced screenshot exists on disk."""
    trajectory_file = traj_dir / "trajectory.json"
    if not trajectory_file.exists():
        return False
    try:
        traj = json.loads(trajectory_file.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    steps = traj.get("steps", [])
    if not steps:
        return False
    has_valid_coords = any(
        any(a.get("x", 0) > 0 or a.get("y", 0) > 0 for a in step.get("actions", [])) for step in steps
    )
    screenshots_exist = all((traj_dir / step.get("screenshot", "")).exists() for step in steps)
    return has_valid_coords and screenshots_exist


def _has_valid_trajectories(repo_traj_dir: Path) -> bool:
    return repo_traj_dir.exists() and any(_is_valid_trajectory(d) for d in repo_traj_dir.glob("task_*/"))


def capture_attempts(traj_dir: Path) -> int:
    """How many finalized capture attempts this trajectory has already cost."""
    try:
        return int(json.loads((traj_dir / ATTEMPTS_FILE).read_text()).get("attempts", 0))
    except (OSError, ValueError, TypeError):
        return 0


def record_capture_attempt(traj_dir: Path, outcome: str) -> int:
    """Count one finalized capture attempt against this trajectory's retry bound."""
    # Host-only sidecar: every copy_out overwrites capture_meta.json, so a counter kept there resets.
    attempts = capture_attempts(traj_dir) + 1
    (traj_dir / ATTEMPTS_FILE).write_text(json.dumps({"attempts": attempts, "last_outcome": outcome}, indent=2))
    return attempts


def capture_verdict(traj_dir: Path) -> str | None:
    """A recorded trajectory's settled verdict, or None when the verifier never judged it."""
    try:
        if "verifier" not in json.loads((traj_dir / "capture_meta.json").read_text()):
            return None
        succeeded = json.loads((traj_dir / "trajectory.json").read_text()).get("success")
    except (OSError, ValueError):
        return None
    return "success" if succeeded else "failed"


def is_capture_exhausted(traj_dir: Path) -> bool:
    """A recorded trajectory that failed verification and has no attempts left."""
    return capture_verdict(traj_dir) == "failed" and capture_attempts(traj_dir) >= CAPTURE_MAX_ATTEMPTS


def is_capture_settled(traj_dir: Path) -> bool:
    """Whether task-level resume must leave this recorded trajectory alone."""
    if not _is_valid_trajectory(traj_dir):
        return False
    # A None verdict is an interrupted run's orphan awaiting finalization, not a failure to re-capture.
    return capture_verdict(traj_dir) != "failed" or is_capture_exhausted(traj_dir)


def _safe_task_id(task_id: str) -> str:
    """Mirror of the driver's task-id sanitization, so host code can predict task dir names."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(task_id))[:120]


def _use_discovered(account_index: int, ratio: float) -> bool:
    """Decide whether the Nth `auth=="account"` task should run in `discovered` mode (start
    signed out WITH creds and only sign in if the app forces it — the realistic 'hit the wall,
    then log in' pattern) instead of `pre_session` (start already signed in via the captured
    session). Deterministic so task-level resume picks the same mode on re-run. `ratio` is the
    target fraction of account tasks that exercise the discover-then-login pattern; the rest reuse
    the session — which keeps login from being skewed onto every trajectory."""
    if ratio <= 0:
        return False
    step = max(2, round(1 / ratio))
    return account_index % step == 0


def _load_proposed_tasks(repo_traj_dir: Path) -> list[dict] | None:
    proposals_file = repo_traj_dir / "proposed_tasks.json"
    if not proposals_file.exists():
        return None
    try:
        return json.loads(proposals_file.read_text()).get("tasks") or None
    except (json.JSONDecodeError, OSError):
        return None


def _agent_capture_complete(repo_traj_dir: Path) -> bool:
    """Task-level resume: capture is complete only when EVERY proposed task has a valid
    recording on disk. A repo-level any-valid check forfeited the unrun tail of the task
    list whenever a run died mid-repo. Legacy dirs without a persisted proposal fall back
    to the old any-valid semantics."""
    tasks = _load_proposed_tasks(repo_traj_dir)
    if tasks is None:
        return _has_valid_trajectories(repo_traj_dir)
    return all(_is_valid_trajectory(repo_traj_dir / f"task_{_safe_task_id(t['id'])}") for t in tasks)


async def _capture_suite(rlog, repo: dict, profile, repo_traj_dir: Path, test_name_prefix: str):
    """Run the instrumented test suite in a sandbox and copy results to repo_traj_dir. Raises
    RetryableFailure on clone/copy infra errors. NOT idempotent (re-running re-captures from
    scratch) — generate_for_repo guards this behind a resume check so it is skipped when valid
    captures already exist on disk."""
    repo_name = repo["full_name"]
    sandbox = DockerSandbox(image=profile.sandbox_image, platform=profile.sandbox_platform)
    try:
        await _run(sandbox.start, repo_name)

        _, stderr, exit_code = await _run(sandbox.clone_repo, repo_name, repo.get("default_branch"))
        if exit_code != 0:
            raise RetryableFailure(FailureCategory.CLONE_FAILED, f"clone failed: {stderr[:200]}")

        await _run(setup_repo_instrumentation, sandbox, profile)

        # Start the repo's bundled app (if any) so the suite has something to drive; mirrors stage 5.
        app_spec = await _run(app_runner.detect_app_start, sandbox, profile)
        await _run(app_runner.build_app, sandbox, app_spec)

        # Run the full instrumented test suite directly (no agent — install/wiring/viewport are
        # all handled in code). Whatever the suite executes is captured to /workspace/results;
        # a timeout just yields partial trajectories, which is fine.
        suite_cmd = profile.test_command_suite.format(
            repo_name=repo_name,
            test_name_prefix=test_name_prefix,
            framework=profile.test_framework,
            framework_language=profile.language,
        )
        suite_cmd = app_runner.wrap_test_command(suite_cmd, app_spec)
        out, err, code = await _run(sandbox.exec_command, suite_cmd, SUITE_RUN_TIMEOUT)
        rlog.info(f"suite run exit={code}; {(out or err)[-300:].strip()[:300]}")

        repo_traj_dir.mkdir(parents=True, exist_ok=True)
        try:
            await _run(sandbox.copy_out, "/workspace/results", str(repo_traj_dir))
        except Exception as e:
            raise RetryableFailure(FailureCategory.SANDBOX_ERROR, f"copy results failed: {str(e)[:200]}")
    finally:
        await _run(sandbox.stop)


def _read_app_context(sandbox) -> tuple[str, str]:
    """README + test-source excerpts from the cloned repo, as grounding for the task proposer."""
    readme = ""
    for name in ("README.md", "readme.md", "README.rst", "README"):
        try:
            readme = sandbox.read_file(f"{REPO_DIR}/{name}")
            break
        except Exception:
            continue

    test_hints = []
    seen = 0
    for glob in ("*.spec.ts", "*.test.ts", "*.spec.js", "test_*.py"):
        for path in sandbox.list_files(glob, REPO_DIR):
            if "/node_modules/" in path or seen >= 3:
                continue
            try:
                content = sandbox.read_file(path)
            except Exception:
                continue
            test_hints.append(f"--- {path.replace(REPO_DIR + '/', '')} ---\n{content[:1500]}")
            seen += 1
        if seen >= 3:
            break
    return readme[:4000], "\n\n".join(test_hints)


async def _capture_agent(rlog, registry: RepoRegistry, repo: dict, profile, repo_traj_dir: Path):
    """Agent capture for one repo: boot its app, propose user tasks, run the teacher on each,
    and copy whatever was recorded out to the host — even on usage-limit aborts, so partial
    sessions resume instead of being lost."""
    repo_name = repo["full_name"]
    # Default Playwright image regardless of the repo's own framework: the teacher never runs
    # the repo's tests, and the browser driver needs this image's /ms-playwright browsers.
    sandbox = DockerSandbox(container_timeout=AGENT_CONTAINER_TIMEOUT)
    captured_anything = False
    # Auth-flow capture: bring up the shared mail stack and attach this sandbox to it, so the app's
    # verification/reset mail lands in GreenMail and the agent can read it through Roundcube.
    auth_flow = schemas.AUTH_FLOW_CAPTURE
    mail_stack = await _run(mail_service.ensure_mail_stack) if auth_flow else None
    # App env for auth-flow capture: SMTP pointed at GreenMail + dummy auth secrets (JWT_SECRET,
    # NEXTAUTH_URL, ...) most auth apps need just to boot. Without the latter a Next/Express auth
    # app throws at startup before the agent ever sees it.
    mail_env = {**mail_service.mail_env_for_app(mail_stack), **app_secrets.boot_env_secrets()} if auth_flow else None
    known = app_secrets.known_account(repo_name) if auth_flow else None
    try:
        await _run(sandbox.start, repo_name)
        if auth_flow:
            await _run(mail_service.attach_sandbox, sandbox.client, sandbox.container)

        _, stderr, exit_code = await _run(sandbox.clone_repo, repo_name, repo.get("default_branch"))
        if exit_code != 0:
            raise RetryableFailure(FailureCategory.CLONE_FAILED, f"clone failed: {stderr[:200]}")

        # No blind root `npm install` here: start_app owns the dependency install (it detects the
        # repo's own package manager and installs in the app's dir), exactly as the stage-5 boot
        # gate does. A pre-install would populate node_modules *before* detection runs, and the
        # thousands of vendored package.json files starve app-dir detection (see _find_app_dir),
        # making a repo that booted in stage 5 falsely fail to boot here.
        app_url = await _run(driver_session.start_app, sandbox, profile, mail_env)
        if app_url is None:
            log_tail, _, _ = await _run(sandbox.exec_command, f"tail -15 {driver_session.APP_LOG} 2>/dev/null")
            raise QualityReject(
                FailureCategory.NO_CAPTURE,
                f"app did not boot for agent capture; app log tail: {log_tail[-300:]}",
            )

        await _run(driver_session.install_driver, sandbox)
        await _run(driver_session.start_driver, sandbox)
        client = driver_session.DriverClient(sandbox)

        # Task list: reuse a persisted proposal on resume (no LLM, and the task set stays
        # stable across runs so task-level resume is meaningful); otherwise ground the
        # proposer in the LIVE app and persist what it returns.
        tasks = _load_proposed_tasks(repo_traj_dir)
        if tasks is None:
            probe = await _run(client.session, "context_probe", app_url)
            homepage_snapshot = probe.get("snapshot", "")
            homepage_screenshot = await _run(client.screenshot_b64)
            try:
                affordances = await _run(client.affordances)
            except Exception:
                affordances = {}
            await _run(client.finish, "probe", "")
            await _run(sandbox.exec_command, f"rm -rf {RESULTS_DIR}/task_context_probe")

            readme, test_hints = await _run(_read_app_context, sandbox)
            proposal = await propose_tasks(
                repo,
                homepage_snapshot,
                homepage_screenshot,
                readme,
                test_hints,
                registry=registry,
                affordances=affordances,
                auth_flow=auth_flow,
                inbox=(known is None),
            )
            if proposal is None:
                raise RetryableFailure(FailureCategory.SDK_ERROR, "task proposal failed or unparseable")
            health = proposal.get("app_health") or {}
            if not health.get("usable", True) or not proposal["tasks"]:
                # The app booted but cannot host real tasks (dead backend, error content) —
                # reject BEFORE spending a single teacher call on it.
                raise QualityReject(
                    FailureCategory.NO_CAPTURE,
                    f"proposer judged app unusable: {health.get('notes', 'no tasks proposed')[:200]}",
                )
            tasks = proposal["tasks"]
            repo_traj_dir.mkdir(parents=True, exist_ok=True)
            (repo_traj_dir / "proposed_tasks.json").write_text(json.dumps(proposal, indent=2))
            rlog.info(
                f"proposed {len(tasks)} tasks for {app_url} "
                f"(health: {health.get('notes', 'ok')[:80]}): "
                + "; ".join(t["instruction"][:60] for t in tasks[:4])
                + " ..."
            )
        else:
            rlog.info(f"resuming persisted proposal: {len(tasks)} tasks")

        # Tasks share one app server (state accumulates realistically) but each gets a fresh
        # browser context, so login/session state never leaks between trajectories.
        #
        # Auth-mode distribution (so login is NOT skewed onto every trajectory):
        #   - register/login tasks (the proposer orders these FIRST) sign in / create the account
        #     and, on success, MINT the captured session below.
        #   - "account" tasks (an ordinary action behind login) mostly start ALREADY signed in via
        #     that captured session (auth_mode=pre_session) — no login ritual recorded; a minority
        #     (AUTH_DISCOVERED_RATIO) start signed out with creds and log in only when blocked.
        #   - "none" tasks never touch auth.
        # One app account per run (`scenario_address(repo_name, "account")`): the register task
        # creates it, every later auth task reuses that same address/password.
        captured_session = None
        account_index = 0
        consecutive_fails = 0
        for i, task in enumerate(tasks):
            if _is_valid_trajectory(repo_traj_dir / f"task_{_safe_task_id(task['id'])}"):
                rlog.info(f"task '{task['id']}': already recorded, skipping")
                continue
            if rate_limit_tracker.limit_reached():
                raise UsageLimitReached("usage limit reached mid agent capture")
            if sandbox.remaining_seconds() < AGENT_TASK_TIME_RESERVE:
                rlog.info(
                    f"container time low ({sandbox.remaining_seconds():.0f}s left), "
                    f"stopping after {i} of {len(tasks)} tasks"
                )
                break
            # App-health gate: the shared server can crash mid-session (an unhandled error, OOM,
            # a deploy action that kills the dev server), after which every remaining task fails
            # with CONNECTION_REFUSED. Probe and restart before the task so the list isn't wasted.
            if not driver_session.app_is_up(sandbox, app_url):
                rlog.warning(f"app at {app_url} unresponsive before task '{task['id']}'; restarting")
                restarted = await _run(driver_session.start_app, sandbox, profile, mail_env)
                if restarted is None:
                    rlog.warning(f"app restart failed; stopping after {i} of {len(tasks)} tasks")
                    break
                app_url = restarted
            # Set before the task runs: a mid-task abort (usage limit, SDK crash) must still
            # copy whatever steps the driver already recorded for this task.
            captured_anything = True
            # Per-task isolation: ONE task's failure — a blown per-task budget (the long, hard
            # tasks we most want are the priciest), an SDK error, or a browser crash — must
            # NEVER discard the repo's already-captured trajectories. The repo is judged on
            # what it captured, in post-processing. Only a global usage-limit stop propagates.
            # Per-task auth wiring (see the distribution comment above). The account is either a
            # pre-existing KNOWN login (demo/seeded account — pre-verified, no inbox needed) or a
            # generated disposable address whose verification mail the agent reads in the webmail.
            # Auth tasks type the creds as masked <secret:...>; pre_session tasks instead get the
            # captured storage_state injected and no creds.
            auth = task.get("auth", "none") if auth_flow else "none"
            if known and auth == "register":
                # The account already exists and is verified, and there is no inbox to complete a
                # NEW registration — so a register task can't be done. Skip it rather than relabel
                # (its instruction still says "register", which would mismatch a login and fail).
                rlog.info(f"task '{task['id']}': skipping register task (known account, no inbox)")
                continue
            acct_email = known["email"] if known else mail_service.scenario_address(repo_name, "account")
            acct_password = known["password"] if known else mail_service.scenario_password(repo_name, "account")
            acct_creds = {"email": acct_email, "password": acct_password}
            # A generated account must read its verification mail in the webmail; a known account is
            # already verified, so no inbox is wired (its mail may not even reach GreenMail).
            acct_mail = None if known else {"url": mail_stack.webmail_url, "address": acct_email, "password": "qa"}

            task_secrets = task_mail = task_authcfg = None
            auth_mode = "none"
            capture_session = False
            if auth in ("register", "login"):
                auth_mode = "login"
                capture_session = True
                task_secrets = acct_creds
                task_mail = acct_mail
            elif auth == "account":
                if captured_session is not None and not _use_discovered(account_index, schemas.AUTH_DISCOVERED_RATIO):
                    auth_mode = "pre_session"
                    task_authcfg = {"auth_mode": "pre_session", "storage_state": captured_session}
                else:
                    auth_mode = "discovered"
                    task_secrets = acct_creds
                    task_mail = acct_mail
                account_index += 1
            unproductive = False
            try:
                state = await run_teacher_task(
                    client,
                    task,
                    app_url,
                    registry=registry,
                    full_name=repo_name,
                    auth_mode=auth_mode,
                    secrets=task_secrets,
                    auth_config=task_authcfg,
                    mail_config=task_mail,
                    capture_session=capture_session,
                )
                rlog.info(
                    f"task '{task['id']}': {state['actions']} actions, auth={auth_mode}, "
                    f"outcome={state['agent_outcome'] or 'none'}"
                )
                unproductive = (state.get("agent_outcome") or "") == "failed"
                # A successful register/login mints the session every later pre_session task reuses.
                # Only on a CLAIMED-complete run, so a half-finished login can't inject a logged-out
                # session that makes pre_session tasks start broken.
                if capture_session and state.get("storage_state") and state.get("agent_outcome") == "completed":
                    captured_session = state["storage_state"]
                    rlog.info(f"captured auth session from '{task['id']}' for pre_session reuse")
            except UsageLimitReached:
                raise
            except Exception as e:
                rlog.warning(
                    f"task '{task['id']}' aborted ({type(e).__name__}: {str(e)[:120]}); "
                    f"keeping prior captures and continuing"
                )
                unproductive = True
            consecutive_fails = consecutive_fails + 1 if unproductive else 0
            # Incremental copy-out: teacher turns can be slow under rate-limit waits, and a
            # container-timeout mid-task must never cost the tasks already recorded.
            repo_traj_dir.mkdir(parents=True, exist_ok=True)
            try:
                await _run(sandbox.copy_out, RESULTS_DIR, str(repo_traj_dir))
            except Exception as e:
                logger.warning(f"incremental copy_out failed for {repo_name}: {e}")
            if consecutive_fails >= AGENT_MAX_CONSECUTIVE_FAILS:
                rlog.warning(
                    f"early-abort: {consecutive_fails} consecutive unproductive tasks "
                    f"(stopped after {i + 1} of {len(tasks)}) — likely a dead backend or auth wall"
                )
                break
    finally:
        if captured_anything:
            repo_traj_dir.mkdir(parents=True, exist_ok=True)
            try:
                await _run(sandbox.copy_out, RESULTS_DIR, str(repo_traj_dir))
            except Exception as e:
                logger.warning(f"final copy_out after agent capture failed for {repo_name}: {e}")
        await _run(sandbox.stop)


def _excluded(traj_dir: Path, reason: str):
    """Mark a trajectory as not-for-training (success=false) with the reason in capture_meta."""
    trajectory_file = traj_dir / "trajectory.json"
    traj = json.loads(trajectory_file.read_text())
    traj["success"] = False
    trajectory_file.write_text(json.dumps(traj, indent=2))
    meta_file = traj_dir / "capture_meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    meta["excluded_reason"] = reason
    meta_file.write_text(json.dumps(meta, indent=2))
    # convert/audit gate on metadata.json, so the exclusion must land there too when it exists.
    metadata_file = traj_dir / "metadata.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        metadata["excluded_reason"] = reason
        metadata_file.write_text(json.dumps(metadata, indent=2))


async def _postprocess_agent_capture(rlog, registry, repo, repo_traj_dir: Path) -> tuple[list[Path], int]:
    """Verify every captured trajectory (idempotent: already-verified dirs are skipped), then
    apply the per-repo diversity cap. Returns (valid_dirs_for_metadata, verified_count)."""
    trajectory_dirs = sorted(repo_traj_dir.glob("task_*/")) if repo_traj_dir.exists() else []
    valid_dirs = []
    for traj_dir in trajectory_dirs:
        if not _is_valid_trajectory(traj_dir):
            continue
        valid_dirs.append(traj_dir)
        capture_meta = _load_capture_meta(traj_dir)
        if "verifier" in capture_meta:
            continue
        verdict = await verify_trajectory(traj_dir, registry=registry, full_name=repo["full_name"])
        rlog.info(
            f"verify {traj_dir.name}: success={verdict['final_success']}"
            f"{' (relabeled)' if verdict.get('relabeled') else ''} — {verdict.get('reasoning', '')[:100]}"
        )

    # Dedup + cap among verified-successful trajectories: identical action-shape duplicates are
    # excluded, then the longest AGENT_MAX_KEPT survive (length is the scarce resource).
    verified = []
    for traj_dir in valid_dirs:
        traj = json.loads((traj_dir / "trajectory.json").read_text())
        if traj.get("success"):
            verified.append((traj_dir, traj))
    verified.sort(key=lambda pair: len(pair[1].get("steps", [])), reverse=True)

    seen_hashes = set()
    kept = 0
    for traj_dir, traj in verified:
        h = _dedup_hash(traj.get("steps", []))
        if h in seen_hashes:
            _excluded(traj_dir, "duplicate action sequence")
            continue
        if kept >= AGENT_MAX_KEPT:
            _excluded(traj_dir, "per_repo_cap")
            continue
        seen_hashes.add(h)
        kept += 1

    return valid_dirs, kept


async def generate_for_repo(
    registry: RepoRegistry,
    repo: dict,
    trajectory_base_dir: Path,
    capture_mode: str = None,
):
    repo_name = repo["full_name"]

    if repo.get("current_stage", 0) >= 6:
        return

    capture_mode = capture_mode or schemas.CAPTURE_MODE
    rlog = repo_logger(6, repo_name)

    deep_analysis = {}
    if repo.get("deep_analysis"):
        try:
            deep_analysis = (
                json.loads(repo["deep_analysis"]) if isinstance(repo["deep_analysis"], str) else repo["deep_analysis"]
            )
        except (json.JSONDecodeError, TypeError):
            pass

    profile = resolve_profile(deep_analysis.get("test_framework"), deep_analysis.get("framework_language"))
    test_name_prefix = repo_name.replace("/", "_")
    repo_traj_dir = trajectory_base_dir / f"repo_{test_name_prefix}"

    # Resume: a prior run may have captured some-but-not-all proposed tasks before dying.
    # Agent mode resumes at TASK granularity (_capture_agent skips tasks already on disk and
    # reuses the persisted proposal); capture is skipped entirely only when every proposed
    # task has a valid recording. Replay keeps the repo-level any-valid check.
    if capture_mode == "agent":
        if _agent_capture_complete(repo_traj_dir):
            rlog.info(f"resuming: all proposed tasks already recorded ({repo_traj_dir.name}), skipping capture")
        else:
            rlog.info("agent capture: booting app + running teacher tasks")
            await _capture_agent(rlog, registry, repo, profile, repo_traj_dir)
    elif _has_valid_trajectories(repo_traj_dir):
        rlog.info(f"resuming: valid trajectories already on disk ({repo_traj_dir.name}), skipping capture")
    else:
        rlog.info("replay capture: running instrumented test suite")
        await _capture_suite(rlog, repo, profile, repo_traj_dir, test_name_prefix)

    # ---- Post-capture processing. Idempotent, so it runs identically on a fresh capture or a
    # resumed one. Agent mode: verify + dedup/cap (instructions came from the task proposer).
    # Replay mode: generate instructions where missing. ----
    if capture_mode == "agent":
        valid_dirs, valid_count = await _postprocess_agent_capture(rlog, registry, repo, repo_traj_dir)
        instruction_generated = {d: True for d in valid_dirs}
        thought_source = "teacher_raw"
        rlog.info(f"{valid_count} verified trajectories from {len(valid_dirs)} captured")
        if valid_count == 0:
            raise QualityReject(FailureCategory.NO_CAPTURE, "no verified trajectories produced")
    else:
        trajectory_dirs = list(repo_traj_dir.glob("task_*/")) if repo_traj_dir.exists() else []
        valid_dirs = []
        instruction_generated = {}
        thought_source = "generated"

        for traj_dir in trajectory_dirs:
            trajectory_file = traj_dir / "trajectory.json"
            if not _is_valid_trajectory(traj_dir):
                continue
            with open(trajectory_file) as f:
                traj = json.load(f)
            valid_dirs.append(traj_dir)

            if not traj.get("instruction"):
                test_source = _load_test_source(traj_dir)
                instruction = await generate_instruction(
                    traj,
                    _load_capture_meta(traj_dir),
                    repo=repo,
                    test_source=test_source,
                    registry=registry,
                )
                traj["instruction"] = instruction
                instruction_generated[traj_dir] = True
                with open(trajectory_file, "w") as f:
                    json.dump(traj, f, indent=2)
            else:
                instruction_generated[traj_dir] = False

        valid_count = len(valid_dirs)
        rlog.info(f"{valid_count} valid trajectories from {len(trajectory_dirs)} total")
        if valid_count == 0:
            raise QualityReject(FailureCategory.NO_CAPTURE, "no valid trajectories produced")

    # Think blocks are NOT generated here — that is the separate, resumable stage 7 (think). Keeping
    # the per-step think generation out of capture means a think-gen stall can never block or fail a
    # capture: trajectories + instructions + metadata land, and the repo is marked `captured`.
    for traj_dir in valid_dirs:
        await _enrich_metadata(
            traj_dir,
            repo,
            profile,
            instruction_generated.get(traj_dir, True),
            registry=registry,
            thought_source=thought_source,
        )

    registry.update_status(
        repo_name,
        RepoStatus.CAPTURED.value,
        stage=6,
        trajectory_count=valid_count,
        trajectory_dir=str(repo_traj_dir),
    )
    return valid_count


def _element_label(step: dict, step_meta: dict) -> str:
    """A short 'role "name"' label for a step's target element, when known."""
    role = step.get("role") or step_meta.get("role")
    name = step.get("name") or step_meta.get("name")
    if role and name:
        return f'{role} "{name[:40]}"'
    if name:
        return f'"{name[:40]}"'
    return role or ""


async def generate_instruction(
    trajectory: dict,
    capture_meta: dict | None = None,
    repo: dict | None = None,
    test_source: str = "",
    registry=None,
) -> str:
    """Generate an intent-level instruction for one trajectory.

    Inputs in order of signal strength: the test function NAME (`task_id`), the
    `test_source` (function body + docstring), the `repo` name+description, and finally
    the recorded action sequence (used only when the upstream signals are uninformative).
    The prompt is explicit that action enumeration is a failure mode; whatever Sonnet
    returns is what lands on `trajectory.instruction` — no regex post-filters.

    Falls back to a `task_id`-derived placeholder ONLY on genuine SDK failure (timeout,
    budget exhaustion). The fallback is a safety net for infrastructure, not for
    second-guessing the model's content.
    """
    steps = trajectory.get("steps", [])
    test_name = trajectory.get("task_id", "unknown")
    if not steps:
        return f"Verify the {test_name.replace('_', ' ')} behavior."

    steps_meta = (capture_meta or {}).get("steps", {})
    action_summary = []
    for step in steps[:10]:
        label = _element_label(step, steps_meta.get(str(step.get("step")), {}))
        for action in step.get("actions", []):
            action_type = action.get("type", "")
            if action_type == "click":
                action_summary.append(f"click {label}" if label else "click")
            elif action_type == "type":
                target = f" into {label}" if label else ""
                action_summary.append(f'type "{action.get("text", "")[:30]}"{target}')
            elif action_type == "keypress":
                keys = action.get("keys", []) or []
                action_summary.append(f"press {'+'.join(keys)}" if keys else "press a key")
            else:
                action_summary.append(f"{action_type} {label}".strip())

    action_str = ", ".join(action_summary[:10]) or "(no actions recorded)"

    repo = repo or {}
    repo_name = repo.get("full_name") or repo.get("name") or "(repository not identified)"
    repo_description = (repo.get("description") or "").strip() or "(no description)"
    test_source_block = (
        test_source.strip()
        if test_source and test_source.strip()
        else "(test source unavailable — rely on the test name and actions)"
    )

    prompt = INSTRUCTION_PROMPT.format(
        repo_name=repo_name,
        repo_description=repo_description,
        test_name=test_name,
        test_source=test_source_block,
        action_summary=action_str,
    )

    try:
        result_text = await run_agent(
            prompt,
            ClaudeAgentOptions(
                model=MODEL_FOR_GROUNDED_GENERATION,
                max_turns=1,
                max_budget_usd=INSTRUCTION_BUDGET_USD,
                permission_mode="bypassPermissions",
            ),
            registry=registry,
            full_name=(repo or {}).get("full_name", ""),
            stage=6,
            purpose="instruction",
            raise_on_error=False,
        )
    except UsageLimitReached:
        raise
    except Exception as e:
        logger.warning(f"Instruction generation failed: {e}")
        return f"Verify the {test_name.replace('_', ' ')} behavior."

    return result_text.strip() or f"Verify the {test_name.replace('_', ' ')} behavior."


def _load_capture_meta(traj_dir: Path) -> dict:
    p = traj_dir / "capture_meta.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_test_source(traj_dir: Path, max_chars: int = 2000) -> str:
    """Read the cached test function source (written by the instrumentation conftest at
    recording time) sitting next to trajectory.json. Returns "" for older trajectories
    that pre-date the capture; the instruction prompt has an explicit fallback path for
    that case so a missing source is not fatal — just lower signal."""
    p = traj_dir / "test_source.txt"
    if not p.exists():
        return ""
    try:
        text = p.read_text()
    except Exception:
        return ""
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""


def _length_bucket(n: int) -> str:
    if n <= 5:
        return "short"
    if n <= 15:
        return "medium"
    return "long"


def _dedup_hash(steps: list) -> str:
    parts = []
    for step in steps:
        role = step.get("role") or ""
        for a in step.get("actions", []):
            parts.append(f"{a.get('type', '')}:{role}")
    return "sha1:" + hashlib.sha1(";".join(parts).encode()).hexdigest()


def _aggregate(steps: list, steps_meta: dict):
    action_hist, role_hist = {}, {}
    n_nav = bbox_steps = n_console = 0
    for step in steps:
        for a in step.get("actions", []):
            t = a.get("type", "")
            action_hist[t] = action_hist.get(t, 0) + 1
        sm = steps_meta.get(str(step.get("step")), {})
        role = step.get("role") or sm.get("role")
        if role:
            role_hist[role] = role_hist.get(role, 0) + 1
        if step.get("bbox") or sm.get("bbox"):
            bbox_steps += 1
        if sm.get("caused_navigation"):
            n_nav += 1
        n_console += sm.get("console_errors", 0) or 0
    total = len(steps)
    valid_bbox_ratio = round(bbox_steps / total, 3) if total else 0.0
    return action_hist, role_hist, n_nav, valid_bbox_ratio, n_console


async def classify_task_type(trajectory: dict, capture_meta: dict, registry=None, full_name: str = "") -> str:
    steps = trajectory.get("steps", [])
    steps_meta = (capture_meta or {}).get("steps", {})
    action_types, elements = [], []
    for step in steps[:20]:
        for a in step.get("actions", []):
            action_types.append(a.get("type", ""))
        label = _element_label(step, steps_meta.get(str(step.get("step")), {}))
        if label:
            elements.append(label)

    prompt = TASK_TYPE_PROMPT.format(
        instruction=trajectory.get("instruction", "")[:300],
        action_types=", ".join(action_types[:30]) or "(none)",
        elements="; ".join(elements[:15]) or "(none)",
    )

    try:
        result_text = await run_agent(
            prompt,
            ClaudeAgentOptions(
                model=MODEL_FOR_CLASSIFICATION,
                max_turns=1,
                max_budget_usd=TASK_TYPE_BUDGET_USD,
                permission_mode="bypassPermissions",
            ),
            registry=registry,
            full_name=full_name,
            stage=6,
            purpose="classify",
            raise_on_error=False,
        )
    except UsageLimitReached:
        raise
    except Exception as e:
        logger.warning(f"Task-type classification failed: {e}")
        return "other"

    label = result_text.strip().lower().replace(" ", "-").strip(".")
    for t in TASK_TYPES:
        if t.lower() in label:
            return t
    return "other"


async def _enrich_metadata(
    traj_dir: Path,
    repo: dict,
    profile,
    instruction_was_generated: bool,
    registry=None,
    thought_source: str = "generated",
):
    """Fill the trajectory-level aggregate / provenance fields in metadata.json from the
    captured trajectory + capture_meta. Best-effort: never raise into the generation flow."""
    try:
        traj_file = traj_dir / "trajectory.json"
        meta_file = traj_dir / "metadata.json"
        if not traj_file.exists():
            return

        with open(traj_file) as f:
            traj = json.load(f)
        capture_meta = _load_capture_meta(traj_dir)

        metadata = {}
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}

        steps = traj.get("steps", [])
        steps_meta = capture_meta.get("steps", {})
        action_hist, role_hist, n_nav, valid_bbox_ratio, n_console = _aggregate(steps, steps_meta)

        metadata["instruction"] = traj.get("instruction", metadata.get("instruction", ""))
        metadata["total_steps"] = len(steps)
        metadata["domain"] = repo.get("domain") or RepoRegistry.classify_domain(
            repo.get("description"), repo.get("topics")
        )
        metadata["website"] = _host(traj.get("start_url", ""))
        metadata["framework"] = profile.test_framework
        metadata["framework_language"] = profile.language
        metadata["source_query"] = repo.get("source_query") or ""
        metadata["repo_stars"] = repo.get("stars", 0) or 0
        metadata["action_type_histogram"] = action_hist
        metadata["element_role_histogram"] = role_hist
        metadata["length_bucket"] = _length_bucket(len(steps))
        metadata["n_navigations"] = n_nav
        metadata["valid_bbox_ratio"] = valid_bbox_ratio
        metadata["n_console_errors"] = n_console
        metadata["instruction_source"] = "generated" if instruction_was_generated else "human"
        metadata["thought_source"] = thought_source
        metadata["capture_source"] = "agent" if thought_source.startswith("teacher") else "replay"
        verifier = capture_meta.get("verifier") or {}
        if verifier:
            metadata["efficiency"] = verifier.get("efficiency")
            metadata["efficiency_ratio"] = verifier.get("efficiency_ratio")
            metadata["relabeled"] = verifier.get("relabeled", False)
        metadata["dedup_hash"] = _dedup_hash(steps)
        metadata["task_type"] = await classify_task_type(
            traj, capture_meta, registry=registry, full_name=repo.get("full_name", "")
        )

        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)
    except UsageLimitReached:
        raise
    except Exception as e:
        logger.warning(f"Metadata enrichment failed for {traj_dir.name}: {e}")


async def run_generate(
    registry: RepoRegistry,
    trajectory_base_dir: Path,
    limit: int = None,
    workers: int = 2,
    budget_remaining: float = None,
    capture_mode: str = None,
):
    capture_mode = capture_mode or schemas.CAPTURE_MODE
    repos = registry.get_repos_for_stage(6, limit=limit)
    logger.info(f"Generating trajectories for {len(repos)} repos with {workers} workers (mode: {capture_mode})")

    semaphore = asyncio.Semaphore(workers)
    progress = StageProgress(6, len(repos))

    async def process_with_semaphore(repo):
        if rate_limit_tracker.limit_reached():
            return

        async def core(reg, r):
            return await generate_for_repo(reg, r, trajectory_base_dir, capture_mode=capture_mode)

        async with semaphore:
            try:
                await run_repo(6, repo, registry, core, progress)
            except UsageLimitReached:
                return

    await asyncio.gather(*[process_with_semaphore(repo) for repo in repos])

    return progress.summary()
