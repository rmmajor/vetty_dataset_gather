import asyncio
import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

from claude_agent_sdk import ClaudeAgentOptions

from cua.pipeline import app_runner, app_secrets, driver_session, mail_service, schemas
from cua.pipeline._processing import run_repo
from cua.pipeline.agent_runner import run_agent
from cua.pipeline.driver_session import boot_driver, run_blocking
from cua.pipeline.frameworks import resolve_profile
from cua.pipeline.instrumentation import REPO_DIR, RESULTS_DIR, setup_repo_instrumentation
from cua.pipeline.lanes.capture_lane import copy_out_site, is_valid_trajectory
from cua.pipeline.llm_config import MODEL_FOR_CLASSIFICATION, MODEL_FOR_GROUNDED_GENERATION
from cua.pipeline.logging_utils import StageProgress, repo_logger
from cua.pipeline.registry import RepoRegistry
from cua.pipeline.sandbox import DockerSandbox
from cua.pipeline.schemas import FailureCategory, QualityReject, RepoStatus, RetryableFailure
from cua.pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from cua.pipeline.sites.registry import load_proposed_tasks
from cua.pipeline.task_proposer import propose_tasks
from cua.pipeline.teacher import run_teacher_task, verify_trajectory
from cua.settings import CAPTURE, PIPELINE
from cua.trajectory.io import CAPTURE_META_FILE, load_json, write_json
from cua.trajectory.schema import metadata_from, safe_task_id

logger = logging.getLogger(__name__)

SUITE_RUN_TIMEOUT = 480  # seconds; partial capture on timeout is fine for trajectory mining

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


def _has_valid_trajectories(repo_traj_dir: Path) -> bool:
    return repo_traj_dir.exists() and any(is_valid_trajectory(d) for d in repo_traj_dir.glob("task_*/"))


def _use_discovered(account_index: int, ratio: float) -> bool:
    """Whether the Nth account task starts signed OUT with credentials rather than on the captured session."""
    # Deterministic, so a task-level resume picks the same mode and login stays a minority of the corpus.
    if ratio <= 0:
        return False
    step = max(2, round(1 / ratio))
    return account_index % step == 0


def _teacher_capture_complete(repo_traj_dir: Path) -> bool:
    """Capture is complete only when EVERY proposed task has a valid recording on disk."""
    tasks = load_proposed_tasks(repo_traj_dir)
    # A repo-level any-valid check forfeited the unrun tail of the task list whenever a run died mid-repo.
    if tasks is None:
        return _has_valid_trajectories(repo_traj_dir)
    return all(is_valid_trajectory(repo_traj_dir / f"task_{safe_task_id(t['id'])}") for t in tasks)


async def _capture_suite(rlog, repo: dict, profile, repo_traj_dir: Path, test_name_prefix: str):
    """Run the instrumented test suite in a sandbox and copy results to repo_traj_dir."""
    # NOT idempotent: re-running re-captures from scratch, so generate_for_repo guards it behind a resume check.
    repo_name = repo["full_name"]
    sandbox = DockerSandbox(image=profile.sandbox_image, platform=profile.sandbox_platform)
    try:
        await run_blocking(sandbox.start, repo_name)

        _, stderr, exit_code = await run_blocking(sandbox.clone_repo, repo_name, repo.get("default_branch"))
        if exit_code != 0:
            raise RetryableFailure(FailureCategory.CLONE_FAILED, f"clone failed: {stderr[:200]}")

        await run_blocking(setup_repo_instrumentation, sandbox, profile)

        # Start the repo's bundled app (if any) so the suite has something to drive; mirrors stage 5.
        app_spec = await run_blocking(app_runner.detect_app_start, sandbox, profile)
        await run_blocking(app_runner.build_app, sandbox, app_spec)

        # No agent here: install/wiring/viewport are handled in code, and a timeout just yields partial trajectories.
        suite_cmd = profile.test_command_suite.format(
            repo_name=repo_name,
            test_name_prefix=test_name_prefix,
            framework=profile.test_framework,
            framework_language=profile.language,
        )
        suite_cmd = app_runner.wrap_test_command(suite_cmd, app_spec)
        out, err, code = await run_blocking(sandbox.exec_command, suite_cmd, SUITE_RUN_TIMEOUT)
        rlog.info(f"suite run exit={code}; {(out or err)[-300:].strip()[:300]}")

        repo_traj_dir.mkdir(parents=True, exist_ok=True)
        try:
            await run_blocking(sandbox.copy_out, "/workspace/results", str(repo_traj_dir))
        except Exception as e:
            raise RetryableFailure(FailureCategory.SANDBOX_ERROR, f"copy results failed: {str(e)[:200]}") from e
    finally:
        await run_blocking(sandbox.stop)


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


async def _boot_repo_app(sandbox, repo: dict, profile, mail_env: dict | None) -> str:
    """Clone the repo into the sandbox, start its app and return the URL the teacher will drive."""
    _, stderr, exit_code = await run_blocking(sandbox.clone_repo, repo["full_name"], repo.get("default_branch"))
    if exit_code != 0:
        raise RetryableFailure(FailureCategory.CLONE_FAILED, f"clone failed: {stderr[:200]}")
    # No blind root `npm install`: a pre-install buries _find_app_dir under vendored package.json files.
    app_url = await run_blocking(driver_session.start_app, sandbox, profile, mail_env)
    if app_url is None:
        log_tail, _, _ = await run_blocking(sandbox.exec_command, f"tail -15 {driver_session.APP_LOG} 2>/dev/null")
        raise QualityReject(
            FailureCategory.NO_CAPTURE,
            f"app did not boot for agent capture; app log tail: {log_tail[-300:]}",
        )
    return app_url


async def _propose(rlog, registry, sandbox, client, repo: dict, app_url: str, repo_traj_dir: Path, mail: dict) -> list:
    """This repo's task list: a persisted proposal on resume, else the proposer grounded in the LIVE app."""
    # A persisted proposal costs no LLM call and keeps the task set stable, which is what resume needs.
    tasks = load_proposed_tasks(repo_traj_dir)
    if tasks is not None:
        rlog.info(f"resuming persisted proposal: {len(tasks)} tasks")
        return tasks

    probe = await run_blocking(client.session, "context_probe", app_url)
    homepage_screenshot = await run_blocking(client.screenshot_b64)
    try:
        affordances = await run_blocking(client.affordances)
    except Exception:
        affordances = {}
    await run_blocking(client.finish, "probe", "")
    await run_blocking(sandbox.exec_command, f"rm -rf {RESULTS_DIR}/task_context_probe")

    readme, test_hints = await run_blocking(_read_app_context, sandbox)
    proposal = await propose_tasks(
        repo,
        probe.get("snapshot", ""),
        homepage_screenshot,
        readme,
        test_hints,
        registry=registry,
        affordances=affordances,
        auth_flow=mail["auth_flow"],
        inbox=(mail["known"] is None),
    )
    if proposal is None:
        raise RetryableFailure(FailureCategory.SDK_ERROR, "task proposal failed or unparseable")
    health = proposal.get("app_health") or {}
    if not health.get("usable", True) or not proposal["tasks"]:
        # The app booted but cannot host real tasks, so reject it before a single teacher call is spent.
        raise QualityReject(
            FailureCategory.NO_CAPTURE,
            f"proposer judged app unusable: {health.get('notes', 'no tasks proposed')[:200]}",
        )
    repo_traj_dir.mkdir(parents=True, exist_ok=True)
    (repo_traj_dir / "proposed_tasks.json").write_text(json.dumps(proposal, indent=2))
    rlog.info(
        f"proposed {len(proposal['tasks'])} tasks for {app_url} "
        f"(health: {health.get('notes', 'ok')[:80]}): "
        + "; ".join(t["instruction"][:60] for t in proposal["tasks"][:4])
        + " ..."
    )
    return proposal["tasks"]


def _task_auth(task: dict, mail: dict, captured_session, account_index: int) -> dict | None:
    """The teacher's auth kwargs for one task, or None when a known account makes the task impossible."""
    auth = task.get("auth", "none") if mail["auth_flow"] else "none"
    known, repo_name = mail["known"], mail["repo_name"]
    # A known account is already registered and has no inbox, so a register task can neither run nor be relabeled.
    if known and auth == "register":
        return None
    acct_email = known["email"] if known else mail_service.scenario_address(repo_name, "account")
    acct_password = known["password"] if known else mail_service.scenario_password(repo_name, "account")
    acct_creds = {"email": acct_email, "password": acct_password}
    # Only a generated account needs the webmail: a known account is pre-verified and may never mail GreenMail.
    acct_mail = None if known else {"url": mail["webmail_url"], "address": acct_email, "password": "qa"}
    wiring = {"auth_mode": "none", "secrets": None, "mail_config": None, "auth_config": None, "capture_session": False}
    if auth in ("register", "login"):
        wiring.update(auth_mode="login", capture_session=True, secrets=acct_creds, mail_config=acct_mail)
    elif auth == "account":
        if captured_session is not None and not _use_discovered(account_index, PIPELINE.auth_discovered_ratio):
            session_config = {"auth_mode": "pre_session", "storage_state": captured_session}
            wiring.update(auth_mode="pre_session", auth_config=session_config)
        else:
            wiring.update(auth_mode="discovered", secrets=acct_creds, mail_config=acct_mail)
    wiring["uses_account"] = auth == "account"
    return wiring


async def _revive_app(rlog, sandbox, app_url: str, profile, mail_env: dict | None, task_id: str) -> str | None:
    """The URL to drive this task, restarting the shared dev server first when it stopped answering."""
    # The server can die mid-session, after which every remaining task fails on CONNECTION_REFUSED.
    if driver_session.app_is_up(sandbox, app_url):
        return app_url
    rlog.warning(f"app at {app_url} unresponsive before task '{task_id}'; restarting")
    return await run_blocking(driver_session.start_app, sandbox, profile, mail_env)


async def _run_task(rlog, client, task: dict, app_url: str, repo_name: str, registry, wiring: dict):
    """Run the teacher on one task, returning (state, unproductive); only a usage-limit stop propagates."""
    try:
        state = await run_teacher_task(
            client,
            task,
            app_url,
            registry=registry,
            full_name=repo_name,
            auth_mode=wiring["auth_mode"],
            secrets=wiring["secrets"],
            auth_config=wiring["auth_config"],
            mail_config=wiring["mail_config"],
            capture_session=wiring["capture_session"],
        )
    except UsageLimitReached:
        raise
    except Exception as e:
        # One task's blown budget, SDK error or browser crash must never discard the repo's other captures.
        rlog.warning(
            f"task '{task['id']}' aborted ({type(e).__name__}: {str(e)[:120]}); keeping prior captures and continuing"
        )
        return None, True
    rlog.info(
        f"task '{task['id']}': {state['actions']} actions, auth={wiring['auth_mode']}, "
        f"outcome={state['agent_outcome'] or 'none'}"
    )
    return state, (state.get("agent_outcome") or "") == "failed"


async def _copy_out_repo(sandbox, repo_traj_dir: Path, repo_name: str, note: str) -> None:
    """Best-effort copy-out: a failed copy must never cost a capture the recordings already on disk."""
    try:
        await copy_out_site(sandbox, RESULTS_DIR, repo_traj_dir)
    except Exception as e:
        logger.warning(f"{note} failed for {repo_name}: {e}")


async def _mail_wiring(repo_name: str) -> tuple[dict, dict | None]:
    """The repo's auth-flow context and the app env that reaches it, bringing the shared mail stack up when it is on."""
    auth_flow = PIPELINE.auth_flow_capture
    context = {"auth_flow": auth_flow, "repo_name": repo_name, "known": None, "webmail_url": ""}
    if not auth_flow:
        return context, None
    context["known"] = app_secrets.known_account(repo_name)
    stack = await run_blocking(mail_service.ensure_mail_stack)
    context["webmail_url"] = stack.webmail_url
    # Dummy auth secrets ride along with the SMTP env: most auth apps throw at startup without them.
    return context, {**mail_service.mail_env_for_app(stack), **app_secrets.boot_env_secrets()}


async def _capture_agent(rlog, registry: RepoRegistry, repo: dict, profile, repo_traj_dir: Path):
    """Agent capture for one repo: boot its app, propose user tasks, run the teacher on each, copy out."""
    repo_name = repo["full_name"]
    # Always the default Playwright image: the driver needs the /ms-playwright browsers only it ships.
    sandbox = DockerSandbox(container_timeout=CAPTURE.container_timeout)
    captured_anything = False
    mail, mail_env = await _mail_wiring(repo_name)
    try:
        await run_blocking(sandbox.start, repo_name)
        if mail["auth_flow"]:
            await run_blocking(mail_service.attach_sandbox, sandbox.client, sandbox.container)
        app_url = await _boot_repo_app(sandbox, repo, profile, mail_env)
        client = await boot_driver(sandbox)
        tasks = await _propose(rlog, registry, sandbox, client, repo, app_url, repo_traj_dir, mail)

        # Tasks share one app server so state accumulates realistically, but each gets a fresh browser context.
        captured_session = None
        account_index = 0
        consecutive_fails = 0
        for i, task in enumerate(tasks):
            if is_valid_trajectory(repo_traj_dir / f"task_{safe_task_id(task['id'])}"):
                rlog.info(f"task '{task['id']}': already recorded, skipping")
                continue
            if rate_limit_tracker.limit_reached():
                raise UsageLimitReached("usage limit reached mid agent capture")
            if sandbox.remaining_seconds() < CAPTURE.task_time_reserve:
                rlog.info(
                    f"container time low ({sandbox.remaining_seconds():.0f}s left), "
                    f"stopping after {i} of {len(tasks)} tasks"
                )
                break
            app_url = await _revive_app(rlog, sandbox, app_url, profile, mail_env, task["id"])
            if app_url is None:
                rlog.warning(f"app restart failed; stopping after {i} of {len(tasks)} tasks")
                break
            # Set before the task runs: a mid-task abort must still copy the steps the driver already recorded.
            captured_anything = True
            wiring = _task_auth(task, mail, captured_session, account_index)
            if wiring is None:
                rlog.info(f"task '{task['id']}': skipping register task (known account, no inbox)")
                continue
            account_index += 1 if wiring.pop("uses_account") else 0
            state, unproductive = await _run_task(rlog, client, task, app_url, repo_name, registry, wiring)
            minted = (state or {}).get("storage_state") and state.get("agent_outcome") == "completed"
            # Only a CLAIMED-complete login mints the session, so a half-finished one cannot break later tasks.
            if wiring["capture_session"] and minted:
                captured_session = state["storage_state"]
                rlog.info(f"captured auth session from '{task['id']}' for pre_session reuse")
            consecutive_fails = consecutive_fails + 1 if unproductive else 0
            await _copy_out_repo(sandbox, repo_traj_dir, repo_name, "incremental copy_out")
            if consecutive_fails >= CAPTURE.max_consecutive_fails:
                rlog.warning(
                    f"early-abort: {consecutive_fails} consecutive unproductive tasks "
                    f"(stopped after {i + 1} of {len(tasks)}) — likely a dead backend or auth wall"
                )
                break
    finally:
        if captured_anything:
            await _copy_out_repo(sandbox, repo_traj_dir, repo_name, "final copy_out after agent capture")
        await run_blocking(sandbox.stop)


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


async def _postprocess_teacher_capture(rlog, registry, repo, repo_traj_dir: Path) -> tuple[list[Path], int]:
    """Verify every captured trajectory then apply the per-repo cap; returns (valid_dirs, verified_count)."""
    # Idempotent: a dir that already carries a verifier verdict is skipped, so a resumed run re-judges nothing.
    trajectory_dirs = sorted(repo_traj_dir.glob("task_*/")) if repo_traj_dir.exists() else []
    valid_dirs = []
    for traj_dir in trajectory_dirs:
        if not is_valid_trajectory(traj_dir):
            continue
        valid_dirs.append(traj_dir)
        capture_meta = load_json(traj_dir / CAPTURE_META_FILE)
        if "verifier" in capture_meta:
            continue
        verdict = await verify_trajectory(traj_dir, registry=registry, full_name=repo["full_name"])
        rlog.info(
            f"verify {traj_dir.name}: success={verdict['final_success']}"
            f"{' (relabeled)' if verdict.get('relabeled') else ''} — {verdict.get('reasoning', '')[:100]}"
        )

    # Identical action shapes are excluded, then the longest survive: trajectory length is the scarce resource.
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
        if kept >= CAPTURE.max_kept_per_repo:
            _excluded(traj_dir, "per_repo_cap")
            continue
        seen_hashes.add(h)
        kept += 1

    return valid_dirs, kept


async def _capture_for_repo(rlog, registry, repo: dict, profile, repo_traj_dir: Path, capture_mode: str, prefix: str):
    """Capture this repo in the configured mode, or skip when a prior run already recorded everything."""
    # Agent mode resumes at TASK granularity, reusing the persisted proposal; replay mode resumes per repo.
    if capture_mode == "agent":
        if _teacher_capture_complete(repo_traj_dir):
            rlog.info(f"resuming: all proposed tasks already recorded ({repo_traj_dir.name}), skipping capture")
            return
        rlog.info("agent capture: booting app + running teacher tasks")
        await _capture_agent(rlog, registry, repo, profile, repo_traj_dir)
    elif _has_valid_trajectories(repo_traj_dir):
        rlog.info(f"resuming: valid trajectories already on disk ({repo_traj_dir.name}), skipping capture")
    else:
        rlog.info("replay capture: running instrumented test suite")
        await _capture_suite(rlog, repo, profile, repo_traj_dir, prefix)


async def _instructions_for_replay(rlog, registry, repo: dict, repo_traj_dir: Path) -> tuple[list, dict]:
    """Fill in a generated instruction for every replay recording that has none, returning (valid_dirs, generated)."""
    trajectory_dirs = list(repo_traj_dir.glob("task_*/")) if repo_traj_dir.exists() else []
    valid_dirs: list = []
    instruction_generated: dict = {}
    for traj_dir in trajectory_dirs:
        trajectory_file = traj_dir / "trajectory.json"
        if not is_valid_trajectory(traj_dir):
            continue
        with open(trajectory_file) as f:
            traj = json.load(f)
        valid_dirs.append(traj_dir)
        instruction_generated[traj_dir] = not traj.get("instruction")
        if not traj.get("instruction"):
            traj["instruction"] = await generate_instruction(
                traj,
                load_json(traj_dir / CAPTURE_META_FILE),
                repo=repo,
                test_source=load_test_source(traj_dir),
                registry=registry,
            )
            with open(trajectory_file, "w") as f:
                json.dump(traj, f, indent=2)
    rlog.info(f"{len(valid_dirs)} valid trajectories from {len(trajectory_dirs)} total")
    if not valid_dirs:
        raise QualityReject(FailureCategory.NO_CAPTURE, "no valid trajectories produced")
    return valid_dirs, instruction_generated


def _deep_analysis(repo: dict) -> dict:
    """The repo's stage-4 analysis as a dict, tolerating the raw JSON string the registry stores."""
    raw = repo.get("deep_analysis")
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return {}


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
    deep_analysis = _deep_analysis(repo)
    profile = resolve_profile(deep_analysis.get("test_framework"), deep_analysis.get("framework_language"))
    test_name_prefix = repo_name.replace("/", "_")
    repo_traj_dir = trajectory_base_dir / f"repo_{test_name_prefix}"

    await _capture_for_repo(rlog, registry, repo, profile, repo_traj_dir, capture_mode, test_name_prefix)

    # Post-capture processing is idempotent, so it runs identically on a fresh capture or a resumed one.
    if capture_mode == "agent":
        valid_dirs, valid_count = await _postprocess_teacher_capture(rlog, registry, repo, repo_traj_dir)
        instruction_generated = {d: True for d in valid_dirs}
        thought_source = "teacher_raw"
        rlog.info(f"{valid_count} verified trajectories from {len(valid_dirs)} captured")
        if valid_count == 0:
            raise QualityReject(FailureCategory.NO_CAPTURE, "no verified trajectories produced")
    else:
        valid_dirs, instruction_generated = await _instructions_for_replay(rlog, registry, repo, repo_traj_dir)
        valid_count = len(valid_dirs)
        thought_source = "generated"

    # Think blocks belong to the separate resumable stage 7, so a think-gen stall can never block or fail a capture.
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


def _action_summary(trajectory: dict, capture_meta: dict | None) -> str:
    """The recorded actions rendered for the prompt's weakest signal, the disambiguation line."""
    steps_meta = (capture_meta or {}).get("steps", {})
    summary = []
    for step in trajectory.get("steps", [])[:10]:
        label = _element_label(step, steps_meta.get(str(step.get("step")), {}))
        for action in step.get("actions", []):
            action_type = action.get("type", "")
            if action_type == "click":
                summary.append(f"click {label}" if label else "click")
            elif action_type == "type":
                target = f" into {label}" if label else ""
                summary.append(f'type "{action.get("text", "")[:30]}"{target}')
            elif action_type == "keypress":
                keys = action.get("keys", []) or []
                summary.append(f"press {'+'.join(keys)}" if keys else "press a key")
            else:
                summary.append(f"{action_type} {label}".strip())
    return ", ".join(summary[:10]) or "(no actions recorded)"


async def generate_instruction(
    trajectory: dict,
    capture_meta: dict | None = None,
    repo: dict | None = None,
    test_source: str = "",
    registry=None,
) -> str:
    """An intent-level instruction for one trajectory, from the test name, its source, the repo and the actions."""
    steps = trajectory.get("steps", [])
    test_name = trajectory.get("task_id", "unknown")
    if not steps:
        return f"Verify the {test_name.replace('_', ' ')} behavior."

    repo = repo or {}
    test_source_block = (
        test_source.strip()
        if test_source and test_source.strip()
        else "(test source unavailable — rely on the test name and actions)"
    )
    prompt = INSTRUCTION_PROMPT.format(
        repo_name=repo.get("full_name") or repo.get("name") or "(repository not identified)",
        repo_description=(repo.get("description") or "").strip() or "(no description)",
        test_name=test_name,
        test_source=test_source_block,
        action_summary=_action_summary(trajectory, capture_meta),
    )

    try:
        result_text = await run_agent(
            prompt,
            ClaudeAgentOptions(
                model=MODEL_FOR_GROUNDED_GENERATION,
                max_turns=1,
                max_budget_usd=PIPELINE.instruction_budget_usd,
                permission_mode="bypassPermissions",
            ),
            registry=registry,
            full_name=repo.get("full_name", ""),
            stage=6,
            purpose="instruction",
            raise_on_error=False,
        )
    except UsageLimitReached:
        raise
    except Exception as e:
        # A safety net for SDK infrastructure failures, never a second guess at the model's own content.
        logger.warning(f"Instruction generation failed: {e}")
        return f"Verify the {test_name.replace('_', ' ')} behavior."

    return result_text.strip() or f"Verify the {test_name.replace('_', ' ')} behavior."


def load_test_source(traj_dir: Path, max_chars: int = 2000) -> str:
    """The test function source the instrumentation conftest cached next to trajectory.json, or ""."""
    # Trajectories predating that capture have none, and the instruction prompt has a fallback for it.
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
                max_budget_usd=PIPELINE.task_type_budget_usd,
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
    """Fill metadata.json's aggregate / provenance fields from the capture; best-effort, never raising into the flow."""
    try:
        traj_file = traj_dir / "trajectory.json"
        meta_file = traj_dir / "metadata.json"
        if not traj_file.exists():
            return

        with open(traj_file) as f:
            traj = json.load(f)
        capture_meta = load_json(traj_dir / CAPTURE_META_FILE)
        metadata = load_json(meta_file)
        task_type = await classify_task_type(traj, capture_meta, registry=registry, full_name=repo.get("full_name", ""))

        metadata.update(
            _enrichment_fields(
                traj, capture_meta, metadata, repo, profile, instruction_was_generated, thought_source, task_type
            )
        )
        write_json(meta_file, metadata)
    except UsageLimitReached:
        raise
    except Exception as e:
        logger.warning(f"Metadata enrichment failed for {traj_dir.name}: {e}")


def _enrichment_fields(
    traj: dict,
    capture_meta: dict,
    metadata: dict,
    repo: dict,
    profile,
    instruction_was_generated: bool,
    thought_source: str,
    task_type: str,
) -> dict:
    """The trajectory-level aggregates and provenance stage 6 writes over whatever the recorder already left on disk."""
    steps = traj.get("steps", [])
    action_hist, role_hist, n_nav, valid_bbox_ratio, n_console = _aggregate(steps, capture_meta.get("steps", {}))
    verifier = capture_meta.get("verifier") or {}
    verdict_fields = (
        {
            "efficiency": verifier.get("efficiency"),
            "efficiency_ratio": verifier.get("efficiency_ratio"),
            "relabeled": verifier.get("relabeled", False),
        }
        if verifier
        else {}
    )
    return metadata_from(
        instruction=traj.get("instruction", metadata.get("instruction", "")),
        total_steps=len(steps),
        domain=repo.get("domain") or RepoRegistry.classify_domain(repo.get("description"), repo.get("topics")),
        website=_host(traj.get("start_url", "")),
        framework=profile.test_framework,
        framework_language=profile.language,
        source_query=repo.get("source_query") or "",
        repo_stars=repo.get("stars", 0) or 0,
        action_type_histogram=action_hist,
        element_role_histogram=role_hist,
        length_bucket=_length_bucket(len(steps)),
        n_navigations=n_nav,
        valid_bbox_ratio=valid_bbox_ratio,
        n_console_errors=n_console,
        instruction_source="generated" if instruction_was_generated else "human",
        thought_source=thought_source,
        capture_source="agent" if thought_source.startswith("teacher") else "replay",
        **verdict_fields,
        dedup_hash=_dedup_hash(steps),
        task_type=task_type,
    )


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
