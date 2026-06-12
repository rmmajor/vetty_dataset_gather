import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from claude_agent_sdk import ClaudeAgentOptions

from repo_pipeline import app_runner, driver_session
from repo_pipeline._processing import run_repo
from repo_pipeline.agent_capture import run_teacher_task, verify_trajectory
from repo_pipeline.agent_runner import run_agent
from repo_pipeline.frameworks import resolve_profile
from repo_pipeline.instrumentation import REPO_DIR, RESULTS_DIR, setup_repo_instrumentation
from repo_pipeline.llm_config import MODEL_FOR_CLASSIFICATION, MODEL_FOR_GROUNDED_GENERATION
from repo_pipeline.logging_utils import StageProgress, repo_logger
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox
from repo_pipeline import schemas
from repo_pipeline.schemas import FailureCategory, QualityReject, RepoStatus, RetryableFailure
from repo_pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from repo_pipeline.task_proposer import propose_tasks

logger = logging.getLogger(__name__)

# Each Claude Agent SDK `query()` carries a fixed overhead (~$0.08 for Sonnet 4.6 — see
# memory pathb-e2e-and-sdk-budget-floor). Defaults sit comfortably above that floor; tune
# down only after verifying a smaller cap doesn't trip `error_max_budget_usd`.
INSTRUCTION_BUDGET_USD = float(os.environ.get("INSTRUCTION_BUDGET_USD", "0.15"))
TASK_TYPE_BUDGET_USD = float(os.environ.get("TASK_TYPE_BUDGET_USD", "0.15"))
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
        any(a.get("x", 0) > 0 or a.get("y", 0) > 0 for a in step.get("actions", []))
        for step in steps
    )
    screenshots_exist = all((traj_dir / step.get("screenshot", "")).exists() for step in steps)
    return has_valid_coords and screenshots_exist


def _has_valid_trajectories(repo_traj_dir: Path) -> bool:
    return repo_traj_dir.exists() and any(_is_valid_trajectory(d) for d in repo_traj_dir.glob("task_*/"))


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
            repo_name=repo_name, test_name_prefix=test_name_prefix,
            framework=profile.test_framework, framework_language=profile.language,
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
    try:
        await _run(sandbox.start, repo_name)

        _, stderr, exit_code = await _run(sandbox.clone_repo, repo_name, repo.get("default_branch"))
        if exit_code != 0:
            raise RetryableFailure(FailureCategory.CLONE_FAILED, f"clone failed: {stderr[:200]}")

        await _run(sandbox.install_node_deps)
        app_url = await _run(driver_session.start_app, sandbox, profile)
        if app_url is None:
            log_tail, _, _ = await _run(sandbox.exec_command, f"tail -15 {driver_session.APP_LOG} 2>/dev/null")
            raise QualityReject(
                FailureCategory.NO_CAPTURE,
                f"app did not boot for agent capture; app log tail: {log_tail[-300:]}",
            )

        await _run(driver_session.install_driver, sandbox)
        await _run(driver_session.start_driver, sandbox)
        client = driver_session.DriverClient(sandbox)

        # Ground the task proposer in the LIVE app: homepage snapshot + screenshot from a
        # probe session (discarded afterwards), plus README and test sources from the clone.
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
        tasks = await propose_tasks(
            repo, homepage_snapshot, homepage_screenshot, readme, test_hints,
            registry=registry, affordances=affordances,
        )
        if not tasks:
            raise RetryableFailure(FailureCategory.SDK_ERROR, "task proposal produced no tasks")
        rlog.info(f"proposed {len(tasks)} tasks for {app_url}: " +
                  "; ".join(t["instruction"][:60] for t in tasks[:4]) + " ...")

        # Tasks share one app server (state accumulates realistically) but each gets a fresh
        # browser context, so login/session state never leaks between trajectories.
        for task in tasks:
            if rate_limit_tracker.limit_reached():
                raise UsageLimitReached("usage limit reached mid agent capture")
            if sandbox.remaining_seconds() < AGENT_TASK_TIME_RESERVE:
                rlog.info(
                    f"container time low ({sandbox.remaining_seconds():.0f}s left), "
                    f"stopping after {tasks.index(task)} of {len(tasks)} tasks"
                )
                break
            # Set before the task runs: a mid-task abort (usage limit, SDK crash) must still
            # copy whatever steps the driver already recorded for this task.
            captured_anything = True
            state = await run_teacher_task(client, task, app_url, registry=registry, full_name=repo_name)
            rlog.info(
                f"task '{task['id']}': {state['actions']} actions, "
                f"outcome={state['agent_outcome'] or 'none'}"
            )
            # Incremental copy-out: teacher turns can be slow under rate-limit waits, and a
            # container-timeout mid-task must never cost the tasks already recorded.
            repo_traj_dir.mkdir(parents=True, exist_ok=True)
            try:
                await _run(sandbox.copy_out, RESULTS_DIR, str(repo_traj_dir))
            except Exception as e:
                logger.warning(f"incremental copy_out failed for {repo_name}: {e}")
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
            deep_analysis = json.loads(repo["deep_analysis"]) if isinstance(repo["deep_analysis"], str) else repo["deep_analysis"]
        except (json.JSONDecodeError, TypeError):
            pass

    profile = resolve_profile(deep_analysis.get("test_framework"), deep_analysis.get("framework_language"))
    test_name_prefix = repo_name.replace("/", "_")
    repo_traj_dir = trajectory_base_dir / f"repo_{test_name_prefix}"

    # Resume: a prior run may have captured trajectories but died during verification/instruction
    # generation, leaving the repo stuck before completion. Re-running would re-capture from
    # scratch — wasteful, so skip capture when valid trajectories already exist and just finish
    # the (idempotent) post-processing below.
    if _has_valid_trajectories(repo_traj_dir):
        rlog.info(f"resuming: valid trajectories already on disk ({repo_traj_dir.name}), skipping capture")
    elif capture_mode == "agent":
        rlog.info("agent capture: booting app + running teacher tasks")
        await _capture_agent(rlog, registry, repo, profile, repo_traj_dir)
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
            traj_dir, repo, profile, instruction_generated.get(traj_dir, True),
            registry=registry, thought_source=thought_source,
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
    test_source_block = test_source.strip() if test_source and test_source.strip() else "(test source unavailable — rely on the test name and actions)"

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


async def _enrich_metadata(traj_dir: Path, repo: dict, profile, instruction_was_generated: bool,
                           registry=None, thought_source: str = "generated"):
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
