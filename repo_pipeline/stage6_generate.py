import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from claude_agent_sdk import ClaudeAgentOptions

from repo_pipeline import app_runner
from repo_pipeline._processing import run_repo
from repo_pipeline.agent_runner import run_agent
from repo_pipeline.frameworks import resolve_profile
from repo_pipeline.instrumentation import setup_repo_instrumentation
from repo_pipeline.llm_config import MODEL_FOR_CLASSIFICATION, MODEL_FOR_GROUNDED_GENERATION
from repo_pipeline.logging_utils import StageProgress, repo_logger
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox
from repo_pipeline.schemas import FailureCategory, QualityReject, RepoStatus, RetryableFailure
from repo_pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker

logger = logging.getLogger(__name__)

# Each Claude Agent SDK `query()` carries a fixed overhead (~$0.08 for Sonnet 4.6 — see
# memory pathb-e2e-and-sdk-budget-floor). Defaults sit comfortably above that floor; tune
# down only after verifying a smaller cap doesn't trip `error_max_budget_usd`.
INSTRUCTION_BUDGET_USD = float(os.environ.get("INSTRUCTION_BUDGET_USD", "0.15"))
TASK_TYPE_BUDGET_USD = float(os.environ.get("TASK_TYPE_BUDGET_USD", "0.15"))
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


async def generate_for_repo(
    registry: RepoRegistry,
    repo: dict,
    trajectory_base_dir: Path,
):
    repo_name = repo["full_name"]

    if repo.get("current_stage", 0) >= 6:
        return

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

    # Resume: a prior run may have captured trajectories but died during instruction/think
    # generation, leaving the repo stuck before completion. Re-running the suite would re-capture
    # from scratch — wasteful, and on dynamic/anti-bot sites it loses the good capture. So skip
    # the suite when valid trajectories already exist and just finish the (idempotent) supervision
    # + completion steps below.
    if _has_valid_trajectories(repo_traj_dir):
        rlog.info(f"resuming: valid trajectories already on disk ({repo_traj_dir.name}), skipping capture")
    else:
        rlog.info("generating trajectories")
        await _capture_suite(rlog, repo, profile, repo_traj_dir, test_name_prefix)

    # ---- Post-capture processing. Idempotent, so it runs identically on a fresh capture or a
    # resumed one: instructions and think blocks are only generated where missing. ----
    trajectory_dirs = list(repo_traj_dir.glob("task_*/")) if repo_traj_dir.exists() else []
    valid_dirs = []
    instruction_generated = {}

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
        await _enrich_metadata(traj_dir, repo, profile, instruction_generated.get(traj_dir, True), registry=registry)

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


async def _enrich_metadata(traj_dir: Path, repo: dict, profile, instruction_was_generated: bool, registry=None):
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
        metadata["thought_source"] = "generated"
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
):
    repos = registry.get_repos_for_stage(6, limit=limit)
    logger.info(f"Generating trajectories for {len(repos)} repos with {workers} workers")

    semaphore = asyncio.Semaphore(workers)
    progress = StageProgress(6, len(repos))

    async def process_with_semaphore(repo):
        if rate_limit_tracker.limit_reached():
            return
        async def core(reg, r):
            return await generate_for_repo(reg, r, trajectory_base_dir)
        async with semaphore:
            try:
                await run_repo(6, repo, registry, core, progress)
            except UsageLimitReached:
                return

    await asyncio.gather(*[process_with_semaphore(repo) for repo in repos])

    return progress.summary()
