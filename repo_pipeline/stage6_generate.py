import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from claude_agent_sdk import ClaudeAgentOptions

from repo_pipeline.frameworks import resolve_profile
from repo_pipeline.instrumentation import setup_repo_instrumentation
from repo_pipeline.llm_config import MODEL_FOR_CLASSIFICATION, MODEL_FOR_GROUNDED_GENERATION
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox
from repo_pipeline.schemas import RepoStatus
from repo_pipeline.sdk_runner import rate_limit_tracker, run_query, UsageLimitReached
from repo_pipeline.think_generator import generate_thinks_for_trajectory

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


async def generate_for_repo(
    registry: RepoRegistry,
    repo: dict,
    trajectory_base_dir: Path,
):
    repo_name = repo["full_name"]

    if repo.get("current_stage", 0) >= 6:
        return

    logger.info(f"Generating trajectories: {repo_name}")

    deep_analysis = {}
    if repo.get("deep_analysis"):
        try:
            deep_analysis = json.loads(repo["deep_analysis"]) if isinstance(repo["deep_analysis"], str) else repo["deep_analysis"]
        except (json.JSONDecodeError, TypeError):
            pass

    profile = resolve_profile(deep_analysis.get("test_framework"), deep_analysis.get("framework_language"))
    test_name_prefix = repo_name.replace("/", "_")

    sandbox = DockerSandbox(image=profile.sandbox_image, platform=profile.sandbox_platform)

    try:
        await _run(sandbox.start, repo_name)

        _, stderr, exit_code = await _run(sandbox.clone_repo, repo_name, repo.get("default_branch"))
        if exit_code != 0:
            registry.reject(repo_name, stage=6, reason=f"clone failed: {stderr[:200]}")
            return

        await _run(setup_repo_instrumentation, sandbox, profile)

        # Run the full instrumented test suite directly (no agent — install/wiring/viewport are
        # all handled in code). Whatever the suite executes is captured to /workspace/results;
        # a timeout just yields partial trajectories, which is fine.
        suite_cmd = profile.test_command_suite.format(
            repo_name=repo_name, test_name_prefix=test_name_prefix,
            framework=profile.test_framework, framework_language=profile.language,
        )
        out, err, code = await _run(sandbox.exec_command, suite_cmd, SUITE_RUN_TIMEOUT)
        logger.info(f"  {repo_name}: suite run exit={code}; {(out or err)[-300:].strip()[:300]}")

        repo_traj_dir = trajectory_base_dir / f"repo_{test_name_prefix}"
        repo_traj_dir.mkdir(parents=True, exist_ok=True)

        try:
            await _run(sandbox.copy_out, "/workspace/results", str(repo_traj_dir))
        except Exception as e:
            logger.warning(f"  Failed to copy results for {repo_name}: {e}")
            registry.reject(repo_name, stage=6, reason=f"copy results failed: {str(e)[:200]}")
            return

        trajectory_dirs = list(repo_traj_dir.glob("task_*/"))
        valid_dirs = []
        instruction_generated = {}

        for traj_dir in trajectory_dirs:
            trajectory_file = traj_dir / "trajectory.json"
            if not trajectory_file.exists():
                continue

            with open(trajectory_file) as f:
                traj = json.load(f)

            steps = traj.get("steps", [])
            if not steps:
                continue

            has_valid_coords = any(
                any(a.get("x", 0) > 0 or a.get("y", 0) > 0 for a in step.get("actions", []))
                for step in steps
            )

            screenshots_exist = all(
                (traj_dir / step["screenshot"]).exists()
                for step in steps
            )

            if not (has_valid_coords and screenshots_exist):
                continue

            valid_dirs.append(traj_dir)

            if not traj.get("instruction"):
                test_source = _load_test_source(traj_dir)
                instruction = await generate_instruction(
                    traj,
                    _load_capture_meta(traj_dir),
                    repo=repo,
                    test_source=test_source,
                )
                traj["instruction"] = instruction
                instruction_generated[traj_dir] = True
                with open(trajectory_file, "w") as f:
                    json.dump(traj, f, indent=2)
            else:
                instruction_generated[traj_dir] = False

        valid_count = len(valid_dirs)
        logger.info(f"  {repo_name}: {valid_count} valid trajectories from {len(trajectory_dirs)} total")

        if valid_count == 0:
            registry.reject(repo_name, stage=6, reason="no valid trajectories produced")
            return

        logger.info(f"  Generating think blocks for {repo_name}...")
        for traj_dir in valid_dirs:
            await generate_thinks_for_trajectory(traj_dir, workers=5)

        # Derive trajectory-level metadata from the captured trajectory + capture_meta, now
        # that instructions and think blocks are filled in.
        for traj_dir in valid_dirs:
            await _enrich_metadata(traj_dir, repo, profile, instruction_generated.get(traj_dir, True))

        registry.update_status(
            repo_name,
            RepoStatus.COMPLETED.value,
            stage=6,
            trajectory_count=valid_count,
            trajectory_dir=str(repo_traj_dir),
        )

    except UsageLimitReached:
        raise
    except Exception as e:
        logger.error(f"Error generating for {repo_name}: {e}")
        registry.reject(repo_name, stage=6, reason=f"generation error: {str(e)[:200]}")
    finally:
        await _run(sandbox.stop)


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
        result_text = await run_query(
            prompt,
            ClaudeAgentOptions(
                model=MODEL_FOR_GROUNDED_GENERATION,
                max_turns=1,
                max_budget_usd=INSTRUCTION_BUDGET_USD,
                permission_mode="bypassPermissions",
            ),
            label="instruction",
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


async def classify_task_type(trajectory: dict, capture_meta: dict) -> str:
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
        result_text = await run_query(
            prompt,
            ClaudeAgentOptions(
                model=MODEL_FOR_CLASSIFICATION,
                max_turns=1,
                max_budget_usd=TASK_TYPE_BUDGET_USD,
                permission_mode="bypassPermissions",
            ),
            label="classify",
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


async def _enrich_metadata(traj_dir: Path, repo: dict, profile, instruction_was_generated: bool):
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
        metadata["task_type"] = await classify_task_type(traj, capture_meta)

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
    total_trajectories = 0

    async def process_with_semaphore(repo):
        nonlocal total_trajectories
        if rate_limit_tracker.limit_reached():
            return
        async with semaphore:
            try:
                await generate_for_repo(registry, repo, trajectory_base_dir)
            except UsageLimitReached:
                return

        repo_after = registry.get_repo(repo["full_name"])
        if repo_after and repo_after["status"] == RepoStatus.COMPLETED.value:
            total_trajectories += repo_after.get("trajectory_count", 0)

    await asyncio.gather(*[process_with_semaphore(repo) for repo in repos])

    stats = registry.get_stats()
    completed = stats.get(RepoStatus.COMPLETED.value, 0)
    rejected = stats.get(RepoStatus.REJECTED_GENERATE.value, 0)
    logger.info(f"Generation complete: {completed} completed, {rejected} rejected, {total_trajectories} new trajectories")
    return {
        "processed": len(repos),
        "passed": completed,
        "rejected": rejected,
        "trajectories": total_trajectories,
    }
