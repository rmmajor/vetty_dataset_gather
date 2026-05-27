import asyncio
import json
import logging
import os
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from repo_pipeline.frameworks import resolve_profile
from repo_pipeline.instrumentation import setup_repo_instrumentation
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox
from repo_pipeline.schemas import RepoStatus
from repo_pipeline.think_generator import generate_thinks_for_trajectory

logger = logging.getLogger(__name__)

INSTRUCTION_BUDGET_USD = float(os.environ.get("INSTRUCTION_BUDGET_USD", "0.05"))
SUITE_RUN_TIMEOUT = 480  # seconds; partial capture on timeout is fine for trajectory mining

INSTRUCTION_PROMPT = """Given this browser test code and the sequence of actions recorded, write a natural language instruction (1-2 sentences) describing what task this test performs from a user's perspective.

Test name: {test_name}
Actions taken: {action_summary}

Write ONLY the instruction, as if you're telling someone what to do on a website. Example: "Navigate to the shopping cart, update the quantity to 2, and proceed to checkout."
"""


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
        suite_cmd = profile.test_command_suite.format(repo_name=repo_name, test_name_prefix=test_name_prefix)
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
        valid_count = 0

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

            if has_valid_coords and screenshots_exist:
                valid_count += 1

                if not traj.get("instruction"):
                    instruction = await generate_instruction(traj)
                    traj["instruction"] = instruction
                    with open(trajectory_file, "w") as f:
                        json.dump(traj, f, indent=2)

        logger.info(f"  {repo_name}: {valid_count} valid trajectories from {len(trajectory_dirs)} total")

        if valid_count == 0:
            registry.reject(repo_name, stage=6, reason="no valid trajectories produced")
            return

        logger.info(f"  Generating think blocks for {repo_name}...")
        for traj_dir in trajectory_dirs:
            if (traj_dir / "trajectory.json").exists():
                await generate_thinks_for_trajectory(traj_dir, workers=5)

        registry.update_status(
            repo_name,
            RepoStatus.COMPLETED.value,
            stage=6,
            trajectory_count=valid_count,
            trajectory_dir=str(repo_traj_dir),
        )

    except Exception as e:
        logger.error(f"Error generating for {repo_name}: {e}")
        registry.reject(repo_name, stage=6, reason=f"generation error: {str(e)[:200]}")
    finally:
        await _run(sandbox.stop)


async def generate_instruction(trajectory: dict) -> str:
    steps = trajectory.get("steps", [])
    if not steps:
        return "Complete the web browsing task."

    action_summary = []
    for step in steps[:10]:
        for action in step.get("actions", []):
            action_type = action.get("type", "")
            if action_type == "click":
                action_summary.append(f"click({action.get('x')}, {action.get('y')})")
            elif action_type == "type":
                action_summary.append(f'type("{action.get("text", "")[:30]}")')
            elif action_type == "keypress":
                action_summary.append(f"keypress({action.get('keys', [])})")
            else:
                action_summary.append(action_type)

    test_name = trajectory.get("task_id", "unknown")
    action_str = ", ".join(action_summary[:10])

    prompt = INSTRUCTION_PROMPT.format(
        test_name=test_name,
        action_summary=action_str,
    )

    result_text = ""
    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model="claude-haiku-4-5-20251001",
                max_turns=1,
                max_budget_usd=INSTRUCTION_BUDGET_USD,
                permission_mode="bypassPermissions",
            ),
        ):
            if isinstance(message, ResultMessage) and message.subtype == "success":
                result_text = message.result or ""
    except Exception as e:
        logger.warning(f"Instruction generation failed: {e}")
        return f"Complete the web task: {test_name}"

    return result_text.strip() or f"Complete the web task: {test_name}"


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
        async with semaphore:
            await generate_for_repo(registry, repo, trajectory_base_dir)

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
