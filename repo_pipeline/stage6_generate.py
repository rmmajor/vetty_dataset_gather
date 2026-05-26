import asyncio
import json
import logging
import os
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from repo_pipeline.frameworks import resolve_profile
from repo_pipeline.instrumentation import setup_repo_instrumentation
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox, build_sandbox_mcp_server
from repo_pipeline.schemas import RepoStatus
from repo_pipeline.think_generator import generate_thinks_for_trajectory

logger = logging.getLogger(__name__)

GENERATE_BUDGET_USD = float(os.environ.get("STAGE6_BUDGET_USD", "1.50"))
INSTRUCTION_BUDGET_USD = float(os.environ.get("INSTRUCTION_BUDGET_USD", "0.05"))

GENERATE_PROMPT = """You are performing full trajectory generation for the browser-test repository '{repo_name}' cloned at /workspace/repo.

Framework: {framework}

The instrumentation has already been injected.

Run every shell command through the `exec` tool — there is no local shell, `exec` runs inside the container. Each `exec` call is a fresh shell, so environment variables do NOT persist between calls: put the trajectory env vars inline on the same line as the test command.

Your tasks:

1. **Run the FULL test suite** with instrumentation, passing the env vars inline in the SAME `exec` command. The container has NO display, so always launch the browser under `xvfb-run -a` (required for headless=False repos, harmless otherwise):
   `{test_command}`

2. **Check results**:
   - List all directories in /workspace/results/
   - For each trajectory directory, verify:
     - trajectory.json exists and has steps
     - Screenshots (PNG files) exist
     - Actions have valid coordinates (x,y > 0)
   - Report any broken trajectories

3. **Generate instructions**: For each valid trajectory, look at the test code and the trajectory data, then write a natural language instruction describing what the test does. Save it by updating the trajectory.json file's "instruction" field.

4. **Report summary** as JSON:
```json
{{
  "total_trajectories": <int>,
  "valid_trajectories": <int>,
  "total_steps": <int>,
  "trajectory_dirs": ["task_xxx", "task_yyy", ...]
}}
```

Output ONLY the JSON at the end.
"""

INSTRUCTION_PROMPT = """Given this browser test code and the sequence of actions recorded, write a natural language instruction (1-2 sentences) describing what task this test performs from a user's perspective.

Test name: {test_name}
Actions taken: {action_summary}

Write ONLY the instruction, as if you're telling someone what to do on a website. Example: "Navigate to the shopping cart, update the quantity to 2, and proceed to checkout."
"""


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
        sandbox.start(repo_name)

        stdout, stderr, exit_code = sandbox.clone_repo(repo_name, branch=repo.get("default_branch"))
        if exit_code != 0:
            registry.reject(repo_name, stage=6, reason=f"clone failed: {stderr[:200]}")
            return

        setup_repo_instrumentation(sandbox, profile)

        prompt = GENERATE_PROMPT.format(
            repo_name=repo_name,
            framework=profile.name,
            test_command=profile.test_command_suite.format(repo_name=repo_name, test_name_prefix=test_name_prefix),
        )

        mcp_server, sandbox_tools = build_sandbox_mcp_server(sandbox)

        result_text = ""
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model="claude-sonnet-4-20250514",
                max_turns=35,
                max_budget_usd=GENERATE_BUDGET_USD,
                permission_mode="bypassPermissions",
                mcp_servers={"sandbox": mcp_server},
                allowed_tools=sandbox_tools,
                disallowed_tools=["Bash"],
            ),
        ):
            if isinstance(message, ResultMessage) and message.subtype == "success":
                result_text = message.result or ""

        repo_traj_dir = trajectory_base_dir / f"repo_{test_name_prefix}"
        repo_traj_dir.mkdir(parents=True, exist_ok=True)

        try:
            sandbox.copy_out("/workspace/results", str(repo_traj_dir))
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
        sandbox.stop()


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
    cost_estimate = 0.0
    total_trajectories = 0

    async def process_with_semaphore(repo):
        nonlocal cost_estimate, total_trajectories
        if budget_remaining is not None and cost_estimate > budget_remaining:
            logger.warning("Budget limit reached, stopping")
            return
        async with semaphore:
            await generate_for_repo(registry, repo, trajectory_base_dir)
            cost_estimate += 0.60

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
