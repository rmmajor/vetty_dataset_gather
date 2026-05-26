import asyncio
import json
import logging
import os

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from repo_pipeline.frameworks import resolve_profile
from repo_pipeline.instrumentation import setup_repo_instrumentation
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox, build_sandbox_mcp_server
from repo_pipeline.schemas import RepoStatus, TrialResults

logger = logging.getLogger(__name__)

TRIAL_BUDGET_USD = float(os.environ.get("STAGE5_BUDGET_USD", "1.00"))

TRIAL_RUN_PROMPT = """You are setting up and running a browser-test trial in a Docker container. The repository '{repo_name}' is cloned at /workspace/repo.

Framework: {framework}

Run every shell command through the `exec` tool — there is no local shell, `exec` runs inside the container. Each `exec` call is a fresh shell, so environment variables do NOT persist between calls: put any `VAR=value` assignments on the same line as the command that needs them.

Your tasks:

1. **Inject instrumentation**:
   - The instrumentation file has already been placed at the correct location.
   - {setup_note}

2. **Run ONE test file** as a proof of concept, passing the trajectory env vars inline in the SAME `exec` command. The container has NO display, so always launch the browser under `xvfb-run -a` (this is required for repos that use headless=False, and harmless otherwise):
   - Pick the test file that looks most promising (diverse actions, multiple test cases).
   - Replace <test_file> with that file and run:
     `{test_command}`

3. **Check results**:
   - Look in /workspace/results/ for generated trajectory directories
   - Check if screenshots were captured (PNG files exist)
   - Check if trajectory.json files have valid bounding box coordinates (x,y > 0)
   - Count how many actions were captured

4. **Report results** as a JSON object:
```json
{{
  "pass_rate": <float 0-1>,
  "action_diversity_score": <float 0-1>,
  "domain_relevance_score": <float 0-1>,
  "trajectory_count_estimate": <int>,
  "setup_complexity_score": <float 0-1>,
  "valid_bounding_boxes": <int>,
  "total_actions_captured": <int>,
  "screenshots_captured": <int>,
  "notes": "<brief notes>"
}}
```

Output ONLY the JSON at the end.
"""

TRIAL_SCORE_THRESHOLD = 0.40


async def trial_single_repo(registry: RepoRegistry, repo: dict):
    repo_name = repo["full_name"]

    if repo.get("current_stage", 0) >= 5:
        return

    logger.info(f"Trial run: {repo_name}")

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
            registry.reject(repo_name, stage=5, reason=f"clone failed: {stderr[:200]}")
            return

        setup_repo_instrumentation(sandbox, profile)

        prompt = TRIAL_RUN_PROMPT.format(
            repo_name=repo_name,
            framework=profile.name,
            setup_note=profile.setup_note,
            test_command=profile.test_command_single.format(repo_name=repo_name, test_name_prefix=test_name_prefix),
        )

        mcp_server, sandbox_tools = build_sandbox_mcp_server(sandbox)

        result_text = ""
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model="claude-sonnet-4-20250514",
                max_turns=20,
                max_budget_usd=TRIAL_BUDGET_USD,
                permission_mode="bypassPermissions",
                mcp_servers={"sandbox": mcp_server},
                allowed_tools=sandbox_tools,
                disallowed_tools=["Bash"],
            ),
        ):
            if isinstance(message, ResultMessage) and message.subtype == "success":
                result_text = message.result or ""

        trial = parse_trial_result(result_text)
        if trial is None:
            registry.reject(repo_name, stage=5, reason="failed to parse trial results")
            return

        if trial.pass_rate < 0.5:
            registry.reject(repo_name, stage=5, reason=f"low pass rate ({trial.pass_rate:.0%})")
            return

        if trial.valid_bounding_boxes == 0:
            registry.reject(repo_name, stage=5, reason="no valid bounding boxes captured")
            return

        if trial.screenshots_captured == 0:
            registry.reject(repo_name, stage=5, reason="no screenshots captured")
            return

        weighted = trial.weighted_score
        logger.info(f"  {repo_name}: weighted_score={weighted:.2f}, pass_rate={trial.pass_rate:.0%}, actions={trial.total_actions_captured}")

        if weighted >= TRIAL_SCORE_THRESHOLD:
            registry.update_status(
                repo_name,
                RepoStatus.VALIDATED.value,
                stage=5,
                trial_results=trial.model_dump(),
                trial_score=weighted,
            )
        else:
            registry.reject(
                repo_name,
                stage=5,
                reason=f"weighted score {weighted:.2f} < {TRIAL_SCORE_THRESHOLD}",
            )

    except Exception as e:
        logger.error(f"Error in trial run for {repo_name}: {e}")
        registry.reject(repo_name, stage=5, reason=f"trial error: {str(e)[:200]}")
    finally:
        sandbox.stop()


def parse_trial_result(text: str) -> TrialResults | None:
    try:
        json_start = text.rfind("{")
        json_end = text.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            return None
        json_str = text[json_start:json_end]
        data = json.loads(json_str)
        return TrialResults(**data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse trial JSON: {e}")
        return None


async def run_trial(
    registry: RepoRegistry,
    limit: int = None,
    workers: int = 2,
    budget_remaining: float = None,
):
    repos = registry.get_repos_for_stage(5, limit=limit)
    logger.info(f"Trial running {len(repos)} repos with {workers} workers")

    semaphore = asyncio.Semaphore(workers)
    cost_estimate = 0.0

    async def process_with_semaphore(repo):
        nonlocal cost_estimate
        if budget_remaining is not None and cost_estimate > budget_remaining:
            logger.warning("Budget limit reached, stopping")
            return
        async with semaphore:
            await trial_single_repo(registry, repo)
            cost_estimate += 0.30

    await asyncio.gather(*[process_with_semaphore(repo) for repo in repos])

    stats = registry.get_stats()
    passed = stats.get(RepoStatus.VALIDATED.value, 0)
    rejected = stats.get(RepoStatus.REJECTED_TRIAL.value, 0)
    logger.info(f"Trial runs complete: {passed} validated, {rejected} rejected")
    return {"processed": len(repos), "passed": passed, "rejected": rejected}
