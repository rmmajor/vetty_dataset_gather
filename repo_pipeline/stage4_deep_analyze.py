import asyncio
import json
import logging
import os

from claude_agent_sdk import ClaudeAgentOptions

from repo_pipeline._processing import run_repo
from repo_pipeline.agent_runner import run_agent
from repo_pipeline.frameworks import all_interactive_actions
from repo_pipeline.llm_config import MODEL_FOR_REASONING
from repo_pipeline.logging_utils import StageProgress, repo_logger
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox, build_sandbox_mcp_server
from repo_pipeline.schemas import (
    DeepAnalysis,
    FailureCategory,
    QualityReject,
    RepoStatus,
    RetryableFailure,
)
from repo_pipeline.sdk_runner import UsageLimitReached

logger = logging.getLogger(__name__)

ANALYZE_BUDGET_USD = float(os.environ.get("STAGE4_BUDGET_USD", "0.75"))

DEEP_ANALYSIS_PROMPT = """You are analyzing the browser-test repository '{repo_name}' that has been cloned to /workspace/repo inside a Docker container. The test suite uses either Playwright or Selenium.

Your goal is to perform a STATIC analysis of this repository's test suite — read files only, do NOT install dependencies (installability is verified later in the trial stage) — to determine if it can produce high-quality browser automation trajectories.

Stage 3 shallow scoring gave this repo the following scores (1-5 each):
{shallow_scores_summary}

Use the `exec` tool to run shell commands inside the container, and `read`/`ls` to inspect files. There is no local shell — every command must go through `exec`. Do the following:

1. **Count test files**: Find all files matching test patterns (*.test.ts, *.spec.ts, *_test.py, test_*.py, *.test.js, *.spec.js) and count them.

2. **Count test cases**: Look inside test files for individual test definitions (test(), it(), def test_*).

3. **Parse action types**: Scan test files for UI action calls — categorize which of these are used:
   - Playwright: click, fill/type, press, selectOption, check/uncheck, dblclick, dragTo/drag_and_drop
   - Selenium: click, send_keys, clear, submit, Select.select_by_*, ActionChains
   - navigation (page.goto / driver.get), waits, assertions

4. **Check self-containment**: Is there a docker-compose.yml? A local dev server script (package.json scripts like "dev", "start")? Or do tests hit external URLs?

5. **Identify framework**: Which browser-test framework is this? Look at the imports and dependencies:
   - "selenium" / "webdriver" imports, selenium in requirements.txt → test_framework = "selenium"
   - "@playwright/test" or "pytest-playwright" / "playwright" → test_framework = "playwright"
   Also determine the language: Python (pytest), TypeScript, or JavaScript. NOTE: Selenium is only supported for Python — flag a non-Python Selenium repo in notes.

6. **Estimate trajectory count**: Based on test count and typical actions per test, estimate how many usable trajectories this repo could produce.

After analysis, output your findings as a JSON object with EXACTLY these fields:
```json
{{
  "test_file_count": <int>,
  "test_case_count": <int>,
  "action_types": ["click", "send_keys", ...],
  "is_self_contained": <bool>,
  "has_docker_compose": <bool>,
  "test_framework": "<playwright|selenium>",
  "playwright_version": "<version or null>",
  "framework_language": "<python|typescript|javascript>",
  "estimated_trajectories": <int>,
  "notes": "<brief notes about anything notable>"
}}
```

Output ONLY the JSON at the end of your analysis.
"""

# Gate for promotion to stage 5 (deep_score is 0–10). Env-overridable so it can be tuned from the
# dashboard's score distribution without a code edit.
DEEP_SCORE_THRESHOLD = int(os.environ.get("STAGE4_DEEP_SCORE_THRESHOLD", "4"))


def compute_deep_score(analysis: DeepAnalysis) -> int:
    score = 0

    if analysis.test_file_count >= 10:
        score += 3
    elif analysis.test_file_count >= 5:
        score += 2
    elif analysis.test_file_count >= 2:
        score += 1

    if analysis.test_case_count >= 20:
        score += 3
    elif analysis.test_case_count >= 10:
        score += 2
    elif analysis.test_case_count >= 3:
        score += 1

    found_interactive = {a.lower() for a in analysis.action_types} & all_interactive_actions()
    score += min(len(found_interactive), 3)

    if analysis.is_self_contained or analysis.has_docker_compose:
        score += 1

    return score


async def analyze_single_repo(registry: RepoRegistry, repo: dict):
    """Static deep analysis of one repo's test suite. Raises QualityReject on a low deep score
    (carrying the analysis so it persists for the dashboard) or RetryableFailure on infra/flaky
    failures; run_repo turns those into retries or a categorized terminal rejection."""
    repo_name = repo["full_name"]

    if repo.get("current_stage", 0) >= 4:
        return

    rlog = repo_logger(4, repo_name)
    rlog.info("deep analyzing")
    sandbox = DockerSandbox()

    try:
        sandbox.start(repo_name)

        stdout, stderr, exit_code = sandbox.clone_repo(repo_name, branch=repo.get("default_branch"))
        if exit_code != 0:
            raise RetryableFailure(FailureCategory.CLONE_FAILED, f"clone failed: {stderr[:200]}")

        shallow_scores_raw = repo.get("shallow_scores")
        if shallow_scores_raw:
            scores = json.loads(shallow_scores_raw) if isinstance(shallow_scores_raw, str) else shallow_scores_raw
            shallow_scores_summary = ", ".join(f"{k}={v}" for k, v in scores.items())
            shallow_scores_summary += f" (total: {repo.get('shallow_total', '?')}/35)"
        else:
            shallow_scores_summary = "(not available)"

        prompt = DEEP_ANALYSIS_PROMPT.format(repo_name=repo_name, shallow_scores_summary=shallow_scores_summary)

        mcp_server, sandbox_tools = build_sandbox_mcp_server(sandbox)

        result_text = await run_agent(
            prompt,
            ClaudeAgentOptions(
                model=MODEL_FOR_REASONING,
                max_turns=25,
                max_budget_usd=ANALYZE_BUDGET_USD,
                permission_mode="bypassPermissions",
                mcp_servers={"sandbox": mcp_server},
                allowed_tools=sandbox_tools,
                disallowed_tools=["Bash"],
            ),
            registry=registry,
            full_name=repo_name,
            stage=4,
            purpose="deep_analyze",
            raise_on_error=True,
        )

        analysis = parse_analysis_result(result_text)
        if analysis is None:
            raise RetryableFailure(FailureCategory.PARSE_FAILED, "failed to parse deep analysis")

        deep_score = compute_deep_score(analysis)
        rlog.info(f"deep_score={deep_score}, tests={analysis.test_case_count}, actions={analysis.action_types}")

        if deep_score >= DEEP_SCORE_THRESHOLD:
            registry.update_status(
                repo_name,
                RepoStatus.ANALYZED.value,
                stage=4,
                deep_analysis=analysis.model_dump(),
                deep_score=deep_score,
            )
        else:
            raise QualityReject(
                FailureCategory.QUALITY_GATE,
                f"deep score {deep_score}/{DEEP_SCORE_THRESHOLD} (tests: {analysis.test_case_count}, self_contained: {analysis.is_self_contained})",
                deep_analysis=analysis.model_dump(),
                deep_score=deep_score,
            )
    finally:
        sandbox.stop()


def parse_analysis_result(text: str) -> DeepAnalysis | None:
    try:
        json_start = text.rfind("{")
        json_end = text.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            return None
        json_str = text[json_start:json_end]
        data = json.loads(json_str)
        return DeepAnalysis(**data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse analysis JSON: {e}")
        return None


async def run_deep_analyze(
    registry: RepoRegistry,
    limit: int = None,
    workers: int = 3,
    budget_remaining: float = None,
):
    repos = registry.get_repos_for_stage(4, limit=limit)
    logger.info(f"Deep analyzing {len(repos)} repos with {workers} workers")

    semaphore = asyncio.Semaphore(workers)
    progress = StageProgress(4, len(repos))
    cost_estimate = 0.0

    async def process_with_semaphore(repo):
        nonlocal cost_estimate
        if rate_limit_tracker.limit_reached():
            return
        if budget_remaining is not None and cost_estimate > budget_remaining:
            logger.warning("Budget limit reached, stopping")
            return
        async with semaphore:
            try:
                await run_repo(4, repo, registry, analyze_single_repo, progress)
            except UsageLimitReached:
                return
            cost_estimate += 0.20

    await asyncio.gather(*[process_with_semaphore(repo) for repo in repos])

    return progress.summary()
