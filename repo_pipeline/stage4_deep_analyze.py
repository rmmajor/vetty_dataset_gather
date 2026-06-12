import asyncio
import json
import logging
import os

from claude_agent_sdk import ClaudeAgentOptions

from repo_pipeline._processing import run_repo
from repo_pipeline.agent_runner import run_agent
from repo_pipeline.frameworks import all_interactive_actions, is_supported_profile
from repo_pipeline.llm_config import MODEL_FOR_REASONING
from repo_pipeline.logging_utils import StageProgress, repo_logger
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox, build_sandbox_mcp_server
from repo_pipeline import schemas
from repo_pipeline.schemas import (
    DeepAnalysis,
    FailureCategory,
    QualityReject,
    RepoStatus,
    RetryableFailure,
)
from repo_pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker

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

4. **Assess capturability** — can this suite actually run and drive a UI inside a bare sandbox (no host network, no external services, no secrets)? This is the single biggest predictor of whether we get usable trajectories. Determine each of:
   - has_local_dev_server: does package.json define a "dev"/"start"/"serve"/"preview" script (or equivalent) that boots the app locally? The pipeline can start it before the tests.
   - has_webserver_config: does playwright.config.* declare a `webServer` block (Playwright starts the app itself)?
   - has_docker_compose: is there a docker-compose.yml?
   - needs_external_backend: do tests require a separate backend/API/database, login credentials, paid third-party services, or a browser wallet/extension that won't exist in the sandbox?
   - tests_public_url: do tests target a hardcoded PUBLIC https URL (a real hosted site) rather than localhost?
   - is_self_contained: overall, can it run end-to-end with only dependency install + (optionally) starting its own bundled app?

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
  "has_local_dev_server": <bool>,
  "has_webserver_config": <bool>,
  "needs_external_backend": <bool>,
  "tests_public_url": <bool>,
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


def can_self_host(analysis: DeepAnalysis) -> bool:
    return (
        analysis.has_local_dev_server
        or analysis.has_webserver_config
        or analysis.has_docker_compose
        or analysis.is_self_contained
    )


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

    # Bootability — the dominant axis. Agent capture (the primary trajectory source) needs the
    # repo's own app serving inside the sandbox: a dev server / webServer / docker-compose is
    # exactly what start_app can boot. A suite that only targets a public URL has nothing for
    # the agent to drive (hard-gated below); one needing an external backend we cannot provide
    # is penalized.
    bootable = can_self_host(analysis)
    if bootable:
        score += 3
    if analysis.needs_external_backend and not bootable:
        score -= 2

    return max(0, score)


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

        # Replay mode runs the repo's own tests, so frameworks with no capture hook (e.g. Java
        # Playwright, non-Python Selenium) are rejected before burning a stage-5 sandbox. Agent
        # mode never runs the tests — the framework is irrelevant to capture there.
        if schemas.CAPTURE_MODE == "replay" and not is_supported_profile(analysis.test_framework, analysis.framework_language):
            raise QualityReject(
                FailureCategory.QUALITY_GATE,
                f"unsupported framework: {analysis.test_framework}/{analysis.framework_language}",
                deep_analysis=analysis.model_dump(),
                deep_score=0,
            )

        # Agent mode hard gate: nothing bootable AND the suite targets a public site means the
        # sandbox will never have an app for the teacher to drive — guaranteed stage-5 reject,
        # cheaper to decide here from the static analysis.
        if schemas.CAPTURE_MODE == "agent" and analysis.tests_public_url and not can_self_host(analysis):
            raise QualityReject(
                FailureCategory.QUALITY_GATE,
                "public-URL-only suite with no bootable app; agent capture needs a local app",
                deep_analysis=analysis.model_dump(),
                deep_score=0,
            )

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
