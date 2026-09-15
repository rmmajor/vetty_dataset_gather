import asyncio
import json
import logging

from claude_agent_sdk import ClaudeAgentOptions

from cua.pipeline import schemas
from cua.pipeline._processing import run_repo
from cua.pipeline.agent_runner import run_agent
from cua.pipeline.frameworks import all_interactive_actions, is_supported_profile
from cua.pipeline.llm_config import MODEL_FOR_REASONING
from cua.pipeline.logging_utils import StageProgress, repo_logger
from cua.pipeline.registry import RepoRegistry
from cua.pipeline.sandbox import DockerSandbox, build_sandbox_mcp_server
from cua.pipeline.schemas import (
    DeepAnalysis,
    FailureCategory,
    QualityReject,
    RepoStatus,
    RetryableFailure,
)
from cua.pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from cua.settings import PIPELINE

logger = logging.getLogger(__name__)

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

6. **Estimate trajectory count**: How many usable, distinct browser-automation trajectories could this repo realistically yield? In AGENT capture (the default) a trajectory is a multi-step USER TASK performed in the repo's booted app — so base the estimate on the app's interactive surface (distinct features, pages, forms, and end-to-end flows a user could complete), NOT on the number of existing test cases. A repo with many unit tests but a trivial or non-existent UI yields ~0; a rich app with no tests can still yield many.

7. **Fit verdict** (`good_fit`): Decide whether this repo is a GOOD FIT for collecting diverse, rich, multi-step web-task trajectories to train a computer-use agent. Set `good_fit: false` (with a short `veto_reason`) if it is any of: a toy / boilerplate / starter template; a tutorial, course, or "learning" project; a test-suite-only repo or a testing library / framework / tool (its own code is what's under test — there is no real product UI); a repo whose UI is trivial; one that needs an external backend / database / paid service / browser wallet the sandbox cannot provide; or one whose functionality is entirely behind a third-party login (GitHub/Google OAuth, etc.) that cannot be completed in an isolated sandbox. Otherwise `good_fit: true`. Be decisive — this verdict vetoes the repo before any expensive boot/capture is spent on it.

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
  "good_fit": <bool>,
  "veto_reason": "<short reason if good_fit is false, else empty string>",
  "notes": "<brief notes about anything notable>"
}}
```

Output ONLY the JSON at the end of your analysis.
"""


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


def fit_veto(analysis: DeepAnalysis, shallow_scores: dict | None) -> str | None:
    """Return a rejection reason if this repo is an obvious bad fit for the trajectory dataset,
    else None. The SAME predicate guards stage 4 going forward and the retroactive prune_unfit
    cleanup, so a toy is rejected by identical logic whether analyzed now or before the veto
    existed. Generic + env-tunable: every rule reads a threshold; none hardcodes a repo name.

    The deterministic rules are a cheap, unambiguous backstop; the model's holistic ``good_fit``
    verdict (filled by the stage-4 prompt, or by judge_fit over a stored analysis) is the main gate
    and catches boilerplates / course repos / login-walled apps that the raw numbers rate highly."""
    if schemas.CAPTURE_MODE == "agent":
        if PIPELINE.stage4_min_est_trajectories > 0 and (
            analysis.estimated_trajectories < PIPELINE.stage4_min_est_trajectories
        ):
            return (
                f"only ~{analysis.estimated_trajectories} usable trajectories "
                f"(< {PIPELINE.stage4_min_est_trajectories}) — too little for a capture-worthy app"
            )
        if analysis.needs_external_backend and not can_self_host(analysis):
            return "needs an external backend/service the sandbox cannot provide"

    production_complexity = (shallow_scores or {}).get("production_complexity")
    if isinstance(production_complexity, int) and production_complexity <= PIPELINE.stage4_toy_complexity_max:
        return f"toy/tutorial (production_complexity {production_complexity} <= {PIPELINE.stage4_toy_complexity_max})"

    if not analysis.good_fit:
        return f"not a good fit — {analysis.veto_reason or 'model vetoed'}"

    return None


FIT_VERDICT_PROMPT = """You are deciding whether a GitHub repository belongs in a dataset of diverse, rich, multi-step web-browsing task trajectories used to train a computer-use agent.

A GOOD fit is a self-containable web APPLICATION that boots locally and offers a rich, multi-section interactive UI where a user can perform many distinct multi-step tasks (create / edit / configure / transact / navigate).

A BAD fit (good_fit=false) is any of: a toy / boilerplate / starter template; a tutorial, course, or "learning" project; a test-suite-only repo or a testing library / framework / tool (its own code is what's under test — there is no real product UI); a repo whose UI is trivial; one that needs an external backend / database / paid service / browser wallet the sandbox cannot provide; or one whose functionality is entirely behind a third-party login (GitHub/Google OAuth, etc.) that cannot be completed in an isolated sandbox.

Repository: {repo_name}
Description: {description}
Shallow scores (1-5): {shallow_scores_summary}
Static analysis already gathered:
{analysis_summary}

Decide based on the evidence above. Output ONLY this JSON, nothing else:
{{"good_fit": <bool>, "veto_reason": "<short reason if good_fit is false, else empty string>"}}
"""


async def judge_fit(
    registry: RepoRegistry, repo: dict, analysis: DeepAnalysis, shallow_scores: dict | None
) -> tuple[bool, str]:
    """Holistic, text-only 'is this a good fit?' verdict over an ALREADY-computed analysis — no
    clone, no sandbox. Renders the same judgment the stage-4 prompt now produces inline, for repos
    analyzed before that verdict existed (used by prune_unfit). Parse failure → (True, "") so a
    flaky call never wrongly prunes; the deterministic fit_veto rules still apply on top."""
    summary = ", ".join(f"{k}={v}" for k, v in (shallow_scores or {}).items()) or "(none)"
    prompt = FIT_VERDICT_PROMPT.format(
        repo_name=repo["full_name"],
        description=repo.get("description") or "(none)",
        shallow_scores_summary=summary,
        analysis_summary=json.dumps(analysis.model_dump(), indent=2),
    )
    text = await run_agent(
        prompt,
        ClaudeAgentOptions(
            model=MODEL_FOR_REASONING,
            max_turns=1,
            permission_mode="bypassPermissions",
        ),
        registry=registry,
        full_name=repo["full_name"],
        stage=4,
        purpose="fit_verdict",
    )
    try:
        start = text.rfind("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
        return bool(data.get("good_fit", True)), str(data.get("veto_reason", "") or "")
    except (json.JSONDecodeError, ValueError, TypeError):
        return True, ""


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
        shallow_scores_dict = {}
        if shallow_scores_raw:
            shallow_scores_dict = (
                json.loads(shallow_scores_raw) if isinstance(shallow_scores_raw, str) else shallow_scores_raw
            )
            shallow_scores_summary = ", ".join(f"{k}={v}" for k, v in shallow_scores_dict.items())
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
                max_budget_usd=PIPELINE.stage4_budget_usd,
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
        if schemas.CAPTURE_MODE == "replay" and not is_supported_profile(
            analysis.test_framework, analysis.framework_language
        ):
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

        # Fit veto — the decisive "is this worth capturing at all?" gate. Runs before the numeric
        # threshold so an obvious toy/boilerplate/login-wall is rejected here (cheaply) instead of
        # surviving to a stage-5 boot or stage-6 capture. The model's good_fit verdict came back in
        # the analysis JSON; fit_veto folds it together with the deterministic backstops.
        if reason := fit_veto(analysis, shallow_scores_dict):
            raise QualityReject(
                FailureCategory.QUALITY_GATE,
                f"fit veto: {reason}",
                deep_analysis=analysis.model_dump(),
                deep_score=deep_score,
            )

        if deep_score >= PIPELINE.stage4_deep_score_threshold:
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
                f"deep score {deep_score}/{PIPELINE.stage4_deep_score_threshold} "
                f"(tests: {analysis.test_case_count}, self_contained: {analysis.is_self_contained})",
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
    effective_limit = min(limit, PIPELINE.stage4_top_k_per_pass) if limit else PIPELINE.stage4_top_k_per_pass
    repos = registry.get_repos_for_stage(4, limit=effective_limit)
    logger.info(
        f"Deep analyzing {len(repos)} repos with {workers} workers (top-{effective_limit} by richness ordering)"
    )

    semaphore = asyncio.Semaphore(workers)
    progress = StageProgress(4, len(repos))
    cost_estimate = 0.0

    async def process_with_semaphore(repo):
        nonlocal cost_estimate
        # Guards live INSIDE the semaphore: all repo coroutines are gathered up front, so a check
        # placed before `async with` is evaluated by every coroutine while cost_estimate is still
        # ~0 — they all pass it, then queue past the gate, and the budget cap never fires. Re-checked
        # here under the semaphore, the cap sees the cost accrued by already-finished repos and stops.
        async with semaphore:
            if rate_limit_tracker.limit_reached():
                return
            if budget_remaining is not None and cost_estimate > budget_remaining:
                logger.warning("Budget limit reached, stopping")
                return
            try:
                await run_repo(4, repo, registry, analyze_single_repo, progress)
            except UsageLimitReached:
                return
            cost_estimate += 0.20

    await asyncio.gather(*[process_with_semaphore(repo) for repo in repos])

    return progress.summary()
