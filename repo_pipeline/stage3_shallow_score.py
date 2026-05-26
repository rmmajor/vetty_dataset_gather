import asyncio
import json
import logging
import os
import re
import subprocess

from claude_agent_sdk import ClaudeAgentOptions, AssistantMessage, ResultMessage, query

from repo_pipeline.registry import RepoRegistry
from repo_pipeline.schemas import RepoStatus, ShallowScores

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 7
SCORE_BUDGET_USD = float(os.environ.get("STAGE3_BUDGET_USD", "0.15"))

SCORING_PROMPT = """You are evaluating a GitHub repository for its suitability as a source of browser test trajectories (Playwright or Selenium). We need trajectories from PRODUCTION apps with complex UIs — not demos, tutorials, or practice repos.

Repository: {repo_name}
URL: {repo_url}
Stars: {stars}
Language: {language}
Description: {description}

File tree (top 2 levels):
{file_tree}

Dependencies file:
{deps_content}

Test directory listing:
{test_listing}

Sample test code (first 200 lines of up to 2 test files):
{sample_tests}

Score this repository on each dimension from 1-5:

1. PRODUCTION_COMPLEXITY (1-5): Is this a real application or a demo/tutorial?
   - 1: Tutorial, practice site, demo app, "learning playwright" repo
   - 2: Simple toy app built for testing (todo list, fake store)
   - 3: Small but real app with actual business logic
   - 4: Medium-complexity production app (CRM, CMS, e-commerce)
   - 5: Complex production app (admin dashboards, data-heavy UIs, workflow builders)

2. UI_DIFFICULTY (1-5): How confusing/complex are the UI elements being tested?
   - 1: Just basic clicks on obvious buttons and text inputs
   - 2: Simple forms with dropdowns and checkboxes
   - 3: Tables with sorting/filtering, multi-step forms
   - 4: Data grids, drag-and-drop, rich text editors, nested menus
   - 5: Spreadsheet-like UIs, workflow builders, calendar drag, canvas interactions

3. INTERACTION_RICHNESS (1-5): Do tests perform diverse UI actions?
   - 1: Only navigate and assert
   - 2: Click + type only
   - 3: Click, type, select, check
   - 4: Above + drag, hover, keyboard shortcuts, file upload
   - 5: Above + complex gestures, context menus, resize, multi-select

4. TEST_COUNT (1-5): How many distinct test files exist?
   - 1=1-2, 2=3-5, 3=6-10, 4=11-20, 5=20+

5. MULTI_PAGE_FLOWS (1-5): Do tests navigate across multiple pages/views?
   - 1: Single page/form
   - 2: 2-3 pages
   - 3: 4-6 pages with some navigation
   - 4: Full user journeys across many pages
   - 5: Complex multi-step workflows spanning entire app sections

6. RUNNABILITY (1-5): Can tests run without complex external services?
   - 5: Self-contained (docker-compose, built-in dev server, public URL)
   - 3: Needs some setup but straightforward
   - 1: Requires external paid services, proprietary backends

7. FRAMEWORK_CLARITY (1-5): Standard Playwright or Selenium project structure? Clear config, proper imports, organized test files?

Respond with ONLY a JSON object (no markdown, no explanation):
{{"production_complexity": N, "ui_difficulty": N, "interaction_richness": N, "test_count": N, "multi_page_flows": N, "runnability": N, "framework_clarity": N}}
"""


def _run_gh(args: list[str], timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _fetch_file_tree(repo_name: str, default_branch: str) -> str:
    raw = _run_gh(["api", f"repos/{repo_name}/git/trees/{default_branch}?recursive=1"])
    if not raw:
        return "(not available)"
    try:
        tree_data = json.loads(raw)
        paths = [item["path"] for item in tree_data.get("tree", [])]
        top_level_paths = [p for p in paths if p.count("/") <= 1]
        return "\n".join(top_level_paths[:100])
    except (json.JSONDecodeError, KeyError):
        return raw[:2000]


def _fetch_file_content(repo_name: str, file_path: str) -> str:
    raw = _run_gh(["api", f"repos/{repo_name}/contents/{file_path}", "--jq", ".content"])
    if not raw:
        return "(not available)"
    try:
        import base64
        return base64.b64decode(raw).decode("utf-8", errors="replace")
    except Exception:
        return raw[:2000]


def _find_test_files(tree_paths: list[str]) -> list[str]:
    test_patterns = [".test.ts", ".spec.ts", ".test.js", ".spec.js", "_test.py", "test_"]
    test_dirs = ["test/", "tests/", "e2e/", "spec/", "__tests__/"]
    test_files = []
    for path in tree_paths:
        in_test_dir = any(f"/{d}" in f"/{path}" or path.startswith(d) for d in test_dirs)
        has_test_ext = any(pat in path for pat in test_patterns)
        if in_test_dir or has_test_ext:
            test_files.append(path)
    return test_files


async def fetch_repo_context_via_agent(repo: dict) -> dict:
    repo_name = repo["full_name"]
    default_branch = repo.get("default_branch", "main")

    raw_tree = _run_gh(["api", f"repos/{repo_name}/git/trees/{default_branch}?recursive=1"])
    all_paths = []
    if raw_tree:
        try:
            tree_data = json.loads(raw_tree)
            all_paths = [item["path"] for item in tree_data.get("tree", [])]
        except (json.JSONDecodeError, KeyError):
            pass

    file_tree = "\n".join([p for p in all_paths if p.count("/") <= 1][:100])

    deps_content = "(not available)"
    for deps_file in ["package.json", "requirements.txt", "pyproject.toml"]:
        if deps_file in all_paths:
            deps_content = _fetch_file_content(repo_name, deps_file)
            if deps_content != "(not available)":
                deps_content = deps_content[:2000]
                break

    test_files = _find_test_files(all_paths)
    test_listing = "\n".join(test_files[:50]) if test_files else "(no test files found)"

    sample_tests = ""
    for test_file in test_files[:2]:
        content = _fetch_file_content(repo_name, test_file)
        if content != "(not available)":
            lines = content.split("\n")[:200]
            sample_tests += f"\n--- {test_file} ---\n" + "\n".join(lines) + "\n"
    if not sample_tests:
        sample_tests = "(no test samples available)"

    raw_combined = f"FILE_TREE:\n{file_tree}\n\nDEPS:\n{deps_content}\n\nTEST_LISTING:\n{test_listing}\n\nSAMPLE_TESTS:\n{sample_tests}"
    logger.info(f"Fetched context for {repo_name}: {len(raw_combined)} chars, {len(test_files)} test files found")

    return {
        "file_tree": file_tree,
        "deps_content": deps_content,
        "test_listing": test_listing,
        "sample_tests": sample_tests,
        "raw": raw_combined,
    }


async def score_single_repo(repo: dict, context: dict) -> ShallowScores | None:
    prompt = SCORING_PROMPT.format(
        repo_name=repo["full_name"],
        repo_url=repo["html_url"],
        stars=repo["stars"],
        language=repo.get("language", "Unknown"),
        description=repo.get("description", ""),
        file_tree=context.get("file_tree", "(not available)")[:3000],
        deps_content=context.get("deps_content", "(not available)")[:2000],
        test_listing=context.get("test_listing", "(not available)")[:1000],
        sample_tests=context.get("sample_tests", "(not available)")[:4000],
    )

    result_text = ""
    last_assistant_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            model="claude-sonnet-4-20250514",
            max_turns=3,
            max_budget_usd=SCORE_BUDGET_USD,
            permission_mode="bypassPermissions",
        ),
    ):
        if isinstance(message, AssistantMessage) and hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    last_assistant_text = block.text
        if isinstance(message, ResultMessage):
            if message.subtype == "success":
                result_text = message.result or ""
            else:
                logger.warning(f"Scoring for {repo['full_name']} ended with: {message.subtype}")

    if not result_text and last_assistant_text:
        logger.info(f"Using last assistant text as scoring result for {repo['full_name']}")
        result_text = last_assistant_text

    try:
        cleaned = result_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        try:
            scores_dict = json.loads(cleaned)
        except json.JSONDecodeError:
            code_block_match = re.search(r"```(?:json)?\s*\n(.*?)```", cleaned, re.DOTALL)
            if code_block_match:
                scores_dict = json.loads(code_block_match.group(1).strip())
            else:
                brace_match = re.search(r"\{[^{}]*\}", cleaned)
                if brace_match:
                    scores_dict = json.loads(brace_match.group(0))
                else:
                    raise
        return ShallowScores(**scores_dict)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse scores for {repo['full_name']}: {e}")
        logger.debug(f"Raw response: {result_text[:500]}")
        return None


async def process_repo(registry: RepoRegistry, repo: dict):
    repo_name = repo["full_name"]

    if repo.get("current_stage", 0) >= 3:
        return

    logger.info(f"Scoring: {repo_name}")

    try:
        context = await fetch_repo_context_via_agent(repo)
        scores = await score_single_repo(repo, context)
    except Exception as e:
        logger.error(f"Error scoring {repo_name}: {e}")
        registry.reject(repo_name, stage=3, reason=f"scoring error: {str(e)[:200]}")
        return

    if scores is None:
        registry.reject(repo_name, stage=3, reason="failed to parse scores")
        return

    total = scores.total
    logger.info(f"  {repo_name}: {total}/35 — {scores.model_dump()}")

    if scores.production_complexity <= 1 and total < SCORE_THRESHOLD:
        registry.reject(repo_name, stage=3, reason=f"demo/tutorial repo ({total}/35)")
        return

    registry.update_status(
        repo_name,
        RepoStatus.SCORED.value,
        stage=3,
        shallow_scores=scores.model_dump(),
        shallow_total=total,
    )


async def run_shallow_score(
    registry: RepoRegistry,
    limit: int = None,
    workers: int = 5,
    budget_remaining: float = None,
):
    repos = registry.get_repos_for_stage(3, limit=limit)
    logger.info(f"Shallow scoring {len(repos)} repos with {workers} workers")

    semaphore = asyncio.Semaphore(workers)
    passed = 0
    rejected = 0
    cost_estimate = 0.0

    async def process_with_semaphore(repo):
        nonlocal passed, rejected, cost_estimate

        if budget_remaining is not None and cost_estimate > budget_remaining:
            logger.warning("Budget limit reached, stopping")
            return

        async with semaphore:
            await process_repo(registry, repo)
            cost_estimate += 0.20  # rough estimate per repo

        repo_after = registry.get_repo(repo["full_name"])
        if repo_after and repo_after["status"] == RepoStatus.SCORED.value:
            passed += 1
        else:
            rejected += 1

    await asyncio.gather(*[process_with_semaphore(repo) for repo in repos])

    logger.info(f"Shallow scoring complete: {passed} passed, {rejected} rejected")
    return {"processed": len(repos), "passed": passed, "rejected": rejected}
