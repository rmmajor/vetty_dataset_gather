"""Propose user-intent tasks for an app booted in the sandbox.

One Claude call per repo, grounded in what the app actually is: the homepage screenshot +
interactive-element snapshot from the live app, the README, and the repo's test sources as
feature hints (tests document what the app can do — they are hints here, not the generator).

The output is the dataset's difficulty/diversity lever: tasks must be phrased as genuine user
goals (never QA assertions), skew long-horizon (most need 8+ actions), and mix action kinds
(typing/forms mandatory, scroll/drag where the UI affords them). The teacher agent in
agent_capture.py then pursues each task in the live app.
"""
import json
import logging
import os

from claude_agent_sdk import ClaudeAgentOptions

from repo_pipeline.agent_runner import run_agent
from repo_pipeline.llm_config import MODEL_FOR_REASONING
from repo_pipeline.sdk_runner import UsageLimitReached

logger = logging.getLogger(__name__)

TASKS_PER_REPO = int(os.environ.get("AGENT_TASKS_PER_REPO", "8"))
PROPOSE_BUDGET_USD = float(os.environ.get("TASK_PROPOSE_BUDGET_USD", "0.50"))

PROPOSE_PROMPT = """You are designing realistic user tasks for a computer-use agent to perform in a \
live web application. The tasks become training data for a model that must learn LONG-HORIZON \
planning and NON-OBVIOUS UI decisions, so the quality bar is: would a demanding human user \
actually want this done, and does doing it require genuinely navigating the app?

The app is running locally. You are given its homepage screenshot (attached image), the \
interactive elements visible on it, the repository README, and excerpts from the repo's own \
E2E tests (treat tests ONLY as hints about which features exist — never copy a test scenario \
verbatim, and never phrase a task like a test).

App repository: {repo_name} — {repo_description}

Homepage interactive elements:
{snapshot}

README excerpt:
{readme}

Test excerpts (feature hints):
{test_hints}

Measured UI affordances on the live page (counts of real elements — trust these over guesses):
{affordances_block}

Propose exactly {n_tasks} tasks. Requirements:
- Each instruction is a USER GOAL in imperative voice ("Create a project called X, add two \
tasks to it, and mark the first one done"), 8-30 words. NEVER use verify/test/assert/check \
phrasing. NEVER mention selectors, the DOM, or testing.
- At least {n_long} tasks must plausibly require 8 or more user actions (multi-page or \
multi-stage flows: create-then-edit, fill-then-review, search-then-compare, configure-then-use).
- At least 2 tasks must require typing into forms or fields.
{affordance_quotas}- Include scroll-dependent or drag-dependent tasks IF the visible UI affords them (long lists, \
kanban boards, sliders, reorderable items); otherwise do not invent them.
- Tasks must be achievable with what this app visibly offers — no external accounts, no email \
verification, no payment with real cards. If the app has a login/demo account visible on the \
page or in the README, tasks may use it.
- Make tasks DIVERSE from each other: different features, different parts of the app, different \
action mixes. Concrete values (names, amounts, dates) should be specific and varied.
- difficulty: easy = obvious 3-6 action flow; medium = 6-12 actions or one non-obvious step; \
hard = 12+ actions, multiple stages, or requires discovering a non-obvious UI affordance.
- Skew hard: at most 2 easy tasks.

Output ONLY a JSON array (no markdown fence, no prose):
[{{"id": "<kebab-slug>", "instruction": "<the user goal>", "difficulty": "easy|medium|hard",
   "expected_actions": <int>, "action_kinds": ["click", "type", ...]}}, ...]"""


# Affordance count -> a hard quota line in the proposal prompt. Generic: a new sophistication
# signal is a new entry here (and a counter in the driver's /affordances), not new prompt code.
_AFFORDANCE_QUOTAS = {
    "draggables": "- The page has {n} draggable elements: at least 1 task MUST require dragging "
                  "(reorder, move between columns, drop onto a target).\n",
    "sliders": "- The page has {n} sliders: at least 1 task MUST involve setting a slider to a "
               "specific value by dragging it.\n",
    "rich_text_editors": "- The page has {n} rich-text editors: at least 1 task MUST involve "
                         "composing or formatting multi-line content in one.\n",
    "date_inputs": "- The page has {n} date/time inputs: at least 1 task MUST involve picking "
                   "specific dates or times.\n",
    "canvases": "- The page has {n} canvas surfaces: at least 1 task MUST involve drawing or "
                "direct manipulation on the canvas.\n",
    "autocompletes": "- The page has {n} autocomplete fields: at least 1 task MUST involve "
                     "typing a partial value and selecting a suggestion.\n",
    "multi_selects": "- The page has {n} multi-select inputs: at least 1 task MUST involve "
                     "selecting several options in one.\n",
}


def render_affordances(affordances: dict | None) -> tuple[str, str]:
    """(human-readable block for the prompt, hard quota lines) from the driver's counts."""
    if not affordances:
        return "(not measured)", ""
    present = {k: v for k, v in affordances.items() if v}
    block = ", ".join(f"{k}={v}" for k, v in present.items()) or "(none beyond basic click/type targets)"
    quotas = "".join(
        template.format(n=present[key]) for key, template in _AFFORDANCE_QUOTAS.items() if key in present
    )
    return block, quotas


def _image_prompt_stream(text: str, screenshot_b64: str | None):
    """Single user message with optional attached screenshot, in SDK streaming-input form."""
    content = [{"type": "text", "text": text}]
    if screenshot_b64:
        content.append(
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}}
        )

    async def stream():
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": None,
            "session_id": "",
        }

    return stream()


def parse_tasks(text: str) -> list[dict]:
    """Extract the JSON task array from the model output. Returns [] when unparseable —
    the stage treats that as a retryable condition, not a content filter."""
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end <= start:
        return []
    try:
        raw = json.loads(text[start:end])
    except json.JSONDecodeError:
        return []
    tasks = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not (item.get("instruction") or "").strip():
            continue
        slug = (item.get("id") or f"task-{i + 1}").strip().replace(" ", "-")[:60]
        tasks.append({
            "id": slug,
            "instruction": item["instruction"].strip(),
            "difficulty": item.get("difficulty", "medium"),
            "expected_actions": int(item.get("expected_actions", 8) or 8),
            "action_kinds": item.get("action_kinds", []),
        })
    return tasks


async def propose_tasks(
    repo: dict,
    snapshot: str,
    screenshot_b64: str | None,
    readme: str,
    test_hints: str,
    n_tasks: int = None,
    registry=None,
    affordances: dict | None = None,
) -> list[dict]:
    n_tasks = n_tasks or TASKS_PER_REPO
    affordances_block, affordance_quotas = render_affordances(affordances)
    prompt_text = PROPOSE_PROMPT.format(
        repo_name=repo.get("full_name", ""),
        repo_description=(repo.get("description") or "").strip() or "(no description)",
        snapshot=(snapshot or "(unavailable)")[:6000],
        readme=(readme or "(no README)")[:4000],
        test_hints=(test_hints or "(no test sources available)")[:5000],
        n_tasks=n_tasks,
        n_long=max(1, int(n_tasks * 0.6)),
        affordances_block=affordances_block,
        affordance_quotas=affordance_quotas,
    )

    try:
        result_text = await run_agent(
            _image_prompt_stream(prompt_text, screenshot_b64),
            ClaudeAgentOptions(
                model=MODEL_FOR_REASONING,
                max_turns=1,
                max_budget_usd=PROPOSE_BUDGET_USD,
                permission_mode="bypassPermissions",
            ),
            registry=registry,
            full_name=repo.get("full_name", ""),
            stage=6,
            purpose="propose_tasks",
            raise_on_error=False,
        )
    except UsageLimitReached:
        raise
    except Exception as e:
        logger.warning(f"Task proposal failed for {repo.get('full_name')}: {e}")
        return []

    tasks = parse_tasks(result_text)
    if not tasks:
        logger.warning(f"Task proposal unparseable for {repo.get('full_name')}: {result_text[:200]}")
    return tasks
