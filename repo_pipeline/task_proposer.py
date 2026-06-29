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
PROPOSE_BUDGET_USD = float(os.environ.get("TASK_PROPOSE_BUDGET_USD", "2.0"))

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

FIRST judge whether the app is actually usable from the screenshot and elements: an app whose
content area shows an error ("An error occurred", stack traces, blank data regions needing a
dead backend) cannot host real tasks no matter how good its shell looks. Judging health wrong
in the optimistic direction wastes an expensive agent run per task; be strict.

Then propose UP TO {n_tasks} tasks — ONLY as many as the working parts of the app can
genuinely support. A broken app gets an empty tasks list. Requirements for each task:
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
- Tasks run IN ORDER against ONE app instance whose data PERSISTS between tasks (each task gets \
a fresh browser session, but the app keeps its state). Order tasks so early ones create data and \
later ones operate on accumulated data — finding, editing, reorganizing, or cleaning up existing \
items is HARDER and more valuable than creating from scratch. Phrase existing-data tasks \
generically ("archive every completed item", "rename the oldest entry") rather than naming \
artifacts from a specific earlier task, so they work against whatever state is actually present.
- difficulty: easy = obvious 3-6 action flow; medium = 6-12 actions or one non-obvious step; \
hard = 12+ actions, multiple stages, or requires discovering a non-obvious UI affordance.
- Skew hard: at most 2 easy tasks.
- Include 1-2 EXPLORATORY tasks (difficulty "hard"): state WHAT the user wants but not WHERE \
it lives, so the agent must hunt through menus, settings, or hover-revealed controls — e.g. \
"Find where this app lets you export your data, and export it" rather than "Open Settings > \
Export". Only propose goals the app genuinely supports (use the README/tests as evidence).
- Vary instruction PHRASING across tasks like real users would: some imperative and precise, \
some terse ("dark mode on, then find the oldest entry"), some conversational ("I need a list \
of everything I finished this week"). Never test/QA voice.
{avoid_block}
Output ONLY a JSON object (no markdown fence, no prose):
{{"app_health": {{"usable": <true|false>, "notes": "<one sentence: what works / what is broken>"}},
 "tasks": [{{"id": "<kebab-slug>", "instruction": "<the user goal>", "difficulty": "easy|medium|hard",
            "expected_actions": <int>, "action_kinds": ["click", "type", ...]}}, ...]}}"""


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


def render_avoid(avoid_instructions: list[str] | None) -> str:
    """A prompt block listing tasks already captured for this app, so the proposer covers NEW
    ground instead of regenerating them. Empty when nothing is covered yet (repo lane's default)."""
    if not avoid_instructions:
        return ""
    lines = "\n".join(f"- {a.strip()}" for a in avoid_instructions[:40] if a and a.strip())
    if not lines:
        return ""
    return ("\nALREADY-COVERED for this app — do NOT propose these or trivial rephrasings of them; "
            f"choose DIFFERENT features, flows, and parts of the app:\n{lines}\n")


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
        # The driver serves agent-facing screenshots as JPEG (SDK 1MB message-buffer limit).
        content.append(
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": screenshot_b64}}
        )

    async def stream():
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": None,
            "session_id": "",
        }

    return stream()


def parse_proposal(text: str) -> dict | None:
    """Extract {"app_health": {...}, "tasks": [...]} from the model output. Returns None when
    unparseable — the stage treats that as a retryable condition, not a content filter. A
    bare task array (older prompt shape) still parses, with health assumed usable."""
    obj_start = text.find("{")
    arr_start = text.find("[")
    raw = None
    if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
        try:
            raw = json.loads(text[obj_start:text.rfind("}") + 1])
        except json.JSONDecodeError:
            raw = None
    if raw is None and arr_start != -1:
        try:
            raw = {"app_health": {"usable": True, "notes": ""}, "tasks": json.loads(text[arr_start:text.rfind("]") + 1])}
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None

    tasks = []
    for i, item in enumerate(raw.get("tasks") or []):
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
    health = raw.get("app_health") if isinstance(raw.get("app_health"), dict) else {"usable": True, "notes": ""}
    return {"app_health": health, "tasks": tasks}


async def propose_tasks(
    repo: dict,
    snapshot: str,
    screenshot_b64: str | None,
    readme: str,
    test_hints: str,
    n_tasks: int = None,
    registry=None,
    affordances: dict | None = None,
    avoid_instructions: list[str] | None = None,
) -> dict | None:
    """Returns {"app_health": {"usable", "notes"}, "tasks": [...]} or None on SDK/parse failure
    (retryable). An unusable app comes back as usable=False with an empty task list — the
    caller rejects the repo without spending a single teacher call."""
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
        avoid_block=render_avoid(avoid_instructions),
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
        return None

    proposal = parse_proposal(result_text)
    if proposal is None:
        logger.warning(f"Task proposal unparseable for {repo.get('full_name')}: {result_text[:200]}")
    return proposal
