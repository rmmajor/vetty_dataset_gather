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
{avoid_block}{auth_flow_block}{style_block}
Output ONLY a JSON object (no markdown fence, no prose):
{{"app_health": {{"usable": <true|false>, "notes": "<one sentence: what works / what is broken>"}},
 "tasks": [{{"id": "<kebab-slug>", "instruction": "<the user goal>", "difficulty": "easy|medium|hard",
            "expected_actions": <int>, "action_kinds": ["click", "type", ...]}}, ...]}}"""


# Opt-in block (auth-flow capture): lifts the "no email verification" restriction, asks for the
# account/auth flows the agent can now complete end-to-end (it has a disposable inbox it opens in
# the browser), and has the proposer CLASSIFY each task's auth need so login appears organically —
# a minority of trajectories — rather than as a forced prefix on every one.
_AUTH_FLOW_BLOCK = """
EMAIL-BASED FLOWS ARE ENABLED for this run. The agent has its own disposable email address and a \
webmail inbox it can open in the browser. OVERRIDING the "no email verification" guidance above:

Tag EVERY task with an "auth" field describing how it relates to signing in:
- "none": the task works without being signed in.
- "account": the task REQUIRES being signed in to do it (an ordinary in-app action behind login).
- "login": signing in is itself the GOAL of the task.
- "register": creating a new account (and verifying the email) is the GOAL of the task.

Most tasks should be "none" or "account". Keep explicit "login"/"register" tasks a SMALL minority \
(the agent will be started already-signed-in for most "account" tasks, so don't make every task a \
login). IF — and only if — this app visibly supports accounts (judge from the page/README), ALSO \
include, ordered FIRST:
- ONE "register" task: "Register a new account and verify the email address." (hard, 10+ actions).
- If there is a forgot/reset-password link, ONE "login" task: "Reset the account password using the \
link emailed to you, then sign in." (hard).
Do NOT invent account flows if the app has no visible sign-up or login.
"""

_AUTH_VALUES = {"none", "account", "login", "register"}


# Opt-in block (search-goal capture): steers proposals toward constrained info-seeking journeys —
# the Online-Mind2Web-shaped distribution the action-UI corpus lacks. Difficulty must live in the
# CONSTRAINT SET (dates, prices, filters, superlatives), never in procedural phrasing.
_SEARCH_GOAL_BLOCK = """
THIS SITE IS FOR INFO-SEEKING / CONSTRAINED-SEARCH TASKS. Override the long-horizon-app guidance
above where it conflicts: most tasks must be goal-phrased SEARCH JOURNEYS whose difficulty lives in
the constraint set, not the route:
- multi-constraint search: concrete dates, price ceilings, ratings, features, and superlatives
  ("cheapest", "earliest", "highest-rated", "shortest") that force comparing results;
- filter + sort + compare: apply 2-3 real filters, sort, and end with the single matching result
  open or visibly selected;
- multi-hop lookups: find fact X, then use X to locate Y, ending with Y on screen.
State the GOAL and its constraints only — never the route. Every task still ends in a visually
checkable state (the matching result/quote/details screen). No purchases, no checkout past the
quote/summary screen, no account creation.
"""

# Opt-in block (qa_verify capture): test-objective voice, change-then-re-check tasks — the shape the goal-phrased corpus lacks.
_QA_VERIFY_BLOCK = """
THIS SITE IS FOR QA-VERIFICATION TASKS. OVERRIDING the "never test/QA voice" rules above for this
site only: every task is a TEST OBJECTIVE, not a user goal. Phrase each instruction as a short
assertion-style title sentence followed by ONE "Verify ..." sentence stating what should hold and
how it shows on screen (two sentences, up to ~40 words in total), e.g. "Toggling Email
notifications in Settings persists after save and reload. Verify unchecking Email notifications
and saving retains that state on page reload." Never a numbered step list, never selectors, never
the DOM.
Every task is inherently TWO-PHASE: (1) change something that PERSISTS — create, rename, toggle,
reorder, delete, or change a setting — then (2) confirm it persisted by re-opening the page or
reloading it, never by trusting a toast or inline confirmation, and end on the screen where the
expected state is visible.
Quotas: at least ONE task whose evidence sits on page 2 or later of a paginated list or behind a
"load more" control; at least ONE task that signs out and signs back in — the account is already
signed in at the start and the session is restorable, so phrase it as "sign out, sign back in, and
confirm X is still there". Keep tasks idempotent-ish and reversible where possible (restoring the
original state at the end is optional, not required). Difficulty mostly medium/hard. Propose 6-10
tasks (this range replaces the count above).
"""

# Opt-in per-site task styles (site config key "task_style"); a new style is a new entry here.
_STYLE_BLOCKS = {"search": _SEARCH_GOAL_BLOCK, "qa_verify": _QA_VERIFY_BLOCK}


# Opt-in block (known-account capture): the app already has a working, pre-verified login and there
# is NO readable inbox in the sandbox (its mail leaves to a cloud provider, or it's a demo account).
# So the proposer must NOT generate registration / email-verification / reset-via-email / invite-via-
# email tasks — they can never be completed without an inbox — and should focus on signed-in usage.
_KNOWN_ACCOUNT_BLOCK = """
YOU ALREADY HAVE A WORKING ACCOUNT for this app and can sign in with it. There is NO email inbox \
available in this environment, so any flow that depends on reading an email CANNOT be completed.

Tag EVERY task with an "auth" field:
- "none": works without being signed in.
- "account": requires being signed in (an ordinary in-app action behind the login).
- "login": signing in with the existing account is itself the GOAL of the task.

Rules:
- Most tasks should be "account". You MAY include ONE "login" task (sign in with the account).
- Do NOT propose account REGISTRATION, email VERIFICATION, password RESET-via-email, or \
email-INVITATION tasks, or anything else that needs to read an email — there is no inbox.
- Focus on what a signed-in user does inside the app (create/edit/find/configure/manage).
"""


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
        auth = (item.get("auth") or "none").strip().lower()
        if auth not in _AUTH_VALUES:
            auth = "none"
        tasks.append({
            "id": slug,
            "instruction": item["instruction"].strip(),
            "difficulty": item.get("difficulty", "medium"),
            "expected_actions": int(item.get("expected_actions", 8) or 8),
            "action_kinds": item.get("action_kinds", []),
            "auth": auth,
        })
    # register/login tasks must run FIRST: they MINT the account whose captured session the later
    # "account" tasks reuse (auth_mode=pre_session). Stable sort preserves the proposer's order
    # within each bucket.
    tasks.sort(key=lambda t: {"register": 0, "login": 1}.get(t["auth"], 2))
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
    auth_flow: bool = False,
    inbox: bool = True,
    task_style: str = "",
) -> dict | None:
    """Returns {"app_health": {"usable", "notes"}, "tasks": [...]} or None on SDK/parse failure
    (retryable). An unusable app comes back as usable=False with an empty task list — the
    caller rejects the repo without spending a single teacher call. `auth_flow=True` enables auth
    task classification; `inbox=False` (a known pre-verified account, no readable sandbox inbox)
    switches to the known-account block so no email-dependent tasks are proposed."""
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
        auth_flow_block=("" if not auth_flow else (_AUTH_FLOW_BLOCK if inbox else _KNOWN_ACCOUNT_BLOCK)),
        style_block=_STYLE_BLOCKS.get(task_style, ""),
    )

    try:
        result_text = await run_agent(
            _image_prompt_stream(prompt_text, screenshot_b64),
            ClaudeAgentOptions(
                model=MODEL_FOR_REASONING,
                # A long prompt (style/auth blocks) can push the JSON past one turn; the budget cap,
                # not the turn count, is what bounds cost here.
                max_turns=6,
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
