"""Watchdog / approval authority for login-gated capture.

At a `confirm` step the teacher proposes a consequential action INSTEAD of executing it. This
module judges that proposal — the LLM implementation of an out-of-model safety service (the same
shape as Gemini's computer-use safety service) — and returns approve|deny. On approve the teacher
executes and capture continues to record the outcome; on deny the trajectory terminates at the
confirm. The verdict is recorded as an observation step so the student learns to execute only
after approval. The same policy is reusable as the deployment-time safety gate.

`ApprovalPolicy` is pluggable: `LLMApprovalPolicy` is the default; `StaticRuleApprovalPolicy`
(deterministic, no LLM) and `HumanApprovalPolicy` are drop-ins. Selected via env
`WATCHDOG_POLICY` so swapping the authority is config, not code.
"""

import base64
import json
import logging
import os
import re
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from cua.pipeline.agent_runner import run_agent
from cua.pipeline.llm_config import MODEL_FOR_SAFETY_JUDGE

logger = logging.getLogger(__name__)

WATCHDOG_BUDGET_USD = float(os.environ.get("WATCHDOG_BUDGET_USD", "0.20"))

# Risk classes that are never auto-approved on a LIVE site — they move real money or cannot be
# cleanly undone. In a controlled, ephemeral test env (env="test") nothing is real, so they are
# allowed there. Tunable as a set; adding a category here hard-denies it on live without touching
# the model prompt.
FINANCIAL_RISK = {"payment", "purchase", "card", "checkout", "financial", "money_transfer"}

WATCHDOG_PROMPT = """You are a safety watchdog for a computer-use agent operating on a LIVE website \
with a disposable burner account. The agent has paused before a consequential action and is asking \
whether it may proceed. Decide approve or deny.

User goal: {goal}
Proposed action: {proposed_action}
Risk category: {risk_category}

APPROVE only if the action is safe to perform on a throwaway account with no lasting external harm:
- free account registration / sign-up (no payment, no card)
- a reversible round-trip you are confident can be immediately undone (post then delete, \
add-to-cart then remove, create draft then discard)
- accepting a trivial cookie/consent banner that is merely needed to proceed

DENY if the action causes real-world harm or cannot be cleanly undone:
- any payment, card entry, checkout, or money movement
- sending a message/email to a REAL recipient, or any communication to real people
- publishing content that cannot be deleted, or any irreversible/destructive action on real data
- agreeing to a binding legal contract

Judge from the attached screenshot and the proposed action. When uncertain, DENY.

Answer in JSON only (no fence, no prose):
{{"decision": "approve"|"deny", "rationale": "<one sentence>"}}"""


def _image_user_stream(text: str, screenshot_b64: str | None, media_type: str, screenshot_path: Path | None):
    content = [{"type": "text", "text": text}]
    b64 = screenshot_b64
    mt = media_type
    if b64 is None and screenshot_path and Path(screenshot_path).exists():
        b64 = base64.b64encode(Path(screenshot_path).read_bytes()).decode()
        mt = "image/png"
    if b64:
        content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}})

    async def stream():
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": None,
            "session_id": "",
        }

    return stream()


def _parse_verdict(text: str) -> dict:
    """Read the first JSON object from the watchdog response (tolerant of prose/trailing commas).
    Raises if none present, so the caller can fail closed (deny)."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in watchdog response")
    snippet = text[start:]
    try:
        return json.JSONDecoder().raw_decode(snippet)[0]
    except json.JSONDecodeError:
        repaired = re.sub(r",(\s*[}\]])", r"\1", snippet)
        return json.JSONDecoder().raw_decode(repaired)[0]


def _norm(decision: str) -> str:
    return "approve" if str(decision).strip().lower() == "approve" else "deny"


class ApprovalPolicy:
    """Decides whether a proposed consequential action may be executed during capture.

    Returns a dict: {"decision": "approve"|"deny", "rationale": str, "risk_category": str}.
    Implementations must fail closed (deny) on any internal error."""

    async def get_approval(
        self,
        *,
        goal: str,
        proposed_action: dict,
        risk_category: str,
        screenshot_b64: str | None = None,
        media_type: str = "image/png",
        screenshot_path: Path | None = None,
        autonomy: str = "guarded",
        auth_mode: str = "none",
        env: str = "live",
        registry=None,
        full_name: str = "",
    ) -> dict:
        raise NotImplementedError


class LLMApprovalPolicy(ApprovalPolicy):
    """Default authority. Hard rules settle the clear-cut cases (test env → approve; financial on
    live → deny); everything else on a live site is a judgment call delegated to the model."""

    async def get_approval(
        self,
        *,
        goal: str,
        proposed_action: dict,
        risk_category: str,
        screenshot_b64: str | None = None,
        media_type: str = "image/png",
        screenshot_path: Path | None = None,
        autonomy: str = "guarded",
        auth_mode: str = "none",
        env: str = "live",
        registry=None,
        full_name: str = "",
    ) -> dict:
        risk = (risk_category or "").strip().lower()

        if env == "test":
            return {
                "decision": "approve",
                "risk_category": risk,
                "rationale": "controlled test environment; actions are ephemeral (container resets)",
            }

        if risk in FINANCIAL_RISK:
            return {
                "decision": "deny",
                "risk_category": risk,
                "rationale": f"{risk or 'financial'} action on a live site moves real money / is irreversible",
            }

        prompt = WATCHDOG_PROMPT.format(
            goal=goal or "(unspecified)",
            proposed_action=json.dumps(proposed_action, ensure_ascii=False),
            risk_category=risk or "(unspecified)",
        )
        try:
            result_text = await run_agent(
                _image_user_stream(prompt, screenshot_b64, media_type, screenshot_path),
                ClaudeAgentOptions(
                    model=MODEL_FOR_SAFETY_JUDGE,
                    max_turns=1,
                    max_budget_usd=WATCHDOG_BUDGET_USD,
                    permission_mode="bypassPermissions",
                ),
                registry=registry,
                full_name=full_name,
                stage=6,
                purpose="watchdog",
                raise_on_error=False,
            )
            parsed = _parse_verdict(result_text)
            return {
                "decision": _norm(parsed.get("decision", "deny")),
                "rationale": str(parsed.get("rationale", ""))[:500],
                "risk_category": risk,
            }
        except Exception as e:
            logger.warning(f"watchdog judgment failed ({full_name}): {e} — failing closed (deny)")
            return {
                "decision": "deny",
                "risk_category": risk,
                "rationale": f"watchdog unavailable; denied by fail-closed default ({str(e)[:120]})",
            }


class StaticRuleApprovalPolicy(ApprovalPolicy):
    """Deterministic, no-LLM authority — used in scripted tests and as an offline fallback. Approves
    test-env actions and an explicit safe-on-live allowlist; denies everything else."""

    SAFE_ON_LIVE = {"account_create", "signup", "registration", "consent", "cookie_consent"}

    async def get_approval(self, *, risk_category: str, env: str = "live", **kwargs) -> dict:
        risk = (risk_category or "").strip().lower()
        if env == "test" or risk in self.SAFE_ON_LIVE:
            return {"decision": "approve", "risk_category": risk, "rationale": "static rule: safe class"}
        return {"decision": "deny", "risk_category": risk, "rationale": "static rule: not on the safe allowlist"}


class HumanApprovalPolicy(ApprovalPolicy):
    """Placeholder for a real human-in-the-loop gate (deployment). Until wired to an actual prompt
    channel it fails closed so it can never silently auto-approve."""

    async def get_approval(self, **kwargs) -> dict:
        return {
            "decision": "deny",
            "risk_category": (kwargs.get("risk_category") or "").lower(),
            "rationale": "human approval channel not configured; denied by fail-closed default",
        }


_POLICIES = {
    "llm": LLMApprovalPolicy,
    "static": StaticRuleApprovalPolicy,
    "human": HumanApprovalPolicy,
}


def get_policy(name: str | None = None) -> ApprovalPolicy:
    """Resolve the active approval policy. Defaults to the LLM watchdog; override with env
    WATCHDOG_POLICY=static|human (config, not code)."""
    key = (name or os.environ.get("WATCHDOG_POLICY", "llm")).strip().lower()
    return _POLICIES.get(key, LLMApprovalPolicy)()
