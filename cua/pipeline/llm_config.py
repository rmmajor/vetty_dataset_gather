"""Centralized model selection for every Claude SDK call site in the pipeline.

All call sites import a NAMED purpose (`MODEL_FOR_REASONING`, `MODEL_FOR_GROUNDED_GENERATION`,
`MODEL_FOR_CLASSIFICATION`) rather than a model string. Swapping the model for one purpose
(or all of them) is a one-line edit here; the call sites never change.

The purposes capture intent, not capability — if a future re-split sends classification
to a cheaper model again, the constant changes here and every classifier follows. No grep
across stage files, no hardcoded model names anywhere else in `cua/pipeline/`.

Model IDs come from `LATEST_MODEL_BY_CLASS`, one entry per capability class. When a new
generation ships (e.g. a future Sonnet), bump the class entry and every purpose on that
class follows.
"""

LATEST_MODEL_BY_CLASS = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
}
"""Latest model ID per capability class. The single place a new model generation lands."""

MODEL_FOR_REASONING = LATEST_MODEL_BY_CLASS["sonnet"]
"""Deep multi-turn reasoning over code/repos. Used by stage 3 shallow scoring and stage 4
deep analysis. These calls run with higher `max_turns` and a larger budget per call."""

MODEL_FOR_GROUNDED_GENERATION = LATEST_MODEL_BY_CLASS["sonnet"]
"""Single-turn generation grounded in concrete inputs — a screenshot for `<think>` blocks,
test source + repo context for trajectory `instruction`s. Sonnet-class reasoning quality
matters here because the output IS the SFT supervision signal."""

MODEL_FOR_TEACHER_AGENT = LATEST_MODEL_BY_CLASS["sonnet"]
"""The teacher driving the browser in agent capture (stage 6 agent mode). Multi-turn,
tool-heavy, and the source of both the recorded actions and the per-step thoughts — the
highest-leverage model choice in the pipeline. Point at LATEST_MODEL_BY_CLASS["opus"]
for harder task batches."""

MODEL_FOR_CLASSIFICATION = LATEST_MODEL_BY_CLASS["sonnet"]
"""Single-label categorization (task-type tagging). Tight output schema, low-token in
and out. Kept on Sonnet for consistency with the rest of the pipeline; if cost becomes
a concern, this is the safest constant to drop to LATEST_MODEL_BY_CLASS["haiku"]."""

MODEL_FOR_SAFETY_JUDGE = LATEST_MODEL_BY_CLASS["sonnet"]
"""The watchdog / approval authority that judges a proposed sensitive action before it is
executed at a `confirm` step (login-gated capture). Sees the goal, the screenshot, the
proposed action, its risk category, and the autonomy/auth/env context, and returns
approve|deny with a rationale. Used at capture time and reusable as the deployment-time
safety service. Kept on Sonnet for consistency."""
