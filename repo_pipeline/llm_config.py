"""Centralized model selection for every Claude SDK call site in the pipeline.

All call sites import a NAMED purpose (`MODEL_FOR_REASONING`, `MODEL_FOR_GROUNDED_GENERATION`,
`MODEL_FOR_CLASSIFICATION`) rather than a model string. Swapping the model for one purpose
(or all of them) is a one-line edit here; the call sites never change.

The three purposes capture intent, not capability — if a future re-split sends classification
to a cheaper model again, the constant changes here and every classifier follows. No grep
across stage files, no hardcoded model names anywhere else in `repo_pipeline/`.
"""

MODEL_FOR_REASONING = "claude-sonnet-4-6"
"""Deep multi-turn reasoning over code/repos. Used by stage 3 shallow scoring and stage 4
deep analysis. These calls run with higher `max_turns` and a larger budget per call."""

MODEL_FOR_GROUNDED_GENERATION = "claude-sonnet-4-6"
"""Single-turn generation grounded in concrete inputs — a screenshot for `<think>` blocks,
test source + repo context for trajectory `instruction`s. Sonnet 4.6's reasoning quality
matters here because the output IS the SFT supervision signal."""

MODEL_FOR_CLASSIFICATION = "claude-sonnet-4-6"
"""Single-label categorization (task-type tagging). Tight output schema, low-token in
and out. Kept on Sonnet for consistency with the rest of the pipeline; if cost becomes
a concern, this is the safest constant to drop to a smaller model."""
