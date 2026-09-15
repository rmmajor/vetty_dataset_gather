"""The Claude teacher that records trajectories: the tools it acts through, the policy it is given, its run loop and the verifier that judges the result."""

from cua.pipeline.teacher.policy import (
    SECRET_KEY_DESC,
    TEACHER_PROMPT,
    TEACHER_STYLE_BLOCKS,
    render_identity_block,
    render_mode_block,
    render_policy_block,
)
from cua.pipeline.teacher.run import run_teacher_task
from cua.pipeline.teacher.tools import build_driver_mcp_server
from cua.pipeline.teacher.verify import (
    GUARDED_VERIFY_PROMPT,
    SESSION_PROBE_PROMPT,
    VERIFIER_UNAVAILABLE_REASON,
    VERIFY_PROMPT,
    VERIFY_STYLE_RULES,
    degenerate_loop_evidence,
    session_authenticated,
    step_summary,
    verdict_from_parsed,
    verify_trajectory,
)

__all__ = [
    "GUARDED_VERIFY_PROMPT",
    "SECRET_KEY_DESC",
    "SESSION_PROBE_PROMPT",
    "TEACHER_PROMPT",
    "TEACHER_STYLE_BLOCKS",
    "VERIFIER_UNAVAILABLE_REASON",
    "VERIFY_PROMPT",
    "VERIFY_STYLE_RULES",
    "build_driver_mcp_server",
    "degenerate_loop_evidence",
    "render_identity_block",
    "render_mode_block",
    "render_policy_block",
    "run_teacher_task",
    "session_authenticated",
    "step_summary",
    "verdict_from_parsed",
    "verify_trajectory",
]
