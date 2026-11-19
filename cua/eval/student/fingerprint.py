"""What must match for two runs to be comparable, plus the diagnosis for a run measured in the wrong space."""

import subprocess

from cua.eval.blocks import BLOCK_SIGNATURES, BLOCK_SIGNATURES_VERSION
from cua.eval.student.config import EvalConfig
from cua.eval.student.grammar import ACTION_VERBS
from cua.pipeline.llm_config import MODEL_FOR_REASONING
from cua.prompt.format import HISTORY_WINDOW, PREV_IMAGE_SIZE, PROMPT_FORMAT_VERSION
from cua.settings import EVAL
from cua.student.executor import COORD_SPACE_RANGE, COORD_SPACE_SUSPECT

# A before/after pair MAY differ here: an sft-v1/v2 checkpoint answers in viewport_abs, each side in its own space.
EXPECTED_TO_DIFFER_KEYS = ("coord_space",)
# Provenance, not harness settings — these can never make a delta unattributable.
PARITY_EXEMPT_KEYS = ("git_commit", "task_source_description", "level_distribution") + EXPECTED_TO_DIFFER_KEYS


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()[
            :12
        ]
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def step_cap_policy(cfg: EvalConfig, source) -> str:
    if cfg.max_steps:
        return f"flat {cfg.max_steps}"
    return (
        f"{cfg.step_cap_multiple}x reference_length, clamped to "
        f"[{cfg.step_cap_floor}, {cfg.step_cap_ceiling}] "
        f"(ungraded tasks: {source.default_step_cap})"
    )


def subset_strategy(cfg: EvalConfig) -> str:
    if not cfg.limit:
        return "full set"
    return f"first {cfg.limit} by dataset order" if not cfg.stratify else f"level-stratified {cfg.limit}"


def harness_fingerprint(cfg: EvalConfig, source, provenance: dict) -> dict:
    """Everything that must match for two runs to be comparable. `compare` diffs exactly this."""
    return {
        "task_source": source.name,
        "task_source_description": source.description,
        "dataset": provenance.get("dataset", source.access_url),
        "dataset_revision": provenance.get("revision", ""),
        "dataset_split": provenance.get("split", ""),
        "coord_space": cfg.coord_space,
        # Parity keys: the observation layout. A run prompted in an older format measures the prompt, not the model.
        "prompt_format": PROMPT_FORMAT_VERSION,
        "observation": cfg.observation,
        "prev_image_size": f"{PREV_IMAGE_SIZE[0]}x{PREV_IMAGE_SIZE[1]}",
        "effect_annotations": True,
        "type_mode": cfg.type_mode,
        # Parity key, deliberately NOT expected-to-differ: both sides of a delta must have been parsed the same way.
        "action_grammar": "tolerant" if cfg.tolerant_grammar else "canonical",
        # Also a parity key: adding a verb changes what counts as a parse failure.
        "action_verbs": ",".join(ACTION_VERBS),
        # Parity key: the signature set decides which episodes leave the success denominator.
        "block_signatures": f"v{BLOCK_SIGNATURES_VERSION}:{','.join(n for n, _ in BLOCK_SIGNATURES)}",
        # Parity key: 58% of Online-Mind2Web is reachable from a US residential exit against ~42% from this machine.
        "browser_backend": cfg.browser_backend,
        "egress": cfg.egress_note,
        "step_cap_policy": step_cap_policy(cfg, source),
        "max_unproductive_steps": cfg.max_unproductive_steps,
        "max_parse_failures": cfg.max_parse_failures,
        "subset_strategy": subset_strategy(cfg),
        "level_distribution": provenance.get("level_distribution", {}),
        "screenshot_format": cfg.screenshot_format,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "student_backend": cfg.student,
        "judge_enabled": cfg.judge,
        "judge_model": MODEL_FOR_REASONING if cfg.judge else "",
        "judge_mid_screenshots": cfg.judge_mid_screenshots,
        "driver_loop_guard_repeats": cfg.loop_guard_repeats,
        "history_window": HISTORY_WINDOW,
        "git_commit": git_commit(),
    }


def coord_space_diagnosis(fingerprint: dict, metrics: dict) -> str:
    """The run-level version of the per-episode warning: is this number even measuring the model?"""
    space = fingerprint.get("coord_space")
    if space not in COORD_SPACE_SUSPECT:
        return ""
    rate = metrics.get("emitted_out_of_viewport_rate", 0)
    if rate < EVAL.coord_space_suspect_rate or not metrics.get("emitted_coords"):
        return ""
    hint, other_space = COORD_SPACE_SUSPECT[space]
    max_x, max_y = COORD_SPACE_RANGE[space]
    return (
        f"COORDINATE SPACE SUSPECT: {rate * 100:.0f}% of emitted coordinates "
        f"({metrics['emitted_out_of_viewport']}/{metrics['emitted_coords']}) fall outside "
        f"0-{max_x} x 0-{max_y} under coord_space={space}. {hint}; re-run with --coord-space {other_space} "
        f"before reading any success rate below."
    )
