"""The eval run's configuration and the per-episode record every metric, report and resume reads."""

from dataclasses import dataclass, field
from pathlib import Path

from cua.eval.student.observation import DEFAULT_OBSERVATION
from cua.eval.task_sources import EvalTask
from cua.prompt.format import PROMPT_FORMAT_VERSION
from cua.settings import EVAL

EVAL_RUNS_DIR = Path(EVAL.runs_dir)


@dataclass
class EvalConfig:
    task_source: str = "online_mind2web"
    tasks_file: str = ""
    limit: int | None = None
    label: str = "baseline"
    run_id: str = ""
    student: str = "openai"
    endpoint: str = ""
    model: str = ""
    adapter: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 512
    student_timeout_s: int = 180
    student_retries: int = 2
    mock_trajectory: str = ""
    coord_space: str = "normalized_1000"
    type_mode: str = "replace"
    browser_backend: str = "docker"
    browser_host: str = ""
    egress_note: str = ""
    tolerant_grammar: bool = True
    max_unproductive_steps: int = 5
    max_parse_failures: int = 3
    max_steps: int = 0  # >0 pins a FLAT cap; 0 derives one per task (below)
    step_cap_multiple: float = 3.0  # cap = multiple x the task's own reference_length
    step_cap_floor: int = 15
    step_cap_ceiling: int = 50
    stratify: bool = False
    episode_timeout_s: int = 900
    screenshot_format: str = "png"
    judge: bool = True
    judge_mid_screenshots: int = 2
    parallel: int = 1
    observation: str = DEFAULT_OBSERVATION
    resume: bool = True
    dataset_revision: str = "main"
    trajectories_dir: str = "trajectories"
    loop_guard_repeats: int = 0
    sandbox_image: str = ""

    def __post_init__(self):
        if self.screenshot_format != "png":
            raise ValueError(
                f"screenshot_format={self.screenshot_format!r}: {PROMPT_FORMAT_VERSION} effect annotations compare "
                f"consecutive screenshots pixel-wise and a lossy frame never compares identical — use png"
            )


@dataclass
class EpisodeRecord:
    task_id: str
    instruction: str
    start_url: str
    website: str = ""
    domain: str = ""
    split_tag: str = ""
    level: str = ""
    reference_length: int = 0
    step_cap: int = 0
    outcome: str = "not_run"
    verified_success: bool = False
    judged_success: bool = False
    judge_deferred: bool = False
    efficiency: str = ""
    judge_reasoning: str = ""
    model_steps: int = 0
    driver_steps: int = 0
    actions_executed: int = 0
    parse_failure_steps: int = 0
    malformed_lines: int = 0
    actions_parsed: int = 0
    actions_canonical: int = 0
    form_deviations: dict = field(default_factory=dict)
    think_steps: int = 0
    driver_errors: int = 0
    out_of_bounds: int = 0
    emitted_coords: int = 0
    emitted_out_of_viewport: int = 0
    coord_space_warning: str = ""
    done_emitted: bool = False
    navigations: int = 0
    handoff_kind: str = ""
    handoff_reason: str = ""
    block_reason: str = ""
    block_step: int = 0
    error: str = ""
    duration_s: float = 0.0
    episode_dir: str = ""
    raw_last_response: str = ""
    # The observation the episode ended on: screenshots alone cannot be re-scanned as the signature set grows.
    final_url: str = ""
    final_snapshot: str = ""


# Environment failures, not model failures: excluded from the denominator, reported separately.
UNSCORED_OUTCOMES = ("site_blocked", "unreachable", "harness_error")


def step_cap_for(task: EvalTask, cfg: EvalConfig, source_default: int) -> int:
    """Per-task step budget: `--max-steps` when pinned, else the multiple of the task's own reference_length."""
    if cfg.max_steps:
        return cfg.max_steps
    if not task.reference_length:
        return source_default
    derived = int(round(cfg.step_cap_multiple * task.reference_length))
    return max(cfg.step_cap_floor, min(cfg.step_cap_ceiling, derived))
