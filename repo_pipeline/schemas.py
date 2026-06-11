import os
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RepoStatus(str, Enum):
    PENDING_DISCOVERY = "pending_discovery"
    DISCOVERED = "discovered"
    FILTERED = "filtered"
    REJECTED_FILTER = "rejected_filter"
    SCORED = "scored"
    REJECTED_SCORE = "rejected_score"
    ANALYZED = "analyzed"
    REJECTED_ANALYZE = "rejected_analyze"
    VALIDATED = "validated"
    REJECTED_TRIAL = "rejected_trial"
    CAPTURED = "captured"
    REJECTED_GENERATE = "rejected_generate"
    COMPLETED = "completed"


STAGE_TO_PASSED_STATUS = {
    1: RepoStatus.DISCOVERED,
    2: RepoStatus.FILTERED,
    3: RepoStatus.SCORED,
    4: RepoStatus.ANALYZED,
    5: RepoStatus.VALIDATED,
    6: RepoStatus.CAPTURED,    # stage 6 now only captures trajectories (no think blocks)
    7: RepoStatus.COMPLETED,   # stage 7 fills think blocks, then completes
}

STAGE_TO_REJECTED_STATUS = {
    2: RepoStatus.REJECTED_FILTER,
    3: RepoStatus.REJECTED_SCORE,
    4: RepoStatus.REJECTED_ANALYZE,
    5: RepoStatus.REJECTED_TRIAL,
    6: RepoStatus.REJECTED_GENERATE,
    # Stage 7 (think) never rejects: a captured repo with good trajectories stays `captured`
    # until its thoughts are complete (resumable across runs), it is never thrown away.
}

STAGE_TO_INPUT_STATUS = {
    2: RepoStatus.DISCOVERED,
    3: RepoStatus.FILTERED,
    4: RepoStatus.SCORED,
    5: RepoStatus.ANALYZED,
    6: RepoStatus.VALIDATED,
    7: RepoStatus.CAPTURED,
}


class FailureCategory(str, Enum):
    """Why a repo failed a stage. Orthogonal to RepoStatus (which encodes the stage): the
    status says where it died, the category says *why* — a quality decision vs an infra/flaky
    failure. Stored in repos.failure_category alongside the free-text filter_reason so the
    dashboard can split genuine rejections from recoverable infrastructure noise."""

    # Quality decisions — the funnel did its job; terminal.
    QUALITY_GATE = "quality_gate"      # score / gate below threshold
    NO_TESTS = "no_tests"              # no test files matched the framework globs
    NO_CAPTURE = "no_capture"          # tests ran but produced no usable bbox/screenshot
    NO_BOOT = "no_boot"                # boot gate: the repo's app never opened its port
    DEDUP = "dedup"                    # fork / duplicate-description clone
    EMPTY_REPO = "empty_repo"
    TOO_LARGE = "too_large"

    # Infrastructure / flakiness — recoverable; retried before becoming terminal.
    CLONE_FAILED = "clone_failed"
    SDK_ERROR = "sdk_error"            # Claude returned a non-success result
    RATE_LIMITED = "rate_limited"
    CONTROL_TIMEOUT = "control_timeout"
    EXEC_TIMEOUT = "exec_timeout"
    SANDBOX_ERROR = "sandbox_error"
    PARSE_FAILED = "parse_failed"      # model output could not be parsed (LLM nondeterminism)

    # Infra, but auto-retry cannot help (re-running hits the same wall) — surfaced, not retried.
    MAX_TURNS = "max_turns"
    BUDGET_EXCEEDED = "budget_exceeded"


# Categories a transient retry can plausibly recover from. Tunable via env so the operator can
# re-classify a category (e.g. move NO_CAPTURE in/out) without code edits.
RETRYABLE_CATEGORIES = {
    FailureCategory.CLONE_FAILED,
    FailureCategory.SDK_ERROR,
    FailureCategory.RATE_LIMITED,
    FailureCategory.CONTROL_TIMEOUT,
    FailureCategory.EXEC_TIMEOUT,
    FailureCategory.SANDBOX_ERROR,
    FailureCategory.PARSE_FAILED,
}

# Infra categories that are NOT auto-retried but ARE recoverable by re-queue (they need a config
# change first, e.g. a higher max_turns). Used by --requeue-infra.
NON_QUALITY_CATEGORIES = RETRYABLE_CATEGORIES | {
    FailureCategory.MAX_TURNS,
    FailureCategory.BUDGET_EXCEEDED,
}

MAX_RETRIES = int(os.environ.get("PIPELINE_MAX_RETRIES", "3"))

# How stages 4-6 produce trajectories. "agent": a Claude teacher pursues proposed user goals in
# the repo's booted app (the browser driver records coordinates; framework-independent).
# "replay": the legacy instrumented-test-suite path. Defined here — next to the state machine it
# shapes — so every stage reads the same switch instead of each re-reading the environment.
CAPTURE_MODE = os.environ.get("PIPELINE_CAPTURE_MODE", "agent")


class PipelineFailure(Exception):
    """Base for stage failures. ``category`` is a FailureCategory, ``reason`` the human detail,
    and ``extra`` any registry fields to persist with the outcome (e.g. the scores behind it)."""

    def __init__(self, category: "FailureCategory", reason: str, **extra):
        super().__init__(reason)
        self.category = category
        self.reason = reason
        self.extra = extra


class QualityReject(PipelineFailure):
    """The repo genuinely failed a quality gate — terminal, never retried."""


class RetryableFailure(PipelineFailure):
    """An infrastructure / flaky failure — retried up to MAX_RETRIES before becoming terminal."""

# Per-stage cutoff: rejections recorded before this instant were produced by superseded code and
# should be re-queued (--requeue-stale). Extend when a stage's behavior changes again.
STAGE_BEHAVIOR_CUTOFFS = {
    5: "2026-05-27T14:29:03+03:00",  # commit 31c698d replaced agent-driven stage 5 with code-driven
}


class RepoCandidate(BaseModel):
    full_name: str
    html_url: str
    stars: int = 0
    last_pushed: Optional[str] = None
    size_kb: int = 0
    language: Optional[str] = None
    archived: bool = False
    fork: bool = False
    description: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    default_branch: str = "main"
    source_query: Optional[str] = None


class ShallowScores(BaseModel):
    production_complexity: int = Field(ge=1, le=5)
    ui_difficulty: int = Field(ge=1, le=5)
    interaction_richness: int = Field(ge=1, le=5)
    test_count: int = Field(ge=1, le=5)
    multi_page_flows: int = Field(ge=1, le=5)
    runnability: int = Field(ge=1, le=5)
    framework_clarity: int = Field(ge=1, le=5)
    application_vs_library: int = Field(ge=1, le=5)

    @property
    def total(self) -> int:
        return (
            self.production_complexity
            + self.ui_difficulty
            + self.interaction_richness
            + self.test_count
            + self.multi_page_flows
            + self.runnability
            + self.framework_clarity
            + self.application_vs_library
        )


class DeepAnalysis(BaseModel):
    test_file_count: int = 0
    test_case_count: int = 0
    action_types: list[str] = Field(default_factory=list)
    is_self_contained: bool = False
    has_docker_compose: bool = False
    has_local_dev_server: bool = False
    has_webserver_config: bool = False
    needs_external_backend: bool = False
    tests_public_url: bool = False
    test_framework: Optional[str] = None
    playwright_version: Optional[str] = None
    framework_language: Optional[str] = None
    deps_install_success: bool = False
    estimated_trajectories: int = 0
    notes: str = ""


class TrialResults(BaseModel):
    pass_rate: float = 0.0
    action_diversity_score: float = 0.0
    domain_relevance_score: float = 0.0
    trajectory_count_estimate: int = 0
    setup_complexity_score: float = 0.0
    valid_bounding_boxes: int = 0
    total_actions_captured: int = 0
    screenshots_captured: int = 0
    notes: str = ""

    @property
    def weighted_score(self) -> float:
        return (
            self.pass_rate * 0.30
            + self.action_diversity_score * 0.25
            + self.domain_relevance_score * 0.20
            + min(self.trajectory_count_estimate / 50.0, 1.0) * 0.15
            + (1.0 - self.setup_complexity_score) * 0.10
        )


class TrajectoryStep(BaseModel):
    step: int
    screenshot: str
    thought: str = ""
    actions: list[dict] = Field(default_factory=list)
    # Inline, high-value capture fields (additive; train_qwen.py convert ignores unknown
    # step keys). The full diagnostic record lives in the sibling capture_meta.json.
    bbox: Optional[dict] = None
    role: Optional[str] = None
    name: Optional[str] = None


class Trajectory(BaseModel):
    task_id: str
    instruction: str
    start_url: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    total_steps: int = 0
    success: bool = True


class TrajectoryMetadata(BaseModel):
    task_id: str
    instruction: str
    start_url: str
    domain: str = ""
    website: str = ""
    total_steps: int = 0
    success: bool = True
    viewport: dict = Field(default_factory=lambda: {"width": 1280, "height": 720})
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    source_repo: str = ""
    # Aggregate / provenance fields, derived after capture (stage 6 for Path B). All optional
    # so trajectories recorded before this existed still validate.
    framework: str = ""
    framework_language: str = ""
    source_query: str = ""
    repo_stars: int = 0
    action_type_histogram: dict = Field(default_factory=dict)
    element_role_histogram: dict = Field(default_factory=dict)
    length_bucket: str = ""
    n_navigations: int = 0
    valid_bbox_ratio: float = 0.0
    n_console_errors: int = 0
    instruction_source: str = ""
    thought_source: str = ""
    dedup_hash: str = ""
    task_type: str = ""


class StepCaptureMeta(BaseModel):
    """Full per-step diagnostic record. Written to capture_meta.json, keyed by step number.
    Element fields are best-effort and default gracefully when the DOM probe is unavailable
    (Path A, screenshot-only) or fails."""
    selector: str = ""
    tag: Optional[str] = None
    role: Optional[str] = None
    name: Optional[str] = None
    attrs: dict = Field(default_factory=dict)
    bbox: Optional[dict] = None
    is_visible: Optional[bool] = None
    is_enabled: Optional[bool] = None
    in_viewport: Optional[bool] = None
    url_before: Optional[str] = None
    url_after: Optional[str] = None
    caused_navigation: bool = False
    dom_changed: Optional[bool] = None
    error: Optional[str] = None
    console_errors: int = 0
    screenshot_sha: str = ""
    screenshot_width: int = 0
    screenshot_height: int = 0
    scroll_position: dict = Field(default_factory=lambda: {"x": 0, "y": 0})
    dt_ms: int = 0
    wait_state: Optional[str] = None


class CaptureMeta(BaseModel):
    schema_version: int = 1
    task_id: str = ""
    framework: str = ""
    framework_language: str = ""
    steps: dict[str, StepCaptureMeta] = Field(default_factory=dict)


class CapturedAction(BaseModel):
    action_type: str
    x: int = 0
    y: int = 0
    selector: str = ""
    value: str = ""
    button: str = "left"
    screenshot_before: str = ""
    bounding_box: Optional[dict] = None
