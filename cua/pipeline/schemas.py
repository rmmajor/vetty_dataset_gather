from enum import Enum

from pydantic import BaseModel, Field

from cua.settings import PIPELINE

# The recorded-trajectory contract moved to cua.trajectory.schema; re-exported so pipeline callers keep one import site.
from cua.trajectory.schema import CaptureMeta as CaptureMeta
from cua.trajectory.schema import StepCaptureMeta as StepCaptureMeta
from cua.trajectory.schema import Trajectory as Trajectory
from cua.trajectory.schema import TrajectoryMetadata as TrajectoryMetadata
from cua.trajectory.schema import TrajectoryStep as TrajectoryStep


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
    6: RepoStatus.CAPTURED,  # stage 6 now only captures trajectories (no think blocks)
    7: RepoStatus.COMPLETED,  # stage 7 fills think blocks, then completes
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
    QUALITY_GATE = "quality_gate"  # score / gate below threshold
    NO_TESTS = "no_tests"  # no test files matched the framework globs
    NO_CAPTURE = "no_capture"  # tests ran but produced no usable bbox/screenshot
    NO_BOOT = "no_boot"  # boot gate: the repo's app never opened its port
    NOT_BOOTABLE = "not_bootable"  # stage-2 gate: not a bootable interactive web app (test suite / non-app)
    DEDUP = "dedup"  # fork / duplicate-description clone
    EMPTY_REPO = "empty_repo"
    TOO_LARGE = "too_large"

    # Infrastructure / flakiness — recoverable; retried before becoming terminal.
    CLONE_FAILED = "clone_failed"
    SDK_ERROR = "sdk_error"  # Claude returned a non-success result
    RATE_LIMITED = "rate_limited"
    CONTROL_TIMEOUT = "control_timeout"
    EXEC_TIMEOUT = "exec_timeout"
    SANDBOX_ERROR = "sandbox_error"
    PARSE_FAILED = "parse_failed"  # model output could not be parsed (LLM nondeterminism)

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

# The one module attribute the orchestrator rebinds: --capture-mode overrides it for the whole run.
CAPTURE_MODE = PIPELINE.capture_mode


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
    test_framework: str | None = None
    playwright_version: str | None = None
    framework_language: str | None = None
    deps_install_success: bool = False
    estimated_trajectories: int = 0
    notes: str = ""
    # Stage-4 fit verdict. The holistic "is this worth capturing at all?" judgment, set by the
    # deep-analysis prompt (agent mode) or by judge_fit over a stored analysis. Defaults keep
    # analyses recorded before the verdict existed parsing cleanly — good_fit=True means "the
    # holistic layer did not veto it" (the deterministic fit_veto rules still apply on top).
    good_fit: bool = True
    veto_reason: str = ""


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
    # Boot-gate UI sophistication profile (agent mode): counts of draggables, sliders,
    # canvases, rich-text editors etc. from the driver's /affordances endpoint. Drives the
    # task proposer's drag/complex-input quotas and the dashboard's sophistication view.
    affordances: dict = Field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        return (
            self.pass_rate * 0.30
            + self.action_diversity_score * 0.25
            + self.domain_relevance_score * 0.20
            + min(self.trajectory_count_estimate / 50.0, 1.0) * 0.15
            + (1.0 - self.setup_complexity_score) * 0.10
        )
