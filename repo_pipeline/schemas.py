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
    COMPLETED = "completed"
    REJECTED_GENERATE = "rejected_generate"


STAGE_TO_PASSED_STATUS = {
    1: RepoStatus.DISCOVERED,
    2: RepoStatus.FILTERED,
    3: RepoStatus.SCORED,
    4: RepoStatus.ANALYZED,
    5: RepoStatus.VALIDATED,
    6: RepoStatus.COMPLETED,
}

STAGE_TO_REJECTED_STATUS = {
    2: RepoStatus.REJECTED_FILTER,
    3: RepoStatus.REJECTED_SCORE,
    4: RepoStatus.REJECTED_ANALYZE,
    5: RepoStatus.REJECTED_TRIAL,
    6: RepoStatus.REJECTED_GENERATE,
}

STAGE_TO_INPUT_STATUS = {
    2: RepoStatus.DISCOVERED,
    3: RepoStatus.FILTERED,
    4: RepoStatus.SCORED,
    5: RepoStatus.ANALYZED,
    6: RepoStatus.VALIDATED,
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
        )


class DeepAnalysis(BaseModel):
    test_file_count: int = 0
    test_case_count: int = 0
    action_types: list[str] = Field(default_factory=list)
    is_self_contained: bool = False
    has_docker_compose: bool = False
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


class CapturedAction(BaseModel):
    action_type: str
    x: int = 0
    y: int = 0
    selector: str = ""
    value: str = ""
    button: str = "left"
    screenshot_before: str = ""
    bounding_box: Optional[dict] = None
