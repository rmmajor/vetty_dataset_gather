"""The recorded-trajectory contract every producer writes and every consumer reads."""

from datetime import datetime

from pydantic import BaseModel, Field


class TrajectoryStep(BaseModel):
    step: int
    screenshot: str
    thought: str = ""
    actions: list[dict] = Field(default_factory=list)
    # Inline capture fields, additive: convert ignores unknown step keys, capture_meta.json holds the full record.
    bbox: dict | None = None
    role: str | None = None
    name: str | None = None
    # Watchdog verdict {kind:"approval", decision, rationale, by} — an observation step: empty actions, never an SFT target, threaded into later history.
    event: dict | None = None


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
    # Derived after capture (stage 6 for Path B); optional so pre-existing trajectories still validate.
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
    # Login-gated lane, additive: capture policy, session provenance, terminal risk class, and the replay manifest convert ignores.
    autonomy: str = "autonomous"  # "guarded" | "autonomous"
    auth_mode: str = "none"  # "pre_session" | "handoff_login" | "none"
    risk_category: str = ""
    preconditions: dict = Field(default_factory=dict)


class StepCaptureMeta(BaseModel):
    """Per-step diagnostics; element fields default gracefully when the DOM probe is absent or fails."""

    selector: str = ""
    tag: str | None = None
    role: str | None = None
    name: str | None = None
    attrs: dict = Field(default_factory=dict)
    bbox: dict | None = None
    is_visible: bool | None = None
    is_enabled: bool | None = None
    in_viewport: bool | None = None
    url_before: str | None = None
    url_after: str | None = None
    caused_navigation: bool = False
    dom_changed: bool | None = None
    error: str | None = None
    console_errors: int = 0
    screenshot_sha: str = ""
    screenshot_width: int = 0
    screenshot_height: int = 0
    scroll_position: dict = Field(default_factory=lambda: {"x": 0, "y": 0})
    dt_ms: int = 0
    wait_state: str | None = None


class CaptureMeta(BaseModel):
    schema_version: int = 1
    task_id: str = ""
    framework: str = ""
    framework_language: str = ""
    steps: dict[str, StepCaptureMeta] = Field(default_factory=dict)
