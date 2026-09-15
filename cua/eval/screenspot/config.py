"""One ScreenSpot run's knobs: the dataset conventions, the per-model spec and the defaults each is scored against."""

from dataclasses import dataclass, field

from cua.settings import EVAL, MAX_PIXELS_DEFAULT
from cua.trajectory import VIEWPORT

DEFAULT_MAX_PIXELS = MAX_PIXELS_DEFAULT
DEFAULT_VIEWPORT_W, DEFAULT_VIEWPORT_H = VIEWPORT
# `os-copilot/ScreenSpot-v2` is raw JSON + an image zip that load_dataset() cannot read at all; see docs/evaluation.md.
DEFAULT_DATASET = "rootsautomation/ScreenSpot"
DEFAULT_SPLIT = "test"

DEFAULT_PARSE_PATTERNS = [
    r"<ref>\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)</ref>",
    # Trailing `[,)]` so `click(x, y, button=left)` parses: the point is unambiguous either way.
    r"click\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[,)]",
    # `<x, y>` is the base model's own spelling; a digit must follow `<`, so the canonical pattern above still wins.
    r"<\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*>",
    r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)",
    r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]",
    r'"x"\s*:\s*(-?\d+\.?\d*).*?"y"\s*:\s*(-?\d+\.?\d*)',
]

# What every arm is SHOWN, independent of how its answer is read back: "WxH" resizes, "native" sends the source image.
PREPARED_IMAGE_SIZE = EVAL.screenspot_image_size


@dataclass
class ModelSpec:
    label: str
    model: str
    adapters: str | None = None
    backend: str | None = None  # overrides EvalConfig.backend when set
    # Overrides EvalConfig.coord_space for THIS model; a single global space mis-scores one side of a comparison.
    coord_space: str | None = None


@dataclass
class EvalConfig:
    dataset: str = DEFAULT_DATASET
    split: str = DEFAULT_SPLIT
    # Defaults match DEFAULT_DATASET's convention; a wrong one reports a confident wrong number instead of crashing.
    bbox_format: str = "xyxy"
    bbox_normalized: bool = True
    viewport_w: int = DEFAULT_VIEWPORT_W
    viewport_h: int = DEFAULT_VIEWPORT_H
    coord_space: str = "normalized_1000"
    parse_patterns: list = field(default_factory=lambda: list(DEFAULT_PARSE_PATTERNS))
    backend: str = "swift"
    limit: int | None = None
    output_dir: str = "eval_results"
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_tokens: int = 2048
    temperature: float = 0.0
