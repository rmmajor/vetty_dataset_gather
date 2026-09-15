"""ScreenSpot grounding evaluation: does a predicted click point land inside the target element's box (docs/evaluation.md)."""

from cua.eval.screenspot.cli import main
from cua.eval.screenspot.config import EvalConfig, ModelSpec
from cua.eval.screenspot.dataset import Sample, load_screenspot, normalize_bbox
from cua.eval.screenspot.predictors import PREDICTOR_BACKENDS, build_predictor
from cua.eval.screenspot.scoring import (
    COORD_SPACES,
    Record,
    aggregate,
    build_coord_space,
    evaluate,
    parse_point,
    point_in_bbox,
)
from cua.eval.screenspot.smoke import run_smoke_test

__all__ = [
    "COORD_SPACES",
    "PREDICTOR_BACKENDS",
    "EvalConfig",
    "ModelSpec",
    "Record",
    "Sample",
    "aggregate",
    "build_coord_space",
    "build_predictor",
    "evaluate",
    "load_screenspot",
    "main",
    "normalize_bbox",
    "parse_point",
    "point_in_bbox",
    "run_smoke_test",
]
