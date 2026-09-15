"""ScreenSpot scoring conventions: the three settings that turn a grounding number wrong without ever raising."""

from cua.eval.screenspot.config import EvalConfig
from cua.eval.screenspot.dataset import Sample, normalize_bbox
from cua.eval.screenspot.predictors import MockPredictor
from cua.eval.screenspot.scoring import aggregate, build_coord_space, evaluate, parse_point, point_in_bbox

ORIG_W = 1920
ORIG_H = 1080
# The element at the centre of a 1920x1080 screen: normalized_1000 answers (500, 500), viewport_abs answers (640, 360).
CENTRE_BBOX = (900.0, 500.0, 1020.0, 580.0)
CENTRE_RESPONSE = "action: click(<ref>(500, 500)</ref>, button=left)"


def _centre_sample():
    return Sample("Click the centre button", CENTRE_BBOX, ORIG_W, ORIG_H, "text", "web")


def _score(coord_space_name, sample, response):
    cfg = EvalConfig(coord_space=coord_space_name)
    coord_space = build_coord_space(cfg)
    records = evaluate([sample], MockPredictor(lambda s, c, space: response), cfg, coord_space)
    return records[0]


def test_same_prediction_hits_in_normalized_1000_and_misses_in_viewport_abs():
    hit = _score("normalized_1000", _centre_sample(), CENTRE_RESPONSE)
    miss = _score("viewport_abs", _centre_sample(), CENTRE_RESPONSE)
    assert hit.parsed_point == miss.parsed_point == (500.0, 500.0)
    assert hit.mapped_point == (960.0, 540.0) and hit.hit is True
    assert miss.mapped_point == (750.0, 750.0) and miss.hit is False


def test_native_abs_reads_the_prediction_as_source_pixels():
    record = _score("native_abs", _centre_sample(), "action: click(<ref>(960, 540)</ref>, button=left)")
    assert record.mapped_point == (960.0, 540.0)
    assert record.hit is True


def test_normalized_bbox_read_as_pixels_collapses_to_a_sub_pixel_box():
    as_pixels = normalize_bbox([0.47, 0.46, 0.53, 0.54], "xyxy", False, ORIG_W, ORIG_H)
    as_fractions = normalize_bbox([0.47, 0.46, 0.53, 0.54], "xyxy", True, ORIG_W, ORIG_H)
    assert as_pixels == (0.47, 0.46, 0.53, 0.54)
    assert point_in_bbox((960.0, 540.0), as_pixels) is False
    assert point_in_bbox((960.0, 540.0), as_fractions) is True


def test_xywh_and_xyxy_describe_different_boxes_from_the_same_numbers():
    assert normalize_bbox([10, 20, 30, 40], "xywh", False, 100, 100) == (10, 20, 40, 60)
    assert normalize_bbox([10, 20, 30, 40], "xyxy", False, 100, 100) == (10, 20, 30, 40)
    assert point_in_bbox((35, 55), normalize_bbox([10, 20, 30, 40], "xywh", False, 100, 100)) is True
    assert point_in_bbox((35, 55), normalize_bbox([10, 20, 30, 40], "xyxy", False, 100, 100)) is False


def test_unknown_bbox_format_is_refused():
    try:
        normalize_bbox([1, 2, 3, 4], "cxcywh", False, 100, 100)
    except ValueError as error:
        assert "cxcywh" in str(error)
    else:
        raise AssertionError("an unknown bbox_format must raise rather than score silently")


def test_point_in_bbox_counts_the_border_and_refuses_an_unparsed_point():
    assert point_in_bbox((50, 50), (50, 50, 150, 150)) is True
    assert point_in_bbox((150, 150), (50, 50, 150, 150)) is True
    assert point_in_bbox((151, 150), (50, 50, 150, 150)) is False
    assert point_in_bbox(None, (50, 50, 150, 150)) is False


def test_an_unparsed_answer_is_a_miss_not_a_crash():
    record = _score("normalized_1000", _centre_sample(), "I cannot tell where that element is.")
    assert record.parsed_point is None
    assert record.mapped_point is None
    assert record.hit is False
    assert parse_point("I cannot tell where that element is.", EvalConfig().parse_patterns) is None


def test_aggregate_reports_accuracy_and_parse_rate_separately():
    hit = _score("normalized_1000", _centre_sample(), CENTRE_RESPONSE)
    parsed_miss = _score("viewport_abs", _centre_sample(), CENTRE_RESPONSE)
    unparsed = _score("normalized_1000", _centre_sample(), "no idea")
    summary = aggregate([hit, parsed_miss, unparsed])
    assert summary["total"] == 3
    assert summary["hits"] == 1
    assert abs(summary["accuracy"] - 1 / 3) < 1e-9
    assert abs(summary["parse_rate"] - 2 / 3) < 1e-9
    assert summary["by_data_type"]["text"] == {"total": 3, "hits": 1, "accuracy": 1 / 3}
