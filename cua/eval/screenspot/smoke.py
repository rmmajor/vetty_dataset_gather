"""The no-model self-test: prompt parity, parsing, coordinate mapping, hit-testing and reporting, on synthetics."""

import json
from pathlib import Path

from cua.eval.screenspot.config import DEFAULT_PARSE_PATTERNS, EvalConfig, ModelSpec
from cua.eval.screenspot.dataset import Sample, normalize_bbox
from cua.eval.screenspot.predictors import MockPredictor, build_messages, build_openai_messages
from cua.eval.screenspot.report import print_comparison, write_comparison, write_results
from cua.eval.screenspot.scoring import aggregate, build_coord_space, evaluate, parse_point, point_in_bbox
from cua.prompt.format import GROUNDING_SYSTEM_PROMPT, PROMPT_FORMAT_VERSION, build_grounding_observation


def build_synthetic_samples():
    return [
        Sample("Click the login button", (100, 100, 300, 200), 1920, 1080, "text", "web"),
        Sample("Tap the settings icon", (10, 20, 50, 60), 720, 1280, "icon", "mobile"),
        Sample("Open the file menu", (500, 500, 600, 560), 1280, 720, "text", "desktop"),
    ]


def smoke_response_fn(sample, cfg, coord_space):
    if sample.data_source == "desktop":
        return "I'm not sure where to click on this screen."
    center_x = (sample.bbox_xyxy[0] + sample.bbox_xyxy[2]) / 2
    center_y = (sample.bbox_xyxy[1] + sample.bbox_xyxy[3]) / 2
    model_x, model_y = coord_space.to_model(center_x, center_y, sample.orig_w, sample.orig_h)
    return (
        "<think>aiming at the target element</think>\n"
        f"action: click(<ref>({model_x:.1f}, {model_y:.1f})</ref>, button=left)"
    )


def assert_default_coord_space():
    assert EvalConfig().coord_space == "normalized_1000", "stock Qwen and an sft3 checkpoint both answer normalized"
    normalized = build_coord_space(EvalConfig())
    assert normalized.to_model(960.0, 540.0, 1920, 1080) == (500.0, 500.0)
    assert normalized.to_original(500.0, 500.0, 1920, 1080) == (960.0, 540.0)


def assert_prompt_parity():
    """The grounding prompt IS training's grounding slice: GROUNDING_SYSTEM_PROMPT plus the one-image sft3 layout."""
    messages = build_messages("the login button", "/tmp/x.png")
    assert messages[0] == {"role": "system", "content": GROUNDING_SYSTEM_PROMPT}
    observation = build_grounding_observation('the element described as "the login button"', "/tmp/x.png")
    assert messages[1] == {"role": "user", "content": observation["content"]}, messages[1]
    assert observation["images"] == ["/tmp/x.png"]
    assert observation["text"] == (
        'Current screenshot:\n<image>\nTask: Click the element described as "the login button".\n'
        "What action should you take?"
    ), observation["text"]
    rendered = build_openai_messages("the login button", "QUJD")
    assert rendered[0] == messages[0]
    assert [item["type"] for item in rendered[1]["content"]] == ["text", "image_url", "text"], rendered[1]
    assert rendered[1]["content"][1]["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert "".join(i.get("text", "<image>") for i in rendered[1]["content"]) == observation["text"]


def assert_scoring(cfg, coord_space):
    assert parse_point("foo <ref>(151, 509)</ref> bar", cfg.parse_patterns) == (151.0, 509.0)
    assert parse_point("action: click(640, 360)", DEFAULT_PARSE_PATTERNS) == (640.0, 360.0)
    assert parse_point("the point is (12, 34)", DEFAULT_PARSE_PATTERNS) == (12.0, 34.0)
    assert parse_point("coords [12, 34]", DEFAULT_PARSE_PATTERNS) == (12.0, 34.0)
    assert parse_point('{"x": 5, "y": 9}', DEFAULT_PARSE_PATTERNS) == (5.0, 9.0)
    assert parse_point("I cannot determine a location", DEFAULT_PARSE_PATTERNS) is None

    model_point = coord_space.to_model(960.0, 540.0, 1920, 1080)
    assert abs(model_point[0] - 640.0) < 1e-6 and abs(model_point[1] - 360.0) < 1e-6
    back = coord_space.to_original(model_point[0], model_point[1], 1920, 1080)
    assert abs(back[0] - 960.0) < 1e-6 and abs(back[1] - 540.0) < 1e-6

    assert point_in_bbox((100, 100), (50, 50, 150, 150)) is True
    assert point_in_bbox((0, 0), (50, 50, 150, 150)) is False
    assert point_in_bbox(None, (50, 50, 150, 150)) is False

    assert normalize_bbox([10, 20, 30, 40], "xywh", False, 100, 100) == (10, 20, 40, 60)
    assert normalize_bbox([0.1, 0.2, 0.3, 0.4], "xyxy", True, 100, 200) == (10.0, 40.0, 30.0, 80.0)


def assert_aggregation(cfg, coord_space):
    records = evaluate(build_synthetic_samples(), MockPredictor(smoke_response_fn), cfg, coord_space)
    summary = aggregate(records)

    assert summary["total"] == 3
    assert summary["hits"] == 2
    assert abs(summary["accuracy"] - 2 / 3) < 1e-9
    assert abs(summary["parse_rate"] - 2 / 3) < 1e-9
    assert summary["by_data_type"]["text"] == {"total": 2, "hits": 1, "accuracy": 0.5}
    assert summary["by_data_type"]["icon"] == {"total": 1, "hits": 1, "accuracy": 1.0}
    assert summary["by_data_source"]["web"]["accuracy"] == 1.0
    assert summary["by_data_source"]["mobile"]["accuracy"] == 1.0
    assert summary["by_data_source"]["desktop"]["accuracy"] == 0.0
    return summary, records


def assert_reporting(cfg, summary, records):
    spec = ModelSpec(label="smoke", model="mock", backend="mock")
    if not cfg.output_dir:
        return
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_results(out_dir / "smoke.json", spec, cfg, summary, records)
    write_comparison(out_dir / "smoke_comparison.json", cfg, {"smoke": summary})
    assert json.loads((out_dir / "smoke.json").read_text())["prompt_format"] == PROMPT_FORMAT_VERSION
    assert json.loads((out_dir / "smoke_comparison.json").read_text())["prompt_format"] == PROMPT_FORMAT_VERSION


def run_smoke_test(cfg):
    assert_default_coord_space()
    cfg.coord_space = "viewport_abs"
    coord_space = build_coord_space(cfg)
    assert_prompt_parity()
    assert_scoring(cfg, coord_space)
    summary, records = assert_aggregation(cfg, coord_space)
    assert_reporting(cfg, summary, records)

    print()
    print_comparison({"smoke": summary})
    print()
    print("SMOKE TEST PASSED")
    return 0
