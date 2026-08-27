"""ScreenSpot grounding evaluation for GUI computer-use models.

Measures grounding accuracy: given a screenshot and an instruction, the model
predicts a click point and we score whether that point lands inside the target
bounding box. Built to compare a model before and after SFT, and reusable for
arbitrary models via a coordinate-space registry, a configurable output parser,
a pluggable inference backend, and a configurable dataset/bbox adapter.

The model trained in this repo emits absolute pixel coordinates in a fixed
1280x720 viewport (see train_qwen.py); the `viewport_abs` coordinate space
reproduces that exactly. Other conventions (normalized 0-1000, native-resolution
absolute) are selectable for other models.

End-to-end evaluation needs a GPU and a real checkpoint. The `--smoke-test` mode
exercises every non-model component (prompt parity, parsing, coordinate mapping,
hit-testing, aggregation, reporting) against a mock predictor with synthetic
samples, requiring no model, GPU, or network.
"""

import argparse
import base64
import concurrent.futures
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from train_qwen import SYSTEM_PROMPT


GROUNDING_USER_TEMPLATE = "Task: {instruction}\nWhat action should you take?"

DEFAULT_MAX_PIXELS = 1003520
DEFAULT_VIEWPORT_W = 1280
DEFAULT_VIEWPORT_H = 720
# `os-copilot/ScreenSpot-v2` is NOT usable here: that repo is raw JSON + an image zip, with no
# `test` split and no image column, so load_dataset() cannot read it. This mirror ships real
# parquet with a 1272-row `test` split and a proper image column. Its annotation convention —
# bbox as (x1,y1,x2,y2) NORMALIZED to 0-1 — is what the bbox defaults below encode.
DEFAULT_DATASET = "rootsautomation/ScreenSpot"
DEFAULT_SPLIT = "test"

DEFAULT_PARSE_PATTERNS = [
    r"<ref>\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)</ref>",
    r"click\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)",
    r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)",
    r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]",
    r'"x"\s*:\s*(-?\d+\.?\d*).*?"y"\s*:\s*(-?\d+\.?\d*)',
]


@dataclass
class Sample:
    instruction: str
    bbox_xyxy: tuple                  # absolute pixels in the original image
    orig_w: int
    orig_h: int
    data_type: str = "unknown"
    data_source: str = "unknown"
    image: object = None              # PIL.Image, or None when no image is needed


@dataclass
class ModelSpec:
    label: str
    model: str
    adapters: Optional[str] = None
    backend: Optional[str] = None     # overrides EvalConfig.backend when set
    # Overrides EvalConfig.coord_space for THIS model. Different checkpoints answer in different
    # coordinate conventions: a stock Qwen emits `normalized_1000`, while a model fine-tuned on
    # this project's corpus emits `viewport_abs` (1280x720). A before/after comparison is
    # therefore only meaningful if each spec is scored in its own space — a single global
    # --coord-space silently mis-scores one side of the comparison.
    coord_space: Optional[str] = None


@dataclass
class EvalConfig:
    dataset: str = DEFAULT_DATASET
    split: str = DEFAULT_SPLIT
    # Defaults match DEFAULT_DATASET's convention. Get these wrong and the eval does not crash —
    # it reports a confident, wrong number. Scoring normalized bbox as absolute pixels shrinks
    # every target to a sub-pixel box at the origin: 0% accuracy regardless of the model.
    bbox_format: str = "xyxy"
    bbox_normalized: bool = True
    viewport_w: int = DEFAULT_VIEWPORT_W
    viewport_h: int = DEFAULT_VIEWPORT_H
    coord_space: str = "viewport_abs"
    parse_patterns: list = field(default_factory=lambda: list(DEFAULT_PARSE_PATTERNS))
    backend: str = "swift"
    limit: Optional[int] = None
    output_dir: str = "eval_results"
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass
class Record:
    instruction: str
    data_type: str
    data_source: str
    raw: str
    parsed_point: Optional[tuple]     # coordinate as the model emitted it
    mapped_point: Optional[tuple]     # mapped into original-image pixels
    bbox_xyxy: tuple
    hit: bool


# ----------------------------------------------------------------------------
# Coordinate-space registry: how model output relates to original-image pixels.
# Each strategy resizes the image for inference and maps points both directions.
# Add a new convention by registering another factory, not by editing callers.
# ----------------------------------------------------------------------------


class CoordSpace:
    def prepare_image(self, image, orig_w, orig_h):
        raise NotImplementedError

    def to_original(self, x, y, orig_w, orig_h):
        raise NotImplementedError

    def to_model(self, x, y, orig_w, orig_h):
        raise NotImplementedError


class ViewportAbsCoordSpace(CoordSpace):
    def __init__(self, viewport_w, viewport_h):
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h

    def prepare_image(self, image, orig_w, orig_h):
        if image is None:
            return None
        return image.resize((self.viewport_w, self.viewport_h))

    def to_original(self, x, y, orig_w, orig_h):
        return (x * orig_w / self.viewport_w, y * orig_h / self.viewport_h)

    def to_model(self, x, y, orig_w, orig_h):
        return (x * self.viewport_w / orig_w, y * self.viewport_h / orig_h)


class Normalized1000CoordSpace(CoordSpace):
    def prepare_image(self, image, orig_w, orig_h):
        return image

    def to_original(self, x, y, orig_w, orig_h):
        return (x / 1000.0 * orig_w, y / 1000.0 * orig_h)

    def to_model(self, x, y, orig_w, orig_h):
        return (x / orig_w * 1000.0, y / orig_h * 1000.0)


class NativeAbsCoordSpace(CoordSpace):
    def prepare_image(self, image, orig_w, orig_h):
        return image

    def to_original(self, x, y, orig_w, orig_h):
        return (x, y)

    def to_model(self, x, y, orig_w, orig_h):
        return (x, y)


COORD_SPACES: dict = {
    "viewport_abs": lambda cfg: ViewportAbsCoordSpace(cfg.viewport_w, cfg.viewport_h),
    "normalized_1000": lambda cfg: Normalized1000CoordSpace(),
    "native_abs": lambda cfg: NativeAbsCoordSpace(),
}


def resolve_coord_space(cfg, spec=None) -> str:
    """The coord space for one spec: its own if set, else the global default."""
    return (spec.coord_space if spec is not None and spec.coord_space else cfg.coord_space)


def build_coord_space(cfg, spec=None):
    name = resolve_coord_space(cfg, spec)
    if name not in COORD_SPACES:
        raise ValueError(
            f"Unknown coord space {name!r}. Available: {sorted(COORD_SPACES)}"
        )
    return COORD_SPACES[name](cfg)


# ----------------------------------------------------------------------------
# Parsing, hit-testing, prompt building, bbox normalization.
# ----------------------------------------------------------------------------


def parse_point(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        try:
            return (float(match.group(1)), float(match.group(2)))
        except (ValueError, IndexError):
            continue
    return None


def point_in_bbox(point, bbox_xyxy):
    if point is None:
        return False
    x, y = point
    x0, y0, x1, y1 = bbox_xyxy
    return x0 <= x <= x1 and y0 <= y <= y1


def build_messages(instruction, image_ref):
    user_text = GROUNDING_USER_TEMPLATE.format(instruction=instruction)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def normalize_bbox(raw_bbox, bbox_format, normalized, width, height):
    values = [float(v) for v in raw_bbox]
    if bbox_format == "xywh":
        x, y, w, h = values
        x0, y0, x1, y1 = x, y, x + w, y + h
    elif bbox_format == "xyxy":
        x0, y0, x1, y1 = values
    else:
        raise ValueError(f"Unknown bbox_format {bbox_format!r}. Use 'xywh' or 'xyxy'.")
    if normalized:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    return (x0, y0, x1, y1)


# ----------------------------------------------------------------------------
# Dataset adapter. `datasets`/Pillow are pulled in lazily so the smoke-test path
# (mock backend, synthetic samples) needs neither installed.
# ----------------------------------------------------------------------------


def load_screenspot(cfg):
    from datasets import load_dataset

    dataset = load_dataset(cfg.dataset, split=cfg.split)
    count = 0
    for row in dataset:
        if cfg.limit is not None and count >= cfg.limit:
            break
        image = row.get("image")
        instruction = row.get("instruction") or row.get("task") or ""
        raw_bbox = row.get("bbox") or row.get("bounding_box")
        if image is None or raw_bbox is None or not instruction:
            continue
        width, height = image.size
        bbox = normalize_bbox(raw_bbox, cfg.bbox_format, cfg.bbox_normalized, width, height)
        yield Sample(
            instruction=instruction,
            bbox_xyxy=bbox,
            orig_w=width,
            orig_h=height,
            data_type=str(row.get("data_type", "unknown")),
            data_source=str(row.get("data_source", row.get("source", "unknown"))),
            image=image,
        )
        count += 1


# ----------------------------------------------------------------------------
# Predictor registry. `swift` shells out to the `swift infer` CLI (mirroring how
# train_qwen.py shells out to `swift sft`), so it needs no in-process model
# imports. `mock` returns canned text for testing and debugging.
# ----------------------------------------------------------------------------


class Predictor:
    def predict_batch(self, samples, cfg, coord_space):
        raise NotImplementedError


class MockPredictor(Predictor):
    def __init__(self, response_fn):
        self.response_fn = response_fn

    def predict_batch(self, samples, cfg, coord_space):
        return [self.response_fn(sample, cfg, coord_space) for sample in samples]


def default_mock_response(sample, cfg, coord_space):
    center_x = cfg.viewport_w / 2
    center_y = cfg.viewport_h / 2
    return (
        "<think>mock prediction</think>\n"
        f"action: click(<ref>({int(center_x)}, {int(center_y)})</ref>, button=left)"
    )


class SwiftPredictor(Predictor):
    """Runs `swift infer` once per model spec over all samples.

    The exact `swift infer` flag names, the result-file schema, and how image
    paths in `messages` are consumed are version-dependent and MUST be validated
    on the GPU box. The result reader (`extract_response_text`) is intentionally
    tolerant of several common schemas.
    """

    def __init__(self, model, adapters, cfg):
        self.model = model
        self.adapters = adapters
        self.cfg = cfg

    def predict_batch(self, samples, cfg, coord_space):
        with tempfile.TemporaryDirectory(prefix="screenspot_infer_") as tmp:
            tmp_dir = Path(tmp)
            infer_path = tmp_dir / "infer.jsonl"
            result_path = tmp_dir / "result.jsonl"

            with open(infer_path, "w") as handle:
                for index, sample in enumerate(samples):
                    prepared = coord_space.prepare_image(sample.image, sample.orig_w, sample.orig_h)
                    image_path = tmp_dir / f"image_{index:06d}.png"
                    prepared.save(image_path)
                    messages = build_messages(sample.instruction, str(image_path))
                    handle.write(json.dumps({"messages": messages}) + "\n")

            cmd = ["swift", "infer", "--model", self.model]
            if self.adapters:
                cmd += ["--adapters", self.adapters]
            cmd += [
                "--val_dataset", str(infer_path),
                "--result_path", str(result_path),
                "--max_new_tokens", str(cfg.max_tokens),
                "--temperature", str(cfg.temperature),
            ]

            env = os.environ.copy()
            if cfg.max_pixels:
                env["MAX_PIXELS"] = str(cfg.max_pixels)

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, env=env, check=True)
            return read_swift_results(result_path, len(samples))


def extract_response_text(obj):
    if not isinstance(obj, dict):
        return ""
    for key in ("response", "generated_text", "completion"):
        if key in obj and obj[key] is not None:
            return str(obj[key])
    messages = obj.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", ""))
    return ""


def read_swift_results(result_path, expected):
    responses = []
    with open(result_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                responses.append(extract_response_text(json.loads(line)))
    if len(responses) < expected:
        responses += [""] * (expected - len(responses))
    return responses[:expected]


class OpenAIPredictor(Predictor):
    """Any OpenAI-compatible /chat/completions endpoint, same wire format the student eval uses."""

    def __init__(self, model, cfg):
        self.model = model
        self.endpoint = (os.environ.get("SCREENSPOT_ENDPOINT") or "").rstrip("/")
        self.api_key = os.environ.get("SCREENSPOT_API_KEY", "")
        self.workers = int(os.environ.get("SCREENSPOT_WORKERS", "8"))
        if not self.endpoint:
            raise ValueError("set SCREENSPOT_ENDPOINT to the /v1 base url of the served model")
        self.cfg = cfg

    def _encode(self, image):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _one(self, sample, coord_space):
        image = coord_space.prepare_image(sample.image, sample.orig_w, sample.orig_h)
        messages = build_messages(sample.instruction, "")
        messages[1]["content"] = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{self._encode(image)}"}},
            {"type": "text", "text": messages[1]["content"][1]["text"]},
        ]
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": 0.0, "max_tokens": 512}).encode()
        req = urllib.request.Request(
            f"{self.endpoint}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.load(r)["choices"][0]["message"]["content"] or ""
            except Exception:
                if attempt == 2:
                    return ""
                time.sleep(2 * (attempt + 1))
        return ""

    def predict_batch(self, samples, cfg, coord_space):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda s: self._one(s, coord_space), samples))


def build_openai_predictor(spec, cfg):
    return OpenAIPredictor(model=spec.model, cfg=cfg)


def build_swift_predictor(spec, cfg):
    return SwiftPredictor(model=spec.model, adapters=spec.adapters, cfg=cfg)


def build_mock_predictor(spec, cfg):
    return MockPredictor(default_mock_response)


PREDICTOR_BACKENDS: dict = {
    "swift": build_swift_predictor,
    "openai": build_openai_predictor,
    "mock": build_mock_predictor,
}


def build_predictor(spec, cfg):
    backend = spec.backend or cfg.backend
    if backend not in PREDICTOR_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Available: {sorted(PREDICTOR_BACKENDS)}"
        )
    return PREDICTOR_BACKENDS[backend](spec, cfg)


# ----------------------------------------------------------------------------
# Evaluation loop and metric aggregation.
# ----------------------------------------------------------------------------


def evaluate(samples, predictor, cfg, coord_space):
    raws = predictor.predict_batch(samples, cfg, coord_space)
    records = []
    for sample, raw in zip(samples, raws):
        parsed = parse_point(raw, cfg.parse_patterns)
        if parsed is None:
            mapped = None
            hit = False
        else:
            mapped = coord_space.to_original(parsed[0], parsed[1], sample.orig_w, sample.orig_h)
            hit = point_in_bbox(mapped, sample.bbox_xyxy)
        records.append(
            Record(
                instruction=sample.instruction,
                data_type=sample.data_type,
                data_source=sample.data_source,
                raw=raw,
                parsed_point=parsed,
                mapped_point=mapped,
                bbox_xyxy=sample.bbox_xyxy,
                hit=hit,
            )
        )
    return records


def group_accuracy(records, key):
    groups = {}
    for record in records:
        bucket = groups.setdefault(key(record), {"total": 0, "hits": 0})
        bucket["total"] += 1
        bucket["hits"] += 1 if record.hit else 0
    return {
        name: {
            "total": bucket["total"],
            "hits": bucket["hits"],
            "accuracy": bucket["hits"] / bucket["total"],
        }
        for name, bucket in sorted(groups.items())
    }


def aggregate(records):
    total = len(records)
    hits = sum(1 for record in records if record.hit)
    parsed = sum(1 for record in records if record.parsed_point is not None)
    return {
        "total": total,
        "hits": hits,
        "accuracy": hits / total if total else 0.0,
        "parse_rate": parsed / total if total else 0.0,
        "by_data_type": group_accuracy(records, lambda r: r.data_type),
        "by_data_source": group_accuracy(records, lambda r: r.data_source),
    }


def run_eval_for_spec(spec, samples, cfg):
    coord_space = build_coord_space(cfg, spec)
    predictor = build_predictor(spec, cfg)
    return evaluate(samples, predictor, cfg, coord_space)


# ----------------------------------------------------------------------------
# Reporting.
# ----------------------------------------------------------------------------


def write_results(path, spec, cfg, summary, records):
    payload = {
        "label": spec.label,
        "model": spec.model,
        "adapters": spec.adapters,
        "backend": spec.backend or cfg.backend,
        "dataset": cfg.dataset,
        "split": cfg.split,
        "coord_space": resolve_coord_space(cfg, spec),
        "bbox_format": cfg.bbox_format,
        "bbox_normalized": cfg.bbox_normalized,
        "viewport": [cfg.viewport_w, cfg.viewport_h],
        "summary": summary,
        "records": [
            {
                "instruction": r.instruction,
                "data_type": r.data_type,
                "data_source": r.data_source,
                "parsed_point": r.parsed_point,
                "mapped_point": r.mapped_point,
                "bbox_xyxy": r.bbox_xyxy,
                "hit": r.hit,
                "raw": r.raw,
            }
            for r in records
        ],
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def write_comparison(path, cfg, summaries):
    payload = {
        "dataset": cfg.dataset,
        "split": cfg.split,
        "coord_space": cfg.coord_space,
        "summaries": summaries,
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def format_group(group):
    if not group:
        return "-"
    return f"{group['accuracy'] * 100:.1f}% ({group['hits']}/{group['total']})"


def print_comparison(summaries):
    labels = list(summaries.keys())

    rows = [("overall accuracy", [f"{summaries[l]['accuracy'] * 100:.1f}%" for l in labels])]
    rows.append(("parse rate", [f"{summaries[l]['parse_rate'] * 100:.1f}%" for l in labels]))
    rows.append(("samples", [str(summaries[l]["total"]) for l in labels]))

    data_types = sorted({t for l in labels for t in summaries[l]["by_data_type"]})
    for data_type in data_types:
        rows.append(
            (f"  type:{data_type}", [format_group(summaries[l]["by_data_type"].get(data_type)) for l in labels])
        )

    data_sources = sorted({s for l in labels for s in summaries[l]["by_data_source"]})
    for data_source in data_sources:
        rows.append(
            (f"  source:{data_source}", [format_group(summaries[l]["by_data_source"].get(data_source)) for l in labels])
        )

    label_width = max([len("metric")] + [len(name) for name, _ in rows])
    col_widths = [max(len(label), *(len(row[1][i]) for row in rows)) for i, label in enumerate(labels)]

    header = "metric".ljust(label_width) + "  " + "  ".join(
        label.ljust(col_widths[i]) for i, label in enumerate(labels)
    )
    print(header)
    print("-" * len(header))
    for name, cells in rows:
        print(name.ljust(label_width) + "  " + "  ".join(
            cell.ljust(col_widths[i]) for i, cell in enumerate(cells)
        ))


# ----------------------------------------------------------------------------
# Model-spec sourcing: inline single spec, or a YAML list of specs.
# ----------------------------------------------------------------------------


def load_model_specs(args):
    if args.models_config:
        with open(args.models_config) as handle:
            raw_specs = yaml.safe_load(handle)
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ValueError(f"{args.models_config} must contain a non-empty list of model specs.")
        return [
            ModelSpec(
                label=spec["label"],
                model=spec["model"],
                adapters=spec.get("adapters"),
                backend=spec.get("backend"),
                coord_space=spec.get("coord_space"),
            )
            for spec in raw_specs
        ]
    if args.model:
        return [ModelSpec(label=args.label or "model", model=args.model, adapters=args.adapters,
                          coord_space=args.coord_space)]
    raise ValueError("Provide --models-config or --model (or use --smoke-test).")


# ----------------------------------------------------------------------------
# Smoke test: full pipeline against a mock predictor and synthetic samples.
# No model, GPU, or network. Asserts every non-model component.
# ----------------------------------------------------------------------------


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


def run_smoke_test(cfg):
    cfg.coord_space = "viewport_abs"
    coord_space = build_coord_space(cfg)

    messages = build_messages("Click the login button", "/tmp/x.png")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    user = messages[1]
    assert user["content"][0] == {"type": "image", "image": "/tmp/x.png"}
    assert user["content"][1]["text"] == "Task: Click the login button\nWhat action should you take?"

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

    samples = build_synthetic_samples()
    predictor = MockPredictor(smoke_response_fn)
    records = evaluate(samples, predictor, cfg, coord_space)
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

    spec = ModelSpec(label="smoke", model="mock", backend="mock")
    if cfg.output_dir:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_results(out_dir / "smoke.json", spec, cfg, summary, records)
        write_comparison(out_dir / "smoke_comparison.json", cfg, {"smoke": summary})

    print()
    print_comparison({"smoke": summary})
    print()
    print("SMOKE TEST PASSED")
    return 0


# ----------------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------------


def parse_viewport(value):
    width, height = value.lower().split("x")
    return int(width), int(height)


def config_from_args(args):
    viewport_w, viewport_h = parse_viewport(args.viewport)
    return EvalConfig(
        dataset=args.dataset,
        split=args.split,
        bbox_format=args.bbox_format,
        bbox_normalized=args.bbox_normalized,
        viewport_w=viewport_w,
        viewport_h=viewport_h,
        coord_space=args.coord_space,
        parse_patterns=args.parse_pattern or list(DEFAULT_PARSE_PATTERNS),
        backend=args.backend,
        limit=args.limit,
        output_dir=args.output_dir,
        max_pixels=args.max_pixels,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


def run(args):
    cfg = config_from_args(args)

    if args.smoke_test:
        return run_smoke_test(cfg)

    specs = load_model_specs(args)
    samples = list(load_screenspot(cfg))
    if not samples:
        print("No samples loaded; check --dataset/--split/--bbox-format.", file=sys.stderr)
        return 1
    print(f"Loaded {len(samples)} samples from {cfg.dataset} [{cfg.split}].")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for spec in specs:
        print(f"\nEvaluating '{spec.label}' (model={spec.model}, adapters={spec.adapters}) ...")
        records = run_eval_for_spec(spec, samples, cfg)
        summary = aggregate(records)
        summaries[spec.label] = summary
        write_results(out_dir / f"{spec.label}.json", spec, cfg, summary, records)

    write_comparison(out_dir / "comparison.json", cfg, summaries)
    print()
    print_comparison(summaries)
    print(f"\nResults written to {out_dir}/")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="ScreenSpot grounding evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the evaluation.")
    run_parser.add_argument("--model", help="Base model id or path (single-spec mode).")
    run_parser.add_argument("--adapters", help="LoRA adapter path for the after-SFT run.")
    run_parser.add_argument("--label", help="Label for the single-spec run.")
    run_parser.add_argument("--models-config", help="YAML list of {label, model, adapters?, backend?} specs.")
    run_parser.add_argument("--backend", default="swift", choices=sorted(PREDICTOR_BACKENDS))
    run_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    run_parser.add_argument("--split", default=DEFAULT_SPLIT)
    run_parser.add_argument("--bbox-format", default="xyxy", choices=["xywh", "xyxy"],
                            help="Annotation convention of --dataset (default matches DEFAULT_DATASET).")
    run_parser.add_argument("--bbox-normalized", action=argparse.BooleanOptionalAction, default=True,
                            help="Dataset bbox is 0-1 rather than pixels. Use --no-bbox-normalized "
                                 "for a mirror that stores absolute pixels.")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--viewport", default=f"{DEFAULT_VIEWPORT_W}x{DEFAULT_VIEWPORT_H}")
    run_parser.add_argument("--coord-space", default="viewport_abs", choices=sorted(COORD_SPACES),
                            help="Global default. A spec in --models-config may override it with "
                                 "`coord_space:` — a stock Qwen answers in normalized_1000, a model "
                                 "fine-tuned on this corpus answers in viewport_abs, so a before/after "
                                 "comparison MUST score each side in its own space.")
    run_parser.add_argument("--parse-pattern", action="append", help="Override default regex(es); repeatable.")
    run_parser.add_argument("--output-dir", default="eval_results")
    run_parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    run_parser.add_argument("--max-tokens", type=int, default=512)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--smoke-test", action="store_true",
                            help="Run the no-model pipeline self-test and exit.")
    run_parser.set_defaults(func=run)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
