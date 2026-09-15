"""The `run` entrypoint: flags into an EvalConfig, the model specs sourced, every arm scored, the comparison written."""

import argparse
import sys
from pathlib import Path

import yaml

from cua.eval.screenspot.config import (
    DEFAULT_DATASET,
    DEFAULT_MAX_PIXELS,
    DEFAULT_PARSE_PATTERNS,
    DEFAULT_SPLIT,
    DEFAULT_VIEWPORT_H,
    DEFAULT_VIEWPORT_W,
    EvalConfig,
    ModelSpec,
)
from cua.eval.screenspot.dataset import load_screenspot
from cua.eval.screenspot.predictors import PREDICTOR_BACKENDS, build_predictor
from cua.eval.screenspot.report import print_comparison, write_comparison, write_results
from cua.eval.screenspot.scoring import COORD_SPACES, aggregate, build_coord_space, evaluate
from cua.eval.screenspot.smoke import run_smoke_test
from cua.prompt.coords import parse_viewport


def load_model_specs(args):
    """The specs to score: a YAML list from --models-config, else the single spec the flags describe."""
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
        return [
            ModelSpec(
                label=args.label or "model", model=args.model, adapters=args.adapters, coord_space=args.coord_space
            )
        ]
    raise ValueError("Provide --models-config or --model (or use --smoke-test).")


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


def run_eval_for_spec(spec, samples, cfg):
    coord_space = build_coord_space(cfg, spec)
    predictor = build_predictor(spec, cfg)
    return evaluate(samples, predictor, cfg, coord_space)


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
    run_parser.add_argument(
        "--bbox-format",
        default="xyxy",
        choices=["xywh", "xyxy"],
        help="Annotation convention of --dataset (default matches DEFAULT_DATASET).",
    )
    run_parser.add_argument(
        "--bbox-normalized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dataset bbox is 0-1 rather than pixels. Use --no-bbox-normalized "
        "for a mirror that stores absolute pixels.",
    )
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--viewport", default=f"{DEFAULT_VIEWPORT_W}x{DEFAULT_VIEWPORT_H}")
    run_parser.add_argument(
        "--coord-space",
        default="normalized_1000",
        choices=sorted(COORD_SPACES),
        help="Global default. A spec in --models-config may override it with "
        "`coord_space:` — a stock Qwen and an sft3 checkpoint answer in "
        "normalized_1000, an sft-v1/v2 checkpoint answers in viewport_abs, so a "
        "before/after comparison MUST score each side in its own space.",
    )
    run_parser.add_argument("--parse-pattern", action="append", help="Override default regex(es); repeatable.")
    run_parser.add_argument("--output-dir", default="eval_results")
    run_parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    run_parser.add_argument("--max-tokens", type=int, default=2048)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--smoke-test", action="store_true", help="Run the no-model pipeline self-test and exit.")
    run_parser.set_defaults(func=run)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args) or 0
