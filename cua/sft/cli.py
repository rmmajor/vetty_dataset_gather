"""`python -m cua.sft convert|train` — the corpus converter and the ms-swift launcher."""

import argparse

from cua.sft.convert import convert_trajectories
from cua.sft.grounding import GROUNDING_DEFAULT_SHARE, GROUNDING_SEED
from cua.sft.swift_launch import train_model


def _add_convert_arguments(parser):
    parser.add_argument("--input", type=str, required=True, help="Input trajectories directory")
    parser.add_argument("--output", type=str, default="training_data/", help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument(
        "--exclude-waste-steps",
        action="store_true",
        help="Drop retry/backtrack click steps as training targets (kept in "
        "history); non-destructive, trajectory files untouched",
    )
    parser.add_argument(
        "--meandering-mode",
        choices=("keep", "subsample", "exclude"),
        default="keep",
        help="Verifier-graded meandering trajectories: keep all steps, subsample "
        "mid-wander targets (head/tail kept, MEANDER_* env knobs), or drop whole",
    )
    parser.add_argument(
        "--exclude-meandering", action="store_true", help="Deprecated alias for --meandering-mode exclude"
    )
    parser.add_argument(
        "--include-replay",
        action="store_true",
        help="Also train on legacy test-replay trajectories (QA-style supervision; off by default)",
    )
    parser.add_argument(
        "--health-config",
        type=str,
        default="corpus_health.json",
        help="Corpus health thresholds JSON (missing file skips the gate)",
    )
    parser.add_argument(
        "--skip-health-gate",
        action="store_true",
        help="Report health violations but write the dataset anyway (experiments only)",
    )
    parser.add_argument(
        "--grounding-share",
        type=float,
        default=GROUNDING_DEFAULT_SHARE,
        help="Ceiling on element-grounding click examples as a share of agent step examples "
        "(mined from recorded click targets; GROUNDING_* env knobs tune sampling)",
    )
    parser.add_argument("--no-grounding", action="store_true", help="Alias for --grounding-share 0")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the split shuffles and the grounding sampler for a reproducible dataset "
        "(default: unseeded splits, GROUNDING_SEED for grounding)",
    )


def _add_train_arguments(parser):
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")


def main():
    parser = argparse.ArgumentParser(description="Convert trajectories and train Qwen 3.5-9B")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_convert_arguments(subparsers.add_parser("convert", help="Convert trajectories to ms-swift SFT format"))
    _add_train_arguments(subparsers.add_parser("train", help="Launch ms-swift SFT training"))

    args, passthrough = parser.parse_known_args()

    if args.command == "convert":
        convert_trajectories(
            args.input,
            args.output,
            args.val_split,
            include_replay=args.include_replay,
            meandering_mode="exclude" if args.exclude_meandering else args.meandering_mode,
            exclude_waste_steps=args.exclude_waste_steps,
            health_config=args.health_config,
            skip_health_gate=args.skip_health_gate,
            grounding_share=0.0 if args.no_grounding else args.grounding_share,
            grounding_seed=GROUNDING_SEED if args.seed is None else args.seed,
            seed=args.seed,
        )
    elif args.command == "train":
        train_model(args.config, args.dataset, args.epochs, args.lr, args.batch_size, passthrough)


if __name__ == "__main__":
    main()
