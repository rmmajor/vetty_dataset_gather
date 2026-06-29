"""CLI for provider-pluggable remote training.

    python -m remote_train run                 # provision → train → pull adapter → destroy
    python -m remote_train run --smoke         # no-cost self-test (mock provider, no instance)
    python -m remote_train search              # dry run: print the cheapest allowed offer only
    python -m remote_train cleanup             # destroy any instance left behind (orphan sweep)

Subcommand dispatch mirrors `eval_screenspot.py` (`set_defaults(func=…)`); `--provider`
choices come from the registry (`sorted(PROVIDERS)`).
"""
import argparse
import logging
import sys

from remote_train import lifecycle, state
from remote_train.base import PROVIDERS, ProviderError, build_provider
from remote_train.config import load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cli_overrides(args) -> dict:
    return {
        "provider": getattr(args, "provider", None),
        "gpu_name": getattr(args, "gpu", None),
        "min_vram_gb": getattr(args, "min_vram", None),
        "disk_gb": getattr(args, "disk", None),
        "max_price_per_hour": getattr(args, "max_price", None),
        "max_runtime_hours": getattr(args, "max_hours", None),
        "run_eval": True if getattr(args, "eval", False) else None,
        "push_hf_repo": getattr(args, "push_hf", None),
        "notify": True if getattr(args, "notify", False) else None,
    }


def run_cmd(args) -> int:
    if args.smoke:
        return lifecycle.run_smoke_test()
    cfg = load_config(args.config, _cli_overrides(args))
    provider = build_provider(cfg.provider, cfg)
    return lifecycle.run_remote_training(provider, cfg)


def search_cmd(args) -> int:
    cfg = load_config(args.config, _cli_overrides(args))
    provider = build_provider(cfg.provider, cfg)
    offer = provider.search_offer(cfg)
    print(
        f"Cheapest allowed offer: id={offer.id}  {offer.gpu_name} x{offer.num_gpus}  "
        f"{offer.gpu_ram_gb}GB  ${offer.price_per_hour:.3f}/hr  {offer.country}  "
        f"reliability={offer.reliability:.2f}"
    )
    print(f"(dry run — excluded {sorted(cfg.exclude_countries)}; nothing provisioned)")
    return 0


def cleanup_cmd(args) -> int:
    cfg = load_config(args.config, _cli_overrides(args))
    provider = build_provider(cfg.provider, cfg)
    return lifecycle.cleanup(provider, cfg)


def _add_common(p) -> None:
    p.add_argument("--config", default="remote_train.yaml", help="Path to remote_train.yaml")
    p.add_argument("--provider", default=None, choices=sorted(PROVIDERS), help="Compute provider")
    p.add_argument("--gpu", default=None, help="GPU name substring filter (e.g. A100)")
    p.add_argument("--min-vram", type=int, default=None, help="Minimum GPU VRAM in GB")
    p.add_argument("--disk", type=int, default=None, help="Minimum/requested disk in GB (must hold model weights)")
    p.add_argument("--max-price", type=float, default=None, help="Max $/hr cap")
    p.add_argument("--max-hours", type=float, default=None, help="Max wall-clock hours (teardown watchdog)")
    p.add_argument("--verbose", action="store_true", help="Debug logging")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="remote_train", description="Provider-pluggable remote training (Vast.ai).")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Provision, train, pull adapter, and destroy.")
    _add_common(run_p)
    run_p.add_argument("--smoke", action="store_true", help="No-cost self-test (mock provider).")
    run_p.add_argument("--eval", action="store_true", help="Run eval_screenspot.py before/after on the box.")
    run_p.add_argument("--push-hf", default=None, help="HuggingFace repo to upload the adapter to (needs HF_TOKEN).")
    run_p.add_argument("--notify", action="store_true", help="Telegram ping on completion.")
    run_p.set_defaults(func=run_cmd)

    search_p = sub.add_parser("search", help="Dry run: print the cheapest allowed offer, provision nothing.")
    _add_common(search_p)
    search_p.set_defaults(func=search_cmd)

    cleanup_p = sub.add_parser("cleanup", help="Destroy any leftover instance (state file + labeled orphans).")
    _add_common(cleanup_p)
    cleanup_p.set_defaults(func=cleanup_cmd)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except (ProviderError, FileNotFoundError) as e:
        logging.getLogger("remote_train").error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
