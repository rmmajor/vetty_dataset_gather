"""CLI for provider-pluggable remote jobs (train / eval) on rented GPUs.

    python -m remote_train train                       # rent -> SFT -> pull adapter -> destroy
    python -m remote_train eval --model Qwen/Qwen3.5-9B-Base --model Qwen/Qwen3.5-9B --limit 100
    python -m remote_train run --smoke                 # no-cost self-test (mock provider)
    python -m remote_train grpo                        # rent -> GRPO against the local env service -> pull adapters -> destroy
    python -m remote_train serve --adapter output      # rent -> serve the student on 127.0.0.1:8001 via tunnel -> destroy
    python -m remote_train search                      # dry run: cheapest allowed offer, rents nothing
    python -m remote_train cleanup                     # destroy instances whose owning run is dead or released (--all: every labeled box)
    python -m remote_train release --instance ID       # a supervisor hands its box to cleanup instead of editing the state file

`--provider` / `--benchmark` choices come straight from the registries, so a new provider or a
new eval shows up in the CLI with no changes here.
"""
import argparse
import logging
import sys

from remote_train import lifecycle, state
from remote_train.base import PROVIDERS, ProviderError, build_provider
from remote_train.benchmarks import BENCHMARKS
from remote_train.config import load_config
from remote_train.jobs import JOBS


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cli_overrides(args) -> dict:
    models = getattr(args, "models", None)
    return {
        "provider": getattr(args, "provider", None),
        "gpu_name": getattr(args, "gpu", None),
        "min_vram_gb": getattr(args, "min_vram", None),
        "disk_gb": getattr(args, "disk", None),
        "max_price_per_hour": getattr(args, "max_price", None),
        "max_runtime_hours": getattr(args, "max_hours", None),
        "resume_from": getattr(args, "resume_from", None),
        "trajectories_dir": getattr(args, "trajectories_dir", None),
        "keep_on_signal": getattr(args, "keep_on_signal", None),
        "benchmark": getattr(args, "benchmark", None),
        "eval_limit": getattr(args, "limit", None),
        "eval_models": tuple(models) if models else None,
        "eval_dataset": getattr(args, "dataset", None),
        "eval_split": getattr(args, "split", None),
        "eval_coord_space": getattr(args, "coord_space", None),
        "push_hf_repo": getattr(args, "push_hf", None),
        "notify": True if getattr(args, "notify", False) else None,
        "rl_config": getattr(args, "rl_config", None),
        "env_url": getattr(args, "env_url", None),
        "serve_adapter": getattr(args, "adapter", None),
        "serve_seconds": getattr(args, "for_seconds", None),
    }


def _job_cmd(args, job_name: str) -> int:
    cfg = load_config(args.config, _cli_overrides(args))
    provider = build_provider(cfg.provider, cfg)
    return lifecycle.run_remote_job(provider, cfg, job_name)


def train_cmd(args) -> int:
    if getattr(args, "smoke", False):
        return lifecycle.run_smoke_test()
    return _job_cmd(args, "train")


def eval_cmd(args) -> int:
    return _job_cmd(args, "eval")


def grpo_cmd(args) -> int:
    return _job_cmd(args, "grpo")


def serve_cmd(args) -> int:
    return _job_cmd(args, "serve")


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
    return lifecycle.cleanup(provider, cfg, sweep_all=args.sweep_all)


def release_cmd(args) -> int:
    known = {str(r.get("id")) for r in state.read_instances()}
    if args.instance not in known:
        raise ValueError(f"instance {args.instance} is not in {state.STATE_FILE} (recorded: {sorted(known) or 'none'})")
    state.mark_abandoned(args.instance)
    return 0


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
    parser = argparse.ArgumentParser(prog="remote_train", description="Remote GPU jobs (train / eval).")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Fine-tune on a rented GPU, pull the adapter, destroy.")
    _add_common(train_p)
    train_p.add_argument("--resume-from", dest="resume_from", default=None,
                         help="Local trainer checkpoint dir to continue from instead of step 0.")
    train_p.add_argument("--trajectories-dir", dest="trajectories_dir", default=None,
                         help="Corpus directory to ship (default: ./trajectories).")
    train_p.add_argument("--keep-on-signal", dest="keep_on_signal", action="store_true", default=None,
                         help="If this supervisor is killed, leave the box running so the detached "
                              "job finishes; recover with `remote_train cleanup`.")
    train_p.add_argument("--push-hf", default=None, help="HuggingFace repo to upload the adapter to.")
    train_p.add_argument("--notify", action="store_true", help="Telegram ping on completion.")
    train_p.set_defaults(func=train_cmd)

    # `run` kept as an alias of `train` so `run --smoke` (the documented self-test) still works.
    run_p = sub.add_parser("run", help="Alias of `train`; `run --smoke` is the no-cost self-test.")
    _add_common(run_p)
    run_p.add_argument("--smoke", action="store_true", help="No-cost self-test (mock provider).")
    run_p.add_argument("--push-hf", default=None, help=argparse.SUPPRESS)
    run_p.add_argument("--notify", action="store_true", help=argparse.SUPPRESS)
    run_p.set_defaults(func=train_cmd)

    eval_p = sub.add_parser("eval", help="Benchmark model(s) on a rented GPU. No training.")
    _add_common(eval_p)
    eval_p.add_argument("--model", action="append", dest="models",
                        help="Model id to evaluate (repeatable, e.g. Qwen/Qwen3.5-9B-Base).")
    eval_p.add_argument("--benchmark", default=None, choices=sorted(BENCHMARKS), help="Which eval to run.")
    eval_p.add_argument("--limit", type=int, default=None, help="Cap samples (omit = full dataset).")
    eval_p.add_argument("--dataset", default=None, help="Override the benchmark's default HF dataset.")
    eval_p.add_argument("--split", default=None, help="Override the dataset split.")
    eval_p.add_argument("--coord-space", default=None,
                        choices=["normalized_1000", "viewport_abs", "native_abs"],
                        help="Coordinate convention THE MODEL answers in. normalized_1000 for a "
                             "stock Qwen; viewport_abs for a checkpoint fine-tuned on this corpus.")
    eval_p.add_argument("--notify", action="store_true", help="Telegram ping on completion.")
    eval_p.set_defaults(func=eval_cmd)

    grpo_p = sub.add_parser("grpo", help="GRPO on a rented GPU against the local env service (make rl-serve first); ALWAYS destroys.")
    _add_common(grpo_p)
    grpo_p.add_argument("--rl-config", default=None, help="RL run config (default rl_config.yaml)")
    grpo_p.add_argument("--env-url", default=None, help="Env service URL as seen from the box (default from rl_config; mock:// for smokes)")
    grpo_p.add_argument("--notify", action="store_true", help="Telegram ping on completion.")
    grpo_p.set_defaults(func=grpo_cmd)

    serve_p = sub.add_parser("serve", help="Serve the student on 127.0.0.1:8001 through a tunnel until --for-seconds or Ctrl-C.")
    _add_common(serve_p)
    serve_p.add_argument("--rl-config", default=None, help="RL run config (default rl_config.yaml)")
    serve_p.add_argument("--adapter", default=None, help="LoRA adapter dir to serve on top of the base (e.g. output)")
    serve_p.add_argument("--for-seconds", type=int, default=None, help="Auto-stop after N seconds (0 = until Ctrl-C)")
    serve_p.set_defaults(func=serve_cmd)

    search_p = sub.add_parser("search", help="Dry run: print the cheapest allowed offer, provision nothing.")
    _add_common(search_p)
    search_p.set_defaults(func=search_cmd)

    cleanup_p = sub.add_parser("cleanup", help="Destroy instances abandoned by dead runs (see --all).")
    _add_common(cleanup_p)
    cleanup_p.add_argument("--all", action="store_true", dest="sweep_all",
                           help="Also destroy every instance carrying the run label, INCLUDING boxes a "
                                "concurrent run is using. Only when you know no sibling run is live.")
    cleanup_p.set_defaults(func=cleanup_cmd)

    release_p = sub.add_parser("release", help="Hand a recorded instance over to `cleanup` (writes the pid null sentinel; destroys nothing).")
    release_p.add_argument("--instance", required=True, help="Instance id as recorded in the state file.")
    release_p.add_argument("--verbose", action="store_true", help="Debug logging")
    release_p.set_defaults(func=release_cmd)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except (ProviderError, FileNotFoundError, ValueError) as e:
        logging.getLogger("remote_train").error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
