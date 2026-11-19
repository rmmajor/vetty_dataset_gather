"""The `run | judge | compare | sources` command surface; every run flag is one row of the RUN_ARGUMENTS table."""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from cua.eval.student.backends import BROWSER_BACKENDS, STUDENT_BACKENDS
from cua.eval.student.config import EVAL_RUNS_DIR, EvalConfig
from cua.eval.student.fingerprint import coord_space_diagnosis
from cua.eval.student.report import render_comparison
from cua.eval.student.run import judge_run, load_run, run_eval
from cua.eval.task_sources import TASK_SOURCES, TaskSourceUnavailable
from cua.pipeline.sandbox import install_sigterm_handler
from cua.settings import EVAL
from cua.student.executor import COORD_SPACES
from cua.student.observation import DEFAULT_OBSERVATION, OBSERVATION_FORMATS

logger = logging.getLogger(__name__)

JUDGE_MID_SCREENSHOTS_HELP = "mid-run frames per verdict on top of the final one; the main cost dial"

# One row per `run` flag, added to the parser in this order — a new knob is a row, never a new branch.
RUN_ARGUMENTS = (
    (
        ("--task-source",),
        {"default": "online_mind2web", "choices": sorted(TASK_SOURCES), "help": "which task set to evaluate on"},
    ),
    (("--tasks-file",), {"default": "", "help": "JSON/JSONL of EvalTask rows, bypassing the registry loader"}),
    (("--limit",), {"type": int, "default": None, "help": "evaluate only the first N tasks"}),
    (
        ("--label",),
        {"default": "baseline", "help": "run label, e.g. 'before' / 'after' — names the run dir and the report"},
    ),
    (("--run-id",), {"default": "", "help": "override the generated run directory name"}),
    (("--student",), {"default": "openai", "choices": sorted(STUDENT_BACKENDS)}),
    (("--endpoint",), {"default": EVAL.endpoint, "help": "OpenAI-compatible base URL (e.g. http://host:8000/v1)"}),
    (("--model",), {"default": EVAL.model, "help": "served model id"}),
    (
        ("--adapter",),
        {"default": EVAL.adapter, "help": "LoRA/checkpoint path served behind the endpoint (recorded in the report)"},
    ),
    (("--api-key",), {"default": EVAL.api_key}),
    (("--temperature",), {"type": float, "default": 0.0}),
    (("--max-tokens",), {"type": int, "default": 512}),
    (("--student-timeout",), {"type": int, "default": 180}),
    (
        ("--student-retries",),
        {"type": int, "default": 2, "help": "retries for transport errors only; HTTP errors are surfaced"},
    ),
    (("--mock-trajectory",), {"default": "", "help": "trajectory.json replayed by the 'mock' student backend"}),
    (
        ("--coord-space",),
        {
            "default": EVAL.coord_space,
            "choices": sorted(COORD_SPACES),
            "help": "the convention THIS model answers in: a stock Qwen and an sft3 checkpoint emit "
            "normalized_1000 (the prompt's 0-1000 grid); an sft-v1/v2 checkpoint emits "
            "viewport_abs (1280x720 pixels). Set it explicitly per model — it is never "
            "auto-detected",
        },
    ),
    (
        ("--type-mode",),
        {
            "default": "replace",
            "choices": ("replace", "append"),
            "help": "driver typing mode; replace selects the focused field's text before typing, as capture and RL do",
        },
    ),
    (
        ("--strict-action-grammar",),
        {
            "action": "store_true",
            "help": "accept ONLY the canonical form format_action_for_training emits. Off "
            "by default: any unambiguous spelling of a call is accepted, and how "
            "canonical each model's output was is reported as canonical_form_rate "
            "instead of being charged to its success rate",
        },
    ),
    (
        ("--max-unproductive-steps",),
        {
            "type": int,
            "default": EVAL.max_unproductive_steps,
            "help": "abort the episode as 'stuck' after this many steps in a row that execute no action at all",
        },
    ),
    (
        ("--max-parse-failures",),
        {
            "type": int,
            "default": EVAL.max_parse_failures,
            "help": "abort as 'parse_failure_abort' after this many unparseable steps in a row",
        },
    ),
    (
        ("--max-steps",),
        {
            "type": int,
            "default": 0,
            "help": "FLAT step cap per episode; 0 derives one from each task's reference_length",
        },
    ),
    (
        ("--step-cap-multiple",),
        {"type": float, "default": 3.0, "help": "derived cap = this x the task's reference_length"},
    ),
    (("--step-cap-floor",), {"type": int, "default": 15}),
    (("--step-cap-ceiling",), {"type": int, "default": 50}),
    (
        ("--stratify",),
        {
            "action": "store_true",
            "help": "with --limit, keep the task set's easy/medium/hard mix instead of taking the first N rows",
        },
    ),
    (("--episode-timeout",), {"type": int, "default": 900}),
    (
        ("--screenshot-format",),
        {
            "default": "png",
            "choices": ("png", "jpeg"),
            "help": "png only under sft3: effect annotations compare consecutive frames pixel-wise",
        },
    ),
    (
        ("--no-judge",),
        {
            "action": "store_true",
            "help": "record episodes and DEFER the verdict; a later resume judges them off "
            "the GPU, so a usage limit can never burn rental hours",
        },
    ),
    (("--judge-mid-screenshots",), {"type": int, "default": 2, "help": JUDGE_MID_SCREENSHOTS_HELP}),
    (
        ("--observation",),
        {
            "default": DEFAULT_OBSERVATION,
            "choices": OBSERVATION_FORMATS,
            "help": "observation layout: v3 sends prev+current screenshots, v2 sends current "
            "only (required for sft-v1/v2 and any GRPO LoRA over them)",
        },
    ),
    (
        ("--parallel",),
        {"type": int, "default": 1, "help": "concurrent sandboxes (clamped to what the Docker VM can hold)"},
    ),
    (("--no-resume",), {"action": "store_true", "help": "re-run episodes that already have an episode.json"}),
    (("--dataset-revision",), {"default": "main"}),
    (
        ("--trajectories-dir",),
        {"default": "trajectories", "help": "corpus checked by the held_out source to prove a task is unseen"},
    ),
    (
        ("--loop-guard-repeats",),
        {"type": int, "default": 0, "help": "driver loop guard for eval; 0 lets a stuck model actually repeat itself"},
    ),
    (("--sandbox-image",), {"default": ""}),
    (
        ("--browser-backend",),
        {
            "default": "docker",
            "choices": sorted(BROWSER_BACKENDS),
            "help": "where the browser runs, i.e. which IP the sites see. "
            "`vast_residential` rents a US consumer-ISP box and destroys it after.",
        },
    ),
    (
        ("--browser-host",),
        {"default": "", "help": "remote_ssh backend: host or host:port of an already-running box"},
    ),
)


def _add_run_arguments(parser):
    for flags, kwargs in RUN_ARGUMENTS:
        parser.add_argument(*flags, **kwargs)


def _config_from_args(args) -> EvalConfig:
    return EvalConfig(
        task_source=args.task_source,
        tasks_file=args.tasks_file,
        limit=args.limit,
        label=args.label,
        run_id=args.run_id,
        student=args.student,
        endpoint=args.endpoint,
        model=args.model,
        adapter=args.adapter,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        student_timeout_s=args.student_timeout,
        student_retries=args.student_retries,
        mock_trajectory=args.mock_trajectory,
        coord_space=args.coord_space,
        type_mode=args.type_mode,
        tolerant_grammar=not args.strict_action_grammar,
        max_unproductive_steps=args.max_unproductive_steps,
        max_parse_failures=args.max_parse_failures,
        max_steps=args.max_steps,
        step_cap_multiple=args.step_cap_multiple,
        step_cap_floor=args.step_cap_floor,
        step_cap_ceiling=args.step_cap_ceiling,
        stratify=args.stratify,
        episode_timeout_s=args.episode_timeout,
        screenshot_format=args.screenshot_format,
        judge=not args.no_judge,
        judge_mid_screenshots=args.judge_mid_screenshots,
        parallel=args.parallel,
        observation=args.observation,
        resume=not args.no_resume,
        dataset_revision=args.dataset_revision,
        trajectories_dir=args.trajectories_dir,
        loop_guard_repeats=args.loop_guard_repeats,
        sandbox_image=args.sandbox_image,
        browser_backend=args.browser_backend,
        browser_host=args.browser_host,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a served computer-use student model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a task set against a served model")
    _add_run_arguments(run_parser)

    judge_parser = subparsers.add_parser("judge", help="score the episodes a --no-judge run deferred")
    judge_parser.add_argument("run_dir")
    judge_parser.add_argument("--judge-mid-screenshots", type=int, default=2, help=JUDGE_MID_SCREENSHOTS_HELP)

    compare_parser = subparsers.add_parser("compare", help="render a before/after table")
    compare_parser.add_argument("runs", nargs="+", help="two or more eval_runs/<id> directories")
    compare_parser.add_argument(
        "--output", default="", help="write the table here (default: print, plus comparison.md next to the first run)"
    )

    list_parser = subparsers.add_parser("sources", help="list the registered task sources")
    list_parser.add_argument("--json", action="store_true")
    return parser


def _cmd_sources(args):
    rows = [
        {
            "name": s.name,
            "description": s.description,
            "default_step_cap": s.default_step_cap,
            "live_web": s.live_web,
            "dataset": s.hf_dataset or s.access_url,
        }
        for s in TASK_SOURCES.values()
    ]
    print(
        json.dumps(rows, indent=2)
        if args.json
        else "\n".join(f"{r['name']:<20} step cap {r['default_step_cap']:<4} {r['description']}" for r in rows)
    )


def _cmd_compare(args):
    runs = [load_run(path) for path in args.runs]
    table = render_comparison(runs)
    print(table)
    output = Path(args.output) if args.output else Path(args.runs[0]).parent / "comparison.md"
    output.write_text(table)
    print(f"\nwrote {output}")


def _cmd_judge(args):
    asyncio.run(judge_run(Path(args.run_dir), EvalConfig(judge_mid_screenshots=args.judge_mid_screenshots)))


def _cmd_run(args, parser):
    try:
        cfg = _config_from_args(args)
    except ValueError as e:
        parser.error(str(e))
    try:
        results = asyncio.run(run_eval(cfg))
    except TaskSourceUnavailable as e:
        logger.error(str(e))
        raise SystemExit(2) from e
    metrics = results["metrics"]
    print(
        f"\nverified task success: {metrics['verified_success_rate'] * 100:.1f}% across "
        f"{metrics['episodes']} {results['harness']['task_source']} tasks, "
        f"parse-failure rate {metrics['parse_failure_rate'] * 100:.1f}%, "
        f"site-blocked {metrics['site_blocked_rate'] * 100:.1f}%"
    )
    warning = coord_space_diagnosis(results["harness"], metrics)
    if warning:
        print(f"\n!! {warning}")
    print(f"report: {EVAL_RUNS_DIR / results['run_id'] / 'report.md'}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sources":
        return _cmd_sources(args)
    if args.command == "compare":
        return _cmd_compare(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    install_sigterm_handler()

    if args.command == "judge":
        return _cmd_judge(args)
    return _cmd_run(args, parser)
