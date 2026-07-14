"""Benchmark registry — adding an eval is a DATA entry, not a code change.

A `Benchmark` declares everything the lifecycle needs to run one evaluation on a rented box:
which harness files must land there, what to pip-install, the command to run, where results
are written, and how to summarize them locally. Add a new eval by appending one `Benchmark`
to `BENCHMARKS`; `jobs.py`, `lifecycle.py`, and the CLI never change (same registry idiom as
`PROVIDERS` / eval_screenspot's `PREDICTOR_BACKENDS`).

`source_ref` materializes a harness that isn't on this branch yet from a git ref — ScreenSpot
lives on the PR #12 branch until it merges, so the benchmark stays runnable without vendoring
a copy of the file here.
"""
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Benchmark:
    name: str
    files: tuple           # harness files that must exist on the box (incl. the requirements file)
    requirements: str      # pip requirements file to install on the box
    results_dir: str       # dir (relative to remote workdir) the harness writes results into
    command: str           # template; gets {models_yaml} {limit} {results_dir}
    source_ref: str = ""   # git ref to fetch missing files from
    summarize: Callable = None  # (local_results_dir: Path) -> str

    def build_command(self, cfg, models_yaml: str) -> str:
        limit = f"--limit {cfg.eval_limit}" if cfg.eval_limit else ""
        return self.command.format(
            models_yaml=models_yaml, limit=limit, results_dir=self.results_dir
        ).strip()


def _summarize_screenspot(results_dir: Path) -> str:
    path = results_dir / "comparison.json"
    if not path.exists():
        return f"(no {path} — see the pulled train.log for the harness output)"
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return f"(comparison.json unreadable: {e})"

    summaries = data.get("summaries", data)
    lines = []
    for label, s in summaries.items():
        if not isinstance(s, dict) or "accuracy" not in s:
            continue
        lines.append(
            f"  {label:<18s} accuracy {s['accuracy'] * 100:5.1f}%   "
            f"parse-rate {s['parse_rate'] * 100:5.1f}%   n={s['total']}"
        )
        for data_type, g in sorted((s.get("by_data_type") or {}).items()):
            lines.append(
                f"      type:{data_type:<12s} {g['accuracy'] * 100:5.1f}% ({g['hits']}/{g['total']})"
            )
    return "\n".join(lines) if lines else "(comparison.json contained no summaries)"


BENCHMARKS: dict = {
    "screenspot": Benchmark(
        name="screenspot",
        # eval_screenspot imports SYSTEM_PROMPT from train_qwen for train/eval prompt parity,
        # so train_qwen.py has to ship too.
        files=("eval_screenspot.py", "train_qwen.py", "requirements-eval.txt"),
        requirements="requirements-eval.txt",
        results_dir="eval_results",
        command="python eval_screenspot.py run --models-config {models_yaml} {limit} --output-dir {results_dir}",
        source_ref="origin/worktree-feat+screenspot-eval",
        summarize=_summarize_screenspot,
    ),
}


def build_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name!r}. Available: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]


def materialize(benchmark: Benchmark, staging: Path) -> list:
    """Every harness file as a local path, fetching whatever is missing from `source_ref`."""
    staging.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in benchmark.files:
        local = Path(name)
        if local.is_file():
            paths.append(local)
            continue
        if not benchmark.source_ref:
            raise FileNotFoundError(
                f"{name!r} not found locally and benchmark {benchmark.name!r} declares no source_ref"
            )
        result = subprocess.run(
            ["git", "show", f"{benchmark.source_ref}:{name}"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise FileNotFoundError(
                f"could not fetch {name!r} from {benchmark.source_ref}: {result.stderr.strip()[:200]}"
            )
        staged = staging / name
        staged.write_text(result.stdout)
        logger.info(f"materialized {name} from {benchmark.source_ref}")
        paths.append(staged)
    return paths
