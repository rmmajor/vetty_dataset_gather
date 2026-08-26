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
    command: str           # template; gets {models_yaml} {results_dir}
    dataset: str = ""      # default HF dataset for this benchmark (cfg.eval_dataset overrides)
    split: str = ""
    extra_args: tuple = () # benchmark-specific flags the harness needs to score correctly
    source_ref: str = ""   # git ref to fetch missing files from
    summarize: Callable = None  # (local_results_dir: Path) -> str

    def build_command(self, cfg, models_yaml: str) -> str:
        parts = [self.command.format(models_yaml=models_yaml, results_dir=self.results_dir)]
        dataset = cfg.eval_dataset or self.dataset
        split = cfg.eval_split or self.split
        if dataset:
            parts.append(f"--dataset {dataset}")
        if split:
            parts.append(f"--split {split}")
        if cfg.eval_limit:
            parts.append(f"--limit {cfg.eval_limit}")
        parts.extend(self.extra_args)
        if cfg.eval_extra_args:
            parts.append(cfg.eval_extra_args)
        return " ".join(str(p) for p in parts if p)


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
        # eval_screenspot imports SYSTEM_PROMPT from train_qwen (train/eval prompt parity), and
        # train_qwen has a TOP-LEVEL `from repo_pipeline.corpus_audit import ...`, so that package
        # must ship too or the eval dies on import. corpus_audit is stdlib-only and __init__ is
        # empty, so this is two small files, not the whole pipeline.
        files=(
            "eval_screenspot.py",
            "train_qwen.py",
            "requirements-eval.txt",
            "repo_pipeline/__init__.py",
            "repo_pipeline/corpus_audit.py",
        ),
        requirements="requirements-eval.txt",
        results_dir="eval_results",
        command="python eval_screenspot.py run --models-config {models_yaml} --output-dir {results_dir}",
        # NOT the harness default (os-copilot/ScreenSpot-v2): that repo is raw JSON + an image
        # zip with no `test` split and no image column, so load_dataset() cannot read it.
        # This fork ships real parquet with a 1272-row `test` split and a proper image column.
        dataset="rootsautomation/ScreenSpot",
        split="test",
        # Pinned explicitly even though they now match the harness defaults: these are properties
        # of THIS dataset, and getting one wrong reports a confident wrong number rather than
        # crashing. Both were verified against real predictions, not assumed.
        #   xyxy            this mirror stores (x1,y1,x2,y2), not xywh.
        #   bbox-normalized its bbox is 0-1, NOT pixels. Without it every target collapses to a
        #                   sub-pixel box at the origin -> 0% accuracy regardless of the model.
        # coord_space is deliberately NOT set here: it is a property of the MODEL, not the
        # dataset, and is written per-spec by jobs.EvalJob (cfg.eval_coord_space).
        extra_args=("--bbox-format", "xyxy", "--bbox-normalized"),
        summarize=_summarize_screenspot,
    ),
}


def build_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name!r}. Available: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]


def materialize(benchmark: Benchmark, staging: Path) -> list:
    """[(local_path, remote_relpath)] for every harness file; missing ones fetched from `source_ref`.

    The remote path keeps the DECLARED relative path, not the basename — a harness that imports a
    package (eval_screenspot -> train_qwen -> repo_pipeline.corpus_audit) needs that package to land
    at `repo_pipeline/corpus_audit.py` on the box, not flattened next to it.
    """
    staging.mkdir(parents=True, exist_ok=True)
    pairs = []
    for name in benchmark.files:
        local = Path(name)
        if local.is_file():
            pairs.append((local, name))
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
        staged = staging / Path(name).name
        staged.write_text(result.stdout)
        logger.info(f"materialized {name} from {benchmark.source_ref}")
        pairs.append((staged, name))
    return pairs
