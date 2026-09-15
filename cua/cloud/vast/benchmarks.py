"""Benchmark registry: one `Benchmark` entry declares the files, deps, command, results dir and summarizer of an eval that runs on a rented box."""

import json
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Benchmark:
    name: str
    files: tuple  # harness files that must exist on the box (incl. what an editable install reads)
    requirements: str  # pip install target for this benchmark's dependencies on the box
    results_dir: str  # dir (relative to remote workdir) the harness writes results into
    command: str  # template; gets {models_yaml} {results_dir}
    dataset: str = ""  # default HF dataset for this benchmark (cfg.eval_dataset overrides)
    split: str = ""
    extra_args: tuple = ()  # benchmark-specific flags the harness needs to score correctly
    source_ref: str = ""  # git ref to fetch missing files from
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
            lines.append(f"      type:{data_type:<12s} {g['accuracy'] * 100:5.1f}% ({g['hits']}/{g['total']})")
    return "\n".join(lines) if lines else "(comparison.json contained no summaries)"


BENCHMARKS: dict = {
    "screenspot": Benchmark(
        name="screenspot",
        # The whole train/eval prompt chain screenspot imports: Pillow-only, no cua.pipeline, no pydantic.
        files=(
            "cua/__init__.py",
            "cua/eval/__init__.py",
            "cua/eval/screenspot/__init__.py",
            "cua/eval/screenspot/__main__.py",
            "cua/eval/screenspot/cli.py",
            "cua/eval/screenspot/config.py",
            "cua/eval/screenspot/dataset.py",
            "cua/eval/screenspot/predictors.py",
            "cua/eval/screenspot/report.py",
            "cua/eval/screenspot/scoring.py",
            "cua/eval/screenspot/smoke.py",
            "cua/prompt/__init__.py",
            "cua/prompt/coords.py",
            "cua/prompt/effects.py",
            "cua/prompt/format.py",
            "cua/settings.py",
            "cua/trajectory/__init__.py",
            "cua/trajectory/actions.py",
            "pyproject.toml",
            "README.md",
        ),
        requirements=".[eval]",
        results_dir="eval_results",
        command="python -m cua.eval.screenspot run --models-config {models_yaml} --output-dir {results_dir}",
        # NOT the harness default (os-copilot/ScreenSpot-v2), which load_dataset() cannot read at all.
        dataset="rootsautomation/ScreenSpot",
        split="test",
        # Properties of THIS mirror, pinned because getting one wrong reports a confident wrong number (coord_space is per MODEL, so EvalJob writes it).
        extra_args=("--bbox-format", "xyxy", "--bbox-normalized"),
        summarize=_summarize_screenspot,
    ),
}


def build_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name!r}. Available: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]


def materialize(benchmark: Benchmark, staging: Path) -> list:
    """[(local_path, remote_relpath)] for every harness file, keeping each DECLARED relative path so an imported package lands as a package."""
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
            capture_output=True,
            text=True,
            timeout=60,
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
