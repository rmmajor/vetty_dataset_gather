"""Job registry — WHAT runs on the rented box.

The lifecycle (rent → ssh → run → pull → ALWAYS destroy) is job-agnostic. A job only answers
five questions: what to check locally, what to upload, what command to run, what to pull back,
and how to summarize it. Adding a job type = a class + one line in `JOBS`; `lifecycle.py` and
the teardown guarantees never change.

  train — fine-tune: ship trajectories/ + code, run the UNCHANGED train_qwen.py convert+train.
  eval  — benchmark only (no training): ship a harness from `benchmarks.py` and run it.
"""
import logging
import tempfile
from pathlib import Path

import yaml

from remote_train.benchmarks import BENCHMARKS, build_benchmark, materialize

logger = logging.getLogger(__name__)


class RemoteJob:
    """Contract for anything the lifecycle can run on a box."""
    name: str = "base"

    def preflight(self, cfg) -> None:
        """Local checks + staging. Raise to abort BEFORE anything is rented."""
        raise NotImplementedError

    def upload(self, provider, instance, cfg) -> None:
        raise NotImplementedError

    def remote_command(self, cfg) -> str:
        """The shell body to run on the box (wrapped by the lifecycle in a sentinel + log)."""
        raise NotImplementedError

    def collect(self, provider, instance, cfg) -> None:
        """Pull results back down."""
        raise NotImplementedError

    def results_dest(self, cfg) -> str:
        """Local dir results + the remote log land in."""
        raise NotImplementedError

    def report(self, cfg) -> str:
        return ""


class TrainJob(RemoteJob):
    name = "train"

    def __init__(self, cfg):
        self.cfg = cfg

    def preflight(self, cfg) -> None:
        traj = Path(cfg.trajectories_dir)
        if not traj.is_dir() or not any(traj.iterdir()):
            raise FileNotFoundError(f"trajectories dir '{traj}' missing or empty — nothing to train on")
        if not Path(cfg.train_config).is_file():
            raise FileNotFoundError(f"train config '{cfg.train_config}' not found")

    def upload(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        provider.exec(instance, f"mkdir -p {wd}/trajectories", timeout=60)
        logger.info("Uploading training bundle (trajectories + code)…")
        provider.upload(instance, cfg.trajectories_dir.rstrip("/") + "/", f"{wd}/trajectories/")
        for f in ("train_qwen.py", cfg.train_config, "requirements.txt"):
            if Path(f).is_file():
                provider.upload(instance, f, f"{wd}/{Path(f).name}")

    def remote_command(self, cfg) -> str:
        return (
            "pip install -r requirements.txt && \\\n"
            '  pip install "ms-swift[llm]" && \\\n'
            f"  python train_qwen.py convert --input trajectories/ --output training_data/ "
            f"--val-split {cfg.val_split} --exclude-waste-steps --exclude-meandering && \\\n"
            f"  python train_qwen.py train --config {Path(cfg.train_config).name} "
            f"--dataset training_data/sft_dataset.jsonl"
        )

    def collect(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        logger.info("Downloading trained adapter…")
        provider.download(instance, f"{wd}/output/", cfg.output_dir.rstrip("/") + "/")
        adapter = Path(cfg.output_dir) / "adapter_config.json"
        if not adapter.exists():
            logger.warning(f"Expected {adapter} not found after download — inspect ./{cfg.output_dir}")

    def results_dest(self, cfg) -> str:
        return cfg.output_dir

    def report(self, cfg) -> str:
        return f"LoRA adapter in ./{cfg.output_dir}"


class EvalJob(RemoteJob):
    name = "eval"

    def __init__(self, cfg):
        self.benchmark = build_benchmark(cfg.benchmark)
        self.staging = Path(tempfile.mkdtemp(prefix="remote_eval_"))
        self.models_yaml = self.staging / "eval_models_remote.yaml"
        self.local_files: list = []

    def preflight(self, cfg) -> None:
        if not cfg.eval_models:
            raise FileNotFoundError("no models to evaluate — pass --model <id> (repeatable)")
        self.local_files = materialize(self.benchmark, self.staging)
        specs = [{"label": str(m).split("/")[-1], "model": str(m)} for m in cfg.eval_models]
        self.models_yaml.write_text(yaml.safe_dump(specs, sort_keys=False))
        logger.info(f"benchmark '{self.benchmark.name}' on {[s['model'] for s in specs]}"
                    f"{f' (limit {cfg.eval_limit})' if cfg.eval_limit else ' (full dataset)'}")

    def upload(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        provider.exec(instance, f"mkdir -p {wd}", timeout=60)
        logger.info("Uploading eval harness…")
        for path in self.local_files:
            provider.upload(instance, str(path), f"{wd}/{path.name}")
        provider.upload(instance, str(self.models_yaml), f"{wd}/{self.models_yaml.name}")

    def remote_command(self, cfg) -> str:
        return (
            f"pip install -r {self.benchmark.requirements} && \\\n"
            f"  {self.benchmark.build_command(cfg, self.models_yaml.name)}"
        )

    def collect(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        logger.info("Downloading eval results…")
        provider.download(
            instance,
            f"{wd}/{self.benchmark.results_dir}/",
            cfg.eval_results_dir.rstrip("/") + "/",
        )

    def results_dest(self, cfg) -> str:
        return cfg.eval_results_dir

    def report(self, cfg) -> str:
        if not self.benchmark.summarize:
            return ""
        return self.benchmark.summarize(Path(cfg.eval_results_dir))


JOBS: dict = {
    "train": TrainJob,
    "eval": EvalJob,
}


def build_job(name: str, cfg) -> RemoteJob:
    if name not in JOBS:
        raise ValueError(f"Unknown job {name!r}. Available: {sorted(JOBS)}")
    return JOBS[name](cfg)
