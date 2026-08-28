"""Job registry — WHAT runs on the rented box.

The lifecycle (rent → ssh → run → pull → ALWAYS destroy) is job-agnostic. A job only answers
five questions: what to check locally, what to upload, what command to run, what to pull back,
and how to summarize it. Adding a job type = a class + one line in `JOBS`; `lifecycle.py` and
the teardown guarantees never change.

  train — fine-tune: ship trajectories/ + code, run the UNCHANGED train_qwen.py convert+train.
  eval  — benchmark only (no training): ship a harness from `benchmarks.py` and run it.
  grpo  — RL: ship the SFT adapter + rl/ code, run rl_train.py against the Mac env service over a tunnel.
  serve — serve the student on serve_port through a tunnel (for rl.eval / live harness smokes).
"""
import contextlib
import hashlib
import json
import logging
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import yaml

from remote_train.benchmarks import BENCHMARKS, build_benchmark, materialize
from rl_train import build_deploy_command, make_dataset

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

    def resources(self, provider, instance, cfg):
        """Context entered after wait_ready and left BEFORE destroy (tunnels, local helpers)."""
        return contextlib.nullcontext()

    def on_poll(self, provider, instance, cfg) -> bool:
        """Called every poll while the box runs; return True to finish the run now."""
        return False


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
        # coord_space is per-MODEL, not per-dataset: a stock Qwen answers in normalized_1000, a
        # model fine-tuned on this corpus answers in viewport_abs. Scoring one in the other's
        # space silently reports a wrong number, so it is written onto every spec explicitly.
        specs = []
        for m in cfg.eval_models:
            spec = {"label": str(m).split("/")[-1], "model": str(m)}
            if cfg.eval_coord_space:
                spec["coord_space"] = cfg.eval_coord_space
            specs.append(spec)
        self.models_yaml.write_text(yaml.safe_dump(specs, sort_keys=False))
        logger.info(f"benchmark '{self.benchmark.name}' on {[s['model'] for s in specs]}"
                    f"{f' (limit {cfg.eval_limit})' if cfg.eval_limit else ' (full dataset)'}")

    def upload(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        provider.exec(instance, f"mkdir -p {wd}", timeout=60)
        logger.info("Uploading eval harness…")
        dirs = {str(Path(rel).parent) for _, rel in self.local_files if Path(rel).parent != Path(".")}
        for d in sorted(dirs):
            provider.exec(instance, f"mkdir -p {wd}/{d}", timeout=60)
        for local, remote_rel in self.local_files:
            provider.upload(instance, str(local), f"{wd}/{remote_rel}")
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


RL_CODE_FILES = ("rl_train.py", "train_qwen.py", "requirements-rl.txt", "repo_pipeline/__init__.py", "repo_pipeline/corpus_audit.py",
                 "rl/__init__.py", "rl/actions.py", "rl/observation.py", "rl/trainer/__init__.py", "rl/trainer/core.py",
                 "rl/trainer/swift_plugin.py")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _http_ok(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status < 500
    except Exception:
        return False


def _upload_code(provider, instance, cfg, files):
    wd = cfg.remote_workdir
    for rel in files:
        if Path(rel).is_file():
            provider.exec(instance, f"mkdir -p {wd}/{Path(rel).parent}", timeout=60)
            provider.upload(instance, rel, f"{wd}/{rel}")


def _apply_compute(rl, cfg):
    for key, value in (rl.get("compute") or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


class GrpoJob(RemoteJob):
    name = "grpo"

    def __init__(self, cfg):
        self.rl = yaml.safe_load(Path(cfg.rl_config).read_text())
        self.run_id = time.strftime("grpo-%Y%m%d-%H%M%S")
        self.run_dir = Path(cfg.rl_runs_dir) / self.run_id
        self.dataset_path = self.run_dir / "rl_tasks.jsonl"
        self.evals_done: list = []
        self.polls = 0
        self.http_ok = _http_ok

    def preflight(self, cfg) -> None:
        _apply_compute(self.rl, cfg)
        if cfg.env_url:
            self.rl["env_url"] = cfg.env_url
        adapter = Path(cfg.output_dir) / "adapter_config.json"
        if not adapter.is_file():
            raise FileNotFoundError(f"SFT adapter '{adapter}' missing — run `make remote-train` first (spec Phase 2)")
        suite_path = Path("rl/rl_tasks.json")
        if not suite_path.is_file():
            raise FileNotFoundError("rl/rl_tasks.json missing — run `make rl-tasks`")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        rows = make_dataset(suite_path, self.dataset_path, "train", 1, self.rl.get("train_max_expected_actions"))
        if rows == 0:
            raise ValueError("no train tasks in rl/rl_tasks.json")
        env_url = self.rl["env_url"]
        if not env_url.startswith("mock://") and not self.http_ok(f"{env_url.split(',')[0].strip()}/health"):
            raise ConnectionError(f"env service not answering at {env_url}/health — start `make rl-serve` first")
        weights = Path(cfg.output_dir) / "adapter_model.safetensors"
        header = {
            "run_id": self.run_id, "base_model": self.rl["base_model"], "sft_adapter": cfg.output_dir,
            "sft_adapter_sha256": _sha256(weights) if weights.is_file() else "", "rl_config_sha256": _sha256(cfg.rl_config),
            "suite_sha256": json.loads(suite_path.read_text()).get("suite_sha256", ""), "train_rows": rows,
            "git_head": _git_head(), "env_url": env_url, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.run_dir / "run_header.json").write_text(json.dumps(header, indent=2))
        logger.info(f"grpo run {self.run_id}: {rows} train rows, env {env_url}, compute {self.rl.get('compute')}")

    def upload(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        provider.exec(instance, f"mkdir -p {wd}/sft_adapter {wd}/rl_runs/{self.run_id}", timeout=60)
        logger.info("Uploading GRPO bundle (code + SFT adapter + config)…")
        _upload_code(provider, instance, cfg, RL_CODE_FILES + (cfg.rl_config,))
        provider.upload(instance, str(self.dataset_path), f"{wd}/{Path(self.rl['dataset']).name}")
        provider.upload(instance, cfg.output_dir.rstrip("/") + "/", f"{wd}/sft_adapter/")

    def resources(self, provider, instance, cfg):
        compute = self.rl.get("compute") or {}
        return provider.tunnel(instance, cfg, [("R", int(compute.get("env_port", 8900))), ("L", int(compute.get("serve_port", 8001)))])

    def remote_command(self, cfg) -> str:
        wd = cfg.remote_workdir
        return (
            "pip install -r requirements-rl.txt && \\\n"
            f"  python rl_train.py run --config {Path(cfg.rl_config).name} --run-dir {wd}/rl_runs/{self.run_id} --workdir {wd}"
        )

    def run_eval(self, phase):
        tag = "before" if "BEFORE" in phase else "after"
        port = (self.rl.get("compute") or {}).get("serve_port", 8001)
        argv = [sys.executable, "-m", "rl.eval", "--endpoint", f"http://127.0.0.1:{port}/v1", "--model", "policy",
                "--run-id", f"{self.run_id}-{tag}", "--out", str(self.run_dir / "eval_runs")]
        logger.info(f"running student eval ({tag}) through the tunnel: {' '.join(argv)}")
        result = subprocess.run(argv, capture_output=True, text=True)
        (self.run_dir / f"eval_{tag}.log").write_text(result.stdout + result.stderr)
        return {"phase": phase, "returncode": result.returncode}

    def on_poll(self, provider, instance, cfg) -> bool:
        wd = cfg.remote_workdir
        self.polls += 1
        out, _, _ = provider.exec(instance, f"cat {wd}/PHASE 2>/dev/null || true", timeout=30)
        phase = (out.strip().splitlines() or [""])[-1]
        if phase.endswith("_READY") and phase not in self.evals_done:
            self.run_eval(phase)
            self.evals_done.append(phase)
            provider.exec(instance, f"echo {phase} > {wd}/PHASE_ACK", timeout=30)
        if self.polls % max(1, cfg.collect_every_polls) == 0:
            try:
                provider.download(instance, f"{wd}/rl_output/latest/", str(self.run_dir / "adapter_latest") + "/")
            except Exception as e:
                logger.debug(f"latest adapter not available yet: {e}")
        return False

    def collect(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        logger.info("Downloading RL outputs…")
        provider.download(instance, f"{wd}/rl_output/", str(self.run_dir / "rl_output") + "/")
        provider.download(instance, f"{wd}/rl_runs/{self.run_id}/", str(self.run_dir / "box") + "/")

    def results_dest(self, cfg) -> str:
        return str(self.run_dir)

    def report(self, cfg) -> str:
        return f"run {self.run_id}: evals {self.evals_done}; outputs in {self.run_dir}"


class ServeJob(RemoteJob):
    """Rent a box, serve the base (+ adapter) student on serve_port through the tunnel until serve_seconds or Ctrl-C."""
    name = "serve"

    def __init__(self, cfg):
        self.rl = yaml.safe_load(Path(cfg.rl_config).read_text())
        self.started = None
        self.clock = time.time
        self.http_ok = _http_ok
        self.ready_logged = False

    def preflight(self, cfg) -> None:
        _apply_compute(self.rl, cfg)
        if cfg.serve_adapter and not (Path(cfg.serve_adapter) / "adapter_config.json").is_file():
            raise FileNotFoundError(f"adapter '{cfg.serve_adapter}' has no adapter_config.json")

    def upload(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        provider.exec(instance, f"mkdir -p {wd}/serve_adapter", timeout=60)
        _upload_code(provider, instance, cfg, RL_CODE_FILES + (cfg.rl_config,))
        if cfg.serve_adapter:
            provider.upload(instance, cfg.serve_adapter.rstrip("/") + "/", f"{wd}/serve_adapter/")

    def resources(self, provider, instance, cfg):
        return provider.tunnel(instance, cfg, [("L", int((self.rl.get("compute") or {}).get("serve_port", 8001)))])

    def remote_command(self, cfg) -> str:
        wd = cfg.remote_workdir
        adapters = f"{wd}/serve_adapter" if cfg.serve_adapter else None
        deploy = build_deploy_command(dict(self.rl, merged_model=self.rl["base_model"]), adapters=adapters)
        return f"pip install -r requirements-rl.txt && \\\n  MAX_PIXELS={self.rl['max_pixels']} {shlex.join(deploy)}"

    def on_poll(self, provider, instance, cfg) -> bool:
        now = self.clock()
        self.started = self.started if self.started is not None else now
        port = (self.rl.get("compute") or {}).get("serve_port", 8001)
        if not self.ready_logged and self.http_ok(f"http://127.0.0.1:{port}/v1/models"):
            logger.info(f"student endpoint ready at http://127.0.0.1:{port}/v1 — run `make rl-eval` in another shell")
            self.ready_logged = True
        return bool(cfg.serve_seconds) and now - self.started >= cfg.serve_seconds

    def collect(self, provider, instance, cfg) -> None:
        return None

    def results_dest(self, cfg) -> str:
        return str(Path(cfg.rl_runs_dir) / "serve")

    def report(self, cfg) -> str:
        return "serve session ended; instance destroyed"


JOBS: dict = {
    "train": TrainJob,
    "eval": EvalJob,
    "grpo": GrpoJob,
    "serve": ServeJob,
}


def build_job(name: str, cfg) -> RemoteJob:
    if name not in JOBS:
        raise ValueError(f"Unknown job {name!r}. Available: {sorted(JOBS)}")
    return JOBS[name](cfg)
