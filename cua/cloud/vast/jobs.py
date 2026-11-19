"""Job registry, the WHAT that runs on a rented box: `train`, `eval`, `grpo` against the Mac env service, `serve`."""

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

from cua.cloud.ignore import gitignored_excludes
from cua.cloud.vast.benchmarks import build_benchmark, materialize
from cua.rl.train import (
    build_deploy_command,
    build_merge_command,
    build_vllm_serve_command,
    load_task_ids,
    make_dataset,
)
from cua.sft.recipe import convert_command, train_command

logger = logging.getLogger(__name__)


class RemoteJob:
    """Contract for anything the lifecycle can run on a box: five questions, answered once per `JOBS` entry."""

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
        if not Path("cua").is_dir():
            raise FileNotFoundError("cua/ not found — the box installs the package and would die at import")
        # Fail before renting: a resume without optimizer state silently restarts from step 0.
        if cfg.resume_from:
            ckpt = Path(cfg.resume_from)
            missing = [f for f in ("optimizer.pt", "trainer_state.json") if not (ckpt / f).is_file()]
            if missing:
                raise FileNotFoundError(f"resume checkpoint '{ckpt}' missing {missing}")

    def upload(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        provider.exec(instance, f"mkdir -p {wd}/trajectories", timeout=60)
        logger.info("Uploading training bundle (trajectories + code)…")
        provider.upload(instance, cfg.trajectories_dir.rstrip("/") + "/", f"{wd}/trajectories/")
        if Path(cfg.train_config).is_file():
            provider.upload(instance, cfg.train_config, f"{wd}/{Path(cfg.train_config).name}")
        upload_package(provider, instance, cfg)
        if cfg.resume_from:
            logger.info(f"Uploading resume checkpoint {cfg.resume_from}…")
            provider.upload(instance, cfg.resume_from.rstrip("/") + "/", f"{wd}/{self._resume_name(cfg)}/")

    @staticmethod
    def _resume_name(cfg) -> str:
        return Path(cfg.resume_from.rstrip("/")).name

    def remote_command(self, cfg) -> str:
        resume = ("--resume_from_checkpoint", self._resume_name(cfg)) if cfg.resume_from else ()
        convert = " ".join(convert_command("python", "trajectories/", "training_data/", cfg.val_split))
        train = " ".join(
            train_command("python", Path(cfg.train_config).name, "training_data/sft_dataset.jsonl", resume)
        )
        return (
            # Modal's proven train stack, NOT ".[pipeline]": installing that over the image clobbers torch.
            'pip install --no-cache-dir "ms-swift[llm]>=4.5,<5" "transformers>=4.57" '
            '"datasets>=3.0" "qwen_vl_utils>=0.0.14" "peft>=0.14" "accelerate>=1.0" '
            "pyyaml Pillow tqdm && \\\n"
            "  pip install --no-cache-dir --no-deps -e . && \\\n"
            f"  {convert} && \\\n"
            f"  {train}"
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
        # coord_space is per-MODEL, never per-dataset, so it is written onto every spec explicitly.
        specs = []
        for m in cfg.eval_models:
            spec = {"label": str(m).split("/")[-1], "model": str(m)}
            if cfg.eval_coord_space:
                spec["coord_space"] = cfg.eval_coord_space
            specs.append(spec)
        self.models_yaml.write_text(yaml.safe_dump(specs, sort_keys=False))
        logger.info(
            f"benchmark '{self.benchmark.name}' on {[s['model'] for s in specs]}"
            f"{f' (limit {cfg.eval_limit})' if cfg.eval_limit else ' (full dataset)'}"
        )

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
            f"pip install -e '{self.benchmark.requirements}' && \\\n"
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


# What `pip install -e ".[rl]" -c constraints/rl-modal.txt` reads on the box; README.md is pyproject's declared readme.
PACKAGE_FILES = ("pyproject.toml", "README.md", "constraints/rl-modal.txt", "corpus_health.json")


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


def upload_package(provider, instance, cfg):
    """The installable package plus what an editable install reads, minus gitignored paths: no manifest to go stale."""
    wd = cfg.remote_workdir
    provider.exec(instance, f"mkdir -p {wd}/cua", timeout=60)
    provider.upload(instance, "cua/", f"{wd}/cua/", exclude=gitignored_excludes())
    _upload_code(provider, instance, cfg, PACKAGE_FILES)


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
        # Suite and frontier band are config-driven: training on every train task is what starved grpo-v2 of gradient.
        suite_path = Path(self.rl.get("tasks_file") or "cua/rl/rl_tasks.json")
        if not suite_path.is_file():
            raise FileNotFoundError(f"{suite_path} missing — run `make rl-tasks` or `make rl-candidates`")
        selection = self.rl.get("task_ids_file")
        task_ids = load_task_ids(selection) if selection else None
        if selection and not task_ids:
            raise ValueError(f"{selection} selected no tasks — the frontier band is empty")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        rows = make_dataset(
            suite_path, self.dataset_path, "train", 1, self.rl.get("train_max_expected_actions"), task_ids
        )
        if rows == 0:
            raise ValueError(f"no train tasks in {suite_path} after the band filter")
        env_url = self.rl["env_url"]
        if not env_url.startswith("mock://"):
            base = env_url.split(",")[0].strip()
            if not self.http_ok(f"{base}/health"):
                raise ConnectionError(f"env service not answering at {base}/health — start `make rl-serve` first")
            self.assert_tasks_served(base, task_ids or self.dataset_task_ids())
        weights = Path(cfg.output_dir) / "adapter_model.safetensors"
        header = {
            "run_id": self.run_id,
            "base_model": self.rl["base_model"],
            "sft_adapter": cfg.output_dir,
            "sft_adapter_sha256": _sha256(weights) if weights.is_file() else "",
            "rl_config_sha256": _sha256(cfg.rl_config),
            "suite_sha256": json.loads(suite_path.read_text()).get("suite_sha256", ""),
            "train_rows": rows,
            "git_head": _git_head(),
            "env_url": env_url,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.run_dir / "run_header.json").write_text(json.dumps(header, indent=2))
        logger.info(f"grpo run {self.run_id}: {rows} train rows, env {env_url}, compute {self.rl.get('compute')}")

    def dataset_task_ids(self) -> list:
        with open(self.dataset_path) as f:
            return [json.loads(line).get("task_id") for line in f if line.strip()]

    def assert_tasks_served(self, base_url, task_ids) -> None:
        """A task the service does not serve becomes a fabricated reward 0.0, so catch it before renting a box."""
        try:
            with urllib.request.urlopen(f"{base_url}/tasks", timeout=15) as resp:
                served = {t.get("task_id") for t in json.loads(resp.read().decode()).get("tasks", [])}
        except Exception as e:
            raise ConnectionError(f"could not list served tasks at {base_url}/tasks: {e}") from e
        missing = sorted({t for t in task_ids if t} - served)
        if missing:
            raise ValueError(
                f"the env service does not serve {len(missing)} of the run's tasks (e.g. {missing[:3]}) — "
                f"it was started with a different --tasks/--split. Restart it with the run's suite."
            )
        logger.info(f"env service serves all {len(set(task_ids))} task ids of this run")

    def upload(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        provider.exec(instance, f"mkdir -p {wd}/sft_adapter {wd}/rl_runs/{self.run_id}", timeout=60)
        logger.info("Uploading GRPO bundle (code + SFT adapter + config)…")
        upload_package(provider, instance, cfg)
        _upload_code(provider, instance, cfg, (cfg.rl_config,))
        provider.upload(instance, str(self.dataset_path), f"{wd}/{Path(self.rl['dataset']).name}")
        provider.upload(instance, cfg.output_dir.rstrip("/") + "/", f"{wd}/sft_adapter/")

    def resources(self, provider, instance, cfg):
        compute = self.rl.get("compute") or {}
        return provider.tunnel(
            instance, cfg, [("R", int(compute.get("env_port", 8900))), ("L", int(compute.get("serve_port", 8001)))]
        )

    def remote_command(self, cfg) -> str:
        wd = cfg.remote_workdir
        # The uv/--break-system-packages preamble and the xet download knob are explained in docs/architecture.md.
        return (
            "pip install -q --break-system-packages uv && uv venv --system-site-packages /workspace/venv && "
            ". /workspace/venv/bin/activate && \\\n"
            "  uv pip install -q -e '.[rl]' -c constraints/rl-modal.txt 'huggingface_hub[hf_xet]' && \\\n"
            f"  HF_XET_HIGH_PERFORMANCE=1 python -m cua.rl.train run --config {Path(cfg.rl_config).name} "
            f"--run-dir {wd}/rl_runs/{self.run_id} --workdir {wd}"
        )

    def run_eval(self, phase):
        tag = "before" if "BEFORE" in phase else "after"
        port = (self.rl.get("compute") or {}).get("serve_port", 8001)
        argv = [
            sys.executable,
            "-m",
            "cua.rl.eval",
            "--endpoint",
            f"http://127.0.0.1:{port}/v1",
            "--model",
            "policy",
            "--run-id",
            f"{self.run_id}-{tag}",
            "--out",
            str(self.run_dir / "eval_runs"),
        ]
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
                # rl_output/, not latest/: `latest` appears only after swift exits, so checkpoints died with the box.
                provider.download(instance, f"{wd}/rl_output/", str(self.run_dir / "rl_output") + "/")
                logger.info(f"pulled in-progress checkpoints to {self.run_dir / 'rl_output'}")
            except Exception as e:
                logger.debug(f"no checkpoint to pull yet: {e}")
        return False

    def collect(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        logger.info("Downloading RL outputs…")
        for remote, local in (
            (f"{wd}/rl_output/", self.run_dir / "rl_output"),
            (f"{wd}/rl_runs/{self.run_id}/", self.run_dir / "box"),
        ):
            for attempt in range(1, 4):
                try:
                    provider.download(instance, remote, str(local) + "/")
                    break
                except Exception as e:
                    logger.warning(f"pull of {remote} failed (attempt {attempt}/3): {e}")
                    if attempt == 3:
                        raise
                    time.sleep(10 * attempt)

    def results_dest(self, cfg) -> str:
        return str(self.run_dir)

    def report(self, cfg) -> str:
        return f"run {self.run_id}: evals {self.evals_done}; outputs in {self.run_dir}"


def parse_serve_adapters(spec):
    """ "name=path,name=path" -> {name: path}; a bare path keeps the canonical single-adapter name."""
    if not spec:
        return {}
    entries = [part.strip() for part in str(spec).split(",") if part.strip()]
    if len(entries) == 1 and "=" not in entries[0]:
        return {"cua-student-lora": entries[0]}
    return dict(entry.split("=", 1) for entry in entries)


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
        for name, local in parse_serve_adapters(cfg.serve_adapter).items():
            if not (Path(local) / "adapter_config.json").is_file():
                raise FileNotFoundError(f"adapter '{name}' at '{local}' has no adapter_config.json")

    def upload(self, provider, instance, cfg) -> None:
        wd = cfg.remote_workdir
        upload_package(provider, instance, cfg)
        _upload_code(provider, instance, cfg, (cfg.rl_config,))
        for name, local in parse_serve_adapters(cfg.serve_adapter).items():
            provider.exec(instance, f"mkdir -p {wd}/serve_adapter/{name}", timeout=60)
            provider.upload(instance, local.rstrip("/") + "/", f"{wd}/serve_adapter/{name}/")
        if self.rl.get("serve_merged_base"):
            provider.exec(instance, f"mkdir -p {wd}/sft_adapter", timeout=60)
            provider.upload(
                instance, str(Path(self.rl["sft_adapter"]).expanduser()).rstrip("/") + "/", f"{wd}/sft_adapter/"
            )

    def resources(self, provider, instance, cfg):
        return provider.tunnel(instance, cfg, [("L", int((self.rl.get("compute") or {}).get("serve_port", 8001)))])

    def remote_command(self, cfg) -> str:
        wd = cfg.remote_workdir
        adapters = {name: f"{wd}/serve_adapter/{name}" for name in parse_serve_adapters(cfg.serve_adapter)} or None
        deploy = build_deploy_command(dict(self.rl, merged_model=self.rl["base_model"]), adapters=adapters)
        if self.rl.get("serve_backend") == "vllm":
            serve = shlex.join(build_vllm_serve_command(self.rl, adapters=adapters))
            if not self.rl.get("serve_merged_base"):
                return serve
            # ms-swift is only needed for the merge; keep it out of the image's Python with a throwaway uv env.
            merge = shlex.join(build_merge_command(self.rl))
            return (
                "pip install -q --break-system-packages uv && uv venv /workspace/merge-env --python 3.12 -q && "
                ". /workspace/merge-env/bin/activate && uv pip install -q -e '/workspace[rl]' "
                "-c /workspace/constraints/rl-modal.txt torchvision && \\\n"
                f"  {merge} && deactivate && \\\n"
                f"  {serve}"
            )
        # A uv venv over the image's site-packages keeps its torch/CUDA while sidestepping Debian's unremovable pins.
        return (
            "pip install -q --break-system-packages uv && uv venv --system-site-packages /workspace/venv && "
            ". /workspace/venv/bin/activate && \\\n"
            "  uv pip install -q -e '.[rl]' -c constraints/rl-modal.txt && \\\n"
            f"  MAX_PIXELS={self.rl['max_pixels']} {shlex.join(deploy)}"
        )

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
