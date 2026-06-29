"""Provider-agnostic remote-training lifecycle, with teardown as the first-class concern.

`run_remote_training` provisions a GPU, ships `trajectories/` + the training code, runs the
EXISTING `train_qwen.py convert` + `train_qwen.py train` on the box (no training logic is
duplicated here), pulls the LoRA adapter back to `./output`, and — always — destroys the
instance.

Teardown is defense-in-depth so credits are never wasted:
  1. `try/finally` around the whole run → destroy on success, exception, or early return.
  2. Signal traps (SIGINT/SIGTERM/SIGHUP) → destroy then exit.
  3. A `destroy` that polls until the instance is CONFIRMED gone (in the provider).
  4. A crash-safe state file (`state.py`) + the `cleanup` orphan sweep for hard kills.
  5. An independent max-runtime watchdog thread that force-destroys a hung run.
"""
import asyncio
import logging
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

from remote_train import state
from remote_train.base import Instance, ProviderError
from remote_train.config import RemoteTrainConfig
from remote_train.mock import MockProvider

try:
    from repo_pipeline import notifier as _notifier
except Exception:  # telegram/repo_pipeline optional — notify is off by default
    _notifier = None

logger = logging.getLogger(__name__)

_STATUS_FILE = "STATUS"
_LOG_FILE = "train.log"
_RUN_SCRIPT = "run.sh"


def run_remote_training(provider, cfg: RemoteTrainConfig, *, enable_safety: bool = True) -> int:
    _preflight(cfg)
    _sweep_orphans(provider, cfg)

    instance = None
    stop = threading.Event()
    watchdog = None
    try:
        offer = provider.search_offer(cfg)
        instance = provider.provision(offer, cfg)
        state.record_instance(instance)
        if enable_safety:
            _install_signal_handlers(provider, instance)
            watchdog = _start_watchdog(provider, instance, cfg, stop)

        provider.wait_ready(instance, cfg.ready_timeout_s)
        _sync_up(provider, instance, cfg)
        _launch_training(provider, instance, cfg)

        deadline = time.time() + cfg.max_runtime_hours * 3600
        if not _wait_for_completion(provider, instance, cfg, deadline):
            logger.error("Remote training reported FAILED — pulling logs, skipping adapter download")
            _pull_logs(provider, instance, cfg)
            return 1

        _sync_down(provider, instance, cfg)
        _post_train(provider, instance, cfg)
        logger.info(f"Remote training complete; LoRA adapter in ./{cfg.output_dir}")
        return 0
    finally:
        stop.set()
        if instance is not None:
            _guaranteed_destroy(provider, instance)
        state.clear_instance()


def cleanup(provider, cfg: RemoteTrainConfig) -> int:
    """Destroy the state-file instance AND every instance carrying the run label. Idempotent."""
    targets: dict = {}
    record = state.read_instance()
    if record and record.get("id"):
        targets[str(record["id"])] = Instance(
            id=str(record["id"]), provider=record.get("provider", cfg.provider),
            label=record.get("label", cfg.instance_label),
        )
    try:
        for inst in provider.list_owned(cfg.instance_label):
            targets[inst.id] = inst
    except ProviderError as e:
        logger.error(f"Could not list instances for orphan sweep: {e}")

    if not targets:
        logger.info(f"Nothing to clean up (no state file, no instance labeled '{cfg.instance_label}')")
        state.clear_instance()
        return 0

    all_ok = True
    for inst in targets.values():
        logger.info(f"Cleanup: destroying {inst.id}")
        if not _guaranteed_destroy(provider, inst):
            all_ok = False
    state.clear_instance()
    return 0 if all_ok else 1


# ----- teardown primitives -----

def _guaranteed_destroy(provider, instance) -> bool:
    logger.info(f"Tearing down instance {instance.id} …")
    try:
        ok = provider.destroy(instance)
    except Exception as e:
        logger.error(f"destroy raised {e!r} — CHECK THE PROVIDER CONSOLE for instance {instance.id}")
        ok = False
    if not ok:
        logger.error(f"⚠️  Could not confirm teardown of {instance.id}. Run `make remote-train-cleanup`.")
    return ok


def _install_signal_handlers(provider, instance) -> None:
    def handler(signum, frame):
        logger.warning(f"Signal {signum} received — destroying instance {instance.id} before exit")
        _guaranteed_destroy(provider, instance)
        state.clear_instance()
        sys.exit(130)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass  # not on the main thread; the try/finally still covers teardown


def _start_watchdog(provider, instance, cfg, stop: threading.Event) -> threading.Thread:
    limit_s = cfg.max_runtime_hours * 3600

    def run():
        if stop.wait(limit_s):
            return  # run finished and signalled us
        logger.error(f"⏰ max_runtime_hours ({cfg.max_runtime_hours}h) exceeded — force-destroying {instance.id}")
        _guaranteed_destroy(provider, instance)
        state.clear_instance()
        os._exit(124)

    t = threading.Thread(target=run, daemon=True, name="remote-train-watchdog")
    t.start()
    return t


def _sweep_orphans(provider, cfg) -> None:
    try:
        orphans = provider.list_owned(cfg.instance_label)
    except ProviderError as e:
        logger.warning(f"Orphan pre-sweep skipped (list failed): {e}")
        return
    for inst in orphans:
        logger.warning(f"Found pre-existing instance {inst.id} labeled '{cfg.instance_label}' — destroying before run")
        _guaranteed_destroy(provider, inst)


# ----- lifecycle steps -----

def _preflight(cfg) -> None:
    traj = Path(cfg.trajectories_dir)
    if not traj.is_dir() or not any(traj.iterdir()):
        raise FileNotFoundError(f"trajectories dir '{traj}' missing or empty — nothing to train on")
    if not Path(cfg.train_config).is_file():
        raise FileNotFoundError(f"train config '{cfg.train_config}' not found")


def _sync_up(provider, instance, cfg) -> None:
    wd = cfg.remote_workdir
    provider.exec(instance, f"mkdir -p {wd}/trajectories", timeout=60)
    logger.info("Uploading training bundle (trajectories + code)…")
    provider.upload(instance, cfg.trajectories_dir.rstrip("/") + "/", f"{wd}/trajectories/")
    for f in ("train_qwen.py", cfg.train_config, "requirements.txt"):
        if Path(f).is_file():
            provider.upload(instance, f, f"{wd}/{Path(f).name}")


def _remote_script(cfg) -> str:
    wd = cfg.remote_workdir
    return (
        "set -o pipefail\n"
        f"cd {wd}\n"
        f"echo RUNNING > {wd}/{_STATUS_FILE}\n"
        "{\n"
        "  pip install -r requirements.txt && \\\n"
        '  pip install "ms-swift[llm]" && \\\n'
        f"  python train_qwen.py convert --input trajectories/ --output training_data/ "
        f"--val-split {cfg.val_split} --exclude-waste-steps --exclude-meandering && \\\n"
        f"  python train_qwen.py train --config {Path(cfg.train_config).name} "
        f"--dataset training_data/sft_dataset.jsonl\n"
        f"}} > {wd}/{_LOG_FILE} 2>&1\n"
        f"if [ $? -eq 0 ]; then echo DONE > {wd}/{_STATUS_FILE}; else echo FAILED > {wd}/{_STATUS_FILE}; fi\n"
    )


def _launch_training(provider, instance, cfg) -> None:
    wd = cfg.remote_workdir
    script = _remote_script(cfg)
    write_cmd = f"cat > {wd}/{_RUN_SCRIPT} <<'REMOTE_EOF'\n{script}REMOTE_EOF"
    _, err, code = provider.exec(instance, write_cmd, timeout=60)
    if code != 0:
        raise ProviderError(f"Failed to write remote run script: {err[:200]}")
    _, err, code = provider.exec(instance, f"cd {wd} && nohup bash {_RUN_SCRIPT} >/dev/null 2>&1 & echo launched", timeout=60)
    if code != 0:
        raise ProviderError(f"Failed to launch remote training: {err[:200]}")
    logger.info("Remote training launched (detached). Polling for completion…")


def _wait_for_completion(provider, instance, cfg, deadline: float) -> bool:
    wd = cfg.remote_workdir
    last_line = ""
    while time.time() < deadline:
        out, _, _ = provider.exec(instance, f"cat {wd}/{_STATUS_FILE} 2>/dev/null || echo MISSING", timeout=45)
        status = (out.strip().splitlines() or ["MISSING"])[-1].strip()
        tail, _, _ = provider.exec(instance, f"tail -n 3 {wd}/{_LOG_FILE} 2>/dev/null", timeout=45)
        tail_line = (tail.strip().splitlines() or [""])[-1]
        if tail_line and tail_line != last_line:
            logger.info(f"[remote] {tail_line}")
            last_line = tail_line
        if status == "DONE":
            return True
        if status == "FAILED":
            return False
        time.sleep(cfg.poll_interval_s)
    raise ProviderError(f"Training exceeded max_runtime_hours ({cfg.max_runtime_hours}h)")


def _sync_down(provider, instance, cfg) -> None:
    wd = cfg.remote_workdir
    logger.info("Downloading trained adapter…")
    provider.download(instance, f"{wd}/output/", cfg.output_dir.rstrip("/") + "/")
    adapter = Path(cfg.output_dir) / "adapter_config.json"
    if not adapter.exists():
        logger.warning(f"Expected {adapter} not found after download — inspect ./{cfg.output_dir}")


def _pull_logs(provider, instance, cfg) -> None:
    wd = cfg.remote_workdir
    try:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        provider.download(instance, f"{wd}/{_LOG_FILE}", str(Path(cfg.output_dir) / _LOG_FILE))
        logger.info(f"Remote training log saved to ./{cfg.output_dir}/{_LOG_FILE}")
    except ProviderError as e:
        logger.warning(f"Could not pull remote log: {e}")


def _post_train(provider, instance, cfg) -> None:
    if cfg.run_eval:
        _maybe_eval(provider, instance, cfg)
    if cfg.push_hf_repo:
        _maybe_push_hf(provider, instance, cfg)
    if cfg.notify:
        _notify(f"✅ remote training done on {instance.id}; adapter pulled to ./{cfg.output_dir}")


def _maybe_eval(provider, instance, cfg) -> None:
    wd = cfg.remote_workdir
    eval_files = [f for f in ("eval_screenspot.py", "eval_models.yaml", "requirements-eval.txt") if Path(f).is_file()]
    if "eval_screenspot.py" not in eval_files:
        logger.warning("--eval requested but eval_screenspot.py not present locally — skipping (merge PR #12 first)")
        return
    try:
        for f in eval_files:
            provider.upload(instance, f, f"{wd}/{f}")
        provider.exec(instance, f"cd {wd} && pip install -r requirements-eval.txt && python eval_screenspot.py run",
                      stream=True, timeout=0)
        provider.download(instance, f"{wd}/eval_results/", "eval_results/")
        logger.info("Eval complete; results in ./eval_results")
    except ProviderError as e:
        logger.error(f"Post-train eval failed (non-fatal): {e}")


def _maybe_push_hf(provider, instance, cfg) -> None:
    if not cfg.hf_token:
        logger.warning("--push-hf requested but HF_TOKEN unset — skipping")
        return
    wd = cfg.remote_workdir
    push = (
        f"cd {wd} && pip install -U huggingface_hub && "
        f"HF_TOKEN={cfg.hf_token} huggingface-cli upload {cfg.push_hf_repo} output . --repo-type model"
    )
    try:
        _, err, code = provider.exec(instance, push, timeout=1800)
        if code == 0:
            logger.info(f"Pushed adapter to HuggingFace repo {cfg.push_hf_repo}")
        else:
            logger.error(f"HF push failed (non-fatal): {err[:200]}")
    except ProviderError as e:
        logger.error(f"HF push failed (non-fatal): {e}")


def _notify(message: str) -> None:
    if _notifier is None:
        logger.info(f"(notify skipped — repo_pipeline.notifier unavailable) {message}")
        return
    try:
        token, chat_id = _notifier.get_bot_config()
    except ValueError as e:
        logger.warning(f"Telegram not configured: {e}")
        return
    try:
        asyncio.run(_notifier.send_telegram_message(token, chat_id, message))
    except Exception as e:
        logger.error(f"notify failed: {e}")


# ----- no-cost self-test (the teardown guarantee is the central assertion) -----

def run_smoke_test() -> int:
    failures = []
    tmp = Path(tempfile.mkdtemp(prefix="remote_train_smoke_"))
    (tmp / "trajectories").mkdir()
    (tmp / "trajectories" / "task_x").mkdir()
    (tmp / "trajectories" / "task_x" / "trajectory.json").write_text("{}")
    (tmp / "config.yaml").write_text("model: test\noutput_dir: ./output\n")
    (tmp / "train_qwen.py").write_text("# stub\n")
    (tmp / "requirements.txt").write_text("\n")
    state.STATE_FILE = tmp / ".remote_train_state.json"

    def make_cfg():
        cfg = RemoteTrainConfig()
        cfg.trajectories_dir = str(tmp / "trajectories")
        cfg.train_config = str(tmp / "config.yaml")
        cfg.output_dir = str(tmp / "output")
        cfg.poll_interval_s = 0
        return cfg

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    cfg = make_cfg()

    provider = MockProvider(cfg)
    offer = provider.search_offer(cfg)
    check("geo-exclusion drops RU/CN; cheapest allowed is SE", offer.id == "se-1" and offer.country == "SE")
    check("excluded offers were actually present", any(o.country in ("RU", "CN") for o in provider._fake_offers))

    inst = provider.provision(offer, cfg)
    check("instance is labeled roman.maior", inst.label == "roman.maior")

    happy = MockProvider(make_cfg())
    rc = run_remote_training(happy, make_cfg(), enable_safety=False)
    check("happy path returns 0", rc == 0)
    check("happy path destroyed the instance", len(happy.destroyed) == 1)

    class _RaiseProvider(MockProvider):
        def wait_ready(self, instance, timeout):
            raise ProviderError("simulated mid-run failure")

    boom = _RaiseProvider(make_cfg())
    try:
        run_remote_training(boom, make_cfg(), enable_safety=False)
    except ProviderError:
        pass
    check("exception path still destroyed the instance", len(boom.destroyed) == 1)

    class _InterruptProvider(MockProvider):
        def wait_ready(self, instance, timeout):
            raise KeyboardInterrupt()

    interrupted = _InterruptProvider(make_cfg())
    try:
        run_remote_training(interrupted, make_cfg(), enable_safety=False)
    except KeyboardInterrupt:
        pass
    check("SIGINT/KeyboardInterrupt path still destroyed the instance", len(interrupted.destroyed) == 1)

    print(f"\nremote_train self-test: {'ALL PASS' if not failures else f'{len(failures)} FAILURES'}")
    return 0 if not failures else 1
