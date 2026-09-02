"""Provider-agnostic remote JOB lifecycle, with teardown as the first-class concern.

The skeleton — rent a GPU → wait for SSH → upload → run (detached, sentinel-tracked) → pull
results → ALWAYS destroy — is job-agnostic. What actually runs comes from a `RemoteJob`
(`jobs.py`): `train` fine-tunes, `eval` benchmarks. Adding a job type or a new benchmark never
touches this file.

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
from remote_train.jobs import RL_CODE_ROOTS, GrpoJob, build_job, first_party_closure
from remote_train.mock import MockProvider
from remote_train.vastai import VastAIProvider

try:
    from repo_pipeline import notifier as _notifier
except Exception:  # telegram/repo_pipeline optional — notify is off by default
    _notifier = None

logger = logging.getLogger(__name__)

_STATUS_FILE = "STATUS"
# Consecutive failed status polls before the box is considered lost; one slow ssh must never end a run.
_MAX_POLL_FAILURES = 5
_LOG_FILE = "run.log"
_RUN_SCRIPT = "run.sh"


def run_remote_job(provider, cfg: RemoteTrainConfig, job_name: str, *, enable_safety: bool = True) -> int:
    return _run_job(provider, cfg, build_job(job_name, cfg), enable_safety=enable_safety)


def _run_job(provider, cfg, job, *, enable_safety: bool = True) -> int:
    job.preflight(cfg)
    _sweep_orphans(provider, cfg)

    instance = None
    stop = threading.Event()
    try:
        offer = provider.search_offer(cfg)
        instance = provider.provision(offer, cfg)
        state.record_instance(instance)
        logger.info(
            f"RENTED {instance.id} — {offer.gpu_name} {offer.gpu_ram_gb}GB @ ${offer.price_per_hour:.3f}/hr "
            f"({offer.country}). It WILL be destroyed when this run ends."
        )
        if enable_safety:
            _install_signal_handlers(provider, instance, cfg)
            _start_watchdog(provider, instance, cfg, stop, job)

        try:
            provider.wait_ready(instance, cfg.ready_timeout_s)
        except Exception as e:
            # Selection is cheapest-first and deterministic, so without this the retry rents the
            # SAME dead host every time and the whole run gives up on one bad machine.
            state.record_bad_offer(offer.id, e)
            raise
        # Job resources (SSH tunnels) live inside the try/finally and close BEFORE the instance is destroyed.
        with job.resources(provider, instance, cfg):
            job.upload(provider, instance, cfg)
            _launch(provider, instance, cfg, job.remote_command(cfg))

            deadline = time.time() + cfg.max_runtime_hours * 3600
            try:
                ok = _wait_for_completion(provider, instance, cfg, deadline, job)
            except Exception:
                # A lost box still has hours of paid work on it; try to rescue it before the finally destroys it.
                _rescue(provider, instance, cfg, job)
                raise
            _pull_logs(provider, instance, cfg, job.results_dest(cfg))
            if not ok:
                logger.error(f"Remote {job.name} reported FAILED — see ./{job.results_dest(cfg)}/{_LOG_FILE}")
                _rescue(provider, instance, cfg, job)
                return 1

            job.collect(provider, instance, cfg)
        summary = job.report(cfg)
        if summary:
            print(f"\n=== {job.name} results ===\n{summary}\n")
        return 0
    finally:
        stop.set()
        if instance is not None:
            _guaranteed_destroy(provider, instance, cfg)
            state.clear_instance(instance.id)


def _rescue(provider, instance, cfg, job) -> None:
    """Best-effort pull of whatever the box produced. Never raises — teardown must still run."""
    try:
        job.collect(provider, instance, cfg)
        logger.info(f"rescued partial results to ./{job.results_dest(cfg)}")
    except Exception as e:
        logger.warning(f"could not rescue results from {instance.id}: {e}")


def cleanup(provider, cfg: RemoteTrainConfig, *, sweep_all: bool = False) -> int:
    """Destroy abandoned instances. Idempotent.

    Default targets only records whose owning run is dead. `sweep_all` additionally takes every
    instance carrying our label — including one a sibling run is training on right now — so it
    stays an explicit choice (`--all`) rather than the behaviour you get by reflex.
    """
    targets: dict = {}
    skipped_live: list = []
    for record in state.read_instances():
        inst = Instance(
            id=str(record["id"]), provider=record.get("provider", cfg.provider),
            label=record.get("label", cfg.instance_label),
        )
        if state.owner_alive(record) and not sweep_all:
            skipped_live.append(inst.id)
            continue
        targets[inst.id] = inst

    if sweep_all:
        try:
            for inst in provider.list_owned(cfg.instance_label):
                targets[inst.id] = inst
        except ProviderError as e:
            logger.error(f"Could not list instances for the --all sweep: {e}")

    for inst_id in skipped_live:
        logger.warning(f"Instance {inst_id} is owned by a RUNNING process — skipping (use --all to force)")

    if not targets:
        logger.info(f"Nothing to clean up (no abandoned record, no instance labeled '{cfg.instance_label}')")
        return 0

    all_ok = True
    for inst in targets.values():
        logger.info(f"Cleanup: destroying {inst.id}")
        if _guaranteed_destroy(provider, inst, cfg):
            state.clear_instance(inst.id)
        else:
            all_ok = False
    return 0 if all_ok else 1


# ----- teardown primitives -----

def _owned_by_us(provider, instance, cfg) -> bool:
    """The ownership gate every teardown path passes through: the account is SHARED.

    Refuses only on positive evidence of foreign ownership — a label the provider reports as
    non-empty and different from ours. An unknown or unreported label leaves teardown free to
    run, so a provider API hiccup can never turn into a leaked billing instance.
    """
    try:
        live = provider.live_label(instance)
    except Exception as e:
        logger.warning(f"Could not read the live label of {instance.id} ({e!r}) — proceeding with teardown")
        return True
    if live and live != cfg.instance_label:
        logger.error(
            f"REFUSING to destroy instance {instance.id}: the provider reports label {live!r}, not "
            f"{cfg.instance_label!r}. This account is shared — that instance is someone else's work."
        )
        return False
    return True


def _guaranteed_destroy(provider, instance, cfg) -> bool:
    if not _owned_by_us(provider, instance, cfg):
        return False
    logger.info(f"Tearing down instance {instance.id} …")
    try:
        ok = provider.destroy(instance)
    except Exception as e:
        logger.error(f"destroy raised {e!r} — CHECK THE PROVIDER CONSOLE for instance {instance.id}")
        ok = False
    if not ok:
        logger.error(f"⚠️  Could not confirm teardown of {instance.id}. Run `make remote-train-cleanup`.")
    return ok


def _install_signal_handlers(provider, instance, cfg) -> None:
    def handler(signum, frame):
        logger.warning(f"Signal {signum} received — destroying instance {instance.id} before exit")
        _guaranteed_destroy(provider, instance, cfg)
        state.clear_instance(instance.id)
        sys.exit(130)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass  # not on the main thread; the try/finally still covers teardown


def _start_watchdog(provider, instance, cfg, stop: threading.Event, job=None) -> threading.Thread:
    limit_s = cfg.max_runtime_hours * 3600

    def run():
        if stop.wait(limit_s):
            return  # run finished and signalled us
        logger.error(f"⏰ max_runtime_hours ({cfg.max_runtime_hours}h) exceeded — force-destroying {instance.id}")
        if job is not None:
            _rescue(provider, instance, cfg, job)
        _guaranteed_destroy(provider, instance, cfg)
        state.clear_instance(instance.id)
        os._exit(124)

    t = threading.Thread(target=run, daemon=True, name="remote-train-watchdog")
    t.start()
    return t


def _sweep_orphans(provider, cfg) -> None:
    """Pre-run safety sweep — designed for a SHARED/team account AND for concurrent local runs.

    Auto-destroys ONLY an instance this tool recorded whose owning run is gone: provably ours and
    provably abandoned. A record whose owner process is still alive belongs to a sibling run and is
    left strictly alone — several runs share one state file whenever they share a directory.
    Anything else carrying our label is reported, never killed. Teammates' boxes are invisible
    either way: `list_owned` is an exact label match. Sweeping the rest is an explicit,
    user-initiated act: `make remote-train-cleanup`.
    """
    orphans = {str(r["id"]) for r in state.read_instances() if not state.owner_alive(r)}
    live_siblings = {str(r["id"]) for r in state.read_instances() if state.owner_alive(r)}

    try:
        labeled = provider.list_owned(cfg.instance_label)
    except ProviderError as e:
        logger.warning(f"Orphan pre-sweep skipped (list failed): {e}")
        return

    for inst in labeled:
        if inst.id in orphans:
            logger.warning(f"Orphan {inst.id} left by a dead run (state file) — destroying before start")
            if _guaranteed_destroy(provider, inst, cfg):
                state.clear_instance(inst.id)
        elif inst.id in live_siblings:
            logger.info(f"Instance {inst.id} belongs to a RUNNING sibling run — leaving it alone")
        else:
            logger.warning(
                f"Instance {inst.id} also carries label '{cfg.instance_label}' but was not recorded by this "
                f"tool — LEAVING IT ALONE (may be a concurrent run). If it is an orphan, run "
                f"`make remote-train-cleanup`."
            )

    live_ids = {i.id for i in labeled}
    for gone in orphans - live_ids:
        state.clear_instance(gone)  # recorded instance is already gone


# ----- run steps -----

def _remote_script(cfg, body: str) -> str:
    wd = cfg.remote_workdir
    return (
        "set -o pipefail\n"
        f"cd {wd}\n"
        f"echo RUNNING > {wd}/{_STATUS_FILE}\n"
        "{\n"
        f"{body}\n"
        f"}} > {wd}/{_LOG_FILE} 2>&1\n"
        f"if [ $? -eq 0 ]; then echo DONE > {wd}/{_STATUS_FILE}; else echo FAILED > {wd}/{_STATUS_FILE}; fi\n"
    )


def _launch(provider, instance, cfg, body: str) -> None:
    wd = cfg.remote_workdir
    script = _remote_script(cfg, body)
    write_cmd = f"cat > {wd}/{_RUN_SCRIPT} <<'REMOTE_EOF'\n{script}REMOTE_EOF"
    _, err, code = provider.exec(instance, write_cmd, timeout=60)
    if code != 0:
        raise ProviderError(f"Failed to write remote run script: {err[:200]}")
    # sshd can hold the exec channel open for a detached child even with setsid + all three fds
    # redirected, so the ssh CLIENT may never return here. That is not evidence of failure, and
    # its exit code is not evidence of success. Fire it, tolerate a non-returning client, then
    # prove the job actually started by looking for the STATUS sentinel it writes on disk.
    launch = f"cd {wd} && setsid nohup bash {_RUN_SCRIPT} </dev/null >/dev/null 2>&1 & echo launched"
    try:
        provider.exec(instance, launch, timeout=20)
    except ProviderError as e:
        logger.info(f"launch client did not return ({e}) — verifying via the STATUS sentinel instead")

    for _ in range(12):
        out, _, _ = provider.exec(instance, f"cat {wd}/{_STATUS_FILE} 2>/dev/null || true", timeout=30)
        if out.strip():
            logger.info(f"Remote job is running (STATUS={out.strip()}). Polling for completion…")
            return
        time.sleep(5)
    raise ProviderError("remote job never started — no STATUS sentinel appeared within 60s")


def _wait_for_completion(provider, instance, cfg, deadline: float, job=None) -> bool:
    wd = cfg.remote_workdir
    last_line = ""
    poll_failures = 0
    while time.time() < deadline:
        if job is not None:
            try:
                if job.on_poll(provider, instance, cfg):
                    return True
            except Exception as e:
                logger.warning(f"on_poll hook failed (ignored): {e}")
        # A slow ssh is not a dead box: one timed-out status poll once destroyed a healthy instance
        # that was serving 200s seconds earlier. Only a run of consecutive failures means anything.
        try:
            out, _, _ = provider.exec(instance, f"cat {wd}/{_STATUS_FILE} 2>/dev/null || echo MISSING", timeout=45)
            poll_failures = 0
        except ProviderError as e:
            poll_failures += 1
            if poll_failures >= _MAX_POLL_FAILURES:
                raise ProviderError(f"status poll failed {poll_failures}x in a row; treating the box as lost: {e}") from e
            logger.warning(f"status poll failed ({str(e)[:120]}); retry {poll_failures}/{_MAX_POLL_FAILURES - 1}")
            time.sleep(cfg.poll_interval_s)
            continue
        status = (out.strip().splitlines() or ["MISSING"])[-1].strip()
        try:
            tail, _, _ = provider.exec(instance, f"tail -n 3 {wd}/{_LOG_FILE} 2>/dev/null", timeout=45)
        except ProviderError:
            tail = ""
        tail_line = (tail.strip().splitlines() or [""])[-1]
        if tail_line and tail_line != last_line:
            logger.info(f"[remote] {tail_line}")
            last_line = tail_line
        if status == "DONE":
            return True
        if status == "FAILED":
            return False
        time.sleep(cfg.poll_interval_s)
    raise ProviderError(f"Job exceeded max_runtime_hours ({cfg.max_runtime_hours}h)")


def _pull_logs(provider, instance, cfg, dest_dir: str) -> None:
    wd = cfg.remote_workdir
    try:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        provider.download(instance, f"{wd}/{_LOG_FILE}", str(Path(dest_dir) / _LOG_FILE))
        logger.info(f"Remote log saved to ./{dest_dir}/{_LOG_FILE}")
    except ProviderError as e:
        logger.warning(f"Could not pull remote log: {e}")


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


# ----- no-cost self-test (teardown guarantee is the central assertion) -----

def run_smoke_test() -> int:
    failures = []
    tmp = Path(tempfile.mkdtemp(prefix="remote_train_smoke_"))
    (tmp / "trajectories" / "task_x").mkdir(parents=True)
    (tmp / "trajectories" / "task_x" / "trajectory.json").write_text("{}")
    (tmp / "config.yaml").write_text("model: test\noutput_dir: ./output\n")
    (tmp / "output").mkdir()
    (tmp / "output" / "adapter_config.json").write_text("{}")
    (tmp / "output" / "adapter_model.safetensors").write_bytes(b"0" * 16)
    state.STATE_FILE = tmp / ".remote_train_state.json"
    state.BAD_OFFER_FILE = tmp / ".remote_train_bad_offers.json"  # the mock's simulated failures must not blocklist real offers

    def make_cfg(job="train"):
        cfg = RemoteTrainConfig()
        cfg.trajectories_dir = str(tmp / "trajectories")
        cfg.train_config = str(tmp / "config.yaml")
        cfg.output_dir = str(tmp / "output")
        cfg.eval_results_dir = str(tmp / "eval_results")
        cfg.eval_models = ("Qwen/Qwen3.5-9B-Base",)
        cfg.poll_interval_s = 0
        cfg.rl_runs_dir = str(tmp / "rl_runs")
        cfg.env_url = "mock://"
        cfg.serve_adapter = str(tmp / "output")
        cfg.serve_seconds = 1
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
    check("instance is labeled roman.maior", provider.provision(offer, cfg).label == "roman.maior")

    foreign = MockProvider(make_cfg())
    foreign.reported_label = "someone.else"
    kept = _guaranteed_destroy(foreign, Instance(id="theirs-1", provider="mock", label="someone.else"), make_cfg())
    check("a foreign label is never destroyed", kept is False and foreign.destroyed == [])

    unknown = MockProvider(make_cfg())
    unknown.reported_label = None
    _guaranteed_destroy(unknown, Instance(id="ours-1", provider="mock", label=cfg.instance_label), make_cfg())
    check("an unreadable label still tears our own box down", unknown.destroyed == ["ours-1"])

    blank_cfg = make_cfg()
    blank_cfg.instance_label = ""
    try:
        VastAIProvider(blank_cfg).list_owned(blank_cfg.instance_label)
        blank_refused = False
    except ProviderError:
        blank_refused = True
    check("a blank label cannot claim unlabeled instances", blank_refused)

    shipped = set(first_party_closure(RL_CODE_ROOTS))
    unshipped = {imp for f in shipped for imp in first_party_closure((f,))} - shipped
    check("the GRPO upload manifest is import-closed", not unshipped)
    if unshipped:
        print(f"        missing from the bundle: {sorted(unshipped)}")

    sibling = MockProvider(make_cfg())
    state.STATE_FILE = Path(tempfile.mkdtemp()) / "state.json"
    state.record_instance(Instance(id="sibling-live", provider="mock", label=cfg.instance_label))
    cleanup(sibling, make_cfg())
    check("cleanup spares an instance whose owner run is alive", sibling.destroyed == [])
    _sweep_orphans(sibling, make_cfg())
    check("a pre-run sweep never touches a live sibling's box", sibling.destroyed == [])

    dead = MockProvider(make_cfg())
    state.STATE_FILE = Path(tempfile.mkdtemp()) / "state.json"
    state.record_instance(Instance(id="orphan-1", provider="mock", label=cfg.instance_label))
    records = state.read_instances()
    records[0]["pid"] = 2 ** 22  # a pid that cannot be running
    state._write(records)
    cleanup(dead, make_cfg())
    check("cleanup still reclaims a box whose owner run died", dead.destroyed == ["orphan-1"])
    check("a reclaimed instance leaves the state file", state.read_instances() == [])

    real_run_eval = GrpoJob.run_eval
    GrpoJob.run_eval = lambda self, phase: {"phase": phase}
    for job_name in ("train", "eval", "grpo", "serve"):
        state.BAD_OFFER_FILE.unlink(missing_ok=True)  # the simulated failures below would blocklist the mock fleet
        happy = MockProvider(make_cfg())
        rc = run_remote_job(happy, make_cfg(), job_name, enable_safety=False)
        check(f"{job_name}: happy path returns 0", rc == 0)
        check(f"{job_name}: happy path destroyed the instance", len(happy.destroyed) == 1)
        if job_name == "grpo":
            check("grpo: tunnel opened R:8900 + L:8001", happy.tunnels == [[("R", 8900), ("L", 8001)]])
            check("grpo: both eval phases acked", sum("PHASE_ACK" in c for c in happy.exec_log) == 2)
        if job_name == "serve":
            check("serve: tunnel opened L:8001", happy.tunnels == [[("L", 8001)]])

        class _RaiseProvider(MockProvider):
            def wait_ready(self, instance, timeout):
                raise ProviderError("simulated mid-run failure")

        state.BAD_OFFER_FILE.unlink(missing_ok=True)
        boom = _RaiseProvider(make_cfg())
        try:
            run_remote_job(boom, make_cfg(), job_name, enable_safety=False)
        except ProviderError:
            pass
        check(f"{job_name}: exception path still destroyed the instance", len(boom.destroyed) == 1)

        class _InterruptProvider(MockProvider):
            def wait_ready(self, instance, timeout):
                raise KeyboardInterrupt()

        state.BAD_OFFER_FILE.unlink(missing_ok=True)
        interrupted = _InterruptProvider(make_cfg())
        try:
            run_remote_job(interrupted, make_cfg(), job_name, enable_safety=False)
        except KeyboardInterrupt:
            pass
        check(f"{job_name}: SIGINT path still destroyed the instance", len(interrupted.destroyed) == 1)
    GrpoJob.run_eval = real_run_eval

    print(f"\nremote_train self-test: {'ALL PASS' if not failures else f'{len(failures)} FAILURES'}")
    return 0 if not failures else 1
