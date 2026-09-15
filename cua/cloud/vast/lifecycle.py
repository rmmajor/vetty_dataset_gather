"""Provider-agnostic remote job lifecycle — rent, upload, run detached, pull, ALWAYS destroy — whose five layers of teardown are listed in docs/architecture.md."""

import asyncio
import logging
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

from cua.cloud.ignore import ALWAYS_EXCLUDE, STATIC_UPLOAD_EXCLUDES
from cua.cloud.vast import state
from cua.cloud.vast.base import Instance, Offer, ProviderError
from cua.cloud.vast.config import RemoteTrainConfig
from cua.cloud.vast.jobs import GrpoJob, build_job, upload_package
from cua.cloud.vast.mock import MockProvider
from cua.cloud.vast.vastai import VastAIProvider

try:
    from cua.pipeline import notifier as _notifier
except Exception:  # telegram/cua.pipeline optional — notify is off by default
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
    outcome = "FAILED"
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
            # Selection is deterministic, so without this every retry rents the SAME dead host.
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
                # Selection is deterministic, so without this a retry re-rents the same host that just failed the workload.
                state.record_bad_offer(offer.id, f"{job.name} reported FAILED on this host")
                _rescue(provider, instance, cfg, job)
                return 1

            job.collect(provider, instance, cfg)
        summary = job.report(cfg)
        if summary:
            print(f"\n=== {job.name} results ===\n{summary}\n")
        outcome = "DONE"
        return 0
    finally:
        stop.set()
        if instance is not None:
            _guaranteed_destroy(provider, instance, cfg)
            state.clear_instance(instance.id)
        # After teardown, so a wedged Telegram call can never delay destroying the box.
        if cfg.notify:
            _notify(f"cua-remote {job.name}: {outcome} (instance {instance.id if instance else 'none'})")


def _rescue(provider, instance, cfg, job) -> None:
    """Best-effort pull of whatever the box produced. Never raises — teardown must still run."""
    try:
        job.collect(provider, instance, cfg)
        logger.info(f"rescued partial results to ./{job.results_dest(cfg)}")
    except Exception as e:
        logger.warning(f"could not rescue results from {instance.id}: {e}")


def cleanup(provider, cfg: RemoteTrainConfig, *, sweep_all: bool = False) -> int:
    """Destroy abandoned instances, idempotently: records whose owning run is dead, or with `sweep_all` every box carrying our label."""
    targets: dict = {}
    skipped_live: list = []
    for record in state.read_instances():
        inst = Instance(
            id=str(record["id"]),
            provider=record.get("provider", cfg.provider),
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
    """The ownership gate every teardown path passes through: refuse only on POSITIVE evidence of a foreign label, since the account is shared."""
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
        # The detached remote job outlives this supervisor, so keep_on_signal trades a possible orphan for hours of paid work.
        if cfg.keep_on_signal:
            logger.warning(
                f"Signal {signum} received — LEAVING instance {instance.id} running so the detached "
                f"job finishes. Collect/destroy later with: python -m cua.cloud.vast cleanup"
            )
            # os._exit, not sys.exit: SystemExit would unwind through the finally that destroys the instance.
            state.mark_abandoned(instance.id)
            os._exit(130)
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
    """Pre-run sweep that auto-destroys ONLY a recorded instance whose owning run is gone; a live sibling's box and any unrecorded box are reported, never killed."""
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
    # The ssh client may never return for a detached child, so the STATUS sentinel below is the only proof the job started.
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
        # A slow ssh is not a dead box: only a run of consecutive failed polls means anything.
        try:
            out, _, _ = provider.exec(instance, f"cat {wd}/{_STATUS_FILE} 2>/dev/null || echo MISSING", timeout=45)
            poll_failures = 0
        except ProviderError as e:
            poll_failures += 1
            if poll_failures >= _MAX_POLL_FAILURES:
                raise ProviderError(
                    f"status poll failed {poll_failures}x in a row; treating the box as lost: {e}"
                ) from e
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
        logger.info(f"(notify skipped — cua.pipeline.notifier unavailable) {message}")
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


class _SmokeEnv:
    """Scratch corpus, adapter and state files plus the pass/fail log every smoke case shares."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="remote_train_smoke_"))
        self.failures: list = []
        (self.tmp / "trajectories" / "task_x").mkdir(parents=True)
        (self.tmp / "trajectories" / "task_x" / "trajectory.json").write_text("{}")
        (self.tmp / "config.yaml").write_text("model: test\noutput_dir: ./output\n")
        (self.tmp / "output").mkdir()
        (self.tmp / "output" / "adapter_config.json").write_text("{}")
        (self.tmp / "output" / "adapter_model.safetensors").write_bytes(b"0" * 16)
        # The mock's simulated failures must never blocklist real offers.
        state.BAD_OFFER_FILE = self.tmp / ".remote_train_bad_offers.json"
        self.use_main_state()

    def make_cfg(self):
        cfg = RemoteTrainConfig()
        cfg.trajectories_dir = str(self.tmp / "trajectories")
        cfg.train_config = str(self.tmp / "config.yaml")
        cfg.output_dir = str(self.tmp / "output")
        cfg.eval_results_dir = str(self.tmp / "eval_results")
        cfg.eval_models = ("Qwen/Qwen3.5-9B-Base",)
        cfg.poll_interval_s = 0
        cfg.rl_runs_dir = str(self.tmp / "rl_runs")
        cfg.env_url = "mock://"
        cfg.serve_adapter = str(self.tmp / "output")
        cfg.serve_seconds = 1
        return cfg

    def check(self, name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            self.failures.append(name)

    def use_main_state(self):
        state.STATE_FILE = self.tmp / ".remote_train_state.json"

    def use_fresh_state(self):
        state.STATE_FILE = Path(tempfile.mkdtemp()) / "state.json"

    def clear_bad_offers(self):
        state.BAD_OFFER_FILE.unlink(missing_ok=True)


def _smoke_offer_selection(env):
    cfg = env.make_cfg()
    provider = MockProvider(cfg)
    offer = provider.search_offer(cfg)
    env.check("geo-exclusion drops RU/CN; cheapest allowed is SE", offer.id == "se-1" and offer.country == "SE")
    env.check("excluded offers were actually present", any(o.country in ("RU", "CN") for o in provider._fake_offers))
    env.check("instance is labeled cua-remote-train", provider.provision(offer, cfg).label == "cua-remote-train")


def _smoke_ownership_gate(env):
    foreign = MockProvider(env.make_cfg())
    foreign.reported_label = "someone.else"
    kept = _guaranteed_destroy(foreign, Instance(id="theirs-1", provider="mock", label="someone.else"), env.make_cfg())
    env.check("a foreign label is never destroyed", kept is False and foreign.destroyed == [])

    unknown = MockProvider(env.make_cfg())
    unknown.reported_label = None
    label = env.make_cfg().instance_label
    _guaranteed_destroy(unknown, Instance(id="ours-1", provider="mock", label=label), env.make_cfg())
    env.check("an unreadable label still tears our own box down", unknown.destroyed == ["ours-1"])

    blank_cfg = env.make_cfg()
    blank_cfg.instance_label = ""
    try:
        VastAIProvider(blank_cfg).list_owned(blank_cfg.instance_label)
        blank_refused = False
    except ProviderError:
        blank_refused = True
    env.check("a blank label cannot claim unlabeled instances", blank_refused)


def _smoke_upload_excludes(env):
    cfg = env.make_cfg()
    provider = MockProvider(cfg)
    instance = provider.provision(provider.search_offer(cfg), cfg)
    upload_package(provider, instance, cfg)
    package = [u for u in provider.uploads if u["local"] == "cua/"]
    excluded = package[0]["exclude"] if package else ()
    env.check("the package upload carries rsync excludes", bool(excluded))
    env.check("bytecode never reaches a shared box", set(ALWAYS_EXCLUDE) <= set(excluded))
    env.check(
        "the burner and guarded site registries never reach a shared box", set(STATIC_UPLOAD_EXCLUDES) <= set(excluded)
    )
    env.check(
        "every derived exclude is anchored at the transfer root",
        all(p.startswith("/") for p in excluded if p not in ALWAYS_EXCLUDE),
    )


def _smoke_cleanup_spares_a_live_sibling(env):
    sibling = MockProvider(env.make_cfg())
    env.use_fresh_state()
    state.record_instance(Instance(id="sibling-live", provider="mock", label=env.make_cfg().instance_label))
    cleanup(sibling, env.make_cfg())
    env.check("cleanup spares an instance whose owner run is alive", sibling.destroyed == [])
    _sweep_orphans(sibling, env.make_cfg())
    env.check("a pre-run sweep never touches a live sibling's box", sibling.destroyed == [])


def _smoke_cleanup_reclaims_a_dead_owner(env):
    dead = MockProvider(env.make_cfg())
    env.use_fresh_state()
    state.record_instance(Instance(id="orphan-1", provider="mock", label=env.make_cfg().instance_label))
    records = state.read_instances()
    records[0]["pid"] = 2**22  # a pid that cannot be running
    state._write(records)
    cleanup(dead, env.make_cfg())
    env.check("cleanup still reclaims a box whose owner run died", dead.destroyed == ["orphan-1"])
    env.check("a reclaimed instance leaves the state file", state.read_instances() == [])


def _smoke_owner_alive_sentinels(env):
    env.check("owner_alive: pid null is the abandoned sentinel", state.owner_alive({"id": "x", "pid": None}) is False)
    env.check("owner_alive: this process is alive", state.owner_alive({"id": "x", "pid": os.getpid()}) is True)
    env.check("owner_alive: a record without a pid is unknown, so alive", state.owner_alive({"id": "x"}) is True)

    released = MockProvider(env.make_cfg())
    env.use_fresh_state()
    state.record_instance(Instance(id="released-1", provider="mock", label=env.make_cfg().instance_label))
    records = state.read_instances()
    records[0]["pid"] = None  # what a supervisor writes when it hands the box over to cleanup
    state._write(records)
    cleanup(released, env.make_cfg())
    env.check("cleanup reclaims a box whose record was released with pid null", released.destroyed == ["released-1"])


def _smoke_release_hands_one_box_over(env):
    handed_over = MockProvider(env.make_cfg())
    env.use_fresh_state()
    label = env.make_cfg().instance_label
    state.record_instance(Instance(id="kept-1", provider="mock", label=label))
    state.record_instance(Instance(id="released-2", provider="mock", label=label))
    state.mark_abandoned("released-2")
    by_id = {r["id"]: r for r in state.read_instances()}
    env.check(
        "mark_abandoned writes the pid null sentinel",
        by_id["released-2"]["pid"] is None and not state.owner_alive(by_id["released-2"]),
    )
    env.check("mark_abandoned leaves the other record owned", by_id["kept-1"]["pid"] == os.getpid())
    cleanup(handed_over, env.make_cfg())
    env.check("cleanup reclaims only the released box", handed_over.destroyed == ["released-2"])
    env.use_main_state()


def _smoke_every_job_destroys_on_every_exit_path(env):
    real_run_eval = GrpoJob.run_eval
    GrpoJob.run_eval = lambda self, phase: {"phase": phase}
    for job_name in ("train", "eval", "grpo", "serve"):
        env.clear_bad_offers()
        happy = MockProvider(env.make_cfg())
        rc = run_remote_job(happy, env.make_cfg(), job_name, enable_safety=False)
        env.check(f"{job_name}: happy path returns 0", rc == 0)
        env.check(f"{job_name}: happy path destroyed the instance", len(happy.destroyed) == 1)
        if job_name == "grpo":
            env.check("grpo: tunnel opened R:8900 + L:8001", happy.tunnels == [[("R", 8900), ("L", 8001)]])
            env.check("grpo: both eval phases acked", sum("PHASE_ACK" in c for c in happy.exec_log) == 2)
        if job_name == "serve":
            env.check("serve: tunnel opened L:8001", happy.tunnels == [[("L", 8001)]])
        for label, provider in (
            ("exception", _RaiseProvider(env.make_cfg())),
            ("SIGINT", _InterruptProvider(env.make_cfg())),
        ):
            env.clear_bad_offers()
            try:
                run_remote_job(provider, env.make_cfg(), job_name, enable_safety=False)
            except (ProviderError, KeyboardInterrupt):
                pass
            env.check(f"{job_name}: {label} path still destroyed the instance", len(provider.destroyed) == 1)
    GrpoJob.run_eval = real_run_eval


def _smoke_a_failed_create_destroys_its_contract(env):
    destroyed_ids = []
    provider = VastAIProvider(env.make_cfg())
    provider._public_key = lambda: "ssh-ed25519 AAAA test"
    provider._cli_json = lambda *a, **k: {"success": False, "new_contract": 4242}
    provider.destroy = lambda instance: destroyed_ids.append(instance.id) or True
    try:
        provider.provision(
            Offer(id="1", price_per_hour=0.1, gpu_name="x", gpu_ram_gb=24, num_gpus=1, country="US", reliability=1.0),
            env.make_cfg(),
        )
    except ProviderError:
        pass
    env.check("vast: a failed create that allocated a contract destroys it", destroyed_ids == ["4242"])


class _RaiseProvider(MockProvider):
    def wait_ready(self, instance, timeout):
        raise ProviderError("simulated mid-run failure")


class _InterruptProvider(MockProvider):
    def wait_ready(self, instance, timeout):
        raise KeyboardInterrupt()


SMOKE_CASES = (
    _smoke_offer_selection,
    _smoke_ownership_gate,
    _smoke_upload_excludes,
    _smoke_cleanup_spares_a_live_sibling,
    _smoke_cleanup_reclaims_a_dead_owner,
    _smoke_owner_alive_sentinels,
    _smoke_release_hands_one_box_over,
    _smoke_every_job_destroys_on_every_exit_path,
    _smoke_a_failed_create_destroys_its_contract,
)


def run_smoke_test() -> int:
    """Run every case in SMOKE_CASES against the mock provider; adding a case is one function plus one entry."""
    env = _SmokeEnv()
    for case in SMOKE_CASES:
        case(env)
    failures = env.failures
    print(f"\ncua-remote self-test: {'ALL PASS' if not failures else f'{len(failures)} FAILURES'}")
    return 0 if not failures else 1
