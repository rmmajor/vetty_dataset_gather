"""Vast.ai compute provider — shells out to the `vastai` CLI (`pip install vastai`).

Implements the `ComputeProvider` contract via the official CLI, matching the repo's
shell-out idiom (`subprocess.run` with capture/text/timeout + custom exception + audit
logging, like `repo_pipeline/bootable_gate.py:probe_repo` and `train_qwen.py:train_model`).
File transfer and command execution go over the instance's `ssh_host:ssh_port` using the
private key in `cfg.ssh_key_path` (its public half must be attached to the Vast.ai account).

Schema validated against `vastai` CLI 1.1.3 (2026-06): search-DSL `gpu_ram`/`disk_space`
are GB while the raw offer `gpu_ram` field is MB (`_offer_from_raw` normalizes by /1024);
`geolocation` is `"Country, CC"` (the trailing code is the country); reliability lives in
`reliability2`; `verified` is often `None` even on good hosts, so the query gates on
`reliability` + `disk_space` rather than `verified=true`. The disk gate matters: large
models (e.g. 27B ≈ 54GB of weights) won't fit hosts with tiny `disk_space`. Field reads
stay tolerant (`.get` with fallbacks); re-confirm with `make remote-train-dry` if the CLI
version changes.
"""
import json
import logging
import os
import subprocess
import time
from pathlib import Path

from remote_train.base import (
    ComputeProvider,
    Instance,
    Offer,
    PROVIDERS,
    ProviderError,
    normalize_country,
    select_cheapest_allowed,
)

logger = logging.getLogger(__name__)

_CLI_TIMEOUT = 120
_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15"]


class VastAIProvider(ComputeProvider):
    name = "vastai"

    def __init__(self, cfg):
        self.cfg = cfg

    # ----- offer selection / provisioning -----

    def search_offer(self, cfg) -> Offer:
        query = (
            f"reliability > {cfg.min_reliability} num_gpus={cfg.num_gpus} "
            f"gpu_ram >= {cfg.min_vram_gb} compute_cap >= {cfg.min_compute_cap} "
            f"dph_total <= {cfg.max_price_per_hour} "
            f"disk_space >= {cfg.disk_gb} rentable=true"
        )
        if cfg.gpu_name:
            query += f" gpu_name={cfg.gpu_name}"
        raw = self._cli_json(["search", "offers", query, "-o", "dph_total", "--limit", "200"])
        records = raw if isinstance(raw, list) else raw.get("offers", [])
        offers = [self._offer_from_raw(r) for r in records]
        offer = select_cheapest_allowed(offers, cfg)
        logger.info(
            f"Selected Vast.ai offer {offer.id}: {offer.gpu_name} x{offer.num_gpus}, "
            f"{offer.gpu_ram_gb}GB, compute_cap {offer.compute_cap}, "
            f"${offer.price_per_hour:.3f}/hr, {offer.country}, reliability {offer.reliability:.2f}"
        )
        return offer

    def provision(self, offer, cfg) -> Instance:
        pub = self._public_key()
        pub_min = " ".join(pub.split()[:2])  # drop the comment — it goes through a shell echo
        onstart = (
            "mkdir -p /root/.ssh && "
            f"echo '{pub_min}' >> /root/.ssh/authorized_keys && "
            "chmod 600 /root/.ssh/authorized_keys && touch /workspace/.provisioned"
        )
        result = self._cli_json([
            "create", "instance", offer.id,
            "--image", cfg.image,
            "--disk", str(cfg.disk_gb),
            "--label", cfg.instance_label,
            "--onstart-cmd", onstart,
            "--ssh", "--direct",
        ])
        if isinstance(result, dict) and result.get("success") is False:
            raise ProviderError(f"Vast.ai create failed for offer {offer.id}: {result}")
        instance_id = ""
        if isinstance(result, dict):
            instance_id = str(result.get("new_contract") or result.get("id") or "")
        if not instance_id:
            raise ProviderError(f"Vast.ai create returned no instance id: {result}")
        logger.info(f"Provisioned Vast.ai instance {instance_id} (label {cfg.instance_label})")
        self._attach_ssh_key(instance_id, pub)
        return Instance(id=instance_id, provider=self.name, label=cfg.instance_label, raw=result if isinstance(result, dict) else {})

    def _public_key(self) -> str:
        """Our SSH public key, from `<ssh_key_path>.pub` or derived from the private key."""
        key = Path(os.path.expanduser(self.cfg.ssh_key_path))
        pub = Path(str(key) + ".pub")
        if pub.is_file():
            return pub.read_text().strip()
        result = subprocess.run(["ssh-keygen", "-y", "-f", str(key)], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise ProviderError(f"could not derive a public key from {key}: {result.stderr.strip()[:200]}")
        return result.stdout.strip()

    def _attach_ssh_key(self, instance_id: str, pub: str) -> None:
        """Attach our key to THIS instance.

        Team-scoped API keys cannot register account-wide keys (Vast: "Team SSH keys are not
        supported"), so we attach per-instance — which is tighter anyway, and keeps a shared
        team account free of one-off keys. The onstart hook above is the fallback.
        """
        try:
            self._cli(["attach", "ssh", str(instance_id), pub])
            logger.info(f"Attached local SSH key to instance {instance_id}")
        except ProviderError as e:
            logger.warning(f"attach ssh failed ({e}) — falling back to the onstart-injected authorized_keys")

    def wait_ready(self, instance, timeout) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            info = self._show_instance(instance.id)
            status = info.get("actual_status") or info.get("cur_state") or ""
            host = info.get("ssh_host") or info.get("public_ipaddr")
            port = info.get("ssh_port")
            if status == "running" and host and port:
                instance.ssh_host = str(host)
                instance.ssh_port = int(port)
                instance.raw = info
                if self._ssh_probe(instance):
                    logger.info(f"Instance {instance.id} ready at {host}:{port}")
                    return
            time.sleep(10)
        raise ProviderError(f"Instance {instance.id} not SSH-ready within {timeout}s")

    # ----- remote exec / transfer -----

    def exec(self, instance, command, *, stream=False, timeout=0) -> tuple:
        cmd = self._ssh_base(instance) + [command]
        logger.debug(f"[vastai exec] {command[:200]}")
        if stream:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            lines = []
            for line in proc.stdout:
                print(line, end="")
                lines.append(line)
            proc.wait()
            return "".join(lines), "", proc.returncode
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout or _CLI_TIMEOUT)
        except subprocess.TimeoutExpired as e:
            raise ProviderError(f"ssh exec timed out after {timeout or _CLI_TIMEOUT}s: {command[:120]}") from e
        return r.stdout, r.stderr, r.returncode

    def upload(self, instance, local_path, remote_path) -> None:
        self._rsync(instance, local_path, remote_path, upload=True)

    def download(self, instance, remote_path, local_path) -> None:
        self._rsync(instance, remote_path, local_path, upload=False)

    # ----- teardown -----

    def destroy(self, instance) -> bool:
        for attempt in range(1, 6):
            try:
                # `-y` is NOT optional: without it the CLI stops at an interactive confirmation,
                # reads EOF, prints "Aborted." and still EXITS 0 — i.e. a silent no-op that looks
                # like success and leaks a running (billing) instance.
                self._cli(["destroy", "instance", instance.id, "-y"])
            except ProviderError as e:
                logger.warning(f"destroy call failed (attempt {attempt}): {e}")
            time.sleep(5)
            if not self._instance_alive(instance.id):
                logger.info(f"Instance {instance.id} confirmed destroyed")
                return True
            logger.warning(f"Instance {instance.id} still present after destroy attempt {attempt}; retrying")
        logger.error(
            f"FAILED to confirm destruction of instance {instance.id} — CHECK THE VAST.AI CONSOLE "
            f"NOW to avoid charges (or run `make remote-train-cleanup`)"
        )
        return False

    def list_owned(self, label) -> list:
        raw = self._cli_json(["show", "instances"])
        records = raw if isinstance(raw, list) else raw.get("instances", [])
        owned = []
        for r in records:
            if str(r.get("label") or "") == label:
                owned.append(Instance(
                    id=str(r.get("id")), provider=self.name, label=label,
                    ssh_host=str(r.get("ssh_host") or ""), ssh_port=int(r.get("ssh_port") or 0), raw=r,
                ))
        return owned

    # ----- internals -----

    def _ssh_base(self, instance) -> list:
        key = os.path.expanduser(self.cfg.ssh_key_path)
        return ["ssh", "-p", str(instance.ssh_port), "-i", key, *_SSH_OPTS,
                f"{instance.ssh_user}@{instance.ssh_host}"]

    def _rsync(self, instance, src, dst, *, upload) -> None:
        key = os.path.expanduser(self.cfg.ssh_key_path)
        ssh = "ssh -p {port} -i {key} {opts}".format(port=instance.ssh_port, key=key, opts=" ".join(_SSH_OPTS))
        remote_side = f"{instance.ssh_user}@{instance.ssh_host}:{dst if upload else src}"
        args = ["rsync", "-az", "-e", ssh]
        args += [src, remote_side] if upload else [remote_side, dst]
        logger.info(f"rsync {'up' if upload else 'down'}: {src} -> {dst}")
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired as e:
            raise ProviderError(f"rsync timed out: {src} -> {dst}") from e
        if r.returncode != 0:
            raise ProviderError(f"rsync failed ({src} -> {dst}): {r.stderr.strip()[:300]}")

    def _ssh_probe(self, instance) -> bool:
        try:
            _, _, code = self.exec(instance, "true", timeout=20)
        except ProviderError:
            return False
        return code == 0

    def _show_instance(self, instance_id) -> dict:
        try:
            raw = self._cli_json(["show", "instance", str(instance_id)])
        except ProviderError as e:
            logger.warning(f"show instance {instance_id} failed: {e}")
            return {}
        if isinstance(raw, list):
            return raw[0] if raw else {}
        return raw if isinstance(raw, dict) else {}

    def _instance_alive(self, instance_id) -> bool:
        try:
            raw = self._cli_json(["show", "instances"])
        except ProviderError:
            return True  # can't confirm absence → assume alive so destroy keeps retrying
        records = raw if isinstance(raw, list) else raw.get("instances", [])
        return any(str(r.get("id")) == str(instance_id) for r in records)

    def _offer_from_raw(self, r: dict) -> Offer:
        gpu_ram = float(r.get("gpu_ram", 0) or 0)
        gpu_ram_gb = round(gpu_ram / 1024, 1) if gpu_ram > 1024 else gpu_ram  # raw field is MB
        return Offer(
            id=str(r.get("id") or r.get("ask_contract_id") or ""),
            price_per_hour=float(r.get("dph_total", 0.0) or 0.0),
            gpu_name=str(r.get("gpu_name", "")),
            gpu_ram_gb=gpu_ram_gb,
            num_gpus=int(r.get("num_gpus", 1) or 1),
            country=normalize_country(r.get("geolocation")),
            reliability=float(r.get("reliability2", r.get("reliability", 0)) or 0),
            compute_cap=int(r.get("compute_cap", 0) or 0),
            raw=r,
        )

    def _cli(self, args: list) -> str:
        full = ["vastai", *args]
        if self.cfg.api_key:
            full += ["--api-key", self.cfg.api_key]
        logger.debug("Running: vastai " + " ".join(args))
        try:
            r = subprocess.run(full, capture_output=True, text=True, timeout=_CLI_TIMEOUT)
        except FileNotFoundError as e:
            raise ProviderError("`vastai` CLI not found — run `make install-remote` (pip install vastai)") from e
        except subprocess.TimeoutExpired as e:
            raise ProviderError(f"vastai {' '.join(args[:2])} timed out after {_CLI_TIMEOUT}s") from e
        if r.returncode != 0:
            raise ProviderError(f"vastai {' '.join(args[:2])} failed: {r.stderr.strip()[:300]}")
        # A vastai subcommand that hits an interactive confirmation with no TTY prints "Aborted."
        # and STILL exits 0. Treat that as the failure it is, so no caller can mistake a no-op
        # for success (this is how a destroy silently left a billing instance alive).
        if "Aborted." in r.stdout:
            raise ProviderError(
                f"vastai {' '.join(args[:2])} aborted at an interactive prompt (exit 0, no action taken) "
                f"— a confirmation flag such as `-y` is missing"
            )
        return r.stdout

    def _cli_json(self, args: list):
        out = self._cli([*args, "--raw"])
        try:
            return json.loads(out)
        except json.JSONDecodeError as e:
            raise ProviderError(f"vastai {' '.join(args[:2])} returned non-JSON: {out[:200]}") from e


def build_vastai_provider(cfg) -> ComputeProvider:
    if not cfg.api_key:
        raise ProviderError("VAST_API_KEY env var required for the vastai provider")
    return VastAIProvider(cfg)


PROVIDERS["vastai"] = build_vastai_provider
