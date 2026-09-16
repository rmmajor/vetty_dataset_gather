"""Run the browser driver on a REMOTE host over ssh, so a live site sees that host's IP instead of this one's."""

import json
import logging
import re
import shlex
import subprocess
import time
from pathlib import Path

from cua.pipeline.sandbox import ShellSandbox

logger = logging.getLogger(__name__)

REMOTE_WORKDIR = "/root/eval"
SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "ServerAliveInterval=15",
]
# Measured: a datacenter exit recovers 1 of 24 blocked sites where a residential one recovers 9.
RESIDENTIAL_ASN = re.compile(
    r"charter|comcast|at&t|verizon|frontier|cox communication|spectrum|centurylink|"
    r"sonic|windstream|optimum|altice|t-mobile|lumen",
    re.I,
)
PLAYWRIGHT_IMAGE = "mcr.microsoft.com/playwright:v1.55.0-noble"
DEFAULT_TIMEOUT = 1800


class RemoteSandbox(ShellSandbox):
    """A ShellSandbox backed by ssh to an already-running host."""

    def __init__(
        self,
        host: str,
        port: int = 22,
        user: str = "root",
        key_path: str = "~/.ssh/id_rsa",
        container_timeout: int = DEFAULT_TIMEOUT,
        workdir: str = REMOTE_WORKDIR,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.key_path = str(Path(key_path).expanduser())
        self.container_timeout = container_timeout
        self.workdir = workdir
        self.start_time = None

    def _ssh_base(self) -> list:
        return ["ssh", "-i", self.key_path, "-p", str(self.port), *SSH_OPTS, f"{self.user}@{self.host}"]

    def exec_command(self, command: str, timeout: int = 300, max_output: int = 50000):
        try:
            r = subprocess.run(self._ssh_base() + [command], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "", f"ssh exec timed out after {timeout}s", 124
        return r.stdout[:max_output], r.stderr[:max_output], r.returncode

    def put_bytes(self, path: str, data: bytes):
        """Stream bytes in over ssh stdin, so the payload never crosses the command line."""
        cmd = f"mkdir -p $(dirname {shlex.quote(path)}) && cat > {shlex.quote(path)}"
        r = subprocess.run(self._ssh_base() + [cmd], input=data, capture_output=True, timeout=300)
        if r.returncode != 0:
            raise OSError(f"put_bytes {path} failed: {r.stderr.decode()[:200]}")

    def copy_out(self, remote_path: str, host_path: str):
        Path(host_path).mkdir(parents=True, exist_ok=True)
        ssh = f"ssh -i {self.key_path} -p {self.port} " + " ".join(SSH_OPTS)
        r = subprocess.run(
            ["rsync", "-az", "-e", ssh, f"{self.user}@{self.host}:{remote_path}/", f"{host_path}/"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r.returncode != 0:
            raise OSError(f"copy_out {remote_path} failed: {r.stderr[:200]}")

    def start(self, name: str) -> str:
        self.start_time = time.time()
        # The Playwright image ships the browsers but not the npm package the driver imports.
        setup = (
            f"mkdir -p {self.workdir} /opt/driver && cd /opt/driver && "
            "([ -d node_modules/playwright-core ] || (npm init -y >/dev/null 2>&1 && "
            "PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm i playwright-core@1.55.0 >/dev/null 2>&1))"
        )
        stdout, stderr, rc = self.exec_command(setup, timeout=600)
        if rc != 0:
            raise RuntimeError(f"remote sandbox setup failed on {self.host}: {stderr[:300]}")
        logger.info(f"Remote sandbox ready on {self.host}:{self.port} ({name})")
        return self.host

    def remaining_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return self.container_timeout - (time.time() - self.start_time)

    def stop(self):
        """No-op for a host we did not rent — the caller owns its lifetime."""
        self.start_time = None


class VastResidentialSandbox(RemoteSandbox):
    """A RemoteSandbox on a freshly rented Vast box with a consumer-ISP IP; destroyed on stop."""

    def __init__(
        self, container_timeout: int = DEFAULT_TIMEOUT, max_price: float = 0.35, key_path: str = "~/.ssh/id_rsa"
    ):
        super().__init__(host="", key_path=key_path, container_timeout=container_timeout)
        self.max_price = max_price
        self.instance = None
        self.provider = None
        self.egress = {}

    def start(self, name: str) -> str:
        # Keeps the Vast rental stack out of the import graph of every capture lane that never rents
        from cua.cloud.vast.base import Offer
        from cua.cloud.vast.config import RemoteTrainConfig
        from cua.cloud.vast.vastai import VastAIProvider

        cfg = RemoteTrainConfig()
        cfg.image = PLAYWRIGHT_IMAGE
        cfg.disk_gb = 32
        self.provider = VastAIProvider(cfg)
        last_error = None
        for offer_id in residential_offer_ids(self.max_price):
            try:
                offer = Offer(
                    id=offer_id,
                    price_per_hour=0.0,
                    gpu_name="",
                    gpu_ram_gb=0,
                    num_gpus=1,
                    country="US",
                    reliability=1.0,
                )
                self.instance = self.provider.provision(offer, cfg)
                self.provider.wait_ready(self.instance, cfg.ready_timeout_s)
                break
            except Exception as e:
                last_error = e
                logger.warning(f"vast offer {offer_id} unusable ({str(e)[:120]}) — next")
                self._destroy()
        if self.instance is None:
            raise RuntimeError(f"no residential Vast offer booted; last error: {last_error}")
        self.host, self.port = self.instance.ssh_host, self.instance.ssh_port
        self.key_path = str(Path(cfg.ssh_key_path).expanduser())
        try:
            self.egress = json.loads(
                self.exec_command("curl -s --max-time 20 https://ipinfo.io/json", timeout=60)[0] or "{}"
            )
        except Exception:
            self.egress = {}
        logger.info(f"Rented egress: {self.egress.get('city')} {self.egress.get('country')} {self.egress.get('org')}")
        return super().start(name)

    def _destroy(self):
        if self.instance is not None and self.provider is not None:
            try:
                self.provider.destroy(self.instance)
            except Exception as e:
                logger.error(f"FAILED to destroy Vast instance {self.instance.id}: {e}")
            self.instance = None

    def stop(self):
        self._destroy()
        super().stop()


def residential_offer_ids(max_price: float = 0.35, want: int = 8) -> list:
    """Vast offer ids whose advertised public IP resolves to a consumer ISP, cheapest first."""
    query = f"geolocation=US rentable=true dph<{max_price} disk_space>40 inet_down>100 num_gpus=1"
    raw = subprocess.run(
        ["vastai", "search", "offers", query, "--raw", "--limit", "200"], capture_output=True, text=True, timeout=180
    ).stdout
    try:
        offers = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("vastai search offers returned non-JSON")
        return []
    picked = []
    for offer in sorted(offers, key=lambda o: o.get("dph_total", 9)):
        ip = offer.get("public_ipaddr")
        if not ip:
            continue
        try:
            info = json.loads(
                subprocess.run(
                    ["curl", "-s", "--max-time", "8", f"https://ipinfo.io/{ip}/json"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                ).stdout
            )
        except Exception:
            continue
        if info.get("country") == "US" and RESIDENTIAL_ASN.search(info.get("org", "")):
            picked.append(str(offer["id"]))
            logger.info(
                f"candidate offer {offer['id']} ${offer.get('dph_total'):.3f}/h {ip} {info.get('org', '')[:40]}"
            )
        if len(picked) >= want:
            break
    return picked
