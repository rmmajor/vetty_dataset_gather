"""Provider-agnostic compute abstraction for remote training.

A `ComputeProvider` is the entire surface the training lifecycle (`lifecycle.py`) sees:
find a GPU offer, rent it, run commands / move files over SSH, and tear it down. Vast.ai
is the first implementation (`vastai.py`); a second provider (RunPod, Lambda, …) is a new
module implementing the eight methods below plus one line appending its factory to
`PROVIDERS` — the lifecycle, CLI, and Makefile never change. This mirrors
`eval_screenspot.py`'s predictor-backend registry: a dict of name -> factory function,
selected by a `--provider` flag validated against `sorted(PROVIDERS)`.

Offer selection (`select_cheapest_allowed`) and the country deny-list (`is_excluded_country`)
live here as pure functions so every provider shares one policy and the self-test can
exercise them without renting anything.
"""
from dataclasses import dataclass, field

from remote_train.tunnel import NullTunnel


class ProviderError(RuntimeError):
    """A provider operation failed (search, provision, ssh, transfer, destroy)."""


@dataclass(frozen=True)
class Offer:
    """One rentable GPU offer, normalized across providers. `raw` keeps the native record."""
    id: str
    price_per_hour: float
    gpu_name: str
    gpu_ram_gb: float
    num_gpus: int
    country: str
    reliability: float
    compute_cap: int = 0  # 750=Turing (no bf16), 800=Ampere, 890=Ada, 1200=Blackwell
    raw: dict = field(default_factory=dict)


@dataclass
class Instance:
    """A provisioned instance. `ssh_host`/`ssh_port` are filled once `wait_ready` succeeds."""
    id: str
    provider: str
    label: str
    ssh_host: str = ""
    ssh_port: int = 0
    ssh_user: str = "root"
    raw: dict = field(default_factory=dict)


class ComputeProvider:
    """The contract every GPU provider implements. Methods raise `ProviderError` on failure.

    Implement these eight methods and append a factory to `PROVIDERS`; nothing else in the
    codebase needs to change to support a new provider.
    """
    name: str = "base"

    def search_offer(self, cfg) -> Offer:
        """Cheapest offer meeting cfg constraints and NOT in an excluded country."""
        raise NotImplementedError

    def provision(self, offer: Offer, cfg) -> Instance:
        """Rent `offer`, tagged with cfg.instance_label. Returns an Instance with a stable id."""
        raise NotImplementedError

    def wait_ready(self, instance: Instance, timeout: int) -> None:
        """Block until the instance runs and SSH answers; fill ssh_host/ssh_port in place."""
        raise NotImplementedError

    def exec(self, instance: Instance, command: str, *, stream: bool = False, timeout: int = 0) -> tuple[str, str, int]:
        """Run a shell command over SSH. Returns (stdout, stderr, exit_code)."""
        raise NotImplementedError

    def upload(self, instance: Instance, local_path: str, remote_path: str) -> None:
        """Copy a local file/dir to the instance."""
        raise NotImplementedError

    def download(self, instance: Instance, remote_path: str, local_path: str) -> None:
        """Copy a file/dir from the instance to local."""
        raise NotImplementedError

    def destroy(self, instance: Instance) -> bool:
        """Tear the instance down and VERIFY it is gone. Idempotent. Returns True once confirmed."""
        raise NotImplementedError

    def list_owned(self, label: str) -> list:
        """All live instances carrying `label` — powers the orphan sweep in cleanup."""
        raise NotImplementedError

    def tunnel(self, instance: Instance, cfg, forwards: list):
        """Optional: a context manager forwarding loopback ports (("R"|"L", port) pairs). Default: no tunnel."""
        return NullTunnel(forwards)


PROVIDERS: dict = {}


def build_provider(name: str, cfg) -> ComputeProvider:
    if name not in PROVIDERS:
        raise ProviderError(f"Unknown provider {name!r}. Available: {sorted(PROVIDERS)}")
    return PROVIDERS[name](cfg)


_COUNTRY_ALIASES = {
    "RU": ("RUSSIA", "RUSSIAN FEDERATION"),
    "CN": ("CHINA", "PEOPLE'S REPUBLIC OF CHINA"),
    "HK": ("HONG KONG",),
}


def normalize_country(geolocation) -> str:
    """Vast geolocation may be 'US', 'CN', or 'Sweden, SE' — return the best-effort upper code."""
    raw = str(geolocation or "").strip()
    if not raw:
        return ""
    return raw.split(",")[-1].strip().upper()


def is_excluded_country(geolocation, exclude_codes) -> bool:
    """True if an offer's location matches the deny-list, by code or country-name alias."""
    raw = str(geolocation or "").strip().upper()
    code = normalize_country(geolocation)
    for ex in exclude_codes:
        ex = str(ex).strip().upper()
        if not ex:
            continue
        if code == ex or raw == ex:
            return True
        if any(alias in raw for alias in _COUNTRY_ALIASES.get(ex, ())):
            return True
    return False


def select_cheapest_allowed(offers: list, cfg) -> Offer:
    """Pure shared selection: drop excluded countries, enforce price/vram, cheapest wins.

    Used by every provider so the geo/price/vram policy is defined once. Raises
    `ProviderError` (with a diagnostic) if nothing qualifies.
    """
    allowed = [
        o for o in offers
        if not is_excluded_country(o.country, cfg.exclude_countries)
        and o.price_per_hour <= cfg.max_price_per_hour
        and o.gpu_ram_gb >= cfg.min_vram_gb
    ]
    if not allowed:
        raise ProviderError(
            f"No offer matched: need >= {cfg.min_vram_gb}GB VRAM, <= ${cfg.max_price_per_hour}/hr, "
            f"excluding countries {sorted(cfg.exclude_countries)} "
            f"(scanned {len(offers)} offers). Loosen constraints in remote_train.yaml."
        )
    return min(allowed, key=lambda o: o.price_per_hour)
