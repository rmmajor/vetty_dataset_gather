"""Provider-pluggable remote jobs: importing this package registers every provider into `base.PROVIDERS`, which populates `--provider`."""

from cua.cloud.vast import (
    mock,  # noqa: F401 — registers "mock"
    vastai,  # noqa: F401 — registers "vastai"
)
from cua.cloud.vast.base import (
    PROVIDERS,
    ComputeProvider,
    Instance,
    Offer,
    ProviderError,
    build_provider,
    is_excluded_country,
    select_cheapest_allowed,
)
from cua.cloud.vast.benchmarks import BENCHMARKS, Benchmark, build_benchmark
from cua.cloud.vast.config import RemoteTrainConfig, load_config
from cua.cloud.vast.jobs import JOBS, RemoteJob, build_job

__all__ = [
    "ComputeProvider",
    "Instance",
    "Offer",
    "PROVIDERS",
    "ProviderError",
    "build_provider",
    "is_excluded_country",
    "select_cheapest_allowed",
    "RemoteTrainConfig",
    "load_config",
    "BENCHMARKS",
    "Benchmark",
    "build_benchmark",
    "JOBS",
    "RemoteJob",
    "build_job",
]
