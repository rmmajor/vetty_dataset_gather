"""Provider-pluggable remote training (Vast.ai first).

Importing this package registers every provider into `base.PROVIDERS`, so `build_provider`
and the `--provider` CLI choices are populated. Add a provider by writing a module that
appends its factory to `PROVIDERS` and importing it here.
"""
from remote_train.base import (
    ComputeProvider,
    Instance,
    Offer,
    PROVIDERS,
    ProviderError,
    build_provider,
    is_excluded_country,
    select_cheapest_allowed,
)
from remote_train.config import RemoteTrainConfig, load_config
from remote_train.benchmarks import BENCHMARKS, Benchmark, build_benchmark
from remote_train.jobs import JOBS, RemoteJob, build_job
from remote_train import vastai  # noqa: F401 — registers "vastai"
from remote_train import mock  # noqa: F401 — registers "mock"

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
