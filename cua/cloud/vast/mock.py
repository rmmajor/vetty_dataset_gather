"""In-memory provider for the no-cost self-test: fake offers, an SSH-answering instance, a record of every call."""

import json
import logging
from pathlib import Path

from cua.cloud.vast.base import PROVIDERS, ComputeProvider, Instance, Offer, select_cheapest_allowed
from cua.cloud.vast.tunnel import NullTunnel

logger = logging.getLogger(__name__)


class MockProvider(ComputeProvider):
    name = "mock"

    def __init__(self, cfg):
        self.cfg = cfg
        self.destroyed: list = []
        self.exec_log: list = []
        self.uploads: list = []
        self.tunnels: list = []
        self.reported_label = cfg.instance_label  # tests set this to a foreign label to exercise the ownership gate
        self.phases = list(getattr(cfg, "mock_phases", ()))
        self.phase_calls = 0
        self._fake_offers = [
            Offer("cn-1", 0.30, "RTX 4090", 48, 1, "CN", 0.99),
            Offer("ru-1", 0.35, "RTX 4090", 48, 1, "RU", 0.99),
            Offer("se-1", 0.45, "A100 SXM4", 80, 1, "SE", 0.99),
            Offer("us-1", 0.55, "A100 SXM4", 80, 1, "US", 0.99),
            Offer("us-2", 0.60, "A100 SXM4", 80, 2, "US", 0.99),  # server-mode GRPO needs two GPUs
        ]

    def search_offer(self, cfg) -> Offer:
        return select_cheapest_allowed(self._fake_offers, cfg)

    def provision(self, offer, cfg) -> Instance:
        return Instance(id=f"mock-{offer.id}", provider=self.name, label=cfg.instance_label)

    def wait_ready(self, instance, timeout) -> None:
        instance.ssh_host, instance.ssh_port = "127.0.0.1", 22

    def exec(self, instance, command, *, stream=False, timeout=0) -> tuple:
        self.exec_log.append(command)
        if "PHASE_ACK" in command:
            return "", "", 0
        if "/PHASE" in command:
            phase = self.phases[min(self.phase_calls, len(self.phases) - 1)] if self.phases else ""
            self.phase_calls += 1
            return phase, "", 0
        if "STATUS" in command:
            # A job that reads phases keeps the box "RUNNING" until its scripted phases are consumed.
            running = 0 < self.phase_calls < len(self.phases)
            return ("RUNNING" if running else "DONE"), "", 0
        return "", "", 0

    def tunnel(self, instance, cfg, forwards):
        self.tunnels.append(list(forwards))
        return NullTunnel(forwards)

    def upload(self, instance, local_path, remote_path, *, exclude=()) -> None:
        self.uploads.append({"local": local_path, "remote": remote_path, "exclude": tuple(exclude)})
        self.exec_log.append(f"UPLOAD {local_path} -> {remote_path}")

    def download(self, instance, remote_path, local_path) -> None:
        self.exec_log.append(f"DOWNLOAD {remote_path} -> {local_path}")
        out = Path(local_path)
        if not (remote_path.endswith("/") or local_path.endswith("/")):
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("mock remote log\n")
            return
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text("{}")
        (out / "comparison.json").write_text(
            json.dumps(
                {
                    "summaries": {
                        "mock": {
                            "total": 2,
                            "hits": 1,
                            "accuracy": 0.5,
                            "parse_rate": 1.0,
                            "by_data_type": {},
                            "by_data_source": {},
                        }
                    }
                }
            )
        )

    def destroy(self, instance) -> bool:
        self.destroyed.append(instance.id)
        logger.info(f"[mock] destroyed {instance.id}")
        return True

    def list_owned(self, label) -> list:
        return []

    def live_label(self, instance) -> str | None:
        return self.reported_label


def build_mock_provider(cfg) -> ComputeProvider:
    return MockProvider(cfg)


PROVIDERS["mock"] = build_mock_provider
