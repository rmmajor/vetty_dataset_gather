"""In-memory provider for the no-cost self-test (`python -m remote_train run --provider mock`).

Simulates the whole lifecycle without renting anything: fixed fake offers (including a cheap
RU and a cheap CN one so the geo-exclusion is observable), an instance that "answers" SSH,
exec that reports training DONE, and a download that drops a fake adapter. Records every
`destroy` so the self-test can assert teardown ran in every exit path. Mirrors
`eval_screenspot.py`'s MockPredictor.
"""
import logging
from pathlib import Path

from remote_train.base import ComputeProvider, Instance, Offer, PROVIDERS, select_cheapest_allowed

logger = logging.getLogger(__name__)


class MockProvider(ComputeProvider):
    name = "mock"

    def __init__(self, cfg):
        self.cfg = cfg
        self.destroyed: list = []
        self.exec_log: list = []
        self._fake_offers = [
            Offer("cn-1", 0.30, "RTX 4090", 48, 1, "CN", 0.99),
            Offer("ru-1", 0.35, "RTX 4090", 48, 1, "RU", 0.99),
            Offer("se-1", 0.45, "A100 SXM4", 80, 1, "SE", 0.99),
            Offer("us-1", 0.55, "A100 SXM4", 80, 1, "US", 0.99),
        ]

    def search_offer(self, cfg) -> Offer:
        return select_cheapest_allowed(self._fake_offers, cfg)

    def provision(self, offer, cfg) -> Instance:
        return Instance(id=f"mock-{offer.id}", provider=self.name, label=cfg.instance_label)

    def wait_ready(self, instance, timeout) -> None:
        instance.ssh_host, instance.ssh_port = "127.0.0.1", 22

    def exec(self, instance, command, *, stream=False, timeout=0) -> tuple:
        self.exec_log.append(command)
        if "STATUS" in command:
            return "DONE", "", 0
        return "", "", 0

    def upload(self, instance, local_path, remote_path) -> None:
        self.exec_log.append(f"UPLOAD {local_path} -> {remote_path}")

    def download(self, instance, remote_path, local_path) -> None:
        self.exec_log.append(f"DOWNLOAD {remote_path} -> {local_path}")
        out = Path(local_path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text("{}")

    def destroy(self, instance) -> bool:
        self.destroyed.append(instance.id)
        logger.info(f"[mock] destroyed {instance.id}")
        return True

    def list_owned(self, label) -> list:
        return []


def build_mock_provider(cfg) -> ComputeProvider:
    return MockProvider(cfg)


PROVIDERS["mock"] = build_mock_provider
