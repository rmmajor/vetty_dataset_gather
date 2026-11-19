"""`--notify` was inert until the lifecycle called the notifier: exactly one ping per run, on either outcome."""

import tempfile
from pathlib import Path

from cua.cloud.vast import lifecycle, state
from cua.cloud.vast.config import RemoteTrainConfig
from cua.cloud.vast.lifecycle import run_remote_job
from cua.cloud.vast.mock import MockProvider


class FailedStatusProvider(MockProvider):
    def exec(self, instance, command, *, stream=False, timeout=0):
        if "STATUS" in command:
            return "FAILED", "", 0
        return super().exec(instance, command, stream=stream, timeout=timeout)


def _train_cfg(root: Path) -> RemoteTrainConfig:
    cfg = RemoteTrainConfig()
    cfg.trajectories_dir = str(root / "trajectories")
    cfg.train_config = str(root / "config.yaml")
    cfg.output_dir = str(root / "output")
    cfg.poll_interval_s = 0
    cfg.notify = True
    (root / "trajectories" / "task_x").mkdir(parents=True)
    (root / "trajectories" / "task_x" / "trajectory.json").write_text("{}")
    (root / "config.yaml").write_text("model: test\noutput_dir: ./output\n")
    (root / "output").mkdir()
    (root / "output" / "adapter_config.json").write_text("{}")
    (root / "output" / "adapter_model.safetensors").write_bytes(b"0" * 16)
    return cfg


def test_notify_fires_once_on_success_and_once_on_failure():
    with tempfile.TemporaryDirectory() as tmp:
        sent = []
        saved = (lifecycle._notify, state.STATE_FILE, state.BAD_OFFER_FILE)
        lifecycle._notify = sent.append
        state.STATE_FILE = Path(tmp) / "state.json"
        state.BAD_OFFER_FILE = Path(tmp) / "bad_offers.json"
        try:
            happy_cfg = _train_cfg(Path(tmp) / "happy")
            assert run_remote_job(MockProvider(happy_cfg), happy_cfg, "train", enable_safety=False) == 0
            assert len(sent) == 1 and "DONE" in sent[0] and "train" in sent[0]

            failed_cfg = _train_cfg(Path(tmp) / "failed")
            failing = FailedStatusProvider(failed_cfg)
            assert run_remote_job(failing, failed_cfg, "train", enable_safety=False) == 1
            assert len(sent) == 2 and "FAILED" in sent[1]

            quiet_cfg = _train_cfg(Path(tmp) / "quiet")
            quiet_cfg.notify = False
            assert run_remote_job(MockProvider(quiet_cfg), quiet_cfg, "train", enable_safety=False) == 0
            assert len(sent) == 2
        finally:
            lifecycle._notify, state.STATE_FILE, state.BAD_OFFER_FILE = saved
