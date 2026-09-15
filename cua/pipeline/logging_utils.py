"""Per-repo, per-stage logging context and a live progress tracker, so interleaved worker output stays attributable."""

import logging


class _StageRepoAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[stage {self.extra['stage']}][{self.extra['repo']}] {msg}", kwargs


def repo_logger(stage: int, repo_name: str) -> logging.LoggerAdapter:
    return _StageRepoAdapter(
        logging.getLogger(f"cua.pipeline.stage{stage}"),
        {"stage": stage, "repo": repo_name},
    )


class StageProgress:
    """Live counters for one stage run, mutated without locks from the per-repo coroutines on the one event loop."""

    def __init__(self, stage: int, total: int):
        self.stage = stage
        self.total = total
        self.logger = logging.getLogger(f"cua.pipeline.stage{stage}")
        self.started = 0
        self.passed = 0
        self.rejected = 0
        self.rejected_infra = 0
        self.retries = 0
        self.trajectories = 0

    @property
    def done(self) -> int:
        return self.passed + self.rejected + self.rejected_infra

    @property
    def running(self) -> int:
        return self.started - self.done

    def start(self):
        self.started += 1

    def passed_repo(self, trajectories: int = 0):
        self.passed += 1
        self.trajectories += trajectories
        self._tick()

    def rejected_repo(self, infra: bool = False):
        if infra:
            self.rejected_infra += 1
        else:
            self.rejected += 1
        self._tick()

    def retry(self):
        self.retries += 1

    def _tick(self):
        parts = [f"{self.passed} passed", f"{self.rejected} rejected"]
        if self.rejected_infra:
            parts.append(f"{self.rejected_infra} infra-rejected")
        if self.running:
            parts.append(f"{self.running} running")
        if self.retries:
            parts.append(f"{self.retries} retries")
        if self.trajectories:
            parts.append(f"{self.trajectories} trajectories")
        self.logger.info(f"[stage {self.stage}] {self.done}/{self.total} done — " + ", ".join(parts))

    def summary(self) -> dict:
        self.logger.info(
            f"[stage {self.stage}] complete — {self.passed} passed, {self.rejected} rejected, "
            f"{self.rejected_infra} infra-rejected, {self.retries} retries, "
            f"{self.trajectories} trajectories"
        )
        return {
            "processed": self.total,
            "passed": self.passed,
            "rejected": self.rejected + self.rejected_infra,
            "rejected_infra": self.rejected_infra,
            "retries": self.retries,
            "trajectories": self.trajectories,
        }
