"""One corpus pass: it names its candidates and what to do with one, and inherits the walk, the dry run, the concurrency, the resume and the report."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from cua.pipeline.sdk_runner import UsageLimitReached
from cua.trajectory.io import TrajectoryDir, iter_trajectory_dirs

logger = logging.getLogger(__name__)

# One wording for every pass: their resume story is the same — untouched candidates stay selectable.
USAGE_LIMIT_MESSAGE = "usage limit reached — untouched candidates remain selectable; re-run to resume"


@dataclass
class Outcome:
    """What one `apply` did: `note` is the status the tally counts, `detail` whatever the pass's own report row needs."""

    changed: bool
    note: str = ""
    cost: float | None = None
    path: Path | None = None
    detail: object = None


def tally(outcomes: list) -> dict:
    """Outcome notes counted in completion order — the dict every pass prints as its result line."""
    counts: dict = {}
    for outcome in outcomes:
        counts[outcome.note] = counts.get(outcome.note, 0) + 1
    return counts


class CorpusPass:
    """Subclasses set `name`, implement `select` and `apply`, and override a hook only where their output differs."""

    name = ""
    workers = 3
    report_path: Path | None = None

    def select(self, recorded: TrajectoryDir) -> bool:
        raise NotImplementedError

    async def apply(self, recorded: TrajectoryDir) -> Outcome:
        raise NotImplementedError

    def candidates(self, corpus: Path, limit: int | None = None) -> list[TrajectoryDir]:
        """Selected dirs under `corpus` in walk order, cut to `limit`."""
        selected = [r for r in (TrajectoryDir.load(path) for path in iter_trajectory_dirs(corpus)) if self.select(r)]
        return selected[:limit] if limit else selected

    def describe(self, corpus: Path, recorded: TrajectoryDir) -> str:
        """One dry-run line for one candidate."""
        return str(recorded.path.relative_to(corpus))

    def summary(self, corpus: Path, candidates: list) -> str:
        """The count line the pass prints under its dry run and logs before applying."""
        return f"{len(candidates)} candidates"

    def print_dry_run(self, corpus: Path, candidates: list) -> None:
        for recorded in candidates:
            print(self.describe(corpus, recorded))
        print(self.summary(corpus, candidates))

    def observe(self, corpus: Path, recorded: TrajectoryDir, outcome: Outcome) -> None:
        """Per-candidate progress, reported as each apply lands."""

    def report(self, corpus: Path, outcomes: list) -> str:
        """The pass's report file body; the default is the same tally the run returns."""
        return "\n".join(f"{note}: {count}" for note, count in tally(outcomes).items()) + "\n"

    def write_report(self, corpus: Path, outcomes: list) -> None:
        if self.report_path is None:
            return
        self.report_path.write_text(self.report(corpus, outcomes))
        logger.info(f"report: {self.report_path} ({len(outcomes)} rows)")

    async def run(self, corpus: Path, *, apply: bool = True, limit=None, workers=None, candidates=None) -> dict:
        """Walk, select, then either list the candidates or apply to each; returns the note tally."""
        # A caller that hands in its own candidate list owns reporting it too, so the summary is not repeated.
        own_selection = candidates is None
        candidates = self.candidates(corpus, limit) if own_selection else candidates
        if not apply:
            self.print_dry_run(corpus, candidates)
            return {}
        if own_selection:
            logger.info(self.summary(corpus, candidates))
        semaphore = asyncio.Semaphore(workers or self.workers)
        outcomes: list[Outcome] = []

        async def guarded(recorded: TrajectoryDir):
            async with semaphore:
                outcome = await self.apply(recorded)
            outcome.path = recorded.path
            outcomes.append(outcome)
            self.observe(corpus, recorded, outcome)

        try:
            await asyncio.gather(*(guarded(recorded) for recorded in candidates))
        except UsageLimitReached:
            logger.warning(USAGE_LIMIT_MESSAGE)
        self.write_report(corpus, outcomes)
        return tally(outcomes)
