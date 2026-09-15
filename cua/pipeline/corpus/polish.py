"""Stage-7 thought polish over trajectory dirs the registry flow cannot reach (site_* lanes)."""

import asyncio
import logging
import sys
from pathlib import Path

from cua.pipeline.corpus.cli import build_parser
from cua.pipeline.corpus.pass_base import CorpusPass, Outcome
from cua.pipeline.think_generator import needs_polish, polish_thoughts_for_trajectory
from cua.trajectory.io import TrajectoryDir


class PolishPass(CorpusPass):
    """Resumable by construction: polishing flips thought_source to 'teacher', which `needs_polish` then skips."""

    name = "polish"

    def __init__(self, include_untrainable: bool = False):
        self.include_untrainable = include_untrainable

    def select(self, recorded: TrajectoryDir) -> bool:
        # Only trainable dirs by default — polishing unverified rejects buys nothing until a salvage pass promotes them.
        return needs_polish(recorded.path) and (self.include_untrainable or recorded.is_trainable())

    async def apply(self, recorded: TrajectoryDir) -> Outcome:
        polished = await polish_thoughts_for_trajectory(recorded.path, full_name=f"polish_pass:{recorded.path.name}")
        return Outcome(changed=bool(polished), note="polished" if polished else "failed")

    def summary(self, corpus: Path, candidates: list) -> str:
        return f"{len(candidates)} trajectories need polish"


def main():
    parser = build_parser(__doc__, "corpus", "workers", "limit")
    parser.add_argument("--all", action="store_true", help="Include untrainable dirs too")
    parser.add_argument("--dry-run", action="store_true", help="List the targets and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # This pass applies by default, and its --dry-run lists every target rather than the --limit it would have polished.
    counts = asyncio.run(
        PolishPass(args.all).run(
            args.corpus, apply=not args.dry_run, limit=None if args.dry_run else args.limit, workers=args.workers
        )
    )
    if args.dry_run:
        return
    print(f"polished {counts.get('polished', 0)}, failed {counts.get('failed', 0)}")
    sys.exit(1 if counts.get("failed") and not counts.get("polished") else 0)


if __name__ == "__main__":
    main()
