"""Stage-7 thought polish over trajectory dirs the registry flow cannot reach (site_* lanes).

Resumable by construction: `polish_thoughts_for_trajectory` flips metadata.thought_source to
'teacher' on success, and `needs_polish` skips those on the next run.

  python -m cua.pipeline.polish_pass trajectories/ [--limit N] [--workers 3] [--dry-run]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from cua.pipeline.corpus_audit import is_trainable
from cua.pipeline.sdk_runner import UsageLimitReached
from cua.pipeline.think_generator import needs_polish, polish_thoughts_for_trajectory
from cua.trajectory.io import iter_trajectory_dirs

logger = logging.getLogger(__name__)


def find_polish_targets(corpus_dir: Path, include_untrainable: bool = False) -> list[Path]:
    # Only trainable dirs by default — polishing unverified rejects buys nothing until a salvage pass promotes them.
    return [d for d in iter_trajectory_dirs(corpus_dir) if needs_polish(d) and (include_untrainable or is_trainable(d))]


async def run_pass(
    corpus_dir: Path, workers: int, limit: int | None, include_untrainable: bool = False
) -> tuple[int, int]:
    targets = find_polish_targets(corpus_dir, include_untrainable)
    if limit:
        targets = targets[:limit]
    logger.info(f"{len(targets)} trajectories need polish under {corpus_dir}")
    semaphore = asyncio.Semaphore(workers)
    done = failed = 0

    async def polish(traj_dir: Path):
        nonlocal done, failed
        async with semaphore:
            ok = await polish_thoughts_for_trajectory(traj_dir, full_name=f"polish_pass:{traj_dir.name}")
        done += ok
        failed += not ok

    try:
        await asyncio.gather(*(polish(d) for d in targets))
    except UsageLimitReached:
        logger.warning("usage limit reached — remaining trajectories stay teacher_raw; re-run to resume")
    return done, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("trajectories/"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Include untrainable dirs too")
    parser.add_argument("--dry-run", action="store_true", help="List the targets and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.dry_run:
        targets = find_polish_targets(args.corpus, args.all)
        for d in targets:
            print(d.relative_to(args.corpus))
        print(f"{len(targets)} trajectories need polish")
        return

    done, failed = asyncio.run(run_pass(args.corpus, args.workers, args.limit, args.all))
    print(f"polished {done}, failed {failed}")
    sys.exit(1 if failed and not done else 0)


if __name__ == "__main__":
    main()
