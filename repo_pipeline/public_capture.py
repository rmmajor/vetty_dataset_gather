"""Public-site teacher capture — the real-web diversity lane.

Funnel-sourced trajectories all live on localhost apps; rich bootable repos are ~2% of GitHub.
This lane points the SAME teacher/driver/verifier machinery at a curated registry of stable,
login-free, bot-tolerant public sites (`public_sites.json`) with hand-authored hard tasks —
cookie banners, dense layouts, real search ranking, map canvases — at the same ~$1/trajectory.

Tasks are non-destructive by curation policy (browse/search/navigate/configure-view only).
Output lands as `trajectories/site_<name>/task_*/`, fully inside the existing contract:
verify -> polish inline (these dirs have no registry row, so stage 7 never visits them),
then `make convert` / `make audit-corpus` ingest them like any agent capture.

Run: python -m repo_pipeline.public_capture [--sites wikipedia,mdn] [--max-tasks N]
     (or `make capture-public`)
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

from repo_pipeline import driver_session
from repo_pipeline.agent_capture import run_teacher_task, verify_trajectory
from repo_pipeline.instrumentation import RESULTS_DIR
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox
from repo_pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from repo_pipeline.stage6_generate import _is_valid_trajectory, _safe_task_id
from repo_pipeline.think_generator import needs_polish, polish_thoughts_for_trajectory

logger = logging.getLogger(__name__)

SITES_CONFIG = Path(__file__).parent / "public_sites.json"
PUBLIC_CONTAINER_TIMEOUT = 10800


def load_sites(only: set[str] | None = None) -> list[dict]:
    sites = json.loads(SITES_CONFIG.read_text())["sites"]
    if only:
        sites = [s for s in sites if s["name"] in only]
    return sites


async def _verify_and_finalize(traj_dir: Path, site: dict, task: dict, state: dict, registry):
    verdict = await verify_trajectory(traj_dir, state=state, registry=registry, full_name=f"site:{site['name']}")
    traj = json.loads((traj_dir / "trajectory.json").read_text())
    capture_meta = json.loads((traj_dir / "capture_meta.json").read_text())
    verifier = capture_meta.get("verifier", {})
    metadata = {
        "task_id": traj["task_id"],
        "instruction": traj["instruction"],
        "start_url": traj["start_url"],
        "total_steps": traj["total_steps"],
        "success": traj["success"],
        "website": urlparse(site["start_url"]).netloc,
        "domain": site.get("domain", "public_web"),
        "framework": "agent",
        "capture_source": "agent",
        "instruction_source": "human",
        "thought_source": "teacher_raw",
        "task_type": "other",
        "difficulty": task.get("difficulty", "medium"),
        "efficiency": verifier.get("efficiency"),
        "efficiency_ratio": verifier.get("efficiency_ratio"),
        "relabeled": verifier.get("relabeled", False),
        "viewport": {"width": 1280, "height": 720},
        "source_repo": f"public:{site['name']}",
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if traj["success"] and needs_polish(traj_dir):
        await polish_thoughts_for_trajectory(traj_dir, registry=registry, full_name=f"site:{site['name']}")
    return verdict


async def capture_site(site: dict, trajectory_base_dir: Path, registry, max_tasks: int | None = None) -> dict:
    """One sandbox per site (the driver's browser is in-container; no app boot needed).
    Task-level resume: tasks already recorded on disk are skipped, mirroring stage 6."""
    site_dir = trajectory_base_dir / f"site_{site['name']}"
    tasks = site["tasks"][:max_tasks] if max_tasks else site["tasks"]
    pending = [t for t in tasks if not _is_valid_trajectory(site_dir / f"task_{_safe_task_id(t['id'])}")]
    summary = {"site": site["name"], "captured": 0, "verified": 0, "skipped": len(tasks) - len(pending)}
    if not pending:
        logger.info(f"[{site['name']}] all {len(tasks)} tasks already recorded")
        return summary

    sandbox = DockerSandbox(container_timeout=PUBLIC_CONTAINER_TIMEOUT)
    try:
        await asyncio.get_running_loop().run_in_executor(None, sandbox.start, f"public/{site['name']}")
        await asyncio.get_running_loop().run_in_executor(None, driver_session.install_driver, sandbox)
        await asyncio.get_running_loop().run_in_executor(None, driver_session.start_driver, sandbox)
        client = driver_session.DriverClient(sandbox)

        for task in pending:
            if rate_limit_tracker.limit_reached():
                raise UsageLimitReached("usage limit reached during public capture")
            state = await run_teacher_task(
                client, task, site["start_url"], registry=registry, full_name=f"site:{site['name']}"
            )
            summary["captured"] += 1
            site_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.get_running_loop().run_in_executor(
                None, sandbox.copy_out, RESULTS_DIR, str(site_dir)
            )
            traj_dir = site_dir / f"task_{_safe_task_id(task['id'])}"
            if traj_dir.exists():
                verdict = await _verify_and_finalize(traj_dir, site, task, state, registry)
                if verdict.get("final_success"):
                    summary["verified"] += 1
                logger.info(
                    f"[{site['name']}] task '{task['id']}': {state['actions']} actions, "
                    f"outcome={state['agent_outcome']}, verified={verdict.get('final_success')}"
                    f" ({verdict.get('efficiency')})"
                )
    finally:
        await asyncio.get_running_loop().run_in_executor(None, sandbox.stop)
    return summary


async def main():
    parser = argparse.ArgumentParser(description="Public-site teacher capture lane")
    parser.add_argument("--sites", type=str, default=None, help="Comma-separated site names (default: all)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks per site this run")
    parser.add_argument("--traj-dir", type=str, default="trajectories/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    only = set(args.sites.split(",")) if args.sites else None
    sites = load_sites(only)
    logger.info(f"Public capture over {len(sites)} sites: {', '.join(s['name'] for s in sites)}")

    registry = RepoRegistry("pipeline_data/registry.db")
    results = []
    try:
        for site in sites:
            try:
                results.append(await capture_site(site, Path(args.traj_dir), registry, args.max_tasks))
            except UsageLimitReached:
                logger.warning("usage limit reached — stopping; finished sites keep their captures")
                break
            except Exception as e:
                logger.error(f"[{site['name']}] failed: {e}")
    finally:
        registry.close()

    for r in results:
        logger.info(f"  {r['site']}: {r['captured']} captured, {r['verified']} verified, {r['skipped']} already on disk")


if __name__ == "__main__":
    asyncio.run(main())
