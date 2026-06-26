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
import os
from pathlib import Path
from urllib.parse import urlparse

from repo_pipeline import driver_session
from repo_pipeline.agent_capture import run_teacher_task, verify_trajectory
from repo_pipeline.instrumentation import RESULTS_DIR
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox, install_sigterm_handler, safe_parallel_sandboxes
from repo_pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from repo_pipeline.stage6_generate import _is_valid_trajectory, _load_proposed_tasks, _safe_task_id
from repo_pipeline.task_proposer import propose_tasks
from repo_pipeline.think_generator import needs_polish, polish_thoughts_for_trajectory

logger = logging.getLogger(__name__)

SITES_CONFIG = Path(__file__).parent / "public_sites.json"
PUBLIC_CONTAINER_TIMEOUT = 10800
PUBLIC_TASKS_PER_SITE = int(os.environ.get("PUBLIC_TASKS_PER_SITE", "6"))


async def _run(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def _norm_instruction(text: str) -> str:
    return " ".join((text or "").lower().split())


def load_sites(only: set[str] | None = None, config_path: Path | None = None) -> list[dict]:
    sites = json.loads((config_path or SITES_CONFIG).read_text())["sites"]
    if only:
        # Explicitly named sites are honored even if skip-tagged (lets you re-probe a dud).
        return [s for s in sites if s["name"] in only]
    return [s for s in sites if not s.get("skip_capture")]


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
        "instruction_source": task.get("instruction_source", "curated"),
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


def _merge_tasks(curated: list[dict], proposed: list[dict], max_tasks: int | None) -> list[dict]:
    """Curated config tasks first (instruction_source=curated: generated offline, reviewed into
    the JSON), then live proposer tasks that don't duplicate a curated one (generated). Slug
    collisions get a `-gen` suffix so each task maps to a distinct task_*/ directory."""
    merged = [{**t, "instruction_source": "curated"} for t in curated]
    seen_norm = {_norm_instruction(t.get("instruction", "")) for t in merged}
    seen_ids = {t["id"] for t in merged}
    for t in proposed:
        if _norm_instruction(t.get("instruction", "")) in seen_norm:
            continue
        tid = t["id"] if t["id"] not in seen_ids else f"{t['id']}-gen"
        seen_ids.add(tid)
        seen_norm.add(_norm_instruction(t.get("instruction", "")))
        merged.append({**t, "id": tid, "instruction_source": "generated"})
    return merged[:max_tasks] if max_tasks else merged


async def _propose_for_site(client, sandbox, site: dict, avoid: list[str], registry) -> dict | None:
    """Ground the proposer in the LIVE public site (homepage snapshot + screenshot + affordances),
    mirroring stage 6's context probe. Returns the proposal dict, or None on SDK/parse failure."""
    probe = await _run(client.session, "context_probe", site["start_url"])
    snapshot = probe.get("snapshot", "")
    screenshot = await _run(client.screenshot_b64)
    try:
        affordances = await _run(client.affordances)
    except Exception:
        affordances = {}
    await _run(client.finish, "probe", "")
    await _run(sandbox.exec_command, f"rm -rf {RESULTS_DIR}/task_context_probe")

    repo = {
        "full_name": f"public:{site['name']}",
        "description": site.get("domain") or site.get("category") or site["name"],
    }
    return await propose_tasks(
        repo, snapshot, screenshot, "", "",
        n_tasks=PUBLIC_TASKS_PER_SITE, registry=registry,
        affordances=affordances, avoid_instructions=avoid,
    )


async def capture_site(site: dict, trajectory_base_dir: Path, registry,
                       max_tasks: int | None = None, propose: bool = True) -> dict:
    """One sandbox per site (the driver's browser is in-container; no app boot needed).

    Lever 1: curated config tasks are deepened with proposer-generated tasks grounded in the live
    site. The proposal is persisted to proposed_tasks.json and reused on resume, so re-runs are
    stable and task-level resume (skip tasks already on disk) is meaningful — mirroring stage 6."""
    site_dir = trajectory_base_dir / f"site_{site['name']}"
    curated = site.get("tasks", [])
    persisted = _load_proposed_tasks(site_dir)

    # With a persisted proposal (or proposing disabled) the full task set is known without the
    # driver, so a fully-recorded site short-circuits before a sandbox is ever started.
    if persisted is not None or not propose:
        full = _merge_tasks(curated, persisted or [], max_tasks)
        if full and all(_is_valid_trajectory(site_dir / f"task_{_safe_task_id(t['id'])}") for t in full):
            logger.info(f"[{site['name']}] all {len(full)} tasks already recorded")
            return {"site": site["name"], "captured": 0, "verified": 0, "skipped": len(full)}

    summary = {"site": site["name"], "captured": 0, "verified": 0, "skipped": 0}
    ran: list[dict] = []
    states: dict = {}
    full: list[dict] = []
    sandbox = DockerSandbox(container_timeout=PUBLIC_CONTAINER_TIMEOUT)
    try:
        await _run(sandbox.start, f"public/{site['name']}")
        await _run(driver_session.install_driver, sandbox)
        await _run(driver_session.start_driver, sandbox)
        client = driver_session.DriverClient(sandbox)

        proposed = persisted
        if proposed is None and propose:
            proposal = await _propose_for_site(client, sandbox, site, [t["instruction"] for t in curated], registry)
            if proposal is not None:
                proposed = proposal.get("tasks") or []
                site_dir.mkdir(parents=True, exist_ok=True)
                (site_dir / "proposed_tasks.json").write_text(json.dumps(proposal, indent=2))
                health = proposal.get("app_health") or {}
                logger.info(f"[{site['name']}] proposed {len(proposed)} extra tasks "
                            f"(health: {health.get('notes', 'ok')[:80]})")
            else:
                logger.warning(f"[{site['name']}] task proposal failed — using curated config tasks only")

        full = _merge_tasks(curated, proposed or [], max_tasks)
        pending = [t for t in full if not _is_valid_trajectory(site_dir / f"task_{_safe_task_id(t['id'])}")]
        summary["skipped"] = len(full) - len(pending)
        if not pending:
            logger.info(f"[{site['name']}] all {len(full)} tasks already recorded")
            return summary

        for task in pending:
            if rate_limit_tracker.limit_reached():
                raise UsageLimitReached("usage limit reached during public capture")
            state = await run_teacher_task(
                client, task, site["start_url"], registry=registry, full_name=f"site:{site['name']}"
            )
            states[task["id"]] = state
            ran.append(task)
            summary["captured"] += 1
            site_dir.mkdir(parents=True, exist_ok=True)
            await _run(sandbox.copy_out, RESULTS_DIR, str(site_dir))
            logger.info(f"[{site['name']}] task '{task['id']}': {state['actions']} actions, "
                        f"outcome={state['agent_outcome']}")
    finally:
        await _run(sandbox.stop)

    # Finalize ONLY after the capture loop: each copy_out re-copies the container's
    # capture_meta/trajectory files over the host copies, so per-task verdicts written
    # mid-loop were clobbered by the next task's copy (cost the first sweep 4 verdicts).
    for task in ran:
        traj_dir = site_dir / f"task_{_safe_task_id(task['id'])}"
        if not traj_dir.exists():
            continue
        verdict = await _verify_and_finalize(traj_dir, site, task, states[task["id"]], registry)
        if verdict.get("final_success"):
            summary["verified"] += 1
        logger.info(
            f"[{site['name']}] task '{task['id']}': verified={verdict.get('final_success')}"
            f" ({verdict.get('efficiency')})"
        )
    return summary


async def main():
    parser = argparse.ArgumentParser(description="Public-site teacher capture lane")
    parser.add_argument("--sites", type=str, default=None, help="Comma-separated site names (default: all)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks per site this run")
    parser.add_argument("--no-propose", action="store_true",
                        help="Disable proposer-generated tasks; run only curated config tasks")
    parser.add_argument("--parallel", type=int, default=3, help="Concurrent site sandboxes")
    parser.add_argument("--traj-dir", type=str, default="trajectories/")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a sites JSON (default: public_sites.json)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    install_sigterm_handler()
    only = set(args.sites.split(",")) if args.sites else None
    sites = load_sites(only, Path(args.config) if args.config else None)
    logger.info(f"Public capture over {len(sites)} sites: {', '.join(s['name'] for s in sites)}")

    registry = RepoRegistry("pipeline_data/registry.db")
    results = []
    # Sites are independent (one sandbox each) — run a few concurrently, but never more than the
    # Docker VM's RAM can hold: each sandbox can claim its mem cap + RAM-backed tmpfs, so an
    # over-subscribed VM thrashes and strangles every sandbox. Auto-clamp to what fits.
    parallel = safe_parallel_sandboxes(int(args.parallel))
    semaphore = asyncio.Semaphore(parallel)
    logger.info(f"Running {parallel} site(s) concurrently (requested {args.parallel})")

    async def run_site(site):
        async with semaphore:
            if rate_limit_tracker.limit_reached():
                return
            try:
                r = await capture_site(
                    site, Path(args.traj_dir), registry, args.max_tasks, propose=not args.no_propose)
                results.append(r)
                # Log per-site as it finishes (not all at the end) so a long run is observable.
                logger.info(f"[{r['site']}] done ({len(results)}/{len(sites)} sites): "
                            f"{r['captured']} captured, {r['verified']} verified, {r['skipped']} already on disk")
            except UsageLimitReached:
                logger.warning(f"[{site['name']}] usage limit reached — captures so far are kept")
            except Exception as e:
                logger.error(f"[{site['name']}] failed: {e}")

    try:
        await asyncio.gather(*[run_site(site) for site in sites])
    finally:
        registry.close()

    logger.info(f"Done: {len(results)}/{len(sites)} sites — "
                f"{sum(r['captured'] for r in results)} captured, "
                f"{sum(r['verified'] for r in results)} verified")


if __name__ == "__main__":
    asyncio.run(main())
