"""Public-site teacher capture — the real-web diversity lane.

Funnel-sourced trajectories all live on localhost apps; rich bootable repos are ~2% of GitHub.
This lane points the SAME teacher/driver/verifier machinery at a curated registry of stable,
login-free, bot-tolerant public sites (`public_sites.json`) with hand-authored hard tasks —
cookie banners, dense layouts, real search ranking, map canvases — at the same ~$1/trajectory.

Tasks are non-destructive by curation policy (browse/search/navigate/configure-view only).
Output lands as `trajectories/site_<name>/task_*/`, fully inside the existing contract:
verify -> polish inline (these dirs have no registry row, so stage 7 never visits them),
then `make convert` / `make audit-corpus` ingest them like any agent capture.

Run: python -m cua.pipeline.lanes.public [--sites wikipedia,mdn] [--max-tasks N]
     (or `make capture-public`)
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from cua.pipeline import app_secrets
from cua.pipeline.driver_session import boot_driver, run_blocking
from cua.pipeline.instrumentation import RESULTS_DIR
from cua.pipeline.lanes.capture_lane import capture_task, finalize_tasks, is_capture_settled
from cua.pipeline.registry import RepoRegistry
from cua.pipeline.sandbox import DockerSandbox, install_sigterm_handler, safe_parallel_sandboxes
from cua.pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from cua.pipeline.sites.registry import PUBLIC_CONFIG, load_proposed_tasks, load_sites
from cua.pipeline.task_proposer import propose_tasks
from cua.settings import CAPTURE, PIPELINE
from cua.trajectory.schema import norm_instruction, safe_task_id

logger = logging.getLogger(__name__)


def _merge_tasks(curated: list[dict], proposed: list[dict], max_tasks: int | None) -> list[dict]:
    """Curated config tasks first, then live proposer tasks that do not duplicate one of them."""
    merged = [{**t, "instruction_source": "curated"} for t in curated]
    seen_norm = {norm_instruction(t.get("instruction", "")) for t in merged}
    seen_ids = {t["id"] for t in merged}
    for t in proposed:
        if norm_instruction(t.get("instruction", "")) in seen_norm:
            continue
        # A slug collision gets a `-gen` suffix so each task still maps to a distinct task_*/ directory.
        tid = t["id"] if t["id"] not in seen_ids else f"{t['id']}-gen"
        seen_ids.add(tid)
        seen_norm.add(norm_instruction(t.get("instruction", "")))
        merged.append({**t, "id": tid, "instruction_source": "generated"})
    return merged[:max_tasks] if max_tasks else merged


async def propose_for_site(
    client, sandbox, site: dict, avoid: list[str], registry, auth_config: dict | None = None
) -> dict | None:
    """Ground the proposer in the LIVE public site, mirroring stage 6's context probe."""
    # The probe carries the site's session so a login-gated site is proposed from its signed-in state, not its login wall.
    probe = await run_blocking(lambda: client.session("context_probe", site["start_url"], auth_config=auth_config))
    snapshot = probe.get("snapshot", "")
    screenshot = await run_blocking(client.screenshot_b64)
    try:
        affordances = await run_blocking(client.affordances)
    except Exception:
        affordances = {}
    await run_blocking(client.finish, "probe", "")
    await run_blocking(sandbox.exec_command, f"rm -rf {RESULTS_DIR}/task_context_probe")

    repo = {
        "full_name": f"public:{site['name']}",
        "description": site.get("domain") or site.get("category") or site["name"],
    }
    return await propose_tasks(
        repo,
        snapshot,
        screenshot,
        "",
        "",
        n_tasks=PIPELINE.public_tasks_per_site,
        registry=registry,
        affordances=affordances,
        avoid_instructions=avoid,
        task_style=site.get("task_style", ""),
    )


def _resolve_site_auth(site: dict):
    """A site's login posture (see sites/schema.md) as (autonomy, auth_mode, secrets, auth_config), or None to skip it."""
    autonomy = site.get("autonomy", "autonomous")
    auth_mode = site.get("auth_mode", "none")
    if autonomy == "autonomous" and auth_mode == "none":
        return autonomy, auth_mode, None, None

    # FAIL-CLOSED: a site needing auth without a vetted burner is skipped before a real login is ever touched.
    burner_id = site.get("burner_account_id")
    if not app_secrets.is_registered_burner(burner_id):
        logger.warning(
            f"[{site['name']}] FAIL-CLOSED: needs auth ({autonomy}/{auth_mode}) but burner "
            f"'{burner_id}' is not a registered burner — skipping site"
        )
        return None
    auth_config = None
    if auth_mode == "pre_session":
        storage_state = app_secrets.load_storage_state(site.get("storage_state_ref"))
        if storage_state is None:
            logger.warning(
                f"[{site['name']}] FAIL-CLOSED: pre_session but storage_state "
                f"'{site.get('storage_state_ref')}' is missing — skipping site"
            )
            return None
        auth_config = {"auth_mode": "pre_session", "storage_state": storage_state}
    # Guarded capture NEVER types the password: the agent hands off at a secret wall and a human supplies it.
    burner = app_secrets.burner_account(burner_id)
    secrets = None if autonomy == "guarded" else {"email": burner["email"], "password": burner["password"]}
    return autonomy, auth_mode, secrets, auth_config


async def _site_tasks(
    client, sandbox, site: dict, site_dir: Path, registry, auth_config, max_tasks, persisted, propose
):
    """This site's task list: curated config tasks deepened with a proposal, persisted so resume is stable."""
    curated = site.get("tasks", [])
    if persisted is not None or not propose:
        return _merge_tasks(curated, persisted or [], max_tasks)
    proposal = await propose_for_site(
        client, sandbox, site, [t["instruction"] for t in curated], registry, auth_config=auth_config
    )
    if proposal is None:
        logger.warning(f"[{site['name']}] task proposal failed — using curated config tasks only")
        return _merge_tasks(curated, [], max_tasks)
    proposed = proposal.get("tasks") or []
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "proposed_tasks.json").write_text(json.dumps(proposal, indent=2))
    health = proposal.get("app_health") or {}
    logger.info(f"[{site['name']}] proposed {len(proposed)} extra tasks (health: {health.get('notes', 'ok')[:80]})")
    return _merge_tasks(curated, proposed, max_tasks)


def _all_recorded(site: dict, site_dir: Path, tasks: list) -> bool:
    """Whether every task already has a settled recording, so this site needs no sandbox at all."""
    if not tasks or not all(is_capture_settled(site_dir / f"task_{safe_task_id(t['id'])}") for t in tasks):
        return False
    logger.info(f"[{site['name']}] all {len(tasks)} tasks already recorded")
    return True


async def capture_site(
    site: dict, trajectory_base_dir: Path, registry, max_tasks: int | None = None, propose: bool = True
) -> dict:
    """One sandbox per site: the driver's browser is in-container, so no app boot is needed."""
    site_dir = trajectory_base_dir / f"site_{site['name']}"
    curated = site.get("tasks", [])

    resolved = _resolve_site_auth(site)
    if resolved is None:
        return {"site": site["name"], "captured": 0, "verified": 0, "skipped": 0, "fail_closed": True}
    autonomy, auth_mode, site_secrets, site_auth_config = resolved

    persisted = load_proposed_tasks(site_dir)
    # With a persisted proposal (or proposing off) the full task set is known without the driver, so a recorded site never starts a sandbox.
    if persisted is not None or not propose:
        full = _merge_tasks(curated, persisted or [], max_tasks)
        if _all_recorded(site, site_dir, full):
            return {"site": site["name"], "captured": 0, "verified": 0, "skipped": len(full)}

    summary = {"site": site["name"], "captured": 0, "verified": 0, "skipped": 0}
    ran: list[dict] = []
    states: dict = {}
    sandbox = DockerSandbox(container_timeout=CAPTURE.public_container_timeout)
    try:
        await run_blocking(sandbox.start, f"public/{site['name']}")
        client = await boot_driver(sandbox)
        full = await _site_tasks(
            client, sandbox, site, site_dir, registry, site_auth_config, max_tasks, persisted, propose
        )
        pending = [t for t in full if not is_capture_settled(site_dir / f"task_{safe_task_id(t['id'])}")]
        summary["skipped"] = len(full) - len(pending)
        if not pending:
            logger.info(f"[{site['name']}] all {len(full)} tasks already recorded")
            return summary

        for task in pending:
            if rate_limit_tracker.limit_reached():
                raise UsageLimitReached("usage limit reached during public capture")
            states[task["id"]] = await capture_task(
                client,
                task,
                site["start_url"],
                sandbox=sandbox,
                site_dir=site_dir,
                label=site["name"],
                registry=registry,
                full_name=f"site:{site['name']}",
                autonomy=autonomy,
                auth_mode=auth_mode,
                secrets=site_secrets,
                auth_config=site_auth_config,
                task_style=site.get("task_style", ""),
            )
            ran.append(task)
            summary["captured"] += 1
    finally:
        await run_blocking(sandbox.stop)

    results = await finalize_tasks(ran, site, site_dir, states, registry)
    summary["verified"] = sum(1 for r in results.values() if r["verdict"].get("final_success"))
    return summary


def _warn_when_nothing_to_capture(sites: list[dict], config: Path) -> None:
    """--no-propose against a registry whose sites carry no curated tasks would capture nothing, silently."""
    taskless = [s["name"] for s in sites if not s.get("tasks")]
    if taskless:
        logger.warning(
            f"--no-propose with {len(taskless)} of {len(sites)} site(s) that have no curated tasks in "
            f"{config}: {', '.join(taskless[:8])}{' ...' if len(taskless) > 8 else ''} — nothing will be captured "
            f"for them (drop --no-propose to let the proposer fill them in)"
        )


async def main():
    parser = argparse.ArgumentParser(description="Public-site teacher capture lane")
    parser.add_argument("--sites", type=str, default=None, help="Comma-separated site names (default: all)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks per site this run")
    parser.add_argument(
        "--no-propose", action="store_true", help="Disable proposer-generated tasks; run only curated config tasks"
    )
    parser.add_argument("--parallel", type=int, default=3, help="Concurrent site sandboxes")
    parser.add_argument("--traj-dir", type=str, default="trajectories/")
    parser.add_argument("--config", type=str, default=None, help="Path to a sites JSON (default: public_sites.json)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    install_sigterm_handler()
    only = [n.strip() for n in args.sites.split(",") if n.strip()] if args.sites else None
    config = Path(args.config) if args.config else PUBLIC_CONFIG
    try:
        sites = load_sites(config, only=only)
    except ValueError as e:
        raise SystemExit(f"--sites: {e}")
    logger.info(f"Public capture over {len(sites)} sites: {', '.join(s['name'] for s in sites)}")
    if args.no_propose:
        _warn_when_nothing_to_capture(sites, config)

    registry = RepoRegistry("pipeline_data/registry.db")
    results = []
    # Never more sandboxes than the Docker VM's RAM can hold: an over-subscribed VM thrashes and strangles every one of them.
    parallel = safe_parallel_sandboxes(int(args.parallel))
    semaphore = asyncio.Semaphore(parallel)
    logger.info(f"Running {parallel} site(s) concurrently (requested {args.parallel})")

    async def run_site(site):
        async with semaphore:
            if rate_limit_tracker.limit_reached():
                return
            try:
                r = await capture_site(site, Path(args.traj_dir), registry, args.max_tasks, propose=not args.no_propose)
                results.append(r)
                # Log per site as it finishes, not all at the end, so a long run is observable.
                logger.info(
                    f"[{r['site']}] done ({len(results)}/{len(sites)} sites): "
                    f"{r['captured']} captured, {r['verified']} verified, {r['skipped']} already on disk"
                )
            except UsageLimitReached:
                logger.warning(f"[{site['name']}] usage limit reached — captures so far are kept")
            except Exception as e:
                logger.error(f"[{site['name']}] failed: {e}")

    try:
        await asyncio.gather(*[run_site(site) for site in sites])
    finally:
        registry.close()

    logger.info(
        f"Done: {len(results)}/{len(sites)} sites — "
        f"{sum(r['captured'] for r in results)} captured, "
        f"{sum(r['verified'] for r in results)} verified"
    )


if __name__ == "__main__":
    asyncio.run(main())
