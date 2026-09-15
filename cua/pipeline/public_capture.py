"""Public-site teacher capture — the real-web diversity lane.

Funnel-sourced trajectories all live on localhost apps; rich bootable repos are ~2% of GitHub.
This lane points the SAME teacher/driver/verifier machinery at a curated registry of stable,
login-free, bot-tolerant public sites (`public_sites.json`) with hand-authored hard tasks —
cookie banners, dense layouts, real search ranking, map canvases — at the same ~$1/trajectory.

Tasks are non-destructive by curation policy (browse/search/navigate/configure-view only).
Output lands as `trajectories/site_<name>/task_*/`, fully inside the existing contract:
verify -> polish inline (these dirs have no registry row, so stage 7 never visits them),
then `make convert` / `make audit-corpus` ingest them like any agent capture.

Run: python -m cua.pipeline.public_capture [--sites wikipedia,mdn] [--max-tasks N]
     (or `make capture-public`)
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

from cua.pipeline import app_secrets, driver_session
from cua.pipeline.agent_capture import run_teacher_task, verify_trajectory
from cua.pipeline.instrumentation import RESULTS_DIR
from cua.pipeline.provision_human import find_site
from cua.pipeline.registry import RepoRegistry
from cua.pipeline.sandbox import DockerSandbox, install_sigterm_handler, safe_parallel_sandboxes
from cua.pipeline.sdk_runner import UsageLimitReached, rate_limit_tracker
from cua.pipeline.stage6_generate import (
    _load_proposed_tasks,
    _safe_task_id,
    is_capture_settled,
    record_capture_attempt,
)
from cua.pipeline.task_proposer import propose_tasks
from cua.pipeline.think_generator import needs_polish, polish_thoughts_for_trajectory
from cua.settings import CAPTURE, PIPELINE
from cua.trajectory import VIEWPORT

logger = logging.getLogger(__name__)

SITES_CONFIG = Path(__file__).parent / "sites" / "public_sites.json"
# Every site registry ships as a JSON here, so a new one is searched by --sites with no code change.
SITE_CONFIG_GLOB = "public_sites*.json"


async def _run(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def _norm_instruction(text: str) -> str:
    return " ".join((text or "").lower().split())


def known_site_configs(preferred: Path | None = None) -> list[Path]:
    """Every site-registry JSON in the package, `preferred` first so an explicit --config wins ties."""
    configs = sorted((Path(__file__).parent / "sites").glob(SITE_CONFIG_GLOB))
    if preferred:
        configs = [preferred] + [c for c in configs if c != preferred]
    return configs


def resolve_named_sites(names: list[str], preferred: Path | None = None) -> list[dict]:
    """Resolve explicitly-named sites across every known config, in the order the caller named them."""
    configs = known_site_configs(preferred)
    resolved, unknown = [], []
    for name in names:
        matches = [(cfg, find_site(name, cfg)) for cfg in configs]
        matches = [(cfg, site) for cfg, site in matches if site]
        if not matches:
            unknown.append(name)
            continue
        winning_cfg, site = matches[0]
        # The same name can appear in several registries with different task sets; say which one won.
        if len(matches) > 1:
            logger.warning(
                f"'{name}' is defined in {len(matches)} configs — using "
                f"{winning_cfg.name} ({len(site.get('tasks', []))} tasks), ignoring "
                f"{', '.join(c.name for c, _ in matches[1:])}"
            )
        resolved.append(site)
    # Hard error, never a warning: silently dropping a site the caller asked for by name reads as success.
    if unknown:
        raise ValueError(
            f"no site named {', '.join(unknown)} in any known config ({', '.join(c.name for c in configs)})"
        )
    return resolved


def load_sites(only: list[str] | set[str] | None = None, config_path: Path | None = None) -> list[dict]:
    if only:
        # Explicitly named sites are honored even if skip-tagged (lets you re-probe a dud).
        return resolve_named_sites(list(only), config_path)
    sites = json.loads((config_path or SITES_CONFIG).read_text())["sites"]
    return [s for s in sites if not s.get("skip_capture")]


async def _verify_and_finalize(traj_dir: Path, site: dict, task: dict, state: dict, registry):
    verdict = await verify_trajectory(traj_dir, state=state, registry=registry, full_name=f"site:{site['name']}")
    traj = json.loads((traj_dir / "trajectory.json").read_text())
    capture_meta = json.loads((traj_dir / "capture_meta.json").read_text())
    verifier = capture_meta.get("verifier", {})
    task_meta = capture_meta.get("task_meta") or {}
    # The trajectory's terminal risk class: the last consequential action the agent paused on.
    risks = [
        a.get("risk_category", "") for s in traj["steps"] for a in s.get("actions", []) if a.get("type") == "confirm"
    ]
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
        "viewport": {"width": VIEWPORT[0], "height": VIEWPORT[1]},
        "source_repo": f"public:{site['name']}",
        "autonomy": task_meta.get("autonomy", "autonomous"),
        "auth_mode": task_meta.get("auth_mode", "none"),
        "task_style": task_meta.get("task_style", ""),
        "risk_category": risks[-1] if risks else "",
        # Replay manifest: what an eval/deploy harness must reconstruct for these actions to be valid.
        "preconditions": {
            "requires_session": task_meta.get("auth_mode") == "pre_session",
            "session_ref": site.get("storage_state_ref", ""),
            "start_url": traj["start_url"],
            "env": task_meta.get("env", "live"),
        },
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    attempts = record_capture_attempt(traj_dir, "success" if traj["success"] else "failed")
    if not traj["success"] and attempts >= CAPTURE.max_attempts:
        logger.warning(
            f"[{site['name']}] task '{task['id']}' has now failed {attempts}x — resume will "
            f"stop re-capturing it (raise CAPTURE_MAX_ATTEMPTS to keep trying)"
        )

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


async def _propose_for_site(
    client, sandbox, site: dict, avoid: list[str], registry, auth_config: dict | None = None
) -> dict | None:
    """Ground the proposer in the LIVE public site (homepage snapshot + screenshot + affordances),
    mirroring stage 6's context probe. Returns the proposal dict, or None on SDK/parse failure."""
    # The probe carries the site's session so a login-gated site is proposed from its signed-in state, not its login wall.
    probe = await _run(lambda: client.session("context_probe", site["start_url"], auth_config=auth_config))
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
    """Resolve a site's login posture, FAIL-CLOSED. Returns
    (autonomy, auth_mode, secrets, auth_config), or None when the site needs auth but has no
    registered burner / a missing storage_state — in which case the caller skips the site entirely
    rather than touch a real login with an unvetted identity.

    Per-site config (all optional; absent ⇒ the existing login-free autonomous behaviour):
      autonomy: "autonomous" | "guarded"   (guarded ⇒ confirm/handoff, human owns secrets)
      auth_mode: "none" | "pre_session"     (pre_session ⇒ start signed in via a saved session)
      burner_account_id: id into credentials/burner_accounts.json (required if auth is needed)
      storage_state_ref: name of credentials/storage_state/<ref>.json (required for pre_session)
    """
    autonomy = site.get("autonomy", "autonomous")
    auth_mode = site.get("auth_mode", "none")
    if autonomy == "autonomous" and auth_mode == "none":
        return autonomy, auth_mode, None, None

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
    # Guarded capture NEVER types the password: the agent hands off at a login/secret wall and a
    # human supplies it. Credentials are passed (masked at type time) ONLY for autonomous runs.
    burner = app_secrets.burner_account(burner_id)
    secrets = None if autonomy == "guarded" else {"email": burner["email"], "password": burner["password"]}
    return autonomy, auth_mode, secrets, auth_config


async def capture_site(
    site: dict, trajectory_base_dir: Path, registry, max_tasks: int | None = None, propose: bool = True
) -> dict:
    """One sandbox per site (the driver's browser is in-container; no app boot needed).

    Lever 1: curated config tasks are deepened with proposer-generated tasks grounded in the live
    site. The proposal is persisted to proposed_tasks.json and reused on resume, so re-runs are
    stable and task-level resume (skip tasks already on disk) is meaningful — mirroring stage 6."""
    site_dir = trajectory_base_dir / f"site_{site['name']}"
    curated = site.get("tasks", [])

    # Resolve login posture before doing any work; a guarded/authenticated site without a vetted
    # burner (or its saved session) is skipped here, before a sandbox or any real-site hit.
    resolved = _resolve_site_auth(site)
    if resolved is None:
        return {"site": site["name"], "captured": 0, "verified": 0, "skipped": 0, "fail_closed": True}
    autonomy, auth_mode, site_secrets, site_auth_config = resolved

    persisted = _load_proposed_tasks(site_dir)

    # With a persisted proposal (or proposing disabled) the full task set is known without the
    # driver, so a fully-recorded site short-circuits before a sandbox is ever started.
    if persisted is not None or not propose:
        full = _merge_tasks(curated, persisted or [], max_tasks)
        if full and all(is_capture_settled(site_dir / f"task_{_safe_task_id(t['id'])}") for t in full):
            logger.info(f"[{site['name']}] all {len(full)} tasks already recorded")
            return {"site": site["name"], "captured": 0, "verified": 0, "skipped": len(full)}

    summary = {"site": site["name"], "captured": 0, "verified": 0, "skipped": 0}
    ran: list[dict] = []
    states: dict = {}
    full: list[dict] = []
    sandbox = DockerSandbox(container_timeout=CAPTURE.public_container_timeout)
    try:
        await _run(sandbox.start, f"public/{site['name']}")
        await _run(driver_session.install_driver, sandbox)
        await _run(driver_session.start_driver, sandbox)
        client = driver_session.DriverClient(sandbox)

        proposed = persisted
        if proposed is None and propose:
            proposal = await _propose_for_site(
                client, sandbox, site, [t["instruction"] for t in curated], registry, auth_config=site_auth_config
            )
            if proposal is not None:
                proposed = proposal.get("tasks") or []
                site_dir.mkdir(parents=True, exist_ok=True)
                (site_dir / "proposed_tasks.json").write_text(json.dumps(proposal, indent=2))
                health = proposal.get("app_health") or {}
                logger.info(
                    f"[{site['name']}] proposed {len(proposed)} extra tasks (health: {health.get('notes', 'ok')[:80]})"
                )
            else:
                logger.warning(f"[{site['name']}] task proposal failed — using curated config tasks only")

        full = _merge_tasks(curated, proposed or [], max_tasks)
        pending = [t for t in full if not is_capture_settled(site_dir / f"task_{_safe_task_id(t['id'])}")]
        summary["skipped"] = len(full) - len(pending)
        if not pending:
            logger.info(f"[{site['name']}] all {len(full)} tasks already recorded")
            return summary

        for task in pending:
            if rate_limit_tracker.limit_reached():
                raise UsageLimitReached("usage limit reached during public capture")
            state = await run_teacher_task(
                client,
                task,
                site["start_url"],
                registry=registry,
                full_name=f"site:{site['name']}",
                autonomy=autonomy,
                auth_mode=auth_mode,
                secrets=site_secrets,
                auth_config=site_auth_config,
                task_style=site.get("task_style", ""),
            )
            states[task["id"]] = state
            ran.append(task)
            summary["captured"] += 1
            site_dir.mkdir(parents=True, exist_ok=True)
            await _run(sandbox.copy_out, RESULTS_DIR, str(site_dir))
            logger.info(
                f"[{site['name']}] task '{task['id']}': {state['actions']} actions, outcome={state['agent_outcome']}"
            )
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
    try:
        sites = load_sites(only, Path(args.config) if args.config else None)
    except ValueError as e:
        raise SystemExit(f"--sites: {e}")
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
                r = await capture_site(site, Path(args.traj_dir), registry, args.max_tasks, propose=not args.no_propose)
                results.append(r)
                # Log per-site as it finishes (not all at the end) so a long run is observable.
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
