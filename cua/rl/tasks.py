"""Build / check / list the RL task suite from the site registries, keeping it disjoint from anything ever trained on."""

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from cua.pipeline import app_secrets
from cua.pipeline.corpus_audit import is_trainable
from cua.pipeline.public_capture import _norm_instruction
from cua.pipeline.stage6_generate import _safe_task_id

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
SITES = Path(__file__).resolve().parents[1] / "pipeline" / "sites"
SUITE_PATH = Path(__file__).parent / "rl_tasks.json"
SEEN_CONFIG = SITES / "public_sites.json"
TASK_SOURCES = [
    {
        "path": SITES / "public_sites_actions_loginfree.json",
        "backend": "public_site",
        "env": "live",
        "lane": "autonomous",
        "auth_mode": "none",
    },
    {
        "path": SITES / "public_sites_actions_logingated.json",
        "backend": "public_site",
        "env": "live",
        "lane": "autonomous",
        "auth_mode": "pre_session",
    },
    {
        "path": SITES / "public_sites_human.json",
        "backend": "public_site",
        "env": "live",
        "lane": "autonomous",
        "auth_mode": "pre_session",
    },
    {
        "path": SITES / "public_sites_saucedemo.json",
        "backend": "public_site",
        "env": "test",
        "lane": "autonomous",
        "auth_mode": "login",
    },
]
_MUTATING = re.compile(
    r"\b(delete|remove|publish|post|send|submit|purchase|buy|pay|order|book|subscribe|register|sign up|"
    r"create an account|upload|share|invite|tweet|comment|reply|unsubscribe|cancel|transfer|donate)\b",
    re.I,
)


def is_idempotent(task):
    if "idempotent" in task:
        return bool(task["idempotent"])
    if task.get("requires_upload"):
        return False
    return not _MUTATING.search(task.get("instruction", ""))


def split_for(site, task_index, idempotent, unseen_site_pct):
    if int(hashlib.sha1(site.encode()).hexdigest()[:8], 16) % 100 < unseen_site_pct:
        return "eval_unseen_site"
    if not idempotent:
        return "eval_mutating"
    if task_index == 0:
        return "eval_seen_site_unseen_task"
    return "train"


def _preconditions(site, auth_mode):
    if auth_mode == "none":
        return {}, True
    burner = site.get("burner_account_id")
    ref = site.get("storage_state_ref")
    ok = bool(
        burner
        and app_secrets.is_registered_burner(burner)
        and (auth_mode != "pre_session" or app_secrets.load_storage_state(ref))
    )
    return {
        "requires_session": auth_mode == "pre_session",
        "session_ref": ref,
        "burner_account_id": burner,
        "env": site.get("env", "live"),
    }, ok


def _trained_keys(trajectories_dir, seen_config):
    dirs, instructions = set(), set()
    if seen_config and Path(seen_config).exists():
        for s in json.loads(Path(seen_config).read_text())["sites"]:
            for t in s["tasks"]:
                dirs.add((s["name"], _safe_task_id(t["id"])))
                instructions.add(_norm_instruction(t["instruction"]))
    traj_root = Path(trajectories_dir) if trajectories_dir else None
    for traj in traj_root.glob("**/task_*/trajectory.json") if traj_root and traj_root.exists() else []:
        # Only what SFT actually trained on is contaminated; failed captures never entered the corpus.
        if not is_trainable(traj.parent):
            continue
        site_dir = traj.parent.parent.name
        dirs.add(
            (site_dir[len("site_") :] if site_dir.startswith("site_") else site_dir, traj.parent.name[len("task_") :])
        )
        try:
            instructions.add(_norm_instruction(json.loads(traj.read_text()).get("instruction", "")))
        except (json.JSONDecodeError, OSError):
            continue
    return dirs, instructions


def build_suite(sources, trajectories_dir, seen_config, unseen_site_pct=15):
    trained_dirs, trained_instructions = _trained_keys(trajectories_dir, seen_config)
    tasks, built_from = [], []
    excluded_trained = 0
    for src in sources:
        path = Path(src["path"])
        if not path.exists():
            logger.warning(f"site registry {path} is missing — no tasks contributed from it")
            continue
        sites = json.loads(path.read_text())["sites"]
        built_from.append(
            {"path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path), "sites": len(sites)}
        )
        for site in sites:
            if site.get("skip_capture"):
                continue
            auth_mode = site.get("auth_mode", src["auth_mode"])
            preconditions, available = _preconditions(site, auth_mode)
            if not available:
                continue
            for index, t in enumerate(sorted(site["tasks"], key=lambda t: t["id"])):
                if (site["name"], _safe_task_id(t["id"])) in trained_dirs or _norm_instruction(
                    t["instruction"]
                ) in trained_instructions:
                    excluded_trained += 1
                    continue
                idempotent = is_idempotent(t)
                tasks.append(
                    {
                        "task_id": f"{site['name']}__{t['id']}",
                        "site": site["name"],
                        "start_url": site["start_url"],
                        "instruction": t["instruction"],
                        "difficulty": t.get("difficulty", "medium"),
                        "expected_actions": t.get("expected_actions", 8),
                        "action_kinds": t.get("action_kinds", []),
                        "backend": src["backend"],
                        "env": src["env"],
                        "lane": site.get("autonomy", src["lane"]),
                        "auth_mode": auth_mode,
                        "idempotent": idempotent,
                        "split": split_for(site["name"], index, idempotent, unseen_site_pct),
                        "preconditions": preconditions,
                        "allowed_hosts": [urlparse(site["start_url"]).netloc],
                        "success_check": t.get("success_check"),
                        "milestones": t.get("milestones"),
                        "throttle": {"min_interval_s": site.get("min_interval_s", 30)},
                        "source": path.name,
                    }
                )
    tasks.sort(key=lambda t: t["task_id"])
    digest = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()
    return {"suite_sha256": digest, "built_from": built_from, "excluded_trained": excluded_trained, "tasks": tasks}


def check_suite(suite, trajectories_dir, seen_config):
    problems = []
    trained_dirs, trained_instructions = _trained_keys(trajectories_dir, seen_config)
    ids, instructions = set(), {}
    for t in suite["tasks"]:
        if t["task_id"] in ids:
            problems.append(f"duplicate task_id {t['task_id']}")
        ids.add(t["task_id"])
        norm = _norm_instruction(t["instruction"])
        if norm in instructions and instructions[norm] != t["split"]:
            problems.append(f"instruction shared across splits: {t['task_id']} / {instructions[norm]}")
        instructions[norm] = t["split"]
        if (t["site"], _safe_task_id(t["task_id"].split("__", 1)[1])) in trained_dirs or norm in trained_instructions:
            problems.append(f"trained-on task in suite: {t['task_id']}")
        if t["split"] == "train" and not t["idempotent"]:
            problems.append(f"non-idempotent task in train: {t['task_id']}")
    return problems


FRONTIER_PATH = SITES / "public_sites_actions_frontier.json"
CANDIDATES_PATH = Path(__file__).parent / "rl_tasks_candidates.json"


def build_candidates(frontier_path, base_suite_path, trajectories_dir, seen_config, max_expected=12):
    """Sweep pool for frontier selection: the frontier registry plus the base suite's short train tasks, all forced into train."""
    registry = json.loads(Path(frontier_path).read_text())
    config = registry.get("_config", {})
    exclude_sites = set(config.get("exclude_sites", []))
    exclude_domains = tuple(config.get("exclude_domains", []))
    source = {"path": frontier_path, "backend": "public_site", "env": "live", "lane": "autonomous", "auth_mode": "none"}
    built = build_suite([source], trajectories_dir, seen_config, unseen_site_pct=0)
    tasks = {t["task_id"]: t for t in built["tasks"]}
    dropped_unoracled = []
    for t in load_suite(base_suite_path)["tasks"]:
        if t["split"] == "train" and (t.get("expected_actions") or 99) <= max_expected:
            # The sweep scores oracle-only, so a base task without a success_check cannot be a candidate.
            if not t.get("success_check"):
                dropped_unoracled.append(t["task_id"])
                continue
            tasks.setdefault(t["task_id"], dict(t))

    def blocked(t):
        host = urlparse(t["start_url"]).netloc
        return t["site"] in exclude_sites or any(host == d or host.endswith("." + d) for d in exclude_domains)

    picked = sorted(
        (dict(t, split="train") for t in tasks.values() if t["idempotent"] and not blocked(t)),
        key=lambda t: t["task_id"],
    )
    digest = hashlib.sha256(json.dumps(picked, sort_keys=True).encode()).hexdigest()
    built_from = [
        {"path": str(frontier_path), "sites": len(registry.get("sites", []))},
        {"path": str(base_suite_path), "sites": None},
    ]
    return {
        "suite_sha256": digest,
        "built_from": built_from,
        "excluded_trained": built["excluded_trained"],
        "dropped_unoracled": dropped_unoracled,
        "tasks": picked,
    }


def load_suite(path=SUITE_PATH):
    return json.loads(Path(path).read_text())


def select_tasks(suite, split=None, task_ids=None):
    tasks = suite["tasks"]
    if task_ids:
        wanted = set(task_ids)
        return [t for t in tasks if t["task_id"] in wanted]
    if split and split != "all":
        return [t for t in tasks if t["split"] == split or (split == "eval" and t["split"].startswith("eval"))]
    return list(tasks)


def main():
    parser = argparse.ArgumentParser(description="RL task suite: build | build-candidates | check | list")
    parser.add_argument("command", choices=["build", "build-candidates", "check", "list"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--trajectories", default=str(REPO / "trajectories"))
    parser.add_argument("--seen-config", default=str(SEEN_CONFIG))
    parser.add_argument("--unseen-site-pct", type=int, default=15)
    parser.add_argument("--split", default=None)
    parser.add_argument("--frontier", default=str(FRONTIER_PATH))
    parser.add_argument("--base-suite", default=str(SUITE_PATH))
    parser.add_argument("--max-expected", type=int, default=12)
    args = parser.parse_args()
    args.out = args.out or (str(CANDIDATES_PATH) if args.command == "build-candidates" else str(SUITE_PATH))
    if args.command == "build-candidates":
        suite = build_candidates(args.frontier, args.base_suite, args.trajectories, args.seen_config, args.max_expected)
        missing = [t["task_id"] for t in suite["tasks"] if not t.get("success_check")]
        problems = check_suite(suite, args.trajectories, args.seen_config)
        Path(args.out).write_text(json.dumps(suite, indent=1))
        print(
            f"built {len(suite['tasks'])} candidates ({suite['excluded_trained']} excluded as trained-on, "
            f"{len(suite['dropped_unoracled'])} base tasks dropped without an oracle); sha {suite['suite_sha256'][:12]} -> {args.out}"
        )
        for p in problems:
            print(f"  - {p}")
        if missing:
            print(f"  - {len(missing)} candidates lack a success_check: {', '.join(missing)}")
        return 1 if problems or missing else 0
    if args.command == "build":
        suite = build_suite(TASK_SOURCES, args.trajectories, args.seen_config, args.unseen_site_pct)
        Path(args.out).write_text(json.dumps(suite, indent=1))
        sources = ", ".join(f"{Path(b['path']).name}:{b['sites']}" for b in suite["built_from"])
        print(
            f"built {len(suite['tasks'])} tasks ({suite['excluded_trained']} excluded as trained-on) from {sources}; "
            f"sha {suite['suite_sha256'][:12]} -> {args.out}"
        )
        return 0
    suite = load_suite(args.out)
    if args.command == "check":
        problems = check_suite(suite, args.trajectories, args.seen_config)
        counts = {}
        for t in suite["tasks"]:
            counts[t["split"]] = counts.get(t["split"], 0) + 1
        print(f"suite sha256 {suite['suite_sha256']} splits {counts} problems {len(problems)}")
        for p in problems:
            print(f"  - {p}")
        return 1 if problems else 0
    for t in select_tasks(suite, split=args.split):
        print(f"{t['task_id']:50} {t['split']:28} {t['lane']:10} {t['auth_mode']:12} {t['instruction'][:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
