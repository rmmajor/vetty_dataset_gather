import concurrent.futures
import hashlib
import logging
import os

from repo_pipeline.bootable_gate import ProbeError, gate_repo
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.schemas import FailureCategory, RepoStatus

logger = logging.getLogger(__name__)

MAX_SIZE_KB = 500_000

# No-LLM bootable-app gate: reject test-suite / non-app repos before any LLM stage. On by default;
# set BOOTABLE_GATE=0 to fall back to the old metadata-only filter. Probes are network I/O, run in
# a small thread pool. A probe error never rejects (infra) — the repo stays `discovered` to retry.
BOOTABLE_GATE = os.environ.get("BOOTABLE_GATE", "1") != "0"
BOOTABLE_GATE_WORKERS = int(os.environ.get("BOOTABLE_GATE_WORKERS", "8"))


def run_bootable_gate(registry: RepoRegistry, repos: list[dict]) -> tuple[list[dict], int, int]:
    """Probe each repo (no LLM) and split into bootable-apps vs rejected. Returns
    (accepted, n_rejected, n_probe_errors). Probe errors are left at their input status (retryable)."""
    def probe(repo):
        try:
            accept, reason = gate_repo(repo["full_name"])
            return repo, accept, reason
        except ProbeError as e:
            return repo, None, str(e)[:160]

    accepted, rejected, errors = [], 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=BOOTABLE_GATE_WORKERS) as ex:
        for repo, accept, reason in ex.map(probe, repos):
            if accept is True:
                accepted.append(repo)
            elif accept is False:
                registry.reject(repo["full_name"], stage=2, reason=f"not a bootable app: {reason}",
                                category=FailureCategory.NOT_BOOTABLE)
                rejected += 1
            else:
                errors += 1  # probe error: leave at input status, retry next run
    return accepted, rejected, errors


def check_filter(repo: dict) -> tuple[bool, str, FailureCategory | None]:
    if repo.get("size_kb", 0) == 0:
        return False, "empty repo", FailureCategory.EMPTY_REPO

    if repo.get("size_kb", 0) > MAX_SIZE_KB:
        return False, "too large (>500MB, likely monorepo)", FailureCategory.TOO_LARGE

    return True, "", None


def deduplicate_forks(registry: RepoRegistry, repos: list[dict]) -> tuple[list[dict], int]:
    kept = []
    deduped = 0

    for repo in repos:
        if not repo.get("fork"):
            kept.append(repo)
            continue

        parent_name = repo.get("parent_repo")
        if parent_name and registry.has_repo(parent_name):
            registry.reject(repo["full_name"], stage=2, reason=f"fork of {parent_name} (parent already in registry)", category=FailureCategory.DEDUP)
            deduped += 1
            continue

        kept.append(repo)

    return kept, deduped


def deduplicate_clones(registry: RepoRegistry, repos: list[dict]) -> tuple[list[dict], int]:
    description_groups = {}
    for repo in repos:
        desc = (repo.get("description") or "").strip()
        if not desc or len(desc) < 30:
            continue
        desc_hash = hashlib.md5(desc.lower().encode()).hexdigest()
        if desc_hash not in description_groups:
            description_groups[desc_hash] = []
        description_groups[desc_hash].append(repo)

    duplicates_to_reject = set()
    for desc_hash, group in description_groups.items():
        if len(group) < 2:
            continue

        group.sort(key=lambda r: r.get("stars", 0), reverse=True)
        canonical = group[0]
        for dupe in group[1:]:
            if dupe["full_name"] == canonical["full_name"]:
                continue
            duplicates_to_reject.add(dupe["full_name"])

    kept = []
    deduped = 0
    for repo in repos:
        if repo["full_name"] in duplicates_to_reject:
            registry.reject(repo["full_name"], stage=2, reason="duplicate description (likely clone)", category=FailureCategory.DEDUP)
            deduped += 1
        else:
            kept.append(repo)

    return kept, deduped


def run_filter(registry: RepoRegistry, limit: int = None):
    repos = registry.get_repos_for_stage(2, limit=limit)
    logger.info(f"Filtering {len(repos)} discovered repos")

    passed = 0
    rejected = 0

    basic_passed = []
    for repo in repos:
        passes, reason, category = check_filter(repo)
        if passes:
            basic_passed.append(repo)
        else:
            registry.reject(repo["full_name"], stage=2, reason=reason, category=category)
            rejected += 1

    logger.info(f"Basic filter: {len(basic_passed)} passed, {rejected} rejected")

    after_fork_dedup, fork_deduped = deduplicate_forks(registry, basic_passed)
    logger.info(f"Fork dedup: removed {fork_deduped} forks whose parent is already in registry")

    after_clone_dedup, clone_deduped = deduplicate_clones(registry, after_fork_dedup)
    logger.info(f"Clone dedup: removed {clone_deduped} repos with identical descriptions")

    gate_rejected = gate_errors = 0
    if BOOTABLE_GATE:
        survivors, gate_rejected, gate_errors = run_bootable_gate(registry, after_clone_dedup)
        logger.info(f"Bootable-app gate: {len(survivors)} bootable, {gate_rejected} not-bootable "
                    f"rejected (no LLM), {gate_errors} probe errors left for retry")
    else:
        survivors = after_clone_dedup

    for repo in survivors:
        registry.update_status(
            repo["full_name"],
            RepoStatus.FILTERED.value,
            stage=2,
        )
        passed += 1

    rejected += fork_deduped + clone_deduped + gate_rejected
    logger.info(f"Filter complete: {passed} passed, {rejected} rejected total")
    return {"passed": passed, "rejected": rejected, "fork_deduped": fork_deduped,
            "clone_deduped": clone_deduped, "gate_rejected": gate_rejected, "gate_errors": gate_errors}