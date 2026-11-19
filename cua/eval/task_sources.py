"""Eval task-source registry — adding an eval task set is a DATA entry, not a code change."""

import json
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from cua.settings import EVAL, env_values
from cua.trajectory.schema import safe_task_id

logger = logging.getLogger(__name__)

PUBLIC_SITES_CONFIG = Path(__file__).resolve().parents[1] / "pipeline" / "sites" / "public_sites.json"
HF_ENDPOINT = EVAL.hf_endpoint
HF_TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"
# HF libraries read only the first two, so this repo's own HF_READ_TOKEN needs explicit handling.
HF_TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_READ_TOKEN")


class TaskSourceUnavailable(Exception):
    """The task set could not be loaded; the message names the exact operator action required."""


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    instruction: str
    start_url: str
    website: str = ""
    domain: str = ""
    reference_length: int = 0  # the benchmark's own human step count; the efficiency denominator
    level: str = ""  # easy | medium | hard, when the source grades difficulty
    split_tag: str = ""
    source: str = ""


@dataclass(frozen=True)
class TaskLoadConfig:
    limit: int | None = None
    revision: str = "main"
    cache_dir: Path = Path("eval_runs/.task_cache")
    trajectories_dir: Path = Path("trajectories")
    sites_config: Path = PUBLIC_SITES_CONFIG
    tasks_file: Path | None = None
    stratify: bool = False  # with --limit, keep the level mix instead of taking the head


@dataclass(frozen=True)
class TaskSource:
    name: str
    description: str
    loader: Callable  # (TaskSource, TaskLoadConfig) -> (list[EvalTask], provenance)
    default_step_cap: int
    live_web: bool = False  # third-party sites: expect bot-blocking as a real outcome
    hf_dataset: str = ""
    hf_file: str = ""
    hf_split: str = ""
    columns: dict = field(default_factory=dict)  # EvalTask field -> dataset column
    access_url: str = ""
    citation: str = ""


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    return url if url.startswith(("http://", "https://")) else f"https://{url}"


def _env_file_candidates() -> list[Path]:
    """`.env` next to the invocation and next to the main checkout (a worktree has none of its own)."""
    candidates = [Path(EVAL.env_file)] if EVAL.env_file else []
    candidates.append(Path.cwd() / ".env")
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True, check=True
        ).stdout.strip()
        candidates.append(Path(common).resolve().parent / ".env")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return candidates


def _read_env_file(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_hf_token() -> str | None:
    """Explicit env var, then a `.env` file, then the token `huggingface-cli login` cached."""
    tokens = env_values(EVAL)
    for var in HF_TOKEN_VARS:
        token = tokens.get(var)
        if token:
            return token.strip()
    for env_file in _env_file_candidates():
        if not env_file.exists():
            continue
        values = _read_env_file(env_file)
        for var in HF_TOKEN_VARS:
            if values.get(var):
                logger.info(f"using {var} from {env_file}")
                return values[var]
    if HF_TOKEN_FILE.exists():
        token = HF_TOKEN_FILE.read_text().strip()
        if token:
            return token
    return None


def _hf_get(url: str, token: str | None, stream: bool = False):
    # requests is an optional extra, and only the gated-dataset source reaches the hub.
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return requests.get(url, headers=headers, timeout=120, allow_redirects=True, stream=stream)


def _hf_resolve_revision(source: TaskSource, token: str | None, revision: str) -> str:
    """The dataset's commit sha for `revision`, recorded so a before/after pair provably shares a task set."""
    response = _hf_get(f"{HF_ENDPOINT}/api/datasets/{source.hf_dataset}/revision/{revision}", token)
    if response.status_code != 200:
        return revision
    return (response.json() or {}).get("sha") or revision


def _hf_download(source: TaskSource, token: str | None, revision: str, sha: str, cache_dir: Path) -> Path:
    """Fetch the dataset's single data file, cached by resolved revision; gating failures are fatal and actionable."""
    if not token:
        raise TaskSourceUnavailable(
            f"{source.name} needs a HuggingFace token: the dataset {source.hf_dataset} is "
            f"gated (gated:auto), and none of {', '.join(HF_TOKEN_VARS)} was found in the "
            f"environment, in {', '.join(str(p) for p in _env_file_candidates())}, or in "
            f"{HF_TOKEN_FILE}.\n"
            f"  1. Accept the terms while logged in at {source.access_url} (auto-approved)\n"
            f"  2. Export HF_TOKEN=<your token>, put HF_READ_TOKEN=<token> in .env, or run "
            f"`huggingface-cli login`"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{source.name}_{sha[:12]}_{source.hf_file}"
    if cached.exists():
        logger.info(f"{source.name}: using cached {cached}")
        return cached

    url = f"{HF_ENDPOINT}/datasets/{source.hf_dataset}/resolve/{revision}/{source.hf_file}"
    response = _hf_get(url, token)
    if response.status_code in (401, 403):
        raise TaskSourceUnavailable(
            f"HuggingFace refused {source.hf_dataset} with HTTP {response.status_code}: "
            f"{response.text.strip()[:200]}\n"
            f"The dataset is gated:auto — the account behind this token has not been granted "
            f"access yet.\n"
            f"  1. Visit {source.access_url} while logged in as that account and accept the terms "
            f"(auto-approved).\n"
            f"  2. Re-run. No stub or sample is substituted: a number from fake tasks would be worse "
            f"than no number."
        )
    if response.status_code != 200:
        raise TaskSourceUnavailable(
            f"Downloading {url} failed with HTTP {response.status_code}: {response.text.strip()[:200]}"
        )
    cached.write_bytes(response.content)
    logger.info(f"{source.name}: downloaded {len(response.content)} bytes at revision {sha[:12]}")
    return cached


def _rows_from_json(path: Path) -> list[dict]:
    """The HF file is either a JSON array or JSON Lines; both are shipped by HF datasets."""
    text = path.read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_hf_json_tasks(source: TaskSource, cfg: TaskLoadConfig) -> tuple[list[EvalTask], dict]:
    """Load a single-JSON-file HF dataset through `source.columns`, so a renamed column is a registry edit."""
    token = resolve_hf_token()
    sha = _hf_resolve_revision(source, token, cfg.revision) if token else cfg.revision
    path = _hf_download(source, token, cfg.revision, sha, cfg.cache_dir)
    rows = _rows_from_json(path)
    columns = source.columns

    missing = [
        c
        for c in (columns.get("task_id"), columns.get("instruction"), columns.get("start_url"))
        if c and rows and c not in rows[0]
    ]
    if missing:
        raise TaskSourceUnavailable(
            f"{source.name}: dataset columns {missing} are absent from the downloaded rows "
            f"(present: {sorted(rows[0])}). Update TASK_SOURCES['{source.name}'].columns."
        )

    tasks = []
    for row in rows:
        start_url = _normalize_url(str(row.get(columns["start_url"], "")))
        if not start_url:
            continue
        tasks.append(
            EvalTask(
                task_id=str(row[columns["task_id"]]),
                instruction=str(row[columns["instruction"]]),
                start_url=start_url,
                website=urlparse(start_url).netloc,
                domain=urlparse(start_url).netloc,
                reference_length=int(row.get(columns.get("reference_length", ""), 0) or 0),
                level=str(row.get(columns.get("level", ""), "") or ""),
                split_tag="live_web",
                source=source.name,
            )
        )

    provenance = {
        "dataset": source.hf_dataset,
        "split": source.hf_split,
        "revision": sha,
        "file": source.hf_file,
        "rows_available": len(rows),
        "distinct_sites": len({t.website for t in tasks}),
        "level_distribution": level_distribution(tasks),
        "reference_length": {
            "mean": round(sum(t.reference_length for t in tasks) / len(tasks), 1) if tasks else 0,
            "max": max((t.reference_length for t in tasks), default=0),
        },
        "citation": source.citation,
    }
    return apply_limit(tasks, cfg), provenance


def level_distribution(tasks: list[EvalTask]) -> dict:
    counts: dict = {}
    for task in tasks:
        counts[task.level or "ungraded"] = counts.get(task.level or "ungraded", 0) + 1
    return dict(sorted(counts.items()))


def apply_limit(tasks: list[EvalTask], cfg: TaskLoadConfig) -> list[EvalTask]:
    """Deterministic subsetting: the head by default, or a level-stratified sample that keeps the full set's mix."""
    if not cfg.limit or cfg.limit >= len(tasks):
        return tasks
    if not cfg.stratify:
        return tasks[: cfg.limit]
    by_level: dict = {}
    for task in sorted(tasks, key=lambda t: t.task_id):
        by_level.setdefault(task.level or "ungraded", []).append(task)
    picked: list[EvalTask] = []
    order = sorted(by_level)
    index = 0
    while len(picked) < cfg.limit:
        bucket = by_level[order[index % len(order)]]
        depth = index // len(order)
        if depth < len(bucket):
            picked.append(bucket[depth])
        elif all(depth >= len(by_level[level]) for level in order):
            break
        index += 1
    return picked


def _site_start_urls(sites_config: Path) -> dict[str, dict]:
    sites = json.loads(sites_config.read_text())["sites"]
    return {site["name"]: site for site in sites}


def load_held_out_tasks(source: TaskSource, cfg: TaskLoadConfig) -> tuple[list[EvalTask], dict]:
    """Our own tasks with no recording on disk, therefore never trained on; repo-app groups are skipped and counted."""
    sites = _site_start_urls(cfg.sites_config)
    traj_dir = Path(cfg.trajectories_dir)
    tasks: list[EvalTask] = []
    seen: set[str] = set()
    captured = 0
    skipped_repo_groups = 0

    candidates: list[tuple[dict, dict, str]] = []
    for _name, site in sites.items():
        for task in site.get("tasks", []):
            candidates.append((site, task, "curated"))

    if traj_dir.exists():
        for proposal_file in sorted(traj_dir.glob("*/proposed_tasks.json")):
            group = proposal_file.parent.name
            if not group.startswith("site_"):
                skipped_repo_groups += 1
                continue
            site = sites.get(group[len("site_") :])
            if site is None:
                continue
            try:
                proposed = json.loads(proposal_file.read_text()).get("tasks") or []
            except (json.JSONDecodeError, OSError):
                continue
            for task in proposed:
                candidates.append((site, task, "generated"))

    for site, task, origin in candidates:
        task_id = str(task.get("id") or "")
        instruction = (task.get("instruction") or "").strip()
        if not task_id or not instruction:
            continue
        unique_id = f"{site['name']}__{task_id}"
        if unique_id in seen:
            continue
        seen.add(unique_id)
        if (traj_dir / f"site_{site['name']}" / f"task_{safe_task_id(task_id)}").exists():
            captured += 1
            continue
        start_url = _normalize_url(site["start_url"])
        tasks.append(
            EvalTask(
                task_id=unique_id,
                instruction=instruction,
                start_url=start_url,
                website=urlparse(start_url).netloc,
                domain=site.get("domain", "public_web"),
                reference_length=int(task.get("expected_actions") or 0),
                level=str(task.get("difficulty") or ""),
                split_tag=f"unseen_task_seen_site:{origin}",
                source=source.name,
            )
        )

    if not tasks:
        raise TaskSourceUnavailable(
            f"{source.name}: every candidate task already has a recording under "
            f"{traj_dir}/site_*/ ({captured} captured), so nothing is held out. Add tasks to "
            f"{cfg.sites_config} or point --trajectories-dir at a different corpus."
        )

    provenance = {
        "sites_config": str(cfg.sites_config),
        "corpus_checked": str(traj_dir),
        "already_captured_excluded": captured,
        "repo_app_groups_skipped": skipped_repo_groups,
        "level_distribution": level_distribution(tasks),
    }
    return apply_limit(tasks, cfg), provenance


TASK_SOURCES: dict[str, TaskSource] = {
    "online_mind2web": TaskSource(
        name="online_mind2web",
        description="300 tasks across live third-party websites (Online-Mind2Web, COLM 2025)",
        loader=load_hf_json_tasks,
        default_step_cap=30,
        live_web=True,
        hf_dataset="osunlp/Online-Mind2Web",
        hf_file="Online_Mind2Web.json",
        hf_split="test",
        # Verified against the live gated rows: the card says `task_description`, the data ships `confirmed_task`.
        columns={
            "task_id": "task_id",
            "instruction": "confirmed_task",
            "start_url": "website",
            "reference_length": "reference_length",
            "level": "level",
        },
        access_url="https://huggingface.co/datasets/osunlp/Online-Mind2Web",
        citation="Xue et al., An Illusion of Progress? (arXiv:2504.01382), CC-BY-4.0",
    ),
    "held_out": TaskSource(
        name="held_out",
        description="our own never-captured public-site tasks (seen site, unseen task)",
        loader=load_held_out_tasks,
        default_step_cap=25,
        live_web=True,
        access_url="cua/pipeline/sites/public_sites.json",
    ),
}


def load_tasks(source_name: str, cfg: TaskLoadConfig) -> tuple[TaskSource, list[EvalTask], dict]:
    if source_name not in TASK_SOURCES:
        raise TaskSourceUnavailable(f"Unknown task source {source_name!r}. Available: {sorted(TASK_SOURCES)}")
    source = TASK_SOURCES[source_name]
    tasks, provenance = source.loader(source, cfg)
    return source, tasks, provenance


def load_tasks_from_file(path: Path, limit: int | None = None) -> list[EvalTask]:
    """Escape hatch for an ad-hoc task list (JSON array or JSON Lines of EvalTask fields)."""
    rows = _rows_from_json(Path(path))
    tasks = []
    for row in rows:
        start_url = _normalize_url(row["start_url"])
        tasks.append(
            EvalTask(
                task_id=str(row["task_id"]),
                instruction=row["instruction"],
                start_url=start_url,
                website=row.get("website") or urlparse(start_url).netloc,
                domain=row.get("domain") or urlparse(start_url).netloc,
                reference_length=int(row.get("reference_length") or 0),
                level=row.get("level", ""),
                split_tag=row.get("split_tag", "file"),
                source=row.get("source", str(path)),
            )
        )
        if limit and len(tasks) >= limit:
            break
    return tasks


def episode_dir_name(task: EvalTask) -> str:
    return f"task_{safe_task_id(task.task_id)}"
