"""The site registries as one loader: every capture lane resolves its sites here, from the JSON data `schema.md` documents."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SITES_DIR = Path(__file__).parent
# Every site registry ships as a JSON here, so a new one is searched by --sites with no code change.
SITE_CONFIG_GLOB = "public_sites*.json"
PUBLIC_CONFIG = SITES_DIR / "public_sites.json"
LOGINGATED_CONFIG = SITES_DIR / "public_sites_actions_logingated.json"
HUMAN_CONFIG = SITES_DIR / "public_sites_human.json"
# The two worklists a human provisioner draws from; `find_site` searches both when given no config.
SOURCE_CONFIGS = [SITES_DIR / "public_sites_actions_indie.json", LOGINGATED_CONFIG]


def known_site_configs(preferred: Path | None = None) -> list[Path]:
    """Every site-registry JSON in the package, `preferred` first so an explicit --config wins ties."""
    configs = sorted(SITES_DIR.glob(SITE_CONFIG_GLOB))
    if preferred:
        configs = [preferred] + [c for c in configs if c != preferred]
    return configs


def find_site(key: str, config_path: Path | None = None) -> dict | None:
    """The site whose `name` or `start_url` matches `key`, in `config_path` if given else both source configs."""
    # The URL matches too so the operator can paste either column of the worklist straight in.
    key_url = key.rstrip("/")
    for cfg in [config_path] if config_path else SOURCE_CONFIGS:
        if not cfg or not cfg.exists():
            continue
        for site in json.loads(cfg.read_text()).get("sites", []):
            if site.get("name") == key or site.get("start_url", "").rstrip("/") == key_url:
                return site
    return None


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


def load_sites(
    path: Path,
    *,
    only: list[str] | set[str] | None = None,
    allow_provisioning: set[str] | None = None,
    include_skipped: bool = False,
) -> list[dict]:
    """The sites a lane should capture from `path`, filtered by whichever of the two gates that lane uses."""
    if only:
        # Explicitly named sites run even when skip-tagged, and are searched across EVERY config: the worklist is split over several.
        return resolve_named_sites(list(only), path)
    sites = json.loads(Path(path).read_text())["sites"]
    if allow_provisioning is not None:
        sites = [s for s in sites if s.get("provisioning") in allow_provisioning]
    if not include_skipped:
        sites = [s for s in sites if not s.get("skip_capture")]
    return sites


def load_proposed_tasks(traj_dir: Path) -> list[dict] | None:
    """The proposer's persisted task list for one capture dir, or None when it has never been proposed."""
    proposals_file = traj_dir / "proposed_tasks.json"
    if not proposals_file.exists():
        return None
    try:
        return json.loads(proposals_file.read_text()).get("tasks") or None
    except (json.JSONDecodeError, OSError):
        return None
