"""Every packaged site registry loads through the one loader, and schema.md documents every key they use."""

import json
import re

from cua.pipeline.sites.registry import SITES_DIR, find_site, known_site_configs, load_sites

SCHEMA = SITES_DIR / "schema.md"
# schema.md documents each key as the first backticked cell of a table row; `_excluded_<name>` stands for a family.
_TABLE_KEY = re.compile(r"^\| `([^`]+)`", re.MULTILINE)
DOCUMENTED_PREFIXES = ("_excluded_",)
TRACKED_ONLY = {"public_sites_human.json"}


def _documented_keys() -> set:
    return set(_TABLE_KEY.findall(SCHEMA.read_text()))


def _registries() -> list:
    return [c for c in known_site_configs() if c.name not in TRACKED_ONLY]


def test_every_packaged_registry_loads_through_load_sites():
    configs = _registries()
    assert len(configs) >= 8, [c.name for c in configs]
    for config in configs:
        sites = load_sites(config, include_skipped=True)
        assert sites, config.name
        for site in sites:
            assert site.get("name"), (config.name, site)
            assert site.get("start_url", "").startswith("http"), (config.name, site["name"])
            assert isinstance(site.get("tasks", []), list), (config.name, site["name"])


def test_load_sites_filters_are_a_subset_of_the_unfiltered_load():
    for config in _registries():
        every = [s["name"] for s in load_sites(config, include_skipped=True)]
        capturable = [s["name"] for s in load_sites(config)]
        assert set(capturable) <= set(every), config.name
        skipped = {s["name"] for s in load_sites(config, include_skipped=True) if s.get("skip_capture")}
        assert set(every) - set(capturable) == skipped, config.name


def test_schema_md_documents_every_key_the_registries_use():
    documented = _documented_keys()
    top_keys, site_keys, task_keys = set(), set(), set()
    for config in _registries():
        data = json.loads(config.read_text())
        top_keys.update(data.keys())
        for site in data.get("sites", []):
            site_keys.update(site.keys())
            for task in site.get("tasks", []):
                task_keys.update(task.keys())
    undocumented = {
        key
        for key in top_keys | site_keys | task_keys
        if key not in documented and not key.startswith(DOCUMENTED_PREFIXES)
    }
    assert not undocumented, f"add to {SCHEMA.name}: {sorted(undocumented)}"
    assert {"name", "start_url", "tasks"} <= site_keys
    assert {"id", "instruction", "difficulty"} <= task_keys


def test_find_site_resolves_a_site_by_name_and_by_start_url():
    config = SITES_DIR / "public_sites.json"
    site = load_sites(config)[0]
    assert find_site(site["name"], config) == site
    assert find_site(site["start_url"], config) == site
    assert find_site(site["start_url"].rstrip("/") + "/", config) == site
    assert find_site("no-such-site-anywhere", config) is None
