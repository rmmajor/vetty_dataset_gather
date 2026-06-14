"""No-LLM bootable-app gate: decide whether a repo is a bootable interactive web app from a single
GitHub probe (root tree + package.json), before any LLM stage runs. Stage 1/2 admit candidates; this
rejects the test-suite / non-app majority cheaply so shallow_score (LLM) and deep_analyze (LLM +
sandbox) only ever run on plausible apps.

`classify_bootable` is a pure function (no I/O) so it is testable in isolation; `probe_repo` does the
one network call via the `gh` CLI (reusing its auth, like stage 3's metadata fetch).

All signal sets are module-level data — adding a framework / config file / compose name is a data
edit, not a logic change (the project's "generalize, don't special-case" rule).
"""
import json
import subprocess

# A run script + one of these as a dependency = a bootable frontend app. Canonical package names
# (scoped names listed in full) so membership is an exact set test.
FRONTEND_FRAMEWORKS = {
    "react", "react-dom", "vue", "nuxt", "next", "svelte", "@sveltejs/kit", "astro", "solid-js",
    "preact", "gatsby", "@angular/core", "@remix-run/react", "@remix-run/node", "remix",
    "ember-source", "alpinejs", "lit", "qwik", "@builder.io/qwik", "vite", "@vue/cli-service",
    "react-scripts", "@quasar/app", "vitepress", "@nuxt/kit",
}

# Server frameworks + a template engine = a server-rendered bootable app even without a JS frontend
# framework (the false-negative class called out in the spec). Both sets must intersect to accept.
SERVER_FRAMEWORKS = {"express", "fastify", "koa", "@nestjs/core", "hapi", "@hapi/hapi", "h3"}
TEMPLATE_ENGINES = {"ejs", "pug", "handlebars", "express-handlebars", "hbs", "nunjucks", "eta"}

# Run scripts that start a long-lived server we can drive.
RUN_SCRIPTS = {"dev", "start", "serve", "preview", "develop"}

# e2e test tooling: corroborates "this is a test suite, not an app" when no framework is present.
E2E_TEST_TOOLING = {
    "@playwright/test", "playwright", "playwright-core", "cypress", "selenium-webdriver",
    "@wdio/cli", "webdriverio", "puppeteer", "nightwatch", "testcafe", "@testing-library/cypress",
}

# A config file (entry-name prefix) for a framework that serves an app; with a run script = bootable.
APP_CONFIG_PREFIXES = (
    "next.config", "nuxt.config", "vite.config", "svelte.config", "astro.config", "remix.config",
    "gatsby-config", "angular.json", "quasar.config", "vue.config", "rsbuild.config",
)

COMPOSE_FILES = {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}

# Root markers of a JS monorepo: the bootable app lives in a workspace subdir, so the root
# package.json often has no framework dep. Accept conservatively (the per-app deep-probe is
# deferred) — these are real multi-package apps, not test suites.
MONOREPO_MARKERS = {"pnpm-workspace.yaml", "lerna.json", "turbo.json", "nx.json", "rush.json"}

# Directory names that signal app source (vs a pure test repo).
APP_STRUCTURE_DIRS = {"src", "app", "apps", "client", "frontend", "web", "packages", "server"}


def parse_package_json(text: str) -> dict | None:
    """Extract the signal fields from package.json text. None when the text is absent/unparseable
    (the caller then falls through to the no-package-json branch)."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    deps = set(data.get("dependencies", {}) or {}) | set(data.get("devDependencies", {}) or {})
    return {
        "scripts": set(data.get("scripts", {}) or {}),
        "deps": deps,
        "has_workspaces": bool(data.get("workspaces")),
    }


def classify_bootable(entry_names, pkg: dict | None) -> tuple[bool, str]:
    """Pure verdict from a repo's root tree entry names + parsed root package.json (or None).

    ACCEPT a plausible bootable interactive app; REJECT the test-suite / non-app majority. Borderline
    cases lean ACCEPT — the downstream LLM stages still judge quality on this (smaller) survivor set;
    the gate only removes high-confidence junk.
    """
    names = set(entry_names)
    names_lower = {n.lower() for n in names}

    compose_present = bool(names_lower & COMPOSE_FILES)
    app_config_present = any(n.startswith(APP_CONFIG_PREFIXES) for n in names_lower)
    has_app_dir = bool(names & APP_STRUCTURE_DIRS)
    is_monorepo = bool(names_lower & MONOREPO_MARKERS) or bool(pkg and pkg["has_workspaces"])

    scripts = pkg["scripts"] if pkg else set()
    deps = pkg["deps"] if pkg else set()
    has_run = bool(scripts & RUN_SCRIPTS)
    has_frontend = bool(deps & FRONTEND_FRAMEWORKS)
    has_server_rendered = bool(deps & SERVER_FRAMEWORKS) and bool(deps & TEMPLATE_ENGINES)

    # --- ACCEPT signals ---
    if has_run and has_frontend:
        return True, f"node frontend app ({', '.join(sorted(deps & FRONTEND_FRAMEWORKS))})"
    if has_run and has_server_rendered:
        return True, "server-rendered node app (server framework + template engine)"
    if compose_present:
        return True, "docker-compose present (self-hostable app)"
    if app_config_present and has_run:
        return True, "app-config file + run script"
    if is_monorepo:
        # The app is in a workspace package; root deps are not representative. Per-app deep-probe
        # is deferred — accept and let the (now much smaller) LLM stages judge.
        return True, "js monorepo (per-app deep-probe deferred)"

    # --- REJECT signals ---
    if pkg is None and not compose_present and not has_app_dir:
        return False, "no package.json / compose / app structure"
    if pkg is not None and not has_frontend and not has_server_rendered:
        if deps & E2E_TEST_TOOLING:
            return False, "test-suite only (e2e tooling, no app framework)"
        return False, "package.json without a frontend/server-app framework"
    return False, "no bootable-app signal"


_PROBE_QUERY = """
query($owner:String!,$name:String!){
  repository(owner:$owner,name:$name){
    root: object(expression:"HEAD:"){ ... on Tree { entries { name } } }
    pkg: object(expression:"HEAD:package.json"){ ... on Blob { text } }
  }
}
"""


class ProbeError(Exception):
    """Raised when the GitHub probe could not be completed (network/auth/missing repo). The gate
    treats this as a retryable infra failure, never a rejection."""


def probe_repo(full_name: str) -> tuple[list[str], str | None]:
    """Fetch (root entry names, package.json text|None) for a repo via the gh GraphQL API.
    Raises ProbeError on any failure so the caller can retry rather than reject."""
    if "/" not in full_name:
        raise ProbeError(f"bad repo name: {full_name}")
    owner, name = full_name.split("/", 1)
    try:
        result = subprocess.run(
            ["gh", "api", "graphql", "-f", f"owner={owner}", "-f", f"name={name}",
             "-f", f"query={_PROBE_QUERY}"],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise ProbeError(f"gh probe failed for {full_name}: {e}")
    if result.returncode != 0:
        raise ProbeError(f"gh probe non-zero for {full_name}: {result.stderr.strip()[:200]}")
    try:
        repo = json.loads(result.stdout)["data"]["repository"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ProbeError(f"gh probe unparseable for {full_name}: {e}")
    if repo is None:
        raise ProbeError(f"repository not found: {full_name}")
    root = repo.get("root") or {}
    entries = [e["name"] for e in (root.get("entries") or [])]
    pkg_obj = repo.get("pkg") or {}
    return entries, pkg_obj.get("text")


def gate_repo(full_name: str) -> tuple[bool, str]:
    """Full gate for one repo: probe + classify. Raises ProbeError on infra failure."""
    entries, pkg_text = probe_repo(full_name)
    pkg = parse_package_json(pkg_text) if pkg_text else None
    return classify_bootable(entries, pkg)
