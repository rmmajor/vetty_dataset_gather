"""No-LLM bootable-app gate: is this repo a bootable interactive web app, judged from one GitHub probe?"""

import json
import subprocess

# A run script plus one of these as a dependency is a bootable frontend app; scoped names in full for exact set tests.
FRONTEND_FRAMEWORKS = {
    "react",
    "react-dom",
    "vue",
    "nuxt",
    "next",
    "svelte",
    "@sveltejs/kit",
    "astro",
    "solid-js",
    "preact",
    "gatsby",
    "@angular/core",
    "@remix-run/react",
    "@remix-run/node",
    "remix",
    "ember-source",
    "alpinejs",
    "lit",
    "qwik",
    "@builder.io/qwik",
    "vite",
    "@vue/cli-service",
    "react-scripts",
    "@quasar/app",
    "vitepress",
    "@nuxt/kit",
}

# A server framework AND a template engine make a server-rendered app even without a JS frontend framework.
SERVER_FRAMEWORKS = {"express", "fastify", "koa", "@nestjs/core", "hapi", "@hapi/hapi", "h3"}
TEMPLATE_ENGINES = {"ejs", "pug", "handlebars", "express-handlebars", "hbs", "nunjucks", "eta"}

# Run scripts that start a long-lived server we can drive.
RUN_SCRIPTS = {"dev", "start", "serve", "preview", "develop"}

# e2e test tooling: corroborates "this is a test suite, not an app" when no framework is present.
E2E_TEST_TOOLING = {
    "@playwright/test",
    "playwright",
    "playwright-core",
    "cypress",
    "selenium-webdriver",
    "@wdio/cli",
    "webdriverio",
    "puppeteer",
    "nightwatch",
    "testcafe",
    "@testing-library/cypress",
}

# A config file (entry-name prefix) for a framework that serves an app; with a run script = bootable.
APP_CONFIG_PREFIXES = (
    "next.config",
    "nuxt.config",
    "vite.config",
    "svelte.config",
    "astro.config",
    "remix.config",
    "gatsby-config",
    "angular.json",
    "quasar.config",
    "vue.config",
    "rsbuild.config",
)

COMPOSE_FILES = {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}

# In a monorepo the bootable app lives in a workspace subdir, so the root package.json often has no framework dep.
MONOREPO_MARKERS = {"pnpm-workspace.yaml", "lerna.json", "turbo.json", "nx.json", "rush.json"}

# These directory names mark a repo that ships app source rather than a test suite alone.
APP_STRUCTURE_DIRS = {"src", "app", "apps", "client", "frontend", "web", "packages", "server"}

# Every other reason is a known false-positive class (vanilla and jQuery UIs), so it reaches the LLM stages instead.
HARD_REJECT_REASONS = frozenset(
    {
        "no package.json / compose / app structure",
        "test-suite only (e2e tooling, no app framework)",
    }
)


def is_hard_reject(reason: str) -> bool:
    """True when a REJECT reason is confident enough to drop the repo at the gate, with no LLM stage run."""
    return reason in HARD_REJECT_REASONS


def parse_package_json(text: str) -> dict | None:
    """The signal fields of package.json text, or None when it is absent or unparseable."""
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
    """Pure ACCEPT/REJECT verdict from a repo's root entry names and parsed root package.json."""
    # Borderline cases lean ACCEPT: the LLM stages still judge quality, so the gate only removes high-confidence junk.
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

    if has_run and has_frontend:
        return True, f"node frontend app ({', '.join(sorted(deps & FRONTEND_FRAMEWORKS))})"
    if has_run and has_server_rendered:
        return True, "server-rendered node app (server framework + template engine)"
    if compose_present:
        return True, "docker-compose present (self-hostable app)"
    if app_config_present and has_run:
        return True, "app-config file + run script"
    if is_monorepo:
        # Root deps are not representative of a workspace app, and the per-app deep probe is deferred.
        return True, "js monorepo (per-app deep-probe deferred)"

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
    """The GitHub probe could not be completed; the gate treats this as retryable infra, never a rejection."""


def probe_repo(full_name: str) -> tuple[list[str], str | None]:
    """(root entry names, package.json text or None) for a repo, via the gh GraphQL API."""
    if "/" not in full_name:
        raise ProbeError(f"bad repo name: {full_name}")
    owner, name = full_name.split("/", 1)
    try:
        result = subprocess.run(
            ["gh", "api", "graphql", "-f", f"owner={owner}", "-f", f"name={name}", "-f", f"query={_PROBE_QUERY}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise ProbeError(f"gh probe failed for {full_name}: {e}") from e
    if result.returncode != 0:
        raise ProbeError(f"gh probe non-zero for {full_name}: {result.stderr.strip()[:200]}")
    try:
        repo = json.loads(result.stdout)["data"]["repository"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ProbeError(f"gh probe unparseable for {full_name}: {e}") from e
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
