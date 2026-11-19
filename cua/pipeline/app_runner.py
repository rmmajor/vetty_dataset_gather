"""Detect, start and stop a repo's own Node web app inside the sandbox, for replay tests and for agent capture."""

import json
import logging
import re
from dataclasses import dataclass

from cua.pipeline.frameworks import FrameworkProfile
from cua.pipeline.instrumentation import REPO_DIR
from cua.settings import SANDBOX

logger = logging.getLogger(__name__)

# Start scripts to try, in order of preference. `dev` needs no build and is the most likely to work.
_START_SCRIPTS = ("dev", "start", "serve", "preview")

# Suppresses the dev-server lint overlay from every screenshot, and resolves `localhost` to the IPv4 the probe uses.
_DEV_SERVER_ENV = (
    "DISABLE_ESLINT_PLUGIN=true ESLINT_NO_DEV_ERRORS=true BROWSER=none CI='' "
    "NODE_OPTIONS=--dns-result-order=ipv4first HOST=0.0.0.0 npm_config_engine_strict=false"
)
_PW_CONFIG_NAMES = (
    "playwright.config.ts",
    "playwright.config.js",
    "playwright.config.mjs",
    "playwright.config.cjs",
    "playwright.config.mts",
    "playwright.config.cts",
)
# Framework dependency name → its default dev-server port, when the script doesn't state one.
_FRAMEWORK_PORTS = (
    ("next", 3000),
    ("nuxt", 3000),
    ("@remix-run", 3000),
    ("vite", 5173),
    ("@sveltejs", 5173),
    ("react-scripts", 3000),
    ("@angular", 4200),
    ("vue-cli-service", 8080),
    ("astro", 4321),
    ("gatsby", 8000),
    ("vue", 8080),
)


@dataclass(frozen=True)
class AppStartSpec:
    start_cmd: str  # shell command that boots the app
    port: int  # TCP port the app is expected to listen on
    build_cmd: str | None  # one-time build to run before starting, or None
    app_dir: str = REPO_DIR  # directory holding the app's package.json (monorepos: a subdir)
    package_manager: str = "npm"  # npm | pnpm | yarn — drives install + run
    install_cmd: str = ""  # PM-aware dependency install for app_dir (empty ⇒ caller installs)


# Web-app directory names — boosted when scoring monorepo subdirs.
_APP_SUBDIR_NAMES = ("frontend", "client", "web", "app", "ui", "www", "site", "dashboard", "webapp")
# A package.json pulling one of these is almost certainly the bootable UI, not an api or lib package beside it.
_FRONTEND_DEPS = (
    "react",
    "vue",
    "svelte",
    "next",
    "nuxt",
    "vite",
    "@angular",
    "solid-js",
    "preact",
    "astro",
    "@remix-run",
    "gatsby",
)


def _find_app_dir(sandbox) -> str | None:
    """The best directory to boot: a package.json with a start script, scored by framework, name and depth."""
    best = None  # (score, app_dir)
    for path in sandbox.list_files("package.json", REPO_DIR):
        if "/node_modules/" in path:
            continue
        rel = path.replace(REPO_DIR + "/", "")
        depth = rel.count("/")  # 0 = root package.json
        if depth > 3:
            continue
        pkg = _read_json(sandbox, path) or {}
        if not any(s in (pkg.get("scripts", {}) or {}) for s in _START_SCRIPTS):
            continue
        app_dir = REPO_DIR if depth == 0 else f"{REPO_DIR}/{rel.rsplit('/', 1)[0]}"
        deps = " ".join(list(pkg.get("dependencies", {})) + list(pkg.get("devDependencies", {})))
        score = -depth  # prefer shallower (root app, then apps/web over apps/web/inner)
        if any(d in deps for d in _FRONTEND_DEPS):
            score += 10
        if app_dir.rsplit("/", 1)[-1].lower() in _APP_SUBDIR_NAMES:
            score += 5
        if best is None or score > best[0]:
            best = (score, app_dir)
    return best[1] if best else None


def _install_dir(sandbox, app_dir: str) -> str:
    """Where the dependency install runs: the workspace root for a monorepo package, else the app dir."""
    # Installing at the root is what puts a workspace app's own binaries on PATH.
    if app_dir == REPO_DIR:
        return REPO_DIR
    out, _, _ = sandbox.exec_command(
        f"ls {REPO_DIR}/pnpm-lock.yaml {REPO_DIR}/yarn.lock {REPO_DIR}/package-lock.json "
        f"{REPO_DIR}/pnpm-workspace.yaml 2>/dev/null"
    )
    return REPO_DIR if out.strip() else app_dir


def _read_json(sandbox, path: str) -> dict | None:
    try:
        return json.loads(sandbox.read_file(path))
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return None


def _has_webserver_config(sandbox) -> bool:
    """True if a Playwright config declares a `webServer` — Playwright starts the app itself."""
    for name in _PW_CONFIG_NAMES:
        try:
            content = sandbox.read_file(f"{REPO_DIR}/{name}")
        except FileNotFoundError:
            continue
        if re.search(r"\bwebServer\b", content):
            return True
    return False


def _infer_port(script: str, dep_names: str) -> int:
    m = re.search(r"(?:--port[ =]|-p[ =]?)(\d{2,5})", script or "")
    if m:
        return int(m.group(1))
    for needle, port in _FRAMEWORK_PORTS:
        if needle in dep_names:
            return port
    return 3000


def detect_package_manager(sandbox, app_dir: str) -> str:
    """The app's package manager, from the `packageManager` field, else a lockfile, else npm."""
    # A pnpm or yarn project under a blind `npm install` never lands its binaries on PATH: the dominant boot failure.
    field = ((_read_json(sandbox, f"{app_dir}/package.json") or {}).get("packageManager") or "").lower()
    if field.startswith("pnpm"):
        return "pnpm"
    if field.startswith("yarn"):
        return "yarn"
    out, _, _ = sandbox.exec_command(f"ls {app_dir}/pnpm-lock.yaml {app_dir}/yarn.lock 2>/dev/null")
    if "pnpm-lock.yaml" in out:
        return "pnpm"
    if "yarn.lock" in out:
        return "yarn"
    return "npm"


def _install_cmd(pm: str, app_dir: str) -> str:
    """The dependency-install command for the detected package manager."""
    # Globally via npm, not corepack, and pinned to the permissive major: pnpm 10+ gates unattended installs and runs.
    if pm == "npm":
        return f"cd {app_dir} && npm install"
    if pm == "pnpm":
        return (
            "npm install -g pnpm@8 >/dev/null 2>&1; "
            f"cd {app_dir} && npm_config_engine_strict=false pnpm install --no-frozen-lockfile"
        )
    return (  # yarn classic
        f"npm install -g yarn@1 >/dev/null 2>&1; cd {app_dir} && npm_config_engine_strict=false yarn install"
    )


def detect_app_start(sandbox, profile: FrameworkProfile, for_agent: bool = False) -> AppStartSpec | None:
    """How to start this repo's app, or None when there is no Node app, no start script, or Playwright owns it."""
    # for_agent skips the webServer short-circuit: no Playwright run honours it, and such blocks wrap the same script.
    app_dir = _find_app_dir(sandbox)
    if app_dir is None:
        return None  # no Node app to start (python app servers etc. are out of scope for v1)
    pkg = _read_json(sandbox, f"{app_dir}/package.json")
    if not pkg:
        return None

    if not for_agent and profile.test_framework == "playwright" and _has_webserver_config(sandbox):
        logger.info("app-start: Playwright webServer config present — leaving startup to Playwright")
        return None

    scripts = pkg.get("scripts", {}) or {}
    chosen = next((s for s in _START_SCRIPTS if s in scripts), None)
    if not chosen:
        return None

    # Install at the workspace root for a monorepo app, but detect the PM and run the dev script in the app's own dir.
    install_dir = _install_dir(sandbox, app_dir)
    pm = detect_package_manager(sandbox, install_dir)
    dep_names = " ".join(list(pkg.get("dependencies", {})) + list(pkg.get("devDependencies", {})))
    port = _infer_port(scripts.get(chosen, ""), dep_names)
    # `dev` runs unbuilt; production-style starts (start/serve/preview) need a build first.
    build_cmd = f"cd {app_dir} && {pm} run build" if (chosen != "dev" and "build" in scripts) else None
    spec = AppStartSpec(
        start_cmd=f"cd {app_dir} && {_DEV_SERVER_ENV} {pm} run {chosen}",
        port=port,
        build_cmd=build_cmd,
        app_dir=app_dir,
        package_manager=pm,
        install_cmd=_install_cmd(pm, install_dir),
    )
    logger.info(
        f"app-start: will run `{pm} run {chosen}` in {app_dir.replace(REPO_DIR, '.')} on :{port}"
        f"{' (after build, install in ' + install_dir.replace(REPO_DIR, '.') + ')' if build_cmd else ''} [pm={pm}]"
    )
    return spec


def build_app(sandbox, spec: AppStartSpec | None):
    """Run the one-time build (if the chosen start needs it) before the test runs."""
    if spec and spec.build_cmd:
        sandbox.exec_command(spec.build_cmd, timeout=SANDBOX.app_build_timeout)


def wrap_test_command(test_cmd: str, spec: AppStartSpec | None) -> str:
    """Wrap a test command so the app is started, waited on, tested and stopped inside ONE shell."""
    # One shell because each exec is independent: a dev server backgrounded in one call dies before the next.
    if spec is None:
        return test_cmd
    return (
        f"( {spec.start_cmd} >/workspace/app.log 2>&1 ) & APP_PID=$! ; "
        f"for i in $(seq 1 {SANDBOX.app_ready_timeout}); do "
        f"(exec 3<>/dev/tcp/127.0.0.1/{spec.port}) 2>/dev/null && break; sleep 1; done ; "
        f"{test_cmd} ; "
        f"kill $APP_PID 2>/dev/null ; pkill -P $APP_PID 2>/dev/null || true"
    )
