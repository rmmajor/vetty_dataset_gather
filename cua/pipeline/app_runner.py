"""Start a repo's own app before running its E2E tests, inside the sandbox.

Most E2E suites assume an app is already serving (a dev server on localhost). The pipeline ran
the tests with nothing serving, so they failed (`pass_rate 0`) and captured nothing — the dominant
no_capture cause. This module detects a repo's bundled web app and runs it alongside the tests.

The sandbox runs each command in an *independent* shell, so a dev server backgrounded in one
`exec_command` dies before the next call. The fix is to start + wait + test + stop inside a single
compound command (`wrap_test_command`). Repos whose Playwright config already declares a `webServer`
are left alone — Playwright manages that lifecycle itself.

Scope (v1): Node apps with a root `package.json` start script. Python app servers, monorepo
sub-apps, and stacks needing a DB/backend/credentials are out of scope (would need docker-compose /
host execution, which are deliberately not built). Those keep failing no_capture — now with a
persisted log (see stage5) explaining why.
"""

import json
import logging
import os
import re
from dataclasses import dataclass

from cua.pipeline.frameworks import FrameworkProfile
from cua.pipeline.instrumentation import REPO_DIR

logger = logging.getLogger(__name__)

# Seconds to wait for the dev server's port to open before running the tests anyway. Env-tunable.
APP_READY_TIMEOUT = int(os.environ.get("SANDBOX_APP_READY_TIMEOUT", "60"))
BUILD_TIMEOUT = int(os.environ.get("SANDBOX_APP_BUILD_TIMEOUT", "420"))

# Start scripts to try, in order of preference. `dev` needs no build and is the most likely to work.
_START_SCRIPTS = ("dev", "start", "serve", "preview")

# Dev servers must serve the APP, not developer tooling: CRA/webpack splash full-screen ESLint
# walls over the page on lint findings (e.g. CRLF/LF mismatches on checkout), which would land
# in every recorded screenshot. These env vars disable the lint plugin and overlay at the
# source; the browser driver additionally scrubs any overlay that still appears. BROWSER=none
# stops CRA trying to open a browser; CI='' keeps warnings from being promoted to hard errors.
#
# NODE_OPTIONS=--dns-result-order=ipv4first: dev servers (Vite, Next, …) bind to the hostname
# `localhost`, which Node resolves to IPv6 `::1` first on this image — but the boot probe and the
# browser driver reach the app over IPv4 `127.0.0.1`, so a perfectly-served app reads as NO_BOOT.
# Forcing IPv4-first resolution makes `localhost` bind 127.0.0.1, where the harness looks.
# HOST=0.0.0.0 covers servers that honour $HOST (CRA/webpack) by binding all interfaces.
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
# Frontend-framework markers: a package.json pulling one of these is almost certainly the bootable
# UI, vs an api/lib/tooling package sharing the same monorepo.
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
    """Best directory to boot: a package.json that has a dev/start script, scored to prefer one
    pulling in a frontend framework, in a web-ish, shallow dir. Searches root + monorepo subdirs
    (apps/*, packages/*, examples/* up to depth 3, excluding node_modules) so a real app in
    `apps/web` isn't missed — the dominant 'no bootable Node app' miss on monorepos. None when no
    package.json with a recognizable start script exists."""
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
    """Where dependency install runs: the workspace ROOT when the app is a monorepo package (so
    pnpm/yarn install the whole workspace and place the app's binaries on PATH), else the app dir."""
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
    """The app's package manager: the `packageManager` field (authoritative), else a lockfile,
    else npm. pnpm/yarn projects fail under a blind `npm install` — their workspace binaries
    (next, vite, turbo) never land on PATH, the dominant `sh: <tool>: not found` boot failure —
    so we install and run them with their own PM."""
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
    """Dependency-install command for the detected PM. npm is built in. pnpm/yarn are installed
    GLOBALLY via npm — NOT corepack, whose shim throws `DefaultVersion` on this image and then
    shadows any fallback. We pin the PERMISSIVE major line (pnpm 8 / yarn 1): pnpm 10+ blocks
    dependency build scripts behind an interactive `approve-builds` prompt and gates `pnpm run` on
    a deps-status check — both fatal for unattended boots — while pnpm 8 runs builds (esbuild etc.)
    directly. `npm_config_engine_strict=false` + `--no-frozen-lockfile` stop a mismatched
    engines.node or a newer-format lockfile from blocking install (we have one Node, one shot)."""
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
    """Inspect the cloned repo and decide how (if at all) to start its app.

    Returns None when there's nothing to start here: no Node app, or a Playwright config that
    already manages its own `webServer` (left to Playwright), or no recognizable start script.

    ``for_agent=True`` (agent capture / boot gate): there is no Playwright test run to honour a
    `webServer` config, so the webServer short-circuit is skipped and the package.json start
    script is used directly — webServer blocks almost always wrap that same script.
    """
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

    # Install at the workspace root for monorepo apps (pnpm/yarn install the whole workspace there
    # and place the app's binaries), but detect the PM and run the dev script in the app's own dir.
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
        sandbox.exec_command(spec.build_cmd, timeout=BUILD_TIMEOUT)


def wrap_test_command(test_cmd: str, spec: AppStartSpec | None) -> str:
    """Wrap a test command so the app is started, waited on, tested, and stopped in ONE shell.

    No-op when spec is None. Uses the bash `/dev/tcp` builtin for the port check (no curl needed).
    The app's exit/kill is best-effort: the container is ephemeral, and a server left listening on
    the port just keeps serving subsequent test files in the same trial.
    """
    if spec is None:
        return test_cmd
    return (
        f"( {spec.start_cmd} >/workspace/app.log 2>&1 ) & APP_PID=$! ; "
        f"for i in $(seq 1 {APP_READY_TIMEOUT}); do "
        f"(exec 3<>/dev/tcp/127.0.0.1/{spec.port}) 2>/dev/null && break; sleep 1; done ; "
        f"{test_cmd} ; "
        f"kill $APP_PID 2>/dev/null ; pkill -P $APP_PID 2>/dev/null || true"
    )
