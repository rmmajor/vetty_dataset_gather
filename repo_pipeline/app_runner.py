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

from repo_pipeline.frameworks import FrameworkProfile
from repo_pipeline.instrumentation import REPO_DIR

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
_DEV_SERVER_ENV = "DISABLE_ESLINT_PLUGIN=true ESLINT_NO_DEV_ERRORS=true BROWSER=none CI=''"
_PW_CONFIG_NAMES = (
    "playwright.config.ts", "playwright.config.js", "playwright.config.mjs",
    "playwright.config.cjs", "playwright.config.mts", "playwright.config.cts",
)
# Framework dependency name → its default dev-server port, when the script doesn't state one.
_FRAMEWORK_PORTS = (
    ("next", 3000), ("nuxt", 3000), ("@remix-run", 3000), ("vite", 5173),
    ("@sveltejs", 5173), ("react-scripts", 3000), ("@angular", 4200),
    ("vue-cli-service", 8080), ("astro", 4321), ("gatsby", 8000), ("vue", 8080),
)


@dataclass(frozen=True)
class AppStartSpec:
    start_cmd: str           # shell command that boots the app (run from REPO_DIR)
    port: int                # TCP port the app is expected to listen on
    build_cmd: str | None    # one-time build to run before starting, or None


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


def detect_app_start(sandbox, profile: FrameworkProfile, for_agent: bool = False) -> AppStartSpec | None:
    """Inspect the cloned repo and decide how (if at all) to start its app.

    Returns None when there's nothing to start here: no Node app, or a Playwright config that
    already manages its own `webServer` (left to Playwright), or no recognizable start script.

    ``for_agent=True`` (agent capture / boot gate): there is no Playwright test run to honour a
    `webServer` config, so the webServer short-circuit is skipped and the package.json start
    script is used directly — webServer blocks almost always wrap that same script.
    """
    pkg = _read_json(sandbox, f"{REPO_DIR}/package.json")
    if not pkg:
        return None  # no Node app to start (python app servers etc. are out of scope for v1)

    if not for_agent and profile.test_framework == "playwright" and _has_webserver_config(sandbox):
        logger.info("app-start: Playwright webServer config present — leaving startup to Playwright")
        return None

    scripts = pkg.get("scripts", {}) or {}
    chosen = next((s for s in _START_SCRIPTS if s in scripts), None)
    if not chosen:
        return None

    dep_names = " ".join(list(pkg.get("dependencies", {})) + list(pkg.get("devDependencies", {})))
    port = _infer_port(scripts.get(chosen, ""), dep_names)
    # `dev` runs unbuilt; production-style starts (start/serve/preview) need a build first.
    build_cmd = f"cd {REPO_DIR} && npm run build" if (chosen != "dev" and "build" in scripts) else None
    spec = AppStartSpec(
        start_cmd=f"cd {REPO_DIR} && {_DEV_SERVER_ENV} npm run {chosen}",
        port=port,
        build_cmd=build_cmd,
    )
    logger.info(f"app-start: will run `npm run {chosen}` on :{port}"
                f"{' (after build)' if build_cmd else ''}")
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
