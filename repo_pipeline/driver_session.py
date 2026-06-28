"""Host-side companion to instrumentation/browser_driver.js.

Owns everything about getting an agent-drivable browser running inside a sandbox:

  * `install_driver` writes the driver into the container and ensures playwright-core exists
    (baked /opt/driver from the image when available, runtime npm install otherwise);
  * `start_driver` daemonizes it and waits for its port;
  * `start_app` boots the repo's own app daemonized (unlike `app_runner.wrap_test_command`,
    which scopes the app to a single shell, agent capture needs the app alive across many
    independent exec calls);
  * `DriverClient` is a thin typed wrapper over the driver's JSON-over-HTTP endpoints.

Used by stage 5 (boot-gate probe) and stage 6 (agent capture).
"""
import logging
import os
import re

from repo_pipeline import app_runner
from repo_pipeline.instrumentation import INSTRUMENTATION_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

DRIVER_PORT = int(os.environ.get("DRIVER_PORT", "8765"))
DRIVER_DIR = "/workspace/driver"
DRIVER_BAKED_DIR = "/opt/driver"
DRIVER_LOG = "/workspace/driver.log"
APP_LOG = "/workspace/app.log"
# Pinned to the sandbox image's Playwright base (browsers in /ms-playwright are this revision).
PLAYWRIGHT_CORE_VERSION = os.environ.get("DRIVER_PLAYWRIGHT_CORE_VERSION", "1.50.0")
DRIVER_READY_TIMEOUT = int(os.environ.get("DRIVER_READY_TIMEOUT", "60"))

_REF_LINE_RE = re.compile(r"^\[\d+\] ", re.MULTILINE)


class DriverError(Exception):
    """The driver rejected a request (stale ref, loop guard, no session...). The message is
    agent-readable and is surfaced back to the model as a tool error so it can adapt."""


def install_driver(sandbox):
    """Write browser_driver.js into the container and make playwright-core importable.

    The baked /opt/driver (added to the sandbox Dockerfile) makes this a no-op install; on
    older images we npm-install playwright-core pinned to the image's browser revision.
    Browsers are NOT downloaded — the image already ships them in /ms-playwright.
    """
    driver_src = INSTRUMENTATION_DIR.joinpath("browser_driver.js").read_text()
    sandbox.exec_command(f"mkdir -p {DRIVER_DIR} {RESULTS_DIR}")
    sandbox.write_file(f"{DRIVER_DIR}/browser_driver.js", driver_src)

    probe = (
        f"node -e \"require('{DRIVER_BAKED_DIR}/node_modules/playwright-core')\" 2>/dev/null && echo BAKED"
    )
    stdout, _, _ = sandbox.exec_command(probe)
    if "BAKED" in stdout:
        return

    stdout, stderr, exit_code = sandbox.exec_command(
        f"cd {DRIVER_DIR} && npm init -y >/dev/null 2>&1 && "
        f"PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm install playwright-core@{PLAYWRIGHT_CORE_VERSION} "
        f"--no-audit --no-fund 2>&1 | tail -3",
        timeout=240,
    )
    if exit_code != 0:
        raise DriverError(f"playwright-core install failed: {(stdout + stderr)[-300:]}")


def start_driver(sandbox):
    """Daemonize the driver and wait for its HTTP port."""
    sandbox.start_background(f"node {DRIVER_DIR}/browser_driver.js", DRIVER_LOG)
    if not sandbox.wait_for_port(DRIVER_PORT, DRIVER_READY_TIMEOUT):
        log_tail, _, _ = sandbox.exec_command(f"tail -20 {DRIVER_LOG}")
        raise DriverError(f"driver did not open port {DRIVER_PORT}; log tail: {log_tail[-400:]}")


def _wait_for_http_port(sandbox, preferred: int, timeout: int) -> int | None:
    """Poll for a listening TCP port that ANSWERS HTTP and return it. The inferred `preferred` port
    is tried first each round; otherwise any other listening port that returns a 2xx/3xx is accepted
    — so an app on a non-guessed port (webpack-dev-server defaults to 8080, custom servers pick
    anything) is still found instead of falsely read as NO_BOOT. Runs as one in-container shell loop
    (parsing /proc/net/tcp{,6}) to avoid per-second docker-exec overhead; excludes the driver port."""
    script = (
        f"for i in $(seq 1 {timeout}); do "
        f"for p in {preferred} $(cat /proc/net/tcp /proc/net/tcp6 2>/dev/null "
        f"| awk 'NR>1 && $4==\"0A\"{{print $2}}' | sed 's/.*://' "
        f"| while read h; do printf '%d\\n' \"0x$h\" 2>/dev/null; done | sort -un); do "
        f"[ \"$p\" = \"{DRIVER_PORT}\" ] && continue; "
        f"code=$(curl -s -o /dev/null -w '%{{http_code}}' -m 2 \"http://127.0.0.1:$p/\" 2>/dev/null); "
        f"case \"$code\" in 2*|3*) echo \"CUA_PORT=$p\"; exit 0;; esac; "
        f"done; sleep 1; done"
    )
    out, _, _ = sandbox.exec_command(script, timeout=timeout + 20)
    m = re.search(r"CUA_PORT=(\d+)", out or "")
    return int(m.group(1)) if m else None


def app_is_up(sandbox, app_url: str) -> bool:
    """True if the app server still answers HTTP (2xx/3xx) at app_url. Tasks share one long-lived
    app server that can crash mid-session (an unhandled error, an OOM, a route that kills the dev
    server) — after which every remaining task fails with CONNECTION_REFUSED. Cheap in-container
    curl probe so the caller can restart the app before the next task."""
    out, _, _ = sandbox.exec_command(
        f"curl -s -o /dev/null -w '%{{http_code}}' -m 5 '{app_url}/' 2>/dev/null"
    )
    return out.strip()[:1] in ("2", "3")


def start_app(sandbox, profile) -> str | None:
    """Detect, build, and daemonize the repo's bundled app; return its base URL or None.

    Agent capture (unlike test replay) has no Playwright run to honour a `webServer` config,
    so detection runs with for_agent=True: package.json start scripts are used even when a
    webServer block exists — those blocks almost always wrap the same script.
    """
    spec = app_runner.detect_app_start(sandbox, profile, for_agent=True)
    if spec is None:
        return None
    # Install with the app's OWN package manager, in its own dir. A blind root `npm install` is a
    # no-op for monorepo subdir apps and outright fails for pnpm/yarn/workspace repos (their dev
    # binary then never lands on PATH — the dominant boot failure), so the install lives here next
    # to the start that depends on it. A failed install surfaces at start as `<tool>: not found`
    # in APP_LOG, which is what the boot gate reports.
    sandbox.exec_command(spec.install_cmd, timeout=600)
    app_runner.build_app(sandbox, spec)
    sandbox.start_background(spec.start_cmd, APP_LOG)
    port = _wait_for_http_port(sandbox, spec.port, app_runner.APP_READY_TIMEOUT)
    if port is None:
        log_tail, _, _ = sandbox.exec_command(f"tail -25 {APP_LOG}")
        logger.warning(f"app did not open an HTTP port (inferred :{spec.port}); log tail: {log_tail[-500:]}")
        return None
    if port != spec.port:
        logger.info(f"app bound :{port} (inferred :{spec.port}) — using detected port")
    return f"http://127.0.0.1:{port}"


def count_interactive_elements(snapshot: str) -> int:
    """Number of `[N] role "name"` element lines in a driver snapshot — the non-triviality
    signal for the stage-5 boot gate."""
    return len(_REF_LINE_RE.findall(snapshot or ""))


class DriverClient:
    """Typed calls to the in-container driver. Raises DriverError with the driver's own
    (agent-readable) message when a request is rejected."""

    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.base = f"http://127.0.0.1:{DRIVER_PORT}"

    def _post(self, path: str, payload: dict | None = None, timeout: int = 120) -> dict:
        response = self.sandbox.http_json("POST", f"{self.base}{path}", payload or {}, timeout=timeout)
        if not response.get("ok", False):
            raise DriverError(response.get("error", f"driver error on {path}"))
        return response

    def status(self) -> dict:
        return self.sandbox.http_json("GET", f"{self.base}/status", timeout=30)

    def session(
        self,
        task_id: str,
        start_url: str,
        instruction: str = "",
        task_meta: dict | None = None,
        auth_config: dict | None = None,
        secrets: dict | None = None,
    ) -> dict:
        # `auth_config` carries the pre_session `storage_state` (cookies) the driver injects at
        # context creation; `secrets` is the session-scoped credential map the driver uses to
        # substitute `<secret:NAME>` placeholders at type time (never recorded). Both are set once
        # here rather than per /act, to keep the secret-transport surface minimal. Travels
        # base64-encoded via http_json (not plaintext on any command line, not logged).
        body = {
            "task_id": task_id, "start_url": start_url,
            "instruction": instruction, "task_meta": task_meta or {},
        }
        if auth_config:
            body["auth_config"] = auth_config
        if secrets:
            body["secrets"] = secrets
        return self._post("/session", body)

    def act(self, payload: dict) -> dict:
        return self._post("/act", payload)

    def snapshot(self) -> dict:
        return self._post("/snapshot")

    def screenshot_b64(self) -> str:
        return self._post("/screenshot")["b64"]

    def navigate(self, url: str) -> dict:
        return self._post("/navigate", {"url": url})

    def wait(self, seconds: float) -> dict:
        return self._post("/wait", {"seconds": seconds})

    def affordances(self) -> dict:
        return self._post("/affordances")["affordances"]

    def finish(self, agent_outcome: str, agent_summary: str) -> dict:
        return self._post("/finish", {"agent_outcome": agent_outcome, "agent_summary": agent_summary})
