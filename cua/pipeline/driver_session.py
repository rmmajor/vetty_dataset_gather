"""Host side of instrumentation/browser_driver.js: install it, start it, boot the repo's app, and talk to it."""

import asyncio
import logging
import re
import shlex

from cua.pipeline import app_runner
from cua.pipeline.instrumentation import INSTRUMENTATION_DIR, RESULTS_DIR
from cua.settings import PIPELINE, SANDBOX

logger = logging.getLogger(__name__)

DRIVER_DIR = "/workspace/driver"
DRIVER_BAKED_DIR = "/opt/driver"
DRIVER_LOG = "/workspace/driver.log"
APP_LOG = "/workspace/app.log"

_REF_LINE_RE = re.compile(r"^\[\d+\] ", re.MULTILINE)


class DriverError(Exception):
    """The driver rejected a request; its message is agent-readable and reaches the model as a tool error."""


def install_driver(sandbox):
    """Write browser_driver.js into the container and make playwright-core importable."""
    driver_src = INSTRUMENTATION_DIR.joinpath("browser_driver.js").read_text()
    sandbox.exec_command(f"mkdir -p {DRIVER_DIR} {RESULTS_DIR}")
    sandbox.write_file(f"{DRIVER_DIR}/browser_driver.js", driver_src)

    probe = f"node -e \"require('{DRIVER_BAKED_DIR}/node_modules/playwright-core')\" 2>/dev/null && echo BAKED"
    stdout, _, _ = sandbox.exec_command(probe)
    if "BAKED" in stdout:
        return

    # Pinned to the image's browser revision, and never downloading browsers: the image ships them in /ms-playwright.
    stdout, stderr, exit_code = sandbox.exec_command(
        f"cd {DRIVER_DIR} && npm init -y >/dev/null 2>&1 && "
        f"PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm install playwright-core@{PIPELINE.driver_playwright_core_version} "
        f"--no-audit --no-fund 2>&1 | tail -3",
        timeout=240,
    )
    if exit_code != 0:
        raise DriverError(f"playwright-core install failed: {(stdout + stderr)[-300:]}")


def start_driver(sandbox, env: dict | None = None, port: int = PIPELINE.driver_port):
    """Daemonize the driver on `port`, one process per port, with `env` prefixing the node invocation."""
    # `env` stays the positional second argument: the eval harness passes it positionally.
    prefix = "".join(f"{k}={v} " for k, v in {**(env or {}), "DRIVER_PORT": port}.items())
    log_path = DRIVER_LOG if port == PIPELINE.driver_port else f"{DRIVER_LOG}.{port}"
    sandbox.start_background(f"{prefix}node {DRIVER_DIR}/browser_driver.js", log_path)
    if not sandbox.wait_for_port(port, PIPELINE.driver_ready_timeout):
        log_tail, _, _ = sandbox.exec_command(f"tail -20 {log_path}")
        raise DriverError(f"driver did not open port {port}; log tail: {log_tail[-400:]}")


async def run_blocking(fn, *args):
    """Run a blocking sandbox/driver call off the event loop so concurrent workers overlap."""
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def boot_driver_sync(sandbox, env: dict | None = None) -> "DriverClient":
    """Install + start the browser driver in a live sandbox and return a client for it."""
    install_driver(sandbox)
    start_driver(sandbox, env)
    return DriverClient(sandbox)


async def boot_driver(sandbox, env: dict | None = None) -> "DriverClient":
    """`boot_driver_sync` for an async caller, each blocking leg through the executor."""
    await run_blocking(install_driver, sandbox)
    await run_blocking(start_driver, sandbox, env)
    return DriverClient(sandbox)


def _wait_for_http_port(sandbox, preferred: int, timeout: int) -> int | None:
    """Poll for a listening TCP port that ANSWERS HTTP and return it, trying `preferred` first each round."""
    # Accepting any answering port keeps an app that picked a non-guessed one from reading as NO_BOOT.
    script = (
        f"for i in $(seq 1 {timeout}); do "
        f"for p in {preferred} $(cat /proc/net/tcp /proc/net/tcp6 2>/dev/null "
        f"| awk 'NR>1 && $4==\"0A\"{{print $2}}' | sed 's/.*://' "
        f"| while read h; do printf '%d\\n' \"0x$h\" 2>/dev/null; done | sort -un); do "
        f'[ "$p" = "{PIPELINE.driver_port}" ] && continue; '
        f"code=$(curl -s -o /dev/null -w '%{{http_code}}' -m 2 \"http://127.0.0.1:$p/\" 2>/dev/null); "
        f'case "$code" in 2*|3*) echo "CUA_PORT=$p"; exit 0;; esac; '
        f"done; sleep 1; done"
    )
    out, _, _ = sandbox.exec_command(script, timeout=timeout + 20)
    m = re.search(r"CUA_PORT=(\d+)", out or "")
    return int(m.group(1)) if m else None


def app_is_up(sandbox, app_url: str) -> bool:
    """True while the app server still answers HTTP at `app_url`, so a caller can restart it before the next task."""
    out, _, _ = sandbox.exec_command(f"curl -s -o /dev/null -w '%{{http_code}}' -m 5 '{app_url}/' 2>/dev/null")
    return out.strip()[:1] in ("2", "3")


def start_app(sandbox, profile, extra_env: dict | None = None) -> str | None:
    """Detect, build and daemonize the repo's bundled app, returning its base URL or None."""
    # for_agent: no Playwright run honours a `webServer` block here, and those blocks wrap the same start script anyway.
    spec = app_runner.detect_app_start(sandbox, profile, for_agent=True)
    if spec is None:
        return None
    # The app's OWN package manager in its own dir: a root `npm install` no-ops for monorepo apps and fails for pnpm.
    sandbox.exec_command(spec.install_cmd, timeout=600)
    app_runner.build_app(sandbox, spec)
    start_cmd = spec.start_cmd
    if extra_env:
        prefix = "; ".join(f"export {k}={shlex.quote(str(v))}" for k, v in extra_env.items())
        start_cmd = f"{prefix}; {start_cmd}"
    sandbox.start_background(start_cmd, APP_LOG)
    port = _wait_for_http_port(sandbox, spec.port, SANDBOX.app_ready_timeout)
    if port is None:
        log_tail, _, _ = sandbox.exec_command(f"tail -25 {APP_LOG}")
        logger.warning(f"app did not open an HTTP port (inferred :{spec.port}); log tail: {log_tail[-500:]}")
        return None
    if port != spec.port:
        logger.info(f"app bound :{port} (inferred :{spec.port}) — using detected port")
    return f"http://127.0.0.1:{port}"


def count_interactive_elements(snapshot: str) -> int:
    """Number of `[N] role "name"` element lines in a driver snapshot, the boot gate's non-triviality signal."""
    return len(_REF_LINE_RE.findall(snapshot or ""))


class DriverClient:
    """Typed calls to the in-container driver, raising DriverError carrying the driver's own message."""

    def __init__(self, sandbox, port: int = PIPELINE.driver_port):
        self.sandbox = sandbox
        self.base = f"http://127.0.0.1:{port}"

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
        mail_config: dict | None = None,
        type_mode: str | None = None,
    ) -> dict:
        # Session-scoped, not per /act, so the secret-transport surface stays minimal; http_json base64s it, never argv.
        body = {
            "task_id": task_id,
            "start_url": start_url,
            "instruction": instruction,
            "task_meta": task_meta or {},
        }
        if auth_config:
            body["auth_config"] = auth_config
        if secrets:
            body["secrets"] = secrets
        if mail_config:
            body["mail_config"] = mail_config
        if type_mode:
            body["type_mode"] = type_mode
        return self._post("/session", body)

    def act(self, payload: dict) -> dict:
        return self._post("/act", payload)

    def observe(self) -> dict:
        """PNG observation whose bytes become the next recorded step's screenshot (RL parity)."""
        return self._post("/observe")

    def act_sequence(self, thought: str, actions: list[dict]) -> dict:
        return self._post("/act", {"kind": "sequence", "thought": thought, "actions": actions})

    def snapshot(self) -> dict:
        return self._post("/snapshot")

    def page_text(self) -> str:
        """The page's rendered text, for success oracles that must see a computed result the snapshot omits."""
        return self._post("/page_text")["text"]

    def screenshot_b64(self, image_format: str = "jpeg") -> str:
        return self._post("/screenshot", {"format": image_format})["b64"]

    def navigate(self, url: str) -> dict:
        return self._post("/navigate", {"url": url})

    def storage_state(self) -> dict:
        """The active context's cookies and localStorage, so a later task can start already authenticated."""
        return self._post("/storage_state")["storage_state"]

    def wait(self, seconds: float) -> dict:
        return self._post("/wait", {"seconds": seconds})

    def affordances(self) -> dict:
        return self._post("/affordances")["affordances"]

    def finish(self, agent_outcome: str, agent_summary: str) -> dict:
        return self._post("/finish", {"agent_outcome": agent_outcome, "agent_summary": agent_summary})
