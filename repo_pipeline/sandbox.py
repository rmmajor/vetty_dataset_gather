import asyncio
import base64
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import docker

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "repo-pipeline-sandbox:latest"
CONTAINER_TIMEOUT_SECONDS = 1800  # 30 minutes
EXEC_TIMEOUT_SECONDS = 300  # 5 minutes per command

# Per-container resource caps. With workers=2, peak demand is ~2 * (SANDBOX_MEM_LIMIT +
# SANDBOX_TMP_SIZE), which must stay under the Docker VM memory; defaults target ~6g for an 8g VM.
# SANDBOX_MEMSWAP_LIMIT == SANDBOX_MEM_LIMIT disables swap so the cap is hard (a runaway test is
# OOM-killed in-container instead of spilling into host RAM). tmpfs is RAM-backed, hence capped too.
SANDBOX_MEM_LIMIT = os.environ.get("SANDBOX_MEM_LIMIT", "2g")
SANDBOX_MEMSWAP_LIMIT = os.environ.get("SANDBOX_MEMSWAP_LIMIT", SANDBOX_MEM_LIMIT)
SANDBOX_PIDS_LIMIT = int(os.environ.get("SANDBOX_PIDS_LIMIT", "1024"))
SANDBOX_CPUS = float(os.environ.get("SANDBOX_CPUS", "2"))
SANDBOX_TMP_SIZE = os.environ.get("SANDBOX_TMP_SIZE", "1g")
SANDBOX_SHM_SIZE = os.environ.get("SANDBOX_SHM_SIZE", "256m")


class DockerSandbox:
    def __init__(self, image=DEFAULT_IMAGE, platform=None, container_timeout=CONTAINER_TIMEOUT_SECONDS):
        self.image = image
        # Set for images that only exist for a non-host architecture (the Selenium image
        # is linux/amd64-only); None lets Docker pick the host-native variant.
        self.platform = platform
        # Agent-capture sessions (stage 6 agent mode) hold one container across many tasks and
        # need longer than the default 30 minutes; everything else keeps the default.
        self.container_timeout = container_timeout
        self.client = docker.from_env()
        self.container = None
        self.container_id = None
        self.repo_full_name = None
        self.start_time = None

    def start(self, repo_full_name: str) -> str:
        self.repo_full_name = repo_full_name
        safe_name = repo_full_name.replace("/", "_").replace(".", "_")

        create_kwargs = dict(
            image=self.image,
            name=f"sandbox_{safe_name}_{int(time.time())}",
            # Self-expiring + auto-removing: the container's main process sleeps exactly the
            # container timeout, then exits, and Docker removes it. Host-side cleanup (stop()
            # in finally blocks) is the normal path, but it never runs when the pipeline
            # process is SIGKILLed or crashes — this guarantees no orphan outlives its
            # timeout even with a dead owner.
            command=["sleep", str(self.container_timeout)],
            auto_remove=True,
            mem_limit=SANDBOX_MEM_LIMIT,
            memswap_limit=SANDBOX_MEMSWAP_LIMIT,
            pids_limit=SANDBOX_PIDS_LIMIT,
            shm_size=SANDBOX_SHM_SIZE,
            nano_cpus=int(SANDBOX_CPUS * 1_000_000_000),
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            tmpfs={"/tmp": f"size={SANDBOX_TMP_SIZE}"},
            detach=True,
            # /workspace is writable (not read-only for the whole container
            # because we need to clone and install deps)
        )
        if self.platform:
            create_kwargs["platform"] = self.platform
        self.container = self.client.containers.create(**create_kwargs)
        self.container.start()
        self.container_id = self.container.id
        self.start_time = time.time()
        logger.info(f"Started sandbox container {self.container_id[:12]} for {repo_full_name}")
        return self.container_id

    def remaining_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return self.container_timeout - (time.time() - self.start_time)

    def _check_timeout(self):
        if self.start_time and (time.time() - self.start_time) > self.container_timeout:
            cid = (self.container_id or "unknown")[:12]
            logger.error(f"Container {cid} exceeded {self.container_timeout}s timeout")
            self.stop()
            raise TimeoutError(f"Sandbox container exceeded {self.container_timeout}s timeout")

    def exec_command(self, command: str, timeout: int = EXEC_TIMEOUT_SECONDS,
                     max_output: int = 50000) -> tuple[str, str, int]:
        self._check_timeout()
        if self.container is None:
            raise RuntimeError("Sandbox container not started")

        try:
            # docker-py's exec_run has no timeout, so enforce it in-container with
            # coreutils `timeout` (SIGTERM at `timeout`s, SIGKILL 10s later). Without
            # this a hung command (e.g. a stuck browser) blocks until the 30-min
            # container timeout. A timed-out command exits 124.
            exec_result = self.container.exec_run(
                ["timeout", "-k", "10", str(timeout), "bash", "-c", command],
                workdir="/workspace",
                demux=True,
            )
            stdout = (exec_result.output[0] or b"").decode("utf-8", errors="replace") if exec_result.output else ""
            stderr = (exec_result.output[1] or b"").decode("utf-8", errors="replace") if exec_result.output else ""
            exit_code = exec_result.exit_code

            if len(stdout) > max_output:
                stdout = stdout[:max_output] + "\n... (truncated)"
            if len(stderr) > max_output:
                stderr = stderr[:max_output] + "\n... (truncated)"

            return stdout, stderr, exit_code
        except Exception as e:
            logger.error(f"exec_command failed: {e}")
            return "", str(e), 1

    def read_file(self, path: str) -> str:
        stdout, stderr, exit_code = self.exec_command(f"cat '{path}'")
        if exit_code != 0:
            raise FileNotFoundError(f"Cannot read {path}: {stderr}")
        return stdout

    def write_file(self, path: str, content: str):
        escaped_content = content.replace("'", "'\\''")
        _, stderr, exit_code = self.exec_command(
            f"mkdir -p $(dirname '{path}') && printf '%s' '{escaped_content}' > '{path}'"
        )
        if exit_code != 0:
            raise IOError(f"Cannot write {path}: {stderr}")

    def list_files(self, pattern: str = "*", directory: str = "/workspace") -> list[str]:
        stdout, _, exit_code = self.exec_command(
            f"find '{directory}' -maxdepth 5 -name '{pattern}' -type f 2>/dev/null | head -500"
        )
        if exit_code != 0:
            return []
        return [line for line in stdout.strip().split("\n") if line]

    def copy_out(self, container_path: str, host_path: str):
        host_dir = Path(host_path)
        host_dir.mkdir(parents=True, exist_ok=True)

        # `docker cp <container>:<dir>/. <host>` copies the directory contents
        # directly into host_dir.
        try:
            subprocess.run(
                ["docker", "cp", f"{self.container_id}:{container_path}/.", str(host_dir)],
                check=True,
                capture_output=True,
            )
            logger.info(f"Copied {container_path} → {host_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"copy_out failed: {e.stderr.decode()}")
            raise

    def stop(self):
        if self.container:
            try:
                self.container.remove(force=True)
                logger.info(f"Destroyed sandbox container {self.container_id[:12]}")
            except docker.errors.NotFound:
                pass  # auto_remove already reaped it (self-expired)
            except Exception as e:
                logger.warning(f"Failed to remove container: {e}")
            self.container = None
            self.container_id = None
            self.start_time = None

    def clone_repo(self, repo_full_name: str, branch: str = None) -> tuple[str, str, int]:
        branch_flag = f"--branch {branch}" if branch else ""
        return self.exec_command(
            f"git clone --depth=1 {branch_flag} https://github.com/{repo_full_name}.git /workspace/repo",
            timeout=120,
        )

    def install_node_deps(self) -> tuple[str, str, int]:
        return self.exec_command("cd /workspace/repo && npm install", timeout=300)

    def install_python_deps(self) -> tuple[str, str, int]:
        stdout, stderr, exit_code = self.exec_command(
            "cd /workspace/repo && pip3 install -r requirements.txt --break-system-packages 2>&1",
            timeout=300,
        )
        if exit_code != 0:
            stdout2, stderr2, exit_code2 = self.exec_command(
                "cd /workspace/repo && pip3 install -e . --break-system-packages 2>&1",
                timeout=300,
            )
            if exit_code2 == 0:
                return stdout2, stderr2, exit_code2
        return stdout, stderr, exit_code

    def start_background(self, command: str, log_path: str) -> tuple[str, str, int]:
        """Start a long-lived process inside the container that survives across exec calls.

        Each exec_command runs in an independent shell, so a plainly-backgrounded process dies
        with it. `setsid` + full stdio detachment re-parents the process to the container's
        PID 1, where it keeps running until the container is destroyed (there is deliberately
        no stop: containers are ephemeral). Used for the agent-capture app server and the
        browser driver.
        """
        escaped = command.replace("'", "'\\''")
        return self.exec_command(
            f"setsid nohup bash -c '{escaped}' > '{log_path}' 2>&1 < /dev/null & echo $!",
            timeout=30,
        )

    def wait_for_port(self, port: int, timeout_seconds: int = 60) -> bool:
        """Poll an in-container TCP port until it accepts connections (bash /dev/tcp, no curl)."""
        stdout, _, _ = self.exec_command(
            f"for i in $(seq 1 {timeout_seconds}); do "
            f"(exec 3<>/dev/tcp/127.0.0.1/{port}) 2>/dev/null && echo OPEN && break; sleep 1; done",
            timeout=timeout_seconds + 15,
        )
        return "OPEN" in stdout

    def http_json(self, method: str, url: str, payload: dict | None = None, timeout: int = 120) -> dict:
        """One JSON request to an in-container HTTP server, returning the parsed response.

        The payload travels base64-encoded through the exec shell so arbitrary text (quotes,
        newlines, unicode in agent thoughts) survives without quoting issues, in a single exec.
        """
        if payload is not None:
            encoded = base64.b64encode(json.dumps(payload).encode()).decode()
            cmd = (
                f"echo {encoded} | base64 -d | "
                f"curl -s -m {timeout - 5} -X {method} -H 'Content-Type: application/json' "
                f"--data-binary @- '{url}'"
            )
        else:
            cmd = f"curl -s -m {timeout - 5} -X {method} '{url}'"
        # Screenshot responses carry ~hundreds of KB of base64 — far past the default cap.
        stdout, stderr, exit_code = self.exec_command(cmd, timeout=timeout, max_output=4_000_000)
        if exit_code != 0:
            raise ConnectionError(f"http_json {method} {url} failed (exit {exit_code}): {stderr[:300]}")
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            raise ConnectionError(f"http_json {method} {url}: non-JSON response: {stdout[:300]}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @staticmethod
    def test_sandbox():
        logger.info("Testing sandbox...")
        sandbox = DockerSandbox()
        try:
            sandbox.start("test/sandbox-test")
            stdout, stderr, exit_code = sandbox.exec_command("whoami && echo 'Sandbox OK'")
            logger.info(f"whoami output: {stdout.strip()}")
            assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
            assert "Sandbox OK" in stdout, "Missing expected output"

            sandbox.write_file("/workspace/test.txt", "hello from sandbox")
            content = sandbox.read_file("/workspace/test.txt")
            assert content.strip() == "hello from sandbox", f"File content mismatch: {content}"

            files = sandbox.list_files("test.txt", "/workspace")
            assert any("test.txt" in f for f in files), f"File not found in listing: {files}"

            logger.info("Sandbox test PASSED")
            return True
        except Exception as e:
            logger.error(f"Sandbox test FAILED: {e}")
            return False
        finally:
            sandbox.stop()


def install_sigterm_handler():
    """pkill sends SIGTERM, which by default terminates Python WITHOUT running finally
    blocks — leaking every live sandbox until its self-expiry reaps it. Converting to
    SystemExit lets the normal cleanup paths run on graceful restarts. Best-effort:
    SIGKILL and crashes are covered by the containers' auto_remove self-expiry instead."""
    import signal

    def _terminate(signum, frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _terminate)


def build_sandbox_mcp_server(sandbox: DockerSandbox, server_name: str = "sandbox"):
    """Build an in-process MCP server whose tools execute inside `sandbox`'s
    Docker container. Returns (server, allowed_tool_names) for use with
    ClaudeAgentOptions(mcp_servers={server_name: server}, allowed_tools=...).

    The DockerSandbox methods are blocking (docker-py), so each tool runs them
    in an executor to keep the event loop responsive when stages run concurrent
    workers.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    async def _run(func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)

    @tool(
        "exec",
        "Run a bash command INSIDE the Docker container at /workspace. Use this to run "
        "pytest / npx playwright test, install dependencies, and inspect output. Returns "
        "exit_code, stdout and stderr. Each call is a fresh shell, so combine env vars and "
        "the command in one call (e.g. 'TRAJECTORY_DIR=/workspace/results npx playwright test').",
        {"command": str},
    )
    async def exec_tool(args):
        stdout, stderr, exit_code = await _run(sandbox.exec_command, args["command"])
        return {
            "content": [{"type": "text", "text": f"exit_code: {exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"}],
            "is_error": exit_code != 0,
        }

    @tool("read", "Read a file inside the container by absolute path.", {"path": str})
    async def read_tool(args):
        try:
            content = await _run(sandbox.read_file, args["path"])
            return {"content": [{"type": "text", "text": content}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": str(e)}], "is_error": True}

    @tool("write", "Write content to a file inside the container.", {"path": str, "content": str})
    async def write_tool(args):
        try:
            await _run(sandbox.write_file, args["path"], args["content"])
            return {"content": [{"type": "text", "text": "File written successfully"}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": str(e)}], "is_error": True}

    @tool(
        "ls",
        "List files matching a glob pattern inside the container. Optional 'directory' (default /workspace).",
        {"pattern": str, "directory": str},
    )
    async def ls_tool(args):
        files = await _run(sandbox.list_files, args.get("pattern", "*"), args.get("directory", "/workspace"))
        return {"content": [{"type": "text", "text": "\n".join(files) if files else "(no files found)"}]}

    server = create_sdk_mcp_server(
        name=server_name,
        version="1.0.0",
        tools=[exec_tool, read_tool, write_tool, ls_tool],
    )
    allowed = [f"mcp__{server_name}__{name}" for name in ("exec", "read", "write", "ls")]
    return server, allowed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    if "--test" in sys.argv:
        success = DockerSandbox.test_sandbox()
        sys.exit(0 if success else 1)
