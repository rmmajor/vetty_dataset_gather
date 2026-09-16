import asyncio
import base64
import io
import json
import logging
import os
import signal
import subprocess
import tarfile
import time
from pathlib import Path

import docker

from cua.settings import SANDBOX

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "repo-pipeline-sandbox:latest"
CONTAINER_TIMEOUT_SECONDS = 1800  # 30 minutes
EXEC_TIMEOUT_SECONDS = 300  # 5 minutes per command
# Past this size the payload streams in as a file: a longer exec argv trips the kernel's "argument list too long".
HTTP_INLINE_MAX = 100_000


def _parse_size(s) -> int:
    """'2g' / '512m' / '1.5g' / '1024' -> bytes. 0 on garbage (callers treat 0 as 'unknown')."""
    s = str(s).strip().lower()
    mult = 1
    if s and s[-1] in "gmkb":
        mult = {"g": 1024**3, "m": 1024**2, "k": 1024, "b": 1}[s[-1]]
        s = s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        return 0


def _safe_parallel(mem_total_bytes: int, requested: int, per_bytes: int, reserve_bytes: int) -> int:
    """How many sandboxes fit in RAM, clamped to [1, requested]; unknown memory leaves `requested` untouched."""
    if mem_total_bytes <= 0 or per_bytes <= 0:
        return requested
    return max(1, min(requested, int((mem_total_bytes - reserve_bytes) / per_bytes)))


def safe_parallel_sandboxes(requested: int, container_mem_limit: str | None = None) -> int:
    """Clamp concurrent sandboxes so peak RAM fits the Docker VM; unreadable VM memory leaves `requested` unchanged."""
    try:
        mem_total = docker.from_env().info().get("MemTotal", 0)
    except Exception as e:
        logger.warning(f"could not read Docker MemTotal ({e}); leaving parallelism at {requested}")
        return requested
    per = _parse_size(container_mem_limit or SANDBOX.mem_limit) + _parse_size(SANDBOX.tmp_size)
    reserve = _parse_size(SANDBOX.vm_reserve)
    safe = _safe_parallel(mem_total, requested, per, reserve)
    if safe < requested:
        logger.warning(
            f"reducing sandbox parallelism {requested} -> {safe}: Docker VM has "
            f"{mem_total / 1024**3:.1f}g, each sandbox needs ~{per / 1024**3:.1f}g + "
            f"{reserve / 1024**3:.1f}g reserve. Raise the VM's memory or SANDBOX_VM_RESERVE to tune."
        )
    return safe


class ShellSandbox:
    """Everything derivable from `exec_command` + `put_bytes`, shared by every sandbox backend."""

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
            raise OSError(f"Cannot write {path}: {stderr}")

    def start_background(self, command: str, log_path: str) -> tuple[str, str, int]:
        """Start a long-lived process that survives across exec calls."""
        # Each exec runs its own shell, so only setsid plus full stdio detachment re-parents the process to PID 1.
        escaped = command.replace("'", "'\\''")
        return self.exec_command(
            f"setsid nohup bash -c '{escaped}' > '{log_path}' 2>&1 < /dev/null & echo $!",
            timeout=30,
        )

    def wait_for_port(self, port: int, timeout_seconds: int = 60) -> bool:
        """Poll a TCP port until it accepts connections (bash /dev/tcp, no curl)."""
        stdout, _, _ = self.exec_command(
            f"for i in $(seq 1 {timeout_seconds}); do "
            f"(exec 3<>/dev/tcp/127.0.0.1/{port}) 2>/dev/null && echo OPEN && break; sleep 1; done",
            timeout=timeout_seconds + 15,
        )
        return "OPEN" in stdout

    def http_json(self, method: str, url: str, payload: dict | None = None, timeout: int = 120) -> dict:
        """One JSON request to a sandbox-local HTTP server, returning the parsed response."""
        # base64 through the exec shell, so quotes, newlines and unicode in agent thoughts survive one exec unquoted.
        tmp_path = None
        if payload is not None:
            raw = json.dumps(payload).encode()
            encoded = base64.b64encode(raw).decode()
            if len(encoded) > HTTP_INLINE_MAX:
                # /workspace, not /tmp: put_archive writes the overlay layer, which a tmpfs mount hides from curl.
                tmp_path = f"/workspace/.httpreq_{base64.urlsafe_b64encode(os.urandom(9)).decode()}.json"
                self.put_bytes(tmp_path, raw)
                cmd = (
                    f"curl -s -m {timeout - 5} -X {method} -H 'Content-Type: application/json' "
                    f"--data-binary @{tmp_path} '{url}'; rc=$?; rm -f {tmp_path}; exit $rc"
                )
            else:
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
        except json.JSONDecodeError as err:
            raise ConnectionError(f"http_json {method} {url}: non-JSON response: {stdout[:300]}") from err

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class DockerSandbox(ShellSandbox):
    def __init__(self, image=DEFAULT_IMAGE, platform=None, container_timeout=CONTAINER_TIMEOUT_SECONDS):
        self.image = image
        # Only for an image that exists for a non-host architecture; None lets Docker pick the host-native variant.
        self.platform = platform
        # Agent capture holds one container across many tasks and needs longer than the default.
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
            # Self-expiring and auto-removing, so a SIGKILLed owner cannot leak the container past its timeout.
            command=["sleep", str(self.container_timeout)],
            auto_remove=True,
            mem_limit=SANDBOX.mem_limit,
            memswap_limit=SANDBOX.memswap_limit,
            pids_limit=SANDBOX.pids_limit,
            shm_size=SANDBOX.shm_size,
            nano_cpus=int(SANDBOX.cpus * 1_000_000_000),
            cap_drop=["ALL"],
            # Dependency installs fchown and chmod; without these caps node_modules lands empty and boot fails.
            cap_add=["CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID", "SETUID", "SETGID"],
            security_opt=["no-new-privileges:true"],
            tmpfs={"/tmp": f"size={SANDBOX.tmp_size}"},
            detach=True,
            # No read_only: /workspace has to stay writable for the clone and the dependency install.
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

    def exec_command(
        self, command: str, timeout: int = EXEC_TIMEOUT_SECONDS, max_output: int = 50000
    ) -> tuple[str, str, int]:
        self._check_timeout()
        if self.container is None:
            raise RuntimeError("Sandbox container not started")

        try:
            # docker-py's exec_run has no timeout, so coreutils `timeout` enforces one in-container; it exits 124.
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

    def put_bytes(self, container_path: str, data: bytes):
        """Stream raw bytes into the container through the Docker archive API, never across the exec command line."""
        self._check_timeout()
        if self.container is None:
            raise RuntimeError("Sandbox container not started")
        directory = os.path.dirname(container_path) or "/"
        name = os.path.basename(container_path)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        self.exec_command(f"mkdir -p '{directory}'")
        if not self.container.put_archive(directory, buf.getvalue()):
            raise OSError(f"put_archive failed for {container_path}")

    def list_files(self, pattern: str = "*", directory: str = "/workspace") -> list[str]:
        # Prune node_modules and .git first: thousands of vendored package.json files would spend the head -500 cap.
        stdout, _, exit_code = self.exec_command(
            f"find '{directory}' -maxdepth 5 \\( -name node_modules -o -name .git \\) -prune -o "
            f"-name '{pattern}' -type f -print 2>/dev/null | head -500"
        )
        if exit_code != 0:
            return []
        return [line for line in stdout.strip().split("\n") if line]

    def copy_out(self, container_path: str, host_path: str):
        if self.container_id is None:
            raise RuntimeError("Sandbox container not started — nothing to copy out")
        host_dir = Path(host_path)
        host_dir.mkdir(parents=True, exist_ok=True)

        # The trailing `/.` copies the directory CONTENTS into host_dir rather than the directory itself.
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
    """Turn SIGTERM into SystemExit so a pkill'd run still runs the finally blocks that stop its sandboxes."""

    def _terminate(signum, frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _terminate)


def build_sandbox_mcp_server(sandbox: DockerSandbox, server_name: str = "sandbox"):
    """An in-process MCP server whose tools run inside `sandbox`, as (server, allowed_tool_names)."""
    # Keeps the SDK out of the import graph of the callers that never build a server
    from claude_agent_sdk import create_sdk_mcp_server, tool

    # docker-py blocks, so every tool hops to an executor and leaves the event loop free for concurrent workers.
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
