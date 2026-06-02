import asyncio
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
    def __init__(self, image=DEFAULT_IMAGE, platform=None):
        self.image = image
        # Set for images that only exist for a non-host architecture (the Selenium image
        # is linux/amd64-only); None lets Docker pick the host-native variant.
        self.platform = platform
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
            command="sleep infinity",
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

    def _check_timeout(self):
        if self.start_time and (time.time() - self.start_time) > CONTAINER_TIMEOUT_SECONDS:
            cid = (self.container_id or "unknown")[:12]
            logger.error(f"Container {cid} exceeded {CONTAINER_TIMEOUT_SECONDS}s timeout")
            self.stop()
            raise TimeoutError(f"Sandbox container exceeded {CONTAINER_TIMEOUT_SECONDS}s timeout")

    def exec_command(self, command: str, timeout: int = EXEC_TIMEOUT_SECONDS) -> tuple[str, str, int]:
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

            if len(stdout) > 50000:
                stdout = stdout[:50000] + "\n... (truncated)"
            if len(stderr) > 50000:
                stderr = stderr[:50000] + "\n... (truncated)"

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
