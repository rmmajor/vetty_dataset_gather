"""Environment backends (registry) and the sandbox worker pool that hosts rollout browsers."""
import logging
import os
import queue
import threading

from repo_pipeline import app_secrets, driver_session
from repo_pipeline.driver_session import DRIVER_PORT, DriverClient
from repo_pipeline.sandbox import DockerSandbox, safe_parallel_sandboxes

logger = logging.getLogger(__name__)
RL_CONTAINER_TIMEOUT = int(os.environ.get("RL_CONTAINER_TIMEOUT", str(4 * 3600)))
RL_RECYCLE_RESERVE_S = int(os.environ.get("RL_RECYCLE_RESERVE_S", "1200"))
RL_DRIVERS_PER_SANDBOX = int(os.environ.get("RL_DRIVERS_PER_SANDBOX", "1"))
# The capture-time guard (3) ended two thirds of RL episodes; 5 lets a recoverable retry survive while a true loop still terminates.
RL_LOOP_GUARD_REPEATS = int(os.environ.get("RL_LOOP_GUARD_REPEATS", "5"))


def resolve_preconditions(task):
    """FAIL-CLOSED session kwargs: {} when no auth is needed, None when a required burner/session is missing."""
    pre = task.get("preconditions") or {}
    if not pre.get("requires_session"):
        return {}
    burner_id = pre.get("burner_account_id")
    if not app_secrets.is_registered_burner(burner_id):
        logger.warning(f"[{task['task_id']}] FAIL-CLOSED: burner '{burner_id}' is not registered — skipping")
        return None
    storage_state = app_secrets.load_storage_state(pre.get("session_ref"))
    if storage_state is None:
        logger.warning(f"[{task['task_id']}] FAIL-CLOSED: storage_state '{pre.get('session_ref')}' missing — skipping")
        return None
    kwargs = {"auth_config": {"auth_mode": "pre_session", "storage_state": storage_state}}
    if task.get("lane", "autonomous") == "autonomous":
        burner = app_secrets.burner_account(burner_id)
        kwargs["secrets"] = {"email": burner["email"], "password": burner["password"]}
    return kwargs


class EnvBackend:
    key = ""

    def open(self, sandbox):
        pass

    def start_url(self, task):
        return task["start_url"]

    def pre_episode(self, sandbox, task):
        pass

    def session_kwargs(self, task):
        return resolve_preconditions(task)

    def post_episode(self, sandbox, task, container_task_dir):
        sandbox.exec_command(f"rm -rf {container_task_dir}")


class PublicSiteBackend(EnvBackend):
    key = "public_site"


class StaticPageBackend(EnvBackend):
    """Serves one in-container HTML page — the no-network backend for smoke tests."""
    key = "static_page"
    port = 3000

    def __init__(self, html):
        self.html = html

    def open(self, sandbox):
        sandbox.exec_command("mkdir -p /workspace/site")
        sandbox.write_file("/workspace/site/index.html", self.html)
        sandbox.start_background(f"cd /workspace/site && python3 -m http.server {self.port}", "/workspace/site.log")
        if not sandbox.wait_for_port(self.port, 30):
            raise RuntimeError("static page server did not start")

    def start_url(self, task):
        return f"http://127.0.0.1:{self.port}/"

    def session_kwargs(self, task):
        return {}


BACKENDS = {PublicSiteBackend.key: PublicSiteBackend, StaticPageBackend.key: StaticPageBackend}


class RolloutWorker:
    """One sandbox hosting N driver slots; recycled (when idle) before the container self-expires."""

    def __init__(self, worker_id, backend, drivers_per_sandbox=RL_DRIVERS_PER_SANDBOX, driver_env=None):
        self.worker_id = worker_id
        self.backend = backend
        self.drivers_per_sandbox = drivers_per_sandbox
        self.driver_env = driver_env or {}
        self.sandbox = None
        self.clients = []
        self.live = 0
        self.lock = threading.Lock()

    def start(self):
        self.sandbox = DockerSandbox(container_timeout=RL_CONTAINER_TIMEOUT)
        self.sandbox.start(f"rl/worker-{self.worker_id}")
        self.backend.open(self.sandbox)
        driver_session.install_driver(self.sandbox)
        self.clients = []
        for i in range(self.drivers_per_sandbox):
            driver_session.start_driver(self.sandbox, env=self.driver_env, port=DRIVER_PORT + i)
            self.clients.append(DriverClient(self.sandbox, port=DRIVER_PORT + i))
        logger.info(f"worker {self.worker_id}: sandbox up with {len(self.clients)} driver slot(s)")

    def client(self, slot):
        return self.clients[slot]

    def needs_recycle(self):
        if self.sandbox is None or self.sandbox.container is None or self.sandbox.remaining_seconds() < RL_RECYCLE_RESERVE_S:
            return True
        try:
            return not all(c.status().get("ok") for c in self.clients)
        except Exception:
            return True

    def recycle(self):
        self.stop()
        self.start()

    def stop(self):
        if self.sandbox is not None:
            self.sandbox.stop()
        self.sandbox = None
        self.clients = []


class WorkerPool:
    """Free-slot queue over N workers; a slot (worker, index) hosts exactly one live episode."""

    def __init__(self, backend, requested_workers, drivers_per_sandbox=RL_DRIVERS_PER_SANDBOX, loop_guard_repeats=RL_LOOP_GUARD_REPEATS):
        self.loop_guard_repeats = loop_guard_repeats
        driver_env = {"DRIVER_LOOP_GUARD_REPEATS": str(loop_guard_repeats)}
        self.workers = [RolloutWorker(i, backend, drivers_per_sandbox, driver_env) for i in range(safe_parallel_sandboxes(requested_workers))]
        self.free = queue.Queue()

    @property
    def slot_count(self):
        return sum(w.drivers_per_sandbox for w in self.workers)

    def start(self):
        for worker in self.workers:
            worker.start()
            for slot in range(worker.drivers_per_sandbox):
                self.free.put((worker, slot))

    def acquire(self, timeout=None):
        worker, slot = self.free.get(timeout=timeout)
        with worker.lock:
            if worker.live == 0 and worker.needs_recycle():
                worker.recycle()
            worker.live += 1
        return worker, slot

    def release(self, worker, slot):
        with worker.lock:
            worker.live -= 1
        self.free.put((worker, slot))

    def stop(self):
        for worker in self.workers:
            worker.stop()
