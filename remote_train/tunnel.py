"""SSH tunnel resource for remote jobs: -R exposes the Mac env service to the box, -L brings the box's serve port here; supervised like autossh."""
import logging
import subprocess
import threading
import time

logger = logging.getLogger(__name__)


class NullTunnel:
    def __init__(self, forwards=()):
        self.forwards = list(forwards)
        self.alive = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class SshTunnel:
    def __init__(self, ssh_argv_base, forwards, keepalive_s=15, popen=subprocess.Popen, sleep=time.sleep):
        self.base, self.forwards, self.keepalive_s = list(ssh_argv_base), list(forwards), keepalive_s
        self.popen, self.sleep = popen, sleep
        self.alive = False
        self.proc = None
        self.thread = None

    def argv(self):
        target = self.base[-1]
        argv = [*self.base[:-1], "-N", "-o", f"ServerAliveInterval={self.keepalive_s}", "-o", "ServerAliveCountMax=4",
                "-o", "ExitOnForwardFailure=yes"]
        for direction, port in self.forwards:
            argv += [f"-{direction}", f"127.0.0.1:{port}:127.0.0.1:{port}"]
        return [*argv, target]

    def _supervise(self):
        attempt = 0
        while self.alive:
            try:
                self.proc = self.popen(self.argv(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                while self.alive and self.proc.poll() is None:
                    self.sleep(1)
            except Exception as e:
                # A dead supervisor leaves alive=True, so every env call silently becomes reward 0.0.
                logger.warning(f"ssh tunnel supervisor could not spawn ssh: {e!r}")
            if not self.alive:
                break
            attempt += 1
            logger.warning(f"ssh tunnel exited (rc={self.proc.poll() if self.proc else 'spawn-failed'}); restarting in {min(60, 5 * attempt)}s")
            self.sleep(min(60, 5 * attempt))

    def start(self):
        self.alive = True
        self.thread = threading.Thread(target=self._supervise, name="ssh-tunnel", daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.alive = False
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
        return False
