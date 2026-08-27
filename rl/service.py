"""HTTP env service for the GPU-side trainer plugin: reset/step/close over the worker pool (loopback, token-guarded)."""
import argparse
import gzip
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from repo_pipeline.sandbox import install_sigterm_handler
from rl.episode import Episode, EpisodeConfig
from rl.judge import AsyncBridge
from rl.rollout import build_pool
from rl.runner import make_run_context
from rl.tasks import SUITE_PATH, load_suite, select_tasks

logger = logging.getLogger(__name__)
RL_ENV_HOST = os.environ.get("RL_ENV_HOST", "127.0.0.1")
RL_ENV_PORT = int(os.environ.get("RL_ENV_PORT", "8900"))
RL_ENV_TOKEN = os.environ.get("RL_ENV_TOKEN", "")
# Below the ingress's ~100 s origin timeout (Cloudflare): a longer server-side wait is cut off by the tunnel before the 503 reaches the client.
RL_RESET_TIMEOUT_S = float(os.environ.get("RL_RESET_TIMEOUT_S", "45"))
RL_EPISODE_IDLE_S = float(os.environ.get("RL_EPISODE_IDLE_S", "600"))


def _terminal_payload(record):
    return {"observation": None, "reward": record.get("reward"), "done": True, "info": record.get("info", {})}


class EnvService:
    def __init__(self, pool, tasks_by_id, run_ctx_factory, cfg, episode_cls=Episode, reset_timeout_s=RL_RESET_TIMEOUT_S):
        self.pool, self.tasks, self.run_ctx_factory, self.cfg = pool, tasks_by_id, run_ctx_factory, cfg
        self.episode_cls, self.reset_timeout_s = episode_cls, reset_timeout_s
        self.episodes = {}
        self.finished = {}
        self.reset_ids = {}
        self.lock = threading.Lock()

    def health(self):
        return {"ok": True, "free_slots": self.pool.free.qsize(), "live_episodes": len(self.episodes), "tasks": len(self.tasks)}

    def reset(self, body):
        request_id = body.get("request_id")
        replay = self._replay_reset(request_id)
        if replay is not None:
            return replay
        task = self.tasks[body["task_id"]]
        worker, slot = self.pool.acquire(timeout=self.reset_timeout_s)
        run_ctx = self.run_ctx_factory(body.get("run_id", "service"), int(body.get("iter", 0)), body.get("policy_tag", ""))
        episode = self.episode_cls(task, worker, slot, int(body.get("sample_k", 0)), run_ctx, self.cfg)
        try:
            observation = episode.reset()
        except Exception:
            self.pool.release(worker, slot)
            raise
        episode_id = uuid.uuid4().hex
        with self.lock:
            self.episodes[episode_id] = {"episode": episode, "lock": threading.Lock(), "worker": worker, "slot": slot, "last": time.time(),
                                         "observation0": observation, "responses": {}}
            if request_id:
                self.reset_ids[request_id] = episode_id
        return {"episode_id": episode_id, "observation": observation}

    def _replay_reset(self, request_id):
        with self.lock:
            episode_id = self.reset_ids.get(request_id) if request_id else None
            entry = self.episodes.get(episode_id) if episode_id else None
        return {"episode_id": episode_id, "observation": entry["observation0"]} if entry is not None else None

    def _entry(self, episode_id):
        with self.lock:
            entry = self.episodes.get(episode_id)
        if entry is None:
            raise KeyError(f"unknown or finished episode {episode_id}")
        return entry

    def _finish(self, episode_id, entry):
        with self.lock:
            live = self.episodes.pop(episode_id, None)
            self.finished[episode_id] = entry["episode"].record
        if live is not None:
            self.pool.release(entry["worker"], entry["slot"])

    def step(self, body):
        episode_id, seq = body["episode_id"], body.get("seq")
        finished = self.finished.get(episode_id)
        if finished is not None:
            return _terminal_payload(finished)
        entry = self._entry(episode_id)
        with entry["lock"]:
            if seq is not None and seq in entry["responses"]:
                return entry["responses"][seq]
            if getattr(entry["episode"], "closed", False):
                return _terminal_payload(entry["episode"].record or {})
            entry["last"] = time.time()
            observation, reward, done, info = entry["episode"].step(body["response_text"])
            response = {"observation": observation, "reward": reward, "done": done, "info": info}
            if seq is not None:
                entry["responses"] = {seq: response}
        if done:
            self._finish(episode_id, entry)
        return response

    def close(self, body):
        episode_id = body["episode_id"]
        finished = self.finished.get(episode_id)
        if finished is not None:
            return {"reward": finished.get("reward"), "info": finished.get("info", {})}
        entry = self._entry(episode_id)
        with entry["lock"]:
            record = entry["episode"].close(body.get("reason", "external"))
        self._finish(episode_id, entry)
        return {"reward": record.get("reward"), "info": record.get("info", {})}

    def episode(self, episode_id):
        record = self.finished.get(episode_id)
        if record is not None:
            return {"status": "finished", "reward": record.get("reward"), "breakdown": record.get("breakdown"), "episode_dir": record.get("traj_dir")}
        self._entry(episode_id)
        return {"status": "live"}

    def reap(self, max_idle_s=RL_EPISODE_IDLE_S):
        with self.lock:
            stale = [eid for eid, e in self.episodes.items() if time.time() - e["last"] > max_idle_s]
        for episode_id in stale:
            logger.warning(f"reaping idle episode {episode_id}")
            self.close({"episode_id": episode_id, "reason": "abandoned"})


def make_handler(service, token):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            # Observations are base64 PNGs, which gzip by ~25%; that margin matters against a metered ingress.
            if status == 200 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
                data = gzip.compress(data, 6)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _dispatch(self, method):
            if token and self.headers.get("X-RL-Token") != token:
                return self._send(401, {"error": "bad token"})
            path, _, query = self.path.partition("?")
            try:
                body = {}
                if method == "POST":
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length) or b"{}")
                if method == "GET" and path == "/health":
                    return self._send(200, service.health())
                if method == "GET" and path == "/tasks":
                    split = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("split")
                    return self._send(200, {"tasks": [t for t in service.tasks.values() if not split or t.get("split") == split]})
                if method == "GET" and path.startswith("/episode/"):
                    return self._send(200, service.episode(path[len("/episode/"):]))
                if method == "POST" and path == "/reset":
                    return self._send(200, service.reset(body))
                if method == "POST" and path == "/step":
                    return self._send(200, service.step(body))
                if method == "POST" and path == "/close":
                    return self._send(200, service.close(body))
                return self._send(404, {"error": f"no route {method} {path}"})
            except queue.Empty:
                return self._send(503, {"error": "no free slot", "retry_after": 30})
            except KeyError as e:
                return self._send(404, {"error": str(e)})
            except Exception as e:
                logger.exception(f"{method} {path} failed")
                return self._send(500, {"error": str(e)[:300]})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def log_message(self, fmt, *args):
            logger.debug(fmt % args)

    return Handler


def serve(service, host, port, token):
    return ThreadingHTTPServer((host, port), make_handler(service, token))


def _reap_forever(service, interval_s=60):
    while True:
        time.sleep(interval_s)
        service.reap()


def main():
    parser = argparse.ArgumentParser(description="RL env service")
    parser.add_argument("--tasks", default=str(SUITE_PATH))
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--drivers-per-sandbox", type=int, default=1)
    parser.add_argument("--backend", default="public_site")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--verifier-mode", default="fallback", choices=["always", "fallback", "never"])
    parser.add_argument("--run-id", default=time.strftime("service-%Y%m%d-%H%M%S"))
    parser.add_argument("--out", default="rl_runs")
    parser.add_argument("--host", default=RL_ENV_HOST)
    parser.add_argument("--port", type=int, default=RL_ENV_PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    install_sigterm_handler()
    suite = load_suite(args.tasks)
    tasks = {t["task_id"]: t for t in select_tasks(suite, split=args.split)}
    bridge = AsyncBridge()
    pool = build_pool(args)
    cfg = EpisodeConfig(max_steps=args.max_steps, verifier_mode=args.verifier_mode)

    def run_ctx_factory(run_id, iteration, tag):
        return make_run_context(run_id or args.run_id, args.out, iteration, policy_tag=tag, bridge=bridge)

    service = EnvService(pool, tasks, run_ctx_factory, cfg)
    server = serve(service, args.host, args.port, RL_ENV_TOKEN)
    try:
        pool.start()
        threading.Thread(target=_reap_forever, args=(service,), daemon=True).start()
        logger.info(f"env service on {args.host}:{args.port} with {pool.slot_count} slot(s), {len(tasks)} tasks (suite {suite['suite_sha256'][:12]})")
        server.serve_forever()
    finally:
        server.shutdown()
        pool.stop()
        bridge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
