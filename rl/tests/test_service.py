import json
import queue
import threading
import urllib.error
import urllib.request

from rl.service import EnvService, serve


class FakePool:
    def __init__(self, slots=1):
        self.free = queue.Queue()
        for i in range(slots):
            self.free.put(("w", i))
        self.slot_count = slots

    def acquire(self, timeout=None):
        return self.free.get(timeout=timeout)

    def release(self, worker, slot):
        self.free.put((worker, slot))


class FakeEpisode:
    def __init__(self, task, worker, slot, k, ctx, cfg):
        self.episode_id, self.n, self.record = f"{task['task_id']}__k{k}", 0, None

    def reset(self):
        return {"text": "obs0", "image_b64": "", "system": "s", "step_index": 0}

    def step(self, text):
        self.n += 1
        if text == "action: done()":
            self.record = {"reward": 1.0, "info": {"reward_status": "final"}}
            return None, 1.0, True, {"terminal_reason": "done"}
        return {"text": f"obs{self.n}", "image_b64": "", "system": "s", "step_index": self.n}, None, False, {"parse": "ok"}

    def close(self, reason="external"):
        self.record = self.record or {"reward": 0.0, "info": {"reward_status": "final", "terminal_reason": reason}}
        return self.record


def _get(url, token="t"):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"X-RL-Token": token}), timeout=10) as response:
        return json.loads(response.read())


def _post(url, body, token="t"):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "X-RL-Token": token})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def test_http_round_trip_and_slot_exhaustion():
    tasks = {"t1": {"task_id": "t1", "split": "train"}}
    service = EnvService(FakePool(1), tasks, lambda run_id, it, tag: None, None, episode_cls=FakeEpisode, reset_timeout_s=0.2)
    server = serve(service, "127.0.0.1", 0, "t")
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        health = _get(f"{base}/health")
        assert health["ok"] and health["free_slots"] == 1
        assert _get(f"{base}/tasks?split=train")["tasks"] == [tasks["t1"]]
        first = _post(f"{base}/reset", {"task_id": "t1", "run_id": "r", "iter": 0, "sample_k": 0, "request_id": "req-1"})
        assert first["observation"]["text"] == "obs0"
        assert _post(f"{base}/reset", {"task_id": "t1", "request_id": "req-1"}) == first
        try:
            _post(f"{base}/reset", {"task_id": "t1"})
            raise AssertionError("second reset must 503")
        except urllib.error.HTTPError as e:
            assert e.code == 503 and json.loads(e.read())["retry_after"] > 0
        stepped = _post(f"{base}/step", {"episode_id": first["episode_id"], "response_text": "action: wait()", "seq": 1})
        assert stepped["done"] is False and stepped["observation"]["text"] == "obs1"
        assert _post(f"{base}/step", {"episode_id": first["episode_id"], "response_text": "action: wait()", "seq": 1}) == stepped
        final = _post(f"{base}/step", {"episode_id": first["episode_id"], "response_text": "action: done()", "seq": 2})
        assert final["done"] is True and final["reward"] == 1.0 and final["observation"] is None
        replayed = _post(f"{base}/step", {"episode_id": first["episode_id"], "response_text": "action: done()", "seq": 2})
        assert replayed["done"] is True and replayed["reward"] == 1.0
        assert _post(f"{base}/close", {"episode_id": first["episode_id"]})["reward"] == 1.0
        assert _get(f"{base}/episode/{first['episode_id']}")["reward"] == 1.0
        assert service.pool.free.qsize() == 1
        try:
            _post(f"{base}/reset", {"task_id": "t1"}, token="wrong")
            raise AssertionError("bad token must 401")
        except urllib.error.HTTPError as e:
            assert e.code == 401
    finally:
        server.shutdown()
