import json
import tempfile
from pathlib import Path

from repo_pipeline.driver_session import DriverError
from rl.backends import EnvBackend
import rl.runner
from rl.episode import Episode, EpisodeConfig, RunContext
from rl.rewards import load_reward_config


class FakeSandbox:
    def __init__(self, traj_dir):
        self.traj_dir = traj_dir
        self.commands = []

    def copy_out(self, container_path, host_path):
        Path(host_path).mkdir(parents=True, exist_ok=True)
        for p in Path(self.traj_dir).iterdir():
            (Path(host_path) / p.name).write_bytes(p.read_bytes())

    def exec_command(self, cmd, **kw):
        self.commands.append(cmd)
        return "", "", 0


class FakeClient:
    """Mimics the driver: records one step per act_sequence into a fake results dir."""

    def __init__(self, results_dir, refuse_after=None):
        self.dir = Path(results_dir)
        self.steps = []
        self.calls = []
        self.refuse_after = refuse_after

    def session(self, task_id, start_url, **kw):
        self.calls.append(("session", task_id, kw))
        self.dir.mkdir(parents=True, exist_ok=True)
        return {"ok": True}

    def observe(self):
        return {"ok": True, "b64": "UE5H", "url": "http://x/", "title": "t", "step_next": len(self.steps) + 1}

    def act_sequence(self, thought, actions):
        if self.refuse_after is not None and len(self.steps) >= self.refuse_after:
            raise DriverError("loop guard: the same action was attempted 3 times in a row with no progress.")
        n = len(self.steps) + 1
        name = f"step_{n:03d}_screenshot.png"
        (self.dir / name).write_bytes(b"PNG")
        self.steps.append({"step": n, "screenshot": name, "thought": thought, "actions": actions})
        self._write()
        return {"ok": True, "step": n, "url": "http://x/", "executed": len(actions), "error_at": None, "error": None}

    def snapshot(self):
        return {"ok": True, "url": "http://x/done", "title": "t", "snapshot": "## Page text\nall good\n"}

    def page_text(self):
        return "computed answer 36"

    def storage_state(self):
        return {"cookies": []}

    def finish(self, outcome, summary):
        self._write()
        (self.dir / "final_screenshot.png").write_bytes(b"PNG")
        return {"ok": True, "steps": len(self.steps), "dir": str(self.dir)}

    def _write(self):
        (self.dir / "trajectory.json").write_text(json.dumps({"task_id": "t", "instruction": "i", "start_url": "u", "steps": self.steps, "total_steps": len(self.steps), "success": False}))
        (self.dir / "capture_meta.json").write_text(json.dumps({"recorder": "browser_driver", "steps": {}}))


class FakeWorker:
    def __init__(self, client, sandbox):
        self._client, self.sandbox, self.backend = client, sandbox, EnvBackend()

    def client(self, slot):
        return self._client


TASK = {"task_id": "site__t1", "site": "site", "split": "train", "start_url": "http://x/", "instruction": "Do the thing", "difficulty": "easy",
        "expected_actions": 2, "backend": "static_page", "env": "test", "lane": "autonomous", "auth_mode": "none", "idempotent": True,
        "preconditions": {}, "allowed_hosts": [], "success_check": {"text_visible": "all good", "done_emitted": True}}


def _episode(tmp, refuse_after=None, task=None, **cfg):
    results = Path(tmp) / "results" / "task_x"
    client = FakeClient(results, refuse_after)
    worker = FakeWorker(client, FakeSandbox(results))
    ctx = RunContext(run_id="r", out_root=Path(tmp) / "rl_runs", iteration=1, reward_config=load_reward_config())
    return Episode(task or TASK, worker, 0, 2, ctx, EpisodeConfig(**{"max_steps": 4, "verifier_mode": "never", **cfg})), client


def test_reset_observation_and_ids():
    with tempfile.TemporaryDirectory() as tmp:
        ep, client = _episode(tmp)
        obs = ep.reset()
        assert ep.episode_id == "site__t1__i001__k02" and obs["text"] == "Task: Do the thing\nWhat action should you take?"
        assert client.calls[0][2]["task_meta"]["expected_actions"] == 2 and client.calls[0][2]["task_meta"]["autonomy"] == "autonomous"


def test_full_episode_records_steps_history_and_reward():
    with tempfile.TemporaryDirectory() as tmp:
        ep, client = _episode(tmp)
        ep.reset()
        obs, reward, done, info = ep.step("<think>c</think>\naction: click(<ref>(10, 10)</ref>, button=left)\naction: type(\"hi\")")
        assert not done and reward is None and info["parse"] == "ok" and info["executed"] == 2
        assert obs["text"] == 'Task: Do the thing\nPrevious actions: [click(8, 14), type("hi")]\nWhat action should you take?'
        obs, reward, done, info = ep.step("garbage")
        assert not done and info["parse"] == "failure" and len(client.steps) == 1
        obs, reward, done, info = ep.step("<think>d</think>\naction: done()")
        assert done and obs is None and info["terminal_reason"] == "done" and reward > 1.0
        rec = ep.record
        assert rec["success"] is True and rec["success_source"] == "success_check" and rec["ui_steps"] == 1 and rec["steps"] == 2
        assert [t["clean"] for t in rec["turns"]] == [True, False, True]
        traj = json.loads((ep.traj_dir / "trajectory.json").read_text())
        assert [a["type"] for a in traj["steps"][0]["actions"]] == ["click", "type"]
        meta = json.loads((ep.traj_dir / "metadata.json").read_text())
        assert meta["excluded_reason"] == "rl_rollout" and meta["source_repo"] == "rl:site__t1" and meta["autonomy"] == "autonomous"
        assert json.loads((ep.traj_dir / "capture_meta.json").read_text())["success_check"]["passed"] is True
        assert (ep.traj_dir / "episode.json").exists()


def test_parse_abort_after_three_failures():
    with tempfile.TemporaryDirectory() as tmp:
        ep, _ = _episode(tmp)
        ep.reset()
        for _ in range(2):
            _, _, done, _ = ep.step("nope")
            assert not done
        _, reward, done, info = ep.step("nope")
        assert done and info["terminal_reason"] == "parse_abort" and reward < 0


def test_step_cap_and_control_unexpected_and_loop_guard():
    with tempfile.TemporaryDirectory() as tmp:
        ep, _ = _episode(tmp, max_steps=1)
        ep.reset()
        _, _, done, info = ep.step("action: wait()")
        assert done and info["terminal_reason"] == "step_cap"
    with tempfile.TemporaryDirectory() as tmp:
        ep, _ = _episode(tmp)
        ep.reset()
        _, _, done, info = ep.step("action: handoff(reason=login)")
        assert done and info["terminal_reason"] == "control_unexpected"
    with tempfile.TemporaryDirectory() as tmp:
        ep, _ = _episode(tmp, refuse_after=0)
        ep.reset()
        _, _, done, info = ep.step("action: wait()")
        assert done and info["terminal_reason"] == "loop_guard" and info["refusal"] == "loop_guard"


def test_navigate_off_host_is_blocked_not_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        ep, client = _episode(tmp)
        ep.reset()
        obs, _, done, info = ep.step('action: navigate("https://evil.example/")')
        assert not done and info["refusal"] == "navigate_blocked" and client.steps == [] and obs is not None
        _, _, done, _ = ep.step("action: done()")
        assert done and ep.record["navigate_blocked"] == 1


def test_navigate_same_registrable_domain_is_allowed():
    with tempfile.TemporaryDirectory() as tmp:
        ep, client = _episode(tmp, task={**TASK, "start_url": "https://www.example.com/"})
        ep.reset()
        _, _, done, info = ep.step('<think>t</think>\naction: navigate("https://checkout.example.com/cart")')
        assert not done and "refusal" not in info and info["executed"] == 1 and len(client.steps) == 1
        _, _, done, info = ep.step('action: navigate("https://example.co.uk/")')
        assert not done and info["refusal"] == "navigate_blocked" and len(client.steps) == 1


def test_close_external_marks_abandoned():
    with tempfile.TemporaryDirectory() as tmp:
        ep, _ = _episode(tmp)
        ep.reset()
        rec = ep.close("abandoned")
        assert rec["terminal_reason"] == "abandoned" and ep.close() is rec


class OneSlotPool:
    slot_count = 1


def test_run_batch_gives_every_task_sample_n_before_any_task_gets_n_plus_one():
    tasks = [{"task_id": "a"}, {"task_id": "b"}, {"task_id": "c"}]
    seen = []
    original = rl.runner.run_episode
    rl.runner.run_episode = lambda policy, task, pool, sample, ctx, cfg: seen.append((task["task_id"], sample)) or {}
    try:
        rl.runner.run_batch(lambda: None, tasks, 3, OneSlotPool(), None, None)
    finally:
        rl.runner.run_episode = original
    assert seen[:3] == [("a", 0), ("b", 0), ("c", 0)]
    assert seen[3:6] == [("a", 1), ("b", 1), ("c", 1)]
    assert len(seen) == 9


class DeadPool:
    """One slot whose sandbox has vanished, as after an OOM kill and auto-remove."""

    slot_count = 1

    def acquire(self, timeout=None):
        return DeadWorker(), 0

    def release(self, worker, slot):
        pass


class DeadWorker:
    worker_id = 0
    sandbox = None

    def client(self, slot):
        raise ConnectionError("No such container")


def test_a_vanished_sandbox_fails_one_episode_not_the_batch():
    records = rl.runner.run_batch(lambda: None, [TASK], 2, DeadPool(),
                                  RunContext(run_id="r", out_root=Path("/tmp/unused"), iteration=0, reward_config=load_reward_config()),
                                  EpisodeConfig(max_steps=2, verifier_mode="never"))
    assert len(records) == 2
    assert all(r["terminal_reason"] == "harness_error" and r["reward"] is None for r in records)

