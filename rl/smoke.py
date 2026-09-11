"""Docker-only end-to-end check of the RL harness: static page + scripted policy through Episode and the HTTP service, no LLM."""

import base64
import json
import logging
import shutil
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

from repo_pipeline.corpus_audit import audit_trajectory
from repo_pipeline.driver_smoke import TEST_PAGE
from repo_pipeline.instrumentation import RESULTS_DIR
from rl.backends import StaticPageBackend, WorkerPool
from rl.episode import EpisodeConfig
from rl.policies import ScriptedPolicy
from rl.runner import make_run_context, run_episode
from rl.service import EnvService, serve
from train_qwen import convert_trajectory_to_examples

logger = logging.getLogger(__name__)

TASK = {
    "task_id": "smoke__greet",
    "site": "smoke",
    "split": "train",
    "start_url": "http://127.0.0.1:3000/",
    "instruction": "Greet yourself by name",
    "difficulty": "easy",
    "expected_actions": 3,
    "backend": "static_page",
    "env": "test",
    "lane": "autonomous",
    "auth_mode": "none",
    "idempotent": True,
    "preconditions": {},
    "allowed_hosts": [],
    "success_check": {"text_visible": "hello Smokey", "done_emitted": True},
}


def discover_coords(client, needles):
    """Resolve page elements to pixel centers through the teacher's ref path (a throwaway probe session)."""
    session = client.session("smoke_probe", TASK["start_url"], instruction="probe")
    coords = {}
    for needle in needles:
        ref = next(
            int(line[1 : line.index("]")])
            for line in session["snapshot"].splitlines()
            if line.startswith("[") and needle in line
        )
        client.act({"kind": "click", "ref": ref, "thought": f"probe {needle}"})
        recorded = json.loads(client.sandbox.read_file(f"{RESULTS_DIR}/task_smoke_probe/trajectory.json"))
        action = recorded["steps"][-1]["actions"][0]
        coords[needle] = (action["x"], action["y"])
    client.finish("failed", "probe")
    client.sandbox.exec_command(f"rm -rf {RESULTS_DIR}/task_smoke_probe")
    return coords


def script(coords):
    (nx, ny), (gx, gy) = coords["Your name"], coords["Greet"]
    return [
        f'<think>I will type my name into the field.</think>\naction: click(<ref>({nx}, {ny})</ref>, button=left)\naction: type("Smokey")',
        "this is not an action",
        f"<think>Submitting the greeting.</think>\naction: click(<ref>({gx}, {gy})</ref>, button=left)",
        "<think>Checking below the fold.</think>\naction: scroll(<ref>(640, 360)</ref>, scrollX=0, scrollY=300)",
        '<think>Back to the top.</think>\naction: keypress(["Home"])',
        "<think>The page greets me as Smokey; the task is complete.</think>\naction: done()",
    ]


class RecordingPolicy(ScriptedPolicy):
    def __init__(self, texts):
        super().__init__(texts)
        self.observations = []

    def act(self, observation):
        self.observations.append(observation)
        return super().act(observation)


def check_record(record, policy, traj_dir):
    assert record["terminal_reason"] == "done", record["terminal_reason"]
    assert record["check"]["passed"] and record["success"] is True and record["success_source"] == "success_check", (
        record["check"]
    )
    assert [t["clean"] for t in record["turns"]] == [True, False, True, True, True, True], record["turns"]
    assert abs(record["breakdown"]["components"]["format_validity"] - 5 / 6) < 1e-6
    assert record["reward"] > 1.0, record["reward"]
    traj = json.loads((traj_dir / "trajectory.json").read_text())
    assert [a["type"] for a in traj["steps"][0]["actions"]] == ["click", "type"], traj["steps"][0]["actions"]
    assert len(traj["steps"]) == 5 and traj["steps"][-1]["actions"] == [{"type": "done"}]
    emitted = [policy.observations[i] for i in (0, 2, 3, 4, 5)]
    for step, obs in zip(traj["steps"], emitted, strict=False):
        assert (traj_dir / step["screenshot"]).read_bytes() == base64.b64decode(obs["image_b64"]), (
            f"PNG parity broken at step {step['step']}"
        )
    examples, _ = convert_trajectory_to_examples(traj, traj_dir)
    # The RL lane stays on the one-image v2 layout by decision, so parity is asserted on the task/history text only.
    for example, obs in zip(examples, emitted, strict=False):
        assert example["messages"][1]["content"][-1]["text"].removeprefix("\n") == obs["text"], (
            "SFT input != RL observation"
        )
    defects = audit_trajectory(traj_dir)
    assert defects == [], defects
    assert json.loads((traj_dir / "metadata.json").read_text())["excluded_reason"] == "rl_rollout"
    logger.info("episode assertions PASSED")


def run_service_round_trip(pool, cfg, out_root, texts):
    service = EnvService(
        pool, {TASK["task_id"]: TASK}, lambda run_id, it, tag: make_run_context(run_id, out_root, it, tag), cfg
    )
    server = serve(service, "127.0.0.1", 0, "smoke")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, body):
        request = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-RL-Token": "smoke"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())

    try:
        first = post("/reset", {"task_id": TASK["task_id"], "run_id": "smoke-http", "iter": 1, "sample_k": 3})
        assert first["observation"]["text"].startswith("Task: Greet yourself by name")
        step = None
        for text in texts:
            step = post("/step", {"episode_id": first["episode_id"], "response_text": text})
            if step["done"]:
                break
        assert step and step["done"] and step["reward"] is not None and step["reward"] > 1.0, step
        assert (out_root / "iter_001" / TASK["task_id"] / "k00").exists(), "the service numbers k itself"
        logger.info("service round-trip PASSED")
    finally:
        server.shutdown()


def run_smoke():
    out_root = Path(tempfile.mkdtemp(prefix="rl_smoke_"))
    pool = WorkerPool(StaticPageBackend(TEST_PAGE), requested_workers=1, drivers_per_sandbox=1)
    try:
        pool.start()
        coords = discover_coords(pool.workers[0].client(0), ["Your name", "Greet"])
        texts = script(coords)
        cfg = EpisodeConfig(max_steps=10, verifier_mode="never")
        policy = RecordingPolicy(texts)
        record = run_episode(policy, TASK, pool, 0, make_run_context("smoke", out_root, 0, "scripted"), cfg)
        check_record(record, policy, Path(record["traj_dir"]))
        run_service_round_trip(pool, cfg, out_root, texts)
        logger.info("RL smoke PASSED")
        return True
    finally:
        pool.stop()
        shutil.rmtree(out_root, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ok = False
    try:
        ok = run_smoke()
    except AssertionError as e:
        logger.error(f"RL smoke FAILED: {e}")
    except Exception as e:
        logger.exception(f"RL smoke ERRORED: {e}")
    sys.exit(0 if ok else 1)
