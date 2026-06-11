"""Scripted (no-LLM) end-to-end check of the agent-capture browser driver.

Boots a sandbox, serves a tiny in-container test page, installs + starts browser_driver.js,
performs a scripted session (snapshot, click, type, scroll, finish), copies the results out,
and asserts the recorded trajectory matches the repo-wide contract: steps with screenshots on
disk, bbox-center coordinates inside the 1280x720 viewport, thoughts recorded per step.

Run: python -m repo_pipeline.driver_smoke   (or `make test-driver`)
"""
import json
import logging
import sys
import tempfile
from pathlib import Path

from repo_pipeline import driver_session
from repo_pipeline.instrumentation import RESULTS_DIR
from repo_pipeline.sandbox import DockerSandbox

logger = logging.getLogger(__name__)

TEST_PAGE = """<!doctype html>
<html><head><title>Driver smoke</title></head><body>
<h1>Smoke test app</h1>
<p id="status" role="status">waiting</p>
<input id="name" type="text" placeholder="Your name">
<button onclick="document.getElementById('status').innerText='hello '+document.getElementById('name').value">Greet</button>
<select id="color"><option value="red">red</option><option value="blue">blue</option></select>
<div style="height: 1500px"></div>
<a href="#bottom" id="bottom-link">Bottom link</a>
</body></html>"""


def run_smoke() -> bool:
    sandbox = DockerSandbox()
    try:
        sandbox.start("smoke/driver-test")

        sandbox.exec_command("mkdir -p /workspace/site")
        sandbox.write_file("/workspace/site/index.html", TEST_PAGE)
        sandbox.start_background("cd /workspace/site && python3 -m http.server 3000", "/workspace/site.log")
        assert sandbox.wait_for_port(3000, 30), "test page server did not start"

        driver_session.install_driver(sandbox)
        driver_session.start_driver(sandbox)
        client = driver_session.DriverClient(sandbox)

        session = client.session("smoke_test", "http://127.0.0.1:3000/", instruction="Greet yourself by name")
        snapshot = session["snapshot"]
        logger.info(f"session snapshot:\n{snapshot}")
        n_elements = driver_session.count_interactive_elements(snapshot)
        assert n_elements >= 4, f"expected >=4 interactive elements, got {n_elements}"

        def ref_for(needle: str) -> int:
            for line in snapshot.splitlines():
                if needle in line and line.startswith("["):
                    return int(line[1:line.index("]")])
            raise AssertionError(f"no element matching {needle!r} in snapshot")

        result = client.act({
            "kind": "type", "ref": ref_for("Your name"), "text": "Smokey",
            "thought": "I need to introduce myself, so I'm typing my name into the name field.",
        })
        snapshot = result["snapshot"]
        result = client.act({
            "kind": "click", "ref": ref_for("Greet"),
            "thought": "With my name entered, clicking Greet should produce the greeting.",
        })
        assert "hello Smokey" in result["snapshot"], f"greeting missing from snapshot: {result['snapshot'][:400]}"
        snapshot = result["snapshot"]
        result = client.act({
            "kind": "scroll", "dy": 800,
            "thought": "The bottom link is below the fold, scrolling down to reach it.",
        })

        # The loop guard must refuse a third identical scroll from the same position only if no
        # progress was made; instead verify the explicit guard with repeated clicks.
        guard_tripped = False
        for _ in range(3):
            try:
                client.act({"kind": "click", "ref": ref_for("Greet"), "thought": "Clicking greet again."})
            except driver_session.DriverError as e:
                assert "loop guard" in str(e), f"unexpected driver error: {e}"
                guard_tripped = True
                break
        assert guard_tripped, "loop guard did not trip after repeated identical clicks"

        finish = client.finish("completed", "Typed a name, clicked greet, saw the greeting.")
        assert finish["steps"] >= 4, f"expected >=4 recorded steps, got {finish['steps']}"

        with tempfile.TemporaryDirectory() as tmp:
            sandbox.copy_out(RESULTS_DIR, tmp)
            traj_dir = Path(tmp) / "task_smoke_test"
            traj = json.loads((traj_dir / "trajectory.json").read_text())
            meta = json.loads((traj_dir / "capture_meta.json").read_text())

            assert traj["start_url"] == "http://127.0.0.1:3000/"
            assert traj["total_steps"] == len(traj["steps"]) >= 4
            assert traj["success"] is False, "success must stay False until the verifier confirms"
            for step in traj["steps"]:
                assert (traj_dir / step["screenshot"]).exists(), f"missing {step['screenshot']}"
                assert step["thought"].strip(), f"step {step['step']} has no thought"
                for action in step["actions"]:
                    if "x" in action:
                        assert 0 <= action["x"] <= 1280 and 0 <= action["y"] <= 720, f"bad coords: {action}"
            assert (traj_dir / "final_screenshot.png").exists()
            assert meta["steps"]["1"]["raw_thought"], "capture_meta missing raw thoughts"

            type_step = traj["steps"][0]
            kinds = [a["type"] for a in type_step["actions"]]
            assert kinds == ["click", "type"], f"type step should record click+type, got {kinds}"

        logger.info("Driver smoke test PASSED")
        return True
    finally:
        sandbox.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ok = False
    try:
        ok = run_smoke()
    except AssertionError as e:
        logger.error(f"Driver smoke test FAILED: {e}")
    except Exception as e:
        logger.exception(f"Driver smoke test ERRORED: {e}")
    sys.exit(0 if ok else 1)
