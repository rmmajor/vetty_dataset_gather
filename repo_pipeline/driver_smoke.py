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
<html><head><title>Driver smoke</title>
<style>
  .menu-items { display: none; }
  .menu:hover .menu-items { display: block; }
</style></head><body>
<h1>Smoke test app</h1>
<p id="status" role="status">waiting</p>
<input id="name" type="text" placeholder="Your name">
<button onclick="document.getElementById('status').innerText='hello '+document.getElementById('name').value">Greet</button>
<select id="color"><option value="red">red</option><option value="blue">blue</option></select>

<div class="menu"><button>Open menu</button>
  <div class="menu-items"><button id="hidden-action">Hidden action</button></div></div>

<input type="range" id="volume" min="0" max="100" aria-label="Volume">
<canvas id="board" width="300" height="200" style="border:1px solid"></canvas>

<ul id="list" ondragover="event.preventDefault()">
  <li id="item-a" draggable="true" ondragstart="event.dataTransfer.setData('text','item-a')">Alpha</li>
  <li id="item-b" draggable="true" ondragstart="event.dataTransfer.setData('text','item-b')"
      ondrop="var s=document.getElementById(event.dataTransfer.getData('text'));this.parentNode.insertBefore(s,this.nextSibling);document.getElementById('status').innerText='reordered'">Beta</li>
</ul>

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

        # Session-scoped burner secret + a (here empty, but valid) pre_session storage_state exercise
        # the login-gated plumbing: masked typing and storageState injection.
        session = client.session(
            "smoke_test", "http://127.0.0.1:3000/", instruction="Greet yourself by name",
            secrets={"smoke_secret": "RealSecretValue"},
            auth_config={"auth_mode": "pre_session", "storage_state": {"cookies": [], "origins": []}},
        )
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
        client.act({"kind": "scroll", "dy": -800, "thought": "Scrolling back up to the controls."})

        affordances = client.affordances()
        logger.info(f"affordances: {affordances}")
        assert affordances["draggables"] == 2, f"expected 2 draggables, got {affordances}"
        assert affordances["sliders"] == 1 and affordances["canvases"] == 1, f"bad affordances: {affordances}"

        # Hover reveals the CSS-hidden menu item and records a `move` action.
        snapshot = client.snapshot()["snapshot"]
        result = client.act({
            "kind": "hover", "ref": ref_for("Open menu"),
            "thought": "The menu may hide more options; hovering to reveal them.",
        })
        assert "Hidden action" in result["snapshot"], f"hover did not reveal menu: {result['snapshot'][:300]}"

        # HTML5 drag-and-drop: raw mouse drag does nothing here, the DragEvent fallback must fire.
        snapshot = result["snapshot"]
        result = client.act({
            "kind": "drag", "ref": ref_for("Alpha"), "to_ref": ref_for("Beta"),
            "thought": "Reordering the list by dragging Alpha below Beta.",
        })
        assert "reordered" in result["snapshot"], f"drag (html5 fallback) had no effect: {result['snapshot'][:300]}"

        # Coordinate tools for canvas surfaces + a recorded wait.
        client.act({"kind": "click_xy", "x": 200, "y": 400,
                    "thought": "Clicking directly on the canvas area."})
        client.act({"kind": "drag_path", "path": [{"x": 180, "y": 380}, {"x": 260, "y": 420}],
                    "thought": "Drawing a stroke across the canvas."})
        client.act({"kind": "wait", "seconds": 0.5,
                    "thought": "Letting the page settle before continuing."})

        # Masked credential: the recorded action keeps the <secret:...> placeholder; the page
        # receives the real value (proving driver-side substitution + leak-free recording).
        client.act({
            "kind": "type", "ref": ref_for("Your name"), "text": "<secret:smoke_secret>",
            "thought": "Entering the provided test username into the name field.",
        })
        result = client.act({
            "kind": "click", "ref": ref_for("Greet"),
            "thought": "Submitting to confirm the entered value took effect.",
        })
        assert "hello RealSecretValue" in result["snapshot"], \
            f"masked secret was not typed as the real value: {result['snapshot'][:300]}"

        # Guarded-mode driver actions: handoff, confirm, and the watchdog approval_event observation.
        client.act({"kind": "handoff", "reason": "login",
                    "thought": "The next step needs a login I should not perform myself; handing off."})
        client.act({"kind": "confirm", "risk_category": "publish",
                    "proposed_action": {"type": "click", "ref": ref_for("Greet")},
                    "thought": "About to publish, so I pause and ask for confirmation first."})
        client.act({"kind": "approval_event",
                    "event": {"kind": "approval", "decision": "deny",
                              "rationale": "publishing on a live site is irreversible", "by": "watchdog"}})

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

        result = client.act({
            "kind": "done",
            "thought": "The page now greets me as Smokey, which is exactly what the task required.",
        })
        assert result["step"] >= 5, "done step was not recorded"

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
                # Observation steps (watchdog approval events) carry an event + empty actions and no
                # thought by design; every ACTION step must have a thought.
                if step.get("actions"):
                    assert step["thought"].strip(), f"step {step['step']} has no thought"
                for action in step["actions"]:
                    if "x" in action:
                        assert 0 <= action["x"] <= 1280 and 0 <= action["y"] <= 720, f"bad coords: {action}"
            assert (traj_dir / "final_screenshot.png").exists()
            assert meta["steps"]["1"]["raw_thought"], "capture_meta missing raw thoughts"

            type_step = traj["steps"][0]
            kinds = [a["type"] for a in type_step["actions"]]
            assert kinds == ["click", "type"], f"type step should record click+type, got {kinds}"

            last_step = traj["steps"][-1]
            assert [a["type"] for a in last_step["actions"]] == ["done"], "last step must be the done action"
            assert meta.get("done_recorded") is True, "capture_meta missing done_recorded"
            recorded_kinds = {a["type"] for s in traj["steps"] for a in s["actions"]}
            assert {"move", "drag", "wait", "done"} <= recorded_kinds, f"missing action kinds: {recorded_kinds}"

            # --- login-gated additions ---
            # Masked credential: the placeholder is recorded; the resolved value never appears.
            blob = json.dumps(traj)
            assert "RealSecretValue" not in blob, "SECRET LEAK: resolved credential found in trajectory"
            masked = [a for s in traj["steps"] for a in s["actions"]
                      if a.get("type") == "type" and a.get("text") == "<secret:smoke_secret>"]
            assert masked, "masked type action did not record the <secret:smoke_secret> placeholder"

            # handoff + confirm recorded as student-target actions.
            handoffs = [a for s in traj["steps"] for a in s["actions"] if a.get("type") == "handoff"]
            assert handoffs and handoffs[0].get("reason") == "login", f"handoff not recorded: {handoffs}"
            confirms = [a for s in traj["steps"] for a in s["actions"] if a.get("type") == "confirm"]
            assert confirms and confirms[0].get("risk_category") == "publish" and confirms[0].get("proposed_action"), \
                f"confirm not recorded correctly: {confirms}"

            # approval_event: an observation step with the watchdog verdict and EMPTY actions.
            event_steps = [s for s in traj["steps"] if s.get("event")]
            assert event_steps, "no watchdog approval event step recorded"
            ev = event_steps[0]
            assert ev["actions"] == [] and ev["event"].get("decision") == "deny", \
                f"approval event malformed (must be observation-only): {ev}"

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
