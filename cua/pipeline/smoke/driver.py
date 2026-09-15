"""Scripted (no-LLM) end-to-end check of the agent-capture browser driver: a sandboxed test page, a recorded session, and the trajectory contract over what it wrote."""

import base64
import json
import logging
import sys
import tempfile
from pathlib import Path

from cua.pipeline import driver_session
from cua.pipeline.instrumentation import RESULTS_DIR
from cua.pipeline.sandbox import DockerSandbox
from cua.pipeline.smoke._harness import run_checks

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

<iframe id="embed" width="320" height="90" srcdoc="<button id='ibtn' onclick='document.body.innerHTML=&quot;<h2>iframe action done</h2>&quot;'>Inside-frame button</button>"></iframe>

<div style="height: 1500px"></div>
<a href="#bottom" id="bottom-link">Bottom link</a>
</body></html>"""

APP_URL = "http://127.0.0.1:3000/"
TASK_ID = "smoke_test"
SECRET_VALUE = "RealSecretValue"


class DriverSmoke:
    """The one live sandbox, driver and recorded session the checks below share, in the order they run."""

    def __init__(self):
        self.sandbox = None
        self.client = None
        self.snapshot = ""
        self.observed_png = ""
        self.workspace = None

    def start(self) -> list[str]:
        """Boot the sandbox, serve the test page, install the driver and open the recorded session."""
        self.sandbox = DockerSandbox()
        self.sandbox.start("smoke/driver-test")
        self.sandbox.exec_command("mkdir -p /workspace/site")
        self.sandbox.write_file("/workspace/site/index.html", TEST_PAGE)
        self.sandbox.start_background("cd /workspace/site && python3 -m http.server 3000", "/workspace/site.log")
        if not self.sandbox.wait_for_port(3000, 30):
            return ["test page server did not start"]
        driver_session.install_driver(self.sandbox)
        driver_session.start_driver(self.sandbox)
        self.client = driver_session.DriverClient(self.sandbox)
        # A session-scoped burner secret plus a (here empty, but valid) pre_session storage_state exercise the login-gated plumbing.
        session = self.client.session(
            TASK_ID,
            APP_URL,
            instruction="Greet yourself by name",
            secrets={"smoke_secret": SECRET_VALUE},
            auth_config={"auth_mode": "pre_session", "storage_state": {"cookies": [], "origins": []}},
        )
        self.snapshot = session["snapshot"]
        logger.info(f"session snapshot:\n{self.snapshot}")
        return []

    def stop(self) -> None:
        if self.sandbox is not None:
            self.sandbox.stop()
        if self.workspace is not None:
            self.workspace.cleanup()

    def ref_for(self, needle: str) -> int:
        for line in self.snapshot.splitlines():
            if needle in line and line.startswith("["):
                return int(line[1 : line.index("]")])
        raise AssertionError(f"no element matching {needle!r} in snapshot")

    def act(self, payload: dict) -> dict:
        return self.client.act(payload)

    def remember(self, result: dict) -> dict:
        """Make this result's snapshot the one `ref_for` resolves against, as the scripted session walks the page."""
        self.snapshot = result["snapshot"]
        return result

    def refresh(self) -> str:
        self.snapshot = self.client.snapshot()["snapshot"]
        return self.snapshot

    def copied_out(self) -> tuple[Path, dict, dict]:
        """The results directory copied out of the container once, as (dir, trajectory, capture_meta)."""
        if self.workspace is None:
            self.workspace = tempfile.TemporaryDirectory(prefix="driver_smoke_")
            self.sandbox.copy_out(RESULTS_DIR, self.workspace.name)
        traj_dir = Path(self.workspace.name) / f"task_{TASK_ID}"
        return (
            traj_dir,
            json.loads((traj_dir / "trajectory.json").read_text()),
            json.loads((traj_dir / "capture_meta.json").read_text()),
        )


SESSION = DriverSmoke()


def check_boot_and_session() -> list[str]:
    failures = SESSION.start()
    if failures:
        return failures
    n_elements = driver_session.count_interactive_elements(SESSION.snapshot)
    if n_elements < 4:
        failures.append(f"expected >=4 interactive elements, got {n_elements}")
    return failures


def check_type_and_click() -> list[str]:
    SESSION.remember(
        SESSION.act(
            {
                "kind": "type",
                "ref": SESSION.ref_for("Your name"),
                "text": "Smokey",
                "thought": "I need to introduce myself, so I'm typing my name into the name field.",
            }
        )
    )
    result = SESSION.remember(
        SESSION.act(
            {
                "kind": "click",
                "ref": SESSION.ref_for("Greet"),
                "thought": "With my name entered, clicking Greet should produce the greeting.",
            }
        )
    )
    if "hello Smokey" not in result["snapshot"]:
        return [f"greeting missing from snapshot: {result['snapshot'][:400]}"]
    return []


def check_scroll_and_affordances() -> list[str]:
    SESSION.act(
        {"kind": "scroll", "dy": 800, "thought": "The bottom link is below the fold, scrolling down to reach it."}
    )
    SESSION.act({"kind": "scroll", "dy": -800, "thought": "Scrolling back up to the controls."})
    affordances = SESSION.client.affordances()
    logger.info(f"affordances: {affordances}")
    failures = []
    if affordances["draggables"] != 2:
        failures.append(f"expected 2 draggables, got {affordances}")
    if affordances["sliders"] != 1 or affordances["canvases"] != 1:
        failures.append(f"bad affordances: {affordances}")
    return failures


def check_hover_reveals_menu() -> list[str]:
    SESSION.refresh()
    result = SESSION.remember(
        SESSION.act(
            {
                "kind": "hover",
                "ref": SESSION.ref_for("Open menu"),
                "thought": "The menu may hide more options; hovering to reveal them.",
            }
        )
    )
    if "Hidden action" not in result["snapshot"]:
        return [f"hover did not reveal menu: {result['snapshot'][:300]}"]
    return []


def check_html5_drag() -> list[str]:
    """Raw mouse drag does nothing on this list, so the recorded drag proves the DragEvent fallback fires."""
    result = SESSION.act(
        {
            "kind": "drag",
            "ref": SESSION.ref_for("Alpha"),
            "to_ref": SESSION.ref_for("Beta"),
            "thought": "Reordering the list by dragging Alpha below Beta.",
        }
    )
    if "reordered" not in result["snapshot"]:
        return [f"drag (html5 fallback) had no effect: {result['snapshot'][:300]}"]
    return []


def check_iframe_click() -> list[str]:
    """The path that lets the agent click a verification/reset link inside a webmail message body."""
    SESSION.refresh()
    result = SESSION.act(
        {
            "kind": "click",
            "ref": SESSION.ref_for("Inside-frame button"),
            "thought": "Clicking the button inside the embedded frame to test iframe interaction.",
        }
    )
    if "iframe action done" not in result["snapshot"]:
        return [f"iframe element was not reachable/clickable: {result['snapshot'][:300]}"]
    return []


def check_coordinate_tools_and_wait() -> list[str]:
    """Canvas surfaces have no refs, so their coordinate tools and the recorded wait are exercised here."""
    SESSION.act({"kind": "click_xy", "x": 200, "y": 400, "thought": "Clicking directly on the canvas area."})
    SESSION.act(
        {
            "kind": "drag_path",
            "path": [{"x": 180, "y": 380}, {"x": 260, "y": 420}],
            "thought": "Drawing a stroke across the canvas.",
        }
    )
    SESSION.act({"kind": "wait", "seconds": 0.5, "thought": "Letting the page settle before continuing."})
    return []


def check_masked_secret_typing() -> list[str]:
    """The page receives the real value while the recorded action keeps the placeholder — driver-side substitution."""
    SESSION.act(
        {
            "kind": "type",
            "ref": SESSION.ref_for("Your name"),
            "text": "<secret:smoke_secret>",
            "thought": "Entering the provided test username into the name field.",
        }
    )
    result = SESSION.act(
        {
            "kind": "click",
            "ref": SESSION.ref_for("Greet"),
            "thought": "Submitting to confirm the entered value took effect.",
        }
    )
    if f"hello {SECRET_VALUE}" not in result["snapshot"]:
        return [f"masked secret was not typed as the real value: {result['snapshot'][:300]}"]
    return []


def check_guarded_actions() -> list[str]:
    """Guarded-mode driver actions: handoff, confirm, and the watchdog approval event."""
    SESSION.act(
        {
            "kind": "handoff",
            "reason": "login",
            "thought": "The next step needs a login I should not perform myself; handing off.",
        }
    )
    SESSION.act(
        {
            "kind": "confirm",
            "risk_category": "publish",
            "proposed_action": {"type": "click", "ref": SESSION.ref_for("Greet")},
            "thought": "About to publish, so I pause and ask for confirmation first.",
        }
    )
    SESSION.act(
        {
            "kind": "approval_event",
            "event": {
                "kind": "approval",
                "decision": "deny",
                "rationale": "publishing on a live site is irreversible",
                "by": "watchdog",
            },
        }
    )
    return []


def check_observe_and_sequence() -> list[str]:
    """The RL path: /observe's PNG becomes the next recorded step and `sequence` records ONE step of contract dicts."""
    failures = []
    recorded = json.loads(SESSION.sandbox.read_file(f"{RESULTS_DIR}/task_{TASK_ID}/trajectory.json"))
    name_click = recorded["steps"][0]["actions"][0]
    greet_click = recorded["steps"][1]["actions"][0]
    observed = SESSION.client.observe()
    if observed["mime"] != "image/png" or observed["step_next"] != len(recorded["steps"]) + 1:
        failures.append(f"unexpected observation {observed['mime']} step_next={observed['step_next']}")
    seq = SESSION.client.act_sequence(
        "Retyping my name through the coordinate path and submitting.",
        [
            {"type": "click", "x": name_click["x"], "y": name_click["y"], "button": "left"},
            {"type": "type", "text": "Sequence"},
            {"type": "click", "x": greet_click["x"], "y": greet_click["y"], "button": "left"},
        ],
    )
    if seq["executed"] != 3 or seq["error"] is not None or seq["terminal"] is not False:
        failures.append(f"unexpected sequence result {seq}")
    if "hello Sequence" not in SESSION.client.snapshot()["snapshot"]:
        failures.append("sequence actions did not take effect")
    SESSION.observed_png = observed["b64"]
    return failures


def check_loop_guard() -> list[str]:
    """The explicit guard: a third identical click from the same position with no progress is refused."""
    failures = []
    guard_tripped = False
    for _ in range(3):
        try:
            SESSION.act({"kind": "click", "ref": SESSION.ref_for("Greet"), "thought": "Clicking greet again."})
        except driver_session.DriverError as e:
            if "loop guard" not in str(e):
                failures.append(f"unexpected driver error: {e}")
            guard_tripped = True
            break
    if not guard_tripped:
        failures.append("loop guard did not trip after repeated identical clicks")
    return failures


def check_done_and_finish() -> list[str]:
    failures = []
    result = SESSION.act(
        {
            "kind": "done",
            "thought": "The page now greets me as Smokey, which is exactly what the task required.",
        }
    )
    if result["step"] < 5:
        failures.append("done step was not recorded")
    finish = SESSION.client.finish("completed", "Typed a name, clicked greet, saw the greeting.")
    if finish["steps"] < 4:
        failures.append(f"expected >=4 recorded steps, got {finish['steps']}")
    return failures


def check_recorded_contract() -> list[str]:
    """What the driver wrote, against the repo-wide trajectory contract."""
    failures = []
    traj_dir, traj, meta = SESSION.copied_out()
    if traj["start_url"] != APP_URL:
        failures.append(f"unexpected start_url {traj['start_url']!r}")
    if traj["total_steps"] != len(traj["steps"]) or len(traj["steps"]) < 4:
        failures.append(f"total_steps {traj['total_steps']} against {len(traj['steps'])} recorded steps")
    if traj["success"] is not False:
        failures.append("success must stay False until the verifier confirms")
    for step in traj["steps"]:
        if not (traj_dir / step["screenshot"]).exists():
            failures.append(f"missing {step['screenshot']}")
        # Observation steps (watchdog approval events) carry an event and no thought by design.
        if step.get("actions") and not step["thought"].strip():
            failures.append(f"step {step['step']} has no thought")
        for action in step["actions"]:
            if "x" in action and not (0 <= action["x"] <= 1280 and 0 <= action["y"] <= 720):
                failures.append(f"bad coords: {action}")
    if not (traj_dir / "final_screenshot.png").exists():
        failures.append("final_screenshot.png was not written")
    if not meta["steps"]["1"]["raw_thought"]:
        failures.append("capture_meta missing raw thoughts")
    kinds = [a["type"] for a in traj["steps"][0]["actions"]]
    if kinds != ["click", "type"]:
        failures.append(f"type step should record click+type, got {kinds}")
    if [a["type"] for a in traj["steps"][-1]["actions"]] != ["done"]:
        failures.append("last step must be the done action")
    if meta.get("done_recorded") is not True:
        failures.append("capture_meta missing done_recorded")
    recorded_kinds = {a["type"] for s in traj["steps"] for a in s["actions"]}
    if not {"move", "drag", "wait", "done"} <= recorded_kinds:
        failures.append(f"missing action kinds: {recorded_kinds}")
    return failures


def check_recorded_login_gated() -> list[str]:
    """The masked credential, the guarded-mode actions and the observation PNG, as recorded."""
    failures = []
    traj_dir, traj, _ = SESSION.copied_out()
    blob = json.dumps(traj)
    if SECRET_VALUE in blob:
        failures.append("SECRET LEAK: resolved credential found in trajectory")
    masked = [
        a
        for s in traj["steps"]
        for a in s["actions"]
        if a.get("type") == "type" and a.get("text") == "<secret:smoke_secret>"
    ]
    if not masked:
        failures.append("masked type action did not record the <secret:smoke_secret> placeholder")
    handoffs = [a for s in traj["steps"] for a in s["actions"] if a.get("type") == "handoff"]
    if not handoffs or handoffs[0].get("reason") != "login":
        failures.append(f"handoff not recorded: {handoffs}")
    confirms = [a for s in traj["steps"] for a in s["actions"] if a.get("type") == "confirm"]
    if not confirms or confirms[0].get("risk_category") != "publish" or not confirms[0].get("proposed_action"):
        failures.append(f"confirm not recorded correctly: {confirms}")
    event_steps = [s for s in traj["steps"] if s.get("event")]
    if not event_steps:
        return [*failures, "no watchdog approval event step recorded"]
    event_step = event_steps[0]
    if event_step["actions"] != [] or event_step["event"].get("decision") != "deny":
        failures.append(f"approval event malformed (must be observation-only): {event_step}")
    seq_steps = [s for s in traj["steps"] if s["thought"].startswith("Retyping my name")]
    if not seq_steps:
        return [*failures, "the sequence step was not recorded"]
    seq_step = seq_steps[0]
    if [a["type"] for a in seq_step["actions"]] != ["click", "type", "click"]:
        failures.append(f"unexpected sequence actions {seq_step['actions']}")
    if (traj_dir / seq_step["screenshot"]).read_bytes() != base64.b64decode(SESSION.observed_png):
        failures.append("observation PNG must be byte-identical to the recorded step screenshot")
    return failures


CHECKS = [
    check_boot_and_session,
    check_type_and_click,
    check_scroll_and_affordances,
    check_hover_reveals_menu,
    check_html5_drag,
    check_iframe_click,
    check_coordinate_tools_and_wait,
    check_masked_secret_typing,
    check_guarded_actions,
    check_observe_and_sequence,
    check_loop_guard,
    check_done_and_finish,
    check_recorded_contract,
    check_recorded_login_gated,
]


if __name__ == "__main__":
    try:
        sys.exit(run_checks(CHECKS, "Driver smoke test"))
    finally:
        SESSION.stop()
