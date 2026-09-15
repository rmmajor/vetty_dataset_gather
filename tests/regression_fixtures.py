"""Behaviour-regression fixtures: `tests.regression_fixtures build|check` records and diffs today's output."""

import argparse
import dataclasses
import difflib
import hashlib
import io
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from cua.eval.student import EvalConfig, harness_fingerprint, parse_student_output
from cua.eval.student_smoke import REPLAY_STEPS
from cua.eval.task_sources import TASK_SOURCES
from cua.pipeline import schemas
from cua.pipeline.registry import RepoRegistry
from cua.rl.actions import decode_step
from cua.trajectory import VIEWPORT
from cua.trajectory.actions import format_action_for_training, format_action_summary

FIXTURES = Path(__file__).resolve().parent / "fixtures"
# Owned by the prompt contract, not by this generator: never written, never diffed here.
UNMANAGED = ("prompt_golden.json",)
SUBPROCESS_TIMEOUT_S = 600


def repo_root() -> Path:
    """The checkout root, found by its Makefile so moving this module into a package does not break it."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "Makefile").is_file():
            return candidate
    raise SystemExit("no Makefile above this module — cannot locate the repository root")


# Expands every guarded target under `make -n`; nowhere paths make file-prerequisite targets record a rule, not a tree.
GUARD_VARS = {
    "VAST_API_KEY": "x",
    "EVAL_ENDPOINT": "x",
    "RL_ENV_TOKEN": "x",
    "RL_ENV_URL": "x",
    "SITE": "x",
    "EVAL_RUNS": "x",
    "GITHUB_TOKEN": "x",
    "OPENAI_API_KEY": "x",
    "TRAIN_DIR": "/nonexistent/training_data",
    "TASKS_FILE": "/nonexistent/tasks.jsonl",
}
ENV_PASSTHROUGH = ("PATH", "HOME", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR")


def subprocess_env() -> dict:
    """A pinned environment: an inherited COLUMNS rewraps argparse help and MAKEFLAGS reprefixes make errors."""
    # env-read: a fixture subprocess needs the real PATH/HOME, and nothing else, from this process
    env = {name: os.environ[name] for name in ENV_PASSTHROUGH if name in os.environ}
    env.update(GUARD_VARS)
    env.update(
        {
            "COLUMNS": "80",
            "LINES": "24",
            "TERM": "dumb",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CLAUDE_CLI_PATH": "/nonexistent/claude-must-never-run",
        }
    )
    return env


def run_capture(argv, cwd) -> str:
    """Combined output of a subprocess, with the exit status appended when it is not zero."""
    result = subprocess.run(
        argv, cwd=str(cwd), env=subprocess_env(), capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S
    )
    tail = "" if result.returncode == 0 else f"[exit {result.returncode}]\n"
    return result.stdout + result.stderr + tail


def normalise(text: str, root: Path, extra=()) -> str:
    """Replace every machine-specific path with a stable token, most specific first since the repo sits under home."""
    for needle, token in (
        *extra,
        (sys.executable, "<python>"),
        (str(Path(sys.prefix)), "<venv>"),
        (str(root), "<repo>"),
        (str(Path.home()), "<home>"),
    ):
        text = text.replace(str(needle), token)
    return text


_USAGE_BLOCK_RE = re.compile(r"\Ausage:.*?(?:\n[ \t]*\n|\Z)", re.DOTALL)


def strip_usage(text: str) -> str:
    """Drop argparse's whole usage block: it spells the program name and indents continuations by its width."""
    return _USAGE_BLOCK_RE.sub("", text, count=1).lstrip("\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def jsonl_sha256(text: str) -> str:
    """Digest of the SORTED lines, so the fixture records content independently of shuffle order."""
    return sha256_text("\n".join(sorted(text.splitlines())) + "\n")


def dumps(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


SEARCH_BBOX = {"x": 600, "y": 340, "width": 80, "height": 40}
ICON_BBOX = {"x": 1000, "y": 20, "width": 24, "height": 24}
TITLE_BBOX = {"x": 100, "y": 100, "width": 120, "height": 30}


def bbox_centre(bbox) -> tuple:
    return round(bbox["x"] + bbox["width"] / 2), round(bbox["y"] + bbox["height"] / 2)


def frame_bytes(frame: int) -> bytes:
    """A 1280x720 PNG that is a pure function of `frame`, so two steps can differ or be pixel-identical on purpose."""
    rng = random.Random(frame)
    image = Image.new("RGB", VIEWPORT, (248, 248, 250))
    draw = ImageDraw.Draw(image)
    for _ in range(10):
        x, y = rng.randrange(0, VIEWPORT[0] - 120), rng.randrange(0, VIEWPORT[1] - 80)
        draw.rectangle(
            [x, y, x + rng.randrange(40, 120), y + rng.randrange(20, 80)],
            fill=(rng.randrange(256), rng.randrange(256), rng.randrange(256)),
        )
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def step(number, frame, thought, actions, url=None, element=None, event=None) -> dict:
    """One trajectory step plus the capture_meta entry it needs, as (step, meta)."""
    entry = {
        "step": number,
        "screenshot": f"step_{number:03d}_screenshot.png",
        "thought": thought,
        "actions": actions,
        "_frame": frame,
    }
    if event:
        entry["event"] = event
    meta = {}
    if url:
        meta["url"] = url
    if element:
        meta.update(element)
    return entry, meta


def click_on(bbox, button="left") -> dict:
    x, y = bbox_centre(bbox)
    return {"type": "click", "x": x, "y": y, "button": button}


def element(role, name, bbox) -> dict:
    return {"role": role, "name": name, "bbox": bbox}


def forms_trajectory() -> tuple:
    """Grounding-rich agent trajectory with no done action: capture_meta drives the synthesized terminal step."""
    steps = [
        step(
            1,
            1,
            "Start from the search box on the landing page.",
            [click_on(SEARCH_BBOX)],
            url="https://synth.test/",
            element=element("button", "Search", SEARCH_BBOX),
        ),
        step(
            2,
            2,
            "Enter what the user is looking for.",
            [{"type": "type", "text": "laptop stand"}],
            url="https://synth.test/",
        ),
        step(3, 3, "Submit the query.", [{"type": "keypress", "keys": ["Enter"]}], url="https://synth.test/results"),
        step(
            4,
            4,
            "Narrow the results with the filter control.",
            [click_on(ICON_BBOX)],
            url="https://synth.test/results",
            element=element("button", "Filter", ICON_BBOX),
        ),
        step(
            5,
            4,
            "Hover the first result to reveal its controls.",
            [{"type": "move", "x": 300, "y": 300}],
            url="https://synth.test/results",
        ),
        step(
            6,
            5,
            "Open the context menu on the first result.",
            [{"type": "click", "x": 500, "y": 500, "button": "right"}],
            url="https://synth.test/results",
        ),
        step(
            7,
            6,
            "Open the result by activating its title.",
            [{"type": "double_click", **dict(zip(("x", "y"), bbox_centre(TITLE_BBOX), strict=True))}],
            url="https://synth.test/item",
            element=element("a", "Adjustable Stand", TITLE_BBOX),
        ),
    ]
    metadata = {
        "capture_source": "agent",
        "thought_source": "teacher",
        "website": "synth.test",
        "autonomy": "autonomous",
        "auth_mode": "none",
    }
    capture_meta = {
        "verifier": {"verdict": "success", "efficiency": "direct"},
        "agent_outcome": "completed",
        "agent_summary": "The item page is open, so the search goal is met.",
        "task_meta": {"autonomy": "autonomous", "auth_mode": "none"},
    }
    return steps, metadata, capture_meta, True


def keys_trajectory() -> tuple:
    """Every keyboard and pointer-delta shape, including an absurd repeat that convert drops as a target."""
    steps = [
        step(
            1,
            10,
            "Scroll the settings list into view.",
            [{"type": "scroll", "x": 640, "y": 360, "scroll_x": 0, "scroll_y": 500}],
            url="https://synth.test/settings",
        ),
        step(
            2,
            11,
            "Select the whole field before replacing it.",
            [{"type": "keypress", "keys": ["Control", "a"]}],
            url="https://synth.test/settings",
        ),
        step(
            3,
            12,
            "Walk down the option list.",
            [{"type": "keypress", "keys": ["ArrowDown"] * 7}],
            url="https://synth.test/settings",
        ),
        step(
            4,
            13,
            "Reorder the two rows by dragging.",
            [{"type": "drag", "path": [{"x": 100, "y": 200}, {"x": 400, "y": 500}]}],
            url="https://synth.test/settings",
        ),
        step(5, 14, "Let the save indicator settle.", [{"type": "wait"}], url="https://synth.test/settings"),
        step(
            6,
            14,
            "Take a fresh frame before reading the result.",
            [{"type": "screenshot"}],
            url="https://synth.test/settings",
        ),
        step(
            7,
            15,
            "Type the password into the confirmation field.",
            [{"type": "type", "text": "<secret:PASSWORD>"}],
            url="https://synth.test/settings",
            element=element("input", "Password", SEARCH_BBOX),
        ),
        step(
            8,
            16,
            "The settings page shows the new value, so the task is complete.",
            [{"type": "done"}],
            url="https://synth.test/settings",
        ),
    ]
    metadata = {
        "capture_source": "agent",
        "thought_source": "teacher",
        "website": "synth.test",
        "autonomy": "autonomous",
        "auth_mode": "pre_session",
    }
    capture_meta = {
        "verifier": {"verdict": "success", "efficiency": "direct"},
        "task_meta": {"autonomy": "autonomous", "auth_mode": "pre_session"},
    }
    return steps, metadata, capture_meta, True


def meander_trajectory() -> tuple:
    """Sixteen steps graded meandering, so keep and subsample produce measurably different datasets."""
    steps = []
    for index in range(15):
        number = index + 1
        steps.append(
            step(
                number,
                20 + index,
                f"Look for the export control, attempt {number}.",
                [{"type": "click", "x": 120 + index * 30, "y": 140 + index * 20, "button": "left"}],
                url="https://synth.test/reports",
            )
        )
    steps.append(
        step(
            16,
            36,
            "The export finished and the file is listed, so the goal is met.",
            [{"type": "done"}],
            url="https://synth.test/reports",
        )
    )
    metadata = {
        "capture_source": "agent",
        "thought_source": "teacher",
        "website": "synth.test",
        "autonomy": "autonomous",
        "auth_mode": "none",
        "efficiency": "meandering",
    }
    capture_meta = {
        "verifier": {"verdict": "success", "efficiency": "meandering"},
        "task_meta": {"autonomy": "autonomous", "auth_mode": "none"},
    }
    return steps, metadata, capture_meta, True


def guarded_trajectory() -> tuple:
    """The guarded-lane control actions: confirm, its recorded approval observation, and a terminal handoff."""
    steps = [
        step(
            1,
            40,
            "Open the billing page the user asked about.",
            [{"type": "click", "x": 200, "y": 200, "button": "left"}],
            url="https://synth.test/billing",
        ),
        step(
            2,
            41,
            "Placing the order spends money, so propose it instead of clicking.",
            [
                {
                    "type": "confirm",
                    "proposed_action": {"type": "click", "x": 640, "y": 400},
                    "risk_category": "FINANCIAL_RISK",
                }
            ],
            url="https://synth.test/billing",
            event={"kind": "approval", "decision": "approve"},
        ),
        step(
            3,
            42,
            "The watchdog approved, so record the decision before continuing.",
            [{"type": "approval_event"}],
            url="https://synth.test/billing",
        ),
        step(
            4,
            43,
            "The card form needs the real cardholder, so hand control back.",
            [{"type": "handoff", "reason": "payment"}],
            url="https://synth.test/billing/pay",
        ),
    ]
    metadata = {
        "capture_source": "agent",
        "thought_source": "teacher",
        "website": "synth.test",
        "autonomy": "guarded",
        "auth_mode": "pre_session",
    }
    capture_meta = {
        "verifier": {"verdict": "success", "efficiency": "direct"},
        "task_meta": {"autonomy": "guarded", "auth_mode": "pre_session"},
    }
    return steps, metadata, capture_meta, True


def excluded_trajectory() -> tuple:
    steps = [
        step(
            1,
            50,
            "Open the page under evaluation.",
            [{"type": "click", "x": 300, "y": 300, "button": "left"}],
            url="https://synth.test/",
        ),
        step(2, 51, "The page is open, so the episode is finished.", [{"type": "done"}], url="https://synth.test/"),
    ]
    metadata = {
        "capture_source": "agent",
        "thought_source": "teacher",
        "website": "synth.test",
        "excluded_reason": "eval_episode",
    }
    return steps, metadata, {"verifier": {"verdict": "success", "efficiency": "direct"}}, True


def navigate_trajectory() -> tuple:
    """A teacher_raw trajectory: the corpus-health gate counts these, and navigate/wait/save live here."""
    steps = [
        step(
            1,
            60,
            "Go straight to the app's task board.",
            [{"type": "navigate", "url": "https://synth.test/app/board"}],
            url="https://synth.test/",
        ),
        step(
            2, 60, "Wait for the board to render before acting.", [{"type": "wait"}], url="https://synth.test/app/board"
        ),
        step(
            3,
            61,
            "Save the edited card.",
            [click_on(SEARCH_BBOX)],
            url="https://synth.test/app/board",
            element=element("button", "Save", SEARCH_BBOX),
        ),
        step(
            4,
            62,
            "The card shows the saved title, so the change persisted.",
            [{"type": "done"}],
            url="https://synth.test/app/board",
        ),
    ]
    metadata = {
        "capture_source": "agent",
        "thought_source": "teacher_raw",
        "website": "synth.app",
        "autonomy": "autonomous",
        "auth_mode": "none",
    }
    capture_meta = {
        "verifier": {"verdict": "success", "efficiency": "direct"},
        "task_meta": {"autonomy": "autonomous", "auth_mode": "none"},
    }
    return steps, metadata, capture_meta, True


def replay_trajectory() -> tuple:
    steps = [
        step(
            1,
            70,
            "Open the login form the test drives.",
            [{"type": "click", "x": 300, "y": 300, "button": "left"}],
            url="https://synth.app/login",
        ),
        step(
            2,
            71,
            "Fill the user name the test fixture provides.",
            [{"type": "type", "text": "qa-user"}],
            url="https://synth.app/login",
        ),
        step(
            3, 72, "The dashboard is visible, so the flow completed.", [{"type": "done"}], url="https://synth.app/home"
        ),
    ]
    metadata = {"capture_source": "replay", "thought_source": "generated", "website": "synth.app"}
    return steps, metadata, {}, True


def waste_trajectory() -> tuple:
    """Retry clicks within the waste radius, so --exclude-waste-steps has something to drop."""
    steps = [
        step(
            1,
            80,
            "Click the submit button.",
            [{"type": "click", "x": 400, "y": 400, "button": "left"}],
            url="https://synth.app/form",
        ),
        step(
            2,
            80,
            "Nothing happened, so click the submit button again.",
            [{"type": "click", "x": 404, "y": 402, "button": "left"}],
            url="https://synth.app/form",
        ),
        step(
            3,
            81,
            "Still nothing, so try the same spot once more.",
            [{"type": "click", "x": 402, "y": 398, "button": "left"}],
            url="https://synth.app/form",
        ),
        step(
            4,
            82,
            "Type the missing field the form was waiting for.",
            [{"type": "type", "text": "42"}],
            url="https://synth.app/form",
        ),
        step(
            5,
            83,
            "The confirmation banner is on screen, so the form went through.",
            [{"type": "done"}],
            url="https://synth.app/form/done",
        ),
    ]
    metadata = {
        "capture_source": "agent",
        "thought_source": "teacher",
        "website": "synth.app",
        "autonomy": "autonomous",
        "auth_mode": "none",
    }
    capture_meta = {
        "verifier": {"verdict": "success", "efficiency": "direct"},
        "task_meta": {"autonomy": "autonomous", "auth_mode": "none"},
    }
    return steps, metadata, capture_meta, True


def unverified_trajectory() -> tuple:
    steps = [
        step(
            1,
            90,
            "Try to reach the checkout page.",
            [{"type": "click", "x": 200, "y": 400, "button": "left"}],
            url="https://synth.app/cart",
        ),
        step(
            2,
            91,
            "The page never loaded, so nothing was achieved.",
            [{"type": "click", "x": 260, "y": 420, "button": "left"}],
            url="https://synth.app/cart",
        ),
    ]
    metadata = {"capture_source": "agent", "thought_source": "teacher", "website": "synth.app"}
    capture_meta = {"verifier": {"verdict": "failure", "efficiency": "direct"}}
    return steps, metadata, capture_meta, False


def mind2web_trajectory() -> tuple:
    steps = [
        step(
            1,
            100,
            "Focus the site's search field.",
            [{"type": "click", "x": 640, "y": 360, "button": "left"}],
            url="https://synth.news/",
        ),
        step(
            2,
            101,
            "Enter the topic the user asked for.",
            [{"type": "type", "text": "battery research"}],
            url="https://synth.news/",
        ),
        step(3, 102, "Run the search.", [{"type": "keypress", "keys": ["Enter"]}], url="https://synth.news/search"),
        step(
            4,
            103,
            "The results list the requested topic, so the task is done.",
            [{"type": "done"}],
            url="https://synth.news/search",
        ),
    ]
    return steps, {"website": "synth.news"}, {}, True


# Directory -> builder; the directory name decides the split group and the path_a source heuristic.
SYNTHETIC_TRAJECTORIES = (
    (
        "site_synth/task_0001_search",
        "Find an adjustable laptop stand on this shop.",
        "https://synth.test/",
        forms_trajectory,
    ),
    (
        "site_synth/task_0002_settings",
        "Change my notification setting and save it.",
        "https://synth.test/settings",
        keys_trajectory,
    ),
    ("site_synth/task_0003_export", "Export this month's report.", "https://synth.test/reports", meander_trajectory),
    (
        "site_synth/task_0004_billing",
        "Pay the outstanding invoice on my account.",
        "https://synth.test/billing",
        guarded_trajectory,
    ),
    ("site_synth/task_0005_eval", "Open the landing page.", "https://synth.test/", excluded_trajectory),
    (
        "repo_synth_app/task_0001_board",
        "Rename the first card on my board.",
        "https://synth.test/",
        navigate_trajectory,
    ),
    ("repo_synth_app/task_0002_login", "Sign in to the app.", "https://synth.app/login", replay_trajectory),
    ("repo_synth_app/task_0003_form", "Submit the contact form.", "https://synth.app/form", waste_trajectory),
    (
        "repo_synth_app/task_0004_cart",
        "Check out the items in my cart.",
        "https://synth.app/cart",
        unverified_trajectory,
    ),
    ("task_mind2web_0001", "Search this news site for battery research.", "https://synth.news/", mind2web_trajectory),
)


def write_synthetic_corpus(root: Path) -> Path:
    """Materialise the ten-trajectory synthetic corpus; the only input the convert golden depends on."""
    root.mkdir(parents=True, exist_ok=True)
    for relative, instruction, start_url, builder in SYNTHETIC_TRAJECTORIES:
        steps, metadata, capture_meta, success = builder()
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        step_meta = {}
        plain_steps = []
        for entry, meta in steps:
            frame = entry.pop("_frame")
            (directory / entry["screenshot"]).write_bytes(frame_bytes(frame))
            if meta:
                step_meta[str(entry["step"])] = meta
            plain_steps.append(entry)
        (directory / "final_screenshot.png").write_bytes(frame_bytes(900 + len(plain_steps)))
        trajectory = {
            "task_id": directory.name,
            "instruction": instruction,
            "start_url": start_url,
            "steps": plain_steps,
            "total_steps": len(plain_steps),
            "success": success,
        }
        (directory / "trajectory.json").write_text(dumps(trajectory))
        (directory / "metadata.json").write_text(dumps({"task_id": directory.name, **metadata}))
        if capture_meta or step_meta:
            (directory / "capture_meta.json").write_text(dumps({**capture_meta, "steps": step_meta}))
    return root


CONVERT_MODES = (("convert_golden", "subsample"), ("convert_golden_keep", "keep"))


def build_convert_golden(root: Path, corpus: Path | None) -> dict:
    """One dataset per meandering mode over the synthetic corpus, with every machine path tokenised."""
    files = {}
    with tempfile.TemporaryDirectory(prefix="regression_convert_") as tmp:
        work = Path(tmp)
        synthetic = write_synthetic_corpus(work / "corpus")
        for stem, mode in CONVERT_MODES:
            output = work / f"out_{mode}"
            argv = [
                sys.executable,
                "-m",
                "cua.sft",
                "convert",
                "--input",
                str(synthetic),
                "--output",
                str(output),
                "--skip-health-gate",
                "--exclude-waste-steps",
                "--meandering-mode",
                mode,
                "--include-replay",
                "--seed",
                "0",
            ]
            result = subprocess.run(
                argv, cwd=str(root), env=subprocess_env(), capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S
            )
            if result.returncode != 0:
                raise SystemExit(f"convert --meandering-mode {mode} failed:\n{result.stdout}\n{result.stderr}")
            replacements = ((str(synthetic), "<corpus>"), (str(output), "<output>"))
            digests = []
            for suffix, name in (("", "sft_dataset.jsonl"), (".val", "sft_val.jsonl")):
                produced = output / name
                text = normalise(produced.read_text(), root, replacements) if produced.is_file() else ""
                files[f"{stem}{suffix}.jsonl"] = text
                digests.append(f"{jsonl_sha256(text)}  {stem}{suffix}.jsonl")
            files[f"{stem}.sha256"] = "\n".join(digests) + "\n"
    _assert_convert_golden_is_useful(files)
    return files


def _assert_convert_golden_is_useful(files: dict) -> None:
    """A golden that recorded an empty or degenerate dataset would pass `check` forever while proving nothing."""
    subsample = files["convert_golden.jsonl"].splitlines() + files["convert_golden.val.jsonl"].splitlines()
    keep = files["convert_golden_keep.jsonl"].splitlines() + files["convert_golden_keep.val.jsonl"].splitlines()
    assert len(subsample) > 20, f"convert golden has only {len(subsample)} examples"
    assert len(keep) > len(subsample), "subsample must drop mid-wander targets that keep retains"
    blob = "\n".join(keep)
    for marker in (
        "Click the ",
        "action: confirm(",
        "action: handoff(",
        "action: done()",
        "action: drag(",
        "action: navigate(",
        "action: approval_event()",
    ):
        assert marker in blob, f"convert golden never exercised {marker!r}"
    assert "task_0005_eval" not in blob, "an excluded_reason trajectory reached the dataset"
    assert "task_0004_cart" not in blob, "an unverified trajectory reached the dataset"
    assert "task_0002_login" in blob, "--include-replay did not admit the replay trajectory"


ACTION_SHAPES = (
    {"type": "click", "x": 640, "y": 360, "button": "left"},
    {"type": "click", "x": 640, "y": 360, "button": "right"},
    {"type": "click", "x": 12, "y": 34},
    {"type": "double_click", "x": 160, "y": 115},
    {"type": "move", "x": 300, "y": 300},
    {"type": "type", "text": "laptop stand"},
    {"type": "type", "text": "<secret:PASSWORD>"},
    {"type": "keypress", "keys": ["Enter"]},
    {"type": "keypress", "keys": ["Control", "a"]},
    {"type": "keypress", "keys": ["ArrowDown", "ArrowDown", "ArrowDown"]},
    {"type": "scroll", "x": 640, "y": 360, "scroll_x": 0, "scroll_y": 500},
    {"type": "scroll", "x": 640, "y": 360, "scroll_x": -120, "scroll_y": -240},
    {"type": "drag", "path": [{"x": 100, "y": 200}, {"x": 400, "y": 500}]},
    {"type": "drag", "path": [{"x": 100, "y": 200}, {"x": 250, "y": 350}, {"x": 400, "y": 500}]},
    {"type": "navigate", "url": "https://synth.test/app/board"},
    {"type": "wait"},
    {"type": "screenshot"},
    {"type": "confirm", "proposed_action": {"type": "click", "x": 640, "y": 400}, "risk_category": "FINANCIAL_RISK"},
    {"type": "confirm", "proposed_action": {"type": "submit"}, "risk_category": "DATA_LOSS"},
    {"type": "confirm"},
    {"type": "handoff", "reason": "login"},
    {"type": "handoff"},
    {"type": "approval_event"},
    {"type": "done"},
)

# Every accepted spelling from the student-eval grammar table, plus the spellings it must reject.
GRAMMAR_SPELLINGS = (
    "action: click(<ref>(640, 360)</ref>, button=left)",
    "action: click(<ref>(640, 360)</ref>, button=right)",
    "action: click(<ref>(640, 360)</ref>)",
    "action: click((640, 360))",
    "action: click(640, 360)",
    "action: click(640, 360, left)",
    "action: click(640, 360, right)",
    "click(640, 360)",
    "- action: click(640, 360)",
    "`action: click(640, 360)`",
    "action: click(640.4, 360.6)",
    "action: click(<ref>(640, 360)</ref>, button=left, note=ignored)",
    "action: double_click(<ref>(160, 115)</ref>)",
    "action: double_click((160, 115))",
    "action: double_click(160, 115)",
    "action: move(<ref>(300, 300)</ref>)",
    "action: move((300, 300))",
    "action: move(300, 300)",
    "action: drag(<ref>(100, 200)</ref> -> <ref>(400, 500)</ref>)",
    "action: drag((100, 200) -> (400, 500))",
    "action: drag(100, 200 -> 400, 500)",
    "action: drag((100, 200), (400, 500))",
    "action: scroll(<ref>(640, 360)</ref>, scrollX=0, scrollY=500)",
    "action: scroll(<ref>(640, 360)</ref>, scroll_x=0, scroll_y=500)",
    "action: scroll(640, 360, dx=0, dy=500)",
    "action: scroll(640, 360, 0, 500)",
    "action: scroll(640, 360)",
    'action: type("laptop stand")',
    "action: type('laptop stand')",
    'action: keypress(["Enter"])',
    'action: keypress("Enter")',
    "action: keypress(Enter)",
    "action: keypress(Control, a)",
    "action: keypress(Control+a)",
    "action: wait()",
    "action: wait",
    "action: done()",
    "action: done",
    "action: screenshot()",
    "action: screenshot",
    'action: navigate("https://synth.test/app")',
    'action: navigate(url="https://synth.test/app")',
    "action: navigate(https://synth.test/app)",
    "https://synth.test/app",
    "action: handoff(reason=login)",
    "action: confirm(propose=click(<ref>(640, 400)</ref>), risk=FINANCIAL_RISK)",
    "action: type(hello world)",
    "action: type(a, b)",
    "action: scroll(500)",
    "action: click(100)",
    "action: drag((10, 20))",
    "action: keypress()",
    "action: navigate()",
    "action: navigate(example.com)",
    "I will now click the button: action: click(640, 360) and then stop",
    "First I need to find the search box on this page.",
    "<think>Reasoning here.</think>\naction: click(640, 360)",
    "Reasoning without an opening tag.</think>\naction: click(640, 360)",
    'action: click(640, 360)\naction: type("hello")',
)


def build_action_roundtrip(root: Path, corpus: Path | None) -> dict:
    """Formatter, history summary, RL decoder and the tolerant student parser over every shape and spelling."""
    shapes = []
    for action in ACTION_SHAPES:
        line = format_action_for_training(action)
        decoded = decode_step(line)
        shapes.append(
            {
                "action": action,
                "format_action_for_training": line,
                "format_action_summary": format_action_summary(action),
                "decode_step": {**dataclasses.asdict(decoded), "ok": decoded.ok},
            }
        )
    spellings = []
    for text in GRAMMAR_SPELLINGS:
        parsed = parse_student_output(text)
        spellings.append(
            {
                "text": text,
                "parse_student_output": {**dataclasses.asdict(parsed), "parse_failed": parsed.parse_failed},
            }
        )
    return {"action_roundtrip.json": dumps({"shapes": shapes, "spellings": spellings})}


def build_registry_schema(root: Path, corpus: Path | None) -> dict:
    """The SQL a fresh registry creates, plus the status enums and stage maps the state machine is built from."""
    with tempfile.TemporaryDirectory(prefix="regression_registry_") as tmp:
        db_path = Path(tmp) / "registry.db"
        registry = RepoRegistry(db_path=str(db_path))
        registry.close()
        with sqlite3.connect(db_path) as connection:
            rows = connection.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
            ).fetchall()
    schema = "".join(f"{row[0].strip()};\n" for row in rows)
    status_values = {
        "RepoStatus": {member.name: member.value for member in schemas.RepoStatus},
        "FailureCategory": {member.name: member.value for member in schemas.FailureCategory},
        "STAGE_TO_INPUT_STATUS": {str(k): v.value for k, v in schemas.STAGE_TO_INPUT_STATUS.items()},
        "STAGE_TO_PASSED_STATUS": {str(k): v.value for k, v in schemas.STAGE_TO_PASSED_STATUS.items()},
        "STAGE_TO_REJECTED_STATUS": {str(k): v.value for k, v in schemas.STAGE_TO_REJECTED_STATUS.items()},
    }
    return {"registry_schema.sql": schema, "status_values.json": dumps(status_values)}


PRODUCER_FAMILIES = (("site", "site_*"), ("repo", "repo_*"), ("mind2web", "task_mind2web_*"))


def build_metadata_keys(root: Path, corpus: Path | None) -> dict:
    """Per producer family, the key vocabulary of the real corpus — keys only, never a recorded value."""
    if corpus is None or not corpus.is_dir():
        return {}
    families = {}
    for family, pattern in PRODUCER_FAMILIES:
        metadata_keys, trajectory_keys, step_keys = set(), set(), set()
        action_keys: dict = {}
        for directory in sorted(corpus.glob(pattern)):
            targets = [directory] if directory.name.startswith("task_") else sorted(directory.glob("task_*"))
            for task_dir in targets:
                trajectory_file = task_dir / "trajectory.json"
                if not trajectory_file.is_file():
                    continue
                trajectory = _load_json(trajectory_file)
                metadata_keys |= set(_load_json(task_dir / "metadata.json"))
                trajectory_keys |= set(trajectory)
                for entry in trajectory.get("steps", []) or []:
                    step_keys |= set(entry)
                    for action in entry.get("actions", []) or []:
                        action_keys.setdefault(action.get("type", "?"), set()).update(action)
        families[family] = {
            "metadata_keys": sorted(metadata_keys),
            "trajectory_keys": sorted(trajectory_keys),
            "step_keys": sorted(step_keys),
            "action_keys": {name: sorted(keys) for name, keys in sorted(action_keys.items())},
        }
    return {"metadata_keys.json": dumps(families)}


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


# Fixture name -> argv tail reaching the entrypoint; names stay package-free so a move does not rename a fixture.
CLI_ENTRYPOINTS = (
    ("train_qwen", ["-m", "cua.sft"]),
    ("train_qwen.convert", ["-m", "cua.sft", "convert"]),
    ("train_qwen.train", ["-m", "cua.sft", "train"]),
    ("eval_screenspot", ["-m", "cua.eval.screenspot"]),
    ("eval_screenspot.run", ["-m", "cua.eval.screenspot", "run"]),
    ("rl_train", ["-m", "cua.rl.train"]),
    ("rl_train.print-commands", ["-m", "cua.rl.train", "print-commands"]),
    ("rl_train.make-dataset", ["-m", "cua.rl.train", "make-dataset"]),
    ("rl_train.run", ["-m", "cua.rl.train", "run"]),
    ("record_trajectories_from_gpt_api", ["-m", "cua.path_a.recorder"]),
    ("load_tasks", ["-m", "cua.path_a.load_mind2web_tasks"]),
    ("render_registry", ["experiments/render_registry.py"]),
    ("remote_train", ["-m", "cua.cloud.vast"]),
    ("remote_train.train", ["-m", "cua.cloud.vast", "train"]),
    ("remote_train.run", ["-m", "cua.cloud.vast", "run"]),
    ("remote_train.eval", ["-m", "cua.cloud.vast", "eval"]),
    ("remote_train.grpo", ["-m", "cua.cloud.vast", "grpo"]),
    ("remote_train.serve", ["-m", "cua.cloud.vast", "serve"]),
    ("remote_train.search", ["-m", "cua.cloud.vast", "search"]),
    ("remote_train.cleanup", ["-m", "cua.cloud.vast", "cleanup"]),
    ("remote_train.release", ["-m", "cua.cloud.vast", "release"]),
    ("orchestrator", ["-m", "cua.pipeline.orchestrator"]),
    ("student_eval", ["-m", "cua.eval.student"]),
    ("student_eval.run", ["-m", "cua.eval.student", "run"]),
    ("student_eval.judge", ["-m", "cua.eval.student", "judge"]),
    ("student_eval.compare", ["-m", "cua.eval.student", "compare"]),
    ("student_eval.sources", ["-m", "cua.eval.student", "sources"]),
    ("public_capture", ["-m", "cua.pipeline.public_capture"]),
    ("capture_logingated", ["-m", "cua.pipeline.capture_logingated"]),
    ("provision_human", ["-m", "cua.pipeline.provision_human"]),
    ("probe_sandbox", ["-m", "cua.pipeline.probe_sandbox"]),
    ("probe_capturable", ["-m", "cua.pipeline.probe_capturable"]),
    ("polish_pass", ["-m", "cua.pipeline.polish_pass"]),
    ("relabel_recoverable", ["-m", "cua.pipeline.relabel_recoverable"]),
    ("outcome_relabel", ["-m", "cua.pipeline.outcome_relabel"]),
    ("wall_supervision", ["-m", "cua.pipeline.wall_supervision"]),
    ("instruction_style_pass", ["-m", "cua.pipeline.instruction_style_pass"]),
    ("regenerate_supervision", ["-m", "cua.pipeline.regenerate_supervision"]),
    ("verify_pattern_scan", ["-m", "cua.pipeline.verify_pattern_scan"]),
    ("corpus_audit", ["-m", "cua.pipeline.corpus_audit"]),
    ("register_real_site", ["-m", "cua.pipeline.register_real_site"]),
    ("notifier", ["-m", "cua.pipeline.notifier"]),
    ("sandbox", ["-m", "cua.pipeline.sandbox"]),
    ("mail_service", ["-m", "cua.pipeline.mail_service"]),
    ("rl_tasks", ["-m", "cua.rl.tasks"]),
    ("rl_rollout", ["-m", "cua.rl.rollout"]),
    ("rl_eval", ["-m", "cua.rl.eval"]),
    ("rl_service", ["-m", "cua.rl.service"]),
    ("rl_analysis", ["-m", "cua.rl.analysis"]),
    ("rl_smoke", ["-m", "cua.rl.smoke"]),
)
NO_ARGPARSE_NOTE = (
    "NO ARGPARSE SURFACE — this entrypoint reads sys.argv directly, so --help is not "
    "captured; running it would start Docker, network or LLM work.\n"
)
# Named, never probed: running one of these for a --help it lacks boots Docker, the network or a capture.
NO_ARGPARSE_ENTRYPOINTS = frozenset({"register_real_site", "notifier", "sandbox", "mail_service", "rl_smoke"})


def build_cli_surface(root: Path, corpus: Path | None) -> dict:
    files = {}
    for name, argv in CLI_ENTRYPOINTS:
        if name in NO_ARGPARSE_ENTRYPOINTS:
            files[f"cli_surface/{name}.txt"] = NO_ARGPARSE_NOTE
            continue
        text = run_capture([sys.executable, *argv, "--help"], root)
        files[f"cli_surface/{name}.txt"] = strip_usage(normalise(text, root))
    return files


_MAKE_TARGET_RE = re.compile(r"^([A-Za-z0-9_-]+):", re.MULTILINE)
# $(MAKE) expands to the absolute path of the running make, which differs per machine.
_MAKE_BINARY_RE = re.compile(r"\S*/make\b")


def build_make_n(root: Path, corpus: Path | None) -> dict:
    """Every Makefile target's expanded recipe, so a renamed script or flag shows up as a diff."""
    targets = []
    for name in _MAKE_TARGET_RE.findall((root / "Makefile").read_text()):
        if name not in targets:
            targets.append(name)
    sections = []
    for target in targets:
        text = _MAKE_BINARY_RE.sub("<make>", run_capture(["make", "-n", target], root))
        sections.append(f"===== {target} =====\n{normalise(text, root)}")
    return {"make_n.txt": "\n".join(sections)}


# The smoke's mock episode config, minus the temp-file path it points at; see student_eval_smoke.run_episode_smoke.
SMOKE_EVAL_CONFIG = dict(
    task_source="held_out",
    label="smoke",
    student="mock",
    coord_space="viewport_abs",
    max_steps=len(REPLAY_STEPS) + 3,
    judge=False,
    judge_mid_screenshots=2,
    episode_timeout_s=600,
    resume=False,
)
VOLATILE_FINGERPRINT_KEYS = ("git_commit",)


def build_eval_fingerprint(root: Path, corpus: Path | None) -> dict:
    fingerprint = harness_fingerprint(EvalConfig(**SMOKE_EVAL_CONFIG), TASK_SOURCES["held_out"], {})
    removed = [key for key in VOLATILE_FINGERPRINT_KEYS if key in fingerprint]
    stable = {key: value for key, value in fingerprint.items() if key not in removed}
    return {
        "eval_fingerprint.json": dumps(
            {"config": SMOKE_EVAL_CONFIG, "removed_volatile_keys": removed, "harness_fingerprint": stable}
        )
    }


COMPARE_RUNS = ("eval_archive/online_mind2web/baseline", "eval_archive/online_mind2web/after")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")


def build_eval_compare(root: Path, corpus: Path | None) -> dict:
    """Two archived runs compared; --output keeps compare's own comparison.md out of the checkout."""
    with tempfile.TemporaryDirectory(prefix="regression_compare_") as tmp:
        text = run_capture(
            [
                sys.executable,
                "-m",
                "cua.eval.student",
                "compare",
                *COMPARE_RUNS,
                "--output",
                str(Path(tmp) / "comparison.md"),
            ],
            root,
        )
        text = normalise(text, root, ((tmp, "<tmp>"),))
    return {"eval_compare.md": _TIMESTAMP_RE.sub("<timestamp>", text)}


# Fixtures that need the real corpus, so `check` reports SKIP instead of a failure where it is absent.
CORPUS_DEPENDENT = ("metadata_keys.json",)

FIXTURE_BUILDERS = (
    ("convert_golden", build_convert_golden),
    ("action_roundtrip", build_action_roundtrip),
    ("registry_schema", build_registry_schema),
    ("metadata_keys", build_metadata_keys),
    ("cli_surface", build_cli_surface),
    ("make_n", build_make_n),
    ("eval_fingerprint", build_eval_fingerprint),
    ("eval_compare", build_eval_compare),
)


def generate(root: Path, corpus: Path | None, only=()) -> dict:
    """Every managed fixture as relative path -> exact text; build writes this and check diffs against it."""
    files = {}
    for name, builder in FIXTURE_BUILDERS:
        if only and name not in only:
            continue
        files.update(builder(root, corpus))
    return files


def stored_fixtures() -> dict:
    files = {}
    for path in sorted(FIXTURES.rglob("*")):
        if path.is_file() and path.name not in UNMANAGED:
            files[str(path.relative_to(FIXTURES))] = path.read_text()
    return files


def command_build(args) -> int:
    root = repo_root()
    files = generate(root, args.corpus, tuple(args.only))
    for relative, text in sorted(files.items()):
        target = FIXTURES / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    print(f"wrote {len(files)} fixture files under {FIXTURES}")
    if args.corpus is None or not args.corpus.is_dir():
        print(f"NOTE: no corpus at {args.corpus} — metadata_keys.json was not regenerated")
    return 0


def command_check(args) -> int:
    root = repo_root()
    fresh = generate(root, args.corpus, tuple(args.only))
    stored = stored_fixtures()
    if args.only:
        stored = {name: text for name, text in stored.items() if name in fresh}
    failures = []
    for relative in sorted(set(stored) | set(fresh)):
        if relative not in fresh:
            if relative in CORPUS_DEPENDENT and (args.corpus is None or not args.corpus.is_dir()):
                print(f"SKIP  {relative} (no corpus at {args.corpus})")
                continue
            failures.append((relative, "stored fixture was not regenerated"))
            continue
        if relative not in stored:
            failures.append((relative, "no stored fixture — run `make fixtures-build`"))
            continue
        if stored[relative] != fresh[relative]:
            diff = difflib.unified_diff(
                stored[relative].splitlines(),
                fresh[relative].splitlines(),
                fromfile=f"stored/{relative}",
                tofile=f"today/{relative}",
                lineterm="",
            )
            failures.append((relative, "\n".join(list(diff)[: args.diff_lines])))
        else:
            print(f"ok    {relative}")
    for relative, detail in failures:
        print(f"FAIL  {relative}\n{detail}")
    print(f"{len(fresh) - len(failures)}/{len(fresh)} fixtures match")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Record and verify the pre-refactor behaviour fixtures")
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument(
        "--corpus",
        type=Path,
        default=repo_root() / "trajectories",
        help="Real corpus scanned for metadata_keys.json (skipped when absent)",
    )
    parser.add_argument("--only", action="append", default=[], help="Limit to one fixture group (repeatable)")
    parser.add_argument("--diff-lines", type=int, default=60, help="Diff lines printed per failing fixture")
    args = parser.parse_args(argv)
    return command_build(args) if args.command == "build" else command_check(args)


if __name__ == "__main__":
    sys.exit(main())
