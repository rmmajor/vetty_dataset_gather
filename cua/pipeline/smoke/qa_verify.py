"""Scripted (no-LLM, no-Docker, no-network) check that the qa_verify task style reaches the proposer, teacher, verifier, metadata and site config."""

import asyncio
import json
import logging
import sys
from pathlib import Path

from cua.pipeline import app_secrets, task_proposer
from cua.pipeline.lanes import capture_lane, public
from cua.pipeline.sites.registry import SITES_DIR
from cua.pipeline.smoke._harness import run_checks, with_tempdir
from cua.pipeline.teacher import policy
from cua.pipeline.teacher import run as teacher_run
from cua.pipeline.teacher import verify as teacher_verify

logger = logging.getLogger(__name__)

STYLE = "qa_verify"
CONFIG_PATH = SITES_DIR / "public_sites_qa_verify.json"
BENCHMARK_SUBJECTS = ("refapp", "saucedemo", "linkding")
SESSION_INJECTION_UNSUPPORTED = ("affine",)
CREDENTIAL_KEYS = {"password", "cookies", "origins", "token", "storage_state"}
PROPOSER_MUST_MENTION = ("Verify", "reload", "sign out", "load more", "page 2")
TEACHER_MUST_MENTION = ("reload", "toast", "failed")
VERIFIER_MUST_MENTION = ("reload", "FINAL screenshot", "success=false")
VERIFY_FORMAT_FIELDS = {
    "instruction": "Toggling email notifications persists after save and reload",
    "n_steps": 3,
    "expected_actions": 8,
    "step_summary": '1. click on button "Save"',
    "agent_outcome": "completed",
    "agent_summary": "done",
    "image_labels": "final screenshot",
}
TASK = {
    "id": "email-toggle-persists",
    "difficulty": "medium",
    "expected_actions": 10,
    "instruction": "Toggling Email notifications in Settings persists after save and reload. Verify the unchecked state survives a page reload.",
}
PROPOSER_REPLY = json.dumps(
    {
        "app_health": {"usable": True, "notes": "ok"},
        "tasks": [
            {
                "id": TASK["id"],
                "instruction": TASK["instruction"],
                "difficulty": "medium",
                "expected_actions": 10,
                "action_kinds": ["click"],
            }
        ],
    }
)
FAILED_VERDICT = json.dumps(
    {
        "criteria": ["toggle saved", "re-checked after reload"],
        "criteria_met": [True, False],
        "reasoning": "stopped on the toast",
        "success": False,
        "efficiency": "acceptable",
    }
)


class _FakeDriver:
    """Records what the capture code hands the driver; no container behind it."""

    def __init__(self):
        self.task_meta = None
        self.probe_auth_config = None

    def session(self, task_id, app_url, instruction="", task_meta=None, auth_config=None, **kwargs):
        if task_id == "context_probe":
            self.probe_auth_config = auth_config
        self.task_meta = task_meta
        return {"url": app_url, "snapshot": "(no elements)"}

    def screenshot_b64(self):
        return None

    def affordances(self):
        return {}

    def finish(self, outcome, summary):
        return None


class _FakeSandbox:
    def exec_command(self, command):
        return ""


def _capturing_run_agent(captured: list, reply: str):
    async def run_agent(prompt, options, **kwargs):
        if isinstance(prompt, str):
            captured.append(prompt)
        else:
            async for message in prompt:
                captured.append(message["message"]["content"][0]["text"])
        return reply

    return run_agent


def _with_stubbed_agent(module, run_agent, fn):
    original = module.run_agent
    module.run_agent = run_agent
    try:
        return fn()
    finally:
        module.run_agent = original


def _trajectory() -> dict:
    steps = [
        {
            "step": 1,
            "screenshot": "step_001_screenshot.png",
            "role": "checkbox",
            "name": "Email notifications",
            "thought": "I uncheck the box.",
            "actions": [{"type": "click", "x": 300, "y": 200, "button": "left"}],
        },
        {
            "step": 2,
            "screenshot": "step_002_screenshot.png",
            "role": "button",
            "name": "Save",
            "thought": "I save the settings.",
            "actions": [{"type": "click", "x": 500, "y": 400, "button": "left"}],
        },
        {
            "step": 3,
            "screenshot": "step_003_screenshot.png",
            "role": "",
            "name": "",
            "thought": "The toast says saved.",
            "actions": [{"type": "done"}],
        },
    ]
    return {
        "task_id": TASK["id"],
        "instruction": TASK["instruction"],
        "start_url": "https://app.test/settings",
        "steps": steps,
        "total_steps": len(steps),
        "success": False,
    }


def _write_capture(traj_dir: Path, task_meta: dict) -> None:
    traj_dir.mkdir(parents=True)
    (traj_dir / "trajectory.json").write_text(json.dumps(_trajectory(), indent=2))
    (traj_dir / "capture_meta.json").write_text(
        json.dumps(
            {
                "recorder": "browser_driver",
                "agent_outcome": "completed",
                "agent_summary": "done",
                "task_meta": task_meta,
            },
            indent=2,
        )
    )


def _missing_phrases(text: str, phrases: tuple, label: str) -> list[str]:
    return [f"{label}: never mentions {phrase!r}" for phrase in phrases if phrase.lower() not in text.lower()]


def check_proposer() -> list[str]:
    block = task_proposer._STYLE_BLOCKS.get(STYLE)
    if not block:
        return [f"proposer: no _STYLE_BLOCKS entry for {STYLE!r}"]
    failures = _missing_phrases(block, PROPOSER_MUST_MENTION, "proposer block")
    repo = {"full_name": "public:smoke-site", "description": "smoke"}
    for style, expect in ((STYLE, True), ("", False)):
        captured: list[str] = []
        proposal = _with_stubbed_agent(
            task_proposer,
            _capturing_run_agent(captured, PROPOSER_REPLY),
            lambda: asyncio.run(task_proposer.propose_tasks(repo, "(snapshot)", None, "", "", task_style=style)),
        )
        injected = bool(captured) and block in captured[0]
        if injected != expect:
            failures.append(f"proposer: style block injected={injected} for task_style={style!r} (expected {expect})")
        if not proposal or [t["id"] for t in proposal.get("tasks", [])] != [TASK["id"]]:
            failures.append(f"proposer: output schema no longer parses for task_style={style!r}: {proposal!r}")
    return failures


def check_probe() -> list[str]:
    """The proposer must see the SIGNED-IN app: the context probe has to carry the site's session."""
    driver = _FakeDriver()
    site = {"name": "smoke-site", "start_url": "https://app.test/", "domain": "smoke", "task_style": STYLE}
    auth_config = {"auth_mode": "pre_session", "storage_state": {"cookies": [], "origins": []}}
    proposal = _with_stubbed_agent(
        task_proposer,
        _capturing_run_agent([], PROPOSER_REPLY),
        lambda: asyncio.run(public.propose_for_site(driver, _FakeSandbox(), site, [], None, auth_config=auth_config)),
    )
    failures = []
    if driver.probe_auth_config != auth_config:
        failures.append(
            f"probe: context_probe session got auth_config={driver.probe_auth_config!r}, expected the site's session"
        )
    if not proposal or not proposal.get("tasks"):
        failures.append(f"probe: proposal lost: {proposal!r}")
    return failures


def check_teacher() -> list[str]:
    block = policy.TEACHER_STYLE_BLOCKS.get(STYLE)
    if not block:
        return [f"teacher: no TEACHER_STYLE_BLOCKS entry for {STYLE!r}"]
    failures = _missing_phrases(block, TEACHER_MUST_MENTION, "teacher block")
    for style, expect in ((STYLE, True), ("", False)):
        rendered = policy.render_policy_block("autonomous", "pre_session", ["email", "password"], task_style=style)
        if (block in rendered) != expect:
            failures.append(f"teacher: render_policy_block injected={block in rendered} for task_style={style!r}")
        driver = _FakeDriver()
        captured: list[str] = []
        _with_stubbed_agent(
            teacher_run,
            _capturing_run_agent(captured, ""),
            lambda: asyncio.run(
                teacher_run.run_teacher_task(
                    driver, TASK, "https://app.test/", auth_mode="pre_session", task_style=style
                )
            ),
        )
        injected = bool(captured) and block in captured[0]
        if injected != expect:
            failures.append(f"teacher: prompt injected={injected} for task_style={style!r} (expected {expect})")
        if (driver.task_meta or {}).get("task_style") != style:
            failures.append(
                f"teacher: task_meta.task_style={driver.task_meta and driver.task_meta.get('task_style')!r}, expected {style!r}"
            )
    return failures


@with_tempdir
def check_verifier(work_dir: Path) -> list[str]:
    rules = teacher_verify.VERIFY_STYLE_RULES.get(STYLE)
    if not rules:
        return [f"verifier: no VERIFY_STYLE_RULES entry for {STYLE!r}"]
    failures = _missing_phrases(rules, VERIFIER_MUST_MENTION, "verifier rules")
    try:
        default_prompt = teacher_verify.VERIFY_PROMPT.format(style_rules="", **VERIFY_FORMAT_FIELDS)
        if rules in default_prompt:
            failures.append("verifier: style rules leak into the default VERIFY_PROMPT")
    except (KeyError, IndexError) as e:
        failures.append(f"verifier: VERIFY_PROMPT.format with empty style_rules failed: {e!r}")
    if "style_rules" in teacher_verify.GUARDED_VERIFY_PROMPT:
        failures.append("verifier: GUARDED_VERIFY_PROMPT must stay untouched")
    for style, expect in ((STYLE, True), ("", False)):
        traj_dir = work_dir / f"verify_{style or 'default'}"
        _write_capture(traj_dir, {"expected_actions": 8, "autonomy": "autonomous", "task_style": style})
        captured: list[str] = []
        _with_stubbed_agent(
            teacher_verify,
            _capturing_run_agent(captured, FAILED_VERDICT),
            lambda: asyncio.run(teacher_verify.verify_trajectory(traj_dir)),
        )
        injected = bool(captured) and rules in captured[0]
        if injected != expect:
            failures.append(f"verifier: rules injected={injected} for task_style={style!r} (expected {expect})")
    return failures


@with_tempdir
def check_metadata(work_dir: Path) -> list[str]:
    traj_dir = work_dir / "finalize"
    _write_capture(
        traj_dir,
        {
            "expected_actions": 8,
            "autonomy": "autonomous",
            "auth_mode": "pre_session",
            "env": "live",
            "task_style": STYLE,
        },
    )
    site = {
        "name": "smoke-site",
        "start_url": "https://app.test/",
        "domain": "smoke",
        "task_style": STYLE,
        "auth_mode": "pre_session",
        "storage_state_ref": "smoke-ref",
    }
    state = {"agent_outcome": "completed", "agent_summary": "done"}
    _with_stubbed_agent(
        teacher_verify,
        _capturing_run_agent([], FAILED_VERDICT),
        lambda: asyncio.run(capture_lane.verify_and_finalize(traj_dir, site, TASK, state, None)),
    )
    metadata = json.loads((traj_dir / "metadata.json").read_text())
    if metadata.get("task_style") != STYLE:
        return [f"metadata: task_style={metadata.get('task_style')!r}, expected {STYLE!r}"]
    return []


def check_config() -> list[str]:
    if not CONFIG_PATH.exists():
        return [f"config: {CONFIG_PATH.name} is missing"]
    raw = CONFIG_PATH.read_text()
    failures = [f"config: benchmark subject {s!r} appears" for s in BENCHMARK_SUBJECTS if s in raw.lower()]
    sites = json.loads(raw).get("sites") or []
    if not sites:
        failures.append("config: no sites")
    names = [s.get("name", "") for s in sites]
    if len(set(names)) != len(names):
        failures.append(f"config: duplicate site names {sorted(n for n in names if names.count(n) > 1)}")
    sessions_present = app_secrets._STORAGE_STATE_DIR.is_dir()
    if not sessions_present:
        logger.warning("config: credentials/storage_state is absent — storage_state_ref resolution not checked")
    for site in sites:
        name = site.get("name", "?")
        if site.get("task_style") != STYLE:
            failures.append(f"config: {name} task_style={site.get('task_style')!r}")
        failures.extend(f"config: {name} lacks {key!r}" for key in ("name", "start_url", "domain") if not site.get(key))
        if any(name == u or name.startswith(f"{u}-") for u in SESSION_INJECTION_UNSUPPORTED):
            failures.append(f"config: {name} cannot take an injected session")
        leaked = CREDENTIAL_KEYS & set(site)
        if leaked:
            failures.append(f"config: {name} carries credential material {sorted(leaked)}")
        ref = site.get("storage_state_ref")
        if site.get("auth_mode") == "pre_session" and not ref:
            failures.append(f"config: {name} is pre_session without a storage_state_ref")
        elif ref and sessions_present and app_secrets.load_storage_state(ref) is None:
            failures.append(f"config: {name} storage_state_ref {ref!r} does not resolve")
    return failures


CHECKS = [check_proposer, check_probe, check_teacher, check_verifier, check_metadata, check_config]


if __name__ == "__main__":
    sys.exit(run_checks(CHECKS, "qa_verify smoke test"))
