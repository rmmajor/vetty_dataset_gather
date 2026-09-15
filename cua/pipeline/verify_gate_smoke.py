"""Scripted (no-LLM, no-Docker, no-network) check that the capture success gate never rejects a truthful trajectory."""

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

from cua.pipeline import agent_capture

logger = logging.getLogger(__name__)


def _click_step(index: int, x: int, y: int) -> dict:
    return {
        "step": index,
        "screenshot": f"step_{index:03d}_screenshot.png",
        "role": "button",
        "name": f"control {x}",
        "thought": "I click the next control to move toward the goal.",
        "actions": [{"type": "click", "x": x, "y": y, "button": "left"}],
    }


def _done_step(index: int) -> dict:
    return {
        "step": index,
        "screenshot": f"step_{index:03d}_screenshot.png",
        "role": "",
        "name": "",
        "thought": "The goal state is on screen, so I stop here.",
        "actions": [{"type": "done"}],
    }


def _purposeful(count: int, start: int = 1) -> list[dict]:
    return [_click_step(start + i, 100 + i * 7, 200 + i * 3) for i in range(count)]


def _repeated(count: int, start: int = 1) -> list[dict]:
    return [_click_step(start + i, 640, 360) for i in range(count)]


def _trajectory(steps: list[dict]) -> dict:
    return {
        "task_id": "gate-smoke",
        "instruction": "Set up a kanban board with priority and due-date fields",
        "start_url": "http://app.test/",
        "steps": steps + [_done_step(len(steps) + 1)],
        "total_steps": len(steps) + 1,
        "success": False,
    }


# Every case is data: a recorded trajectory shape plus the verdict the judge returned for it.
CASES = [
    {
        "name": "verified + efficient trains",
        "steps": _purposeful(8),
        "expected_actions": 8,
        "verdict": {"success": True, "efficiency": "efficient", "reasoning": "final screenshot shows the board"},
        "trains": True,
        "efficiency_ratio": 1.0,
    },
    {
        "name": "verified + graded meandering at 3.0x still trains (AFFiNE weekly-kanban-tags regression)",
        "steps": _purposeful(54),
        "expected_actions": 18,
        "verdict": {
            "success": True,
            "efficiency": "meandering",
            "reasoning": "the final screenshot confirms the goal was met",
        },
        "trains": True,
        "efficiency_ratio": 3.0,
    },
    {
        "name": "verified + meandering with no expected_actions still trains",
        "steps": _purposeful(30),
        "expected_actions": 0,
        "verdict": {"success": True, "efficiency": "meandering", "reasoning": "goal met over a long route"},
        "trains": True,
        "efficiency_ratio": None,
    },
    {
        "name": "verified + a four-times-retried control inside real work still trains",
        "steps": _purposeful(20) + _repeated(4, start=21),
        "expected_actions": 9,
        "verdict": {"success": True, "efficiency": "meandering", "reasoning": "tag creation retried, goal still met"},
        "trains": True,
        "efficiency_ratio": 2.67,
    },
    {
        "name": "verifier failure never trains",
        "steps": _purposeful(12),
        "expected_actions": 12,
        "verdict": {"success": False, "efficiency": "efficient", "reasoning": "the board was never created"},
        "trains": False,
        "efficiency_ratio": 1.0,
    },
    {
        "name": "degenerate loop never trains even when the judge claims success",
        "steps": _repeated(8),
        "expected_actions": 6,
        "verdict": {"success": True, "efficiency": "acceptable", "reasoning": "looks done to me"},
        "trains": False,
        "efficiency_ratio": 1.33,
    },
]


def _stub_verifier(verdict: dict):
    async def run_agent(prompt_stream, options, **kwargs):
        await prompt_stream.aclose()
        return json.dumps(verdict)

    return run_agent


def _run_case(case: dict, work_dir: Path) -> dict:
    traj_dir = work_dir / case["name"][:40].replace(" ", "_")
    traj_dir.mkdir(parents=True)
    (traj_dir / "trajectory.json").write_text(json.dumps(_trajectory(case["steps"]), indent=2))
    (traj_dir / "capture_meta.json").write_text(
        json.dumps(
            {
                "recorder": "browser_driver",
                "agent_outcome": "completed",
                "agent_summary": "done",
                "task_meta": {"expected_actions": case["expected_actions"], "autonomy": "autonomous"},
            },
            indent=2,
        )
    )

    original_run_agent = agent_capture.run_agent
    agent_capture.run_agent = _stub_verifier(case["verdict"])
    try:
        returned = asyncio.run(agent_capture.verify_trajectory(traj_dir))
    finally:
        agent_capture.run_agent = original_run_agent

    written = json.loads((traj_dir / "trajectory.json").read_text())
    recorded = json.loads((traj_dir / "capture_meta.json").read_text())["verifier"]
    return {
        "returned_success": returned["final_success"],
        "written_success": written["success"],
        "efficiency": recorded.get("efficiency"),
        "efficiency_ratio": recorded.get("efficiency_ratio"),
    }


def run_smoke() -> bool:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in CASES:
            actual = _run_case(case, Path(tmp))
            expected = {
                "returned_success": case["trains"],
                "written_success": case["trains"],
                "efficiency": case["verdict"]["efficiency"],
                "efficiency_ratio": case["efficiency_ratio"],
            }
            for key, want in expected.items():
                got = actual[key]
                if got != want:
                    failures.append(f"{case['name']}: {key} expected {want!r}, got {got!r}")
            status = "FAIL" if any(case["name"] in f for f in failures) else "pass"
            logger.info(
                f"[{status}] {case['name']} -> trains={actual['written_success']} "
                f"efficiency={actual['efficiency']!r} ratio={actual['efficiency_ratio']}"
            )

    for failure in failures:
        logger.error(f"  {failure}")
    return not failures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ok = run_smoke()
    logger.info(f"Verify-gate smoke test {'PASSED' if ok else 'FAILED'} ({len(CASES)} cases)")
    sys.exit(0 if ok else 1)
