"""One rollout episode over a driver slot: observe → decode → execute → record → reward."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from cua.eval.student.blocks import detect_block
from cua.pipeline.driver_session import DriverError
from cua.pipeline.instrumentation import RESULTS_DIR
from cua.prompt.format import history_entries_for_step
from cua.rl import checks
from cua.rl.actions import decode_step
from cua.rl.executor import control_index, to_sequence_payload, validate_actions
from cua.rl.judge import cached_verdict, judge_success
from cua.rl.observation import build_observation
from cua.rl.rewards import compute_reward
from cua.trajectory.schema import safe_task_id

logger = logging.getLogger(__name__)

OUTCOME_BY_REASON = {"done": "completed"}

# Second-level public suffixes (example.co.uk) so the registrable domain keeps three labels there.
_SECOND_LEVEL_SUFFIXES = {"co", "com", "org", "net", "gov", "ac", "edu"}


def registrable_domain(netloc: str) -> str:
    host = netloc.rsplit("@", 1)[-1].split(":")[0].lower()
    labels = [label for label in host.split(".") if label]
    if len(labels) >= 3 and labels[-2] in _SECOND_LEVEL_SUFFIXES and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


@dataclass
class EpisodeConfig:
    max_steps: int = 30
    max_consecutive_parse_failures: int = 3
    loop_guard_terminates: bool = True
    verifier_mode: str = "fallback"
    type_mode: str = "replace"


@dataclass
class RunContext:
    run_id: str
    out_root: Path
    iteration: int = 0
    policy_tag: str = ""
    reward_config: dict = field(default_factory=dict)
    bridge: object = None


class Episode:
    def __init__(self, task, worker, slot, sample_k, run_ctx, cfg):
        self.task, self.worker, self.slot, self.k, self.ctx, self.cfg = task, worker, slot, sample_k, run_ctx, cfg
        self.client = worker.client(slot)
        self.episode_id = f"{task['task_id']}__i{run_ctx.iteration:03d}__k{sample_k:02d}"
        self.safe_id = safe_task_id(self.episode_id)
        self.host_dir = Path(run_ctx.out_root) / f"iter_{run_ctx.iteration:03d}" / task["task_id"] / f"k{sample_k:02d}"
        self.traj_dir = self.host_dir / f"task_{self.safe_id}"
        # Compared by registrable domain, so www./m./checkout. subdomain hops are not refused.
        self.allowed_domains = {
            registrable_domain(h) for h in {urlparse(task["start_url"]).netloc, *task.get("allowed_hosts", [])}
        }
        self.history, self.turns = [], []
        self.step_index = self.consecutive_parse_failures = self.navigate_blocked = 0
        self.blocked = self.oracle_error = None
        self.terminal_reason, self.last_thought, self.record, self.closed = None, "", None, False
        self.started = time.time()

    def reset(self):
        backend = self.worker.backend
        kwargs = backend.session_kwargs(self.task)
        if kwargs is None:
            raise RuntimeError("preconditions_unavailable")
        backend.pre_episode(self.worker.sandbox, self.task)
        task_meta = {
            "difficulty": self.task.get("difficulty"),
            "expected_actions": self.task.get("expected_actions"),
            "autonomy": self.task.get("lane", "autonomous"),
            "auth_mode": self.task.get("auth_mode", "none"),
            "env": self.task.get("env", "live"),
            "rl": {
                "run_id": self.ctx.run_id,
                "iter": self.ctx.iteration,
                "k": self.k,
                "policy_tag": self.ctx.policy_tag,
            },
        }
        self.client.session(
            self.episode_id,
            backend.start_url(self.task),
            instruction=self.task["instruction"],
            task_meta=task_meta,
            type_mode=self.cfg.type_mode,
            **kwargs,
        )
        return self._observe()

    def _observe(self):
        # A navigation can still be committing right after an action, and the driver's title/screenshot reads then fail transiently.
        for attempt in range(3):
            try:
                shot = self.client.observe()
                break
            except DriverError:
                if attempt == 2:
                    raise
                time.sleep(1.0)
        return build_observation(self.task["instruction"], self.history, shot["b64"], self.step_index)

    def step(self, response_text):
        if self.closed:
            raise RuntimeError("episode is closed")
        decoded = decode_step(response_text)
        problems = validate_actions(decoded.actions)
        turn = {
            "step_index": self.step_index,
            "raw": response_text,
            "errors": decoded.errors + problems,
            "clean": decoded.ok and not decoded.errors and not problems,
            "executed": 0,
        }
        self.turns.append(turn)
        info = {"step_index": self.step_index, "parse_errors": turn["errors"]}
        if not decoded.actions or problems:
            self.consecutive_parse_failures += 1
            info["parse"] = "failure"
            if self.consecutive_parse_failures >= self.cfg.max_consecutive_parse_failures:
                return self._terminate("parse_abort", info)
            return self._observe(), None, False, info
        self.consecutive_parse_failures = 0
        info["parse"] = "ok"
        blocked = [
            a
            for a in decoded.actions
            if a["type"] == "navigate" and registrable_domain(urlparse(a["url"]).netloc) not in self.allowed_domains
        ]
        if blocked:
            self.navigate_blocked += 1
            info["refusal"] = "navigate_blocked"
            return self._observe(), None, False, info
        payload = to_sequence_payload(decoded.thought, decoded.actions)
        self.last_thought = decoded.thought
        try:
            result = self.client.act_sequence(payload["thought"], payload["actions"])
        except DriverError as e:
            if "loop guard" in str(e):
                info["refusal"] = "loop_guard"
                if self.cfg.loop_guard_terminates:
                    return self._terminate("loop_guard", info)
                return self._observe(), None, False, info
            info["refusal"], info["error"] = "driver_error", str(e)[:200]
            return self._terminate("driver_error", info)
        turn["executed"] = result.get("executed", 0)
        self.history.extend(history_entries_for_step({"actions": payload["actions"]}))
        self.step_index += 1
        info.update({"url": result.get("url"), "executed": turn["executed"]})
        if result.get("error"):
            info["exec_error"] = result["error"]
        control = control_index(payload["actions"])
        if control is not None:
            return self._terminate(
                "done" if payload["actions"][control]["type"] == "done" else "control_unexpected", info
            )
        if self.step_index >= self.cfg.max_steps:
            return self._terminate("step_cap", info)
        return self._observe(), None, False, info

    def _terminate(self, reason, info):
        self.terminal_reason = reason
        record = self.close(reason)
        return None, record["reward"], True, {**info, "terminal_reason": reason, **record["info"]}

    def close(self, reason="external"):
        if self.closed:
            return self.record
        self.closed = True
        self.terminal_reason = self.terminal_reason or reason
        check_result, milestones = self._run_oracles()
        try:
            self.client.finish(OUTCOME_BY_REASON.get(self.terminal_reason, "failed"), self.last_thought[:400])
        except Exception as e:
            logger.warning(f"{self.episode_id}: finish failed: {e}")
        container_dir = f"{RESULTS_DIR}/task_{self.safe_id}"
        self.host_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.worker.sandbox.copy_out(container_dir, str(self.traj_dir))
        except Exception as e:
            logger.error(f"{self.episode_id}: copy_out failed: {e}")
        try:
            self.worker.backend.post_episode(self.worker.sandbox, self.task, container_dir)
        except Exception as e:
            logger.warning(f"{self.episode_id}: post_episode failed: {e}")
        self._write_metadata(check_result, milestones)
        self.record = self._build_record(check_result, milestones)
        if self.traj_dir.exists():
            (self.traj_dir / "episode.json").write_text(json.dumps(self.record, indent=2))
        return self.record

    def _page_text(self):
        for attempt in (1, 2):
            try:
                return self.client.page_text()
            except Exception as e:
                if attempt == 2:
                    # 14 of the selected oracles read page_text; a blank one silently fails them all.
                    self.oracle_error = f"page_text: {str(e)[:120]}"
                    logger.error(f"{self.episode_id}: page_text unavailable after retry — ORACLES ARE BLIND: {e}")
                    return ""
                time.sleep(1)

    def _run_oracles(self):
        spec = self.task.get("success_check")
        milestone_specs = self.task.get("milestones") or []
        if not spec and not milestone_specs:
            return None, None
        try:
            snapshot, page_text = self.client.snapshot(), self._page_text()
            # A CAPTCHA/WAF wall is the environment refusing, not the policy failing, so the analysis drops these samples.
            self.blocked = detect_block(snapshot.get("url", ""), f"{snapshot.get('snapshot', '')}\n{page_text}")
            if self.blocked:
                logger.warning(
                    f"{self.episode_id}: site blocked ({self.blocked}) — episode is not a measurement of the policy"
                )
            ctx = checks.build_context(
                snapshot, self.client.storage_state, {"terminal_reason": self.terminal_reason}, page_text
            )
        except Exception as e:
            self.oracle_error = f"snapshot: {str(e)[:120]}"
            logger.error(
                f"{self.episode_id}: oracle snapshot failed — scoring this episode as a failure is NOT a measurement: {e}"
            )
            return (
                {"passed": False, "results": {"error": str(e)[:200]}} if spec else None,
                {m.get("name", ""): False for m in milestone_specs} or None,
            )
        check_result = checks.evaluate(spec, ctx) if spec else None
        milestones = {
            m.get("name", ""): checks.evaluate(m.get("check") or {}, ctx)["passed"] for m in milestone_specs
        } or None
        return check_result, milestones

    def _write_metadata(self, check_result, milestones=None):
        traj_file = self.traj_dir / "trajectory.json"
        if not traj_file.exists():
            return
        traj = json.loads(traj_file.read_text())
        meta_file = self.traj_dir / "capture_meta.json"
        capture_meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        if check_result is not None:
            capture_meta["success_check"] = check_result
        if milestones is not None:
            capture_meta["milestones"] = milestones
        capture_meta["excluded_reason"] = "rl_rollout"
        meta_file.write_text(json.dumps(capture_meta, indent=2))
        metadata = {
            "task_id": traj["task_id"],
            "instruction": traj["instruction"],
            "start_url": traj["start_url"],
            "total_steps": traj["total_steps"],
            "success": traj["success"],
            "website": urlparse(self.task["start_url"]).netloc,
            "domain": self.task.get("domain", "public_web"),
            "framework": "agent",
            "capture_source": "agent",
            "instruction_source": "curated",
            "thought_source": "student",
            "task_type": "other",
            "difficulty": self.task.get("difficulty", "medium"),
            "viewport": {"width": 1280, "height": 720},
            "source_repo": f"rl:{self.task['task_id']}",
            "autonomy": self.task.get("lane", "autonomous"),
            "auth_mode": self.task.get("auth_mode", "none"),
            "preconditions": self.task.get("preconditions", {}),
            "excluded_reason": "rl_rollout",
            "rl": {
                "run_id": self.ctx.run_id,
                "iter": self.ctx.iteration,
                "k": self.k,
                "terminal_reason": self.terminal_reason,
            },
        }
        (self.traj_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def _ui_steps(self):
        traj_file = self.traj_dir / "trajectory.json"
        if not traj_file.exists():
            return self.step_index
        steps = json.loads(traj_file.read_text()).get("steps", [])
        return sum(1 for s in steps if s.get("actions") and s["actions"][0].get("type") != "done")

    def _build_record(self, check_result, milestones=None):
        success, source, status = judge_success(
            self.traj_dir, self.task, check_result, self.cfg.verifier_mode, self.ctx.bridge
        )
        verdict = (cached_verdict(self.traj_dir) or {}) if self.traj_dir.exists() else {}
        record = {
            "episode_id": self.episode_id,
            "task_id": self.task["task_id"],
            "site": self.task.get("site"),
            "split": self.task.get("split"),
            "k": self.k,
            "iter": self.ctx.iteration,
            "traj_dir": str(self.traj_dir),
            "terminal_reason": self.terminal_reason,
            "steps": self.step_index,
            "ui_steps": self._ui_steps(),
            "turns": self.turns,
            "success": success,
            "efficiency_grade": verdict.get("efficiency"),
            "verdict_contradiction": bool(verdict.get("contradiction")),
            "success_source": source,
            "reward_status": status,
            "navigate_blocked": self.navigate_blocked,
            "duration_s": round(time.time() - self.started, 1),
            "check": check_result,
            "milestones": milestones,
            "blocked": self.blocked,
            "oracle_error": self.oracle_error,
        }
        reward, breakdown = (
            compute_reward(record, self.task, self.ctx.reward_config) if status == "final" else (None, {})
        )
        record.update(
            {
                "reward": reward,
                "breakdown": breakdown,
                "info": {
                    "reward_status": status,
                    "success": success,
                    "success_source": source,
                    "episode_dir": str(self.traj_dir),
                    "blocked": self.blocked,
                    "oracle_error": self.oracle_error,
                },
            }
        )
        return record
