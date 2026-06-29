"""Trajectory corpus loader.

Walks `trajectories/` (both `task_*/` flat and `repo_*/task_*/` nested layouts), tolerantly
loads `trajectory.json` + `metadata.json` + `capture_meta.json` from each task dir, and returns
two long-form pandas DataFrames plus a list of human-readable warnings for any file that was
skipped or had to be defaulted. Every file load is wrapped in try/except — the dashboard must
never crash on a malformed trajectory.
"""

import json
import math
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import ValidationError

from repo_pipeline.schemas import CaptureMeta, Trajectory, TrajectoryMetadata
from repo_pipeline.dashboard.metrics import (
    PLACEHOLDER_THOUGHT_SUBSTRINGS,
    shannon_entropy_nats,
)


def _find_task_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    found: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "trajectory.json").exists():
            found.append(child)
            continue
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir() and (grandchild / "trajectory.json").exists():
                found.append(grandchild)
    return found


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _classify_path_kind(metadata: dict, task_id: str) -> str:
    if (metadata.get("source_repo") or "").strip():
        return "B"
    if task_id.startswith("mind2web_"):
        return "A"
    return "A"


def _parse_recorded_date(timestamp_str: str) -> Optional[pd.Timestamp]:
    if not timestamp_str:
        return None
    try:
        return pd.to_datetime(timestamp_str, utc=True)
    except (ValueError, TypeError):
        return None


def _is_placeholder(thought: str) -> bool:
    if not thought:
        return False
    return any(substr in thought for substr in PLACEHOLDER_THOUGHT_SUBSTRINGS)


def _bbox_contains(bbox: Optional[dict], x: Optional[float], y: Optional[float]) -> Optional[bool]:
    if bbox is None or x is None or y is None:
        return None
    bx, by = bbox.get("x"), bbox.get("y")
    bw, bh = bbox.get("width"), bbox.get("height")
    if bx is None or by is None or bw is None or bh is None:
        return None
    return bool(bx <= x <= bx + bw and by <= y <= by + bh)


def _click_to_center_px(bbox: Optional[dict], x: Optional[float], y: Optional[float]) -> Optional[float]:
    if bbox is None or x is None or y is None:
        return None
    bx, by = bbox.get("x"), bbox.get("y")
    bw, bh = bbox.get("width"), bbox.get("height")
    if bx is None or by is None or bw is None or bh is None:
        return None
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    return float(math.hypot(x - cx, y - cy))


def _bbox_area_ratio(bbox: Optional[dict], viewport_w: int, viewport_h: int) -> Optional[float]:
    if bbox is None:
        return None
    bw, bh = bbox.get("width"), bbox.get("height")
    if bw is None or bh is None:
        return None
    vp_area = max(1, viewport_w * viewport_h)
    return float(bw * bh) / float(vp_area)


def _first_coord_action(actions: list[dict]) -> tuple[Optional[float], Optional[float]]:
    for action in actions:
        if action.get("type") in ("click", "type", "drag"):
            x = action.get("x")
            y = action.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return float(x), float(y)
            path = action.get("path") or []
            if path and isinstance(path[0], dict):
                px, py = path[0].get("x"), path[0].get("y")
                if isinstance(px, (int, float)) and isinstance(py, (int, float)):
                    return float(px), float(py)
    return None, None


def _step_action_type(actions: list[dict]) -> str:
    if not actions:
        return "none"
    if len(actions) == 1:
        return str(actions[0].get("type") or "none")
    types = {a.get("type") for a in actions}
    if len(types) == 1:
        return str(next(iter(types)) or "none")
    return "multi_action"


def _action_counter(trajectory: Trajectory) -> Counter:
    counter: Counter = Counter()
    for step in trajectory.steps:
        for action in step.actions:
            counter[str(action.get("type") or "none")] += 1
    return counter


def _per_trajectory_action_entropy(trajectory: Trajectory) -> Optional[float]:
    counter = _action_counter(trajectory)
    if not counter:
        return None
    return shannon_entropy_nats(counter)


def load_corpus(traj_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    traj_rows: list[dict] = []
    step_rows: list[dict] = []

    task_dirs = _find_task_dirs(traj_root)
    for task_dir in task_dirs:
        rel = task_dir.relative_to(traj_root) if traj_root in task_dir.parents else task_dir
        trajectory_raw = _read_json(task_dir / "trajectory.json")
        if trajectory_raw is None:
            warnings.append(f"skipped {rel}: trajectory.json missing or unreadable")
            continue
        try:
            trajectory = Trajectory(**trajectory_raw)
        except ValidationError as exc:
            warnings.append(f"skipped {rel}: trajectory.json invalid ({exc.error_count()} errors)")
            continue

        metadata_raw = _read_json(task_dir / "metadata.json") or {}
        if not metadata_raw:
            warnings.append(f"{rel}: metadata.json missing — using defaults")
            metadata_raw = {
                "task_id": trajectory.task_id,
                "instruction": trajectory.instruction,
                "start_url": trajectory.start_url,
                "total_steps": trajectory.total_steps,
                "success": trajectory.success,
            }
        try:
            metadata = TrajectoryMetadata(**metadata_raw)
        except ValidationError as exc:
            warnings.append(f"{rel}: metadata.json invalid ({exc.error_count()} errors) — using defaults")
            metadata = TrajectoryMetadata(
                task_id=trajectory.task_id,
                instruction=trajectory.instruction,
                start_url=trajectory.start_url,
                total_steps=trajectory.total_steps,
                success=trajectory.success,
            )

        capture_meta: Optional[CaptureMeta] = None
        capture_raw = _read_json(task_dir / "capture_meta.json")
        if capture_raw is not None:
            try:
                capture_meta = CaptureMeta(**capture_raw)
            except ValidationError as exc:
                warnings.append(f"{rel}: capture_meta.json invalid ({exc.error_count()} errors) — ignoring")

        path_kind = _classify_path_kind(metadata.model_dump(), trajectory.task_id)
        action_counter = _action_counter(trajectory)
        per_traj_entropy = _per_trajectory_action_entropy(trajectory)
        viewport = metadata.viewport or {"width": 1280, "height": 720}
        viewport_w = int(viewport.get("width") or 1280)
        viewport_h = int(viewport.get("height") or 720)

        # `trajectory.json` is the post-stage-6 source of truth for instruction (Claude can
        # generate it after the test ran). Older `metadata.json` files predate this and
        # carry an empty instruction — prefer the trajectory's copy when metadata is empty.
        effective_instruction = metadata.instruction or trajectory.instruction or ""

        metadata_row = metadata.model_dump()
        metadata_row["instruction"] = effective_instruction

        traj_rows.append({
            **metadata_row,
            "path_kind": path_kind,
            "trajectory_dir": str(rel),
            "n_steps_actual": len(trajectory.steps),
            "n_click":    action_counter.get("click", 0),
            "n_type":     action_counter.get("type", 0),
            "n_keypress": action_counter.get("keypress", 0),
            "n_scroll":   action_counter.get("scroll", 0),
            "n_wait":     action_counter.get("wait", 0),
            "n_drag":     action_counter.get("drag", 0),
            "n_actions_total": sum(action_counter.values()),
            "action_entropy_nats": per_traj_entropy,
            "instruction_len_chars": len(effective_instruction),
            "instruction_token_est": len(effective_instruction) // 4,
            "has_capture_meta": capture_meta is not None,
            "recorded_date": _parse_recorded_date(metadata.timestamp),
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        })

        for step in trajectory.steps:
            step_key = str(step.step)
            cap_step = (capture_meta.steps.get(step_key)
                        if capture_meta is not None else None)
            x, y = _first_coord_action(step.actions)
            bbox = step.bbox if step.bbox is not None else (
                cap_step.bbox if cap_step is not None else None
            )
            role = step.role if step.role else (cap_step.role if cap_step is not None else None)
            name = step.name if step.name else (cap_step.name if cap_step is not None else None)

            step_rows.append({
                "task_id": trajectory.task_id,
                "path_kind": path_kind,
                "step": step.step,
                "screenshot": step.screenshot,
                "screenshot_path": str(task_dir / step.screenshot),
                "action_type": _step_action_type(step.actions),
                "n_actions": len(step.actions),
                "actions": step.actions,
                "x": x,
                "y": y,
                "thought": step.thought or "",
                "has_thought": bool((step.thought or "").strip()),
                "thought_len": len(step.thought or ""),
                "is_placeholder_thought": _is_placeholder(step.thought or ""),
                "bbox": bbox,
                "role": role,
                "name": name,
                "selector":         (cap_step.selector if cap_step is not None else None),
                "tag":              (cap_step.tag      if cap_step is not None else None),
                "is_visible":       (cap_step.is_visible if cap_step is not None else None),
                "is_enabled":       (cap_step.is_enabled if cap_step is not None else None),
                "in_viewport":      (cap_step.in_viewport if cap_step is not None else None),
                "url_before":       (cap_step.url_before  if cap_step is not None else None),
                "url_after":        (cap_step.url_after   if cap_step is not None else None),
                "caused_navigation":(cap_step.caused_navigation if cap_step is not None else None),
                "dom_changed":      (cap_step.dom_changed if cap_step is not None else None),
                "console_errors":   (cap_step.console_errors if cap_step is not None else None),
                "dt_ms":            (cap_step.dt_ms if cap_step is not None else None),
                "wait_state":       (cap_step.wait_state if cap_step is not None else None),
                "bbox_contains_click": _bbox_contains(bbox, x, y),
                "click_to_center_px": _click_to_center_px(bbox, x, y),
                "bbox_area_ratio": _bbox_area_ratio(bbox, viewport_w, viewport_h),
            })

    traj_df = pd.DataFrame(traj_rows)
    step_df = pd.DataFrame(step_rows)
    return traj_df, step_df, warnings


def split_by_path(traj_df: pd.DataFrame, step_df: pd.DataFrame,
                  path_kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if traj_df.empty:
        return traj_df, step_df
    t = traj_df[traj_df["path_kind"] == path_kind].reset_index(drop=True)
    s = step_df[step_df["task_id"].isin(t["task_id"])].reset_index(drop=True)
    return t, s


def summarize_warnings(warnings: list[str], limit: int = 20) -> str:
    if not warnings:
        return ""
    if len(warnings) <= limit:
        return "\n".join(f"- {w}" for w in warnings)
    head = "\n".join(f"- {w}" for w in warnings[:limit])
    return head + f"\n- … and {len(warnings) - limit} more"


# ---------- Registry (pipeline_data/registry.db) loaders -----------------------------------
# The corpus loader above reads finished trajectories from disk; these read the funnel state
# (every repo and why it was rejected) for the Rejections page. Read-only, tolerant of an
# absent DB or a pre-migration schema (returns an empty frame instead of raising).

_SHALLOW_DIMS = [
    "production_complexity", "ui_difficulty", "interaction_richness", "test_count",
    "multi_page_flows", "runnability", "framework_clarity", "application_vs_library",
]
_DEEP_FIELDS = ["test_file_count", "test_case_count", "is_self_contained", "estimated_trajectories",
                "test_framework", "framework_language"]


def _json_get(blob, key):
    if not blob:
        return None
    try:
        return json.loads(blob).get(key) if isinstance(blob, str) else blob.get(key)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def _read_sql(db_path: Path, query: str) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.OperationalError:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(query, con)
    except (pd.errors.DatabaseError, sqlite3.OperationalError):
        return pd.DataFrame()
    finally:
        con.close()


def load_registry(db_path: Path) -> pd.DataFrame:
    """One row per repo from the registry, with shallow/deep score JSON expanded into
    `sc_<dim>` / `da_<field>` columns so the score-distribution charts can read them directly."""
    df = _read_sql(db_path, "SELECT * FROM repos")
    if df.empty:
        return df
    # Guarantee the columns the Rejections page reads, even against a registry that predates the
    # failure-taxonomy migration (the dashboard reads read-only and never migrates).
    if "failure_category" not in df.columns:
        df["failure_category"] = None
    if "retry_count" not in df.columns:
        df["retry_count"] = 0
    for dim in _SHALLOW_DIMS:
        df[f"sc_{dim}"] = df["shallow_scores"].map(lambda b, k=dim: _json_get(b, k))
    for field in _DEEP_FIELDS:
        df[f"da_{field}"] = df["deep_analysis"].map(lambda b, k=field: _json_get(b, k))
    return df


def load_agent_calls(db_path: Path) -> pd.DataFrame:
    """One row per Claude Agent SDK call (turns / cost / tokens / subtype), for the Turns & cost tab."""
    return _read_sql(db_path, "SELECT * FROM agent_calls")


def load_iso_date_range(traj_df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    if traj_df.empty or "recorded_date" not in traj_df.columns:
        return None, None
    dates = traj_df["recorded_date"].dropna()
    if dates.empty:
        return None, None
    lo = dates.min()
    hi = dates.max()
    return (lo.isoformat() if isinstance(lo, datetime) else str(lo),
            hi.isoformat() if isinstance(hi, datetime) else str(hi))
