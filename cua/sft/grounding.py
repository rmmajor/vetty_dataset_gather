"""The grounding slice: recorded clicks land on a named element's bbox centre, a free ScreenSpot-style set on our own distribution."""

import re
from pathlib import Path

from cua.prompt.coords import is_point, to_model_action
from cua.prompt.format import GROUNDING_SYSTEM_PROMPT, build_grounding_observation, build_step_messages
from cua.settings import GROUNDING
from cua.sft.health import scan_text_for_secrets
from cua.trajectory import VIEWPORT
from cua.trajectory.actions import format_action_for_training

GROUNDING_ROLE_NOUNS = {
    "button": "button",
    "a": "link",
    "link": "link",
    "input": "text field",
    "textbox": "text field",
    "searchbox": "text field",
    "textarea": "text area",
    "select": "dropdown",
    "combobox": "dropdown",
    "option": "option",
    "menuitem": "menu item",
    "tab": "tab",
    "checkbox": "checkbox",
    "radio": "radio button",
    "spinbutton": "number field",
    "treeitem": "item",
    "listitem": "item",
    "li": "item",
    "gridcell": "item",
    "summary": "item",
    "switch": "toggle",
    "img": "image",
}
GROUNDING_TEMPLATES = ('the "{name}" {noun}', 'the {noun} labelled "{name}"', 'the {noun} "{name}"')
GROUNDING_NAME_MAX_CHARS = 60
GROUNDING_CENTRE_TOLERANCE_PX = 2
GROUNDING_SMALL_TARGET_PX = 40
GROUNDING_SMALL_OVERSAMPLE = GROUNDING.small_oversample
GROUNDING_PER_GROUP_CAP = GROUNDING.per_group_cap
GROUNDING_DEDUPE_MAX = 2
GROUNDING_DEDUPE_CELL_PX = 8
GROUNDING_SEED = GROUNDING.seed
GROUNDING_DEFAULT_SHARE = 0.25
# Masked-password placeholders, password labels and secret placeholders: never a grounding target.
GROUNDING_SKIP_NAME_PATTERNS = (re.compile(r"[•*]{3,}"), re.compile(r"passw", re.IGNORECASE), re.compile(r"<secret:"))
GROUNDING_SKIP_INPUT_TYPES = ("password",)


def grounding_description(name, noun, variant=0):
    return GROUNDING_TEMPLATES[variant % len(GROUNDING_TEMPLATES)].format(name=name, noun=noun)


def _bbox_centre(bbox):
    return round(bbox["x"] + bbox["width"] / 2), round(bbox["y"] + bbox["height"] / 2)


def _bbox_in_viewport(bbox):
    x, y, w, h = (bbox.get(k) for k in ("x", "y", "width", "height"))
    if not all(isinstance(v, (int, float)) for v in (x, y, w, h)):
        return False
    return w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= VIEWPORT[0] and y + h <= VIEWPORT[1]


def _grounding_skip_reason(action, meta, screenshot_path):
    """Why a step is not a grounding target, or None; the reason keys are the convert report's histogram."""
    if action.get("type") != "click" or not is_point(action):
        return "not_click"
    role = meta.get("role")
    if not isinstance(role, str) or role not in GROUNDING_ROLE_NOUNS:
        return "no_role"
    name = (meta.get("name") or "").strip()
    if not name:
        return "no_name"
    if len(name) > GROUNDING_NAME_MAX_CHARS:
        return "name_too_long"
    if (
        any(p.search(name) for p in GROUNDING_SKIP_NAME_PATTERNS)
        or (meta.get("attrs") or {}).get("type") in GROUNDING_SKIP_INPUT_TYPES
    ):
        return "password_or_masked"
    if scan_text_for_secrets(name):
        return "secret"
    bbox = meta.get("bbox")
    if not isinstance(bbox, dict) or not _bbox_in_viewport(bbox):
        return "bbox_off_viewport"
    cx, cy = _bbox_centre(bbox)
    if abs(action["x"] - cx) > GROUNDING_CENTRE_TOLERANCE_PX or abs(action["y"] - cy) > GROUNDING_CENTRE_TOLERANCE_PX:
        return "off_centre"
    if not screenshot_path.is_file():
        return "no_screenshot"
    return None


def grounding_targets(trajectory, trajectory_dir, capture_meta):
    """Clean grounding targets from a trajectory's click steps, plus a skip-reason histogram for the report."""
    meta_steps = (capture_meta or {}).get("steps")
    meta_steps = meta_steps if isinstance(meta_steps, dict) else {}
    targets, skipped = [], {}
    for step in trajectory.get("steps", []):
        actions = step.get("actions") or ([step["action"]] if "action" in step else [])
        meta = meta_steps.get(str(step.get("step"))) or {}
        screenshot_path = (Path(trajectory_dir) / (step.get("screenshot") or "")).resolve()
        reason = _grounding_skip_reason(actions[0] if actions else {}, meta, screenshot_path)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        bbox = meta["bbox"]
        cx, cy = _bbox_centre(bbox)
        targets.append(
            {
                "screenshot": str(screenshot_path),
                "name": meta["name"].strip(),
                "role": meta["role"],
                "noun": GROUNDING_ROLE_NOUNS[meta["role"]],
                "x": cx,
                "y": cy,
                "small": bbox["width"] <= GROUNDING_SMALL_TARGET_PX and bbox["height"] <= GROUNDING_SMALL_TARGET_PX,
            }
        )
    return targets, skipped


def grounding_example(target, variant=0):
    """Description -> bbox-centre click with no <think>: the grounding slice supervises pointing, not planning."""
    description = grounding_description(target["name"], target["noun"], variant)
    observation = build_grounding_observation(description, target["screenshot"])
    action = to_model_action({"type": "click", "x": target["x"], "y": target["y"], "button": "left"})
    return {
        "messages": [
            *build_step_messages(GROUNDING_SYSTEM_PROMPT, observation),
            {"role": "assistant", "content": format_action_for_training(action)},
        ]
    }


def _dedupe_targets(targets, dedupe_max):
    """At most `dedupe_max` targets per (name, 8px cell): one button clicked in ten trajectories is one element."""
    seen, kept = {}, []
    for target in targets:
        key = (
            target["name"],
            round(target["x"] / GROUNDING_DEDUPE_CELL_PX),
            round(target["y"] / GROUNDING_DEDUPE_CELL_PX),
        )
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= dedupe_max:
            kept.append(target)
    return kept


def _grounding_tier(target, variant):
    """Survival order under a cap: small-target primaries, then their rephrasings, large targets dropped first."""
    return 0 if target["small"] and variant == 0 else 1 if target["small"] else 2


def _ranked(rows, rng):
    rng.shuffle(rows)
    rows.sort(key=lambda row: row["tier"])
    return rows


def _group_rows(group, targets, oversample, rng):
    rows = [
        {
            "group": group,
            "tier": _grounding_tier(t, v),
            "small": t["small"],
            "role": t["role"],
            "example": grounding_example(t, v),
        }
        for t in targets
        for v in range(oversample if t["small"] else 1)
    ]
    return _ranked(rows, rng)


def select_grounding_examples(targets_by_group, budget, rng, oversample=None, per_group_cap=None, dedupe_max=None):
    """Dedupe, oversample small targets, cap per group, then keep <= budget examples with small targets dropped last."""
    oversample = GROUNDING_SMALL_OVERSAMPLE if oversample is None else oversample
    per_group_cap = GROUNDING_PER_GROUP_CAP if per_group_cap is None else per_group_cap
    dedupe_max = GROUNDING_DEDUPE_MAX if dedupe_max is None else dedupe_max
    stats = {"grounding_candidates": 0, "grounding_deduped": 0, "grounding_capped": 0, "grounding_ceiling_dropped": 0}
    rows = []
    for group, targets in sorted(targets_by_group.items()):
        unique = _dedupe_targets(targets, dedupe_max)
        stats["grounding_candidates"] += len(targets)
        stats["grounding_deduped"] += len(targets) - len(unique)
        group_rows = _group_rows(group, unique, oversample, rng)
        stats["grounding_capped"] += max(0, len(group_rows) - per_group_cap)
        rows.extend(group_rows[:per_group_cap])
    rows = _ranked(rows, rng)
    stats["grounding_ceiling_dropped"] = max(0, len(rows) - max(0, budget))
    rows = rows[: max(0, budget)]
    examples_by_group, roles = {}, {}
    for row in rows:
        examples_by_group.setdefault(row["group"], []).append(row["example"])
        roles[row["role"]] = roles.get(row["role"], 0) + 1
    stats["grounding_examples"] = len(rows)
    stats["grounding_small_target_share"] = sum(row["small"] for row in rows) / len(rows) if rows else 0.0
    stats["grounding_roles"] = roles
    return examples_by_group, stats


def print_grounding_report(report, stats, summary):
    print(
        f"Grounding slice (ceiling {summary.grounding_share:.0%} of agent examples, seed {summary.grounding_seed}): "
        f"{stats['grounding_examples']} examples = {summary.grounding_share_actual:.1%} of agent steps, "
        f"{stats['grounding_examples'] / (summary.total_examples or 1):.1%} of all; "
        f"small (<= {GROUNDING_SMALL_TARGET_PX}px) targets {stats['grounding_small_target_share']:.1%}"
    )
    print(
        f"Grounding funnel: {stats['grounding_candidates']} candidates, -{stats['grounding_deduped']} dedupe, "
        f"-{stats['grounding_capped']} per-group cap ({GROUNDING_PER_GROUP_CAP}), "
        f"-{stats['grounding_ceiling_dropped']} ceiling; small targets x{GROUNDING_SMALL_OVERSAMPLE}"
    )
    print(
        "Grounding roles: "
        + (", ".join(f"{r}={n}" for r, n in sorted(stats["grounding_roles"].items(), key=lambda kv: -kv[1])) or "none")
    )
    print(
        "Grounding skipped steps: "
        + (
            ", ".join(f"{r}={n}" for r, n in sorted(report["grounding_skipped"].items(), key=lambda kv: -kv[1]))
            or "none"
        )
    )
