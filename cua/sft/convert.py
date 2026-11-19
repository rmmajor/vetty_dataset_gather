"""trajectories/ -> ms-swift SFT examples: one per step, gated, reported and split by app."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from cua.pipeline.corpus.audit import trainability
from cua.prompt.coords import to_model_action
from cua.prompt.effects import downscale_prev, effect_is_no_change, step_effects
from cua.prompt.format import SYSTEM_PROMPT, build_step_messages, build_step_observation, history_entries_for_step
from cua.sft.corpus_filters import (
    absurd_keypress_indices,
    maybe_synthesize_done_step,
    meander_drop_indices,
    waste_step_indices,
)
from cua.sft.grounding import (
    GROUNDING_DEFAULT_SHARE,
    GROUNDING_SEED,
    grounding_targets,
    print_grounding_report,
    select_grounding_examples,
)
from cua.sft.health import (
    CorpusSummary,
    health_gate,
    health_stats,
    leak_gate,
    print_corpus_footer,
    print_corpus_overview,
    scan_examples_for_secrets,
)
from cua.trajectory.actions import TERMINAL_TYPES, format_action_for_training, normalize_trajectory_actions
from cua.trajectory.io import CAPTURE_META_FILE, METADATA_FILE, TRAJECTORY_FILE, iter_trajectory_dirs, load_json

REPORT_COUNTERS = (
    "trajectories",
    "done_steps",
    "done_trajectories",
    "meandering",
    "meandering_examples",
    "teacher_raw",
    "waste_steps_dropped",
    "subsample_dropped",
    "effect_steps",
    "effect_no_change",
    "effect_url_change",
)
REPORT_HISTOGRAMS = (
    "by_source",
    "actions",
    "target_actions",
    "domains",
    "thought_sources",
    "autonomy",
    "auth_mode",
    "grounding_skipped",
)
SKIP_REASONS = ("excluded", "unverified", "replay_gated", "empty", "meandering", "stuck_loop", "unreadable")


def _empty_report():
    return {**dict.fromkeys(REPORT_COUNTERS, 0), **{key: {} for key in REPORT_HISTOGRAMS}, "lengths": []}


@dataclass
class IngestResult:
    """Everything one pass over the corpus produced, before grounding, gates and the split."""

    examples: list = field(default_factory=list)
    examples_by_group: dict = field(default_factory=dict)
    grounding_targets_by_group: dict = field(default_factory=dict)
    leak_sources: list = field(default_factory=list)
    report: dict = field(default_factory=_empty_report)
    skipped: dict = field(default_factory=lambda: dict.fromkeys(SKIP_REASONS, 0))


def _empty_step_stats():
    return {
        "waste_dropped": 0,
        "subsample_dropped": 0,
        "target_actions": {},
        "done_emitted": False,
        "effect_steps": 0,
        "effect_no_change": 0,
        "effect_url_change": 0,
    }


def _write_prev_image(trajectory_dir, steps, index, prev_dir):
    """Downscaled previous-step screenshot for `index`; None on step 0, without prev_dir, or a missing source."""
    if index == 0 or prev_dir is None:
        return None
    source = Path(trajectory_dir) / steps[index - 1]["screenshot"]
    if not source.is_file():
        return None
    target = Path(prev_dir) / Path(trajectory_dir).name / f"step_{index:03d}_prev.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(downscale_prev(source.read_bytes()))
    return str(target.resolve())


def _step_example(instruction, action_history, screenshot_path, prev_image, thought, actions):
    observation = build_step_observation(instruction, action_history, str(screenshot_path), prev_image)
    action_str = "\n".join(format_action_for_training(to_model_action(a)) for a in actions)
    assistant_content = f"<think>{thought}</think>\n{action_str}" if thought else action_str
    return {
        "messages": [
            *build_step_messages(SYSTEM_PROMPT, observation),
            {"role": "assistant", "content": assistant_content},
        ]
    }


def _count_target_actions(stats, actions):
    for action in actions:
        action_type = action.get("type", "?")
        stats["target_actions"][action_type] = stats["target_actions"].get(action_type, 0) + 1
        # Any terminal stop verb is stop-supervision — wall demos legitimately end in handoff.
        if action_type in TERMINAL_TYPES:
            stats["done_emitted"] = True


def convert_trajectory_to_examples(
    trajectory, trajectory_dir, exclude_waste_steps=False, extra_drop_indices=(), capture_meta=None, prev_dir=None
):
    """One example PER STEP; dropped targets still flow into action_history, so the trajectory keeps all its context."""
    instruction = trajectory["instruction"]
    steps = trajectory["steps"]
    stats = _empty_step_stats()
    if not steps:
        return [], stats

    examples = []
    action_history = []
    waste = waste_step_indices(steps) if exclude_waste_steps else set()
    extra_drop = set(extra_drop_indices or ())
    effects = step_effects(trajectory_dir, steps, capture_meta)
    for effect in filter(None, effects):
        stats["effect_steps"] += 1
        stats["effect_no_change"] += int(effect_is_no_change(effect))
        stats["effect_url_change"] += int(bool(effect.get("path")))

    for i, step in enumerate(steps):
        screenshot_path = (trajectory_dir / step["screenshot"]).resolve()
        actions = step.get("actions", [])
        if not actions and "action" in step:
            actions = [step["action"]]

        emit = screenshot_path.exists() and bool(actions)
        if emit and i in waste:
            stats["waste_dropped"] += 1
        elif emit and i in extra_drop:
            stats["subsample_dropped"] += 1
        elif emit:
            prev_image = _write_prev_image(trajectory_dir, steps, i, prev_dir)
            thought = step.get("thought", "")
            examples.append(_step_example(instruction, action_history, screenshot_path, prev_image, thought, actions))
            _count_target_actions(stats, actions)

        action_history.extend(history_entries_for_step(step, effects[i]))

    return examples, stats


def _load_trajectory(traj_dir, skipped):
    """A trajectory being written by a concurrent capture parses as truncated JSON; skipping it saves the run."""
    try:
        trajectory = json.loads((traj_dir / TRAJECTORY_FILE).read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"SKIP unreadable {traj_dir}: {e}")
        skipped["unreadable"] += 1
        return None
    return trajectory, load_json(traj_dir / METADATA_FILE), load_json(traj_dir / CAPTURE_META_FILE)


def _is_meandering(metadata, capture_meta):
    """A quality tag, not a gate: `success` already means the goal was met."""
    return (capture_meta.get("verifier") or {}).get("efficiency") == "meandering" or metadata.get(
        "efficiency"
    ) == "meandering"


def _split_group(traj_dir):
    """Split unit: the app/site, never the step — trajectories of one app share screenshots and layout."""
    parent = traj_dir.parent.name
    return parent if parent.startswith(("repo_", "site_")) else traj_dir.name


def _extra_drop_indices(steps, meandering_mode, is_meandering):
    extra_drop = absurd_keypress_indices(steps)
    if meandering_mode == "subsample" and is_meandering:
        extra_drop |= meander_drop_indices(steps)
    return extra_drop


def _merge_step_stats(report, traj_stats):
    report["waste_steps_dropped"] += traj_stats["waste_dropped"]
    report["subsample_dropped"] += traj_stats["subsample_dropped"]
    for key in ("effect_steps", "effect_no_change", "effect_url_change"):
        report[key] += traj_stats[key]


def _record_examples(result, traj_dir, group, trajectory, capture_meta, examples):
    result.examples.extend(examples)
    leaks = scan_examples_for_secrets(examples)
    if leaks:
        result.leak_sources.append((traj_dir, leaks))
    result.examples_by_group.setdefault(group, []).extend(examples)
    targets, grounding_skipped = grounding_targets(trajectory, traj_dir, capture_meta)
    result.grounding_targets_by_group.setdefault(group, []).extend(targets)
    for reason, n in grounding_skipped.items():
        result.report["grounding_skipped"][reason] = result.report["grounding_skipped"].get(reason, 0) + n


def _count(histogram, key, n=1):
    histogram[key] = histogram.get(key, 0) + n


def _record_counts(report, trajectory, metadata, capture_meta, source, examples, traj_stats, is_meandering):
    report["trajectories"] += 1
    if is_meandering:
        report["meandering"] += 1
        report["meandering_examples"] += len(examples)
    if traj_stats["done_emitted"]:
        report["done_trajectories"] += 1
    for action_type, n in traj_stats["target_actions"].items():
        _count(report["target_actions"], action_type, n)
    _count(report["by_source"], source)
    report["lengths"].append(len(trajectory.get("steps", [])))
    _count(report["domains"], metadata.get("website") or metadata.get("domain") or "unknown")
    thought_source = metadata.get("thought_source", "unknown")
    if thought_source == "teacher_raw":
        report["teacher_raw"] += 1
    _count(report["thought_sources"], thought_source)
    task_meta = capture_meta.get("task_meta") or {}
    _count(report["autonomy"], metadata.get("autonomy") or task_meta.get("autonomy") or "autonomous")
    _count(report["auth_mode"], metadata.get("auth_mode") or task_meta.get("auth_mode") or "none")
    for step in trajectory.get("steps", []):
        for action in step.get("actions", []):
            action_type = action.get("type", "?")
            _count(report["actions"], action_type)
            if action_type == "done":
                report["done_steps"] += 1


def ingest_corpus(input_path, output_path, include_replay, meandering_mode, exclude_waste_steps):
    """One pass over every trajectory directory, applying the shared trainability gate and the target filters."""
    result = IngestResult()
    trajectory_dirs = list(iter_trajectory_dirs(input_path))
    print(f"Found {len(trajectory_dirs)} trajectory directories")

    for traj_dir in trajectory_dirs:
        loaded = _load_trajectory(traj_dir, result.skipped)
        if loaded is None:
            continue
        trajectory, metadata, capture_meta = loaded
        source = metadata.get("capture_source") or ("path_a" if "mind2web" in traj_dir.name else "replay")

        # The ONE trainability gate, shared with cua.pipeline.corpus.audit so they cannot drift.
        skip_reason = trainability(trajectory, metadata, capture_meta, traj_dir.name, include_replay=include_replay)
        if skip_reason:
            result.skipped[skip_reason] += 1
            continue
        is_meandering = _is_meandering(metadata, capture_meta)
        if meandering_mode == "exclude" and is_meandering:
            result.skipped["meandering"] += 1
            continue

        if source == "agent":
            maybe_synthesize_done_step(trajectory, traj_dir, capture_meta)
        normalize_trajectory_actions(trajectory)
        group = _split_group(traj_dir)
        examples, traj_stats = convert_trajectory_to_examples(
            trajectory,
            traj_dir,
            exclude_waste_steps=exclude_waste_steps,
            extra_drop_indices=_extra_drop_indices(trajectory.get("steps", []), meandering_mode, is_meandering),
            capture_meta=capture_meta,
            prev_dir=output_path / "prev" / group,
        )
        _merge_step_stats(result.report, traj_stats)
        if not examples:
            result.skipped["empty"] += 1
            continue
        _record_examples(result, traj_dir, group, trajectory, capture_meta, examples)
        _record_counts(result.report, trajectory, metadata, capture_meta, source, examples, traj_stats, is_meandering)
    return result


def merge_grounding(ingest, grounding_share, grounding_seed, meandering_mode):
    """Grounding joins each site's split group so val stays leak-free; the ceiling is a share of AGENT examples."""
    agent_examples = len(ingest.examples)
    by_group, stats = select_grounding_examples(
        ingest.grounding_targets_by_group, int(grounding_share * agent_examples), random.Random(grounding_seed)
    )
    for group, examples in by_group.items():
        leaks = scan_examples_for_secrets(examples)
        if leaks:
            ingest.leak_sources.append((f"grounding:{group}", leaks))
        ingest.examples_by_group.setdefault(group, []).extend(examples)
        ingest.examples.extend(examples)
    summary = CorpusSummary(
        meandering_mode=meandering_mode,
        grounding_share=grounding_share,
        grounding_seed=grounding_seed,
        agent_examples=agent_examples,
        total_examples=len(ingest.examples),
        grounding_examples=stats["grounding_examples"],
    )
    return stats, summary


def report_corpus(ingest, grounding, summary):
    print_corpus_overview(ingest.report, summary)
    print_grounding_report(ingest.report, grounding, summary)
    print_corpus_footer(ingest.report, ingest.skipped)


def enforce_gates(ingest, grounding, summary, health_config, skip_health_gate):
    leak_gate(ingest.leak_sources)
    stats = health_stats(ingest.report, ingest.examples_by_group, summary, grounding)
    health_gate(stats, health_config, skip_health_gate)
    return stats


def _write_jsonl(path, examples):
    with open(path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def write_splits(ingest, output_path, val_split, seed):
    """Group-wise split: whole apps go to val, never individual steps."""
    groups = list(ingest.examples_by_group.items())
    if seed is not None:
        random.seed(seed)
    random.shuffle(groups)
    val_target = int(len(ingest.examples) * val_split)
    val_examples, train_examples, val_groups = [], [], []
    for group, examples in groups:
        if val_target and len(val_examples) < val_target and len(groups) > 1:
            val_examples.extend(examples)
            val_groups.append(group)
        else:
            train_examples.extend(examples)
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    if val_groups:
        print(f"Validation groups (held-out apps): {', '.join(val_groups)}")

    train_path = output_path / "sft_dataset.jsonl"
    _write_jsonl(train_path, train_examples)
    print(f"Wrote {len(train_examples)} training examples to {train_path}")
    if val_examples:
        val_path = output_path / "sft_val.jsonl"
        _write_jsonl(val_path, val_examples)
        print(f"Wrote {len(val_examples)} validation examples to {val_path}")


def convert_trajectories(
    input_dir,
    output_dir,
    val_split,
    include_replay=False,
    meandering_mode="keep",
    exclude_waste_steps=False,
    health_config="corpus_health.json",
    skip_health_gate=False,
    grounding_share=GROUNDING_DEFAULT_SHARE,
    grounding_seed=GROUNDING_SEED,
    seed=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ingest = ingest_corpus(Path(input_dir), output_path, include_replay, meandering_mode, exclude_waste_steps)
    grounding, summary = merge_grounding(ingest, grounding_share, grounding_seed, meandering_mode)
    report_corpus(ingest, grounding, summary)
    stats = enforce_gates(ingest, grounding, summary, health_config, skip_health_gate)
    write_splits(ingest, output_path, val_split, seed)
    return stats
