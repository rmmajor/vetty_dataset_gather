import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import yaml


SYSTEM_PROMPT = """You are a GUI agent that operates a web browser at a 1280x720 resolution. \
Given a task, a screenshot of the current page, and your previous actions, reason about the \
state in <think>...</think>, then output one or more actions, one per line.

Available actions (coordinates are absolute pixels on the screenshot):
action: click(<ref>(x, y)</ref>, button=left|right)
action: double_click(<ref>(x, y)</ref>)
action: move(<ref>(x, y)</ref>)
action: type("text")
action: keypress(["Enter"])
action: scroll(<ref>(x, y)</ref>, scrollX=0, scrollY=500)
action: drag(<ref>(x1, y1)</ref> -> <ref>(x2, y2)</ref>)
action: wait()
action: done()

Use done() only when the task is fully accomplished and the screen confirms it."""


def format_action_for_training(action):
    action_type = action["type"]

    if action_type == "click":
        x, y = action["x"], action["y"]
        button = action.get("button", "left")
        return f'action: click(<ref>({x}, {y})</ref>, button={button})'

    elif action_type == "double_click":
        x, y = action["x"], action["y"]
        return f'action: double_click(<ref>({x}, {y})</ref>)'

    elif action_type == "drag":
        path = action["path"]
        start = path[0]
        end = path[-1]
        return f'action: drag(<ref>({start["x"]}, {start["y"]})</ref> -> <ref>({end["x"]}, {end["y"]})</ref>)'

    elif action_type == "move":
        x, y = action["x"], action["y"]
        return f'action: move(<ref>({x}, {y})</ref>)'

    elif action_type == "type":
        text = action["text"]
        return f'action: type("{text}")'

    elif action_type == "keypress":
        keys = action["keys"]
        return f'action: keypress({json.dumps(keys)})'

    elif action_type == "scroll":
        x, y = action["x"], action["y"]
        sx, sy = action["scroll_x"], action["scroll_y"]
        return f'action: scroll(<ref>({x}, {y})</ref>, scrollX={sx}, scrollY={sy})'

    elif action_type == "wait":
        return "action: wait()"

    elif action_type == "done":
        return "action: done()"

    elif action_type == "screenshot":
        return "action: screenshot()"

    return f"action: {action_type}()"


def format_action_summary(action):
    action_type = action["type"]

    if action_type in ("click", "double_click", "move"):
        return f'{action_type}({action["x"]}, {action["y"]})'
    elif action_type == "type":
        return f'type("{action["text"]}")'
    elif action_type == "keypress":
        return f'keypress({action["keys"]})'
    elif action_type == "scroll":
        return f'scroll({action["x"]}, {action["y"]})'
    elif action_type == "drag":
        path = action["path"]
        return f'drag({path[0]["x"]},{path[0]["y"]} -> {path[-1]["x"]},{path[-1]["y"]})'
    return action_type


# Text history shown to the model for each step. Windowed: long trajectories would otherwise
# bloat the prompt, and at inference the runtime keeps a bounded history too.
HISTORY_WINDOW = 10


def _maybe_synthesize_done_step(trajectory, trajectory_dir, capture_meta):
    """Terminal supervision for agent trajectories recorded before the driver emitted `done`
    steps: when the teacher honestly reported completion and the final screenshot exists,
    append a {type:"done"} step whose thought is the teacher's own completion summary. Without
    a terminal step the model is never taught to STOP — it would act forever at inference."""
    steps = trajectory.get("steps", [])
    if any(a.get("type") == "done" for s in steps for a in s.get("actions", [])):
        return
    if capture_meta.get("agent_outcome") != "completed":
        return
    summary = (capture_meta.get("agent_summary") or "").strip()
    final_screenshot = trajectory_dir / "final_screenshot.png"
    if not summary or not final_screenshot.exists():
        return
    steps.append({
        "step": (steps[-1]["step"] + 1) if steps else 1,
        "screenshot": "final_screenshot.png",
        "thought": summary,
        "actions": [{"type": "done"}],
    })


def convert_trajectory_to_examples(trajectory, trajectory_dir):
    """One training example PER STEP: current screenshot + task + windowed text history ->
    <think> + action(s). A whole trajectory as one multi-turn example packs every screenshot
    (~300 visual tokens each) into a single sequence — a 21-step trajectory is ~9k tokens
    against max_length 4096, so the trainer silently truncated exactly the long-horizon tails
    this dataset exists for. Per-step examples match inference conditions (one screenshot +
    text history) and supervise every step uniformly."""
    instruction = trajectory["instruction"]
    steps = trajectory["steps"]
    if not steps:
        return []

    examples = []
    action_history = []

    for step in steps:
        screenshot_path = (trajectory_dir / step["screenshot"]).resolve()
        thought = step.get("thought", "")
        actions = step.get("actions", [])
        if not actions and "action" in step:
            actions = [step["action"]]

        if screenshot_path.exists() and actions:
            user_text = f"Task: {instruction}"
            if action_history:
                window = action_history[-HISTORY_WINDOW:]
                omitted = len(action_history) - len(window)
                prefix = f"... {omitted} earlier actions, " if omitted > 0 else ""
                user_text += f"\nPrevious actions: [{prefix}{', '.join(window)}]"
            user_text += "\nWhat action should you take?"

            action_str = "\n".join(format_action_for_training(a) for a in actions)
            assistant_content = f"<think>{thought}</think>\n{action_str}" if thought else action_str

            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image", "image": str(screenshot_path)},
                        {"type": "text", "text": user_text},
                    ]},
                    {"role": "assistant", "content": assistant_content},
                ]
            })

        for a in actions:
            action_history.append(format_action_summary(a))

    return examples


def convert_trajectories(input_dir, output_dir, val_split, include_replay=False, exclude_meandering=False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_examples = []
    examples_by_group = {}
    skipped = {"excluded": 0, "unverified": 0, "replay_gated": 0, "empty": 0, "meandering": 0}
    report = {
        "trajectories": 0, "by_source": {}, "actions": {}, "lengths": [],
        "domains": {}, "done_steps": 0, "thought_sources": {}, "meandering": 0,
    }

    trajectory_dirs = sorted(
        d for d in input_path.glob("**/task_*")
        if d.is_dir() and (d / "trajectory.json").exists()
    )
    print(f"Found {len(trajectory_dirs)} trajectory directories")

    for traj_dir in trajectory_dirs:
        with open(traj_dir / "trajectory.json") as f:
            trajectory = json.load(f)

        metadata = {}
        metadata_file = traj_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        capture_meta = {}
        capture_meta_file = traj_dir / "capture_meta.json"
        if capture_meta_file.exists():
            try:
                with open(capture_meta_file) as f:
                    capture_meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        source = metadata.get("capture_source") or (
            "path_a" if "mind2web" in traj_dir.name else "replay"
        )

        # An explicit exclusion (quality audit, dedup, per-repo cap, contamination) overrides
        # everything below — these are KNOWN-BAD recordings, regardless of provenance.
        if metadata.get("excluded_reason"):
            skipped["excluded"] += 1
            continue

        # Replay trajectories carry QA-style supervision (test-assertion instructions,
        # retrofitted thoughts) — the exact style the agent pipeline replaced. They train
        # only on explicit opt-in.
        if source == "replay" and not include_replay:
            skipped["replay_gated"] += 1
            continue

        # Agent captures: `success` is the VERIFIER's verdict — false marks unverified
        # trajectories that must never train.
        if not trajectory.get("success", True) and source == "agent":
            skipped["unverified"] += 1
            continue

        # Efficiency is a quality tag, not a gate: meandering goal-completions train by default but
        # can be dropped for a clean run. `success` already means the goal was met.
        is_meandering = ((capture_meta.get("verifier") or {}).get("efficiency") == "meandering"
                         or metadata.get("efficiency") == "meandering")
        if exclude_meandering and is_meandering:
            skipped["meandering"] += 1
            continue

        if source == "agent":
            _maybe_synthesize_done_step(trajectory, traj_dir, capture_meta)

        examples = convert_trajectory_to_examples(trajectory, traj_dir)
        if not examples:
            skipped["empty"] += 1
            continue

        all_examples.extend(examples)
        # Split unit: the APP/SITE, not the step. Steps of one trajectory (and trajectories of
        # one app) share screenshots and layout — splitting below app level leaks pixels into
        # val and turns val loss into a memorization check.
        parent = traj_dir.parent.name
        group = parent if parent.startswith(("repo_", "site_")) else traj_dir.name
        examples_by_group.setdefault(group, []).extend(examples)
        report["trajectories"] += 1
        if is_meandering:
            report["meandering"] += 1
        report["by_source"][source] = report["by_source"].get(source, 0) + 1
        report["lengths"].append(len(trajectory.get("steps", [])))
        domain = metadata.get("website") or metadata.get("domain") or "unknown"
        report["domains"][domain] = report["domains"].get(domain, 0) + 1
        ts = metadata.get("thought_source", "unknown")
        report["thought_sources"][ts] = report["thought_sources"].get(ts, 0) + 1
        for step in trajectory.get("steps", []):
            for a in step.get("actions", []):
                t = a.get("type", "?")
                report["actions"][t] = report["actions"].get(t, 0) + 1
                if t == "done":
                    report["done_steps"] += 1

    # ---- Corpus report: the at-a-glance health check that catches a Booking.com-style
    # regression at convert time instead of in a JSON file weeks later. ----
    lengths = sorted(report["lengths"])
    total_actions = sum(report["actions"].values()) or 1
    print(f"\n=== Corpus report ===")
    print(f"Trajectories trained on: {report['trajectories']}  (examples: {len(all_examples)})")
    print(f"By source: {report['by_source']}")
    print(f"Thought sources: {report['thought_sources']}")
    if lengths:
        mid = lengths[len(lengths) // 2]
        print(f"Steps/trajectory: min={lengths[0]} median={mid} p90={lengths[int(len(lengths)*0.9)]} max={lengths[-1]}")
    print("Action mix: " + ", ".join(
        f"{t}={n} ({n / total_actions:.0%})" for t, n in sorted(report["actions"].items(), key=lambda kv: -kv[1])
    ))
    print(f"Terminal done() steps: {report['done_steps']}/{report['trajectories']} trajectories")
    print(f"Meandering (tagged, included; pass --exclude-meandering to drop): {report['meandering']}")
    print("Domains: " + ", ".join(f"{d}={n}" for d, n in sorted(report["domains"].items(), key=lambda kv: -kv[1])[:10]))
    print(f"Skipped: {skipped}")
    print("======================\n")

    # Group-wise split: whole apps go to val, never individual steps.
    groups = list(examples_by_group.items())
    random.shuffle(groups)
    val_target = int(len(all_examples) * val_split)
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
    with open(train_path, "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")
    print(f"Wrote {len(train_examples)} training examples to {train_path}")

    if val_examples:
        val_path = output_path / "sft_val.jsonl"
        with open(val_path, "w") as f:
            for example in val_examples:
                f.write(json.dumps(example) + "\n")
        print(f"Wrote {len(val_examples)} validation examples to {val_path}")


def train_model(config_path, dataset_override=None, epochs_override=None, lr_override=None, batch_size_override=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if dataset_override:
        config["dataset"] = dataset_override
    if epochs_override:
        config["num_train_epochs"] = epochs_override
    if lr_override:
        config["learning_rate"] = lr_override
    if batch_size_override:
        config["per_device_train_batch_size"] = batch_size_override

    cmd = ["swift", "sft"]

    flag_map = {
        "model": "--model",
        "train_type": "--train_type",
        "dataset": "--dataset",
        "torch_dtype": "--torch_dtype",
        "num_train_epochs": "--num_train_epochs",
        "per_device_train_batch_size": "--per_device_train_batch_size",
        "learning_rate": "--learning_rate",
        "lora_rank": "--lora_rank",
        "lora_alpha": "--lora_alpha",
        "target_modules": "--target_modules",
        "gradient_accumulation_steps": "--gradient_accumulation_steps",
        "max_length": "--max_length",
        "warmup_ratio": "--warmup_ratio",
        "freeze_vit": "--freeze_vit",
        "output_dir": "--output_dir",
    }

    for key, flag in flag_map.items():
        if key in config:
            value = config[key]
            if isinstance(value, bool):
                value = str(value).lower()
            cmd.extend([flag, str(value)])

    env = os.environ.copy()
    max_pixels = config.get("max_pixels")
    if max_pixels:
        env["MAX_PIXELS"] = str(max_pixels)

    print("Running command:")
    print("  " + " ".join(cmd))
    if max_pixels:
        print(f"  MAX_PIXELS={max_pixels}")

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Convert trajectories and train Qwen 3.5-9B")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert trajectories to ms-swift SFT format")
    convert_parser.add_argument("--input", type=str, required=True, help="Input trajectories directory")
    convert_parser.add_argument("--output", type=str, default="training_data/", help="Output directory")
    convert_parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    convert_parser.add_argument("--exclude-meandering", action="store_true",
                                help="Drop trajectories the verifier graded 'meandering' (kept by default)")
    convert_parser.add_argument("--include-replay", action="store_true",
                                help="Also train on legacy test-replay trajectories (QA-style supervision; off by default)")

    train_parser = subparsers.add_parser("train", help="Launch ms-swift SFT training")
    train_parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    train_parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    train_parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    train_parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")

    args = parser.parse_args()

    if args.command == "convert":
        convert_trajectories(args.input, args.output, args.val_split, include_replay=args.include_replay,
                             exclude_meandering=args.exclude_meandering)
    elif args.command == "train":
        train_model(args.config, args.dataset, args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
