import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import yaml


SYSTEM_PROMPT = (
    "You are a GUI agent that interacts with web browsers. "
    "Given a screenshot and a task, reason about the current state "
    "and output a structured action to complete the task."
)


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


def convert_trajectory_to_messages(trajectory, trajectory_dir):
    instruction = trajectory["instruction"]
    steps = trajectory["steps"]

    if not steps:
        return None

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    action_history = []

    for step in steps:
        screenshot_filename = step["screenshot"]
        screenshot_path = str((trajectory_dir / screenshot_filename).resolve())

        if not Path(screenshot_path).exists():
            continue

        user_text = f"Task: {instruction}"
        if action_history:
            history_str = ", ".join(action_history)
            user_text += f"\nPrevious actions: [{history_str}]"
        user_text += "\nWhat action should you take?"

        user_content = [
            {"type": "image", "image": screenshot_path},
            {"type": "text", "text": user_text},
        ]
        messages.append({"role": "user", "content": user_content})

        thought = step.get("thought", "")
        actions = step.get("actions", [])
        if not actions and "action" in step:
            actions = [step["action"]]

        action_strs = [format_action_for_training(a) for a in actions]
        action_str = "\n".join(action_strs)

        if thought:
            assistant_content = f"<think>{thought}</think>\n{action_str}"
        else:
            assistant_content = action_str

        messages.append({"role": "assistant", "content": assistant_content})

        for a in actions:
            action_history.append(format_action_summary(a))

    if len(messages) <= 1:
        return None

    return {"messages": messages}


def convert_trajectories(input_dir, output_dir, val_split):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_examples = []
    skipped_failed = 0
    skipped_empty = 0

    trajectory_dirs = sorted(
        d for d in input_path.glob("**/task_*")
        if d.is_dir() and (d / "trajectory.json").exists()
    )
    print(f"Found {len(trajectory_dirs)} trajectory directories")

    for traj_dir in trajectory_dirs:
        trajectory_file = traj_dir / "trajectory.json"
        if not trajectory_file.exists():
            continue

        with open(trajectory_file) as f:
            trajectory = json.load(f)

        if not trajectory.get("success", False):
            skipped_failed += 1
            continue

        example = convert_trajectory_to_messages(trajectory, traj_dir)
        if example is None:
            skipped_empty += 1
            continue

        all_examples.append(example)

    print(f"Converted {len(all_examples)} trajectories")
    print(f"Skipped {skipped_failed} failed trajectories")
    print(f"Skipped {skipped_empty} empty/invalid trajectories")

    random.shuffle(all_examples)
    val_count = int(len(all_examples) * val_split)
    val_examples = all_examples[:val_count]
    train_examples = all_examples[val_count:]

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

    train_parser = subparsers.add_parser("train", help="Launch ms-swift SFT training")
    train_parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    train_parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    train_parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    train_parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")

    args = parser.parse_args()

    if args.command == "convert":
        convert_trajectories(args.input, args.output, args.val_split)
    elif args.command == "train":
        train_model(args.config, args.dataset, args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
