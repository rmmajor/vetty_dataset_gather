import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import yaml

from repo_pipeline import operator_pii
from repo_pipeline.corpus_audit import LOOP_RUN_LIMIT, _action_signature


# Unambiguous secret literals that must NEVER reach the training corpus. Masked credentials are
# recorded as `<secret:NAME>` placeholders (driver-side substitution), so any of these appearing as
# a literal is a real leak. Scoped to high-entropy keys/tokens that CANNOT be legitimate page
# content — the gate must never false-positive on normal UI text. Credit-card numbers are
# deliberately NOT matched: we never type real cards (payments are denied/handed-off), so the only
# card-shaped strings in the corpus are page content (year dropdowns, UUIDs) and public test cards
# (4111…/4242…) used in test-env checkout tasks — all false positives. Arbitrary leaked
# passwords/usernames can't be pattern-matched without false positives; the structural guarantee
# (driver records placeholders, asserted by driver_smoke) is their primary defense.
_SECRET_PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("github_token", re.compile(r"\bghp_[A-Za-z0-9]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
]


def _scan_text_for_secrets(text: str) -> list[str]:
    return [name for name, pat in _SECRET_PATTERNS if pat.search(text)] + operator_pii.scan_text(text)


def _scan_examples_for_secrets(examples) -> list[str]:
    """Distinct findings across every message of one trajectory's examples — the text that would ship."""
    findings = []
    for example in examples:
        for message in example["messages"]:
            content = message["content"]
            text = content if isinstance(content, str) else " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
            for finding in _scan_text_for_secrets(text):
                if finding not in findings:
                    findings.append(finding)
    return findings


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
action: navigate("https://...")
action: handoff(reason=login)
action: confirm(propose=click(<ref>(x, y)</ref>), risk=purchase)
action: done()

Use done() only when the task is fully accomplished and the screen confirms it.
Use handoff(reason=...) to yield control for a login, payment entry, CAPTCHA, or 2FA — never type \
those yourself. Use confirm(propose=..., risk=...) to PAUSE before a consequential, hard-to-undo \
action (purchase, payment, send, publish, delete, consent) and wait for approval instead of doing it."""


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

    elif action_type == "navigate":
        return f'action: navigate("{action.get("url", "")}")'

    elif action_type == "handoff":
        return f'action: handoff(reason={action.get("reason", "login")})'

    elif action_type == "confirm":
        risk = action.get("risk_category", "")
        proposed = action.get("proposed_action") or {}
        if "x" in proposed and "y" in proposed:
            inner = f'{proposed.get("type") or "click"}(<ref>({proposed["x"]}, {proposed["y"]})</ref>)'
        elif proposed.get("type"):
            inner = f'{proposed["type"]}()'
        else:
            inner = "action()"
        return f'action: confirm(propose={inner}, risk={risk})'

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
    elif action_type == "navigate":
        return f'navigate({action.get("url", "")})'
    elif action_type == "handoff":
        return f'handoff({action.get("reason", "")})'
    elif action_type == "confirm":
        return f'confirm(risk={action.get("risk_category", "")})'
    return action_type


# Text history shown to the model for each step. Windowed: long trajectories would otherwise
# bloat the prompt, and at inference the runtime keeps a bounded history too.
HISTORY_WINDOW = 10


def build_step_user_text(instruction, action_history):
    """The student's user turn: task + windowed text history; shared by convert and the RL env."""
    user_text = f"Task: {instruction}"
    if action_history:
        window = action_history[-HISTORY_WINDOW:]
        omitted = len(action_history) - len(window)
        prefix = f"... {omitted} earlier actions, " if omitted > 0 else ""
        user_text += f"\nPrevious actions: [{prefix}{', '.join(window)}]"
    user_text += "\nWhat action should you take?"
    return user_text


def history_entries_for_step(step):
    actions = step.get("actions", [])
    if not actions and "action" in step:
        actions = [step["action"]]
    entries = [format_action_summary(a) for a in actions]
    event = step.get("event") or {}
    if event.get("kind") == "approval":
        entries.append(f"[watchdog: {event.get('decision', '?')}]")
    return entries


def build_step_messages(system, user_text, image_ref):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image", "image": image_ref},
            {"type": "text", "text": user_text},
        ]},
    ]


# Step-level waste filter (opt-in via --exclude-waste-steps). A single-click step whose target
# lands within WASTE_CLICK_RADIUS_PX of a click in the previous WASTE_LOOKBACK_STEPS is a
# retry/backtrack — the "click didn't register, click again" pattern that survives the stuck-loop
# gate (which only drops >=3 IDENTICAL consecutive actions). These steps are dropped as TRAINING
# TARGETS only; they stay in the text history so later steps' context — and their post-retry
# screenshots — remain coherent. Geometry-based, like the loop/viewport gates, not thought-text.
WASTE_CLICK_RADIUS_PX = int(os.environ.get("WASTE_CLICK_RADIUS_PX", "10"))
WASTE_LOOKBACK_STEPS = int(os.environ.get("WASTE_LOOKBACK_STEPS", "4"))


def _waste_step_indices(steps, radius_px=WASTE_CLICK_RADIUS_PX, lookback=WASTE_LOOKBACK_STEPS):
    """Indices of single-click steps that merely retry a recent click (within radius_px of a click
    in the previous `lookback` steps). Only PURE single-click steps qualify — a step that also
    types/scrolls/keypresses does real work and is never waste; scrolls (which legitimately repeat
    x,y with different deltas) are never considered. A step's own click is recorded only after it
    is checked, so a step never matches itself."""
    waste = set()
    recent = []
    for i, step in enumerate(steps):
        actions = step.get("actions") or ([step["action"]] if "action" in step else [])
        clicks = [a for a in actions if a.get("type") == "click" and a.get("x") is not None]
        if len(actions) == 1 and len(clicks) == 1:
            x, y = clicks[0]["x"], clicks[0]["y"]
            if any(i - j <= lookback and abs(x - px) <= radius_px and abs(y - py) <= radius_px
                   for px, py, j in recent):
                waste.add(i)
        for a in clicks:
            recent.append((a["x"], a["y"], i))
    return waste


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


def convert_trajectory_to_examples(trajectory, trajectory_dir, exclude_waste_steps=False):
    """One training example PER STEP: current screenshot + task + windowed text history ->
    <think> + action(s). A whole trajectory as one multi-turn example packs every screenshot
    (~300 visual tokens each) into a single sequence — a 21-step trajectory is ~9k tokens
    against max_length 4096, so the trainer silently truncated exactly the long-horizon tails
    this dataset exists for. Per-step examples match inference conditions (one screenshot +
    text history) and supervise every step uniformly.

    Returns (examples, waste_dropped). With exclude_waste_steps, retry/backtrack clicks are
    omitted as TARGETS but still flow into action_history, so the trajectory keeps all its steps
    and downstream context stays intact — only the bad supervision target is removed."""
    instruction = trajectory["instruction"]
    steps = trajectory["steps"]
    if not steps:
        return [], 0

    examples = []
    action_history = []
    waste = _waste_step_indices(steps) if exclude_waste_steps else set()
    waste_dropped = 0

    for i, step in enumerate(steps):
        screenshot_path = (trajectory_dir / step["screenshot"]).resolve()
        thought = step.get("thought", "")
        actions = step.get("actions", [])
        if not actions and "action" in step:
            actions = [step["action"]]

        emit = screenshot_path.exists() and bool(actions)
        if emit and i in waste:
            waste_dropped += 1
        elif emit:
            user_text = build_step_user_text(instruction, action_history)
            action_str = "\n".join(format_action_for_training(a) for a in actions)
            assistant_content = f"<think>{thought}</think>\n{action_str}" if thought else action_str
            examples.append({"messages": [
                *build_step_messages(SYSTEM_PROMPT, user_text, str(screenshot_path)),
                {"role": "assistant", "content": assistant_content},
            ]})

        action_history.extend(history_entries_for_step(step))

    return examples, waste_dropped


def convert_trajectories(input_dir, output_dir, val_split, include_replay=False,
                         exclude_meandering=False, exclude_waste_steps=False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_examples = []
    leak_sources = []
    examples_by_group = {}
    skipped = {"excluded": 0, "unverified": 0, "replay_gated": 0, "empty": 0, "meandering": 0, "stuck_loop": 0}
    report = {
        "trajectories": 0, "by_source": {}, "actions": {}, "lengths": [],
        "domains": {}, "done_steps": 0, "thought_sources": {}, "meandering": 0,
        "waste_steps_dropped": 0, "autonomy": {}, "auth_mode": {},
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
        if metadata.get("excluded_reason") or capture_meta.get("excluded_reason"):
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

        # Stuck loops (>= LOOP_RUN_LIMIT identical consecutive actions) are bad supervision —
        # auto-excluded (same detection as corpus_audit) so the corpus self-cleans as capture scales.
        max_run = run = 0
        prev = None
        for step in trajectory.get("steps", []):
            sig = _action_signature(step)
            run = run + 1 if sig == prev and sig else 1
            prev = sig
            max_run = max(max_run, run)
        if max_run >= LOOP_RUN_LIMIT:
            skipped["stuck_loop"] += 1
            continue

        if source == "agent":
            _maybe_synthesize_done_step(trajectory, traj_dir, capture_meta)

        examples, waste_dropped = convert_trajectory_to_examples(
            trajectory, traj_dir, exclude_waste_steps=exclude_waste_steps)
        report["waste_steps_dropped"] += waste_dropped
        if not examples:
            skipped["empty"] += 1
            continue

        all_examples.extend(examples)
        leaks = _scan_examples_for_secrets(examples)
        if leaks:
            leak_sources.append((traj_dir, leaks))
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
        task_meta = capture_meta.get("task_meta") or {}
        au = metadata.get("autonomy") or task_meta.get("autonomy") or "autonomous"
        report["autonomy"][au] = report["autonomy"].get(au, 0) + 1
        am = metadata.get("auth_mode") or task_meta.get("auth_mode") or "none"
        report["auth_mode"][am] = report["auth_mode"].get(am, 0) + 1
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
    print(f"Autonomy: {report['autonomy']}  Auth modes: {report['auth_mode']}")
    print(f"Thought sources: {report['thought_sources']}")
    if lengths:
        mid = lengths[len(lengths) // 2]
        print(f"Steps/trajectory: min={lengths[0]} median={mid} p90={lengths[int(len(lengths)*0.9)]} max={lengths[-1]}")
    print("Action mix: " + ", ".join(
        f"{t}={n} ({n / total_actions:.0%})" for t, n in sorted(report["actions"].items(), key=lambda kv: -kv[1])
    ))
    print(f"Terminal done() steps: {report['done_steps']}/{report['trajectories']} trajectories")
    print(f"Meandering (tagged, included; pass --exclude-meandering to drop): {report['meandering']}")
    print(f"Waste steps dropped as targets (--exclude-waste-steps): {report['waste_steps_dropped']}")
    print("Domains: " + ", ".join(f"{d}={n}" for d, n in sorted(report["domains"].items(), key=lambda kv: -kv[1])[:10]))
    print(f"Skipped: {skipped}")
    print("======================\n")

    identities = operator_pii.configured_identities()
    print("Operator-identity patterns: " + (", ".join(operator_pii.mask(v) for v in identities)
          or "NONE — the PII gate is inert; set OPERATOR_PII or credentials/operator_pii.txt"))

    # Leak gate (FAIL CLOSED): any literal secret or operator identity refuses the whole dataset.
    if leak_sources:
        print(f"SECRET/PII LEAK DETECTED in {len(leak_sources)} trajectories — refusing to write dataset.")
        for traj_dir, findings in leak_sources:
            print(f"  {traj_dir}: {', '.join(findings)}")
        print("Quarantine each one (metadata.json excluded_reason) or re-capture: credentials belong in "
              "<secret:NAME> placeholders, and a real person's identity belongs nowhere in the corpus.")
        sys.exit(1)

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


def train_model(config_path, dataset_override=None, epochs_override=None, lr_override=None,
                batch_size_override=None, passthrough_args=None):
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

    # convert writes sft_val.jsonl beside the train file; without this it was generated and discarded,
    # so no eval_loss was ever computed and overfitting was invisible.
    if not config.get("val_dataset") and config.get("dataset"):
        sibling = Path(config["dataset"]).parent / "sft_val.jsonl"
        if sibling.is_file():
            config["val_dataset"] = str(sibling)
            print(f"Validation split found: {sibling}")
        else:
            # eval_strategy without a val set makes the trainer raise at startup, so honour the
            # warning instead of dying: no split means no evaluation.
            dropped = config.pop("eval_strategy", None)
            print(f"No validation split at {sibling} — training will report no eval_loss"
                  + (f" (eval_strategy={dropped} dropped)" if dropped else ""))

    cmd = ["swift", "sft"]

    flag_map = {
        "model": "--model",
        "train_type": "--tuner_type",
        "dataset": "--dataset",
        "val_dataset": "--val_dataset",
        "eval_strategy": "--eval_strategy",
        "packing": "--packing",
        "attn_impl": "--attn_impl",
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

    # Anything argparse did not recognise is a swift flag; forwarding beats growing flag_map per knob.
    cmd.extend(passthrough_args or [])

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
    convert_parser.add_argument("--exclude-waste-steps", action="store_true",
                                help="Drop retry/backtrack click steps as training targets (kept in "
                                     "history); non-destructive, trajectory files untouched")
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

    args, passthrough = parser.parse_known_args()

    if args.command == "convert":
        convert_trajectories(args.input, args.output, args.val_split, include_replay=args.include_replay,
                             exclude_meandering=args.exclude_meandering,
                             exclude_waste_steps=args.exclude_waste_steps)
    elif args.command == "train":
        train_model(args.config, args.dataset, args.epochs, args.lr, args.batch_size, passthrough)


if __name__ == "__main__":
    main()
