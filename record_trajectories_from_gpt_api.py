import argparse
import asyncio
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from playwright.async_api import async_playwright
from tqdm import tqdm


def parse_viewport(viewport_str):
    width, height = viewport_str.lower().split("x")
    return int(width), int(height)


def encode_screenshot_to_base64(screenshot_bytes):
    return base64.b64encode(screenshot_bytes).decode("utf-8")


def extract_computer_call(response):
    for item in response.output:
        if item.type == "computer_call":
            return item
    return None


def extract_reasoning(response):
    reasoning_parts = []
    for item in response.output:
        if item.type == "reasoning":
            if hasattr(item, "summary") and item.summary:
                for content in item.summary:
                    if hasattr(content, "text"):
                        reasoning_parts.append(content.text)
        elif item.type == "text":
            reasoning_parts.append(item.text)
    return " ".join(reasoning_parts) if reasoning_parts else ""


def has_pending_safety_checks(response):
    computer_call = extract_computer_call(response)
    if computer_call and hasattr(computer_call, "pending_safety_checks") and computer_call.pending_safety_checks:
        return True
    return False


CUA_KEY_TO_PLAYWRIGHT = {
    "CTRL": "Control",
    "CONTROL": "Control",
    "ALT": "Alt",
    "SHIFT": "Shift",
    "META": "Meta",
    "CMD": "Meta",
    "COMMAND": "Meta",
    "ENTER": "Enter",
    "RETURN": "Enter",
    "TAB": "Tab",
    "ESCAPE": "Escape",
    "ESC": "Escape",
    "BACKSPACE": "Backspace",
    "DELETE": "Delete",
    "SPACE": " ",
    "UP": "ArrowUp",
    "DOWN": "ArrowDown",
    "LEFT": "ArrowLeft",
    "RIGHT": "ArrowRight",
    "ARROWUP": "ArrowUp",
    "ARROWDOWN": "ArrowDown",
    "ARROWLEFT": "ArrowLeft",
    "ARROWRIGHT": "ArrowRight",
    "PAGEUP": "PageUp",
    "PAGEDOWN": "PageDown",
    "HOME": "Home",
    "END": "End",
    "INSERT": "Insert",
    "F1": "F1", "F2": "F2", "F3": "F3", "F4": "F4",
    "F5": "F5", "F6": "F6", "F7": "F7", "F8": "F8",
    "F9": "F9", "F10": "F10", "F11": "F11", "F12": "F12",
}


def map_key_for_playwright(key):
    return CUA_KEY_TO_PLAYWRIGHT.get(key.upper(), key)


def is_screenshot_only(computer_call):
    actions = getattr(computer_call, "actions", None) or []
    return len(actions) == 1 and actions[0].type == "screenshot"


async def execute_actions(page, actions):
    for action in actions:
        action_type = action.type

        if action_type == "click":
            button = getattr(action, "button", "left") or "left"
            await page.mouse.click(action.x, action.y, button=button)

        elif action_type == "double_click":
            await page.mouse.dblclick(action.x, action.y)

        elif action_type == "drag":
            path = action.path
            if len(path) >= 2:
                start = path[0]
                end = path[-1]
                await page.mouse.move(start["x"], start["y"])
                await page.mouse.down()
                for point in path[1:]:
                    await page.mouse.move(point["x"], point["y"])
                await page.mouse.up()

        elif action_type == "move":
            await page.mouse.move(action.x, action.y)

        elif action_type == "type":
            await page.keyboard.type(action.text)

        elif action_type == "keypress":
            keys = [map_key_for_playwright(k) for k in action.keys]
            key_combo = "+".join(keys)
            await page.keyboard.press(key_combo)

        elif action_type == "scroll":
            await page.mouse.move(action.x, action.y)
            await page.mouse.wheel(action.scroll_x, action.scroll_y)

        elif action_type == "wait":
            await asyncio.sleep(2)

        elif action_type == "screenshot":
            pass

        else:
            print(f"  Unknown action type: {action_type}")


def actions_to_dicts(actions):
    result = []
    for action in actions:
        action_type = action.type
        d = {"type": action_type}

        if action_type == "click":
            d["x"] = action.x
            d["y"] = action.y
            d["button"] = getattr(action, "button", "left") or "left"

        elif action_type == "double_click":
            d["x"] = action.x
            d["y"] = action.y

        elif action_type == "drag":
            d["path"] = [{"x": p["x"], "y": p["y"]} for p in action.path]

        elif action_type == "move":
            d["x"] = action.x
            d["y"] = action.y

        elif action_type == "type":
            d["text"] = action.text

        elif action_type == "keypress":
            d["keys"] = list(action.keys)

        elif action_type == "scroll":
            d["x"] = action.x
            d["y"] = action.y
            d["scroll_x"] = action.scroll_x
            d["scroll_y"] = action.scroll_y

        result.append(d)
    return result


def send_screenshot(client, response, call_id, b64_screenshot):
    return client.responses.create(
        model="gpt-5.4-mini",
        previous_response_id=response.id,
        tools=[{"type": "computer"}],
        input=[{
            "type": "computer_call_output",
            "call_id": call_id,
            "output": {
                "type": "computer_screenshot",
                "image_url": f"data:image/png;base64,{b64_screenshot}",
            },
        }],
        truncation="auto",
    )


async def record_single_task(client, task, output_dir, max_steps, viewport_width, viewport_height):
    task_id = task["task_id"]
    task_dir = output_dir / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    steps = []
    success = True
    error_message = None

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            device_scale_factor=1,
        )
        page = await context.new_page()

        try:
            await page.goto(task["start_url"], timeout=30000, wait_until="domcontentloaded")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"  Failed to navigate to {task['start_url']}: {e}")
            success = False
            error_message = f"Navigation failed: {e}"
            await browser.close()
            save_metadata(task, task_dir, steps, success, viewport_width, viewport_height, error_message)
            return

        try:
            response = client.responses.create(
                model="gpt-5.4-mini",
                tools=[{"type": "computer"}],
                input=[{
                    "role": "user",
                    "content": task["instruction"],
                }],
                truncation="auto",
            )
        except Exception as e:
            print(f"  API error on initial request: {e}")
            success = False
            error_message = f"API error: {e}"
            await browser.close()
            save_metadata(task, task_dir, steps, success, viewport_width, viewport_height, error_message)
            return

        screenshot_counter = 1

        for step_num in range(1, max_steps + 1):
            if has_pending_safety_checks(response):
                print(f"  Safety check triggered at step {step_num}, stopping task")
                success = False
                error_message = "Safety check triggered"
                break

            computer_call = extract_computer_call(response)
            if not computer_call:
                break

            call_id = computer_call.call_id
            actions = getattr(computer_call, "actions", None) or []

            if is_screenshot_only(computer_call):
                screenshot_bytes = await page.screenshot(type="png")
                screenshot_path = task_dir / f"step_{screenshot_counter:03d}_screenshot.png"
                screenshot_path.write_bytes(screenshot_bytes)
                screenshot_counter += 1
                b64_screenshot = encode_screenshot_to_base64(screenshot_bytes)

                try:
                    response = send_screenshot(client, response, call_id, b64_screenshot)
                except Exception as e:
                    print(f"  API error at step {step_num} (screenshot): {e}")
                    success = False
                    error_message = f"API error at step {step_num}: {e}"
                    break
                continue

            reasoning = extract_reasoning(response)
            action_dicts = actions_to_dicts(actions)

            screenshot_bytes = await page.screenshot(type="png")
            screenshot_filename = f"step_{screenshot_counter:03d}_screenshot.png"
            screenshot_path = task_dir / screenshot_filename
            screenshot_path.write_bytes(screenshot_bytes)
            screenshot_counter += 1

            steps.append({
                "step": len(steps) + 1,
                "screenshot": screenshot_filename,
                "thought": reasoning,
                "actions": action_dicts,
            })

            tqdm.write(f"  Step {len(steps)}: {[a['type'] for a in action_dicts]}")

            await execute_actions(page, actions)
            await asyncio.sleep(1)

            screenshot_bytes = await page.screenshot(type="png")
            b64_screenshot = encode_screenshot_to_base64(screenshot_bytes)

            try:
                response = send_screenshot(client, response, call_id, b64_screenshot)
            except Exception as e:
                print(f"  API error at step {step_num}: {e}")
                success = False
                error_message = f"API error at step {step_num}: {e}"
                break

        await browser.close()

    trajectory = {
        "task_id": task_id,
        "instruction": task["instruction"],
        "start_url": task["start_url"],
        "steps": steps,
        "total_steps": len(steps),
        "success": success,
    }

    trajectory_path = task_dir / "trajectory.json"
    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2)

    save_metadata(task, task_dir, steps, success, viewport_width, viewport_height, error_message)
    return trajectory


def save_metadata(task, task_dir, steps, success, viewport_width, viewport_height, error_message=None):
    metadata = {
        "task_id": task["task_id"],
        "instruction": task["instruction"],
        "start_url": task["start_url"],
        "domain": task.get("domain", ""),
        "website": task.get("website", ""),
        "total_steps": len(steps),
        "success": success,
        "viewport": {"width": viewport_width, "height": viewport_height},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if error_message:
        metadata["error"] = error_message

    metadata_path = task_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


async def run(args):
    client = OpenAI()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    viewport_width, viewport_height = parse_viewport(args.viewport)

    with open(args.tasks) as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(tasks)} tasks")
    print(f"Output directory: {output_dir}")
    print(f"Viewport: {viewport_width}x{viewport_height}")
    print(f"Max steps per task: {args.max_steps}")

    skipped = 0
    for task in tqdm(tasks, desc="Recording trajectories"):
        task_id = task["task_id"]
        task_dir = output_dir / f"task_{task_id}"

        if args.resume and task_dir.exists() and (task_dir / "trajectory.json").exists():
            skipped += 1
            continue

        tqdm.write(f"Recording: {task_id} - {task['instruction'][:60]}...")

        try:
            await record_single_task(client, task, output_dir, args.max_steps, viewport_width, viewport_height)
        except Exception as e:
            tqdm.write(f"  FAILED: {task_id} - {e}")
            task_dir.mkdir(parents=True, exist_ok=True)
            save_metadata(task, task_dir, [], False, viewport_width, viewport_height, str(e))

    if skipped:
        print(f"Skipped {skipped} already-recorded tasks (--resume)")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Record web browsing trajectories using GPT-5.4 CUA")
    parser.add_argument("--tasks", type=str, required=True, help="Path to tasks.jsonl file")
    parser.add_argument("--output", type=str, default="trajectories/", help="Output directory for trajectories")
    parser.add_argument("--max-steps", type=int, default=15, help="Maximum actions per task")
    parser.add_argument("--viewport", type=str, default="1280x720", help="Browser viewport size (WIDTHxHEIGHT)")
    parser.add_argument("--resume", action="store_true", help="Skip tasks that already have trajectories")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
