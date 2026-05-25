import json
import os
from pathlib import Path

TRAJECTORY_DIR = os.environ.get("TRAJECTORY_DIR", "/workspace/results")
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720


class ActionRecorder:
    def __init__(self, test_name: str, output_dir: str = None):
        self.test_name = test_name
        self.output_dir = Path(output_dir or TRAJECTORY_DIR) / test_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps = []
        self.step_counter = 0

    def record_action(
        self,
        action_type: str,
        x: int,
        y: int,
        screenshot_bytes: bytes,
        selector: str = "",
        value: str = "",
        button: str = "left",
        bounding_box: dict = None,
    ):
        self.step_counter += 1
        screenshot_filename = f"step_{self.step_counter:03d}_screenshot.png"
        screenshot_path = self.output_dir / screenshot_filename
        screenshot_path.write_bytes(screenshot_bytes)

        action = {"type": action_type}
        if action_type in ("click", "double_click", "move"):
            action["x"] = x
            action["y"] = y
            if action_type == "click":
                action["button"] = button
        elif action_type == "type":
            action["text"] = value
        elif action_type == "scroll":
            action["x"] = x
            action["y"] = y
            action["scroll_x"] = 0
            action["scroll_y"] = int(value) if value else 0
        elif action_type == "keypress":
            action["keys"] = value.split("+") if isinstance(value, str) else value
        elif action_type == "drag":
            action["path"] = [{"x": x, "y": y}]

        self.steps.append({
            "step": self.step_counter,
            "screenshot": screenshot_filename,
            "thought": "",
            "actions": [action],
            "selector": selector,
            "bounding_box": bounding_box,
        })

    def save(self, instruction: str = "", start_url: str = "", success: bool = True):
        clean_steps = []
        for step in self.steps:
            clean_steps.append({
                "step": step["step"],
                "screenshot": step["screenshot"],
                "thought": step["thought"],
                "actions": step["actions"],
            })

        trajectory = {
            "task_id": self.test_name,
            "instruction": instruction,
            "start_url": start_url,
            "steps": clean_steps,
            "total_steps": len(clean_steps),
            "success": success,
        }

        trajectory_path = self.output_dir / "trajectory.json"
        with open(trajectory_path, "w") as f:
            json.dump(trajectory, f, indent=2)

        metadata = {
            "task_id": self.test_name,
            "instruction": instruction,
            "start_url": start_url,
            "domain": "",
            "website": "",
            "total_steps": len(clean_steps),
            "success": success,
            "viewport": {"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            "source_repo": os.environ.get("SOURCE_REPO", ""),
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return trajectory_path


def calculate_center(bounding_box: dict) -> tuple[int, int]:
    x = int(bounding_box["x"] + bounding_box["width"] / 2)
    y = int(bounding_box["y"] + bounding_box["height"] / 2)
    return x, y
