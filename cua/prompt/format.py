"""The student's observation format: the system prompt, the interleaved user turn, and its wire renderings."""

import base64

from cua.prompt.coords import to_model_action
from cua.prompt.effects import effect_annotation
from cua.settings import PROMPT
from cua.trajectory.actions import format_action_summary

SYSTEM_PROMPT = """You are a GUI agent that operates a web browser at a 1280x720 resolution. \
Given a task, the current screenshot (and, after the first step, a downscaled screenshot of the \
screen before your last action), and your previous actions annotated with their observed effect \
([no change] when the screen did not change, -> /path when the URL path changed), reason about \
the state in <think>...</think>, then output one or more actions, one per line.

Available actions (x, y are normalized to 0-1000 across the screenshot width and height; \
scrollX/scrollY are pixel deltas):
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

GROUNDING_SYSTEM_PROMPT = """You are a GUI agent that operates a web browser at a 1280x720 resolution. \
Given a screenshot and the description of one UI element, output exactly one action that clicks it: \
action: click(<ref>(x, y)</ref>, button=left) with x, y normalized to 0-1000 across the screenshot \
width and height. No reasoning, no other text."""

PROMPT_FORMAT_VERSION = "sft3"
PREV_IMAGE_SIZE = PROMPT.prev_image_size
# 20 covers the corpus's median 24-step auth trajectories; 10 truncated 56.5% of steps.
HISTORY_WINDOW = 20

PREV_IMAGE_HEADER = "Previous screenshot (before your last action, downscaled):\n"
CUR_IMAGE_HEADER = "Current screenshot:\n"
IMAGE_PLACEHOLDER = "<image>"
GROUNDING_INSTRUCTION_TEMPLATE = "Click {description}."


def png_b64(png_bytes):
    return base64.b64encode(png_bytes).decode()


def _text_item(text):
    return {"type": "text", "text": text}


def _image_item(ref):
    return {"type": "image", "image": ref}


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


def history_entries_for_step(step, effect=None):
    """Model-space summaries of a step's (pixel-space) actions; the observed effect annotates the last entry."""
    actions = step.get("actions", [])
    if not actions and "action" in step:
        actions = [step["action"]]
    entries = [format_action_summary(to_model_action(a)) for a in actions]
    event = step.get("event") or {}
    if event.get("kind") == "approval":
        entries.append(f"[watchdog: {event.get('decision', '?')}]")
    if effect is not None and entries:
        entries[-1] += effect_annotation(effect)
    return entries


def build_step_observation(instruction, action_history, cur_image, prev_image=None):
    """The student's user turn as interleaved text/image items; `text` is its ms-swift `<image>` flattening."""
    if prev_image is not None:
        content = [_text_item(PREV_IMAGE_HEADER), _image_item(prev_image), _text_item("\n" + CUR_IMAGE_HEADER)]
    else:
        content = [_text_item(CUR_IMAGE_HEADER)]
    content += [_image_item(cur_image), _text_item("\n" + build_step_user_text(instruction, action_history))]
    images = [item["image"] for item in content if item["type"] == "image"]
    return {"content": content, "images": images, "text": flatten_for_swift(content)}


def build_grounding_observation(description, image_ref):
    """A grounding user turn IS a history-free first-step observation whose task is to click one described element."""
    return build_step_observation(GROUNDING_INSTRUCTION_TEMPLATE.format(description=description), [], image_ref)


def flatten_for_swift(content):
    return "".join(IMAGE_PLACEHOLDER if item["type"] == "image" else item["text"] for item in content)


def build_step_messages(system, observation):
    return [{"role": "system", "content": system}, {"role": "user", "content": observation["content"]}]


def to_openai_content(content, image_b64_for):
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64_for(item['image'])}"}}
        if item["type"] == "image"
        else _text_item(item["text"])
        for item in content
    ]


def to_claude_content(content, image_b64_for):
    """The same observation in Anthropic's block form, so a frontier arm is prompted identically."""
    return [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64_for(item["image"])}}
        if item["type"] == "image"
        else _text_item(item["text"])
        for item in content
    ]
