"""Prompt construction: cua.prompt.format's own observation builders, with PNG bytes inlined as data URLs."""

from cua.prompt.effects import downscale_prev, screenshot_changed_fraction, url_path_change
from cua.prompt.format import (
    SYSTEM_PROMPT,
    build_step_messages,
    build_step_observation,
    png_b64,
    to_openai_content,
)

# Per-ARM, not global: an sft-v1/v2 checkpoint trained on ONE screenshot, so a two-image turn measures the prompt.
OBSERVATION_FORMATS = ("v2", "v3")
DEFAULT_OBSERVATION = "v3"


def observed_effect(prev_png: bytes, cur_png: bytes, prev_url: str, cur_url: str) -> dict:
    """What the next screenshot/url showed the last actions did, in the shape cua.prompt.effects annotates with."""
    return {"changed": screenshot_changed_fraction(prev_png, cur_png), "path": url_path_change(prev_url, cur_url)}


def build_student_messages(
    instruction: str,
    action_history: list,
    cur_png: bytes,
    prev_png: bytes | None = None,
    observation: str = DEFAULT_OBSERVATION,
) -> list:
    """OpenAI chat messages for one step; v3 adds the previous screenshot from step 2 on, v2 never does."""
    if observation not in OBSERVATION_FORMATS:
        raise ValueError(f"unknown observation format {observation!r}; expected one of {OBSERVATION_FORMATS}")
    if observation == "v2":
        prev_png = None
    images = {"cur": cur_png}
    if prev_png is not None:
        images["prev"] = downscale_prev(prev_png)
    turn = build_step_observation(instruction, action_history, "cur", "prev" if prev_png is not None else None)
    content = to_openai_content(turn["content"], lambda ref: png_b64(images[ref]))
    return build_step_messages(SYSTEM_PROMPT, {**turn, "content": content})
