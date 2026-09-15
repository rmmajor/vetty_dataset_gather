"""Every environment knob the package reads, resolved once at import into frozen sections."""

import os
from dataclasses import dataclass


def parse_viewport(text):
    """`1280x720` -> (1280, 720); also the env cast for PREV_IMAGE_SIZE, re-exported by cua.prompt.coords."""
    width, height = text.lower().split("x")
    return int(width), int(height)


def _env(name, default, cast=str):
    return cast(os.environ.get(name, default))


@dataclass(frozen=True)
class PromptSettings:
    prev_image_size: tuple
    effect_no_change_max_fraction: float
    effect_pixel_delta: int
    effect_path_max_chars: int


@dataclass(frozen=True)
class CorpusSettings:
    waste_click_radius_px: int
    waste_lookback_steps: int
    meander_keep_head: int
    meander_keep_tail: int
    meander_mid_keep_every: int
    keypress_max_repeat: int


@dataclass(frozen=True)
class GroundingSettings:
    small_oversample: int
    per_group_cap: int
    seed: int


PROMPT = PromptSettings(
    prev_image_size=_env("PREV_IMAGE_SIZE", "640x360", parse_viewport),
    effect_no_change_max_fraction=_env("EFFECT_NO_CHANGE_MAX_FRACTION", "0.0005", float),
    effect_pixel_delta=_env("EFFECT_PIXEL_DELTA", "16", int),
    effect_path_max_chars=_env("EFFECT_PATH_MAX_CHARS", "40", int),
)

CORPUS = CorpusSettings(
    waste_click_radius_px=_env("WASTE_CLICK_RADIUS_PX", "10", int),
    waste_lookback_steps=_env("WASTE_LOOKBACK_STEPS", "4", int),
    meander_keep_head=_env("MEANDER_KEEP_HEAD", "5", int),
    meander_keep_tail=_env("MEANDER_KEEP_TAIL", "5", int),
    meander_mid_keep_every=_env("MEANDER_MID_KEEP_EVERY", "6", int),
    keypress_max_repeat=_env("KEYPRESS_MAX_REPEAT", "6", int),
)

GROUNDING = GroundingSettings(
    small_oversample=_env("GROUNDING_SMALL_OVERSAMPLE", "3", int),
    per_group_cap=_env("GROUNDING_PER_GROUP_CAP", "40", int),
    seed=_env("GROUNDING_SEED", "0", int),
)
