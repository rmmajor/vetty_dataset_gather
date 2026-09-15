"""What the next screenshot and URL show an action did — the history annotations the student is trained to read."""

from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageChops

from cua.settings import PROMPT

EFFECT_DIFF_SIZE = (320, 180)
EFFECT_NO_CHANGE_MAX_FRACTION = PROMPT.effect_no_change_max_fraction
EFFECT_PIXEL_DELTA = PROMPT.effect_pixel_delta
EFFECT_PATH_MAX_CHARS = PROMPT.effect_path_max_chars


def _diff_plane(png_bytes):
    return Image.open(BytesIO(png_bytes)).convert("L").resize(EFFECT_DIFF_SIZE, Image.Resampling.BOX)


def _changed_fraction(plane_a, plane_b):
    histogram = ImageChops.difference(plane_a, plane_b).histogram()
    return sum(histogram[EFFECT_PIXEL_DELTA + 1 :]) / (plane_a.width * plane_a.height)


def screenshot_changed_fraction(png_a_bytes, png_b_bytes):
    return _changed_fraction(_diff_plane(png_a_bytes), _diff_plane(png_b_bytes))


def url_path_change(url_before, url_after):
    if not url_before or not url_after:
        return None
    before = urlparse(url_before).path.rstrip("/")
    after = urlparse(url_after).path.rstrip("/")
    if before == after:
        return None
    return (after or "/")[:EFFECT_PATH_MAX_CHARS]


def effect_is_no_change(effect):
    return effect.get("changed") is not None and effect["changed"] <= EFFECT_NO_CHANGE_MAX_FRACTION


def effect_annotation(effect):
    if not effect:
        return ""
    text = " [no change]" if effect_is_no_change(effect) else ""
    if effect.get("path"):
        text += f" -> {effect['path']}"
    return text


def step_effects(trajectory_dir, steps, capture_meta):
    """Per step, what the NEXT screenshot/url show the action did; None for the last step or a missing screenshot."""
    trajectory_dir = Path(trajectory_dir)
    meta_steps = (capture_meta or {}).get("steps")
    meta_steps = meta_steps if isinstance(meta_steps, dict) else {}
    planes = {}

    def plane_at(index):
        if index not in planes:
            file = trajectory_dir / steps[index]["screenshot"]
            planes[index] = _diff_plane(file.read_bytes()) if file.is_file() else None
        return planes[index]

    effects = []
    for k, step in enumerate(steps):
        if k + 1 == len(steps) or plane_at(k) is None or plane_at(k + 1) is None:
            effects.append(None)
            continue
        url_before = (meta_steps.get(str(step.get("step"))) or {}).get("url")
        url_after = (meta_steps.get(str(steps[k + 1].get("step"))) or {}).get("url")
        effects.append(
            {"changed": _changed_fraction(plane_at(k), plane_at(k + 1)), "path": url_path_change(url_before, url_after)}
        )
        planes.pop(k, None)
    return effects


def downscale_prev(png_bytes, size=None):
    image = (
        Image.open(BytesIO(png_bytes)).convert("RGB").resize(size or PROMPT.prev_image_size, Image.Resampling.LANCZOS)
    )
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
