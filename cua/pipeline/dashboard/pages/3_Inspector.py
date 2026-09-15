"""Per-trajectory inspector: pick one trajectory and step through its screenshots with action overlays."""

import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from cua.pipeline.dashboard._state import (
    DEFAULT_TRAJ_DIR,
    cached_load_corpus,
    page_header,
    sidebar_global_controls,
)

BBOX_COLOR = (46, 160, 67, 220)
CLICK_COLOR = (214, 39, 40, 220)


def overlay_screenshot(path: Path, bbox: dict | None, x: float | None, y: float | None) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    if bbox is not None:
        bx, by = bbox.get("x"), bbox.get("y")
        bw, bh = bbox.get("width"), bbox.get("height")
        if None not in (bx, by, bw, bh):
            draw.rectangle([bx, by, bx + bw, by + bh], outline=BBOX_COLOR, width=4)
    if x is not None and y is not None:
        r = 10
        draw.ellipse([x - r, y - r, x + r, y + r], fill=CLICK_COLOR, outline=(255, 255, 255, 255), width=2)
    return Image.alpha_composite(img, canvas)


st.set_page_config(page_title="Trajectory inspector", page_icon="🔎", layout="wide")
page_header(
    "Trajectory inspector",
    "Step-by-step screenshots with action overlays. Bbox is drawn when available; "
    "click point is drawn whenever an `(x, y)` action was recorded.",
)

traj_df, step_df, warnings = cached_load_corpus(str(DEFAULT_TRAJ_DIR))
sidebar_global_controls(warnings)

if traj_df.empty:
    st.warning("No trajectories loaded.")
    st.stop()

picker_labels = [f"({row.path_kind})  {row.task_id}" for row in traj_df.itertuples()]
picker_to_task = dict(zip(picker_labels, traj_df["task_id"].tolist(), strict=False))

with st.sidebar:
    st.markdown("### Inspector")
    pick = st.selectbox("Trajectory", picker_labels)
    show_overlay = st.checkbox("Draw bbox + click overlay", value=True)

task_id = picker_to_task[pick]
traj_row = traj_df[traj_df["task_id"] == task_id].iloc[0]
task_steps = step_df[step_df["task_id"] == task_id].sort_values("step")

meta_cols = st.columns(4)
meta_cols[0].metric("Steps", int(traj_row["n_steps_actual"]))
meta_cols[1].metric("Outcome", "✅ success" if traj_row["success"] else "❌ fail")
meta_cols[2].metric("Has capture_meta", "yes" if traj_row["has_capture_meta"] else "no")
meta_cols[3].metric("Source", traj_row["source_repo"] or "—")

st.markdown(f"**Instruction.** {traj_row['instruction'] or '_(empty)_'}")
st.markdown(f"**Start URL.** `{traj_row['start_url']}`")

with st.expander("Raw metadata.json"):
    metadata_path = DEFAULT_TRAJ_DIR / traj_row["trajectory_dir"] / "metadata.json"
    if metadata_path.exists():
        st.code(metadata_path.read_text(), language="json")
    else:
        st.caption("metadata.json missing")

st.markdown("---")
st.subheader(f"Steps ({len(task_steps)})")

for row in task_steps.itertuples():
    st.markdown(f"#### Step {row.step} · `{row.action_type}`")
    img_col, info_col = st.columns([3, 2])
    screenshot_path = Path(row.screenshot_path)
    if screenshot_path.exists():
        try:
            if show_overlay:
                img = overlay_screenshot(screenshot_path, row.bbox, row.x, row.y)
            else:
                img = Image.open(screenshot_path)
            img_col.image(img, width="stretch")
        except (OSError, ValueError) as exc:
            img_col.warning(f"Could not render screenshot: {exc}")
    else:
        img_col.warning(f"Screenshot missing: `{screenshot_path.name}`")

    info_col.markdown(f"**Action** ({row.n_actions})")
    info_col.code(json.dumps(row.actions, indent=2, default=str), language="json")

    info_col.markdown("**Target element** (from `capture_meta.json` / inline)")
    elem_block = {
        "role": row.role,
        "name": row.name,
        "bbox": row.bbox,
        "selector": row.selector,
        "tag": row.tag,
        "in_viewport": row.in_viewport,
        "is_visible": row.is_visible,
        "is_enabled": row.is_enabled,
    }
    info_col.code(json.dumps(elem_block, indent=2, default=str), language="json")

    if (row.thought or "").strip():
        info_col.markdown("**Thought**")
        info_col.write(row.thought)
    else:
        info_col.caption("(no thought recorded)")
    st.markdown("---")
