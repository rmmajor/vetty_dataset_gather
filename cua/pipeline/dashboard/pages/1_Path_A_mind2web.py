"""Path A page — mind2web / GPT-5.4 CUA trajectories.

Real-web trajectories recorded by `cua/path_a/recorder.py`. No in-sandbox DOM
probe, so `bbox` / `role` / `name` / `tag` / `dt_ms` / `dom_changed` are unset on every step.
Distributions here focus on what is available: action mix, instruction & thought shape,
domain / website spread, and click coordinate density on the viewport.
"""

import streamlit as st

from cua.pipeline.dashboard import charts
from cua.pipeline.dashboard._state import (
    DEFAULT_TRAJ_DIR,
    apply_path_filters,
    cached_load_corpus,
    page_header,
    populated_count,
    render_kpi_strip,
    render_populated_strip,
    sidebar_global_controls,
)

st.set_page_config(page_title="Path A · mind2web", page_icon="🅰️", layout="wide")
page_header(
    "Path A — mind2web / GPT-5.4 CUA",
    "Real-web trajectories. Element-level fields (bbox, role, tag, dom_changed, dt_ms) are "
    "intentionally not captured for Path A and show as empty-state.",
)

traj_df, step_df, warnings = cached_load_corpus(str(DEFAULT_TRAJ_DIR))
sidebar_global_controls(warnings)

filtered_traj, filtered_step = apply_path_filters(traj_df, step_df, path_kind="A")
if filtered_traj.empty:
    st.warning("No Path A trajectories match the current filters.")
    st.stop()

render_kpi_strip(filtered_traj, filtered_step)
render_populated_strip(populated_count(filtered_traj), len(filtered_traj))

st.markdown("---")
tab_o, tab_a, tab_r, tab_s, tab_e = st.tabs(
    [
        "📦 Overview",
        "🎮 Actions",
        "🧠 Reasoning",
        "🌐 Sources",
        "🧩 Elements",
    ]
)

with tab_o:
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.steps_per_traj_hist(filtered_traj), width="stretch", key="a_steps_hist")
    c2.plotly_chart(charts.actions_per_traj_hist(filtered_traj), width="stretch", key="a_actions_hist")
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.success_pie(filtered_traj), width="stretch", key="a_success_pie")
    c4.plotly_chart(charts.instruction_length_hist(filtered_traj), width="stretch", key="a_instr_len")
    c5, c6 = st.columns(2)
    c5.plotly_chart(charts.steps_vs_actions_scatter(filtered_traj), width="stretch", key="a_steps_vs_actions")
    c6.plotly_chart(charts.instruction_vs_steps_scatter(filtered_traj), width="stretch", key="a_instr_vs_steps")

with tab_a:
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.action_type_bar(filtered_step), width="stretch", key="a_action_type_bar")
    c2.plotly_chart(charts.per_traj_action_entropy_hist(filtered_traj), width="stretch", key="a_action_entropy_hist")
    st.plotly_chart(
        charts.unique_action_types_per_traj_hist(filtered_traj), width="stretch", key="a_unique_action_types"
    )
    st.plotly_chart(charts.action_mix_per_traj_stack(filtered_traj), width="stretch", key="a_action_mix_stack")
    st.plotly_chart(charts.click_heatmap(filtered_step), width="stretch", key="a_click_heatmap")

with tab_r:
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.thought_presence_bar(filtered_step), width="stretch", key="a_thought_presence")
    c2.plotly_chart(charts.thought_length_hist(filtered_step), width="stretch", key="a_thought_length")

with tab_s:
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.domain_bar(filtered_traj, column="domain"), width="stretch", key="a_domain_bar")
    c2.plotly_chart(charts.domain_bar(filtered_traj, column="website"), width="stretch", key="a_website_bar")
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.per_domain_steps_box(filtered_traj, by="domain"), width="stretch", key="a_domain_steps_box")
    c4.plotly_chart(charts.task_type_bar(filtered_traj), width="stretch", key="a_task_type_bar")
    st.plotly_chart(charts.recorded_timeline(filtered_traj), width="stretch", key="a_recorded_timeline")
    st.markdown("**Top start URLs**")
    st.dataframe(charts.top_start_urls_table(filtered_traj), hide_index=True, width="stretch")

with tab_e:
    st.caption(
        "Path A does not capture DOM-level element metadata, so the charts in this tab "
        "stay empty by design. Use Path B for element / bbox / tag distributions."
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.element_role_bar(filtered_step), width="stretch", key="a_role_bar")
    c2.plotly_chart(charts.element_tag_bar(filtered_step), width="stretch", key="a_tag_bar")
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.bbox_area_hist(filtered_step), width="stretch", key="a_bbox_area_hist")
    c4.plotly_chart(charts.bbox_contained_pie(filtered_step), width="stretch", key="a_bbox_contained_pie")
