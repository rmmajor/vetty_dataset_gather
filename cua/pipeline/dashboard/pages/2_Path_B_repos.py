"""Path B page — QA repo Playwright/Selenium trajectories.

Deterministic test runs in the sandbox. When `capture_meta.json` is present every element
field (`bbox`, `role`, `tag`, `is_visible`, …) and every per-action diagnostic (`dt_ms`,
`dom_changed`, `caused_navigation`) lights up. This is the path where the Elements tab is
meaningful.
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

st.set_page_config(page_title="Path B · QA repos", page_icon="🅱️", layout="wide")
page_header(
    "Path B — QA repos (Playwright / Selenium)",
    "Element fields fill in once stage 6 writes `capture_meta.json`. Until then the "
    "Elements tab and the `n_navigations` / `dom_changed` / `dt_ms` panels stay empty.",
)

traj_df, step_df, warnings = cached_load_corpus(str(DEFAULT_TRAJ_DIR))
sidebar_global_controls(warnings)

filtered_traj, filtered_step = apply_path_filters(traj_df, step_df, path_kind="B")
if filtered_traj.empty:
    st.warning("No Path B trajectories match the current filters.")
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
    c1.plotly_chart(charts.steps_per_traj_hist(filtered_traj), width="stretch", key="b_steps_hist")
    c2.plotly_chart(charts.actions_per_traj_hist(filtered_traj), width="stretch", key="b_actions_hist")
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.success_pie(filtered_traj), width="stretch", key="b_success_pie")
    c4.plotly_chart(charts.instruction_length_hist(filtered_traj), width="stretch", key="b_instr_len")
    c5, c6 = st.columns(2)
    c5.plotly_chart(charts.steps_vs_actions_scatter(filtered_traj), width="stretch", key="b_steps_vs_actions")
    c6.plotly_chart(charts.instruction_vs_steps_scatter(filtered_traj), width="stretch", key="b_instr_vs_steps")
    st.plotly_chart(charts.navigations_hist(filtered_traj), width="stretch", key="b_navigations_hist")

with tab_a:
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.action_type_bar(filtered_step), width="stretch", key="b_action_type_bar")
    c2.plotly_chart(charts.per_traj_action_entropy_hist(filtered_traj), width="stretch", key="b_action_entropy_hist")
    st.plotly_chart(
        charts.unique_action_types_per_traj_hist(filtered_traj), width="stretch", key="b_unique_action_types"
    )
    st.plotly_chart(charts.action_mix_per_traj_stack(filtered_traj), width="stretch", key="b_action_mix_stack")
    st.plotly_chart(charts.click_heatmap(filtered_step), width="stretch", key="b_click_heatmap")
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.dt_ms_hist(filtered_step), width="stretch", key="b_dt_ms_hist")
    c4.plotly_chart(charts.dom_change_pie(filtered_step), width="stretch", key="b_dom_change_pie")

with tab_r:
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.thought_presence_bar(filtered_step), width="stretch", key="b_thought_presence")
    c2.plotly_chart(charts.thought_length_hist(filtered_step), width="stretch", key="b_thought_length")

with tab_s:
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.source_repo_bar(filtered_traj), width="stretch", key="b_source_repo_bar")
    c2.plotly_chart(charts.framework_bar(filtered_traj), width="stretch", key="b_framework_bar")
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.task_type_bar(filtered_traj), width="stretch", key="b_task_type_bar")
    c4.plotly_chart(charts.domain_bar(filtered_traj, column="domain"), width="stretch", key="b_domain_bar")
    st.plotly_chart(
        charts.per_domain_steps_box(filtered_traj, by="source_repo"), width="stretch", key="b_repo_steps_box"
    )
    st.plotly_chart(charts.recorded_timeline(filtered_traj), width="stretch", key="b_recorded_timeline")
    st.markdown("**Top start URLs**")
    st.dataframe(charts.top_start_urls_table(filtered_traj), hide_index=True, width="stretch")

with tab_e:
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.element_role_bar(filtered_step), width="stretch", key="b_role_bar")
    c2.plotly_chart(charts.element_tag_bar(filtered_step), width="stretch", key="b_tag_bar")
    st.plotly_chart(charts.action_role_heatmap(filtered_step), width="stretch", key="b_action_role_heatmap")
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.bbox_area_hist(filtered_step), width="stretch", key="b_bbox_area_hist")
    c4.plotly_chart(charts.click_to_center_hist(filtered_step), width="stretch", key="b_click_center_hist")
    st.plotly_chart(charts.bbox_contained_pie(filtered_step), width="stretch", key="b_bbox_contained_pie")
