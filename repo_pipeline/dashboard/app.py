"""Trajectory dashboard — landing page.

Cross-corpus exploratory view. Per-path breakdowns live on the Path A / Path B subpages;
this page answers the "what does the corpus look like overall?" question with raw
distributions: outcome split, steps/actions per trajectory, action-type mix, domain spread,
source breakdown, recording timeline.

Run: `make dashboard`  →  http://localhost:8501
"""

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_pipeline.dashboard import charts
from repo_pipeline.dashboard._state import (
    DEFAULT_TRAJ_DIR,
    cached_load_corpus,
    page_header,
    populated_count,
    render_kpi_strip,
    render_populated_strip,
    sidebar_global_controls,
)
from repo_pipeline.dashboard.loader import load_iso_date_range


st.set_page_config(page_title="Trajectory dashboard", page_icon="📊", layout="wide")

page_header(
    "Trajectory dashboard",
    "Exploratory view over the trajectory corpus. Pick a path from the sidebar for deeper "
    "per-path distributions, or open the Inspector to look at individual trajectories."
)

traj_df, step_df, warnings = cached_load_corpus(str(DEFAULT_TRAJ_DIR))
sidebar_global_controls(warnings)

if traj_df.empty:
    st.warning(
        f"No trajectories found at `{DEFAULT_TRAJ_DIR}`. Generate some via the repo pipeline "
        "(stage 6) or via `make record` for Path A, then click 🔄 Refresh."
    )
    st.stop()

render_kpi_strip(traj_df, step_df)
render_populated_strip(populated_count(traj_df), len(traj_df))

date_lo, date_hi = load_iso_date_range(traj_df)
if date_lo:
    st.caption(f"Recorded range: {date_lo}  →  {date_hi}")

n_by_kind = traj_df["path_kind"].value_counts().to_dict()
caption = "  ·  ".join(f"Path {k}: {n_by_kind.get(k, 0)} trajectories" for k in ("A", "B"))
st.caption(caption)

st.markdown("---")
st.subheader("Per-trajectory distributions")

c1, c2 = st.columns(2)
c1.plotly_chart(charts.steps_per_traj_hist(traj_df), width="stretch",
                 key="landing_steps_hist")
c2.plotly_chart(charts.actions_per_traj_hist(traj_df), width="stretch",
                 key="landing_actions_hist")

c3, c4 = st.columns(2)
c3.plotly_chart(charts.per_traj_action_entropy_hist(traj_df), width="stretch",
                 key="landing_action_entropy_hist")
c4.plotly_chart(charts.unique_action_types_per_traj_hist(traj_df), width="stretch",
                 key="landing_unique_action_types_hist")

c5, c6 = st.columns(2)
c5.plotly_chart(charts.instruction_length_hist(traj_df), width="stretch",
                 key="landing_instruction_len_hist")
c6.plotly_chart(charts.success_pie(traj_df), width="stretch",
                 key="landing_success_pie")

st.markdown("---")
st.subheader("Actions across all steps")

c7, c8 = st.columns(2)
c7.plotly_chart(charts.action_type_bar(step_df), width="stretch",
                 key="landing_action_type_bar")
c8.plotly_chart(charts.thought_presence_bar(step_df), width="stretch",
                 key="landing_thought_presence_bar")

st.plotly_chart(charts.click_heatmap(step_df), width="stretch",
                 key="landing_click_heatmap")

st.markdown("---")
st.subheader("Sources")

c9, c10 = st.columns(2)
c9.plotly_chart(charts.domain_bar(traj_df, column="domain"), width="stretch",
                 key="landing_domain_bar")
c10.plotly_chart(charts.domain_bar(traj_df, column="website"), width="stretch",
                 key="landing_website_bar")

c11, c12 = st.columns(2)
c11.plotly_chart(charts.source_repo_bar(traj_df), width="stretch",
                 key="landing_source_repo_bar")
c12.plotly_chart(charts.task_type_bar(traj_df), width="stretch",
                 key="landing_task_type_bar")

st.plotly_chart(charts.recorded_timeline(traj_df), width="stretch",
                 key="landing_recorded_timeline")

st.markdown("---")
st.subheader("Cross-cuttings")

c13, c14 = st.columns(2)
c13.plotly_chart(charts.steps_vs_actions_scatter(traj_df), width="stretch",
                  key="landing_steps_vs_actions")
c14.plotly_chart(charts.instruction_vs_steps_scatter(traj_df), width="stretch",
                  key="landing_instr_vs_steps")
