"""Rejections page — why repos leave the funnel, and the data to tune the gates.

Reads pipeline_data/registry.db (not the trajectory corpus). Score-distribution charts split
passed vs rejected with the gate threshold drawn in, so the operator can see exactly how many
repos a threshold change would recover. Existing rejections only have data here after running
`python -m cua.pipeline.backfill_scores`; new ones populate automatically.
"""

import streamlit as st

from cua.pipeline.dashboard import charts
from cua.pipeline.dashboard._state import (
    DEFAULT_DB_PATH,
    cached_load_agent_calls,
    cached_load_registry,
    page_header,
)
from cua.pipeline.schemas import NON_QUALITY_CATEGORIES
from cua.settings import PIPELINE

_INFRA = {c.value for c in NON_QUALITY_CATEGORIES}
# Mirror the stage gates so the threshold lines track the actual (env-tunable) config.
_DEEP_GATE = PIPELINE.stage4_deep_score_threshold
_GATE_APP = PIPELINE.stage3_gate_app_min
_GATE_RUN = PIPELINE.stage3_gate_runnability_min
_GATE_INTERACT = PIPELINE.stage3_gate_interaction_min
_REJECTED_PREFIX = "rejected_"
_PASSED_S3 = {"scored", "analyzed", "validated", "completed"}
_PASSED_S4 = {"analyzed", "validated", "completed"}


st.set_page_config(page_title="Rejections", page_icon="🚫", layout="wide")
page_header(
    "Rejections — funnel losses & gate tuning",
    "Reads pipeline_data/registry.db. Score charts split passed vs rejected with the gate line "
    "drawn in. Run `python -m cua.pipeline.backfill_scores` to populate historical rejections.",
)

repos_df = cached_load_registry(str(DEFAULT_DB_PATH))
calls_df = cached_load_agent_calls(str(DEFAULT_DB_PATH))

with st.sidebar:
    st.caption(f"📁 `{DEFAULT_DB_PATH}`")
    if st.button("🔄 Refresh", width="stretch"):
        cached_load_registry.clear()
        cached_load_agent_calls.clear()
        st.rerun()

if repos_df.empty:
    st.warning("No registry found at pipeline_data/registry.db. Run the pipeline first.")
    st.stop()

# Pass/reject label per scored stage, so the score charts can colour each repo.
status = repos_df["status"].astype(str)
repos_df["s3_outcome"] = "other"
repos_df.loc[status.isin(_PASSED_S3), "s3_outcome"] = "passed"
repos_df.loc[status == "rejected_score", "s3_outcome"] = "rejected"
repos_df["s4_outcome"] = "other"
repos_df.loc[status.isin(_PASSED_S4), "s4_outcome"] = "passed"
repos_df.loc[status == "rejected_analyze", "s4_outcome"] = "rejected"

total = len(repos_df)
completed = int((status == "completed").sum())
rejected = int(status.str.startswith(_REJECTED_PREFIX).sum())
infra_rejected = int(repos_df["failure_category"].isin(_INFRA).sum())
quality_rejected = rejected - infra_rejected

cols = st.columns(5)
cols[0].metric("Repos", f"{total:,}")
cols[1].metric("Completed", f"{completed:,}")
cols[2].metric("Rejected (total)", f"{rejected:,}")
cols[3].metric("Quality rejections", f"{quality_rejected:,}")
cols[4].metric("Infra/flaky rejections", f"{infra_rejected:,}")

st.markdown("---")
tab_funnel, tab_s3, tab_s4, tab_reasons, tab_turns = st.tabs(
    [
        "📉 Funnel",
        "3️⃣ Stage 3 scores",
        "4️⃣ Stage 4 scores",
        "🏷️ Reasons & errors",
        "⏱️ Turns & cost",
    ]
)

with tab_funnel:
    st.plotly_chart(charts.status_bar(repos_df), width="stretch", key="rej_status_bar")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.failure_category_bar(repos_df), width="stretch", key="rej_cat_bar")
    c2.plotly_chart(charts.retry_hist(repos_df), width="stretch", key="rej_retry_hist")

with tab_s3:
    st.caption(
        f"Stage-3 gate rejects if app < {_GATE_APP} OR runnability < {_GATE_RUN} OR "
        f"interaction_richness < {_GATE_INTERACT}. Bars left of each dashed line are rejected."
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        charts.score_dim_bar(
            repos_df, "sc_application_vs_library", "s3_outcome", "application_vs_library", threshold=_GATE_APP
        ),
        width="stretch",
        key="rej_s3_app",
    )
    c2.plotly_chart(
        charts.score_dim_bar(repos_df, "sc_runnability", "s3_outcome", "runnability", threshold=_GATE_RUN),
        width="stretch",
        key="rej_s3_run",
    )
    c3, c4 = st.columns(2)
    c3.plotly_chart(
        charts.score_dim_bar(
            repos_df, "sc_interaction_richness", "s3_outcome", "interaction_richness", threshold=_GATE_INTERACT
        ),
        width="stretch",
        key="rej_s3_interact",
    )
    c4.plotly_chart(
        charts.score_dim_bar(repos_df, "shallow_total", "s3_outcome", "shallow total (/40)"),
        width="stretch",
        key="rej_s3_total",
    )

with tab_s4:
    st.caption(
        f"Stage-4 gate rejects if the deep score (0–10) is below {_DEEP_GATE}. "
        "Bars left of the dashed line are what the gate rejects."
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        charts.score_dim_bar(repos_df, "deep_score", "s4_outcome", "deep_score", threshold=_DEEP_GATE),
        width="stretch",
        key="rej_s4_deep",
    )
    c2.plotly_chart(
        charts.score_dim_bar(repos_df, "da_test_case_count", "s4_outcome", "test_case_count"),
        width="stretch",
        key="rej_s4_tests",
    )

with tab_reasons:
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.rejection_reason_bar(repos_df, "rejected_score"), width="stretch", key="rej_reason_score")
    c2.plotly_chart(
        charts.rejection_reason_bar(repos_df, "rejected_analyze"), width="stretch", key="rej_reason_analyze"
    )
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.rejection_reason_bar(repos_df, "rejected_trial"), width="stretch", key="rej_reason_trial")
    c4.plotly_chart(charts.rejection_reason_bar(repos_df, "rejected_filter"), width="stretch", key="rej_reason_filter")

with tab_turns:
    if calls_df.empty:
        st.info(
            "No agent_calls recorded yet. Run an agent stage (3, 4, or 6) to populate turn / "
            "cost telemetry. Stage 4 (max_turns=25) is the one to watch."
        )
    else:
        st.plotly_chart(charts.agent_turns_box(calls_df), width="stretch", key="rej_turns_box")
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.agent_cost_bar(calls_df), width="stretch", key="rej_cost_bar")
        c2.plotly_chart(charts.agent_subtype_bar(calls_df), width="stretch", key="rej_subtype_bar")
