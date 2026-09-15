"""Yield page — stage-5 capture yield, sliced by framework, with the failure logs.

Reads pipeline_data/registry.db. "Reached stage 5" = a repo that either cleared the trial
(validated → captured → completed) or was rejected at trial. The headline number is how many of
those actually captured. The no_capture table shows the persisted tail of each failed test run so
"why pass_rate=0" (app-needed vs install-failed vs instrumentation-broke) is diagnosable.
"""

import json

import streamlit as st

from cua.pipeline.dashboard import charts
from cua.pipeline.dashboard._state import (
    DEFAULT_DB_PATH,
    cached_load_registry,
    page_header,
)

_PASSED = {"validated", "captured", "completed"}

st.set_page_config(page_title="Yield", page_icon="🎯", layout="wide")
page_header(
    "Yield — stage-5 capture, by framework, with failure logs",
    "Reads pipeline_data/registry.db. 'Reached' = cleared the trial or was rejected at it; "
    "'passed' = produced a usable capture. The no_capture table carries each run's test-output tail.",
)

repos_df = cached_load_registry(str(DEFAULT_DB_PATH))

with st.sidebar:
    st.caption(f"📁 `{DEFAULT_DB_PATH}`")
    if st.button("🔄 Refresh", width="stretch"):
        cached_load_registry.clear()
        st.rerun()

if repos_df.empty:
    st.warning("No registry found at pipeline_data/registry.db. Run the pipeline first.")
    st.stop()

reached = repos_df[repos_df["status"].isin(_PASSED | {"rejected_trial"})].copy()
if reached.empty:
    st.info("No repos have reached stage 5 yet. Run `--stage 5` to populate trial outcomes.")
    st.stop()

reached["passed"] = reached["status"].isin(_PASSED)
reached["no_capture"] = reached["failure_category"] == "no_capture"
n_reached = len(reached)
n_passed = int(reached["passed"].sum())
n_no_capture = int(reached["no_capture"].sum())

cols = st.columns(4)
cols[0].metric("Reached stage 5", f"{n_reached:,}")
cols[1].metric("Captured (passed)", f"{n_passed:,}")
cols[2].metric("Yield", f"{n_passed / n_reached:.0%}")
cols[3].metric("no_capture rejections", f"{n_no_capture:,}")

st.markdown("---")
st.plotly_chart(charts.capture_yield_by_framework(repos_df), width="stretch", key="yield_fw")

lang = reached["da_framework_language"].fillna("?") if "da_framework_language" in reached else "?"
fw = reached["da_test_framework"].fillna("?") if "da_test_framework" in reached else "?"
reached["framework"] = lang.astype(str) + " / " + fw.astype(str)
summary = (
    reached.groupby("framework")
    .agg(
        reached=("status", "size"),
        passed=("passed", "sum"),
        no_capture=("no_capture", "sum"),
    )
    .reset_index()
)
summary["yield_%"] = (100 * summary["passed"] / summary["reached"]).round(0)
summary = summary.sort_values("reached", ascending=False)
st.markdown("#### Per-framework yield")
st.dataframe(summary, width="stretch", hide_index=True)

st.markdown("#### Why `no_capture` — persisted test-output tails")
nc = reached[reached["no_capture"]].copy()
if nc.empty:
    st.caption("No no_capture rejections recorded yet.")
else:

    def _notes(blob):
        try:
            data = json.loads(blob) if isinstance(blob, str) else (blob or {})
            return data.get("notes", "") if isinstance(data, dict) else ""
        except (json.JSONDecodeError, TypeError):
            return ""

    nc["log"] = nc["trial_results"].map(_notes)
    choices = nc["full_name"].tolist()
    picked = st.selectbox("Repo", choices, key="yield_nc_pick")
    row = nc[nc["full_name"] == picked].iloc[0]
    st.caption(row.get("filter_reason", ""))
    st.code(row["log"] or "(no log captured — predates this telemetry)", language="text")
