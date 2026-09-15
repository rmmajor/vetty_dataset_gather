"""Tokens page — total token usage and average tokens per stage.

Reads agent_calls from pipeline_data/registry.db. Usage splits into output, uncached input,
cache-read and cache-write. The cache columns populate only for calls recorded after the
cache-token telemetry fix (registry.record_agent_call); older rows count input+output only, so
historical totals understate true input (most of which was served from cache).
"""

import streamlit as st

from cua.pipeline.dashboard import charts
from cua.pipeline.dashboard._state import (
    DEFAULT_DB_PATH,
    cached_load_agent_calls,
    page_header,
)

st.set_page_config(page_title="Tokens", page_icon="🔢", layout="wide")
page_header(
    "Tokens — total usage & average per stage",
    "Reads agent_calls from pipeline_data/registry.db. Cache-read/write populate for calls recorded "
    "after the cache-token telemetry fix; older rows count input+output only.",
)

calls_df = cached_load_agent_calls(str(DEFAULT_DB_PATH))

with st.sidebar:
    st.caption(f"📁 `{DEFAULT_DB_PATH}`")
    if st.button("🔄 Refresh", width="stretch"):
        cached_load_agent_calls.clear()
        st.rerun()

if calls_df.empty:
    st.info("No agent_calls recorded yet. Run an agent stage (3, 4, 6) to populate token telemetry.")
    st.stop()

df = charts.token_frame(calls_df)

total_tokens = int(df["total_tokens"].sum())
total_input = int(df[["input_tokens", "cache_read_tokens", "cache_creation_tokens"]].sum().sum())
total_output = int(df["output_tokens"].sum())
total_cost = float(df["cost_usd"].fillna(0).sum()) if "cost_usd" in df.columns else 0.0

cols = st.columns(4)
cols[0].metric("Total tokens", f"{total_tokens:,}")
cols[1].metric("Total input (incl. cache)", f"{total_input:,}")
cols[2].metric("Total output", f"{total_output:,}")
cols[3].metric("Total cost (USD)", f"${total_cost:,.2f}")

st.markdown("---")
c1, c2 = st.columns(2)
c1.plotly_chart(charts.tokens_by_stage_bar(calls_df), width="stretch", key="tok_total_stage")
c2.plotly_chart(charts.avg_tokens_by_stage_bar(calls_df), width="stretch", key="tok_avg_stage")

st.markdown("#### Average tokens per call, by stage & purpose")
summary = (
    df.groupby(["stage", "purpose"])
    .agg(
        calls=("stage", "size"),
        avg_input=("input_tokens", "mean"),
        avg_cache_read=("cache_read_tokens", "mean"),
        avg_cache_write=("cache_creation_tokens", "mean"),
        avg_output=("output_tokens", "mean"),
        avg_total=("total_tokens", "mean"),
        avg_cost=("cost_usd", "mean"),
    )
    .reset_index()
    .sort_values(["stage", "purpose"])
)
summary = summary.round(
    {"avg_input": 0, "avg_cache_read": 0, "avg_cache_write": 0, "avg_output": 0, "avg_total": 0, "avg_cost": 4}
)
st.dataframe(summary, width="stretch", hide_index=True)
