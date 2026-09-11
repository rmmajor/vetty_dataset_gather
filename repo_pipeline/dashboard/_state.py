"""Streamlit-side helpers shared by every page (cached loader, sidebar, KPI strip, filters).

Kept separate from loader.py / charts.py / metrics.py so those stay free of `streamlit`
imports and remain unit-reasonable from a plain Python REPL.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_pipeline.dashboard.loader import (
    load_agent_calls,
    load_corpus,
    load_registry,
    summarize_warnings,
)

DEFAULT_TRAJ_DIR = REPO_ROOT / "trajectories"
DEFAULT_DB_PATH = REPO_ROOT / "pipeline_data" / "registry.db"


@st.cache_data(ttl=60, show_spinner="Scanning trajectories/ …")
def cached_load_corpus(traj_root_str: str):
    traj_df, step_df, warnings = load_corpus(Path(traj_root_str))
    return traj_df, step_df, warnings


@st.cache_data(ttl=60, show_spinner="Reading registry.db …")
def cached_load_registry(db_path_str: str):
    return load_registry(Path(db_path_str))


@st.cache_data(ttl=60, show_spinner="Reading agent_calls …")
def cached_load_agent_calls(db_path_str: str):
    return load_agent_calls(Path(db_path_str))


def sidebar_global_controls(warnings: list[str]) -> None:
    with st.sidebar:
        st.caption(f"📁 `{DEFAULT_TRAJ_DIR}`")
        if st.button("🔄 Refresh", width="stretch"):
            cached_load_corpus.clear()
            st.rerun()
        if warnings:
            with st.expander(f"⚠️  Loader warnings ({len(warnings)})"):
                st.code(summarize_warnings(warnings), language="text")


def render_kpi_strip(traj_df: pd.DataFrame, step_df: pd.DataFrame) -> None:
    if traj_df.empty:
        st.info("No trajectories loaded.")
        return
    cols = st.columns(6)
    cols[0].metric("Trajectories", len(traj_df))
    success_rate = float(traj_df["success"].fillna(False).mean()) * 100
    cols[1].metric("Success rate", f"{success_rate:.0f}%")
    cols[2].metric("Total steps", int(traj_df["n_steps_actual"].sum()))
    cols[3].metric("Total actions", int(traj_df["n_actions_total"].sum()))
    n_repos = int(traj_df["source_repo"].fillna("").replace("", pd.NA).dropna().nunique())
    n_domains = int(traj_df["domain"].fillna("").replace("", pd.NA).dropna().nunique())
    cols[4].metric("Unique repos / domains", f"{n_repos} / {n_domains}")
    has_cap = float(traj_df["has_capture_meta"].fillna(False).mean()) * 100
    cols[5].metric("% w/ capture_meta", f"{has_cap:.0f}%")


def page_header(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)


def apply_path_filters(
    traj_df: pd.DataFrame, step_df: pd.DataFrame, *, path_kind: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = traj_df[traj_df["path_kind"] == path_kind]
    if t.empty:
        return t, step_df.iloc[0:0]

    with st.sidebar:
        st.markdown(f"### Path {path_kind} filters")
        success_pick = st.radio(
            "Outcome", ["all", "success only", "fail only"], key=f"success_{path_kind}", horizontal=True
        )
        if success_pick == "success only":
            t = t[t["success"]]
        elif success_pick == "fail only":
            t = t[~t["success"]]

        domains = sorted(d for d in t["domain"].fillna("").unique() if d)
        if domains:
            picked = st.multiselect("Domain", domains, default=domains, key=f"domain_{path_kind}")
            t = t[t["domain"].isin(picked) | (t["domain"].fillna("") == "")]

        if path_kind == "B":
            repos = sorted(r for r in t["source_repo"].fillna("").unique() if r)
            if repos:
                picked = st.multiselect("Source repo", repos, default=repos, key=f"repo_{path_kind}")
                t = t[t["source_repo"].isin(picked)]
            frameworks = sorted(f for f in t["framework"].fillna("").unique() if f)
            if frameworks:
                picked = st.multiselect("Framework", frameworks, default=frameworks, key=f"fw_{path_kind}")
                t = t[t["framework"].isin(picked)]

    s = step_df[step_df["task_id"].isin(t["task_id"])].reset_index(drop=True)
    return t.reset_index(drop=True), s


def populated_count(traj_df: pd.DataFrame) -> dict[str, int]:
    """Diagnostic: how many trajectories have each enriched field populated.

    Surfaced at the top of each page as a small grey strip so the user can tell at a glance
    which charts will be lit up vs which are empty-state today.
    """
    if traj_df.empty:
        return {}
    counts: dict[str, int] = {}
    for col in (
        "framework",
        "task_type",
        "dedup_hash",
        "domain",
        "website",
        "source_repo",
        "n_navigations",
        "repo_stars",
    ):
        if col not in traj_df.columns:
            counts[col] = 0
            continue
        s = traj_df[col]
        if pd.api.types.is_numeric_dtype(s.dropna()):
            counts[col] = int((s.fillna(0) != 0).sum())
        else:
            counts[col] = int((s.fillna("").astype(str).str.strip() != "").sum())
    counts["has_capture_meta"] = int(traj_df["has_capture_meta"].fillna(False).sum())
    return counts


def render_populated_strip(counts: dict[str, int], total: int) -> None:
    if not counts or total == 0:
        return
    parts = []
    for col, n in counts.items():
        marker = "●" if n == total else ("◐" if n > 0 else "○")
        parts.append(f"{marker} `{col}` {n}/{total}")
    st.caption("Field population:  " + "   ".join(parts))
