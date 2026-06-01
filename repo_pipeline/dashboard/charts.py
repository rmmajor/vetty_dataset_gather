"""Pure plotly figure builders. No Streamlit calls in this module.

Every chart here is a plain distribution / count / scatter — no thresholds, no green/red
target bands, no scoring. Each function takes a (possibly filtered) DataFrame and returns
a `plotly.graph_objects.Figure`. Empty / missing-data inputs are caught up front and routed
to a small grey "no data" placeholder figure instead of raising.
"""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


MUTED = "#888888"


def is_populated(series: pd.Series, *, allow_zero: bool = False) -> bool:
    if series is None or len(series) == 0:
        return False
    non_na = series.dropna()
    if non_na.empty:
        return False
    if not pd.api.types.is_numeric_dtype(non_na):
        non_empty = non_na[non_na.astype(str).str.strip() != ""]
        return not non_empty.empty
    if allow_zero:
        return True
    return bool((non_na != 0).any())


def empty_state_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font={"size": 14, "color": MUTED})
    fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False},
                      margin={"l": 10, "r": 10, "t": 10, "b": 10}, height=220)
    return fig


# ---------- Per-trajectory distributions ---------------------------------------------------

def steps_per_traj_hist(traj_df: pd.DataFrame) -> go.Figure:
    if traj_df.empty:
        return empty_state_fig("no trajectories")
    fig = px.histogram(traj_df, x="n_steps_actual", nbins=30,
                       title=f"Steps per trajectory (median {traj_df['n_steps_actual'].median():.0f}, "
                             f"max {int(traj_df['n_steps_actual'].max())})")
    fig.update_layout(xaxis_title="steps", yaxis_title="trajectories",
                      bargap=0.05, height=320)
    return fig


def actions_per_traj_hist(traj_df: pd.DataFrame) -> go.Figure:
    if traj_df.empty:
        return empty_state_fig("no trajectories")
    fig = px.histogram(traj_df, x="n_actions_total", nbins=30,
                       title=f"Actions per trajectory (median {traj_df['n_actions_total'].median():.0f}, "
                             f"max {int(traj_df['n_actions_total'].max())})")
    fig.update_layout(xaxis_title="actions", yaxis_title="trajectories",
                      bargap=0.05, height=320)
    return fig


def unique_action_types_per_traj_hist(traj_df: pd.DataFrame) -> go.Figure:
    if traj_df.empty:
        return empty_state_fig("no trajectories")
    action_cols = [c for c in ("n_click", "n_type", "n_keypress",
                                "n_scroll", "n_wait", "n_drag") if c in traj_df.columns]
    if not action_cols:
        return empty_state_fig("no action breakdown columns")
    unique_count = (traj_df[action_cols] > 0).sum(axis=1)
    fig = px.histogram(unique_count, nbins=int(unique_count.max() or 1) + 1,
                        title="Distinct action types used per trajectory")
    fig.update_layout(xaxis_title="distinct action types in trajectory",
                       yaxis_title="trajectories",
                       bargap=0.05, height=320, showlegend=False)
    return fig


def per_traj_action_entropy_hist(traj_df: pd.DataFrame) -> go.Figure:
    if traj_df.empty:
        return empty_state_fig("no trajectories")
    entropies = traj_df["action_entropy_nats"].dropna()
    if entropies.empty:
        return empty_state_fig("no action entropy data")
    fig = px.histogram(entropies, nbins=20,
                        title=f"Per-trajectory action-type entropy (median {entropies.median():.2f} nats)")
    fig.update_layout(xaxis_title="Shannon entropy (nats)",
                       yaxis_title="trajectories",
                       bargap=0.05, height=320, showlegend=False)
    return fig


def instruction_length_hist(traj_df: pd.DataFrame) -> go.Figure:
    if traj_df.empty:
        return empty_state_fig("no trajectories")
    series = traj_df["instruction_len_chars"]
    fig = px.histogram(series.clip(upper=1200), nbins=40,
                        title=f"Instruction length (chars). median {series.median():.0f}, "
                              f"max {int(series.max())}")
    fig.update_layout(xaxis_title="characters", yaxis_title="trajectories",
                       bargap=0.05, height=320, showlegend=False)
    return fig


# ---------- Outcome / categorical ----------------------------------------------------------

def success_pie(traj_df: pd.DataFrame) -> go.Figure:
    if traj_df.empty:
        return empty_state_fig("no trajectories")
    counts = traj_df["success"].fillna(False).map({True: "success", False: "fail"}).value_counts()
    fig = px.pie(values=counts.values, names=counts.index, title="Outcome",
                 color=counts.index,
                 color_discrete_map={"success": "#2ca02c", "fail": "#d62728"})
    fig.update_layout(height=320)
    return fig


def action_type_bar(step_df: pd.DataFrame) -> go.Figure:
    if step_df.empty:
        return empty_state_fig("no steps")
    counts = step_df["action_type"].value_counts()
    total = int(counts.sum())
    pct = (counts / total * 100).round(1)
    fig = px.bar(x=counts.index, y=counts.values,
                 title=f"Action type mix across {total} steps",
                 hover_data={"share": pct.values})
    fig.update_layout(xaxis_title="action type", yaxis_title="step count", height=320)
    return fig


def action_mix_per_traj_stack(traj_df: pd.DataFrame, *, top_n: int = 25) -> go.Figure:
    if traj_df.empty:
        return empty_state_fig("no trajectories")
    cols = [c for c in ("n_click", "n_type", "n_keypress", "n_scroll", "n_wait", "n_drag")
            if c in traj_df.columns]
    if not cols:
        return empty_state_fig("no action breakdown columns")
    sub = traj_df[["task_id"] + cols].copy()
    sub["total"] = sub[cols].sum(axis=1)
    sub = sub.sort_values("total", ascending=False).head(top_n)
    long = sub.melt(id_vars=["task_id"], value_vars=cols,
                    var_name="action_kind", value_name="count")
    long["action_kind"] = long["action_kind"].str.replace("n_", "", regex=False)
    fig = px.bar(long, x="count", y="task_id", color="action_kind", orientation="h",
                 title=f"Action mix per trajectory (top {top_n} by total actions)")
    fig.update_layout(yaxis={"autorange": "reversed"}, height=max(320, 22 * len(sub) + 80),
                       barmode="stack", xaxis_title="actions", yaxis_title=None,
                       legend_title="action kind")
    return fig


def steps_vs_actions_scatter(traj_df: pd.DataFrame) -> go.Figure:
    if traj_df.empty:
        return empty_state_fig("no trajectories")
    fig = px.scatter(traj_df, x="n_steps_actual", y="n_actions_total",
                      color="path_kind", hover_data=["task_id", "source_repo", "domain"],
                      title="Steps vs total actions (one dot per trajectory)")
    fig.update_layout(xaxis_title="steps", yaxis_title="total actions", height=380)
    return fig


def instruction_vs_steps_scatter(traj_df: pd.DataFrame) -> go.Figure:
    if traj_df.empty:
        return empty_state_fig("no trajectories")
    fig = px.scatter(traj_df, x="instruction_len_chars", y="n_steps_actual",
                      color="path_kind", hover_data=["task_id", "instruction"],
                      title="Instruction length vs step count")
    fig.update_layout(xaxis_title="instruction chars", yaxis_title="steps", height=380)
    return fig


def per_domain_steps_box(traj_df: pd.DataFrame, *, by: str = "domain") -> go.Figure:
    if traj_df.empty or by not in traj_df.columns or not is_populated(traj_df[by]):
        return empty_state_fig(f"{by} unset")
    sub = traj_df[[by, "n_steps_actual"]].copy()
    sub[by] = sub[by].fillna("").replace("", pd.NA)
    sub = sub.dropna(subset=[by])
    if sub.empty:
        return empty_state_fig(f"{by} unset on every trajectory")
    counts = sub[by].value_counts()
    top_categories = counts.head(15).index
    sub = sub[sub[by].isin(top_categories)]
    fig = px.box(sub, x=by, y="n_steps_actual",
                  title=f"Step count by {by} (top {len(top_categories)} categories)")
    fig.update_layout(xaxis_title=by, yaxis_title="steps", height=360,
                       xaxis_tickangle=-30)
    return fig


def domain_bar(traj_df: pd.DataFrame, column: str = "domain", top_n: int = 25) -> go.Figure:
    if traj_df.empty or column not in traj_df.columns:
        return empty_state_fig(f"no {column} data")
    series = traj_df[column].fillna("").replace("", pd.NA).dropna()
    if series.empty:
        return empty_state_fig(f"{column} unset on every trajectory")
    counts = series.value_counts().head(top_n)
    fig = px.bar(x=counts.index, y=counts.values,
                  title=f"{column.title()} distribution ({len(counts)} of {series.nunique()} unique shown)")
    fig.update_layout(xaxis_title=column, yaxis_title="trajectories", height=320,
                       xaxis_tickangle=-30)
    return fig


def source_repo_bar(traj_df: pd.DataFrame) -> go.Figure:
    if traj_df.empty:
        return empty_state_fig("no trajectories")
    repos = traj_df["source_repo"].fillna("").replace("", pd.NA).dropna()
    if repos.empty:
        return empty_state_fig("no source_repo data (Path A only)")
    counts = repos.value_counts()
    fig = px.bar(x=counts.values, y=counts.index, orientation="h",
                  title=f"Trajectories per source repo ({len(counts)} repos)")
    fig.update_layout(xaxis_title="trajectories", yaxis_title=None,
                       yaxis={"autorange": "reversed"},
                       height=max(240, 20 * len(counts) + 80))
    return fig


def task_type_bar(traj_df: pd.DataFrame) -> go.Figure:
    if traj_df.empty or not is_populated(traj_df["task_type"]):
        return empty_state_fig(
            "task_type unset on every trajectory.\n"
            "Stage 6 classifies via Claude; re-run to populate."
        )
    counts = traj_df["task_type"].fillna("").replace("", pd.NA).dropna().value_counts()
    fig = px.bar(x=counts.index, y=counts.values,
                  title=f"Task type distribution ({len(counts)} categories)")
    fig.update_layout(xaxis_title="task_type", yaxis_title="trajectories", height=320)
    return fig


def framework_bar(traj_df: pd.DataFrame) -> go.Figure:
    if traj_df.empty or not is_populated(traj_df["framework"]):
        return empty_state_fig("framework unset (Path B only, set after stage 6 instrumentation)")
    sub = traj_df.copy()
    sub["framework"] = sub["framework"].fillna("").replace("", pd.NA)
    sub = sub.dropna(subset=["framework"])
    if sub.empty:
        return empty_state_fig("framework unset")
    counts = sub.groupby(["framework", "framework_language"]).size().reset_index(name="trajectories")
    fig = px.bar(counts, x="framework", y="trajectories", color="framework_language",
                  title="Framework × language")
    fig.update_layout(xaxis_title="framework", yaxis_title="trajectories", height=320,
                       legend_title="language")
    return fig


def top_start_urls_table(traj_df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if traj_df.empty:
        return pd.DataFrame(columns=["start_url", "trajectories"])
    counts = (traj_df["start_url"].fillna("").replace("", pd.NA).dropna()
                                  .value_counts().head(top_n)
                                  .rename_axis("start_url").reset_index(name="trajectories"))
    return counts


def recorded_timeline(traj_df: pd.DataFrame) -> go.Figure:
    if traj_df.empty or "recorded_date" not in traj_df.columns:
        return empty_state_fig("no trajectories")
    dates = traj_df["recorded_date"].dropna()
    if dates.empty:
        return empty_state_fig("no recorded_date in metadata.json")
    by_day = dates.dt.tz_convert(None).dt.date.value_counts().sort_index()
    fig = px.bar(x=by_day.index, y=by_day.values, title="Trajectories recorded per day")
    fig.update_layout(xaxis_title="date", yaxis_title="trajectories", height=300)
    return fig


# ---------- Reasoning / thought distributions ----------------------------------------------

def thought_length_hist(step_df: pd.DataFrame) -> go.Figure:
    if step_df.empty:
        return empty_state_fig("no steps")
    nonempty = step_df[step_df["has_thought"]]
    if nonempty.empty:
        return empty_state_fig("no steps with non-empty thought")
    fig = px.histogram(nonempty["thought_len"].clip(upper=1500), nbins=40,
                        title=f"Thought length over {len(nonempty)} non-empty steps "
                              f"(median {nonempty['thought_len'].median():.0f} chars)")
    fig.update_layout(xaxis_title="characters", yaxis_title="steps",
                       bargap=0.05, height=320, showlegend=False)
    return fig


def thought_presence_bar(step_df: pd.DataFrame) -> go.Figure:
    if step_df.empty:
        return empty_state_fig("no steps")
    rows = [
        ("empty",       int((~step_df["has_thought"]).sum())),
        ("populated",   int((step_df["has_thought"] & ~step_df["is_placeholder_thought"]).sum())),
        ("placeholder", int(step_df["is_placeholder_thought"].sum())),
    ]
    df = pd.DataFrame(rows, columns=["thought_state", "steps"])
    fig = px.bar(df, x="thought_state", y="steps",
                  title=f"Thought state across {len(step_df)} steps",
                  color="thought_state",
                  color_discrete_map={"empty": "#cccccc",
                                       "populated": "#1f77b4",
                                       "placeholder": "#d62728"})
    fig.update_layout(xaxis_title=None, yaxis_title="steps", height=320, showlegend=False)
    return fig


# ---------- Click coordinates --------------------------------------------------------------

def click_heatmap(step_df: pd.DataFrame, *, viewport_w: int = 1280,
                  viewport_h: int = 720) -> go.Figure:
    coord_steps = step_df.dropna(subset=["x", "y"])
    if coord_steps.empty:
        return empty_state_fig("no coordinate actions recorded")
    fig = px.density_heatmap(coord_steps, x="x", y="y", nbinsx=32, nbinsy=18,
                              title=f"Click / type / drag coordinate density "
                                    f"({len(coord_steps)} action points)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="x (px)", yaxis_title="y (px)", height=420,
                       xaxis={"range": [0, viewport_w]},
                       yaxis={"range": [viewport_h, 0]})
    return fig


# ---------- Element / capture_meta panels --------------------------------------------------

def element_role_bar(step_df: pd.DataFrame) -> go.Figure:
    if "role" not in step_df.columns or not is_populated(step_df["role"]):
        return empty_state_fig(
            "no element role data — `capture_meta.json` missing.\n"
            "Re-run stage 6 with the enriched instrumentation."
        )
    roles = step_df["role"].fillna("").replace("", pd.NA).dropna()
    counts = roles.value_counts().head(20)
    fig = px.bar(x=counts.index, y=counts.values,
                  title=f"Target element role mix (top {len(counts)} of {roles.nunique()})")
    fig.update_layout(xaxis_title="role", yaxis_title="step count", height=320,
                       xaxis_tickangle=-30)
    return fig


def element_tag_bar(step_df: pd.DataFrame) -> go.Figure:
    if "tag" not in step_df.columns or not is_populated(step_df["tag"]):
        return empty_state_fig("tag unset — re-run stage 6 with capture_meta.")
    tags = step_df["tag"].fillna("").replace("", pd.NA).dropna()
    counts = tags.value_counts().head(20)
    fig = px.bar(x=counts.index, y=counts.values,
                  title=f"Target HTML tag (top {len(counts)} of {tags.nunique()})")
    fig.update_layout(xaxis_title="tag", yaxis_title="step count", height=320,
                       xaxis_tickangle=-30)
    return fig


def action_role_heatmap(step_df: pd.DataFrame) -> go.Figure:
    if "role" not in step_df.columns or not is_populated(step_df["role"]):
        return empty_state_fig(
            "no element role data — `capture_meta.json` missing.\n"
            "Re-run stage 6 with the enriched instrumentation."
        )
    df = step_df.copy()
    df["role"] = df["role"].fillna("(none)")
    pivot = df.pivot_table(index="role", columns="action_type",
                            values="step", aggfunc="count", fill_value=0)
    fig = px.imshow(pivot, text_auto=True, aspect="auto",
                     title="Action × element-role heatmap (counts)",
                     color_continuous_scale="Blues")
    fig.update_layout(height=360, margin={"l": 60, "r": 10, "t": 40, "b": 60})
    return fig


def bbox_area_hist(step_df: pd.DataFrame) -> go.Figure:
    if "bbox_area_ratio" not in step_df.columns or not is_populated(step_df["bbox_area_ratio"], allow_zero=True):
        return empty_state_fig(
            "no bbox data — `capture_meta.json` missing.\n"
            "Re-run stage 6 to populate inline `bbox`."
        )
    ratios = step_df["bbox_area_ratio"].dropna()
    if ratios.empty:
        return empty_state_fig("no bboxes")
    fig = px.histogram(ratios.clip(upper=1.0), nbins=40,
                        title=f"Bbox area / viewport area  (median {ratios.median():.4f})")
    fig.update_layout(xaxis_title="area ratio", yaxis_title="step count",
                       bargap=0.05, height=320, showlegend=False)
    return fig


def click_to_center_hist(step_df: pd.DataFrame) -> go.Figure:
    if "click_to_center_px" not in step_df.columns or not is_populated(step_df["click_to_center_px"], allow_zero=True):
        return empty_state_fig(
            "no bbox data — `capture_meta.json` missing."
        )
    distances = step_df["click_to_center_px"].dropna()
    if distances.empty:
        return empty_state_fig("no bbox-contained clicks")
    fig = px.histogram(distances, nbins=30,
                        title=f"Click distance from bbox center (px)  median {distances.median():.0f}")
    fig.update_layout(xaxis_title="distance (px)", yaxis_title="step count",
                       bargap=0.05, height=320, showlegend=False)
    return fig


def bbox_contained_pie(step_df: pd.DataFrame) -> go.Figure:
    if "bbox_contains_click" not in step_df.columns or not is_populated(step_df["bbox_contains_click"], allow_zero=True):
        return empty_state_fig(
            "no bbox data — `capture_meta.json` missing."
        )
    series = step_df["bbox_contains_click"].dropna()
    if series.empty:
        return empty_state_fig("no click/type steps with bbox")
    counts = series.map({True: "inside bbox", False: "outside bbox"}).value_counts()
    fig = px.pie(values=counts.values, names=counts.index,
                  title="Click position vs target bbox",
                  color=counts.index,
                  color_discrete_map={"inside bbox": "#1f77b4",
                                       "outside bbox": "#cccccc"})
    fig.update_layout(height=320)
    return fig


def dt_ms_hist(step_df: pd.DataFrame) -> go.Figure:
    if "dt_ms" not in step_df.columns or not is_populated(step_df["dt_ms"]):
        return empty_state_fig("dt_ms unset — re-run stage 6 with capture_meta.")
    series = step_df["dt_ms"].dropna()
    if series.empty:
        return empty_state_fig("no dt_ms data")
    fig = px.histogram(series.clip(upper=30000), nbins=40,
                        title=f"Inter-action latency dt_ms  (clipped at 30s; median "
                              f"{series.median():.0f})")
    fig.update_layout(xaxis_title="ms", yaxis_title="step count",
                       bargap=0.05, height=320, showlegend=False)
    return fig


def navigations_hist(traj_df: pd.DataFrame) -> go.Figure:
    if "n_navigations" not in traj_df.columns or not is_populated(traj_df["n_navigations"]):
        return empty_state_fig(
            "n_navigations unset — re-run stage 6 with capture_meta enabled."
        )
    fig = px.histogram(traj_df["n_navigations"], nbins=20,
                        title="Navigation count per trajectory")
    fig.update_layout(xaxis_title="n_navigations", yaxis_title="trajectories",
                       bargap=0.05, height=320, showlegend=False)
    return fig


def dom_change_pie(step_df: pd.DataFrame) -> go.Figure:
    if "dom_changed" not in step_df.columns or not is_populated(step_df["dom_changed"], allow_zero=True):
        return empty_state_fig("dom_changed unset — re-run stage 6.")
    series = step_df["dom_changed"].dropna()
    if series.empty:
        return empty_state_fig("no dom_changed data")
    counts = series.map({True: "changed", False: "unchanged"}).value_counts()
    fig = px.pie(values=counts.values, names=counts.index,
                  title=f"DOM change after action across {len(series)} steps",
                  color=counts.index,
                  color_discrete_map={"changed": "#1f77b4",
                                       "unchanged": "#cccccc"})
    fig.update_layout(height=320)
    return fig
