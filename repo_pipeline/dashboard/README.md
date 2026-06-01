# Trajectory dashboard

A local Streamlit dashboard for exploring the trajectory corpus. **Read-only.** Plain
distributions — histograms, bar charts, scatter plots, heatmaps. No scoring, no thresholds,
no red/green coloring.

```bash
make dashboard               # opens http://localhost:8501
# or:
streamlit run repo_pipeline/dashboard/app.py
```

The dashboard reads `trajectory.json` + `metadata.json` + (when present)
`capture_meta.json` from every task directory under `trajectories/`. Any file that fails to
parse is skipped with a warning, never raised.

## Pages

- **Landing (`app.py`)** — cross-corpus overview. KPI strip, per-trajectory distributions
  (steps, actions, instruction length, action entropy, distinct action types), action-type
  mix across all steps, click-coordinate density heatmap, sources (domain / website /
  source_repo / task_type), recording timeline, and two scatter plots (steps vs actions,
  instruction length vs steps).
- **Path A · mind2web** — Path A trajectories only. Same shape but path-scoped, with a
  filter sidebar (outcome, domain).
- **Path B · QA repos** — Path B trajectories only. Adds source-repo and framework filters
  in the sidebar, and an Elements tab that lights up when `capture_meta.json` exists.
- **Inspector** — pick one trajectory; render each step's screenshot with the bbox/click
  overlay and the action + element + thought side-by-side.

Every page is laid out the same way:

1. **KPI strip** — counts (trajectories, total steps, total actions), success rate,
   unique repos / domains, % of trajectories that have `capture_meta.json`.
2. **Field-population strip** — a one-line ●/◐/○ summary of how many of the path's
   trajectories have each enriched field set. Quickly tells you which charts will be
   informative vs empty-state today.
3. **Tabs:** Overview · Actions · Reasoning · Sources · Elements.

## Tabs & what's in them

### Overview
- **Steps per trajectory** — histogram of `n_steps_actual`, with median + max in the title.
- **Actions per trajectory** — histogram of `n_actions_total` (some steps fire multiple
  actions).
- **Outcome** — `success` pie.
- **Instruction length** — chars histogram (clipped at 1200 in the chart only).
- **Steps vs total actions scatter** — colored by `path_kind`; hover for task_id /
  source_repo / domain.
- **Instruction length vs step count scatter** — does a longer instruction mean a longer
  trajectory? Colored by `path_kind`; hover shows the full instruction.
- **Navigation count per trajectory** *(Path B only)* — `n_navigations` histogram.

### Actions
- **Action type mix** — bar over `steps[].actions[].type`. Hover shows percentage share.
- **Per-trajectory action-type entropy** — Shannon entropy in nats of each trajectory's
  internal action mix. Tells you whether trajectories are mono-modal (entropy ≈ 0) or
  varied (entropy ≳ 1).
- **Distinct action types per trajectory** — how many of click/type/keypress/scroll/wait/
  drag does a typical trajectory use.
- **Action mix per trajectory** — horizontal stacked bar for the top-25 trajectories by
  total action count, broken down by action kind.
- **Click coordinate heatmap** — 2D density over the viewport (1280×720) of every recorded
  click / type / drag coordinate, y-axis flipped so the heatmap matches the screen.
- **Inter-action latency (`dt_ms`)** *(Path B only)* — histogram clipped at 30 s.
- **DOM change after action** *(Path B only)* — pie of `dom_changed`.

### Reasoning
- **Thought state** — bar with three buckets: empty / populated / placeholder. A
  "placeholder" thought is one that contains a known refusal substring like *"I need to see
  the screenshot"* or *"Could you provide"* (full list in
  `metrics.PLACEHOLDER_THOUGHT_SUBSTRINGS`).
- **Thought length** — histogram over non-empty thoughts (chars, clipped at 1500).

### Sources
- **Domain** / **Website** — bar chart, top 25.
- **Source repo** *(Path B only)* — horizontal bar, all repos visible.
- **Framework × language** *(Path B only)* — bar grouped by `framework_language`.
- **Task type** — bar over the Claude-classified `task_type` (auth / form-fill / search /
  CRUD / drag-drop / nav / other). Empty until stage 6 has classified.
- **Step count by domain** *(Path A)* / **by source_repo** *(Path B)* — box plot of
  `n_steps_actual` per category, top 15 categories.
- **Trajectories recorded per day** — bar over `metadata.timestamp` dates.
- **Top start URLs** — table.

### Elements *(populated only when `capture_meta.json` exists — Path B after the re-instrumented stage 6)*
- **Target element role** — bar over `capture_meta.steps[*].role` (aria role).
- **Target HTML tag** — bar over `capture_meta.steps[*].tag`.
- **Action × element-role heatmap** — counts of `(action_type, role)` pairs. Rare cells
  point at edge cases (typing into a button, scrolling on a textbox).
- **Bbox area / viewport area** — histogram of bbox area as a fraction of the 1280×720
  viewport. Median annotated in the title.
- **Click distance from bbox center (px)** — for clicks that landed inside their target
  bbox, how far from the center.
- **Click position vs target bbox** — pie of "inside" vs "outside" the bbox.

### Inspector (separate page)
Per-trajectory deep-dive. Pick a trajectory from the sidebar; for each step the page shows:
- the screenshot with a green bbox rectangle (if `bbox` present) and a red dot at `(x, y)`
  for any coordinate-bearing action;
- the action payload as JSON;
- the element record from `capture_meta.json` (role / name / bbox / selector / tag /
  visibility flags);
- the recorded `thought`.

A sidebar toggle hides the overlay if you want to see the raw screenshot.

## Files

```
repo_pipeline/dashboard/
├── README.md          # this file
├── app.py             # landing page
├── pages/
│   ├── 1_Path_A_mind2web.py
│   ├── 2_Path_B_repos.py
│   └── 3_Inspector.py
├── _state.py          # Streamlit-side helpers: cache, sidebar, KPI strip, filters
├── loader.py          # walks trajectories/ → (traj_df, step_df, warnings)
├── metrics.py         # shannon_entropy_nats + a few small constants used by loader.py
└── charts.py          # pure plotly figure builders
```

`loader.py`, `charts.py`, and `metrics.py` are streamlit-free and can be imported from a
REPL or notebook (`from repo_pipeline.dashboard.loader import load_corpus`).

## Refreshing

The corpus is read once per page load and cached for 60 s. Click **🔄 Refresh** in the
sidebar to force a re-scan — e.g. after generating new trajectories.
