# Task: Monitor the running trajectory harvest — catch bugs, errors, and low-quality trajectories

## What you're monitoring

This repo (`/Users/rmajor/unik/maga/dataset_gather`) runs a funnel that captures computer-use
trajectories for fine-tuning Qwen3.5-9B. A capture run is **already in progress** (started by the
user before this session). Your job is to **watch it, diagnose anything that goes wrong, and keep
the trajectory quality high** — not to redesign anything. Read `CLAUDE.md` first for architecture.

The run is one of:
- `--stage 5-7` (boot-gate the analyzed backlog → capture validated → finalize), or
- `--until-limit 100` (full drain loop 3→7).

Find it: `pgrep -fl "cua.pipeline.orchestrator"`. Its log is the newest file in
`pipeline_data/run_logs/`. Use a **Monitor** on that log (one monitor, stop it before re-arming —
see Monitor hygiene below). Useful filter:
`grep -E "boot OK|complete —|verified trajectories from|stage 6.*(proposed|task ')|REJECT|RUN DEGRADED|Stage . failed|Traceback|budget|usage limit"`

## Baseline state at handoff (so you can tell "moved" from "stuck")

- **Corpus: 42 verified trainable trajectories, 275 SFT examples**, median 6 / max 20 steps, 36/42
  with terminal `done()`. Confirm with: `make audit-corpus` and
  `.venv-pipeline/bin/python -m cua.sft convert --input trajectories/ --output /tmp/c --val-split 0 | grep -E "Trajectories trained|Steps/|Domains"`
- Funnel: ~171 `analyzed`, ~1705 `scored`, 7 `completed`. Branch `feat/agent-capture` → PR #23.
- Run the pipeline via `.venv-pipeline/bin/python` (NOT system python). Claude calls auth via the
  logged-in `claude` CLI (OAuth) — **never set ANTHROPIC_API_KEY**.

## What "a trajectory landed" means

Trajectory dirs live under `trajectories/repo_<owner>_<repo>/task_*/` and `trajectories/site_*/task_*/`.
A trajectory **trains** only if `convert` counts it: `metadata.capture_source=="agent"` AND
`trajectory.json success==true` AND no `metadata.excluded_reason`. `success` is the VERIFIER's
verdict (set in stage-6 post-processing), not the teacher's claim. A repo shows
`[stage 6][repo] N verified trajectories from M captured` when it finishes.

## The failure modes to watch for (all seen this session — know them cold)

1. **A stage silently dies and starves everything downstream.** A bare `NameError` once killed
   every stage-4 pass while the run looked like "0 pending." If a stage's pass processes 0 or the
   log shows `Stage N failed` / `RUN DEGRADED` / a `Traceback`, investigate immediately — read the
   traceback, fix the import/bug, commit, restart the run. AST-sweep sibling stages for the same
   class of bug before declaring it fixed.

2. **Budget must NEVER reject anything (hard user rule).** Budget is informational only. If you
   see a repo rejected with `budget_exceeded`, or a trajectory dropped for budget, that's a
   regression — recover via `--requeue-infra` and fix the cause. Per-call caps were deliberately
   raised huge (teacher $25, analyze $5); they exist only as runaway backstops behind
   `max_turns`/action-cap. Do not re-introduce any budget→reject path.

3. **One task's failure discarding a whole repo's captures.** Fixed this session (per-task
   isolation in `stage6_generate.py`), but verify it holds: a repo that has good trajectories on
   disk must never end up `rejected_generate` with `trajectory_count=0`. If you see a repo rejected
   while valid trajectories sit in its dir, that's the bug returning — recover by re-running
   post-processing on its dir (see `_postprocess_agent_capture`) or `--requeue-infra`.

4. **Browser crashes losing a task.** Seen on SSM (`browser target crashed` → task failed, no
   recovery). If frequent, the driver should restart its context and retry the task once
   (`driver_session.py` / `browser_driver.js`). Currently a known gap, not yet fixed.

5. **Low-quality / insufficient trajectories.** Spot-check newly-verified ones. Red flags:
   - Stuck loops (≥3 identical actions) — `make audit-corpus` catches these mechanically; it must
     stay at **0 failing**. If it reports failures, investigate before they reach training.
   - Thoughts that narrate tooling ("clicking element 14", "using snapshot") instead of user
     intent — that's the most toxic supervision failure. Read 3-4 thought fields of each new repo.
   - Trajectories that "succeed" but the final screenshot doesn't show the goal met (verifier
     miss) — read the verifier `reasoning` in `capture_meta.json` against the final screenshot.
   - Health-check passed an app whose subpages need a dead backend → tasks fail on those subpages
     (seen: GomaGoma CRUD, movies-app). Correctly dropped, just wasteful; note the repo, don't act.
   - A goal-COMPLETED long trajectory dropped solely for `meandering` (>2× expected actions). This
     is a PARKED decision — see `docs/decisions/open-questions.md`. Do NOT change the threshold
     without the user; just note occurrences.

6. **Idle containers.** No sandbox should outlive its run. `docker ps --filter name=sandbox` —
   anything older than the running process / from a dead run is an orphan; `docker rm -f` it.
   Containers self-expire (auto_remove + sleep timeout) so this should be rare.

## Monitor hygiene (the user called this out twice — do not repeat the mistake)

- **One monitor per watched thing.** Before restarting a watched process, `TaskStop` its monitor
  FIRST, then re-arm on the new target. When a watched task completes and you've read its result,
  stop its monitor in the same breath. Never let monitors on dead logs pile up.

## How to report

- Surface only: **verified-count changes**, **failures that form a pattern** (≥2-3 occurrences),
  and **anything actionable**. Do NOT narrate every routine `task '...' completed` line — the user
  found that noisy. Lead each report with the conclusion.
- When the run finishes or hits the usage limit: give a corpus summary (count, examples, step
  distribution, domain mix, action mix) via the convert corpus report, and list any bugs found +
  fixed (with commit hashes) and any still open.

## Standing constraints

- Commit + push fixes to `feat/agent-capture` (PR #23); minimal comments, no nested imports,
  generic-over-special-case (user's CLAUDE.md rules).
- A saturated 5h Claude window makes calls wait minutes — if the run looks "hung," time
  `claude -p pong` before debugging code (it's usually rate-limit backoff, not a bug).
- Open improvement items live in `docs/decisions/open-questions.md` and the running suggestion
  list (meandering threshold, browser-crash recovery, health-check subpage probing, no-progress
  action cap, compound-task tier, phrasing diversity). Don't implement these unprompted; the job
  this session is monitoring + bug-fixing, not feature work.

## Definition of done

The run completes (or the user stops it) with: corpus count reported, `make audit-corpus` at 0
failing, every bug encountered either fixed-and-committed or clearly logged as open, and no orphan
containers left behind.
