# SFT v2 — the corpus-gap round (2026-08-31)

The response to `docs/sft-corpus-gaps.md` and the judge2 triple (`grpo-v2-first-live-run.md`):
under the consistent criteria-first judge, sft-v1's task success was indistinguishable from the
base model (14.7% vs 9.4% scored at n≈34, CIs overlapping), so this round changes the corpus, the
harness bug that miscounted a trained verb, and the measurement's resolving power — together.

## What was wrong with the measurement itself

1. **`navigate` was trained but unparseable.** The training `SYSTEM_PROMPT` teaches
   `navigate("url")` (598 occurrences in the corpus), the browser driver implements the `navigate`
   /act kind, but the eval parser's `ACTION_VERBS` lacked it — every emitted `navigate(...)` line
   scored as a parse failure. `handoff`/`confirm` had the same bug. Fixed: all three verbs parse;
   navigate executes; handoff/confirm end the episode as the scored outcome `handoff`. The verb
   set is now part of `harness_fingerprint`, so pre-fix runs correctly compare as a different
   harness. RL's exact-netloc navigate allowlist also loosened to registrable-domain.
2. **n≈34 cannot resolve any plausible delta** (±8pp CI). The v2 protocol runs a pinned 230-task
   file (`eval_archive/online_mind2web/tasks_pinned_v2.jsonl`): the legacy 50 (kept verbatim for
   continuity — reported as a subset, annotated non-rankable against the judge2 triple because the
   fingerprint changed) + 180 level-stratified tasks, sites hard-blocked from this IP in all three
   judge2 runs excluded from the new 180 only. Target ≥150 scored episodes per side.

## Corpus interventions (v2)

- **Shared trainability gate**: convert and `corpus_audit` now call one function
  (`corpus_audit.trainability`) — the hand-copied mirror that could drift is gone.
- **Hard health gates** (`corpus_health.json`, enforced at convert; `--skip-health-gate` for
  experiments): min trainable count, done/stop coverage ≥95%, `teacher_raw` = 0, meandering
  example share ≤25%, per-group share ≤15%, action-share floors (click/type/done). `modal_train`
  runs `corpus_audit` before converting; the remote lane ships `corpus_health.json`.
- **Meandering subsample instead of exclusion** (`--meandering-mode subsample`): sft-v1's clean
  cut dropped all 139 meandering trajectories (~4,300 emitted targets) outright — and done() still
  fell 22%→16%, so the stop-signal problem was structural, not meandering-driven. v2 trains their
  head/tail + every 4th mid-wander step (drop-as-target, keep-in-history — the waste-step
  mechanism), raising the done-target share instead of discarding the interaction data.
- **`HISTORY_WINDOW` 10→20**: 56.5% of steps trained with truncated history exactly on the
  long auth trajectories (median 24 steps). Shared constant — SFT, RL observation and eval all
  move together; recorded in the fingerprint.
- **Keypress hygiene at convert**: `ENTER`→`Enter` canonical casing; steps whose keypress repeats
  one key >6 times (the 10-30-key ArrowDown chains) drop as targets, stay in history.
- **Audited hindsight relabel** (`repo_pipeline/relabel_recoverable.py`): capture-time
  reclassification stays forbidden; this separate pass proposes a goal-shaped instruction for what
  a partial/completed-unverified run actually accomplished (goal-shape rejects navigation-verb
  phrasings — the June failure), then re-judges with the same `verify_trajectory` against the NEW
  instruction; only a fresh success verdict writes, with `metadata.relabeled_from="hindsight_v2"`
  + the original instruction preserved, and a terminal done() appended. Rejected attempts restore
  files byte-for-byte.
- **Wall demos** (`repo_pipeline/wall_supervision.py`): blocked failed/partial runs trimmed to
  the pre-wall prefix + the wall-recognition step, plus a synthesized terminal
  `handoff(reason=captcha)` (interactive checks) or honest `done()` (hard blocks). Capped
  (~150 examples) so stop-on-adversity stays a minority signal; originals kept as
  `trajectory_prewall.json`. This is the format-on-broken-pages supervision behind 13/50
  parse-abort episodes.
- **Stage-7 polish + instruction style**: `polish_pass` clears the 74 trainable `teacher_raw`
  dirs (registry-less site_* lanes stage 7 could never reach); `instruction_style_pass` rewrites
  ~80 procedure-flavored instructions into goals (route dropped, every concrete parameter kept),
  reversible via `original_instruction`.
- **Search-goal capture lane** (`public_sites_search.json`, `task_style: "search"` proposer
  block): the benchmark is constrained-search-heavy while the corpus is action-UI-heavy by design;
  this lane adds goal-phrased multi-constraint search journeys on 21 mainstream sites, all probed
  reachable, none of them among Online-Mind2Web's 145 task sites (leakage rule; note the
  pre-existing corpus overlap: ikea.com, google.com, gov.uk were captured before this rule).
  Travel/retail sites carry date pickers and range sliders, so the proposer's affordance quotas
  double as the gap-7 interaction-verb tranche.

## Measured salvage (2026-08-31 night run)

| pass | input | kept | notes |
| --- | --- | --- | --- |
| orphan judging | 22 never-judged | 3 verified success | 19 confirmed failures |
| hindsight relabel | 157 candidates | **46 relabeled** | 90 rejected by the fresh verdict (restored byte-for-byte), 21 accomplished nothing proposable |
| wall demos | 90 blocked runs | **34 applied** | 150-example cap, domain round-robin; 36 skipped (wall past step 8) |

The 29% relabel acceptance rate is the audit working as intended: the judge rejects runs whose
"accomplishment" does not hold up against its own criteria checklist, which is precisely the check
the banned June version lacked.

Supervision passes on top: 137 trajectories polished into student voice, 81 procedure-flavored
instructions rewritten as goals (4 rejected by the shape check), 50 relabeled trajectories
thought-realigned against their new goal.

Corpus state after the passes, before the search lane finishes (mid-capture snapshot):

| | sft-v1 (corpus v1) | v2 in progress |
| --- | --- | --- |
| trainable trajectories | 456 | **689** |
| SFT examples | 6,093 | **8,928** |
| meandering supervision | excluded outright | 23.8% of examples (23.1% of trajectories) |
| stop supervision | 590 done steps | 645 done + 33 wall demos (handoff/done) |

`corpus_audit`: 689 trainable, 0 failing.

Two defects the counts alone would have hidden, both found by reading the written artifacts:

1. **Wall demos inherited a stale efficiency grade.** 14 of 33 carried `meandering` from the
   verdict on their UNTRIMMED original — a 4-step demo (the median) cannot meander. The tag made
   convert subsample their tiny target set and inflated the corpus meandering share. `apply_plan`
   now re-grades a trimmed demo `acceptable` (keeping `efficiency_before_trim`), and the 33 on
   disk were repaired.
2. **Meandering subsampling was calibrated by feel.** At `MEANDER_MID_KEEP_EVERY=4` meandering
   examples were 26.2% of supervision against a 23.1% trajectory share — still over-weighted, and
   over the gate. The fix was the KNOB, not the threshold: at 6 the example share is 23.8%, i.e.
   meandering trajectories supervise in proportion to how many of them there are. Raising the gate
   to make the violation disappear would have thrown away the only signal that caught this.

A spot-check caught one defect the pass could not see: a relabeled trajectory's THOUGHTS still
pursue the dropped part of the original goal ("…so I can immediately create Dashboard" under an
instruction that now asks only for Auth and Billing). Fixed at the source — a relabel now sets
`thought_source: teacher_raw`, which routes the trajectory through the existing polish pass to be
rewritten against the NEW goal. The `teacher_raw = 0` health gate makes skipping that step
impossible: convert refuses the dataset until the realignment has run.

## Harness bug found while capturing (search lane)

The search-style proposer block lengthened the prompt enough to push the JSON reply past
`max_turns=1`, so every search-lane site silently fell back to its 2 curated tasks — the lane would
have produced ~42 trajectories instead of ~150, reported only as a warning line. `max_turns` is now
2; verified live (12go: 10 well-shaped constrained-search tasks, 6 hard/medium).

Related, and worth remembering: `probe_capturable` measures the HTTP surface from the HOST, not the
sandboxed browser. rome2rio and sbb-rail probed OK yet are Cloudflare-Turnstile-walled (or render
blank) inside the container. Sandbox-side reachability is only knowable by capturing; the cost is
bounded (2-3 teacher runs per dead site) and the failures become wall-demo material.

## Three defects the training lane had been carrying (found launching sft-v2)

None of these would have surfaced without an actual training run; sft-v1 predates all of them.

1. **A broken recording reached the trainable set.** `site_openfoodfacts/task_gluten-free-breakfast-cereals`
   carried an empty `type` action at step 25 — supervision that teaches typing an empty string. The
   pre-train `corpus_audit` step added to `modal_train` caught it BEFORE the GPU started, which is
   the whole point of running the audit remotely rather than trusting the local one. Quarantined
   with the reason recorded; corpus went 739 → 738 trainable.
2. **`config.yaml` was unrunnable.** `packing: true` was added after sft-v1 trained, but ms-swift
   refuses packing without a flash-attention implementation and the Modal image ships none, so
   `swift sft` died at argument parsing. sft-v1's recorded args show `packing=False` — the config
   had drifted away from the only recipe ever actually run. Set back to `false`.
3. **The batch shape was starving the GPU.** `per_device_train_batch_size: 1` with
   `gradient_accumulation_steps: 8` measured **20.4 s/step on an H100** — 2.55 s/sample against
   sft-v1's 1.66 s/sample on a *slower* A100, i.e. eight sequential batch-1 micro-steps per
   optimizer step. Restoring sft-v1's `2 x 4` (identical effective batch of 8) gave **9.7 s/step**,
   a 2.1x speedup that turned an 18-hour run into 8.5 hours.

## The Vast lane had never completed a run (three more defects)

4. **`repo_pipeline/` was never uploaded**, though `train_qwen.py` imports it — the box died at
   import *after* the rental started. Now shipped, and preflight refuses to rent without it.
5. **`pip install -r requirements.txt` clobbered the image's torch.** That file is the whole
   pipeline (playwright, pandas, datasets…); installing it over `pytorch/pytorch` left
   `transformers` half-imported (`NameError: name 'nn' is not defined`). Replaced with only the
   training stack Modal proved, on a torch 2.6 image.
6. **The safety mechanism destroyed the work.** On SIGINT/SIGTERM/SIGHUP the supervisor destroyed
   the rented box, to protect against paying for an orphan. But the remote job is already detached
   (`setsid nohup`), so it would have survived — the dead-man's switch was throwing away hours of
   GPU time to prevent a cost that would not have occurred. `--keep-on-signal` now leaves the box
   running and exits via `os._exit` (a plain `sys.exit` unwinds through the `finally:`, which
   destroys the instance anyway — that subtlety silently defeats the fix if missed).

## What was actually killing the runs: SIGTERM sweeps, not sleep

Repeated unexplained deaths of local processes cost most of a day. Ruled out with evidence: no
jetsam/OOM entries, 58% memory free, no crash reports, `claude` host process alive throughout, and
`uptime` is NOT evidence about sleep (it does not reset on wake — a mistake made here). The machine
*did* sleep overnight (89 DarkWake events), which was a plausible but wrong explanation.

A zero-cost canary heartbeat settled it: at 19:31:50 several Bash background tasks of different ages
(62, 58 and 47 minutes) received **signal 15 (SIGTERM)** in the same instant, together with their
parent shells, while the Claude host process and all **Monitor** tasks survived. Same-instant deaths
across different ages rule out per-task timeouts; SIGTERM rules out sleep, which suspends rather
than signals.

Practical consequences, both now in use: anything that must outlive a sweep belongs in a **Monitor**
(caffeinate included — it was swept twice as a Bash task), and any long GPU run must be detached
remotely with `--keep-on-signal` so a swept supervisor cannot take the work with it.

Operational note: Modal's `modal run` creates an app tied to the client. Something outside these
sessions terminated every local background task twice in ~70 minutes (03:11:10 and ~04:20, across
two unrelated Claude Code sessions, with no OOM, sleep, crash report, or resource pressure — filed
as a Claude Code bug). Training therefore runs with `modal run --detach`, which survives the client
dying; local jobs (capture, eval) do not and must simply be restarted.

## Deviations from the gaps doc's prescriptions

- Gap 3(c) suggested keeping single-retry clicks (no `--exclude-waste-steps`); the lanes keep the
  waste filter — same-spot retries teach re-clicking into no change, and the recovery signal now
  comes from relabeled strategy-change runs instead.
- Gap 6 (icon oversampling) deferred: no duplication flag shipped; revisit if ScreenSpot icon
  regresses further.

## Config note

sft-v2 trains with current `config.yaml` (packing on, val split on) — sft-v1's actual args had
`packing=False`, no val. The corpus is the headline intervention; the delta vs sft-v1 is corpus +
those training-plumbing fixes, and the quoted comparison is vs the base model on the same night.

## Results

(to be filled after the sft-v2 train + 2-way eval)
