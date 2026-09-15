# SFT corpus data gaps — ranked

Measured 2026-08-30 over the canonical corpus `trajectories/` (1,225 trajectory dirs), replicating
`cua/sft/legacy_train_qwen.py convert` gating exactly (excluded_reason, replay gate, verifier `success`, stuck-loop
gate). **Trainable subset: 609 trajectories, 11,318 steps → ~11,291 per-step SFT examples.** Skips:
506 unverified (306 failed / 105 partial / 52 completed-but-unverified / 43 unreported), 45 excluded,
38 replay-gated, 27 stuck-loop. Composition of the trainable set: 573 public-site captures + 33 repo
captures + 3 Mind2Web recordings; 212 (34.8%) behind auth (`pre_session` 155, `discovered` 52,
`login` 5); 190 distinct sites, top-10 site share 15.6%; steps/trajectory median 14, p90 42 (auth:
median 24, p90 47; login-free: 10 / 33). Cross-referenced against the same-night eval triple
`eval_runs/{qwen-base,sft-v1,grpo-v2}-judge2_*` (Online-Mind2Web 50-task sample: base 14.7% scored
success, sft-v1 9.4%, grpo-v2 11.8%; see `docs/decisions/grpo-v2-first-live-run.md`), ScreenSpot
`eval_archive/screenspot_grpo_v2/`, and 143 RL rollouts in `rl_runs/grpo-v2/`.

| # | Gap | One-line evidence | Confidence |
|---|-----|-------------------|------------|
| 1 | Blocked/error-page states absent | 0.80% of trainable steps ever mention an error state; 14/50 eval episodes hit walls, where sft-v1 drops `<think>` and emits unparseable `navigate()` | confirmed |
| 2 | Info-seeking multi-constraint search on mainstream sites underrepresented | 2/40 eval sites appear in corpus; constrained-search tasks = 13/609; sft-v1 0 successes above easy | confirmed symptom |
| 3 | No stuck→recover / backtrack supervision | success-only gating removes all 306 failed + 105 partial runs; 97/143 RL episodes end by loop-guard | confirmed symptom |
| 4 | Stop-signal dilution: meandering steps are 41.8% of supervision, done() 5.2% | done() rate fell 22%→16% base→sft; step_cap episodes 4→11 | confirmed symptom |
| 5 | History window truncates 56.5% of steps | auth trajectories median 24 steps vs HISTORY_WINDOW=10; amplifies 3 and 4 | hypothesis |
| 6 | Icon/small-target clicks scarce | 13.3% of bbox'd clicks ≤40x40; ScreenSpot icon 80.3%→75.4% base→sft | hypothesis |
| 7 | Dead/starved action verbs | double_click 0, horizontal scroll 0, move 0.49%, right-click 0.37% of clicks | dataset-only |
| 8 | Supervision-style residue | 74/609 trajectories still `teacher_raw`; 86/609 procedure-flavored instructions | dataset-only |

## 1. Blocked/error-page states are absent — and the trained escape hatch is unexecutable

**Evidence (dataset).** Only 91 of 11,318 trainable steps (0.80%, in 37 trajectories) mention any
blocked/error state in the thought: captcha/bot-wall 58 steps in 17 trajectories, access-denied/403
11 steps in 7, 404 12 steps in 10, generic error 10 steps in 4, geo-block 0. Only 9 of those are
first steps. The corpus is verified-success-only on healthy pages by construction. Related:
`navigate("url")` appears in the trainable corpus 151 times across 68 trajectories (mostly
auth-lane URL jumps) and is listed in `SYSTEM_PROMPT`, but the eval parser's `ACTION_VERBS`
(`cua/eval/student/grammar.py`) does not include it — every emitted `navigate(...)` line is a
parse failure, and the RL driver refuses it too (`navigate_blocked` in 15/143 rollouts).

**Evidence (runs).** 14/50 eval episodes are `site_blocked` in all three same-night runs. On those
pages sft-v1 degenerates: bare prose with no opening `<think>`, a stray `</think>`, then
`action: navigate("…")` — e.g. `eval_runs/sft-v1-judge2_*/episodes/task_20a460a8fe1971b84411c5b1e6ac4186`
(stubhub bot wall; `raw_last_response` ends `…</think>\n\naction: navigate("https://www.stubhub.com")`),
`task_0059adc6b12a3822305deb68929b2de8` (walmart careers — navigated to ziprecruiter.com, a different
site), `task_2532fd402d3c741b79894e6ff2269f53` (thumbtack geo-block), `task_0e42c3a73f2aece1f854e0ba55b7c8b0`
(google login loop). 13/50 sft episodes ended `parse_failure_abort`.

**Impact.** 28% of eval episodes land on states the model has ~never seen; its response is the one
verb the harness cannot execute, so blocked-adjacent pages convert into parse aborts instead of
recovery or a clean report. This is the largest single bucket of non-success outcomes.

**Fix.** (a) Capture ~60–100 deliberate wall demonstrations (bot checks, geo blocks, login loops,
404s, "temporarily restricted") showing the intended behavior — retry-later, alternate path on the
SAME site, or `handoff(reason=captcha)` / `done()` with an honest report. (b) Resolve the navigate
contradiction one way or the other: either the eval/RL drivers execute `navigate` or the corpus
stops teaching it as the recovery move (relabel/trim those 151 steps). (c) Stop filtering every
wall out of the corpus: an episode that meets a wall and handles it well is trainable signal.

**Confidence:** confirmed-by-runs.

## 2. Info-seeking / multi-constraint search on mainstream public sites is underrepresented

**Evidence (dataset).** Regex task-type mix over 609 trainable instructions: create/submit 217
(35.6%), search/lookup 139 (22.8%), configure 36, purchase/checkout 24, download/export 20,
delete/manage 11, auth/register 11, other 151. Travel/commerce constrained search (dates, prices,
round-trip, "cheapest", filters) matches only 13 instructions (2.1%). Path A (Mind2Web-style GPT
recordings) contributes just 3 trainable trajectories; 33 trainable trajectories run on localhost
dev apps. Site overlap with the eval: 2 of 40 distinct eval-episode sites (google.com, gov.uk)
appear anywhere in the trainable corpus.

**Evidence (runs).** sft-v1 by level: easy 3/11, medium 0/9, hard 0/12 — zero successes above easy
(`eval_runs/sft-v1-judge2_*/results.json`). Medium/hard Online-Mind2Web tasks are precisely
multi-constraint info-seeking on large consumer sites (`reference_length` up to 36).

**Impact.** The corpus is SaaS/action-UI-heavy by design (the deployment goal), but the measured
benchmark is search-heavy on mainstream sites; the trained distribution gives the model near-zero
practice at constraint-tracking search journeys (enter query → filter → sort → compare → pick),
which is where all medium/hard failures sit.

**Fix.** 150–200 trajectories of goal-phrased search tasks on mainstream public sites
(travel/retail/reference: flights with date+price constraints, product filter+sort+compare,
multi-hop lookups), captured through the existing public-site lane with search-goal proposer
templates. Keep instructions terse goals; the constraint set carries the difficulty.

**Confidence:** symptom confirmed by runs; the transfer estimate is dataset-side reasoning.

## 3. No stuck→recover supervision: the corpus never shows a failed attempt being abandoned

**Evidence (dataset).** Gating removes exactly the recovery material: 306 `failed` and 105
`partial` runs are excluded as unverified, 27 more for stuck loops, and the converter's optional
waste-step filter drops retry clicks as targets. Within the trainable set, recovery language
("didn't work / try a different approach / go back") appears in 2.8% of steps (317, spread over 186
trajectories — almost all single soft retries, not visible strategy changes after a failed attempt).

**Evidence (runs).** 97/143 RL rollouts (67.8%) ended by loop-guard — the student repeating one
action ≥5 times (`rl_runs/grpo-v2/*/episode.json` `terminal_reason`); eval `step_cap` episodes rose
4 (base) → 11 (sft-v1).

**Impact.** When a click doesn't take (overlay, hydration lag, wrong element), the policy has no
trained alternative to re-emitting the same action; loop-guard/step-cap terminations dominate RL
and eval failure modes.

**Fix.** (a) Mine the 105 `partial` + 52 completed-but-unverified runs for hindsight-relabelable
prefixes that contain a visible dead end followed by a working path. (b) Capture ~100 deliberate
obstacle tasks where the teacher demonstrably tries, fails, states the failure in the thought, and
switches strategy. (c) Keep single-retry steps (don't default `--exclude-waste-steps`) so "click
didn't register → adjust" remains observable.

**Confidence:** symptom confirmed by runs; dataset numbers direct.

## 4. Stop-signal dilution: wandering supervises 41.8% of steps, stopping 5.2%

**Evidence (dataset).** 139/609 trainable trajectories (22.8%) are verifier-graded `meandering`,
and because they are long they contribute 4,735/11,318 step-examples (41.8%). `efficiency_ratio`
≥2.0 in 90/599 tagged trajectories (p90 = 2.25). `done()` supervision is 590 examples — 5.2% of
the corpus, exactly one per trajectory, always after the goal is already confirmed; 19 trainable
trajectories don't end in `done` at all.

**Evidence (runs).** done() rate 22% (base) → 16% (sft-v1); step_cap 11/50; RL efficiency grades:
64 `meandering` vs 12 `efficient` (143 rollouts). Steps/reference on sft successes 1.33x.

**Impact.** Fine-tuning made the model stop *less* than the base model: the per-step loss is
dominated by mid-wander continuation targets, and near-goal "stop now" boundary states are rare
relative to "keep acting" states.

**Fix.** Data-only options, in order of leverage: trim meandering trajectories to end at the step
the goal is actually reached (the verifier notes + `efficiency_ratio` locate the tail) rather than
excluding them; oversample the 590 done steps; add ~50 captured near-goal boundary cases where the
tempting next action exists but done() is correct. `--exclude-meandering` is the blunt fallback
(costs 41.8% of examples).

**Confidence:** symptom confirmed; attribution to the meandering share is the leading hypothesis
(done-dilution is structural either way).

## 5. History-window truncation hits 56.5% of steps — exactly on the long auth trajectories

**Evidence (dataset).** With `HISTORY_WINDOW = 10` (`cua/sft/legacy_train_qwen.py:166`), 6,393/11,318 trainable
steps (56.5%) train with a truncated "Previous actions" list. The truncation concentrates on the
corpus's core: auth-lane trajectories run median 24 steps / p90 47 vs 10 / 33 login-free; 65
trainable trajectories exceed 40 steps.

**Impact.** Beyond 10 actions the model cannot see what it already did — consistent with repeating
actions (gap 3) and failing to recognize the goal was reached (gap 4). Amplifier, not root cause.

**Fix.** Converter/runtime change rather than capture: carry a one-line summary of the omitted
prefix (the converter already counts it: "... N earlier actions"), or widen the window; verify at
parity in `student_eval` (the window is a parity key). Data-side mitigation: prefer goal-scoped
shorter tasks when capturing.

**Confidence:** dataset-only hypothesis; mechanism consistent with measured loop/stop symptoms.

## 6. Icon and small-target clicks are scarce

**Evidence (dataset).** Path B steps record the clicked element's bbox: of 6,253 trainable clicks
with a bbox, 830 (13.3%) hit icon-sized targets ≤40x40 px; median target area 6,478 px² (~80x80),
p10 976 px². Hover (`move`) appears 66 times total; top-80px chrome clicks are 11.1% of all clicks.
Element roles skew to large controls: button 2,215, input 1,494, link 1,247.

**Evidence (runs).** ScreenSpot desktop-300: base 89.0% → sft 85.3%; icon 80.3% → 75.4%, text
94.9% → 92.1% (`eval_archive/screenspot_grpo_v2/{qwen-base,sft-v1-lora}-served.json`).

**Impact.** The −4.9pt icon regression is the whole grounding drop; the corpus gives the model far
more practice on labeled buttons/inputs than on small unlabeled glyphs.

**Fix.** Oversample the existing 830 icon-click steps; add a grounding-only mix-in (icon-dense
click supervision from a separate source) or capture icon-heavy tasks (toolbars, kebab menus,
star/favorite toggles). Cheap to try since the examples already exist.

**Confidence:** regression confirmed; causal link to corpus composition is a hypothesis (format
shift could contribute).

## 7. Dead or starved action verbs

**Evidence (dataset).** Over 13,341 trainable actions: `double_click` **0**, `move` 66 (0.49%),
`drag` 116 (0.87%), right-click 30/8,176 clicks (0.37%), `wait` 152, `confirm` 27 and `handoff` 12
(guarded lane: 35 trajectories). Scroll (814): down 647 / up 167 / horizontal **0**. Keypress
(1,551, 280 distinct combos): Enter 474 plus 9 as `ENTER` (casing inconsistency), Escape 164,
Control+a 120, Tab 40, arrow-key presses ~25 including 10–30-key `ArrowDown+…` chains. Date/calendar
interactions are mentioned in only 54 trajectories; file upload in 15.

**Impact.** The grammar advertises verbs with zero or near-zero supervision, so the model cannot
have learned them; modest for Online-Mind2Web but real for date pickers, sliders, hover menus,
carousels — and for the guarded-mode deployment verbs.

**Fix.** Targeted capture of ~30–50 trajectories exercising drag-to-reorder, double-click rename,
hover-reveal menus, horizontal carousels, calendar pickers; normalize `ENTER`→`Enter` and collapse
mega ArrowDown chains at convert time.

**Confidence:** dataset-only.

## 8. Supervision-style residue

**Evidence (dataset).** 74/609 trainable trajectories (12.2%) still carry
`thought_source: teacher_raw` (stage-7 polish never reached them); 103 of their 2,719 steps contain
meta-references ("…then take a screenshot to see the dropdown state"). 86/609 instructions are
procedure-flavored (mostly multi-clause tours: "Open X, filter Y, sort Z, then…"). 12 instructions
and 1.1% of type-texts contain literal emails — burner/test addresses that are legitimate page
content (invite flows), not operator PII; masked credentials are 10.3% of type-texts.

**Impact.** Minor style drift from the intent-level supervision standard; the meta thoughts
supervise narrating the capture process rather than the user goal.

**Fix.** Run the stage-7 polish pass over the 74 `teacher_raw` trajectories (one text-only call
each, resumable); tighten proposer templates that emit 3+-clause tour instructions.

**Confidence:** dataset-only.

## Not gaps — checked and healthy

- **done() placement:** 0 mid-trajectory `done` in 11,318 steps; 590/609 trajectories (96.9%) end
  in a real recorded `done` step (no synthesis needed).
- **Secrets/PII:** no operator-identity or credential literals in the trainable set (the 11
  `operator_pii_leak` recordings are already excluded); credential typing is `<secret:*>`-masked.
- **Site concentration:** 190 sites / 210 split groups, top-10 share 15.6% — no single-site
  domination.
- **Format transfer:** canonical action form 18.8% (base) → 100% (sft), per-step parse failures
  20.6% → 8.0% on healthy pages — the grammar itself trained fine (the failures are gap 1).
- **Coordinates:** 0/8,176 clicks outside the 1280x720 viewport; click geometry is spread
  (center-third 38.7%, top-80px 11.1%, edges ~5% each).
- **Thought quality:** 0 thoughts narrate coordinates; median 148 chars, only 27 empty.
- **Type-text diversity:** 1,164 distinct / 1,686 type actions; 29.5% contain digits, 44.1% spaces,
  6.0% date-like, 2.9% non-ASCII.
