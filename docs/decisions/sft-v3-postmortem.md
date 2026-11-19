# SFT v3 — postmortem (2026-09-04)

What the round predicted, what it measured, what caused the gap, and what is proven against what is
merely plausible. Every number below is recomputed from files on disk and each section names them.
Companions: `docs/decisions/sft-v3-plan.md` (why this round), `docs/decisions/sft-v3-corpus-round.md`
(the design and the pre-registered bands), `docs/decisions/sft-v3-results.md` (the first pass at the
results), `docs/decisions/grpo-v3-results.md` (why RL was dropped).

## 0. Sources

| what | where |
| --- | --- |
| sft-v3 episodes, v3 harness | `.../cua-observation-v3/results/econ-sft-v3` |
| sft-v2 re-recorded, v3 harness | `.../cua-observation-v3/results/matched-sft-v2` |
| base Qwen3.5-9B, v3 harness | `.../cua-observation-v3/results/matched-base` |
| sft-v2 / sft-v1 / base, original harness | `.../valiant-kindling-rivest/results/econ-sft2`, `econ-sft1`, `econ-base` |
| training corpus | `trajectories/` (a symlink to the canonical corpus) — 1,398 task dirs, 784 trainable, 774 outcome-relabeled with `trajectory_pre_outcome.json` originals |
| the dataset actually trained | `training_data/sft_dataset.jsonl` 11,308 rows (9,061 agent + 2,247 grounding) + `sft_val.jsonl` 1,278 |
| adapter | `models/cua-sft-v3` (`adapter_config.json`: r64, α128, base `Qwen/Qwen3.5-9B`; `args.json`: 2 epochs, lr 5e-5, cosine, bf16) |
| the relabel prompt under suspicion | `repo_pipeline/outcome_relabel.py:46-84` |
| the observation builders | `train_qwen.py:56` `SYSTEM_PROMPT`, `:162` `effect_annotation`, `:171` `step_effects`, `:309` `history_entries_for_step`, `:336` `build_step_observation` |
| the training gate | `repo_pipeline/corpus_audit.py:122` `trainability` |

Three corrections to `sft-v3-results.md` that this postmortem depends on.

1. **The matched sft-v2 re-record is complete.** That file says it "reached 2 of 6 builds before
   being stopped". On disk `matched-sft-v2` holds 7 completed runs, 105 episodes and a scored
   `report.json`. Every sft-v2 comparison below therefore uses the matched run, not the published one.
2. **The reporter model differs between the two harnesses, and that is a confound the results file
   does not name.** `manifest.json` `models`: the three v3-harness arms all run
   `planner_model` / `reporter_model` = `qwen-3.5-9b` with only the executor swapped, which makes them
   a clean three-way comparison. The original-harness runs (`econ-sft1`, `econ-sft2`) used
   `cua-student-lora` as planner **and** reporter. So sft-v2's DDR falling from 7/33 (original) to
   3/33 (matched) is not a model effect at all — same weights, different harness and different
   reporter.
3. **`error verdicts 62.9% → 54.3%` compares across harnesses.** Against the matched sft-v2 the
   figure is 61.9% → 54.3%, and the two Wilson intervals overlap (§4).

## 1. What the round predicted

`sft-v3-corpus-round.md` fixed the bands before any number existed. Definitive success required the
coordinate-loop row and the premature-done row to pass together with the error-verdict row;
definitive failure was either behavioural row landing in its fail band.

| check | pass | fail | quoted sft-v2 baseline |
| --- | --- | --- | --- |
| coordinate loops (same point ≥6× per episode) | ≤ 6% | ≥ 12% | 18.1% |
| premature done (`done` in ≤5 steps) | ≤ 15% | ≥ 28% | 39.0% |
| success claims absent from the final page text | ≤ 15 | ≥ 30 | 41/105 |
| error verdicts | ≤ 45% | ≥ 55% | 62.9% |
| verification cluster TC-003/004/005/012 judgeable | ≥ 12/28 | ≤ 5/28 | 1/28 |
| DDR strict | 33-42% expected | < 27% contradicts | 21.2% |

The changes meant to deliver this: two images per step (640x360 previous + current 1280x720),
harness-computed effect annotations in the history (` [no change]`, ` -> /path`), `normalized_1000`
coordinates, an outcome-relabel pass over every trainable thought, a 25% grounding slice, QA-voice
instructions, a verify-style capture tranche, and 2 epochs instead of 3.

## 2. What happened — the behavioural rows

Recomputed from the recorded episodes, no judge involved, n=105 case-runs per arm, Wilson 95%.
A "coordinate loop" is an episode in which one `(x, y)` pair is the target of ≥6 recorded actions.
Premature done is `episode.done` with `episode.steps ≤ 5`.

| arm | loops ≥6 | loops ≥5 | premature done |
| --- | --- | --- | --- |
| **sft-v3** | **9.5% [5.3, 16.6]** | 11.4% [6.7, 18.9] | **27.6% [20.0, 36.8]** |
| sft-v2 matched | 26.7% [19.1, 35.8] | 29.5% [21.6, 38.8] | 37.1% [28.5, 46.7] |
| sft-v2 original | 10.5% [6.0, 17.8] | 18.1% [11.9, 26.5] | 39.0% [30.3, 48.6] |
| sft-v1 original | 5.7% [2.6, 11.9] | 11.4% [6.7, 18.9] | 27.6% [20.0, 36.8] |
| base (either harness) | 0.0% [0.0, 3.5] | 0.0% | 0.0% |

**A definitional caveat that has to be stated.** The pre-registered baseline of 18.1% is not
reproducible from the definition the plan gives. At the stated ≥6 threshold, sft-v2 original scores
10.5%; 18.1% is exactly 19/105, which is what a ≥**5** threshold yields on that same run. No script
that produced the pre-registered figure exists on disk, so the row cannot be scored strictly as
pre-registered. Both thresholds are reported above, applied identically to every arm, and both
support the same reading.

**Loops: a real, resolvable improvement.** Against the matched sft-v2 the intervals do not overlap
(9.5% [5.3, 16.6] vs 26.7% [19.1, 35.8]) at either threshold. This is the one place where the
architecture change demonstrably took: the model uses the `[no change]` signal. Against the
pre-registration's own numbers it still lands between the bands — "moved, not decisive".

**Premature done: not resolvable, and no better than sft-v1.** 27.6% vs the matched 37.1%; the
intervals overlap heavily. 27.6% is 0.4 pp below the ≥28% fail threshold, i.e. at the fail edge, and
it is the *identical* rate sft-v1 already had. Whatever moved here, it is not attributable to this
round.

**Success claims absent from the final page.** The pre-registered 41/105 is likewise not
reproducible — no definition on disk matches it. Two defensible measures, applied uniformly:

| arm | closing claim unsupported | any-thought claim unsupported |
| --- | --- | --- |
| sft-v3 | 2/105 = 1.9% [0.5, 6.7] | 42/105 = 40.0% |
| sft-v2 matched | 2/105 = 1.9% [0.5, 6.7] | 25/105 = 23.8% |
| sft-v2 original | 1/105 = 1.0% | 27/105 = 25.7% |
| base matched | 0/105 | 0/105 |

"Closing claim unsupported" = the final thought quotes a confirmation string (`saved`, `created`,
`logged in`, …) that does not appear in the episode's `page_text`. That is the metric the row was
trying to capture and both SFT arms are at 2/105. The "any-thought" column is *not* a valid proxy —
it compares a mid-run quotation against the *final* page, so a claim that was true when made counts
as unsupported. It is reported only because 42/105 sits suspiciously near the pre-registered 41/105
and someone will otherwise re-derive it and read a regression into it.

**Verification cluster** (TC-003/004/005/012 × 7 runs = 28): sft-v3 **6/28** judgeable
(2 passed, 4 failed) against 1/28 for matched sft-v2, 1/28 original, 1/28 sft-v1, 0/28 base. A 6×
move, above the ≤5/28 fail band, far below the ≥12/28 pass band, and the Wilson intervals (21.4%
[10.2, 39.5] vs 3.6% [0.6, 17.7]) touch — borderline, not resolvable at 95%.

**Verdict summary against the pre-registration: no row passed; no row failed outright except by the
plan's own composite rule.** The plan required loops *and* premature done to pass together. Loops
landed between the bands, premature done landed at the fail edge. By the plan's own criterion this
is not a definitive success, and its "moved, not decisive" clause is the honest label.

## 3. What happened — the corpus and the training

Corpus scan, `python -m repo_pipeline.verify_pattern_scan trajectories` on 2026-09-04, 784 trainable
trajectories, against the same scan's 2026-09-01 pre-round numbers in `sft-v3-plan.md`:

| | before | after | plan's bar |
| --- | --- | --- | --- |
| done() steps preceded by a verification thought | 119/739 = 16.1% | **208/740 = 28.1%** | ≥ 60% |
| verification language anywhere | 69.5% | 80.1% | — |
| verification language after a save/submit | 40.9% | 50.5% | — |
| mid-run navigate (re-open to check) | 11.6% | 16.3% | — |
| pagination language | 1.8% | **2.2%** | tranche of ~30 |
| two password entries ≥5 steps apart | 3.4% | **3.3%** | tranche of ~30 |

The verify-style capture tranche landed (16.1% → 28.1% is real) but reached less than half of the
plan's bar. **Plan item 3 — the pagination and re-authentication tranches — did not land at all.**
That matters because `navigation` and `state_dependent` are exactly the defect classes where sft-v3
scored 0/5 and 1/3.

Effect annotations over the 774 relabeled trajectories (13,310 non-first steps, recomputed with
`train_qwen.step_effects`): 13.5% `[no change]`, 15.6% a URL path change, 71.0% changed without a
path move. Consistent with the 13.0% / 15.5% quoted at design time.

Dataset actually trained: 11,308 train rows, of which 2,247 (19.9%) are grounding examples — under
the 25% ceiling. Two epochs; epoch 2 was trained and rejected on rising eval loss
(0.5587 → 0.5791 against train loss falling to 0.464), so the served adapter is the epoch-1
checkpoint at step 1,400.

## 4. What happened — the benchmark

Economy sweep, 7 runs (1 clean + 6 economy builds) × 15 pinned cases = 105 case-runs, 33 catalogue
defects, `pinned_plan_hash 68d4435e…` identical across every arm.

| arm | DDR strict (Wilson) | reports | precision | FPR | error verdicts (Wilson) | passed | failed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base matched | 0/33 = 0.0% [0.0, 10.4] | 5 | 0% | 1.00 | 66.7% [57.2, 75.0] | 0 | 7 |
| base original | 0/33 = 0.0% [0.0, 10.4] | 0 | — | 0.00 | 73.3% [64.2, 80.9] | 0 | 0 |
| sft-v1 original | 9/33 = 27.3% [15.1, 44.2] | 14 | 71.4% | 1.00 | 60.0% [50.4, 68.9] | 25 | 17 |
| sft-v2 original | 7/33 = 21.2% [10.7, 37.8] | 9 | 100% | 0.00 | 62.9% [53.3, 71.5] | 29 | 9 |
| sft-v2 matched | 3/33 = 9.1% [3.1, 23.6] | 3 | 100% | 0.00 | 61.9% [52.4, 70.6] | 33 | 7 |
| **sft-v3** | **7/33 = 21.2% [10.7, 37.8]** | 7 | 100% | 0.00 | **54.3% [44.8, 63.5]** | 31 | 13 |

**What this table can and cannot support.**

- **DDR resolves nothing.** n=33 defects at 1 repeat. Every SFT interval overlaps every other SFT
  interval. sft-v3 21.2% [10.7, 37.8] and matched sft-v2 9.1% [3.1, 23.6] overlap across
  [10.7, 23.6]; sft-v1's 27.3% [15.1, 44.2] overlaps both. **No DDR difference in this table is a
  real difference.** The same weights (sft-v2) score 21.2% and 9.1% under two harnesses, which is
  itself the measurement's error bar made visible.
- **Per-defect-class DDR is 5 defects per class and resolves nothing whatsoever.** The
  `sft-v3-results.md` table (`action` 0% → 20%, `navigation` 40% → 0%, `silent_failure` 20% → 0%)
  is a reshuffle of ±1 detection inside noise and should not be read as a finding. It is retained
  below only as a *pointer* to the mechanism, corroborated by the traces, not as evidence on its own.
- **Error verdicts moved in the intended direction but not resolvably.** 61.9% → 54.3%, intervals
  [52.4, 70.6] and [44.8, 63.5], heavy overlap. Part of the move is also room, not behaviour: the
  matched protocol gave both arms 25 steps, but sft-v3 was *trained* with a 20-entry history it now
  actually gets, whereas sft-v2 at window 20 is off the distribution it was benched on originally.
- **Precision and FPR are a genuine but shared property.** sft-v3 files 7 reports, all 7 match, FPR
  0.00. This is *not* unique to sft-v3: matched sft-v2 (3/3) and original sft-v2 (9/9) also score
  100% precision and FPR 0.00. The arms that hallucinate bugs are sft-v1 (71.4%, FPR 1.00) and the
  **base model under the qwen reporter** (5 reports, 0 matched, FPR 1.00). The correct claim is
  "the SFT lane since v2 does not invent bugs", not "sft-v3 alone does not".

Per-build, with each build's injected defects (`records/refapp/local/*.json` `defect_ids`):

| build | defect classes injected | sft-v3 | sft-v2 matched | sft-v2 orig | sft-v1 orig |
| --- | --- | --- | --- | --- | --- |
| clean | — | 0 bugs, 7 pass / 8 err | 0, 7/8 | 0, 6/9 | 1, 5/9 |
| economy-1 | silent_failure, action, validation, labeling | **3**, 5/6 | 1, 3/10 | 4, 2/9 | 5, 2/8 |
| economy-2 | action, **navigation** ×2, validation, labeling | **0**, 2/13 | 0, 3/12 | 1, 3/10 | 3, 2/10 |
| economy-3 | silent_failure, labeling ×2, validation, data | **2**, 4/8 | 0, 6/8 | 1, 5/9 | 2, 4/9 |
| economy-4 | state_dependent, data, validation, action, labeling | **1**, 7/6 | 0, 6/8 | 0, 5/10 | 0, 5/8 |
| economy-5 | state_dependent ×3, navigation, data | **1**, 5/7 | 0, 4/10 | 1, 4/10 | 1, 4/10 |
| economy-6 | data, **navigation** ×2, silent_failure, data | **0**, 1/9 | **2**, 4/9 | 2, 4/9 | 3, 3/9 |

Cells are raw `execution.bugs_found` lengths per build; they can exceed the precision denominator in
§4, which excludes reports filed on the clean build (sft-v1: 15 raw vs 14 scored; base matched:
6 vs 5). The two SFT arms compared here file nothing on the clean build, so their per-build counts
sum exactly to their §4 totals, 7 and 3.

sft-v3 leads or ties on five of the six economy builds and is beaten on exactly one: economy-6. Its
two zero-bug builds — economy-2 and economy-6 — are the two builds whose defect set puts a
**navigation** fault on the agent's own critical path. That is the pattern §5 explains.

## 5. The central finding: the relabel pass taught a phrase, not a judgement

### 5.1 The two traces

`results/econ-sft-v3/episodes/refapp/local/e7__economy-6__r0/TC-008.json` — TC-008 is "successful
login with valid credentials, redirected to the dashboard"; the build injects
`login_redirects_to_register`. `episode.thinking`, verbatim:

> [3] **Clicking Log in worked: the page moved to /register** and now shows a 'Create account' form
> with Email, Password, and Display name fields, plus a 'Log in' link. I'll click 'Log in' to return
> to the login page and try again.

The defect is observed, named exactly, and labelled "worked". The model then spends 21 further steps
re-entering the same credentials. It gets the judgement right *once*, at step 7 —

> [7] Clicking Log in again took me to /register instead of logging me in, so the credentials still
> aren't being accepted.

— and then reverts to the template for the rest of the episode, ending with the most direct
contradiction in the corpus of runs:

> [17] **Clicking Create account worked**: the page moved to /register and now shows **'This email is
> already registered' in red, confirming the account was created.**
>
> [24] *(identical, six steps later)*

An error banner is read as confirmation of the success the template requires. The episode terminates
at the 25-step cap with `done=false`, `final_url=/login`, and no bug report.

The same mechanism on a different build and a different defect —
`e7__economy-2__r0/TC-010.json`, where `tasks_nav_link_wrong_target` sends the Tasks link to
/settings:

> [4] **Clicking Settings worked: the page moved to /settings** and the Settings page shows the same
> top nav bar … I'll click Tasks next to confirm it routes correctly from this page too.
>
> [11] **Clicking Settings again shows no change, confirming the nav bar is stable and functional.**
> I'll click Tasks next to verify that link also routes correctly.

Two things are visible here that the first trace does not show. First, the model **rewrites its own
intent to match the outcome**: it clicked *Tasks*, landed on /settings, and narrated the action as
"clicking Settings". Second, even the `[no change]` channel — the one that demonstrably works — gets
a positive gloss: a dead nav link is "stable and functional". 25 steps, `done=false`, no report.

### 5.2 The prompt that produced it

`repo_pipeline/outcome_relabel.py:56-62`, the two rules every one of the 774 relabeled trajectories
was rewritten under:

```
- The FIRST sentence states the outcome of the previous action, judged by comparing the previous
  screen with the current one — e.g. "Clicking Save worked: a 'Settings saved' banner is visible."
  or "Nothing changed after that click; the menu is still closed." For step 1 (no previous action)
  describe what the starting screen shows instead.
- The bracketed note after a previous action is what the model will be told: [no change] means the
  screen stayed the same, -> /path means the page moved to that path. Agree with it and say what it
  means on screen.
```

Two defects, and they compound.

1. **The rule defines "outcome" as *change*, not as *match with intent*.** The comparison it asks
   for is previous screen vs current screen. Nowhere does it ask whether the new screen is the one
   the action was *for*. The binary it teaches is `changed / did not change`, and the worked
   example attaches the word "worked" to the changed branch.
2. **"Agree with it" makes the mechanical annotation authoritative.** `effect_annotation`
   (`train_qwen.py:162`) emits ` -> /path` for *any* URL path change, including a redirect away from
   the intended destination. The prompt instructs the relabeller to agree with that note. So a
   defect that manifests **as** a navigation is, by construction, evidence that the action worked.

`SYSTEM_PROMPT` (`train_qwen.py:56`) then describes the annotation to the student in the same
change-only terms: "`[no change]` when the screen did not change, `-> /path` when the URL path
changed". The student is never told that a path change can be the wrong path.

### 5.3 The corpus measurement

Every relabeled thought's opening sentence, classified as **success** (asserts the action worked:
`worked`, `succeeded`, `successful`, `confirming`, `took effect`, `has been saved/created/…`),
**failure** (asserts it did not: `nothing changed`, `no change`, `didn't`, `failed`, `still shows`,
`unchanged`, `rejected`), **mixed** (both) or **neutral**. 14,084 thoughts across 774 trajectories,
cross-tabbed against the harness annotation on the *previous* action.

| | success | failure | mixed | neutral |
| --- | --- | --- | --- | --- |
| all relabeled thoughts | **6,949 (49.3%)** | **2,127 (15.1%)** | 469 (3.3%) | 4,539 (32.2%) |
| the same thoughts before relabel | 1,144 (8.1%) | 972 (6.9%) | 56 (0.4%) | 11,912 (84.6%) |

| previous action annotated | n | success | failure | mixed | neutral |
| --- | --- | --- | --- | --- | --- |
| ` -> /path` (URL moved) | 2,073 | **1,572 (76%)** | **82 (4%)** | 51 (2%) | 368 (18%) |
| changed, no path move | 9,445 | 5,171 (55%) | 912 (10%) | 321 (3%) | 3,041 (32%) |
| ` [no change]` | 1,792 | 206 (11%) | **1,110 (62%)** | 96 (5%) | 380 (21%) |
| first step | 774 | 0 | 23 (3%) | 1 | 750 (97%) |

**This is the mechanism, measured.** The two annotation channels were supervised asymmetrically:

- The `[no change]` channel learned a genuine judgement — 62% failure assertions against 11% success
  — and it works at inference: sft-v3 halves the coordinate-loop rate against matched sft-v2 (§2)
  and raises failure-opening thoughts from 3.0% to 5.3%.
- The `-> /path` channel learned **change ⇒ success**, 76% to 4%, a 19:1 ratio. There is
  effectively no supervision in this corpus for "the page moved, and it moved somewhere wrong".

Before/after pairs make the rewrite visible. Each of these is a step whose previous action was
annotated with a path change; the *only* change the relabel pass made was to prepend a success verdict
to an accurate neutral description:

| trajectory / step | before | after |
| --- | --- | --- |
| `repo_abhii145_ecommerce/task_browse-fragrances-favourite` step 2 (`-> /products`) | "I'm now on the Fragrances category page with several products listed." | "**Clicking Fragrances worked: the page moved to /products** and now lists several fragrance items…" |
| `repo_dantwoashim_SSM/task_create-engagement` step 2 (`-> /login`) | "The sign-in form is visible." | "**Clicking Portal worked:** the page moved to the sign-in form…" |
| `repo_iib0011_omni-tools/task_csv-to-json-with-typed-data` step 2 (`-> /csv/csv-to-json`) | "I'm now on the CSV to JSON converter page…" | "**Clicking the button worked: the page moved to /csv/csv-to-json** and now shows the CSV to JSON converter…" |

And 148 of 1,147 (12.9%) no-effect sentences in the relabeled corpus attach a positive gloss to the
non-event ("Clicking Beautify Code caused no visible change, **confirming** the output was already
correctly beautified") — the pattern reproduced verbatim by economy-2 TC-010 thought [11].

### 5.4 The transfer to inference

Opening-sentence class of every recorded eval thought, same classifier, same harness for the first
three rows:

| arm | thoughts | success | failure | mixed | neutral |
| --- | --- | --- | --- | --- | --- |
| **sft-v3** | 1,189 | **52.4%** | 5.3% | 6.0% | 36.3% |
| sft-v2 matched | 1,267 | 10.3% | 3.0% | 1.9% | 84.8% |
| sft-v2 original | 970 | 8.8% | 2.9% | 1.4% | 86.9% |
| sft-v1 original | 942 | 7.4% | 0.2% | 1.5% | 90.9% |
| base matched | 665 | 25.3% | 0.0% | 0.0% | 74.7% |

The corpus's 49.3% success-opener share reproduces at inference as 52.4%, against 10.3% for the same
architecture trained on the un-relabeled corpus. n>1,100 per side; this is the most resolvable
number in the whole postmortem. The relabel pass transferred, exactly and quantitatively — the
template transferred, and the judgement did not come with it.

A mechanical corollary: sft-v3 produces **167 thoughts** that assert success while naming a
destination path the test case never mentions. Every other arm produces **zero**, because no other
arm was trained to narrate path changes at all. That count over-collects (most such navigations are
benign), so it is reported as a marker of the template's reach, not as an error count.

A hand audit does supply the error count. Of the 46 sft-v3 thoughts that assert success while the
same thought quotes an error, rejection or anomaly string, **10 are flat contradictions** — the
success claim is refuted by the evidence the thought itself cites. Nine of the ten are in
economy-6; the tenth (economy-1 TC-007) reads a password-length error as proof that email validation
fired. The other 36 are correct reasoning about *expected* negative-test errors ("Clicking Save
worked: a red 'Title is required' error message … confirming the server-side validation fired") and
must not be counted against the model. The matched sft-v2 comparison, audited identically: 3
contradictions of 25 candidates. These counts are small and the difference between them is not
statistically resolvable; they are corroboration for the traces and the corpus cross-tab, not
independent evidence.

### 5.5 Why zero reports rather than a wrong report

Two contributing causes, in order of weight.

**The narration is what the reporter reads.** All three v3-harness arms share the same
`qwen-3.5-9b` reporter, so the only input that differed on economy-6 was the executor's episode.
sft-v3 ended 5 of its 15 economy-6 episodes on `/register` — more than matched sft-v2's 2 — and the
reporter still filed nothing, because those 25 steps of narration say the redirect worked. Matched
sft-v2's TC-008 episode is 4 steps long, ends on `/register`, and its final thought is *also* wrong
("confirming I'm logged in as the registered user and have been successfully redirected to the
dashboard") — but the episode stops on the broken screen, and the reporter filed the correct bug
anyway.

**Persistence consumed the budget.** economy-6 per-arm episode outcomes:

| arm | hit cap | done | mean steps | final URLs |
| --- | --- | --- | --- | --- |
| sft-v3 | 10/15 | 5/15 | 20.4 | /login ×7, /register ×5, / ×1, /settings ×1, /tasks/1/edit ×1 |
| sft-v2 matched | 3/15 | 12/15 | 12.9 | spread over 8 distinct routes |
| base matched | 14/15 | 0/15 | 23.4 | /login ×15 |

sft-v3 reached 1 passed / 1 failed / 9 error / **4 skipped** on economy-6 — the only build in the
whole sweep where any sft-v3 case was verdicted `skipped`. Four cases were never executed far enough
for the reporter to form any opinion, because the earlier cases had spent the run fighting a login
that could not succeed.

The uncomfortable corollary for the benchmark: **the failure mode sft-v3 was built to fix — stopping
early on the wrong screen — is what earned sft-v2 its economy-6 bug reports.** A short wrong episode
that halts on the defective page is more legible to a downstream reporter than a long correct-looking
episode that navigates away from it. This is a property of the measurement instrument, not of QA
competence, and any future round that optimises DDR without controlling for episode length is
partly optimising for early stopping.

## 6. The structural cause: no example of "the app is wrong"

The relabel prompt is the proximate cause. The training gate is the deeper one.

`corpus_audit.trainability` (`repo_pipeline/corpus_audit.py:129`) drops any agent trajectory with
`success: false`. Over the 1,398 task dirs on disk:

| gate outcome | dirs |
| --- | --- |
| trainable | **784** |
| `unverified` (agent capture, `success: false`) | **461** |
| `replay_gated` (legacy replay lane) | 76 |
| `excluded` (`metadata.excluded_reason`) | 46 |
| `stuck_loop` | 31 |

461 trajectories — 37% of the agent-captured corpus — are exactly the runs in which the agent did
not achieve the goal, and they are all withheld. What survives:

- Terminal action of the 784 trainable trajectories: **740 `done`**, **31 `handoff`**, 8 `click`,
  5 other. There is no third category.
- All 31 handoffs attribute the stop to an **environment wall** — CAPTCHA, human verification, a
  credential the agent must not type. Verbatim: *"this is a CAPTCHA-style human challenge that I
  shouldn't try to solve myself"*; *"This needs my actual account credentials, so I'll hand off"*.
  None attributes the stop to the application being defective.
- App-fault language (`is a bug`, `seems broken`, `not working as expected`, `a defect`, …) appears
  in **44 thoughts across 37 of the 1,398 dirs**, and only **10 of those dirs are trainable** —
  1.3% of the training corpus. Reading all ten: they say *my action* isn't working ("The textbox is
  showing 100 and my typing isn't working properly"), not *the app* is wrong. The two thoughts that
  do accuse the application ("the GitHub authentication flow itself is broken"; "The app seems
  broken at the root level") are both in **excluded** trajectories.
- **Trainable trajectories whose final thought names an application fault: 0 of 784.**

So the model has never once seen the behaviour the benchmark scores. Every trainable example ends in
one of two ways: I achieved the goal, or a wall I am not permitted to cross stopped me. "The
application did the wrong thing, I am confident about it, here is the evidence, and I stop and
report" is not a supervised behaviour in this corpus, and no prompt fix inside `outcome_relabel.py`
can add it — the examples are not there to relabel.

This also explains the shape of §4's per-build table without appealing to per-class DDR: a defect
that the agent can walk *past* (a wrong label, a wrong count) leaves an intact episode from which
the reporter can extract a bug. A defect that blocks the agent's own path (economy-2's and
economy-6's navigation faults) requires the agent to *give up and say why* — the one thing 784
trajectories never demonstrate.

## 7. Where the round succeeded

Stated plainly, because the failure above is loud.

- **Coordinate loops halved and the interval separates.** 26.7% → 9.5% against the matched sft-v2,
  non-overlapping Wilson intervals. The `[no change]` annotation and the previous-screen image are
  used. This is the round's one clean behavioural win and it validates the observation-architecture
  half of the hypothesis.
- **The verification cluster moved 6×**, 1/28 → 6/28 judgeable, and the corpus metric behind it
  moved 16.1% → 28.1% done-after-verification. Below the bar, above the noise floor, and directly
  attributable to the `qa_verify` capture tranche.
- **Error verdicts fell** 61.9% → 54.3% (matched) — directionally right, not resolvable, and partly
  attributable to the 25-step cap.
- **No hallucinated bugs.** 7 reports, 7 matched, FPR 0.00 on the clean build. Shared with sft-v2,
  but it is the property that makes the lane usable at all, and sft-v1 (71.4% precision, FPR 1.00)
  and the base model under the same reporter (0/5, FPR 1.00) both lack it.
- **The base-model comparison is decisive in the round's favour** and is the only comparison in this
  document with no overlap anywhere: base matched files 5 reports and matches 0, emits `done` in
  0/105 episodes, hits the step cap in 98/105, and reaches 0 passed verdicts. Whatever SFT is doing,
  it is the difference between an agent and a narrator.
- **The prompt-format machinery held.** One builder (`build_step_observation`) feeds convert,
  student eval, ScreenSpot and the bench's vendored copy; the trained dataset respected the 25%
  grounding ceiling (19.9%); epoch 2 was correctly rejected on eval loss.

## 8. Proven, plausible, unknown

**Proven** (recomputable from the files named in §0):

1. The outcome-relabel pass installed a success-assertion template: corpus opening sentences moved
   8.1% → 49.3% success, and the eval-time rate is 52.4% against 10.3% for the same architecture on
   the un-relabeled corpus. n > 1,100 per side.
2. The template is conditioned on *change*, not on *intent*: 76% success openers after a URL path
   change against 4% failure openers, versus 62% failure openers after `[no change]`.
3. The training corpus contains zero trainable examples of concluding that the application is at
   fault, and 461 `success: false` trajectories — 37% of agent captures — are gated out.
4. sft-v3 halves the coordinate-loop rate against the matched sft-v2 with non-overlapping intervals.
5. No pre-registered row passed. DDR did not move resolvably in any direction, for any arm.
6. The same sft-v2 weights score DDR 7/33 and 3/33 under two harnesses that differ in reporter model
   and protocol.

**Plausible, mechanism-supported, not statistically resolved:**

7. The economy-6 zero-report result is caused by the template rather than by grounding or by the
   step cap. Support: two independent traces on two builds and two defect classes showing the same
   rewrite; the corpus cross-tab predicting exactly this behaviour; sft-v3's two zero-bug builds
   being precisely the two navigation-defect builds. Against: n=1 run per build, and per-class DDR
   at 5 defects resolves nothing.
8. Persistence destroying terminal evidence is a *contributing* cause. Support: economy-6 mean steps
   20.4 vs 12.9, 10/15 cap hits, 4 skipped cases. Against: sft-v3 ended on `/register` more often
   than sft-v2 did and still filed nothing, so narration outweighs terminal state.
9. Part of the 61.9% → 54.3% error-verdict move is the 25-step cap rather than behaviour.

**Unknown, and cheap to settle:**

10. Whether a repeated click is a *wrong coordinate* or a *correct coordinate whose action had no
    effect*. `sft-v3-corpus-round.md` flagged this as unmeasured; it still is, and it is the
    difference between a grounding problem and a policy problem.
11. ScreenSpot-300 and Online-Mind2Web-174 for sft-v3. Both pre-registered as no-regression checks,
    neither run. `eval_results/` holds only `smoke.json`; `eval_runs/` is empty.
12. Whether `normalized_1000` helped. It cannot be separated from the relabel pass, the second
    image, the grounding slice and the epoch change — five interventions in one round on the same
    checkpoint.
13. The verdict-tier cost anomaly ($10.68 vs $0.73 per run) remains unexplained, as
    `sft-v3-results.md` states. Verdict-phase token counts are not recorded, only dollars.

## 9. The prompt fix

The defect is one sentence of `OUTCOME_PROMPT` (`repo_pipeline/outcome_relabel.py:56-62`). The
replacement must (a) make the comparison *intent vs outcome* rather than *screen vs screen*,
(b) demote the bracketed annotation from verdict to observation, and (c) supply a worked example of
the third case — the action did something, and that something was wrong.

Replace the two rules with:

```
- The FIRST sentence judges the previous action against what it was FOR, not merely against whether
  the screen changed. Name the intent, then the observed result, then whether they match:
    * they match      -> "Clicking Save worked: a 'Settings saved' banner is visible."
    * nothing happened -> "Nothing changed after that click; the menu is still closed."
    * something happened, but not the intended thing ->
      "Clicking Log in did NOT log me in: the page went to /register and is showing a 'Create
       account' form, which is not where signing in should lead."
  A screen that changed is NOT evidence that the action succeeded. Never write "worked" unless the
  new screen is the one the action was aiming for.
- The bracketed note is a raw observation, not a verdict. [no change] means the pixels did not move;
  it does not by itself mean the action failed (a correct action can be a no-op, e.g. re-selecting an
  already-active filter). -> /path means the URL moved to that path; it does not by itself mean the
  action succeeded — say whether that path is where the action was supposed to lead, and if it is
  not, say so plainly.
- If the screen shows an error, a rejection or a message contradicting what the action was meant to
  do, the first sentence must say the action did not do what was intended. Never gloss a visible
  error as confirmation of success.
```

The system prompt needs the matching change (`train_qwen.py:56`), so the student is not told at
inference that a path change means the action landed:

```
… and your previous actions annotated with their observed effect ([no change] when the screen did
not change, -> /path when the URL path changed). These annotations report WHAT HAPPENED, not whether
it was what you wanted: judge each action by comparing the new screen against what the action was
for.
```

Both are one-line-per-rule edits and both are cheap to validate: re-run `outcome_relabel` in
`--dry-run --sample 40` over trajectories whose steps carry ` -> /path` annotations and check that
the success-opener share on that slice falls from 76% toward the ~55% of the plain-change bucket,
with the difference landing in "changed, but not as intended" rather than in "neutral".

## 10. What the next round should do differently

Ordered by expected effect, and deliberately fewer interventions than this round carried.

1. **Supply the missing behaviour, do not just fix the phrasing.** The prompt fix removes a wrong
   label; it cannot create an example of "the app is broken, report it and stop". Two options,
   preferably both:
   - a **defect-capture lane**: boot a repo app with a deliberately injected fault (the QA bench's
     own defect catalogue is the obvious generator), run the teacher against a task the fault
     blocks, and require the run to terminate in a new terminal action — `report(defect=…)` — that
     the verifier grades as `success: true`. This is the same `_STYLE_BLOCKS` mechanism the
     `qa_verify` style already uses, and it produces exactly the supervision the gate currently
     forbids.
   - **admit a graded subset of the 461 `unverified` trajectories** whose failure is attributable to
     the site rather than the agent, via a verifier verdict that distinguishes *agent failed* from
     *application prevented the goal* — the second class is training signal, not waste. That is an
     audited relabel pass in the shape of `relabel_recoverable`, and it is free of new capture cost.
2. **Change one thing per round.** This round moved observation, coordinate space, supervision
   voice, task distribution, grounding mix and epoch count together, which is why item 12 in §8 is
   unanswerable. The observation change is validated (loops); the supervision change is falsified
   (this document). Those two should never have shipped together.
3. **Fix the measurement before spending on the model.** DDR at n=33 × 1 repeat cannot resolve
   anything this lane will plausibly produce. Either raise repeats until the interval is narrower
   than the effect being chased, or move the headline to the 105-case verdict tier and report DDR
   as descriptive. And control for episode length: an instrument that pays for stopping on the
   broken screen (§5.5) will keep rewarding the wrong behaviour.
4. **Instrument the loop diagnosis** (§8 item 10). A bench-side probe answering "did this click land
   inside any interactive element's box?" costs nothing at inference and tells the next round
   whether grounding or policy is the binding constraint.
5. **Land the two tranches that did not land.** Pagination is still 2.2% of the corpus and mid-flow
   re-authentication 3.3%, essentially unmoved from 1.8% / 3.4%. Both map to named all-fail clusters.
6. **Run the two no-regression evals** (ScreenSpot-300, Mind2Web-174). Until they exist, the claim
   that this round did not cost general browsing is unsupported.
7. **Register the round.** `experiments/results.csv` carries `cua-sft-v3` result rows but
   `experiments/models.csv` has no `cua-sft-v3` or `corpus-v3` row, and neither `matched-sft-v2` nor
   `matched-base` appears in the registry at all. The matched arms are the correct baseline for
   everything above; they need rows before any of these numbers is cited.
