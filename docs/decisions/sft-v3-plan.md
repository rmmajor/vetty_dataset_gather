# SFT v3 — what to build next, and why (2026-09-01)

Written after a second, independent measurement of `sft-v1` arrived from the vetty QA benchmark
(510 cases, 34 runs x 15 test cases, same builds and plans, verdicts on `claude-opus-5`). That
benchmark measures something Online-Mind2Web cannot — whether the agent *finishes a task inside a
known application and correctly reports what it found* — and it names a failure the corpus round
that produced `sft-v2` did not target. This file records which of its findings survive contact with
our own corpus, which are already closed, and what v3 should therefore be.

Companions: `docs/decisions/sft-v2-corpus-round.md` (what v2 changed),
`docs/decisions/sft-v2-results.md` (what v2 measured), `docs/open-items.md` §8 (RL).

## The external measurement

Base `Qwen3.5-9B` vs `sft-v1` on the QA benchmark. Reproduced here because the deltas are much
larger than anything Online-Mind2Web resolved, and because the failure profile is the useful part.

| dimension | base | sft-v1 |
| --- | --- | --- |
| action protocol | 0-5 parsed actions per 15-step episode, ~140 tokens of prose per step | ~1 action per step, canonical form 100% |
| completion | done 0%, every episode hit the 15-step cap | done 88%, mean 8.8 steps |
| judgeable at all (passed+failed) | 0.8% | 37.5% |
| core flows | TC-008 login 0/34 passed | 33/34; TC-001 31/34; TC-013 32/34 |
| defect detection | 0/33 | 10/33 |
| reporting | 4 reports, 0 matched | 65 reports, 10 matched |
| cost | ~8.5 s/step, 22 min/run | ~1 s/step, 5 min/run |

Its three failure clusters, by case:

1. **act -> save -> verify again.** TC-003/004/005 (setting persistence, delete + counter):
   error 33-34/34. TC-012 (toggle Done + Save): done 0%, all 34 hit the cap. This is why
   `silent_failure` detection is 0% and `data` is 20% — the model does not make a verification pass
   after acting.
2. **Long routes.** TC-011 (reach page 3 of pagination): 34/34 error.
3. **Multi-stage flows with two logins.** TC-014/015 (password reset + token re-use): 31-34 error.

And one harness artifact of theirs, which is the honest half of the report: the final `page_text`
in their episode bundle is page inner-text, and `<input>` *values* and native
`validationMessage` bubbles do not appear in inner-text. TC-002 (32/34 failed) and TC-007 (26/34
failed) together account for ~58 of 73 failed verdicts, so the reported precision of 15.4% and
FPR of 2.0 understate the model. Their fix is in the bench, not the model.

## What `sft-v2` already closed

Three of their five priorities are stale as of the v2 round, measured on the pinned 50-task
Online-Mind2Web subset, same night, same criteria-first judge:

| | base | sft-v2 |
| --- | --- | --- |
| parse-failure rate | 37.7% | **0.0%** |
| canonical action form | 25.5% | **100%** |
| done() emission | 12.0% | **30.0%** |

Their priority 1 (their bench harness) is theirs to fix. Their priority 5 (generalisation) was
partly addressed by the search lane (~74 trajectories on 21 sites with zero overlap with the
Online-Mind2Web site set); the v2 scored success rate, 12.1% (4/33) against base 8.8% (3/34), is
inside noise at that sample size and settles nothing either way.

What is **not** closed is their priorities 2, 3 and 4. Those are the substance of v3.

## Measured against our corpus

The claim "the model does not verify after acting" is a claim about the training distribution, so it
was checked there. Scan over the 799 trainable trajectories in `trajectories/` on 2026-09-01,
reproducible with `python -m repo_pipeline.verify_pattern_scan trajectories`. Verification language
is a deliberately generous regex — verify / confirm / re-open / reload / make sure / persisted /
still there / appears in the list — so these shares are upper bounds:

| | count | share |
| --- | --- | --- |
| verification *language* anywhere in the run | 555 | 69.5% |
| verification language *after* a save/submit step | 327 | 40.9% |
| a mid-run `navigate` (re-opening a page to check) | 93 | 11.6% |
| **done() steps preceded by a verification thought** | **119 / 739** | **16.1%** |

The corpus talks about verifying and rarely does it. 84% of our done() supervision is a bare claim
of completion. Our own hindsight-relabel pass made this marginally worse: the 46 synthesised done
steps it appended assert `"The goal is accomplished: {goal}. I stop here."` with nothing checked
first. Wall demos are a legitimate exception — "I am blocked, here is what I saw, stop" is the
correct supervision for a wall — but they are only 33 of the 739.

Two more tranches sized the same way:

| | count | share |
| --- | --- | --- |
| trajectories >= 25 steps | 239 | 29.9% |
| pagination language anywhere (`page 2`, `next page`, `load more`, ...) | 14 | **1.8%** |
| two password entries >= 5 steps apart (re-login proxy) | 27 | **3.4%** |

Worth reading carefully: our trajectories are **not short** — median 13 steps, p90 42, max 61, and
almost a third run past 25 steps. Their cluster 2 is therefore not "we lack long trajectories". It
is "we lack long *routes through a result set*" — depth-first navigation where the target is on
page 3, behind a filter, or after a "load more". That is a much more specific and much cheaper gap
to close than "capture longer runs".

## The uncomfortable consequence for v2

`docs/sft-corpus-gaps.md` treated the low done() rate as a defect and v2 raised it 12% -> 30%. Their
data says the same intervention, taken further, produces done 88% *with 63% error* — an agent that
confidently stops without having checked. Given that only 16.1% of our done() steps follow any
verification, **raising the done rate is only a win if stopping is conditioned on evidence**, and we
have no measurement showing that it is.

So the v2 done-rate improvement should not be reported as an unqualified gain, and the metric for v3
is not done-rate but **done-after-verification rate**, on both sides:

- corpus side: share of done() steps whose preceding step re-checked the changed state;
- eval side: share of `completed` episodes whose final action follows a verification step.

Neither is currently computed. Both are cheap.

## What does not transfer to our lane

Their `page_text` artifact does **not** contaminate our Online-Mind2Web numbers. Our judge is
`agent_capture.verify_trajectory`, which sends real screenshots — `_verification_screenshots`
(`agent_capture.py:798`) attaches `VERIFY_MID_SCREENSHOTS` mid-run images plus the final one — so a
filled `<input>` and a native validation bubble are visible to it exactly as they would be to a
person. Their local columns need the fix; our numbers do not move. Worth telling them, because it
also means the two benchmarks' precision figures are not currently on the same scale.

## Plan

Ordered by expected effect per dollar. RL comes after 1-3, which is also their conclusion and
`docs/open-items.md` §8's.

1. **Verification-loop capture.** The proposer gains a `verify` task style (alongside
   `_SEARCH_GOAL_BLOCK`, same `_STYLE_BLOCKS` mechanism) whose goals are inherently two-phase:
   change something that persists, then confirm it persisted by re-opening the page — not by
   trusting the toast. The teacher policy must require the re-check; the verifier must grade whether
   it happened, so an unverified run does not get `success: true`. Target ~120-150 trajectories on
   sites we already hold sessions for, since persistence tasks need an account. This is the single
   highest-value lever: it is the skill behind their `silent_failure` 0% and `data` 20%, and it is
   the skill our corpus is thinnest on relative to how often the task demands it.
2. **Done calibration in the data, not only in the reward.** Rewrite the 46 synthesised relabel
   done-thoughts so each cites what was observed rather than asserting completion, and add the
   done-after-verification metric to `corpus_audit` with a floor in `corpus_health.json`. The
   `false_done` penalty already exists in `rl/reward_config.json`; it cannot teach a behaviour the
   SFT distribution never demonstrates, which is why this precedes RL rather than replacing it.
3. **Two small targeted tranches.** Pagination depth (1.8% today) and mid-flow re-authentication
   (3.4% today): ~30 trajectories each, authored as tasks rather than hoped for. Cheap, and both map
   to a named all-fail cluster in their results.
4. **Observation architecture — scope it, do not slip it in.** Their point 3 is the deepest one and
   we had not considered it: with one screenshot and no page-state memory, an "is it still there?"
   question is not answerable in principle, no matter how good the policy. A previous-screenshot or
   text-diff channel is a change to `train_qwen.build_step_user_text`, which is the SFT/RL
   byte-parity invariant — SFT input, RL observation and the eval prompt must move together or every
   comparison breaks. This is a design round of its own, not a data pass. Note the honest
   counter-argument before committing: v2 widened `HISTORY_WINDOW` 10 -> 20 and we have no evidence
   yet that the model uses the history it already has, so measure that first.
5. **Then GRPO v3** on the sft-v3 policy, per `docs/open-items.md` §8.

## Success bar

- done-after-verification rate on the corpus: 16.1% -> **>= 60%** of non-wall-demo done steps.
- The QA benchmark re-run: TC-003/004/005/012 (the verification cluster) move off 33-34/34 error.
- Online-Mind2Web: no regression against `sft-v2` on parse-failure (0.0%) and canonical form (100%),
  and the scored success rate measured at n >= 150 per side so the result is resolvable at all —
  n=33 could never settle the v2 comparison and will not settle v3's.

## What would falsify this plan

If the verification tranche lands and the QA benchmark's verification cluster does not move, the
constraint is architectural (item 4), not distributional, and the order above is wrong. That is a
clean experiment and it is worth running item 1 partly *because* it discriminates between the two
explanations.
