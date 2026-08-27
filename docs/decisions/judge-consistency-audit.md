# Judge consistency audit — Online-Mind2Web before/after (2026-08-27)

Manual audit of every verified success in the two Online-Mind2Web runs that produced the first
before/after result. Written down because it changes the headline number, and because the failure
mode is asymmetric: it inflates the fine-tuned side only.

## What was audited

Every `verified_success == true` episode in both runs was read (judge reasoning + task), and the
suspicious ones were checked against `final_screenshot.png` directly.

| run | successes | meandering successes | reasoning contradicts verdict | efficiency mix |
| --- | --- | --- | --- | --- |
| `baseline_qwen35_9b` | 3 / 34 scored | **0** | **0** | 2 efficient, 1 acceptable |
| `after` | 10 / 34 scored | **4** | **2** | 4 meandering, 4 acceptable, 2 efficient |

**Both contradictions are `meandering` episodes**, and every meandering success drew a critical
comment — two of the four criticisms describe an unmet goal, two describe wasted steps. The
baseline produced no meandering successes at all, so this failure mode cannot appear there: it is
reachable only by a model that wanders, which is exactly what fine-tuning produced
(steps/reference 1.36x -> 2.14x).

## Confirmed false positives (checked against pixels)

- **recreation.gov** — "Open the most helpful 5-star reviews of Alpine Ridge." Final screenshot
  shows `Sort by: Most Recent` and a 4-star review. Goal not met. The judge reasoning said exactly
  that ("not filtered/sorted for 5-star or 'most helpful' reviews as requested") and still returned
  success.
- **americashealthrankings.org** — "Display the figure comparing unemployment trends among women in
  Illinois and Michigan." The final chart legend reads **Michigan** and **United States**; Illinois
  was never plotted. Same pattern: correct criticism, success verdict.

## Flagged but NOT false positives

- **gov.uk** — the final state is the Health and Care Worker visa page, which answers the question.
  The judge's complaint is wasted steps, not correctness.
- **ca.gov** — the judge states "the goal was substantively achieved"; the criticism is 32 actions
  against an expected 10.

Efficiency complaints are not correctness failures and must not be scored as such.

## Corrected numbers

| metric | baseline | after (as measured) | after (audited) |
| --- | --- | --- | --- |
| verified success, scored | 8.8% (3/34) | 29.4% (10/34) | **23.5% (8/34)** |
| verified success, all 50 | 6.0% | 20.0% | **16.0%** |

The improvement is large either way. Quote the audited figures.

## Root cause

`repo_pipeline/student_eval.py` sets both fields from the same value:

```python
record.judged_success   = bool(verdict.get("success"))
record.verified_success = bool(verdict.get("success"))
```

`efficiency` is recorded but never qualifies success, so a "goal met" verdict is taken at face
value even when the accompanying reasoning describes the goal not being met. The capture pipeline
does couple these (`agent_capture.py`: a meandering grade normally forces `success:false`); the
eval harness reuses the same verifier without that coupling.

## What to do about it

Coupling `meandering -> success:false` in the eval would drop the count to 6/34 (17.6%), but it
would also discard gov.uk and ca.gov, which are genuine successes. That trades one bias for another.

Preferred, in order:

1. **Make the verdict self-consistent in the prompt.** The judge should not assert success while
   describing an unmet goal. Have it answer the correctness question first and in isolation, then
   grade efficiency separately.
2. **Treat `meandering` + success as a review flag, not an auto-fail.** Both contradictions were
   meandering, so it is a cheap and complete trigger for manual review here — 4 episodes to read,
   both real false positives caught, and the two genuine successes kept.
3. **Keep `judged_success` and `verified_success` genuinely distinct, or collapse them to one.**
   Two names for one value implies a check that does not exist.

## Resolution (2026-08-27)

Done on the RL harness branch, because this verifier is also the GRPO reward:

1. The verdict JSON now lists the goal's `criteria` and a `criteria_met` checklist BEFORE `success`,
   and the recorded `success` is the judge's claim AND every checklist entry
   (`agent_capture.verdict_from_parsed`). The raw claim survives as `claimed_success`, and a
   disagreement is logged and stored as `contradiction` so its rate can be tracked per run.
2. The eval harness records `judged_success` = the raw claim and `verified_success` = the
   consistent verdict, so the two names are two values.
3. The RL reward (`rl/reward_config.json`) discounts a meandering success (x0.6) instead of vetoing
   it, slopes step efficiency to zero at 2x the expected actions, pays the done() bonus only on
   success, and penalises both a false done() and running into either step cap — so wandering,
   under-stopping and overclaiming are each priced separately rather than folded into success.

## Standing caveat for any published number

This harness's judge is `claude-sonnet-5` on the capture pipeline's rubric, not WebJudge/o4-mini.
The Online-Mind2Web authors measure WebJudge itself running 6-10 points above human-rated success.
Numbers from this harness are not rankable against published tables, and the audit above shows our
judge carries its own leniency on top — concentrated on wandering agents.

## Corpus counts for this run

The training run converted **456 trajectories -> 6,093 examples -> 5,477 train / 616 val**, giving
685 optimizer steps per epoch at effective batch 8 (5,477 / 8), and 2,055 steps over 3 epochs.

An earlier local convert of the same directory reported 415 trajectories / 5,354 examples; the
corpus grew between that convert and the upload because capture was still running. Any figure
quoted for this checkpoint must come from the run log, not from an earlier local convert.
