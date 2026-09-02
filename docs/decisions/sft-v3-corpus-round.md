# SFT v3 — the observation round (2026-09-02)

The response to the QA benchmark's SFT-2 failure report and to `sft-v3-plan.md`. SFT-2 reached
DDR 21.2% with 63% of cases ending unjudgeable; its 66 error cases decompose into action cycles
(34.8%), premature `done` (30.3%), `done` on the wrong screen (22.7%) and wandering (12.1%), and it
regressed against SFT-1 on the first two. Three of those buckets need information a
one-screenshot policy does not have; the fourth needs supervision the corpus did not contain. This
round changes the observation, the supervision and the task distribution together, and drops RL
(`grpo-v3-results.md`).

## What changes in the input (`train_qwen.py`, `PROMPT_FORMAT_VERSION = "sft3"`)

- **Two images per step.** A 640x360 thumbnail of the screen before the last action, then the
  current 1280x720 screenshot. Verified on the training image: 880 + 220 image-pad tokens, a
  worst-case step ≈2,020 tokens against `max_length 4096`; a grounding example is 1,028.
- **Effect annotations in the history.** The harness appends ` [no change]` to an action group
  whose screenshot stayed within 0.05% of changed pixels (measured over 2,547 step pairs: the
  threshold that separates caret and hover noise from real change without mislabelling `type`
  steps) and ` -> /path` when the URL path changed. On the corpus: 13.0% of annotated steps are
  `[no change]`, 15.5% move to a new path.
- **Coordinates in `normalized_1000`.** The base model's native convention (ScreenSpot 89% vs 25%
  zero-shot); SFT-1 lost 3.7 pp learning `viewport_abs`. `trajectory.json` stays in pixels; the
  conversion lives only in the builders, so `decode_step` remains the exact inverse of
  `format_action_for_training`.
- **One builder, one fixture.** `build_step_observation` feeds convert, `student_eval`,
  `eval_screenspot` and the bench's vendored copy; `rl/tests/fixtures/prompt_golden.json` is the
  byte-level contract and `prompt_format` is a fingerprint key. The RL lane stays on the one-image
  v2 layout and is retired.

## What changes in the supervision

- **Outcome-checked thoughts** (`make outcome-relabel`): every thought opens with what the
  previous action did, judged from the two consecutive screens ("Nothing changed after that
  click; the menu is still closed"), and every `done()` cites the on-screen evidence for each goal
  criterion. One vision call per window of 12 steps; originals kept as
  `trajectory_pre_outcome.json`; provenance `thought_source: teacher_outcome`. The audit refuses a
  corpus below 90% relabeled (`min_outcome_relabeled_share`).
- **Grounding slice** (`convert --grounding-share 0.25`): every recorded click carries the
  element's role, name and bbox, so 6,602 named clicks (664 of them ≤40x40 px) become
  ScreenSpot-style examples under `GROUNDING_SYSTEM_PROMPT`, small targets oversampled ×3, capped at
  a quarter of the agent examples, joined to the source site's split group.

## What changes in the task distribution

- **QA-voice instructions** (`make instruction-qa-voice`): 144 trajectories whose runs
  demonstrably re-checked now carry the benchmark's test-objective phrasing ("… persists after
  reload. Verify …"), originals in `metadata.instruction_pre_qa_voice`.
- **Verify-style capture** (`make capture-qa-verify`): a `qa_verify` proposer style on the
  session-holding sites; every task changes persistent state and then re-opens or reloads to
  confirm, ending on the screen that shows the checked state, with quotas for pagination depth and
  sign-out/sign-in. The verifier carries a style-keyed rule: a run that stops on the toast without
  re-opening is `success=false`.

## Training recipe

Two epochs (SFT-2's eval loss bottomed at epoch 2), LoRA r64/α128, lr 5e-5, `max_length 4096`,
`max_pixels 1003520`, `freeze_vit true`, per-epoch checkpoints; Modal A100-80GB or H100.

## Measured at the start of the round (2026-09-02, before the capture and relabel passes)

| | corpus v2 (frozen 08-31) | after the free passes |
| --- | --- | --- |
| trainable trajectories | 738 | 738 |
| SFT examples | 9,458 | 11,822 (9,458 agent + 2,364 grounding) |
| done() thoughts citing on-screen evidence | 91.8% | 91.8% |
| done() after a re-check | 16.0% | 16.0% |
| QA-voice instructions | 0 | 144 |
| outcome-relabeled | 0 | in progress |
| `[no change]` / URL-change annotated steps | — | 13.0% / 15.5% |

The freeze numbers (2026-09-05 noon) are appended below when the corpus is frozen.

## Pre-registered decision criteria (written 2026-09-02, before any SFT-3 number exists)

Every row is measured against SFT-2 re-recorded under the matched protocol (window 20, 25-step
cap, v3 executor semantics), never against the published SFT-2 numbers, so harness fixes cannot
masquerade as model gains. The first three rows come from the bench recording alone (n=105 cases,
±9 pp), are cap-independent and need no judge; they are the decisive signal. DDR (n=33 defects,
±16 pp) is reported, expected at 33-42%, but is not a decision criterion.

| check | what it proves | SFT-2 (matched) | pass | fail |
| --- | --- | --- | --- | --- |
| coordinate loops (same point ≥6× per episode) | the no-op signal and the two images are used | 18.1% | ≤ 6% | ≥ 12% |
| premature done (`done` in ≤5 steps) | stopping is evidence-conditioned | 39.0% | ≤ 15% | ≥ 28% |
| success claims absent from the final page text | thoughts describe the screen, not wishes | 41/105 | ≤ 15 | ≥ 30 |
| error verdicts (verdict tier) | execution reaches a judgeable state | 62.9% | ≤ 45% | ≥ 55% |
| verification cluster TC-003/004/005/012 judgeable | the act → reload → check skill exists | 1/28 | ≥ 12/28 | ≤ 5/28 |
| ScreenSpot-300 (grounding prompt, both models) | native coordinates + grounding slice | base 89.0, SFT-1 85.3 | ≥ 87% | ≤ 84% |
| Mind2Web-174 parse failure / canonical form | format not regressed | 0% / 100% | same | any regression |

Definitive success: the loop row and the premature-done row pass together with the error-verdict
row. Definitive failure: either behavioural row lands in its fail band, in which case the
architecture change did not take and no DDR number rescues it. Anything between the bands is
"moved, not decisive", and the write-up says so. A DDR under 27% with the behavioural rows passing
means the errors moved to a different bucket, which the per-case table will show.
