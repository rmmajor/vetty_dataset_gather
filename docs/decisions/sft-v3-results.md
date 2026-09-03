# SFT v3 — measured results (2026-09-03)

Adapter: `models/cua-sft-v3`, the epoch-1 checkpoint (step 1,400). Epoch 2 was trained and rejected:
eval loss rose 0.5587 → 0.5791 while train loss kept falling to 0.464, so epoch 1 is the minimum.
Trained on Vast A100 SXM4 80GB, 14h15m wall, ~$15.50.

## QA benchmark — economy sweep, all four arms

Same shape everywhere: 7 runs (1 clean + 6 economy builds) x 15 pinned cases = 105 case-runs,
33 catalogue defects, verdict and judge on `claude-sonnet-5`.

| arm | DDR strict | precision | FPR | error verdicts | passed | failed |
| --- | --- | --- | --- | --- | --- | --- |
| base Qwen3.5-9B | 0.0% | — (0 reports) | 0.00 | 73.3% | 0 | 0 |
| sft-v1 | 27.3% | 71.4% | 1.00 | 60.0% | 25 | 17 |
| sft-v2 | 21.2% | 100% (9/9) | 0.00 | 62.9% | 29 | 9 |
| **sft-v3** | **21.2%** | **100% (7/7)** | **0.00** | **54.3%** | **31** | **13** |

## Token economics (the executor, per case)

The executor is our own served model, so these are compute, not spend. Recorded because the round's
claim is offloading QA execution to a small model without significant regression, and the cost of
that offload is exactly these numbers.

| arm | input tokens / case | output tokens / case | turns / case | verdict-tier $ / run |
| --- | --- | --- | --- | --- |
| base | 19,386 | 3,617 | 15.0 | $1.32 |
| sft-v1 | 11,920 | 517 | 9.0 | $10.49 |
| sft-v2 | 12,315 | 538 | 9.2 | $10.68 |
| sft-v3 | 18,797 | 746 | 11.3 | $0.73 |

**The verdict-tier dollar column is not comparable across arms.** sft-v3 ran on vetty `428c062`
while the others ran on `409810f`, and cost-reduction work landed in the bench between them. It is a
harness difference, not a model property, and must not be reported as an efficiency win.

**The token columns are comparable and they cut against a naive reading.** sft-v3 consumes ~53% more
input tokens per case than sft-v2 (18,797 vs 12,315) — the second screenshot is doing exactly what it
was designed to do, and it is not free. Output tokens and turns also rise (746 vs 538, 11.3 vs 9.2),
partly the 25-step cap versus 15. The base model's 3,617 output tokens per case is the pathological
case: it narrates without ever emitting parseable actions.

## Against the pre-registered bands

`sft-v3-corpus-round.md` fixed these before any number existed:

| check | predicted pass | predicted fail | actual | verdict |
| --- | --- | --- | --- | --- |
| error verdicts | ≤ 45% | ≥ 55% | 54.3% | **did not pass** (just inside the fail band's edge) |
| DDR strict | 33-42% expected | < 27% contradicts | 21.2% | **did not pass** |

The round did not deliver the predicted improvement. Error verdicts fell 62.9% → 54.3%, a real move
in the intended direction, but well short of the ≤45% the plan required, and DDR did not move at all.

## What did and did not shift, by defect class

| class | sft-v2 | sft-v3 |
| --- | --- | --- |
| action | 0% | **20%** |
| labeling | 20% | **40%** |
| data | 20% | **40%** |
| state_dependent | 33% | 33% |
| validation | 20% | 20% |
| navigation | 40% | **0%** |
| silent_failure | 20% | **0%** |

The `action` class — "the button does nothing" — moved off zero, which is the class the
verify-after-acting work targeted. But `navigation` and `silent_failure` both fell to zero, and the
net is exactly the same 7 of 33 defects. With n=33 and a ±16pp interval, none of these per-class
moves is individually resolvable; the honest reading is that the detection total did not change and
its composition reshuffled within noise.

## Not yet measured

- The three decisive behavioural rows the plan named (coordinate loops, premature `done`, success
  claims absent from the final page) are computable from the recorded episodes on disk without
  further spend, and would say whether the architecture change took hold even though DDR did not.
- The matched sft-v2 re-record under the v3 protocol (window 20, 25-step cap) reached 2 of 6 builds
  before being stopped. Without it, sft-v2's numbers here come from its original protocol
  (window 10, 15-step cap), so part of the error-rate gap may be the extra 10 steps rather than the model.
- ScreenSpot and Online-Mind2Web for sft-v3.
