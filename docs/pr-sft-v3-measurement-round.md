# Measurement round: make the evaluation harnesses report what they measure

## Why

Every number this project reports feeds a thesis. Auditing the two harnesses that produce them
turned up a recurring defect: a value that looks measured but is not — a hard-coded zero standing
in for a missing measurement, a cost copied from an external self-report with nothing able to check
it, a cap or a parser deciding an answer instead of the model.

Several of these flattered the fine-tuned checkpoints by penalising the baseline, which is the
dangerous direction for a result of this shape. The branch fixes them, re-measures grounding on a
basis where the arms are genuinely comparable, and writes down the rule that decides what counts as
cost.

## The headline measurement

ScreenSpot, full 1272-sample test split, **all four arms** scored from one endpoint under one
harness commit, each in its own coordinate space, with an identical 2048-token budget:

| arm | overall | 95% CI | icon (575) | text (697) | parse |
| --- | --- | --- | --- | --- | --- |
| base Qwen3.5-9B | 72.2% | [69.7, 74.6] | **73.2%** | 71.4% | 86.2% |
| cua-sft-v2 | 76.3% | [73.9, 78.6] | **67.5%** | 83.5% | 91.8% |
| cua-grpo-v3 | 82.7% | [80.5, 84.7] | 75.0% | 89.1% | 98.7% |
| **cua-sft-v3** | **91.6%** | **[89.9, 93.0]** | **88.0%** | **94.5%** | 100.0% |

Earlier ScreenSpot rows used `--limit 300`, which head-samples in dataset order, and were spread
across three harness commits with no judge hash recorded — at n=300 no arm was separable from any
other.

### SFT-2 did not improve grounding, it traded it

Split by target type, sft-v2 is **worse than the untuned base at icon grounding** — 67.5%
[63.6, 71.2] against 73.2% [69.4, 76.7], barely-overlapping intervals at n=575 — while gaining
12 points on text. Its overall lead over base is correspondingly marginal.

This is the regression the sft-v3 round set out to fix by moving from `viewport_abs` to
`normalized_1000` coordinates. It was previously an inference from a 300-sample run against a
different harness (ScreenSpot 89.0 -> 85.3, icons 80.3 -> 75.4); it is now measured directly.
sft-v3 fixes it decisively: icons 88.0%, text 94.5%.

### GRPO-3 improved grounding, which its task metrics did not predict

`cua-grpo-v3` is recorded in the registry as yielding no measurable gain, on the basis of reward
and task-success curves — its LoRA-B Frobenius norm is 3.1% of sft-v2's, and 752 episodes moved
reward 1.08 -> 0.89. That reading is incomplete. On grounding it gained **76.3% -> 82.7%** over
its own parent, recovering most of sft-v2's icon regression (67.5% -> 75.0%) while keeping the
text gain (83.5% -> 89.1%), at a parse rate of 98.7%.

A run that produced no task-level improvement still measurably sharpened where the model points.
That is a more useful result for the dropped-RL chapter than "nothing happened", and it is only
visible because all four arms were finally scored on one instrument.

## What was wrong, and what it cost

**A token cap that scored the cap.** `eval_screenspot.OpenAIPredictor` hard-coded `max_tokens: 512`
and ignored the configured value. The base model reasons before answering, so it was cut off
mid-thought and its click never arrived: 462 of 1272 samples failed to parse, every one a
truncation, putting it at 54.1%. The fine-tuned checkpoints answer tersely and were untouched, so
the cap penalised exactly one arm of the comparison. The same 512 default was truncating the
Online-Mind2Web base run from a different code path.

**A parser that scored convention-imitation.** Two unambiguous spellings were rejected:
`click(x, y, button=left)` (the `click(` pattern required a closing paren immediately after `y`)
and `click(<x, y>)`, the base model's own form. The student eval's action grammar already refuses
to charge a model for surface deviation; the grounding eval now agrees.

**A cost with its tokens blanked to zero.** `benchmark/verdicts.py` recorded
`UsageRecord(input_tokens=0, output_tokens=0, cost_usd=cost)`, the cost copied verbatim from the
Claude CLI, so `pricing.yaml` could never cross-check it. Measured against a tiny prompt under the
same options: 124,135 cache-write tokens for content of ~1,375 tokens. The verdict tier's cost was
~98% fixed agent-harness overhead rather than judging work, and the CLI over-reports by a
consistent ~2.5× besides. Tokens are now recorded beside the cost, the verdict cache replays the
usage that produced a hit instead of `0.0`, and `benchmark/cost.py` — dead code imported only by
its own test — is wired into the report so GPU seconds become dollars.

**A judge reading the operator's config.** The verdict call did not set `setting_sources=[]`, so
every judgement loaded the local CLAUDE.md, skills and plugins. That is contamination of the
measurement instrument, not merely cost. The dataset_gather side was already clean — every
`ClaudeAgentOptions` there reaches `harden_options`.

**Two ways a judge failure became a model failure.** `verify_trajectory` returns a `success: False`
fallback for a wedged CLI, a budget cap or an unparseable reply, and the eval scored that as the
model failing; it now defers, as a usage limit already did. And `verified_success` read the judge's
raw verdict rather than `final_success`, so a degenerate loop the capture gate refuses could be
counted as a win. Neither fired on the runs on disk, so no recorded number moves.

**Precision penalised by judging failures.** A bug the judge never ruled on counted in the
denominator, so a judging failure lowered precision while the same failure raised DDR. A bug on a
*clean* run is still a false positive and still counts — there is simply no defect to match.

## New capability

**A frontier arm on grounding.** `PREDICTOR_BACKENDS["claude"]` routes image+text through the
logged-in CLI, with `to_claude_content` beside `to_openai_content` so both wire formats render the
same observation. Grounding is the one lane where a frontier comparison is apples-to-apples: single
turn, one screenshot, one click point, no scaffold advantage.

**A per-arm observation format.** An sft-v1/v2 checkpoint, and any GRPO LoRA over one, trained on a
single screenshot; sending it the sft3 two-image turn measures the prompt rather than the model.
`--observation v2|v3` mirrors the coordinate space, which is already per-arm for the same reason,
and joins the harness fingerprint so a mismatched pair cannot pass the parity check.

**A GRPO-3 serve profile.** GRPO-3 is a LoRA over the *merged* sft-v2 weights, so it cannot be
stacked on the raw base alongside sft-v2 and sft-v3; `rl_config_serve_grpo.yaml` merges first and
serves the adapter on top.

## What counts as cost

`docs/decisions/hybrid-cost-and-the-product-gap.md` leads with the rule rather than the number,
because many more numbers are coming:

> Count what a user pays to run the tool. Do not count what only we pay to grade it.
> Would a customer's run still have to do this? If yes it is cost; if no it is apparatus.

Explore, plan, execute and **conclude** are in. Matching a reported bug against the injected defect
catalogue is out — a real user has no catalogue. Two consequences that are easy to get backwards:
grading is not a product phase even though it costs real money (~$0.45/run, tracked in
`judgements.jsonl`, never in `total_cost_usd`), and concluding **is** one even though our
implementation of it happens to live in the benchmark package.

The doc also records that the shipped product does not implement the architecture it is measured
under: `LocalModelAgentAdapter` routes every phase to the local model, including the structuring
step that turns a run into verdicts and bug reports, so the model grades its own homework. That
dates to the file's first commit (2026-03-13, *"local models integration (untested)"*), not to this
work — but it means the hybrid cost figure composes a measured local execution cost with a measured
frontier verdict cost rather than coming from one instrumented end-to-end run.

## Registry

`experiments/results.csv` grows from 181 to 221 rows and `models.csv` from 15 to 17. Adds the
Sonnet economy sweep (measured 2026-09-02, never recorded), both matched-harness arms, the
`cua-sft-v3` and `corpus-v3` rows, a derived GPU cost per case for every local arm, the
end-to-end cost model, and the ScreenSpot run above.

## Verification

`make test-eval`, `make test-student-eval`, `make test-rl-unit` (143/143) and the vetty unit suite
(1046) all pass. The ScreenSpot numbers come from a live rented GPU; both rentals were destroyed
and confirmed gone, and the account shows zero instances.

## Not in this branch

Online-Mind2Web at n=300 for base / sft-v2 / grpo-v3, and the GRPO-3 ScreenSpot and QA-bench arms.
Those runs were in flight when the session was interrupted; the rentals were reclaimed rather than
left burning. The Online-Mind2Web numbers currently in the registry remain **not comparable across
arms** — the +14.7pp figure pairs two different harness commits, and the only same-harness pairing
reverses its sign — which is why re-running all arms on one harness at n=300 is the outstanding
work rather than an optional extra.
