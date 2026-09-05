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

ScreenSpot, full 1272-sample test split, **all five arms shown the identical native-resolution
image**, each scored in its own coordinate convention, with an identical 2048-token budget:

| arm | overall | 95% CI | icon (575) | text (697) | parse |
| --- | --- | --- | --- | --- | --- |
| claude-sonnet-5 | 68.3% | [65.7, 70.8] | 65.9% | 70.3% | 100.0% |
| base Qwen3.5-9B | 72.2% | [69.7, 74.6] | 73.2% | 71.4% | 86.2% |
| cua-sft-v2 | 85.4% | [83.3, 87.2] | 77.4% | 92.0% | 98.4% |
| cua-grpo-v3 | 85.1% | [83.0, 86.9] | 77.2% | 91.5% | 98.3% |
| **cua-sft-v3** | **91.6%** | **[89.9, 93.0]** | **88.0%** | **94.5%** | 100.0% |

Earlier ScreenSpot rows used `--limit 300`, which head-samples in dataset order, and were spread
across three harness commits with no judge hash recorded — at n=300 no arm was separable from any
other.

### The image every arm saw is a harness setting, not a property of its coordinate space

`prepare_image` was a method on `CoordSpace`, and the `viewport_abs` implementation resized the
input to 1280x720. Declaring a coordinate space therefore silently changed the picture: sft-v2 and
grpo-v3 were scored on squashed screenshots while base, sft-v3 and the frontier arm saw the source.
That is not a comparison between models, it is a comparison between inputs.

Image preparation is now one harness-wide setting (`SCREENSPOT_IMAGE_SIZE`, default `native`),
recorded in every result artifact as `prepared_image_size`, and `prepare_image` is deleted from the
coordinate spaces entirely so the coupling cannot come back. Correcting it moves sft-v2 from 76.3%
to **85.4%** and grpo-v3 from 82.7% to **85.1%**; the superseded rows carry the reason.

### SFT-2 improved grounding on both target types

The earlier reading — that sft-v2 bought text accuracy by giving up icon accuracy, 67.5% against
the base's 73.2% — was an artefact of the squashed input. On the source images sft-v2 scores
**77.4% on icons, above** the base's 73.2%, alongside 92.0% on text: +13 points overall with no
trade. sft-v3 extends it rather than repairing it (icons 88.0%, text 94.5%).

### GRPO-3 gained nothing, on grounding as on everything else

A previous revision of this description claimed grounding as the one lane where grpo-v3 measurably
improved on its parent, 76.3% -> 82.7%. Both arms were squashed at the time. On the source images
they are **85.1% and 85.4%**, a 0.3-point difference with fully overlapping intervals — what the
6.4-point lead measured was grpo-v3's greater robustness to the distortion, not its grounding.

`cua-grpo-v3` therefore has no measured advantage on any lane, which is the reading its LoRA-B
Frobenius norm (3.1% of sft-v2's) and its reward curve (1.08 -> 0.89 over 752 episodes) always
supported. The dropped-RL chapter keeps its original conclusion.

### The frontier arm is the weakest grounder here, and that is not a bug

`claude-sonnet-5` scores **68.3%**, below the untuned base. Four checks before believing it:

- **Not a coordinate-frame artefact.** Sonnet is on the high-resolution tier (long edge 2576px,
  4784 visual tokens), which downsizes only 74 of the 1272 images; rescaling answers out of that
  frame gives 67.8%, marginally worse, and out of the standard 1568px frame 17.1%. One answer in
  1272 falls outside the image, and the same answers read as `normalized_1000` score 4.4%.
- **Misses are wrong-element, not imprecise.** The median gap to the target box is 8.7% of the
  image diagonal; only 13.7% land within 50px.
- **Paired on the identical split**, sft-v3 hits where sonnet misses 335 times against 39 the other
  way. Against base it is 260 to 210 (McNemar z=2.31, p≈0.02), and base's 86.2% parse rate flatters
  sonnet: among samples base actually answered, base scores 83.8% to sonnet's 69.6%.
- **The weakness is by source, not resolution**: 90.7% windows and 89.5% macos — matching the best
  local arm — against 59.9% android, 62.4% ios, 42.7% gitlab.

The asymmetry is one of purpose. ScreenSpot measures a model's native output mode for the Qwen-VL
line, which reports it in its own model cards (Qwen3-VL-8B 94.4%); no Anthropic model appears on a
ScreenSpot leaderboard, because Claude's computer use is delivered through a dedicated tool
interface rather than by emitting coordinates as text, and Anthropic's vision documentation states
plainly that its coordinate outputs are approximate. **This row is evidence that grounding is the
right thing to offload, not that the frontier model is weak.** The frontier comparison that carries
the economic claim is the QA bench, where Sonnet runs inside the full agent with native tooling.

Asterisk carried on every row: each arm is measured under its own coordinate convention, because
requiring one convention scores translation into an unfamiliar output space rather than grounding.
The 0-1000 prompt cost sonnet 32 points of measured accuracy for exactly that reason.

## The QA bench, every arm on one harness

Economy sweep, 7 runs x 15 cases, 33 catalogue defects, harness 428c062:

| arm | DDR | 95% CI | error verdicts | precision |
| --- | --- | --- | --- | --- |
| base (matched) | 0.0% | [0.0, 10.4] | 66.7% | — |
| cua-sft-v2 (matched) | 9.1% | [3.1, 23.6] | 61.9% | 100% (3) |
| cua-sft-v3 | 21.2% | [10.7, 37.7] | 54.3% | 100% (7) |
| **cua-grpo-v3** | **24.2%** | [12.8, 41.0] | 57.1% | 100% (11) |
| claude-sonnet-5 (frontier) | 66.7% | [49.6, 80.3] | 2.9% | 77.1% (48) |

**sft-v2 has two legitimate DDR numbers and the choice between them changes the story.** Its
original run (harness 409810f, 15-step cap, history window 10) detected 7 of 33 defects — 21.2%.
Its matched re-record (428c062, 25-step cap, window 20) detected 3 — 9.1%. Both are real and
independently counted from the bug reports and judgements; the table above uses the matched run
because it is the only sft-v2 recording that shares a harness, step cap, history window and
observation format with the arms it is compared against.

That makes the defensible claim narrower than "sft-v3 doubled detection". Under matched conditions
sft-v3 detects 7 defects to sft-v2's 3; under sft-v2's own original protocol sft-v2 also detected 7.
At n=33 with intervals near +/-15 points, neither pairing separates the two checkpoints. What the
lane does establish is that every fine-tuned arm detects defects the base never does: base 0 of 33,
on both harnesses.

grpo-v3's 8 of 33 is the highest of any local arm, and defect detection is now the ONLY lane where
the run recorded as yielding no measurable gain shows one. An earlier revision counted grounding as
a second; that reading did not survive re-measurement on unsquashed images. This is a claim about
grpo-v3 versus its own parent under one harness, not a ranking against sft-v3, and at n=33 the
interval spans roughly +/-15 points.

## Online-Mind2Web, all four arms on the full 300

One harness commit, one token budget, each arm prompted in the observation format it was trained
for and scored in its own coordinate space:

| arm | scored | 95% CI | all-episode | n scored |
| --- | --- | --- | --- | --- |
| base Qwen3.5-9B | 3.18% | [1.4, 7.2] | 1.67% | 157 |
| cua-sft-v2 | **16.44%** | [11.3, 23.3] | 8.00% | 146 |
| cua-sft-v3 | 11.11% | [7.0, 17.3] | 5.33% | 144 |
| cua-grpo-v3 | 13.38% | [8.7, 20.0] | 6.69% | 142 |

**Every fine-tuned arm is clearly above the base, and none is separable from the others.** The three
checkpoints' intervals overlap almost entirely, so at n≈145 the round-over-round differences on
browsing are noise. sft-v2 has the highest point estimate; that is not the same as being better.

| arm | easy | medium | hard |
| --- | --- | --- | --- |
| base | 12.5% | **0.0%** | **0.0%** |
| cua-sft-v2 | 47.4% | 4.9% | 6.4% |
| cua-sft-v3 | 23.7% | 10.0% | 2.2% |
| cua-grpo-v3 | 32.4% | 4.8% | 9.5% |

The base solves nothing outside easy tasks across 117 scored medium and hard episodes. Every
fine-tuned arm solves some of every level.

### The outcome breakdown carries as much as the success rate

| outcome | base | sft-v2 | sft-v3 | grpo-v3 |
| --- | --- | --- | --- | --- |
| completed | 8 | 56 | 46 | 53 |
| step_cap | 74 | 84 | 94 | 79 |
| parse_failure_abort | **46** | 0 | 0 | 2 |
| handoff | 26 | 5 | 3 | 5 |
| site_blocked | 126 | 135 | 136 | 136 |

The base fails to emit a parseable action on 46 episodes; every fine-tuned checkpoint is at 0 or 2.
Its `parse_failure_rate` is 12.8% and its `canonical_form_rate` 20.1%, against 0.0% and 100% for the
checkpoints — the base has only ever seen the action grammar *described* in a system prompt. Folding
that into the success rate would conflate "can it browse" with "can it imitate our convention",
which is why the parser is tolerant and canonical form is its own metric.

Block rates land within one point across all four arms, so every arm hit the same walls. 143 of 300
tasks were blocked or unreachable from this egress, which is why both denominators are registered
rather than only the flattering one.

These rows supersede the n=33 sft-v2 row on the 50-task set and the n=34/n=164 base rows, none of
which shared a harness with anything they were compared against.

## What the three lanes say together

| | ScreenSpot | QA bench DDR | Online-Mind2Web |
| --- | --- | --- | --- |
| claude-sonnet-5 | 68.3% | **66.7%** | cited, not run |
| base | 72.2% | 0.0% | 3.18% |
| cua-sft-v2 | 85.4% | 9.1% | **16.44%** |
| cua-sft-v3 | **91.6%** | 21.2% | 11.11% |
| cua-grpo-v3 | 85.1% | **24.2%** | 13.38% |

Each round bought what it targeted and did not generalise. SFT on the QA corpus is what buys
grounding: 72.2% -> 85.4% at sft-v2 and -> 91.6% at sft-v3, icons 73.2% -> 77.4% -> 88.0%. The sft3
round additionally aimed at self-verification and more than doubled defect detection, while browsing
did not improve and its point estimate moved the wrong way. The GRPO run gained on defect detection
alone; on grounding it lands within noise of its own parent.

The one difference that is unambiguous everywhere is base versus fine-tuned. Between checkpoints,
only the grounding lane separates them, and only sft-v3 from the other two; on browsing at n≈145
nothing does. Against the frontier arm, every fine-tuned checkpoint grounds better and every one
detects far fewer defects — which is the shape the offloading argument predicts, and the reason the
economic claim rests on the QA bench rather than on grounding alone.

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

**An image the coordinate space silently resized.** `prepare_image` was a method on
`CoordSpace`, and the `viewport_abs` implementation resized the input to 1280x720. Declaring a
coordinate space therefore changed the picture: sft-v2 and grpo-v3 were scored on squashed
screenshots while base, sft-v3 and the frontier arm saw the source, so the cross-arm table compared
inputs rather than models. Preparation is now one harness-wide setting recorded in every artifact as
`prepared_image_size`, and `prepare_image` is deleted from the coordinate spaces so the coupling
cannot return. Correcting it moved sft-v2 76.3% -> 85.4% and grpo-v3 82.7% -> 85.1%, and retired two
conclusions this description previously drew.

**A frontier arm asked for the wrong coordinate convention.** The Claude predictor inherited
`GROUNDING_SYSTEM_PROMPT`, which specifies the 0-1000 space Qwen emits natively. The model
half-complied and half-reverted to pixels, so no single mapping scored it and the value was bounded
only to [36.5, 51.7]. Anthropic's own guidance is explicit — *"Claude does not work well when you ask
for normalized coordinates... Always ask for pixel coordinates"* — and asked in pixels it is
unambiguous: 100% parse, 1 of 1272 answers outside the image, 68.3%. The withdrawn row stays as
`superseded` with its reasoning.

**A rate divided by the wrong denominator.** Seven QA-bench DDR rows carried `n=105`, the case-run
count, but DDR is the share of the 33 catalogue defects detected. The values were right and the
Wilson intervals were too narrow: sft-v2 read [4.9, 16.1] and is [3.1, 23.6]. Once corrected, no
local arm separates from another on DDR.

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
same observation. Each arm is asked for the convention it
natively emits and scored in that space, because requiring one shared convention measures
translation rather than grounding. Single turn, one screenshot, one click point, no scaffold on
either side — which is also the limit of the claim: `docs/decisions/why-a-9b-outgrounds-the-frontier-model.md`
records why GUI-tuned Qwen checkpoints lead here (grounding is a named Qwen3-VL training objective
with purpose-built point-annotation data and GUI screenshots spanning ScreenSpot's three domains),
why the ordering reverses on ScreenSpot-Pro where published Claude figures allow cropping and
planners, and the contamination limitation that ScreenSpot predates Qwen3-VL.

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

`experiments/results.csv` grows from 181 to 271 rows and `models.csv` from 15 to 17. Adds the
five-arm ScreenSpot run on native images, all four local Online-Mind2Web arms on the full 300,
grpo-v3 on the QA bench, the Sonnet economy sweep that was measured on 2026-09-02 and never recorded,
both matched-harness arms, the `cua-sft-v3` and `corpus-v3` rows, a derived GPU cost per case, and
the end-to-end cost model. Superseded rows keep their original values and carry the reason they were
retired, so the corrections are auditable rather than overwritten.

The frontier QA-bench rows carry a verification note: re-judging that arm under 428c062 reproduced its
numbers exactly (DDR 0.6667, precision 0.7708, FPR 0.0), so it is comparable to the 428c062 local arms
despite being recorded at 409810f — the v3 executor changes touch only the CUA path, which the
claude-sdk backbone does not import.

## Verification

`make test-eval`, `make test-student-eval`, `make test-rl-unit` (143/143) and the vetty unit suite
(1046) all pass. Every GPU rental was destroyed and confirmed gone; the account shows zero instances
and the Modal app is stopped with zero containers.

## Not in this branch

A scaffolded frontier grounding arm. Anthropic's guidance for fine targets is crop-and-refine, and
the ScreenSpot-Pro leaderboard entries permit exactly that, so a two-call cropped run would give the
frontier model's upper bound. It is deliberately absent: it is not the protocol the local arms are
held to, and mixing them would repeat the unequal-conditions error this branch exists to fix. The
single-turn number answers what offloading costs; the scaffolded one answers a different question and
can be added as its own row.

Everything measurable on the three lanes is now here, grpo-v3's Online-Mind2Web arm included.

https://claude.ai/code/session_01Dkkp5S3mHoQ3GZcfkNFwkX
