# grpo-v3: the run completed and the model did not improve

Written 2026-09-02, immediately after the run finished. grpo-v2 (2026-08-27) was a *dead* run —
fabricated rewards, no reward contrast, weights that never moved. grpo-v3 fixed every one of those
mechanisms and still produced no measurable behavioural gain. That distinction is the point of this
document: the pipeline is now sound, so the remaining causes are scale and task design, and those
are the things worth spending on next.

## What ran

| | |
| --- | --- |
| adapter in | sft-v2 (`checkpoint-3189`, LoRA r64/α128, 738 trajectories / 9,458 examples) |
| tasks | 25, from the measured frontier band (`rl/frontier_selection.json`) |
| scale | 50 steps × (2 prompts × 8 generations) = **800 episodes**, 752 recorded |
| hyperparameters | lr 5e-6, `max_turns` 36, `epsilon_high` 0.28, `dynamic_sample` false, `async_generate` false |
| box | 2× A100 SXM4, ~10.2 h wall clock, ~$25 including failed attempts |
| final adapter | `B_frobenius` 0.5557 = **3.12%** of sft-v2's norm |

## The result

Reward and success **drifted down** across the run; step efficiency did not improve.

| quarter | n | reward | success | ui_steps | voluntary `done` |
| --- | --- | --- | --- | --- | --- |
| q1 (steps ~1-12) | 188 | +1.078 | 59.6% | 12.2 | 86% |
| q2 | 188 | +0.951 | 53.2% | 13.4 | 82% |
| q3 | 188 | +0.888 | 50.0% | 13.9 | 82% |
| q4 (steps ~38-50) | 188 | +0.889 | 52.1% | 13.1 | 87% |

The q1→q4 decline is ~1.5σ at n=188, so it is **not** a significant regression — but there is no
gain anywhere, and an apparent efficiency improvement at step 20 (ui_steps 14.4 → 11.6) did not
survive to step 25.

## What was genuinely fixed, and is worth keeping

These are the grpo-v2 killers, all confirmed absent over 752 episodes:

- **0 missing rewards** (grpo-v2: 29 of 143 episodes trained on a fabricated `0.0`).
- **1 zero-variance task of 14 observed** (grpo-v2: 10 of 12 groups had no reward contrast).
- **2 oracle errors**, both transient reads that retried.
- **Loop guard collapsed**: 82-87% of episodes terminate voluntarily (grpo-v2: 68% ended on the
  loop guard).
- **The weights moved.** The B-norm curve is a converging plateau, not grpo-v2's flat floor:

  | step | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 50 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `B_frobenius` | 0.191 | 0.313 | 0.402 | 0.466 | 0.508 | 0.532 | 0.546 | 0.556 |
  | Δ | — | +64% | +29% | +16% | +9% | +5% | +2.8% | — |

## Why it did not improve

Ranked by how much each plausibly cost, most first.

### 1. The run is an order of magnitude too small

50 steps × 16 episodes moved the LoRA to 3.12% of the SFT adapter's norm, against a KL anchor
pinned to that same SFT policy and a learning rate of 5e-6. A policy change that small is not
expected to show up in end-to-end task success. The B-norm curve had already flattened
(+2.8% over the last five steps), so more steps *at these settings* buys progressively less —
the ceiling is set by lr and the KL weight, not by step count alone.

### 2. The task band was selected on two samples per task

`rl/frontier_selection.json` was built with `min_scored=2`: every one of the 25 selected tasks had
`pass_rate` exactly 0.5 from a **1-1 split on two episodes**. Total evidence behind the whole
selection was 52 episodes. A task whose true pass rate is 0.2 or 0.8 splits 1-1 about 32% of the
time, so a large fraction of the band was never in the band. Confirmed in flight:
`calculatornet-bmr` came back 7/8 in the validation run. Groups that are near-degenerate waste
generations without contributing gradient.

### 3. Several oracles do not reward the behaviour the run exists to teach

Found by adversarial review before launch, only partly fixable in the time available:

- **All 8 "make/publish" tasks** check only that a published URL exists containing the title
  string. Python syntax and 1-day expiry (`dpaste`), Markdown and 1-month expiry (`privatebin`),
  the byline and bulleted list (`telegra.ph`), the numbered list (`rentry`) are **not verified**.
  The policy is therefore rewarded for skipping exactly the option-setting steps that constitute
  the multi-step behaviour we want.
- **`omnicalculator__frontier-speed-convert`** was satisfiable by *typing* the answer
  (`text_visible "160.9"` matches the snapshot's rendered input values). Dropped before the run;
  it cannot be fixed by `page_text_visible` because Omni renders results inside inputs.
- **Milestones are a constant, not an ordering signal.** Several were true on the start page
  (`archive.org` `url_contains "archive.org"`; `uuidgenerator` `text_visible "Version 4"`;
  `omnicalculator` `text_visible "100"` matching "1000"/"100%"). In the validation run the
  milestone was constant across all 8 samples in 3 of 4 groups, so the `progress` component
  contributed no within-group ordering — it only raised the do-nothing floor.

### 4. The reward initially taught the policy never to stop

`step_cap` cost 0.2 while `false_done` cost 0.4, so below ~50% confidence *not terminating*
strictly dominated. Visible in the validation data: bahn's only successes came from `step_cap` and
`max_turns` while every `done()` scored negative. Fixed before the run
(`step_cap` 0.45 > `false_done` 0.35 > `loop_guard` 0.3), and voluntary termination did rise to
82-87% — but this means the run's reward function differs from every measurement taken before it.

### 5. We flew without trainer telemetry

`rl_config.yaml` sets no `logging_steps`, so ms-swift used the HF default of 500 — with
`max_steps: 50` **no `{'loss': …, 'reward': …}` line was ever emitted**. `frac_reward_zero_std`,
`kl`, `grad_norm` and `clip_ratio` were unavailable for the entire run; every number in this
document was reconstructed from the env service's episode records. That is a serious observability
gap: the trainer-side signal that would have distinguished "no gradient" from "gradient, small
step" was simply not collected.

### 6. Infrastructure tax, ~40% of the night's spend

- **Three consecutive rentals of the same dead host.** `select_cheapest_allowed` is deterministic
  cheapest-first, so "retry" re-rented the identical machine that never became SSH-ready. Fixed
  with a 6-hour offer blocklist.
- **~85 minutes of base-weights download per attempt**, three times. `HF_HUB_ENABLE_HF_TRANSFER`
  is deprecated and silently ignored by current `huggingface_hub`; switched to
  `HF_XET_HIGH_PERFORMANCE`, which helped but did not eliminate the tax.
- **Three simultaneous kills of the harness-tracked background tasks** (01:57, 04:50, ~05:25),
  each severing the GPU from the browser env mid-run. The GPU job survives (it runs detached), so
  the recovery is: restart the env service, re-establish the reverse tunnel, re-attach a
  supervisor. Env service and tunnel now run detached for this reason.
- **Two teardown defects found the hard way.** `collect` never ran on a failed run, and the
  periodic adapter pull targeted `rl_output/latest/` — a path that only exists *after* training
  ends — so every mid-run checkpoint would have died with the box. Both fixed. A third,
  `state.owner_alive()` treating `os.kill(1, 0)`'s `PermissionError` as "alive", meant the
  supervisor's `pid: 1` "abandoned" sentinel made `cleanup` skip the box; it kept billing after the
  supervisor announced release. **Not yet fixed** — see below.

## What to change before spending on grpo-v4

Ordered by expected value per dollar.

1. **Re-measure the band at k=8 and widen it to 40-60 tasks.** The current selection rests on 52
   episodes. `make rl-candidates` plus a k=8 sweep is a few dollars and removes the single largest
   source of wasted generations. Do not describe a k=2 selection as a measured frontier.
2. **Set `logging_steps: 1`.** Free, and it restores `frac_reward_zero_std`, `kl`, `grad_norm` and
   `reward_std` — the metrics that make a kill decision possible at step 5 rather than step 25.
3. **Tighten the publish oracles to check the options.** Expiry, syntax, format and byline are all
   assertable from the published page. Until they are, the run cannot reward multi-step behaviour.
4. **Require every milestone to be false on the start page**, or drop it. A milestone that is true
   at step 0 is a constant offset that cancels in GRPO.
5. **Raise the policy-change budget.** Either more steps (200+), a higher learning rate, or a lower
   KL weight — the current combination is explicitly tuned to keep the policy near sft-v2, which is
   the opposite of what a first RL run needs. Prefer changing one of the three at a time.
6. **Cache the merged weights.** ~85 minutes of paid GPU per attempt goes to re-downloading the
   19 GB base and re-merging. A provider snapshot or a persistent volume removes it entirely and
   pays for itself in two runs.
7. **Fix `state.owner_alive()`**: use an explicit `pid: null` (or an `abandoned: true` flag) to mean
   "no owner", instead of overloading pid 1, which is always alive. Then fold the supervisor loop
   into `remote_train` so a killed wrapper cannot leave a billing box.
8. **Evaluate more than the final checkpoint.** All 10 checkpoints are retained; a paired oracle
   re-run at steps 10/25/50 would show whether the policy moved at all before it plateaued.

## Reproducing the numbers here

All 10 checkpoints and 752 episode records are on disk (`rl_runs/`, gitignored). The B-norm series
is `python -m rl.analysis bnorm <checkpoint>/adapter_model.safetensors --ref <sft-v2>`; the reward
quarters are computed from `rl_runs/**/episode.json` sorted by mtime.
