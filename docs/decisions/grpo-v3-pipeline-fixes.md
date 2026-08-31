# GRPO v3 — fixing the pipeline that produced a dead run (2026-08-31)

`grpo-v2` finished, produced an adapter, and taught the policy nothing: its LoRA-B Frobenius norm is
**0.279 against sft-v1's 14.345 — 1.9%**. This records why, what changed, and how each change was
proven before spending on another run.

## Why grpo-v2 was dead

Measured over its 143 recorded episodes (`rl_runs/grpo-v2/`):

| symptom | measurement |
| --- | --- |
| no contrast inside a group | 8 successes total, from **2 of 12 tasks**; 10 tasks all-fail, rewards differing only by penalty noise (−0.1 vs 0.0) |
| fabricated rewards | **29/143 (20%)** episodes had `reward: None` (deferred verdict) and trained as `0.0` via `float(reward or 0.0)` |
| wrong task pool | 23/27 train tasks `hard`; bhphoto spent 12/12 episodes clicking a Cloudflare wall |
| loop guard as the default ending | **97/143 (68%)** episodes ended `loop_guard` at a flat −0.1 |
| no oracles | 0 of 106 suite tasks had a `success_check`, so every reward waited on the LLM judge |
| smoke-test scale | 15 steps, G=4, `dynamic_sample: false` |

GRPO's gradient is the spread of rewards *within* a task's group. With one distinct reward per group
there is no advantage and no learning, which is exactly what the adapter norm shows.

## What changed

**Rewards can no longer be fabricated.** `rl.trainer.core.require_reward` raises `MissingRewardError`
on a `None`; `BrowserScheduler` calls it for every episode (an `env_error` episode keeps an explicit,
labelled `0.0`). `rl/service.py` and `rl.rollout` refuse to start under `--verifier-mode never` unless
every task carries a `success_check`, so a judge-free run cannot silently produce verdictless episodes.
`BrowserEnv.close` logs its failures instead of swallowing them.

**Failed groups order themselves.** A task may declare `milestones: [{name, check, weight}]`, evaluated
by the same oracle context at close; the new `progress` reward component pays their weighted share
**only when the episode failed**, so "reached the results page" outranks "never left home" without
touching how successes are scored. The `false_done` penalty went 0.2 → 0.4 to fix an inversion where a
failed `done()` (0.0) scored above an honest attempt that tripped the loop guard (−0.1). Failure order
is now: capped attempt > loop guard > false done.

**Oracles can see the answer.** The driver's `/snapshot` renders only headings, alerts and interactive
element names, so a computed result in an ordinary element was invisible to `text_visible` — every
"is the answer 36 on screen" oracle would have silently scored 0. The driver gained `POST /page_text`
(the page's rendered `innerText`, all frames, 40 kB cap) and `checks.build_context` now searches
snapshot + page text. It is never part of an agent observation, so no training input changes.

**Train tasks come from measurement, not taste.** `repo_pipeline/public_sites_actions_frontier.json`
holds short (≤12 expected actions), idempotent, oracle-checked tasks; its `_config` excludes the
bot-walled sites grpo-v2 wasted itself on and **every domain in the archived Online-Mind2Web eval
sample**, so the benchmark cannot leak into training even at domain level. `make rl-candidates` merges
it with base-suite train tasks that have oracles (dropping those that do not), then a pass@8 sweep
scores the pool oracle-only and `rl.analysis select-frontier` keeps the tasks in the 20–80% band.
`rl_train.py make-dataset --task-ids-file` / `modal_grpo.py --task-ids-file` consume that selection.

**Scale and diagnostics.** `num_generations 8`, `temperature 1.0`, `lr 5e-6`, `max_turns 24`,
`max_steps 50`, `dynamic_sample: true` + `max_resample_times 3`, `epsilon_high 0.28` (clip-higher).
`rl/analysis.py` reports per-task pass@k, reward histograms and the zero-variance-group fraction, and
`bnorm --ref` is the dead-run detector: **B-norm under 10% of the SFT adapter's after ~10 steps means
the run is dead — stop it.** ms-swift also logs `frac_reward_zero_std` per step, which is the same
signal live.

**Eval harness: `navigate` is legal.** `navigate("…")` is in the training system prompt and appears
151 times in the corpus, but `student_eval.py` had no such verb, so a correct recovery action on a
blocked page was scored a parse failure. It now parses and dispatches to the driver's `/navigate`,
and `harness_fingerprint` carries `navigate_enabled` so earlier runs are not compared across the
change.

## How each fix was proven, cheapest first

| rung | cost | result |
| --- | --- | --- |
| unit + harness self-tests | $0 | 95/95 `rl.tests`; student-eval smoke PASS; `remote_train --smoke` ALL PASS; Docker `rl.smoke` PASS |
| candidate build + oracle gate | $0 | 33 candidates, disjointness clean; a suite with one oracle stripped is refused by name under `--verifier-mode never` |
| Modal GRPO smoke (new config) | ~$1.5 | one full step, adapter written, rc=0 — **ms-swift 4.5.2 accepts `dynamic_sample`, `max_resample_times`, `epsilon_high`**; degenerate groups resample and then train rather than hang |
| Modal GRPO smoke, verdictless env | ~$1.5 | `MissingRewardError: episode … finished without a reward` → `PHASE FAILED`, rc=1. Fabricated zeros are structurally impossible |
| pass@8 frontier sweep (merged sft-v1) | see below | the measured band that selects the train set |

An earlier smoke died with `StopIteration` in the trainer's dataloader: under the new
`train_max_expected_actions: 12`, the stale suite's train split held a single task, too few to fill a
generation. That is the intended failure — an empty pool now stops the run instead of quietly training
on nothing.

## Standing rules this leaves behind

- **Check the first checkpoint before paying for the rest of the run.** `save_steps: 5`, so pull
  `<run>/rl_output/checkpoint-5` and compare it against the SFT adapter the run started from:
  `python -m rl.analysis bnorm <checkpoint>/adapter_model.safetensors --ref models/cua-sft-v1/adapter_model.safetensors`.
  It prints `B_ratio` and `dead_run`. Under 10% means the weights did not move and no further steps
  will help — kill the run. grpo-v2 measured 0.279 against 14.345, i.e. 1.9%, and was allowed to
  finish anyway; that is the mistake this check exists to prevent. It costs nothing and needs no eval.
- A run whose reward can be `None` is not allowed to train; prove it with the verdictless smoke after
  any change to the reward path.
- Never launch a training run before the pass@k band shows ≥15 tasks with mixed outcomes.
- Check `frac_reward_zero_std` and the LoRA-B norm early; kill a dead run instead of paying it out.
- Serve a GRPO adapter on the **merged SFT weights**, never on raw `Qwen/Qwen3.5-9B`.
