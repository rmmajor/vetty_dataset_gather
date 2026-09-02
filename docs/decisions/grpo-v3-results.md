# GRPO v3 — alive, and flat (2026-09-02)

Operator's run log, written as the run finished: `docs/decisions/grpo-v3-run-results.md`.

The third GRPO run and the last one for the paper. It fixed everything `grpo-v2` died of
(`docs/decisions/grpo-v3-pipeline-fixes.md`): rewards can no longer be fabricated, every train task
carries an oracle, the pool was selected by measurement, G=8 at temperature 1.0, 50 steps. The
trainer saw reward contrast at every step and the LoRA weights moved. Train-pool success did not
rise; it drifted down. This file records the run, the ledger, the diagnosis and the decision to
stop, entirely from what is on disk under `rl_runs/grpo-20260902-025034/` in the `feat/grpo-v3`
worktree (`git_head 058e868`).

Companions: `docs/decisions/grpo-v2-first-live-run.md` (the run before), `docs/open-items.md` §8–§10.

## 1. What the run was

| | |
| --- | --- |
| policy start | `sft-v2` LoRA (`sft_adapter_sha256 9802e033…`) merged into `Qwen/Qwen3.5-9B` on the box; the GRPO LoRA sits on the merged weights |
| trainer | ms-swift GRPO, vLLM server mode, LoRA r64/α128 all-linear, `freeze_vit`, lr 5e-6 cosine to 0, β 0.01, ε 0.2 / ε_high 0.28, G=8, temperature 1.0, top_p 0.95, 16 episodes per step (2 tasks × 8), 50 steps, `dynamic_sample: false` |
| episode | `max_turns 36`, `max_completion_length 512`, `max_length 3072`, one screenshot per prompt, history window 10 |
| compute | Vast.ai, two GPUs (rollout engine on one, trainer on the other), `pytorch/pytorch:2.11.0-cuda12.8` image; `train_runtime` 29 871 s (8.3 h), 480–770 s per step |
| env | `rl.service` on the Mac through the reverse tunnel (`http://127.0.0.1:8900`), live public sites, loop guard 5 |
| train tasks | 25 from `rl/frontier_selection.json`: the pass@2 band [0.2, 0.8] over `rl/rl_tasks_candidates.json` (suite sha `3705ec31…`); every one of the 25 sat at exactly 1/2 at selection; `train_max_expected_actions 18` |
| reward | oracle only (`--verifier-mode never`; all 752 episodes carry `success_source: success_check`). `rl/reward_config.json` autonomous lane: success 1.0, format 0.2, efficiency ≤ 0.5 (zero at 2× expected), done-on-success 0.2, progress ≤ 0.4 on failure; penalties `step_cap` 0.45, `false_done` 0.35, `loop_guard` 0.3, `parse_abort` 0.5, `control_unexpected` 0.3, `navigate_blocked` 0.1 |
| ledger | `run_header.json`, `rl_output/v0-20260902-013334/logging.jsonl` (every 5 steps), `completions.jsonl` (per-turn records), 752 `episode.json` under `iter_000/<task>/kNN/task_*/` |
| artifact | `output/grpo-v3-rescued/adapter_model.safetensors` — checkpoint-50, pulled by the supervisor script after the wrapper process died |
| eval | none: `eval_before: false`, `eval_after: false`. The in-run train-pool numbers below are the only measurement |
| cost | ≈ $25 operator-reported, including three rentals that never became SSH-ready and two setups that failed before training. On disk: `.remote_train_bad_offers.json` blocklists two hosts ("three consecutive rentals never became SSH-ready", "not SSH-ready within 900s") and six earlier `rl_runs/grpo-2026090*` directories precede the surviving run. Balance after teardown $7.56 |

The fixes doc describes `max_turns 24` and `dynamic_sample: true`; the run's config
(`rl_config_grpo50.yaml`, sha `38e57420…` = `run_header.rl_config_sha256`) has 36 and `false`. The
config is authoritative.

## 2. The curve

Trainer log, one row per logged step (`logging_steps` 5; each row is that step's 16 episodes):

| step | reward | reward_std | frac_zero_std | kl | grad_norm | turns/episode | lr | tokens/turn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.104 | 0.962 | 0.000 | 0.00 | 3.73 | 19.6 | 5.0e-6 | 61 |
| 5 | 1.051 | 0.727 | 0.125 | 0.47 | 1.90 | 16.0 | 4.9e-6 | 64 |
| 10 | 0.927 | 0.885 | 0.000 | 0.74 | 2.70 | 12.6 | 4.5e-6 | 62 |
| 15 | 1.108 | 0.848 | 0.000 | 0.84 | 0.71 | 11.8 | 4.0e-6 | 64 |
| 20 | 1.113 | 0.753 | 0.100 | 0.82 | 4.59 | 14.3 | 3.3e-6 | 64 |
| 25 | 0.808 | 0.682 | 0.000 | 0.86 | 0.95 | 15.9 | 2.5e-6 | 61 |
| 30 | 0.994 | 0.868 | 0.000 | 0.87 | 2.77 | 13.5 | 1.7e-6 | 61 |
| 35 | 1.012 | 0.809 | 0.000 | 0.83 | 2.69 | 12.1 | 1.0e-6 | 64 |
| 40 | 0.787 | 0.715 | 0.000 | 0.83 | 1.21 | 17.8 | 4.8e-7 | 64 |
| 45 | 1.019 | 0.786 | 0.000 | 0.88 | 2.23 | 13.9 | 1.2e-7 | 61 |
| 50 | 0.661 | 0.745 | 0.000 | 0.89 | 2.61 | 13.9 | 0 | 60 |

`clip_ratio/region_mean` stayed at 0.004–0.006 and no completion reached the 512-token cap.
`frac_reward_zero_std` is the share of the batch whose group had identical rewards, averaged over
the five steps behind each row: 0.125 and 0.100 are one zero-variance group in 8 and in 10 steps.
The trainer had gradient at every step.

LoRA-B Frobenius norm by checkpoint (`rl.analysis bnorm` definition, recomputed from the files):

| step | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ‖B‖ | 0.191 | 0.313 | 0.402 | 0.466 | 0.508 | 0.532 | 0.546 | 0.554 | 0.5557 | 0.5557 |

The increments shrink with the cosine schedule and stop when the learning rate reaches zero: a
converging plateau, which is what an applied update looks like. `grpo-v2` measured 0.279 after 15
steps at lr 1e-5 and was flat. Against the SFT adapter the run started from (`sft-v2` ‖B‖ 17.785)
the final norm is **3.1%**; the runbook's own kill rule (`B_ratio < 0.10` = dead) would have
classed this run dead at checkpoint-5 (1.1%) and at every checkpoint after it.

## 3. The ledger

752 `episode.json` files, 25 tasks, 16–40 episodes per task (94 groups of 8). 50 steps × 16 is
800; the 48 rollouts not on disk belong to no episode directory. Among the 752: `reward_status:
final` ×752, `reward: None` ×0, `verdict_contradiction` ×0.

**Terminal reasons**

| `done` | `step_cap` | `max_turns` | `control_unexpected` | `loop_guard` | `step_error` |
| --- | --- | --- | --- | --- | --- |
| 633 (84.2%) | 66 (8.8%) | 30 (4.0%) | 11 | 10 | 2 |

**done() against success**

| success | done() | false done (done ∧ ¬success) | success without done() |
| --- | --- | --- | --- |
| 404 / 752 = 53.7% | 633 = 84.2% | **238 = 31.6% of episodes, 37.6% of done()** | 9 (all capped successes) |

**Environment**: 30 episodes flagged `blocked` (`rate_limited` 19, `captcha` 9,
`human_verification` 2), concentrated on telegra.ph (9 of 32), the Standard Ebooks *Time Machine*
task (8 of 16) and the three justpaste.it tasks (7). Twelve of the thirty still passed the oracle.
Two `oracle_error` episodes (`page.title: Target crashed`, 12go), scored 0.2, reward present. 70
episodes attempted `navigate(...)` and were penalised for it; 11 ended `control_unexpected`.

**Reward histogram** (mean 0.952, sd 0.919, min −0.65, max 1.9)

| [−1.0, −0.5) | [−0.5, 0) | [0, 0.5) | [0.5, 1.0) | [1.0, 1.5) | [1.5, 2.0) |
| --- | --- | --- | --- | --- | --- |
| 1 | 179 | 169 | 8 | 56 | 339 |

Bimodal: 339 clean successes at 1.5–1.9 (280 at exactly 1.9) against 348 failures in [−0.5, 0.5).

**What each ending was paid** (mean reward)

| ending | n | mean | median |
| --- | --- | --- | --- |
| success | 404 | 1.778 | 1.9 |
| false done() | 238 | **0.021** | 0.05 |
| loop guard | 10 | −0.010 | |
| capped, failed (`step_cap` + `max_turns`) | 87 | **−0.084** | −0.05 |

The fixes doc set out to order failures "capped attempt > loop guard > false done". With
`false_done` 0.35 below `step_cap` 0.45 the run's order was the reverse: a policy that is unsure
earns more by stopping than by continuing.

**Per task** (sorted by pass rate; selection pass@2 was 0.50 for every row)

| task | n | pass | mean reward | sd | false done |
| --- | --- | --- | --- | --- | --- |
| gutenberg / find Pride and Prejudice | 32 | 0.97 | 1.85 | 0.29 | 1 |
| 12go / Bangkok–Chiang Mai Saturday | 32 | 0.94 | 1.79 | 0.41 | 0 |
| calculator.net / loan 12 500 3 yr | 32 | 0.91 | 1.71 | 0.56 | 3 |
| calculator.net / BMR female 29 165 cm 62 kg | 32 | 0.84 | 1.63 | 0.63 | 5 |
| gutenberg / Two Cities opening | 24 | 0.83 | 1.51 | 0.78 | 1 |
| privatebin / markdown, month | 24 | 0.83 | 1.55 | 0.72 | 3 |
| rentry / publish note | 32 | 0.81 | 1.49 | 0.80 | 2 |
| privatebin / discussion, year | 32 | 0.78 | 1.45 | 0.76 | 4 |
| archive.org / software collection | 40 | 0.78 | 1.39 | 0.74 | 9 |
| flixbus / Berlin–Prague Saturday | 32 | 0.75 | 0.96 | 0.72 | 1 |
| rentry / follow-ups | 32 | 0.66 | 1.21 | 0.93 | 9 |
| standardebooks / Austen page | 32 | 0.66 | 1.02 | 0.71 | 6 |
| standardebooks / Frankenstein | 32 | 0.66 | 1.08 | 0.82 | 5 |
| standardebooks / Time Machine | 16 | 0.50 | 0.84 | 0.91 | 3 |
| dpaste / python paste with expiry | 24 | 0.42 | 0.69 | 0.81 | 0 |
| uuidgenerator / version-4 page | 32 | 0.41 | 0.92 | 0.81 | 19 |
| arxiv / BERT paper | 40 | 0.40 | 0.71 | 0.86 | 23 |
| openlibrary / Dracula work | 24 | 0.33 | 0.78 | 0.79 | 15 |
| standardebooks / Treasure Island | 32 | 0.31 | 0.55 | 0.86 | 11 |
| bahn / Berlin–Hamburg | 24 | 0.21 | −0.01 | 0.60 | 10 |
| calculator.net / final exam score | 32 | 0.16 | 0.28 | 0.60 | 14 |
| justpaste.it / smoke note | 40 | 0.05 | −0.06 | 0.40 | 29 |
| telegra.ph / failed checks | 32 | 0.03 | 0.17 | 0.35 | 27 |
| justpaste.it / pending list | 24 | 0.00 | −0.10 | 0.15 | 19 |
| justpaste.it / run summary | 24 | 0.00 | −0.07 | 0.18 | 19 |

At run scale 13 of 25 tasks are inside the [0.2, 0.8] band they were selected into, 7 are above
it (208 episodes) and 5 below (152 episodes): 48% of the rollouts came from tasks the policy either
nearly always solves or nearly never solves. In the ledger's k-ordered groups of 8, 24 of 94 have a
single outcome; 70 are mixed.

**First versus last window.** The ledger does not record the trainer step, so episodes are ordered
by the write time of `episode.json` (05:24 → 12:46) and the windows are the first and last 160
episodes, i.e. ten steps' worth each:

| | first 160 | last 160 | all 752 |
| --- | --- | --- | --- |
| mean reward | 1.064 (sd 0.894) | 0.840 (sd 0.930) | 0.952 |
| success | 59.4% | 50.0% | 53.7% |
| done() | 85.6% | 86.9% | 84.2% |
| false done() | 27.5% | **38.8%** | 31.6% |
| capped | 10.0% | 10.6% | 12.8% |
| blocked | 4 | 8 | 30 |
| navigate attempts | 10 | 18 | 70 |
| steps per episode | 14.3 | 13.7 | 14.0 |

Reward fell by 0.224 (z = 2.2) and success by 9.4 points (z = 1.7). The operator's figures
(1.078 → 0.889, 59.6% → 52.1%, ~1.5σ) come from a slightly different window and agree in sign and
size. In 80-episode buckets success runs 67.5, 51.2, 57.5, 51.2, 52.5, 56.2, 42.5, 58.8, 46.3,
53.1: one high first bucket, then a flat band between 42% and 59%. The single behavioural change
that is larger than the bucket noise is the false-done share, up eleven points while the done()
rate stayed put.

## 4. Diagnosis

Reward contrast was healthy (`frac_reward_zero_std` ≈ 0, `reward_std` 0.68–0.96, 70 of 94
ledger groups mixed) and the weights moved (‖B‖ 0.19 → 0.56, saturating with the schedule). The
policy did not get better at the tasks. The reasons, in the order of their weight:

1. **On live sites the within-group spread is mostly not the policy.** Thirty episodes hit a rate
   limit, a CAPTCHA or a human-verification wall; twelve of them still passed and eighteen failed,
   so the same wall lands on both sides of a group. Two more crashed the page. The per-task pass
   rates in the run span 0.00–0.97 for tasks that all measured exactly 0.50 at selection. What GRPO
   normalises inside a group is largely which sample drew the slow page, the rate limit or the
   lucky token at temperature 1.0.
2. **800 episodes cannot average that out.** With reward sd 0.92, the mean over one step's 16
   episodes has a standard error of 0.23; the logged per-step reward swings 0.66–1.11, which is
   that noise. Ten-step windows of 160 resolve a change of ~0.2 in reward or ~10 points in success
   and nothing finer.
3. **The KL anchor did not hold.** KL to the SFT policy reached 0.83 by step 15 and stayed at
   0.82–0.89 to the end. At β = 0.01 that term contributes ≈ 0.008 to a loss that ranged 0.01–0.23:
   there was no restoring force, and the policy sat at a fixed distance from `sft-v2` for 35 steps
   while the reward stayed flat.
4. **A false done() was priced above an honest capped attempt.** `false_done` 0.35 against
   `step_cap` 0.45 put the mean false-done reward at +0.02 and the mean capped failure at −0.08.
   The one thing the run measurably taught the policy is to stop earlier: false done() went from
   27.5% to 38.8% of episodes while done() itself stayed at ~86%.
5. **The band was measured at k = 2 and did not hold at k ≈ 30.** A pass@2 of 1/2 carries a
   standard error of ±0.35; at run scale 12 of the 25 tasks were outside [0.2, 0.8], and four
   tasks (152 episodes) were at ≤ 0.05, the all-fail groups `grpo-v2` was supposed to have
   eliminated.
6. **The observation the policy trained on was not the observation it was fine-tuned on.**
   `feat/grpo-v3` carries `train_qwen.py` from 7ea86ab (2026-08-28) with `HISTORY_WINDOW = 10`;
   `sft-v2` was trained from 1ac5d18 (2026-09-01) with `HISTORY_WINDOW = 20`. The logged prompts
   show it (`Previous actions: [... 14 earlier actions, ` followed by ten actions). Every RL step
   past the tenth action showed the policy a truncated history it had never been trained on.

Each of 3–6 is a one-line config or branch fix. None of them fixes 1 and 2, which are what the
environment is.

## 5. Decision

**RL is dropped for the paper.** The three runs are reported as a diagnosed negative result:

| run | what it showed |
| --- | --- |
| v1 (Modal smoke, 2026-08-27) | the machinery: merge, plugin paths, one GRPO step with per-turn records |
| v2 (2026-08-28) | dead: ‖B‖ 1.9% of the SFT adapter, 20% fabricated zero rewards, 68% loop-guard endings, gradient from 2 of 12 tasks |
| v3 (2026-09-02) | alive and flat: contrast at every step, weights moved 3.1%, train-pool success 59% → 50% inside noise, false done() up 11 points |

No rollout data enters SFT. All 752 episodes carry `metadata.excluded_reason: rl_rollout` and live
under `rl_runs/`, outside the `trajectories/**` glob that `convert` walks.

A run that could work would need, all at once: **≥ 3 000 episodes**, so that a five-point change
in success between windows of 1 500 is ~2.7σ instead of the ~0.9σ it is at 160; **per-turn
credit**, because today one scalar is broadcast to every one of an episode's 12–20 turn records and
a failure at turn 19 penalises eighteen correct turns; **the parity prompt** (history window 20);
**`false_done ≥ step_cap`**, so that stopping without success never pays more than continuing; and
**a task pool at the benchmark's distribution**, since gains on rentry and justpaste.it are not the
thing Online-Mind2Web measures. At the observed ~10 min per 16-episode step, 3 000 episodes is
~190 steps and ~31 h of a two-GPU box, plus a day of the Mac serving the sandboxes, against a Vast
balance of $7.56. That buys the measurement of whether RL helps, not the gain itself, and the
corpus work in `docs/decisions/sft-v3-plan.md` has a known, cheaper target.

## 6. The teardown incident

The supervisor script that took over the box after the wrapper died released instance 49593507
by rewriting its ownership record with `pid: 1` and then ran `remote_train cleanup`, which probes
the owner with `os.kill(pid, 0)`; pid 1 answers `PermissionError`, which `owner_alive` read as a
live foreign process, so cleanup skipped the box and it billed until the record was rewritten by
hand. The sentinel is now explicit: a record with `pid: null` means released, `owner_alive`
returns `False` for it while PermissionError still means alive for a real pid, and
`state.mark_abandoned(id)` / `python -m remote_train release --instance ID` are the writers a
supervisor uses instead of editing the JSON. `make test-remote` now covers the null sentinel, the
own-pid and missing-pid cases, and `cleanup` reclaiming a released record.
