# GRPO v2 — first live before/after (2026-08-28)

The first end-to-end GRPO run of the RL harness (`rl/`, `rl_train.py`, `modal_grpo.py`) on real
public sites, and the before/after it produced. Written down because the headline delta is inside
noise and the run's value is in what it proved and what it found.

## Setup

| | |
| --- | --- |
| policy start | `sft-v1` LoRA merged into `Qwen/Qwen3.5-9B` (`merged/sft-v1__adapter`) |
| trainer | ms-swift 4.5.2 GRPO, vLLM 0.23 server mode, LoRA r64/α128 all-linear, lr 1e-5, β 0.01, G=4, 16 episodes per step |
| compute | Modal, 2×L40S (rollout engine and trainer cannot share a device), ~$20 for 15 steps |
| env | `rl.service` on the Mac: 3 sandboxes × 2 drivers, live public sites through an ngrok ingress with localtunnel failover |
| train tasks | 12 (train split filtered to ≤18 expected actions, `train_max_expected_actions`), `max_turns 28` |
| reward | v2 (`rl/reward_config.json`): criteria-consistent judge verdict ×0.6 on `meandering`, step efficiency to 0 at 2× expected, done() bonus on success only, penalties for false done and either cap |
| eval | Online-Mind2Web, the archived 50-task sample (17 easy / 16 medium / 17 hard), `viewport_abs`, temperature 0, 2 mid screenshots, criteria-first judge — both sides the same night on the same harness commit |

## Result

| | `sft-v1` | `grpo-v2` | Δ |
| --- | --- | --- | --- |
| verified success, scored | 9.4% (3/32) | 11.8% (4/34) | +2.4 pp |
| verified success, all 50 | 6.0% | 8.0% | +2.0 pp |
| easy / medium / hard | 3/11 · 0/9 · 0/12 | 3/11 · 0/10 · 1/13 | hard +1 |
| done() rate | 16% | 22% | +6 pp |
| steps / reference on successes | 1.33× | 1.12× | −0.21 |
| parse-failure rate (per step) | 8.0% | 8.7% | +0.7 pp |
| canonical action form | 100% | 100% | — |
| site-blocked | 28% | 28% | — |

All three SFT successes held; the one new success is gov.uk (`efficient`, 8 steps) where SFT hit the
step cap. One task out of fifty is inside noise; what is not noise is that nothing regressed and
the two behaviours the reward was built to move — stopping with done() and step efficiency —
moved in the intended direction.

The SFT baseline under this judge is 9.4%, not the archived 29.4% (old judge): of the ten archived
successes, three hold, two were font-wait screenshot timeouts tonight (carnival, webmd), three are
step-cap episodes the criteria-first judge rejects (gov.uk, recreation.gov, ca.gov — recreation.gov
was the audit's confirmed false positive), and two played out differently. Numbers judged by the two
rubrics must not be put in one table.

## What the run proved

Rollouts → judge → GRPO update → checkpoint → adapter → eval, on real sites, at ~5–9 min per
generation. Group rewards are correctly ordered (efficient success 1.9 › meandering success 1.0 ›
capped success 0.6 › failure ≤ 0), the judge produced no contradictions in ~250 verdicts, and the
per-task GRPO groups had non-zero variance where the student is at its frontier.

## Why the learning signal was thin

- Only 3–4 of the 12 train tasks have mixed outcomes for the SFT student; the other groups are all-fail
  and contribute no advantage.
- The driver's loop guard (3 identical actions) ended ~70% of episodes at −0.1 — now a service knob
  defaulting to 5.
- Steps ~6–9 trained on zeros: the rate-limit tracker's rejected flag is sticky and every verdict was
  deferred after one 429 — fixed so the judge clears a stale flag and waits on a real one.
- Mean episode reward stayed near 0.05 across the run; 15 steps × 16 episodes is a proof of machinery,
  not a training curve.

## Defects found and fixed on the way (all on `feat/rl-harness-phase3`)

Queueing under the ingress's 100 s cut-off; trainer-capped episodes' close reward dropped; env calls
blocking the rollout server's loop; a task's four GRPO samples sharing one trajectory dir; a step
re-executed on retry after an observe race; the sticky usage-limit flag; `start_driver` argument
order breaking the eval harness; eval client not backing off on a preempted serve; the serve falling
back to the raw base because `CUA_SERVE_MODEL` was read in the container. Each is one commit with
the reasoning in its message.

## Next run

1. Tasks at the student's frontier: more short, idempotent tasks on the sites where it sometimes
   succeeds, plus `success_check`s so rewards stop depending on judge throughput.
2. `--loop-guard-repeats 5`, `num_generations 8` (a rare success is more likely to appear in a group
   of eight), 40–60 steps (~$15–25 at the observed rate).
3. Serve the after-adapter on the merged SFT base — the GRPO LoRA's `base_model_name_or_path` is the
   merged model, never the raw `Qwen/Qwen3.5-9B`.
