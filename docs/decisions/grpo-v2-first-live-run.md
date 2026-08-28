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

All three policies, the same night, the same 50 tasks, the same criteria-first judge, harness parity
confirmed by `compare` (`eval_archive/online_mind2web/{qwen-base-judge2,sft-v1-judge2,grpo-v2-judge2}`):

| | base `Qwen3.5-9B` | `sft-v1` | `grpo-v2` |
| --- | --- | --- | --- |
| verified success, scored | 14.7% (5/34) | 9.4% (3/32) | 11.8% (4/34) |
| verified success, all 50 | 10.0% | 6.0% | 8.0% |
| easy / medium / hard | 4/11 · 0/10 · 1/13 | 3/11 · 0/9 · 0/12 | 3/11 · 0/10 · 1/13 |
| per-step parse failure | 20.6% | 8.0% | 8.7% |
| parse-failure aborts (episodes) | 19 | 13 | 13 |
| canonical action form | 18.8% | 100% | 100% |
| done() rate | 22% | 16% | 22% |
| steps / reference on successes | 1.92× | 1.33× | 1.12× |
| site-blocked / unreachable / harness | 14 / 2 / 0 | 14 / 2 / 2 | 14 / 2 / 0 |

Three tasks are solved by every policy (cookpad, Google Finance, theweathernetwork). The base model
adds americashealthrankings (graded `meandering`, the audit's known false-positive site) and a second
Google task; GRPO adds gov.uk (`efficient`, 8 steps, where SFT hit the cap). **Five, three and four
successes out of ~34 are indistinguishable at this sample size** — the honest headline is that task
success did not measurably change with training. What did change, and by far more than the noise:
format compliance (canonical form 19% → 100%), per-step parse failures (21% → 8%), and steps on
successes (1.92× → 1.12× of the reference length). GRPO against SFT moves the two behaviours its
reward targets — stopping with done() (16% → 22%) and step efficiency (1.33× → 1.12×) — the intended
way, with nothing regressing.

The archived "base 8.8% → sft-v1 29.4%" (old judge, 2026-08-26) does not survive re-measurement:
of the ten archived SFT successes, three hold, two were font-wait screenshot timeouts tonight
(carnival, webmd), three are step-cap episodes the criteria-first judge rejects (gov.uk,
recreation.gov, ca.gov — recreation.gov was the audit's confirmed false positive), and two played out
differently on a different night. Numbers judged by the two rubrics must not share a table.

## Grounding did not regress

ScreenSpot (`rootsautomation/ScreenSpot` test, the archived desktop-300 split: 162 windows +
138 macos), the same night, every endpoint warm, each model scored in its own coordinate space
(`eval_archive/screenspot_grpo_v2/`):

| | base `Qwen3.5-9B` | base + `sft-v1` LoRA | merged `sft-v1` | `grpo-v2` | archived (2026-08-26) |
| --- | --- | --- | --- | --- | --- |
| accuracy, n=300 | 89.0% | 85.3% | 85.3% | 85.0% | base 89.0% · sft-v1 85.0% |
| parse rate | 94.3% | 97.3% | 97.3% | 96.7% | 94.7% · 97.0% |
| icon / text | 80.3 / 94.9 | 75.4 / 92.1 | 76.2 / 91.6 | 75.4 / 91.6 | — |

GRPO sits on top of SFT within noise (85.0 vs 85.3); the −4 pp against the base is the SFT-era trade
already recorded on `main`. The merged weights score exactly what base + LoRA scores (85.3 vs 85.3),
which validates the merge the RL run trained on. On the full 1,272-sample split (desktop + mobile +
web) GRPO scores 78.6%.

One harness lesson: a serve that has scaled to zero answers nothing for its first minutes, and the
grounding client's 2–4 s retries then record empty predictions as misses — the first pass of the
merged-SFT spec read 25.7% on the first 300 for exactly that reason. Warm the endpoint before any
grounding run.

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
