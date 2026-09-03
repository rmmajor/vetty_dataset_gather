# SFT-3 evaluation runbook

Every number in the paper's matched column comes from ONE rental serving every arm at once, so the
harness commit, the task sample and the judge are identical across models and only the model varies.
Verified against the code on 2026-09-03 while sft-v3 was training; nothing here has been executed yet.

## 0. Adapters (all four survive locally)

| arm | path | answers in |
| --- | --- | --- |
| base | none, the served base | `normalized_1000` |
| sft-v1 | `models/cua-sft-v1` | `viewport_abs` |
| sft-v2 | `models/cua-sft-v2` | `viewport_abs` |
| sft-v3 | `output/<run>/adapter` (pulled by the training lane) | `normalized_1000` |

`models/` also holds `cua-grpo-v2` and `cua-grpo-v3`; RL is dropped, so they are archive only.

## 1. Serve every arm from one box

`rl_train.build_vllm_serve_command` takes a `name -> path` dict and emits one `--lora-modules` list,
so a single rental exposes `cua-student` (base) plus `sft-v1`, `sft-v2`, `sft-v3` as served names.
Serve with `--limit-mm-per-prompt image=2`: sft3 sends two images and vLLM's default of 1 returns
HTTP 400. Warm the endpoint before any sweep — a cold start reads as a dead policy.

## 2. ScreenSpot grounding (~10 min per arm)

`eval_models_served.yaml` already carries the four specs with their own `coord_space`. Run
`make eval-screenspot` against the endpoint. The base "before" number must be re-measured under
`GROUNDING_SYSTEM_PROMPT` so before and after share a prompt; the historical 89.0% was taken under
the agent prompt and is not comparable.

## 3. Online-Mind2Web (~1.5 h per arm)

Task set `eval_archive/online_mind2web/tasks_reachable_us_174.jsonl`, browser through the
residential-egress backend, model on the rented endpoint. sft-v2 MUST be re-run on this set: its
only existing number is on the archived 50-task set and the two are not comparable. The base side
is partly free — 149 of the 174 are shared with the existing 230-task base run, so only the
remaining 25 need executing.

## 4. QA benchmark (browser local, model remote)

The bench selects the served adapter by name, so one endpoint covers both arms:

| setting | sft-v2 arm | sft-v3 arm |
| --- | --- | --- |
| `VETTY_LOCAL_MODEL_NAME` | `sft-v2` | `sft-v3` |
| `BENCHMARK_CUA_OBSERVATION` | `v2` | `v3` |
| `BENCHMARK_CUA_COORD_SPACE` | `viewport_abs` | `normalized_1000` |
| `BENCHMARK_CUA_HISTORY_WINDOW` | 20 | 20 |
| `BENCHMARK_CUA_MAX_STEPS` | 25 | 25 |

Window 20 and 25 steps are held constant deliberately: sft-v2 trained on a 20-entry history while
the bench ran 10, so its published 21.2% was measured off its own distribution. Record with
`--record-only`, tear the GPU down, then run the verdict tier and the judge on Claude usage.
Report the published numbers and the matched numbers as two separate columns.

## 5. What decides the result

`docs/decisions/sft-v3-corpus-round.md` holds the pre-registered bands. The three recording-only
rows — coordinate loops, premature `done`, and success claims absent from the final page — are the
decisive ones: they need no judge, cost nothing beyond the recording, and are cap-independent. DDR
is reported but does not decide, because n=33 defects carries a ±16 pp interval.
