# SFT v2 — measured results (2026-08-31)

Status: **IN PROGRESS** — this file is written as the runs land. Anything not filled in below did
not finish, and is named as such rather than estimated.

## What is being compared

| | base | sft-v2 |
| --- | --- | --- |
| weights | `Qwen/Qwen3.5-9B` (stock) | base + SFT v2 LoRA |
| served name | `cua-student` | `cua-student-lora` |
| coord space | `normalized_1000` | `viewport_abs` |
| corpus | — | v2: 698 trainable trajectories, 9,011 examples |
| recipe | — | LoRA r64/a128, lr 5e-5, 3 epochs, packing on, val split 0.1 |

Both sides run the SAME night, the SAME 230-task pinned file
(`eval_archive/online_mind2web/tasks_pinned_v2.jsonl` = the archived judge2 50 + 180 level-stratified
tasks, sites blocked in all three judge2 runs excluded from the new 180), the SAME criteria-first
judge on `claude-sonnet-5`, `--parallel 3`, temperature 0. `compare` verifies harness parity; only
`coord_space` is expected to differ.

**These numbers are NOT comparable to the 2026-08-27 judge2 triple**: the harness fingerprint
changed (the action grammar now accepts `navigate` / `handoff` / `confirm`, and `history_window`
went 10 → 20). That is why the base side is re-measured here rather than quoted from before.

## Headline

| metric | base `Qwen3.5-9B` | `sft-v2` |
| --- | --- | --- |
| verified success, scored | **6.7%** (11/164) | (pending) |
| verified success, all 230 | 4.8% | (pending) |
| parse-failure rate (steps) | 23.6% | (pending) |
| canonical action form | 29.7% (401/1351) | (pending) |
| done() emission rate | 11.3% | (pending) |
| steps / reference on success | 1.93x | (pending) |
| navigate actions executed | 42 | (pending) |
| episodes ended by handoff/confirm | 14 | (pending) |

Base run: `eval_runs/qwen-base-v2_online_mind2web_20260830T223946Z`, finished 2026-08-31.
At 6.7% over 164 scored episodes the 95% CI is roughly ±4pp, so sft-v2 needs about **13%+ scored**
to be a defensible improvement rather than noise. The 34-scored-episode runs of 2026-08-27 could
not have resolved anything of that size, which is why the sample was widened to 230 tasks.

**The harness fix is load-bearing, not cosmetic.** The base model executed 42 `navigate` actions
and ended 14 episodes with `handoff`; under the old verb table every one of those would have been
scored as a parse failure instead of a real outcome.

## Per level

| level | n | scored | verified | success rate | base blocked |
| --- | --- | --- | --- | --- | --- |
| easy | 65 | 44 | 11 | **25.0%** | 18 |
| medium | 101 | 71 | 0 | **0.0%** | 24 |
| hard | 64 | 49 | 0 | **0.0%** | 13 |

Every base success is an easy task; medium and hard are zero. Outcome spread: 94
`parse_failure_abort`, 55 `site_blocked`, 30 `step_cap`, 26 `completed`, 14 `handoff`, 8
`unreachable`, 3 `harness_error`.

## Format and behaviour metrics

(pending)

## Grounding (ScreenSpot desktop-300)

(pending)

## Reading

(pending)
