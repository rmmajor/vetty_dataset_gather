# Evaluation archive

Raw artefacts behind the evaluation figures quoted in the root `README.md`. Committed deliberately:
`eval_results/` and `eval_runs/` are gitignored, so without this directory those figures would have
no reproducible backing once the working tree is cleaned. Later rounds are recorded in the
experiment registry instead, each row naming its own `artifacts_path` and `artifacts_availability`.

`manifest.json` carries the headline figures plus a sha256 and byte count for every file here, so a
silent edit to an archived record is detectable.

## What is here

| path | what |
| --- | --- |
| `screenspot_sft_v1/before.json` | 300 per-sample grounding records, base model (`normalized_1000`) |
| `screenspot_sft_v1/after.json` | 300 per-sample grounding records, SFT v1 LoRA (`viewport_abs`) |
| `screenspot_sft_v1/comparison.json` | rendered side-by-side summary of those two |
| `screenspot_grpo_v2/*.json` | the same grounding harness over the served base, SFT v1 (merged and LoRA) and GRPO v2 arms |
| `online_mind2web/baseline/` | 50-episode agentic run, base model: `results.json`, `report.md`, `run.log` |
| `online_mind2web/after/` | the same 50 episodes, SFT v1 LoRA |
| `online_mind2web/*-judge2/` | base / SFT v1 / GRPO v2 re-measured on one criteria-first judge |
| `online_mind2web/tasks_*.jsonl` | the pinned task sets every run above drew from (see that directory's README) |

Each `results.json` carries the full harness fingerprint — dataset revision, judge model, step-cap
policy, temperature, coordinate space, egress — which is what makes a before/after pair comparable
at all. Served endpoint hostnames are recorded as `<modal-workspace>`: the workspace a run happened
to be deployed in is not part of the measurement.

## Headline figures (SFT v1, first round)

| metric | before | after |
| --- | ---: | ---: |
| Online-Mind2Web verified success (scored) | 8.8% | 23.5% audited / 29.4% as measured |
| Online-Mind2Web verified success (all 50) | 6.0% | 16.0% audited / 20.0% as measured |
| ScreenSpot grounding accuracy | 89.0% | 85.0% |
| ScreenSpot parse rate | 94.7% | 97.0% |

Quote the **audited** Online-Mind2Web figures. Two of the ten verified successes are confirmed judge
false positives — see `docs/decisions/judge-consistency-audit.md`. The ScreenSpot row was measured
under a superseded protocol that resized the `viewport_abs` arm's inputs; the root `README.md`
explains why it does not stand as a regression.

## What is NOT here

**Per-episode screenshots (~166 MB).** Every Online-Mind2Web episode recorded a screenshot per step
plus a final screenshot; those are the evidence the success audit was performed against (a claimed
success was checked by reading the final screenshot, which is how both false positives were caught).

They are excluded to keep the repository small, but they are **not** discarded — both run
directories are packed with checksums alongside the adapter, outside git:

| file | size | sha256 |
| --- | ---: | --- |
| `models/om2w-baseline-screenshots.tar.gz` | 55 MB | `b98c01924eda7e52…` |
| `models/om2w-after-screenshots.tar.gz` | 106 MB | `4277ea6e5ee96744…` |
| `models/om2w-screenshots.sha256` | — | both digests, verify with `shasum -a 256 -c` |

Each tarball is a complete run directory (`results.json`, `report.md`, `run.log`, and
`episodes/*/step_NNN_screenshot.png` + `final_screenshot.png`), unpacking to the original
`eval_runs/<run_id>/` layout. `results.json` records each episode's `episode_dir`, so records
re-associate with their screenshots automatically once a tarball is unpacked.

`models/` is gitignored, so these live with the adapter rather than in the repository. Anyone
reproducing the success audit needs them: both confirmed false positives were caught by reading a
final screenshot, not by reading the judge's verdict.

## Reproducing

Root `README.md` has the commands for both harnesses. Note that `eval_models_served.yaml` (not
`eval_models.yaml`) describes what actually ran for ScreenSpot: `backend: openai` against vLLM
served names, rather than the local `swift infer` lane.
