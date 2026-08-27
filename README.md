# GPT-5.4 → Qwen 3.5-9B Computer Use Distillation Pipeline

Distill GPT-5.4's web browsing capabilities into Qwen 3.5-9B via trajectory-based
supervised fine-tuning (SFT). Based on the "Distilling GPT-5.4 Computer Use into
Qwen 3.5-9B" blueprint.

Two data sources feed the same training pipeline:

1. **GPT-5.4 CUA distillation** — record trajectories by having GPT-5.4 perform Mind2Web tasks in a browser ($0.55/trajectory)
2. **QA repo pipeline** — find existing Playwright test repos on GitHub, instrument them to capture screenshots + coordinates, generate thinking blocks with Claude ($0.05–0.15/trajectory)

Both produce identical trajectory formats, so `train_qwen.py convert` works on either.

## Prerequisites

- Python 3.10+
- For GPT-5.4 path: OpenAI API key with CUA access
- For repo pipeline: GitHub token, Anthropic API key (Claude Agent SDK), Docker
- For training: GPU with 16-24GB+ VRAM, ms-swift installed

## Setup

    pip install -r requirements.txt
    playwright install chromium
    export OPENAI_API_KEY=your_key_here

For the repo pipeline:

    pip install -r requirements-pipeline.txt
    export GITHUB_TOKEN=your_token
    export ANTHROPIC_API_KEY=your_key

For training (on GPU machine):

    pip install ms-swift[llm]

## Path A: GPT-5.4 CUA Distillation

### Step 1: Load tasks from Mind2Web

    python load_tasks.py --output tasks.jsonl
    python load_tasks.py --output tasks.jsonl --limit 100  # limit for testing

### Step 2: Record trajectories

    python record_trajectories_from_gpt_api.py --tasks tasks.jsonl --output trajectories/

Options:
- `--max-steps 15` — max actions per task (default: 15)
- `--viewport 1280x720` — browser viewport size (default: 1280x720)
- `--resume` — skip tasks that already have a trajectory directory

## Path B: QA Repo Pipeline

Finds Playwright test repos on GitHub, instruments them to capture screenshots and pixel coordinates, runs the tests in Docker sandboxes, and generates thinking blocks with Claude. The output lands in `trajectories/` in the same format as Path A.

### Architecture

A 6-stage quality funnel — cast the widest net, then reject aggressively:

| Stage | What | LLM | Cost/repo |
|-------|------|-----|-----------|
| 1. Discover | GitHub API search (50+ queries) | None | $0 |
| 2. Filter | Stars, recency, size, language | None | $0 |
| 3. Shallow Score | Claude Sonnet scores repo metadata | Sonnet | ~$0.03 |
| 4. Deep Analyze | Clone + statically inspect tests in Docker (no install) | Sonnet | ~$0.15 |
| 5. Trial Run | POC instrumented test in Docker | Sonnet | ~$0.30 |
| 6. Generate | Full trajectory generation in Docker | Sonnet | ~$0.60 |

Stages 1–2 run on-demand. Stages 3–6 run as an overnight cron job with a configurable budget cap. Results are sent via Telegram in the morning.

All repo code runs inside ephemeral Docker containers (stages 4–6). No repo code touches the host. Dependency install happens at stages 5–6 (each in its own fresh container, in code via `setup_repo_instrumentation`), not at stage 4 — stage 4 is a cheap static triage that only reads files.

### Setup

    make setup-pipeline       # install Python deps
    make build-sandbox        # build Docker sandbox image (~2GB, mostly Chromium)
    make test-sandbox         # verify sandbox works

Optional — Telegram notifications:

    export TELEGRAM_BOT_TOKEN=your_token
    export TELEGRAM_CHAT_ID=your_chat_id
    make test-notifier        # verify bot sends messages

### Usage

    # Stage 1: Discover repos on GitHub
    make pipeline-discover

    # Stage 2: Filter by metadata
    make pipeline-filter

    # Stage 3: Score with Claude (costs money)
    make pipeline-score PIPELINE_LIMIT=50 PIPELINE_BUDGET=5

    # Stages 3-6: Full overnight run
    make pipeline-overnight PIPELINE_BUDGET=30

    # Check progress
    make pipeline-status

Or use the CLI directly:

    python -m repo_pipeline.orchestrator --stage 1
    python -m repo_pipeline.orchestrator --stage 2
    python -m repo_pipeline.orchestrator --stage 3-6 --limit 10 --budget 5
    python -m repo_pipeline.orchestrator --overnight --budget 30 --workers 5
    python -m repo_pipeline.orchestrator --until-limit 75 --budget 30
    python -m repo_pipeline.orchestrator --status

CLI options:
- `--stage N` or `--stage N-M` — run specific stage(s)
- `--overnight` — run stages 3–6 once, send Telegram report when done
- `--until-limit [PCT]` — loop stages 3–6 until Claude subscription usage hits PCT% of the limit (bare flag = 100). Keeps draining the funnel across passes until the threshold, the `--budget`, or no pending repos remain — whichever comes first. Reads the SDK's `RateLimitEvent.utilization`; if no usage events are observed (possible under headless OAuth), it degrades to "until a hard 429 / funnel drained".
- `--status` — print registry stats
- `--limit N` — max repos per stage
- `--workers N` — parallel workers for LLM stages (default: 5)
- `--budget N` — max USD for the run (default: 50)

Rate-limit safety (all modes): a hard 429 from Claude stops the run cleanly and leaves the affected repo in its input status (it is **not** rejected), so the next run resumes it.

### Cron Setup

See `crontab.example`. Example:

    0 1 * * * cd /path/to/dataset_gather && python -m repo_pipeline.orchestrator --overnight --budget 30 >> pipeline_data/run_logs/$(date +\%Y\%m\%d).log 2>&1

### Registry

All repo state lives in `pipeline_data/registry.db` (SQLite). The registry tracks each repo through the pipeline:

    discovered → filtered → scored → analyzed → validated → completed

Each stage is idempotent — running it twice on the same repo is a no-op. New search queries add new repos without touching existing ones.

Re-evaluate rejected repos:

```python
from repo_pipeline.registry import RepoRegistry
registry = RepoRegistry()
registry.reset_rejected(stage=5)  # re-queue repos rejected at trial stage
```

### File Structure

    repo_pipeline/
    ├── orchestrator.py              # Main entry point
    ├── registry.py                  # SQLite repo registry
    ├── schemas.py                   # Pydantic models
    ├── sandbox.py                   # Docker container management
    ├── stage1_discover.py           # GitHub API search
    ├── stage2_filter.py             # Metadata filtering
    ├── stage3_shallow_score.py      # Claude Sonnet scoring
    ├── stage4_deep_analyze.py       # Deep inspection in Docker
    ├── stage5_trial_run.py          # POC instrumented test in Docker
    ├── stage6_generate.py           # Full trajectory generation
    ├── think_generator.py           # Claude Sonnet thinking blocks
    ├── notifier.py                  # Telegram reports
    └── instrumentation/
        ├── capture.py               # Shared capture logic
        ├── pw_python_conftest.py    # Python Playwright instrumentation
        └── pw_ts_fixture.ts         # TS/JS Playwright instrumentation
    docker/
    ├── Dockerfile.sandbox           # Isolated execution environment
    └── docker-compose.sandbox.yml

## Converting and Training

Both paths produce trajectories in the same format. Convert and train:

    python train_qwen.py convert --input trajectories/ --output training_data/
    python train_qwen.py train --config config.yaml --dataset training_data/sft_dataset.jsonl

Options for convert:
- `--val-split 0.1` — fraction of data for validation (default: 0.1)

Options for train:
- `--epochs 3` — override number of epochs
- `--lr 5e-5` — override learning rate
- `--batch-size 1` — override batch size

Or use Make:

    make convert
    make train

## Experiment registry

Every training run and every measured number lives in `experiments/models.csv` (what was
trained) and `experiments/results.csv` (what was measured). One row per
model × eval × metric × repeat, so adding an eval or a dataset version never reshapes the table.

The block below is generated. Edit the CSVs, then:

    python experiments/render_registry.py            # re-render this README block
    python experiments/render_registry.py --fill-ci  # fill Wilson intervals for new rows
    python experiments/render_registry.py --check    # CI guard: fails if the block is stale

`status` is part of the data: `done` is measured, `pending` means the run finished but the
metric has not been computed, `planned` is on the worklist, and `superseded` marks numbers
that must not be quoted.

<!-- registry:start -->

<!-- generated by experiments/render_registry.py; edit the CSVs, not this block -->

### Models and training runs

| model_id | kind | base_model | dataset_version | method | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| qwen35-9b-base | base | Qwen/Qwen3.5-9B |  |  | done | Stock checkpoint. Answers in normalized_1000 coordinates; served as `cua-student`. |
| cua-sft-v1 | sft | Qwen/Qwen3.5-9B-Base | v1 | LoRA r64/a128 3ep lr5e-5 | done | 5477 step-level examples from 504 trainable trajectories. Answers in viewport_abs (1280x720). Served as `cua-student-lora`. |
| cua-sft-v1-ep1 | sft | Qwen/Qwen3.5-9B-Base | v1 | LoRA r64/a128 lr5e-5 | done | Epoch-1 checkpoint of cua-sft-v1; kept to show the epoch trend. |
| cua-sft-v1-ep2 | sft | Qwen/Qwen3.5-9B-Base | v1 | LoRA r64/a128 lr5e-5 | done | Epoch-2 checkpoint of cua-sft-v1. |
| cua-sft-v2 | sft | Qwen/Qwen3.5-9B-Base | v2 | LoRA | planned | Retrain on corpus v2 once the login-gated capture lane lands. |
| cua-sft-v1-nologin | sft | Qwen/Qwen3.5-9B-Base | v1-nologin | LoRA r64/a128 3ep lr5e-5 | planned | Ablation: same recipe with post-authentication trajectories removed. Isolates what the auth-gated data buys. |
| cua-grpo-v1 | grpo | cua-sft-v1 | v1 | GRPO | planned | Experimental; do not report until it has a verified run. |
| claude-sonnet-5 | frontier |  |  |  | done | Frontier executor inside the full vetty agent (explore/plan/execute/report on the claude-sdk backbone). |
| claude-opus-5 | frontier |  |  |  | done | Used as verdict and match-judge for every QA-bench column, never as an executor. |
| scripted-playwright | tool |  |  | 13 hand-written Playwright tests | done | Incumbent-method comparison arm: locator-based (#id) suite in the vetty benchmark. |
| corpus-v1 | dataset |  | v1 |  | done | Snapshot the SFT v1 model was trained on. Frozen; the deck describes this snapshot. |
| corpus-current | dataset |  | v2-wip |  | running | Live corpus, still growing. Basis for the planned v2 retrain. |
| qa-bench-refapp | benchmark |  |  |  | done | Reference app plus one-defect-per-run mutants; defect injected by config, not by patching code. |

### Results

#### `corpus-audit` — corpus-audit-v1

| model_id | subset | metric | value | n | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| corpus-current | all | distinct_sites | 366 |  | done |  |
| corpus-current | all | recorded_steps | 23492 |  | done |  |
| corpus-current | all | trainable_trajectories | 652 |  | done | 44 further trajectories carry an excluded_reason and never train. |
| corpus-current | all | trajectories | 1225 |  | done | Counted on disk over trajectories/ on 2026-08-28; the corpus kept growing after the v1 snapshot. |
| corpus-v1 | all | distinct_sites | 328 |  | done |  |
| corpus-v1 | all | median_steps_after_login | 32.5 |  | done |  |
| corpus-v1 | all | median_steps_to_login | 2 |  | done |  |
| corpus-v1 | all | post_login_work_share | 90.8 % |  | done | Provenance is a written analysis, not a pinned script. Re-run and pin the generating code before the paper. |
| corpus-v1 | all | recorded_steps | 19884 |  | done |  |
| corpus-v1 | all | sft_examples | 5477 |  | done | One example per recorded step of a trainable trajectory. |
| corpus-v1 | all | trainable_trajectories | 504 |  | done | Verified successes that became SFT examples. |
| corpus-v1 | all | trajectories | 1068 |  | done |  |

#### `online-mind2web` — osunlp/Online-Mind2Web@eacad896a84d

| model_id | subset | metric | value | n | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| cua-sft-v1 | all | parse_failure_rate | 7.4 % |  | done | Denominator is harness-internal; not a per-episode proportion. |
| cua-sft-v1 | all-episodes | verified_success_rate | 20.0 % <sub>[11–33]</sub> | 50 | done | Audited equivalent is 0.16. |
| cua-sft-v1 | easy | verified_success_rate | 36.4 % <sub>[15–65]</sub> | 11 | done | As measured; the audit did not publish a per-level breakdown. |
| cua-sft-v1 | hard | verified_success_rate | 23.1 % <sub>[8–50]</sub> | 13 | done | As measured. |
| cua-sft-v1 | medium | verified_success_rate | 30.0 % <sub>[11–60]</sub> | 10 | done | Base scored 0/10 here; the headline qualitative claim. |
| cua-sft-v1 | parsed-actions | canonical_form_rate | 100.0 % <sub>[99–100]</sub> | 584 | done | 584/584 parsed actions in canonical form. |
| cua-sft-v1 | scored | verified_success_rate | 29.4 % <sub>[17–46]</sub> | 34 | done | 10/34 as measured by the judge before audit. |
| cua-sft-v1 | scored | verified_success_rate_audited | 23.5 % <sub>[12–40]</sub> | 34 | done | 8/34 after manual audit removed 2 confirmed judge false positives. THIS is the number quoted in the thesis and the deck. |
| cua-sft-v1-ep1 | scored | verified_success_rate | 11.8 % <sub>[5–27]</sub> | 34 | done | 4/34. |
| cua-sft-v1-ep2 | scored | verified_success_rate | 14.7 % <sub>[6–30]</sub> | 34 | done | 5/34. Epoch trend is monotone: 0.118 -> 0.147 -> 0.294. |
| qwen35-9b-base | all | parse_failure_rate | 22.3 % |  | done | Denominator is harness-internal; not a per-episode proportion. |
| qwen35-9b-base | all-episodes | verified_success_rate | 6.0 % <sub>[2–16]</sub> | 50 | done | Denominator includes blocked and unreachable episodes. |
| qwen35-9b-base | easy | verified_success_rate | 18.2 % <sub>[5–48]</sub> | 11 | done |  |
| qwen35-9b-base | hard | verified_success_rate | 7.7 % <sub>[1–33]</sub> | 13 | done |  |
| qwen35-9b-base | medium | verified_success_rate | 0.0 % <sub>[0–28]</sub> | 10 | done |  |
| qwen35-9b-base | parsed-actions | canonical_form_rate | 26.8 % <sub>[22–32]</sub> | 276 | done | 74/276 parsed actions in canonical form. |
| qwen35-9b-base | scored | verified_success_rate | 8.8 % <sub>[3–23]</sub> | 34 | done | 3/34 scored episodes; 50 sampled, 14 site-blocked, 2 unreachable. |
| cua-grpo-v1 | scored | verified_success_rate | _not measured_ | 34 | planned | Experimental; keep out of any external presentation until verified. |
| cua-sft-v1-nologin | scored | verified_success_rate | _not measured_ | 34 | planned | Ablation that turns the post-authentication novelty claim into a result. |

#### `qa-bench-refapp` — catalog-v1@0f3d7f88e85a

| model_id | subset | metric | value | n | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5 | 6-defect-subset | verdict_cost_per_case_usd | $0.87 | 6 | done | Now dominates hybrid cost; the remaining lever is a cheaper verdict tier. |
| claude-sonnet-5 | 6-defect-subset | cost_per_case_usd | $1.30 | 6 | done | Execution $1.17 + reporting $0.13. Plus a one-time explore+plan of ~$3.68/app. |
| claude-sonnet-5 | 6-defect-subset | ddr | 100.0 % <sub>[61–100]</sub> | 6 | done | 6/6 via the full vetty agent in native browser-tool mode. |
| claude-sonnet-5 | 6-defect-subset | done_rate | 100.0 % <sub>[61–100]</sub> | 6 | done |  |
| claude-sonnet-5 | 6-defect-subset | execution_cost_per_case_usd | $1.17 | 6 | done | The phase the local model replaces. |
| claude-sonnet-5 | 6-defect-subset | explore_plan_cost_per_app_usd | $3.68 |  | done | One-time per app, shared across all cases and mutants. |
| claude-sonnet-5 | 6-defect-subset | reporting_cost_per_case_usd | $0.13 | 6 | done |  |
| claude-sonnet-5 | phase-executing | phase_call_share | 94.3 % <sub>[92–96]</sub> | 511 | done | 482 of 511 calls. The figure quoted on the architecture slide. |
| claude-sonnet-5 | phase-executing | phase_cost_share | 92.5 % |  | done | Single full-pipeline run, which crashed during reporting, so the reporting share is understated and this is a slight overstatement. Corroborated independently by $1.17 of $1.30 per case in vetty#21. |
| claude-sonnet-5 | phase-exploring | phase_call_share | 3.5 % <sub>[2–5]</sub> | 511 | done |  |
| claude-sonnet-5 | phase-exploring | phase_cost_share | 3.4 % |  | done |  |
| claude-sonnet-5 | phase-planning | phase_call_share | 1.6 % <sub>[1–3]</sub> | 511 | done |  |
| claude-sonnet-5 | phase-planning | phase_cost_share | 2.5 % |  | done |  |
| claude-sonnet-5 | phase-reporting | phase_call_share | 0.6 % <sub>[0–2]</sub> | 511 | done |  |
| claude-sonnet-5 | phase-reporting | phase_cost_share | 1.6 % |  | done | Understated: the run crashed in this phase after 3 calls. |
| cua-sft-v1 | 6-defect-subset | completion_tokens_per_case | 385 | 6 | done | ~6.4 model calls per case. |
| cua-sft-v1 | 6-defect-subset | cost_per_case_usd | $0.88 | 6 | done | Opus-5 verdict $0.87 + Modal L40S ~$0.01. The verdict now dominates; a cheaper verdict tier is the remaining lever. |
| cua-sft-v1 | 6-defect-subset | ddr | 83.3 % <sub>[44–97]</sub> | 6 | done | 5/6. Misses delete_button_no_action (confabulates that the delete succeeded). |
| cua-sft-v1 | 6-defect-subset | done_rate | 100.0 % <sub>[61–100]</sub> | 6 | done | 2-12 steps per case. |
| cua-sft-v1 | 6-defect-subset | execution_cost_per_case_usd | $0.01 | 6 | done | Execution phase only: $1.17 -> ~$0.01, about 100x. Metered natively as of vetty#22. |
| qa-bench-refapp | full-48-case-suite | defect_catalog_size | 12 |  | done |  |
| qa-bench-refapp | full-48-case-suite | defect_classes | 6 |  | done | validation, action, navigation, labeling, data, silent_failure. |
| qa-bench-refapp | full-48-case-suite | planned_test_cases | 48 |  | done | Test cases in the cached plan for the full suite. |
| qa-bench-refapp | full-48-case-suite | scripted_suite_size | 13 |  | done | Locator-based Playwright tests forming the incumbent-method arm. |
| qwen35-9b-base | 6-defect-subset | completion_tokens_per_case | 4600 | 6 | done | Rambles without stopping; ~15 model calls per case. |
| qwen35-9b-base | 6-defect-subset | ddr | 50.0 % <sub>[19–81]</sub> | 6 | done | 3/6. The 3 caught are all statically visible in page text; the model never completes a task. |
| qwen35-9b-base | 6-defect-subset | done_rate | 0.0 % <sub>[0–39]</sub> | 6 | done | All 6 hit the 15-step cap; 1 crashed on a malformed keypress. |
| claude-sonnet-5 | 6-defect-subset | ddr | _not measured_ | 6 | pending | All 6 defect runs plus clean completed; the judge pass has not been run so no DDR exists yet. |
| claude-sonnet-5 | skin-v2 | pass_rate_retained | _not measured_ |  | planned | Agent arm of the same comparison. |
| cua-grpo-v1 | 6-defect-subset | ddr | _not measured_ | 6 | planned | Experimental. |
| cua-sft-v1 | skin-v2 | pass_rate_retained | _not measured_ |  | planned | Needs an ids-only skin to separate DOM change from label change. |
| cua-sft-v1-nologin | 6-defect-subset | ddr | _not measured_ | 6 | planned | Same ablation measured in the QA harness. |
| scripted-playwright | skin-v2 | pass_rate_retained | _not measured_ | 13 | planned | Central thesis claim. Skin v2 renames all 32 ids, so an #id suite should collapse. |
| claude-sonnet-5 | full-48-case-suite | bugs_per_clean_run | 9 | 1 | superseded | 4 of 9 describe interactions never performed (0 POST /settings in the access log); the rest are fixture behaviour, not agent noise. |
| claude-sonnet-5 | full-48-case-suite | cost_per_run_usd | $61.20 | 1 | superseded | Reported $61.20 vs $24.87 by the pricing table; the gap is unreconciled. |
| claude-sonnet-5 | full-48-case-suite | flakiness | 0.0 % | 1 | superseded | Not a measurement: flakiness needs >=2 runs and this had 1, so 0 was the only possible output. |
| claude-sonnet-5 | full-48-case-suite | post_login_share | 85.4 % <sub>[73–93]</sub> | 48 | superseded | Regex heuristic over test-case text rather than an observed auth transition. |
| claude-sonnet-5 | full-48-case-suite | ui_coverage_elements | 92.1 % | 1 | superseded | Same crashed run. |
| claude-sonnet-5 | full-48-case-suite | ui_coverage_views | 88.9 % | 1 | superseded | Single run that crashed in the reporter phase; judge pass never ran. |

#### `screenspot` — rootsautomation/ScreenSpot@test

| model_id | subset | metric | value | n | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| cua-sft-v1 | all | accuracy | 85.0 % <sub>[81–89]</sub> | 300 | done | coord_space=viewport_abs. Out-of-domain: ScreenSpot is desktop, the corpus is web-only. |
| cua-sft-v1 | all | parse_rate | 97.0 % <sub>[94–98]</sub> | 300 | done |  |
| cua-sft-v1 | icon | accuracy | 75.4 % <sub>[67–82]</sub> | 122 | done | Most of the aggregate drop sits here. |
| cua-sft-v1 | text | accuracy | 91.6 % <sub>[87–95]</sub> | 178 | done |  |
| cua-sft-v1-ep1 | all | accuracy | 82.0 % <sub>[77–86]</sub> | 300 | done |  |
| cua-sft-v1-ep1 | all | parse_rate | 97.7 % <sub>[95–99]</sub> | 300 | done |  |
| cua-sft-v1-ep2 | all | accuracy | 86.3 % <sub>[82–90]</sub> | 300 | done |  |
| cua-sft-v1-ep2 | all | parse_rate | 95.7 % <sub>[93–97]</sub> | 300 | done |  |
| qwen35-9b-base | all | accuracy | 89.0 % <sub>[85–92]</sub> | 300 | done | coord_space=normalized_1000 |
| qwen35-9b-base | all | parse_rate | 94.7 % <sub>[92–97]</sub> | 300 | done |  |
| qwen35-9b-base | icon | accuracy | 80.3 % <sub>[72–86]</sub> | 122 | done |  |
| qwen35-9b-base | text | accuracy | 94.9 % <sub>[91–97]</sub> | 178 | done |  |
| cua-grpo-v1 | all | accuracy | _not measured_ | 300 | planned | Experimental. |

### Figures cited from other people's studies

| claim | value | n | venue | year | reference | used_on |
| --- | --- | --- | --- | --- | --- | --- |
| Share of record/replay web test breakages caused by stale locators | 73.6 % | 1065 | ICST | 2016 | 10.1109/ICST.2016.16 | deck slide 3, thesis ch.1 |
| Legacy GUI tests on live sites that stopped passing within 18 months | 100.0 % | 120 | ASE | 2025 | 10.1109/ASE63991.2025.00251 | deck slide 3, thesis ch.1 |

_83 registry rows: 68 done, 1 pending, 8 planned, 6 superseded. Percentages carry a 95 % Wilson interval where n is known._

<!-- registry:end -->

## Evaluation (ScreenSpot grounding)

`eval_screenspot.py` measures **grounding** — given a screenshot and an instruction,
the model predicts a click point and we score whether that point lands inside the
target bounding box. Use it to quantify how much SFT improves grounding by comparing
the base model (before) against the fine-tuned adapter (after).

Install deps on the GPU box, then run:

    make install-eval
    make eval-screenspot

`make eval-screenspot` evaluates every model listed in `eval_models.yaml` on the same
dataset and records the results side by side in `eval_results/` (`<label>.json` per
model plus `comparison.json`), printing a comparison table (overall accuracy, parse
rate, and a per-`data_type` / per-`data_source` breakdown). The default
`eval_models.yaml` compares `before` (base Qwen3.5-9B) vs `after` (base + `./output`
LoRA adapter); add or remove specs freely — any number of models is supported.

Quick start during development (no model needed):

    make test-eval        # full pipeline self-test on synthetic data, no GPU/network

Inference runs by shelling out to the `swift infer` CLI (same way training shells out
to `swift sft`), loading the base model and applying the LoRA adapter from `adapters:`.

**Coordinate handling.** The default `viewport_abs` coordinate space matches how this
model was trained: each eval image is resized to the 1280x720 training viewport,
absolute pixel coordinates are parsed in that space, then linearly scaled back to the
original resolution to test against the bounding box. Resizing non-16:9 screenshots to
1280x720 distorts them — the back-mapping is exact, but model accuracy on distorted
images is a measurement caveat. The base ("before") model is not instruction-tuned and
may emit prose rather than coordinates, yielding a low parse rate and near-zero
accuracy; that is the expected, informative baseline.

**Reusable for other models.** Nothing is hardcoded to this model. Flags select the
inference backend (`--backend`), coordinate convention (`--coord-space`:
`viewport_abs` / `normalized_1000` / `native_abs`), output parser
(`--parse-pattern`, repeatable regexes), bounding-box format
(`--bbox-format xywh|xyxy`, `--bbox-normalized`), and dataset
(`--dataset` / `--split`). The default dataset is **ScreenSpot-v2**
(`os-copilot/ScreenSpot-v2`); the original ScreenSpot and ScreenSpot-Pro are
selectable via `--dataset` (set `--bbox-format`/`--bbox-normalized` to match the
chosen mirror's annotation convention).

## Output Structure

    trajectories/
      task_mind2web_0001/            # from GPT-5.4 path
        metadata.json
        trajectory.json
        step_001_screenshot.png
        step_002_screenshot.png
        ...
      repo_owner_repo/               # from repo pipeline
        task_owner_repo_testname/
          metadata.json
          trajectory.json
          step_001_screenshot.png
          ...

## Training Configuration

See `config.yaml` for default hyperparameters. Key settings from the blueprint:
- Model: Qwen/Qwen3.5-9B-Base
- LoRA rank 64, alpha 128
- Learning rate 5e-5, 3 epochs
- Max sequence length 4096
- Vision encoder frozen (freeze_vit: true)

## Cost Estimates

**GPT-5.4 CUA distillation:**
- ~$0.55 per trajectory
- 1,000 trajectories ≈ $550

**QA repo pipeline:**
- Stages 1–2: free
- Stage 3: ~$0.03/repo (scoring)
- Stages 4–6: ~$0.15–0.60/repo (only repos passing prior gates)
- Think blocks: ~$0.005/step (Sonnet)
- Budget is capped per run — pipeline pauses when limit is hit

**Training:**
- 20-40 GPU hours on A100 80GB (~$16-32 on RunPod)
