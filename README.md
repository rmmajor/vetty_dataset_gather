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
| cua-sft-v2 | sft | Qwen/Qwen3.5-9B-Base | v2 | LoRA r64/a128 3ep lr5e-5 | done | 8482 train + 976 val examples from 738 trainable trajectories. Trained in two halves: Modal H100 to step 1500, resumed on a Vast A100 via --resume-from. eval_loss 0.4581 (epoch 2) / 0.4621 (epoch 3). Answers viewport_abs. SERVE WITH --response_prefix '' or ms-swift prefills an empty <think> and no action line is ever emitted; pass the base explicitly, adapter_config records a stale Modal volume path. |
| cua-sft-v1-nologin | sft | Qwen/Qwen3.5-9B-Base | v1-nologin | LoRA r64/a128 3ep lr5e-5 | planned | Ablation: same recipe with post-authentication trajectories removed. Isolates what the auth-gated data buys. |
| claude-sonnet-5 | frontier |  |  |  | done | Frontier executor inside the full vetty agent (explore/plan/execute/report on the claude-sdk backbone). |
| claude-opus-5 | frontier |  |  |  | done | Used as verdict and match-judge for every QA-bench column, never as an executor. |
| scripted-playwright | tool |  |  | 13 hand-written Playwright tests | done | Incumbent-method comparison arm: locator-based (#id) suite in the vetty benchmark. |
| corpus-v1 | dataset |  | v1 |  | done | Snapshot the SFT v1 model was trained on. Frozen; the deck describes this snapshot. |
| corpus-v2 | dataset |  | v2 |  | done | Frozen snapshot cua-sft-v2 trained on: 738 trainable trajectories -> 9458 examples (8482 train / 976 val), health-gated. Built from corpus-v1 plus 46 audited hindsight relabels, 34 wall demos, 9 recovered orphans, ~74 verified search-lane trajectories, and full thought-polish/instruction-style passes. |
| corpus-current | dataset |  | v3-wip |  | running | Live corpus, still growing past the v2 freeze. 6 search-lane sites remain uncaptured. |
| qa-bench-refapp | benchmark |  |  |  | done | Reference app plus one-defect-per-run mutants; defect injected by config, not by patching code. |
| cua-grpo-v2 | grpo | cua-sft-v1 (merged) | rl-suite f5dda31d | GRPO LoRA r64/a128 15 steps G=4 lr1e-5 beta0.01 | done | First live GRPO run: 240 episodes on 12 train tasks against live sites, reward v2 (docs/decisions/grpo-v2-first-live-run.md). LoRA over the MERGED sft-v1 weights; serve it on those, never on the raw base. Answers in viewport_abs. |
| cua-grpo-v3 | grpo | cua-sft-v2 (merged) | rl-suite 3705ec31 | GRPO LoRA r64/a128 50 steps G=8 lr5e-6 oracle rewards | done | No measurable gain; see docs/decisions/grpo-v3-results.md. 752 oracle-scored episodes on 25 frontier tasks (k=2 band 0.2-0.8) against live sites, temperature 1.0, beta 0.01, max_turns 36. LoRA over the MERGED sft-v2 weights. LoRA-B Frobenius norm 0.5557 (3.1% of sft-v2's 17.78). Never served or benchmarked: RL dropped for the paper. |
| cua-sft-v3 | sft | Qwen/Qwen3.5-9B-Base | v3 | LoRA r64/a128 2ep lr5e-5 | done | Epoch-1 checkpoint (step 1400); epoch 2 trained and REJECTED - eval loss rose 0.5587 -> 0.5791 while train loss kept falling to 0.464. 14h15m wall. First checkpoint on the sft3 two-image observation (640x360 previous thumbnail + current 1280x720) and the first to answer in normalized_1000. Did not deliver its pre-registered bands; see docs/decisions/sft-v3-postmortem.md. |
| corpus-v3 | dataset |  | v3 |  | done | Frozen snapshot cua-sft-v3 trained on. corpus-v2 plus the outcome-relabel pass (every thought opens with the judged outcome of the previous action), QA-voice instruction rewrites, and the qa_verify capture tranche. The outcome-relabel prompt is the round's identified defect: it defined 'outcome' as screen-delta rather than intent-match, so success assertions rose 8.1% -> 49.3%. |

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
| cua-grpo-v2 | all | parse_failure_rate | 8.7 % |  | done | Denominator is harness-internal; not a per-episode proportion. |
| cua-grpo-v2 | all-episodes | done_rate | 22.0 % <sub>[13–35]</sub> | 50 | done | Episodes the policy ended with done(). |
| cua-grpo-v2 | all-episodes | verified_success_rate | 8.0 % <sub>[3–19]</sub> | 50 | done | Denominator includes blocked and unreachable sites. |
| cua-grpo-v2 | easy | verified_success_rate | 27.3 % <sub>[10–57]</sub> | 11 | done | 3/11. |
| cua-grpo-v2 | hard | verified_success_rate | 7.7 % <sub>[1–33]</sub> | 13 | done | 1/13. |
| cua-grpo-v2 | medium | verified_success_rate | 0.0 % <sub>[0–28]</sub> | 10 | done | 0/10. |
| cua-grpo-v2 | parsed-actions | canonical_form_rate | 100.0 % <sub>[99–100]</sub> | 518 | done | 518/518 parsed actions in canonical form. |
| cua-grpo-v2 | scored | verified_success_rate | 11.8 % <sub>[5–27]</sub> | 34 | done | 4/34 scored episodes. Criteria-first judge (judge_prompt_sha), all three policies the same night; not comparable to rows without a judge_prompt_sha. |
| cua-grpo-v2 | successes | steps_over_reference_on_success | 1.12 | 4 | done | Mean steps / reference_length over verified successes. |
| cua-sft-v1 | all | parse_failure_rate | 7.4 % |  | done | Denominator is harness-internal; not a per-episode proportion. |
| cua-sft-v1 | all | parse_failure_rate | 8.0 % |  | done | Denominator is harness-internal; not a per-episode proportion. |
| cua-sft-v1 | all-episodes | done_rate | 16.0 % <sub>[8–29]</sub> | 50 | done | Episodes the policy ended with done(). |
| cua-sft-v1 | all-episodes | verified_success_rate | 20.0 % <sub>[11–33]</sub> | 50 | done | Audited equivalent is 0.16. |
| cua-sft-v1 | all-episodes | verified_success_rate | 6.0 % <sub>[2–16]</sub> | 50 | done | Denominator includes blocked and unreachable sites. |
| cua-sft-v1 | easy | verified_success_rate | 36.4 % <sub>[15–65]</sub> | 11 | done | As measured; the audit did not publish a per-level breakdown. |
| cua-sft-v1 | easy | verified_success_rate | 27.3 % <sub>[10–57]</sub> | 11 | done | 3/11. |
| cua-sft-v1 | hard | verified_success_rate | 23.1 % <sub>[8–50]</sub> | 13 | done | As measured. |
| cua-sft-v1 | hard | verified_success_rate | 0.0 % <sub>[0–24]</sub> | 12 | done | 0/12. |
| cua-sft-v1 | medium | verified_success_rate | 30.0 % <sub>[11–60]</sub> | 10 | done | Base scored 0/10 here; the headline qualitative claim. |
| cua-sft-v1 | medium | verified_success_rate | 0.0 % <sub>[0–30]</sub> | 9 | done | 0/9. |
| cua-sft-v1 | parsed-actions | canonical_form_rate | 100.0 % <sub>[99–100]</sub> | 584 | done | 584/584 parsed actions in canonical form. |
| cua-sft-v1 | parsed-actions | canonical_form_rate | 100.0 % <sub>[99–100]</sub> | 509 | done | 509/509 parsed actions in canonical form. |
| cua-sft-v1 | scored | verified_success_rate | 29.4 % <sub>[17–46]</sub> | 34 | done | 10/34 as measured by the judge before audit. |
| cua-sft-v1 | scored | verified_success_rate | 9.4 % <sub>[3–24]</sub> | 32 | done | 3/32 scored episodes. Criteria-first judge (judge_prompt_sha), all three policies the same night; not comparable to rows without a judge_prompt_sha. |
| cua-sft-v1 | scored | verified_success_rate_audited | 23.5 % <sub>[12–40]</sub> | 34 | done | 8/34 after manual audit removed 2 confirmed judge false positives. THIS is the number quoted in the thesis and the deck. |
| cua-sft-v1 | successes | steps_over_reference_on_success | 1.33 | 3 | done | Mean steps / reference_length over verified successes. |
| cua-sft-v1-ep1 | scored | verified_success_rate | 11.8 % <sub>[5–27]</sub> | 34 | done | 4/34. |
| cua-sft-v1-ep2 | scored | verified_success_rate | 14.7 % <sub>[6–30]</sub> | 34 | done | 5/34. Epoch trend is monotone: 0.118 -> 0.147 -> 0.294. |
| cua-sft-v2 | all | parse_failure_rate | 0.0 % <sub>[0–7]</sub> | 50 | done |  |
| cua-sft-v2 | all-episodes | done_emission_rate | 0.3 | 50 | done |  |
| cua-sft-v2 | all-episodes | verified_success_rate | 8.0 % <sub>[3–19]</sub> | 50 | done |  |
| cua-sft-v2 | easy | verified_success_rate | 27.3 % <sub>[10–57]</sub> | 11 | done |  |
| cua-sft-v2 | hard | verified_success_rate | 0.0 % <sub>[0–23]</sub> | 13 | done |  |
| cua-sft-v2 | medium | verified_success_rate | 11.1 % <sub>[2–44]</sub> | 9 | done |  |
| cua-sft-v2 | parsed-actions | canonical_form_rate | 100.0 % <sub>[99–100]</sub> | 644 | done |  |
| cua-sft-v2 | scored | verified_success_rate | 12.1 % <sub>[5–27]</sub> | 33 | done | sft-v2 on the archived 50-task subset, same night and same judge as om2w-base-v2-50. Success delta is inside noise at n=33; parse-failure and done-rate deltas are not. |
| qwen35-9b-base | all | parse_failure_rate | 22.3 % |  | done | Denominator is harness-internal; not a per-episode proportion. |
| qwen35-9b-base | all | parse_failure_rate | 20.6 % |  | done | Denominator is harness-internal; not a per-episode proportion. |
| qwen35-9b-base | all | parse_failure_rate | 23.6 % <sub>[19–30]</sub> | 230 | done |  |
| qwen35-9b-base | all | parse_failure_rate | 37.7 % <sub>[26–52]</sub> | 50 | done |  |
| qwen35-9b-base | all-episodes | done_emission_rate | 0.113 | 230 | done |  |
| qwen35-9b-base | all-episodes | done_emission_rate | 0.12 | 50 | done |  |
| qwen35-9b-base | all-episodes | done_rate | 22.0 % <sub>[13–35]</sub> | 50 | done | Episodes the policy ended with done(). |
| qwen35-9b-base | all-episodes | verified_success_rate | 6.0 % <sub>[2–16]</sub> | 50 | done | Denominator includes blocked and unreachable episodes. |
| qwen35-9b-base | all-episodes | verified_success_rate | 10.0 % <sub>[4–21]</sub> | 50 | done | Denominator includes blocked and unreachable sites. |
| qwen35-9b-base | all-episodes | verified_success_rate | 4.8 % <sub>[3–8]</sub> | 230 | done |  |
| qwen35-9b-base | all-episodes | verified_success_rate | 6.0 % <sub>[2–16]</sub> | 50 | done |  |
| qwen35-9b-base | easy | verified_success_rate | 18.2 % <sub>[5–48]</sub> | 11 | done |  |
| qwen35-9b-base | easy | verified_success_rate | 36.4 % <sub>[15–65]</sub> | 11 | done | 4/11. |
| qwen35-9b-base | easy | verified_success_rate | 25.0 % <sub>[15–39]</sub> | 44 | done |  |
| qwen35-9b-base | easy | verified_success_rate | 27.3 % <sub>[10–57]</sub> | 11 | done |  |
| qwen35-9b-base | hard | verified_success_rate | 7.7 % <sub>[1–33]</sub> | 13 | done |  |
| qwen35-9b-base | hard | verified_success_rate | 7.7 % <sub>[1–33]</sub> | 13 | done | 1/13. |
| qwen35-9b-base | hard | verified_success_rate | 0.0 % <sub>[0–7]</sub> | 49 | done |  |
| qwen35-9b-base | hard | verified_success_rate | 0.0 % <sub>[0–23]</sub> | 13 | done |  |
| qwen35-9b-base | medium | verified_success_rate | 0.0 % <sub>[0–28]</sub> | 10 | done |  |
| qwen35-9b-base | medium | verified_success_rate | 0.0 % <sub>[0–28]</sub> | 10 | done | 0/10. |
| qwen35-9b-base | medium | verified_success_rate | 0.0 % <sub>[0–5]</sub> | 71 | done |  |
| qwen35-9b-base | medium | verified_success_rate | 0.0 % <sub>[0–28]</sub> | 10 | done |  |
| qwen35-9b-base | parsed-actions | canonical_form_rate | 26.8 % <sub>[22–32]</sub> | 276 | done | 74/276 parsed actions in canonical form. |
| qwen35-9b-base | parsed-actions | canonical_form_rate | 18.8 % <sub>[15–24]</sub> | 282 | done | 53/282 parsed actions in canonical form. |
| qwen35-9b-base | parsed-actions | canonical_form_rate | 29.7 % <sub>[27–32]</sub> | 1351 | done |  |
| qwen35-9b-base | parsed-actions | canonical_form_rate | 25.5 % <sub>[19–33]</sub> | 149 | done |  |
| qwen35-9b-base | scored | verified_success_rate | 8.8 % <sub>[3–23]</sub> | 34 | done | 3/34 scored episodes; 50 sampled, 14 site-blocked, 2 unreachable. |
| qwen35-9b-base | scored | verified_success_rate | 14.7 % <sub>[6–30]</sub> | 34 | done | 5/34 scored episodes. Criteria-first judge (judge_prompt_sha), all three policies the same night; not comparable to rows without a judge_prompt_sha. |
| qwen35-9b-base | scored | verified_success_rate | 6.7 % <sub>[4–12]</sub> | 164 | done | Base re-measured on the 230-task pinned set under the navigate/handoff-aware harness. Not comparable to pre-2026-08-31 rows: the action grammar and history_window changed. |
| qwen35-9b-base | scored | verified_success_rate | 8.8 % <sub>[3–23]</sub> | 34 | done | Same base run, restricted to the archived 50-task subset so sft-v2 has a like-for-like comparator. Base scored 0.1471 on these same 50 on 2026-08-27, so ~6pp is run-to-run noise. |
| qwen35-9b-base | successes | steps_over_reference_on_success | 1.92 | 5 | done | Mean steps / reference_length over verified successes. |
| cua-sft-v1-nologin | scored | verified_success_rate | _not measured_ | 34 | planned | Ablation that turns the post-authentication novelty claim into a result. |

#### `qa-bench-refapp` — catalog-v1@0f3d7f88e85a, catalog-v2@cb427058ac9e

| model_id | subset | metric | value | n | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5 | 6-defect-subset | verdict_cost_per_case_usd | $0.87 | 6 | done | Now dominates hybrid cost; the remaining lever is a cheaper verdict tier. |
| claude-sonnet-5 | 6-defect-subset | cost_per_case_usd | $1.30 | 6 | done | Execution $1.17 + reporting $0.13. Plus a one-time explore+plan of ~$3.68/app. |
| claude-sonnet-5 | 6-defect-subset | ddr | 100.0 % <sub>[61–100]</sub> | 6 | done | 6/6 via the full vetty agent in native browser-tool mode. |
| claude-sonnet-5 | 6-defect-subset | done_rate | 100.0 % <sub>[61–100]</sub> | 6 | done |  |
| claude-sonnet-5 | 6-defect-subset | execution_cost_per_case_usd | $1.17 | 6 | done | The phase the local model replaces. |
| claude-sonnet-5 | 6-defect-subset | explore_plan_cost_per_app_usd | $3.68 |  | done | One-time per app, shared across all cases and mutants. |
| claude-sonnet-5 | 6-defect-subset | reporting_cost_per_case_usd | $0.13 | 6 | done |  |
| claude-sonnet-5 | cost-model | e2e_cost_per_case_usd | $1.30 |  | done | FRONTIER CONFIGURATION = every phase on claude-sonnet-5, the reference the hybrid is compared against. Excludes the DDR match-judge, which is benchmark apparatus and has never entered any cost figure. |
| claude-sonnet-5 | cost-model | execution_cost_per_case_usd_e2e | 1.17 |  | done | FRONTIER CONFIGURATION = every phase on claude-sonnet-5, the reference the hybrid is compared against. Excludes the DDR match-judge, which is benchmark apparatus and has never entered any cost figure. |
| claude-sonnet-5 | cost-model | report_cost_per_case_usd | $0.13 |  | done | FRONTIER CONFIGURATION = every phase on claude-sonnet-5, the reference the hybrid is compared against. Excludes the DDR match-judge, which is benchmark apparatus and has never entered any cost figure. |
| claude-sonnet-5 | economy-7run-105case | ddr | 66.7 % <sub>[57–75]</sub> | 105 | done | DDR strict over 33 catalogue defects. Arm 'sonnet'. |
| claude-sonnet-5 | economy-7run-105case | error_verdict_rate | 0.0286 | 105 | done | Share of 105 case-runs the verdict tier could not judge - the execution-competence number. Arm 'sonnet'. |
| claude-sonnet-5 | economy-7run-105case | executor_cost_per_case_usd | $1.40 | 105 | done | Frontier executor cost per case (executing $139.348 + reporting $7.266 over 105 cases). Claude Agent SDK's own cost estimate on a Max subscription via the logged-in CLI - notional, not billed, and known to drift. Reported as an estimate, never as a measurement. |
| claude-sonnet-5 | economy-7run-105case | executor_input_tokens_per_case | 22 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| claude-sonnet-5 | economy-7run-105case | executor_output_tokens_per_case | 3875 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| claude-sonnet-5 | economy-7run-105case | precision_strict | 0.7708 | 105 | done | Share of filed bug reports that matched a real catalogue defect. Arm 'sonnet'. |
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
| cua-sft-v1 | economy-7run-105case | ddr | 27.3 % <sub>[20–36]</sub> | 105 | done | DDR strict over 33 catalogue defects, economy sweep. |
| cua-sft-v1 | economy-7run-105case | error_verdict_rate | 0.6 | 105 | done | Share of the 105 case-runs the verdict tier could not judge - the execution-competence number. |
| cua-sft-v1 | economy-7run-105case | executor_input_tokens_per_case | 11920 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v1 | economy-7run-105case | executor_output_tokens_per_case | 517 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v1 | economy-7run-105case | executor_turns_per_case | 9 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v1 | economy-7run-105case | gpu_cost_per_case_usd | $0.01 | 105 | done | Derived, not self-reported: phase_wall_seconds.executing x Modal L40S $1.95/GPU-hour (benchmark/cost.py, wired into the report 2026-09-04). This is the local arm's REAL cost; its tokens are free because the model is self-hosted. |
| cua-sft-v1 | economy-7run-105case | verdict_cost_per_run_usd | $10.49 | 105 | done | Verdict-tier cost, NOT model cost: executor runs on our own GPU at $0. sft-v3 ran on vetty 428c062 and the others on 409810f, so the per-run cost is confounded by harness changes and is not comparable across arms. |
| cua-sft-v2 | cost-model | e2e_cost_per_case_usd | $0.89 |  | done | HYBRID CONFIGURATION = the proposed architecture: local model executes, frontier model draws the conclusions. Composed from a measured local execution cost (GPU seconds x L40S $1.95/h) and a measured frontier verdict cost - NOT one instrumented end-to-end run, because the shipped product routes its conclusion steps to the local model too. Must be re-measured end to end once LocalModelAgentAdapter takes a frontier model_override. See docs/decisions/hybrid-cost-and-the-product-gap.md |
| cua-sft-v2 | cost-model | e2e_saving_vs_frontier | 1.46 |  | done | Frontier $1.30 / hybrid $0.89. Execution alone falls ~78x; the frontier conclusion step now dominates. HYBRID CONFIGURATION = the proposed architecture: local model executes, frontier model draws the conclusions. Composed from a measured local execution cost (GPU seconds x L40S $1.95/h) and a measured frontier verdict cost - NOT one instrumented end-to-end run, because the shipped product routes its conclusion steps to the local model too. Must be re-measured end to end once LocalModelAgentAdapter takes a frontier model_override. See docs/decisions/hybrid-cost-and-the-product-gap.md |
| cua-sft-v2 | cost-model | execution_cost_per_case_usd_e2e | 0.015 |  | done | HYBRID CONFIGURATION = the proposed architecture: local model executes, frontier model draws the conclusions. Composed from a measured local execution cost (GPU seconds x L40S $1.95/h) and a measured frontier verdict cost - NOT one instrumented end-to-end run, because the shipped product routes its conclusion steps to the local model too. Must be re-measured end to end once LocalModelAgentAdapter takes a frontier model_override. See docs/decisions/hybrid-cost-and-the-product-gap.md |
| cua-sft-v2 | cost-model | report_cost_per_case_usd | $0.87 |  | done | HYBRID CONFIGURATION = the proposed architecture: local model executes, frontier model draws the conclusions. Composed from a measured local execution cost (GPU seconds x L40S $1.95/h) and a measured frontier verdict cost - NOT one instrumented end-to-end run, because the shipped product routes its conclusion steps to the local model too. Must be re-measured end to end once LocalModelAgentAdapter takes a frontier model_override. See docs/decisions/hybrid-cost-and-the-product-gap.md |
| cua-sft-v2 | economy-7run-105case | ddr | 21.2 % <sub>[14–30]</sub> | 105 | done | DDR strict over 33 catalogue defects, economy sweep. |
| cua-sft-v2 | economy-7run-105case | ddr | 9.1 % <sub>[5–16]</sub> | 105 | done | DDR strict over 33 catalogue defects. Arm 'matched-sft-v2'. |
| cua-sft-v2 | economy-7run-105case | error_verdict_rate | 0.6286 | 105 | done | Share of the 105 case-runs the verdict tier could not judge - the execution-competence number. |
| cua-sft-v2 | economy-7run-105case | error_verdict_rate | 0.619 | 105 | done | Share of 105 case-runs the verdict tier could not judge - the execution-competence number. Arm 'matched-sft-v2'. |
| cua-sft-v2 | economy-7run-105case | executor_input_tokens_per_case | 12315 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v2 | economy-7run-105case | executor_input_tokens_per_case | 16541 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v2 | economy-7run-105case | executor_output_tokens_per_case | 538 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v2 | economy-7run-105case | executor_output_tokens_per_case | 708 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v2 | economy-7run-105case | executor_turns_per_case | 9.2 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v2 | economy-7run-105case | gpu_cost_per_case_usd | $0.02 | 105 | done | Derived, not self-reported: phase_wall_seconds.executing x Modal L40S $1.95/GPU-hour (benchmark/cost.py, wired into the report 2026-09-04). This is the local arm's REAL cost; its tokens are free because the model is self-hosted. |
| cua-sft-v2 | economy-7run-105case | gpu_cost_per_case_usd | $0.01 | 105 | done | Derived, not self-reported: phase_wall_seconds.executing x Modal L40S $1.95/GPU-hour (benchmark/cost.py, wired into the report 2026-09-04). This is the local arm's REAL cost; its tokens are free because the model is self-hosted. |
| cua-sft-v2 | economy-7run-105case | precision_strict | 1 | 105 | done | Share of filed bug reports that matched a real catalogue defect. Arm 'matched-sft-v2'. |
| cua-sft-v2 | economy-7run-105case | verdict_cost_per_run_usd | $10.68 | 105 | done | Verdict-tier cost, NOT model cost: executor runs on our own GPU at $0. sft-v3 ran on vetty 428c062 and the others on 409810f, so the per-run cost is confounded by harness changes and is not comparable across arms. |
| cua-sft-v3 | economy-7run-105case | ddr | 21.2 % <sub>[14–30]</sub> | 105 | done | DDR strict over 33 catalogue defects, economy sweep. |
| cua-sft-v3 | economy-7run-105case | error_verdict_rate | 0.5429 | 105 | done | Share of the 105 case-runs the verdict tier could not judge - the execution-competence number. |
| cua-sft-v3 | economy-7run-105case | executor_input_tokens_per_case | 18797 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v3 | economy-7run-105case | executor_output_tokens_per_case | 746 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v3 | economy-7run-105case | executor_turns_per_case | 11.3 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v3 | economy-7run-105case | gpu_cost_per_case_usd | $0.02 | 105 | done | Derived, not self-reported: phase_wall_seconds.executing x Modal L40S $1.95/GPU-hour (benchmark/cost.py, wired into the report 2026-09-04). This is the local arm's REAL cost; its tokens are free because the model is self-hosted. |
| cua-sft-v3 | economy-7run-105case | verdict_cost_per_run_usd | $0.73 | 105 | done | Verdict-tier cost, NOT model cost: executor runs on our own GPU at $0. sft-v3 ran on vetty 428c062 and the others on 409810f, so the per-run cost is confounded by harness changes and is not comparable across arms. |
| qa-bench-refapp | full-48-case-suite | defect_catalog_size | 12 |  | done |  |
| qa-bench-refapp | full-48-case-suite | defect_classes | 6 |  | done | validation, action, navigation, labeling, data, silent_failure. |
| qa-bench-refapp | full-48-case-suite | planned_test_cases | 48 |  | done | Test cases in the cached plan for the full suite. |
| qa-bench-refapp | full-48-case-suite | scripted_suite_size | 13 |  | done | Locator-based Playwright tests forming the incumbent-method arm. |
| qwen35-9b-base | 6-defect-subset | completion_tokens_per_case | 4600 | 6 | done | Rambles without stopping; ~15 model calls per case. |
| qwen35-9b-base | 6-defect-subset | ddr | 50.0 % <sub>[19–81]</sub> | 6 | done | 3/6. The 3 caught are all statically visible in page text; the model never completes a task. |
| qwen35-9b-base | 6-defect-subset | done_rate | 0.0 % <sub>[0–39]</sub> | 6 | done | All 6 hit the 15-step cap; 1 crashed on a malformed keypress. |
| qwen35-9b-base | economy-7run-105case | ddr | 0.0 % <sub>[0–4]</sub> | 105 | done | DDR strict over 33 catalogue defects, economy sweep. |
| qwen35-9b-base | economy-7run-105case | ddr | 0.0 % <sub>[0–4]</sub> | 105 | done | DDR strict over 33 catalogue defects. Arm 'matched-base'. |
| qwen35-9b-base | economy-7run-105case | error_verdict_rate | 0.7333 | 105 | done | Share of the 105 case-runs the verdict tier could not judge - the execution-competence number. |
| qwen35-9b-base | economy-7run-105case | error_verdict_rate | 0.6667 | 105 | done | Share of 105 case-runs the verdict tier could not judge - the execution-competence number. Arm 'matched-base'. |
| qwen35-9b-base | economy-7run-105case | executor_input_tokens_per_case | 19386 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| qwen35-9b-base | economy-7run-105case | executor_input_tokens_per_case | 37135 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| qwen35-9b-base | economy-7run-105case | executor_output_tokens_per_case | 3617 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| qwen35-9b-base | economy-7run-105case | executor_output_tokens_per_case | 9738 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| qwen35-9b-base | economy-7run-105case | executor_turns_per_case | 15 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| qwen35-9b-base | economy-7run-105case | gpu_cost_per_case_usd | $0.13 | 105 | done | Derived, not self-reported: phase_wall_seconds.executing x Modal L40S $1.95/GPU-hour (benchmark/cost.py, wired into the report 2026-09-04). This is the local arm's REAL cost; its tokens are free because the model is self-hosted. |
| qwen35-9b-base | economy-7run-105case | gpu_cost_per_case_usd | $0.03 | 105 | done | Derived, not self-reported: phase_wall_seconds.executing x Modal L40S $1.95/GPU-hour (benchmark/cost.py, wired into the report 2026-09-04). This is the local arm's REAL cost; its tokens are free because the model is self-hosted. |
| qwen35-9b-base | economy-7run-105case | precision_strict | 0 | 105 | done | Share of filed bug reports that matched a real catalogue defect. Arm 'matched-base'. |
| qwen35-9b-base | economy-7run-105case | verdict_cost_per_run_usd | $1.32 | 105 | done | Verdict-tier cost, NOT model cost: executor runs on our own GPU at $0. sft-v3 ran on vetty 428c062 and the others on 409810f, so the per-run cost is confounded by harness changes and is not comparable across arms. |
| claude-sonnet-5 | 6-defect-subset | ddr | _not measured_ | 6 | pending | All 6 defect runs plus clean completed; the judge pass has not been run so no DDR exists yet. |
| claude-sonnet-5 | skin-v2 | pass_rate_retained | _not measured_ |  | planned | Agent arm of the same comparison. |
| cua-sft-v1 | skin-v2 | pass_rate_retained | _not measured_ |  | planned | Needs an ids-only skin to separate DOM change from label change. |
| cua-sft-v1-nologin | 6-defect-subset | ddr | _not measured_ | 6 | planned | Same ablation measured in the QA harness. |
| scripted-playwright | skin-v2 | pass_rate_retained | _not measured_ | 13 | planned | Central thesis claim. Skin v2 renames all 32 ids, so an #id suite should collapse. |
| claude-sonnet-5 | full-48-case-suite | bugs_per_clean_run | 9 | 1 | superseded | 4 of 9 describe interactions never performed (0 POST /settings in the access log); the rest are fixture behaviour, not agent noise. |
| claude-sonnet-5 | full-48-case-suite | cost_per_run_usd | $61.20 | 1 | superseded | Reported $61.20 vs $24.87 by the pricing table; the gap is unreconciled. |
| claude-sonnet-5 | full-48-case-suite | flakiness | 0.0 % | 1 | superseded | Not a measurement: flakiness needs >=2 runs and this had 1, so 0 was the only possible output. |
| claude-sonnet-5 | full-48-case-suite | post_login_share | 85.4 % <sub>[73–93]</sub> | 48 | superseded | Regex heuristic over test-case text rather than an observed auth transition. |
| claude-sonnet-5 | full-48-case-suite | ui_coverage_elements | 92.1 % | 1 | superseded | Same crashed run. |
| claude-sonnet-5 | full-48-case-suite | ui_coverage_views | 88.9 % | 1 | superseded | Single run that crashed in the reporter phase; judge pass never ran. |
| cua-grpo-v2 | 6-defect-subset | ddr | _not measured_ | 6 | superseded | RL dropped for the paper after grpo-v3 showed no measurable gain (docs/decisions/grpo-v3-results.md); no GRPO checkpoint will be evaluated. |

#### `rl-train-pool` — frontier-25@3705ec31

| model_id | subset | metric | value | n | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| cua-grpo-v3 | adapter | lora_b_frobenius | 0.5557 |  | done | Frobenius norm of all lora_B tensors after 50 steps: 0.19 at step 5 and saturating with the cosine schedule; 3.1% of the sft-v2 adapter's 17.7846. grpo-v2 measured 0.279 (1.9% of sft-v1) and flat. |
| cua-grpo-v3 | all-episodes | false_done_rate | 31.6 % <sub>[28–35]</sub> | 752 | done | Within-run number: 238/752 rollouts ended with done() and failed the oracle; 27.5% in the first 160, 38.8% in the last 160. done() rate 84.2% against 53.7% success. |
| cua-grpo-v3 | all-episodes | oracle_success_rate | 53.7 % <sub>[50–57]</sub> | 752 | done | Within-run oracle number, not a benchmark: 404/752 rollouts; the 30 blocked episodes stay in the denominator because they trained. |
| cua-grpo-v3 | first-160-episodes | mean_reward | 1.064 | 160 | done | Within-run number: mean composed reward (rl/reward_config.json, autonomous lane) over the first 160 rollouts, sd 0.894. |
| cua-grpo-v3 | first-160-episodes | oracle_success_rate | 59.4 % <sub>[52–67]</sub> | 160 | done | Within-run oracle number, not a benchmark: the first 160 training rollouts (10 steps x 16) ordered by episode mtime, success_check verdicts only, live sites at temperature 1.0. |
| cua-grpo-v3 | last-160-episodes | mean_reward | 0.84 | 160 | done | Within-run number: last 160 rollouts, sd 0.930; 1.064 -> 0.840 is z=2.2. |
| cua-grpo-v3 | last-160-episodes | oracle_success_rate | 50.0 % <sub>[42–58]</sub> | 160 | done | Within-run oracle number, not a benchmark: the last 160 training rollouts. 59.4% -> 50.0% is z=1.7 against the first window. |

#### `screenspot` — rootsautomation/ScreenSpot@test

| model_id | subset | metric | value | n | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| cua-grpo-v2 | all | accuracy | 85.0 % <sub>[81–89]</sub> | 300 | done | coord_space=viewport_abs. Merged SFT + GRPO LoRA, viewport_abs. No grounding regression against sft-v1. |
| cua-grpo-v2 | all | parse_rate | 96.7 % <sub>[94–98]</sub> | 300 | done |  |
| cua-grpo-v2 | full-split | accuracy | 78.6 % <sub>[76–81]</sub> | 1272 | done | Full 1,272-sample test split (desktop + mobile + web); the desktop-300 rows are the comparable ones. |
| cua-grpo-v2 | icon | accuracy | 75.4 % <sub>[67–82]</sub> | 122 | done |  |
| cua-grpo-v2 | text | accuracy | 91.6 % <sub>[87–95]</sub> | 178 | done |  |
| cua-sft-v1 | all | accuracy | 85.0 % <sub>[81–89]</sub> | 300 | done | coord_space=viewport_abs. Out-of-domain: ScreenSpot is desktop, the corpus is web-only. |
| cua-sft-v1 | all | accuracy | 85.3 % <sub>[81–89]</sub> | 300 | done | coord_space=viewport_abs. Base + SFT LoRA served, viewport_abs, same night. |
| cua-sft-v1 | all | accuracy | 85.3 % <sub>[81–89]</sub> | 300 | done | coord_space=viewport_abs. Merged SFT weights served without a LoRA; equals base + LoRA, so the merge the RL run trained on is faithful. |
| cua-sft-v1 | all | parse_rate | 97.0 % <sub>[94–98]</sub> | 300 | done |  |
| cua-sft-v1 | all | parse_rate | 97.3 % <sub>[95–99]</sub> | 300 | done |  |
| cua-sft-v1 | all | parse_rate | 97.3 % <sub>[95–99]</sub> | 300 | done |  |
| cua-sft-v1 | icon | accuracy | 75.4 % <sub>[67–82]</sub> | 122 | done | Most of the aggregate drop sits here. |
| cua-sft-v1 | icon | accuracy | 75.4 % <sub>[67–82]</sub> | 122 | done |  |
| cua-sft-v1 | icon | accuracy | 76.2 % <sub>[68–83]</sub> | 122 | done |  |
| cua-sft-v1 | text | accuracy | 91.6 % <sub>[87–95]</sub> | 178 | done |  |
| cua-sft-v1 | text | accuracy | 92.1 % <sub>[87–95]</sub> | 178 | done |  |
| cua-sft-v1 | text | accuracy | 91.6 % <sub>[87–95]</sub> | 178 | done |  |
| cua-sft-v1-ep1 | all | accuracy | 82.0 % <sub>[77–86]</sub> | 300 | done |  |
| cua-sft-v1-ep1 | all | parse_rate | 97.7 % <sub>[95–99]</sub> | 300 | done |  |
| cua-sft-v1-ep2 | all | accuracy | 86.3 % <sub>[82–90]</sub> | 300 | done |  |
| cua-sft-v1-ep2 | all | parse_rate | 95.7 % <sub>[93–97]</sub> | 300 | done |  |
| cua-sft-v2 | all | accuracy | 76.3 % <sub>[74–79]</sub> | 1272 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. |
| cua-sft-v2 | all | parse_rate | 91.8 % <sub>[90–93]</sub> | 1272 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. |
| cua-sft-v2 | icon | accuracy | 67.5 % <sub>[64–71]</sub> | 575 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. |
| cua-sft-v2 | text | accuracy | 83.5 % <sub>[81–86]</sub> | 697 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. |
| cua-sft-v3 | all | accuracy | 91.6 % <sub>[90–93]</sub> | 1272 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. |
| cua-sft-v3 | all | parse_rate | 100.0 % <sub>[100–100]</sub> | 1272 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. |
| cua-sft-v3 | icon | accuracy | 88.0 % <sub>[85–90]</sub> | 575 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. |
| cua-sft-v3 | text | accuracy | 94.5 % <sub>[93–96]</sub> | 697 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. |
| qwen35-9b-base | all | accuracy | 89.0 % <sub>[85–92]</sub> | 300 | done | coord_space=normalized_1000 |
| qwen35-9b-base | all | accuracy | 89.0 % <sub>[85–92]</sub> | 300 | done | coord_space=normalized_1000. Served stock base, normalized_1000, same night as the grpo-v2 rows. |
| qwen35-9b-base | all | accuracy | 60.9 % <sub>[58–64]</sub> | 1272 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. BASE PARSE RATE IS A FLOOR: this run predates the parser accepting the base's own `click(<x, y>)` spelling, which recovers 163 of its 343 unparsed answers; the corrected base re-run was killed before it finished, so accuracy here understates the base. |
| qwen35-9b-base | all | parse_rate | 94.7 % <sub>[92–97]</sub> | 300 | done |  |
| qwen35-9b-base | all | parse_rate | 94.3 % <sub>[91–96]</sub> | 300 | done |  |
| qwen35-9b-base | all | parse_rate | 73.0 % <sub>[71–75]</sub> | 1272 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. BASE PARSE RATE IS A FLOOR: this run predates the parser accepting the base's own `click(<x, y>)` spelling, which recovers 163 of its 343 unparsed answers; the corrected base re-run was killed before it finished, so accuracy here understates the base. |
| qwen35-9b-base | icon | accuracy | 80.3 % <sub>[72–86]</sub> | 122 | done |  |
| qwen35-9b-base | icon | accuracy | 80.3 % <sub>[72–86]</sub> | 122 | done |  |
| qwen35-9b-base | icon | accuracy | 58.1 % <sub>[54–62]</sub> | 575 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. BASE PARSE RATE IS A FLOOR: this run predates the parser accepting the base's own `click(<x, y>)` spelling, which recovers 163 of its 343 unparsed answers; the corrected base re-run was killed before it finished, so accuracy here understates the base. |
| qwen35-9b-base | text | accuracy | 94.9 % <sub>[91–97]</sub> | 178 | done |  |
| qwen35-9b-base | text | accuracy | 94.9 % <sub>[91–97]</sub> | 178 | done |  |
| qwen35-9b-base | text | accuracy | 63.3 % <sub>[60–67]</sub> | 697 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. BASE PARSE RATE IS A FLOOR: this run predates the parser accepting the base's own `click(<x, y>)` spelling, which recovers 163 of its 343 unparsed answers; the corrected base re-run was killed before it finished, so accuracy here understates the base. |

### Figures cited from other people's studies

| claim | value | n | venue | year | reference | used_on |
| --- | --- | --- | --- | --- | --- | --- |
| Share of record/replay web test breakages caused by stale locators | 73.6 % | 1065 | ICST | 2016 | 10.1109/ICST.2016.16 | deck slide 3, thesis ch.1 |
| Legacy GUI tests on live sites that stopped passing within 18 months | 100.0 % | 120 | ASE | 2025 | 10.1109/ASE63991.2025.00251 | deck slide 3, thesis ch.1 |

_221 registry rows: 208 done, 1 pending, 5 planned, 7 superseded. Percentages carry a 95 % Wilson interval where n is known._

<!-- registry:end -->

## Evaluation (ScreenSpot grounding)

`eval_screenspot.py` measures **grounding** — given a screenshot and an instruction, the
model predicts a click point and we score whether it lands inside the target bounding box.
It is how you quantify what SFT actually bought you.

    make install-eval          # eval deps (GPU box)
    make eval-screenspot       # evals every model in eval_models.yaml -> eval_results/
    make test-eval             # harness self-test, no model/GPU/network

### Measured results

Baseline for the untrained starting checkpoint, on `rootsautomation/ScreenSpot` (`test`):

| model | accuracy | parse rate | text | icon | n |
|-------|---------:|-----------:|-----:|-----:|--:|
| `Qwen/Qwen3.5-9B` (instruct, no SFT) | **85.0%** | 93% | 96.0% | 86.0% | 100 |

**85% is the bar SFT has to beat** — the starting checkpoint is already a strong grounder.

### Coordinate conventions — read this before trusting any number

Getting these wrong does **not** crash the eval. It reports a confident, completely wrong
number. All three were verified against real model output, not assumed:

- **`--coord-space` is per-model, not global.** A stock Qwen emits **`normalized_1000`**
  (0–1000). A model fine-tuned on this project's corpus emits **`viewport_abs`** (1280x720
  pixels, the training viewport). A before/after comparison therefore needs a *different*
  coord space per spec — set `coord_space:` on the spec in `eval_models.yaml`. Scoring the
  same base-model predictions as `viewport_abs` instead of `normalized_1000` drops the score
  from 85% to 25%.
- **`--bbox-normalized`** — `rootsautomation/ScreenSpot` stores bbox in 0–1, not pixels.
  Omit this flag and every target collapses to a sub-pixel box at the origin: **0% accuracy
  by construction, regardless of the model.**
- **`--bbox-format xyxy`** — that mirror stores `(x1,y1,x2,y2)`, not `xywh`.

The defaults now match the default dataset. If you point `--dataset` somewhere else, re-check
all three against that mirror's annotation convention.

## Remote training / eval on a rented GPU (Vast.ai)

`make train` assumes a GPU box you manage. To rent one instead — provision → ship code/data →
run → pull results → **destroy the instance** — use the provider-pluggable `remote_train/`
package:

    make install-remote                 # one-time: pip install the vastai CLI
    export VAST_API_KEY=...             # required
    make remote-train-dry               # dry run: cheapest allowed offer, rents nothing
    make remote-eval EVAL_MODEL=Qwen/Qwen3.5-9B EVAL_LIMIT=100   # benchmark, no training
    make remote-train                   # full SFT, pulls the adapter back
    make remote-train-cleanup           # safety net: destroy any leftover instance

Two jobs share one lifecycle: `train` (SFT) and `eval` (benchmark only). Adding a benchmark is
one `Benchmark` entry in `remote_train/benchmarks.py`; adding a provider (RunPod, Lambda) is one
module implementing `ComputeProvider` plus a line in `PROVIDERS`. The lifecycle and CLI never change.

**Teardown is guaranteed** — the instance dies in a `finally`, on SIGINT/SIGTERM/SIGHUP, via a
`destroy` that polls until the box is *confirmed gone*, via a crash-safe state file + orphan
sweep, and via a max-runtime watchdog. On a shared/team account the sweep only ever auto-destroys
a box this tool provisioned; anything else carrying the label is reported, never killed.

Offer selection is gated on `compute_cap >= 800` (bf16 needs Ampere+; the cheapest cards are
Turing and **cannot** run this workload), on enough `disk_space` to hold the weights, and on an
`exclude_countries` deny-list (default `RU, CN`). Every instance is labeled `roman.maior`.

### Cost per run (measured, Vast.ai)

| what | GPU | $/hr | wall time | cost |
|------|-----|-----:|----------:|-----:|
| ScreenSpot eval, 9B, 100 samples | A100 PCIE 40GB | $0.401 | ~31 min | **~$0.21** |
| — of which: host image pull + boot | | | ~11 min | ~$0.07 |
| — of which: pip install + 18GB model pull | | | ~15 min | ~$0.10 |
| — of which: actual inference (100 samples) | | | ~4 min | ~$0.03 |
| Failed/aborted attempt (dies in <5 min) | RTX A6000 48GB | $0.428 | 2–3 min | ~$0.02 |
| Teardown smoke-check (rent + immediately destroy) | cheapest | $0.048 | ~20 s | ~$0.001 |

Bandwidth is billed separately (~$2.67/TB down); an 18GB model pull adds roughly $0.05.
Boot + install dominate a short eval — the marginal cost of *more samples* is small, so prefer
one larger run over several small ones. A full 1272-sample ScreenSpot sweep is ~$0.35–0.50.

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
- Model: Qwen/Qwen3.5-9B (post-trained checkpoint; the eval baseline is measured on it)
- LoRA rank 64, alpha 128
- Learning rate 5e-5, 3 epochs
- Max sequence length 4096
- Vision encoder frozen (freeze_vit: true)

## Evaluation: Online-Mind2Web baseline

> **Raw records for every figure in this section and the three below are committed to
> [`eval_archive/`](eval_archive/)** — 300 per-sample ScreenSpot records per model, both 50-episode
> Online-Mind2Web runs (`results.json`, `report.md`, `run.log`), and a `manifest.json` carrying the
> headline figures plus a sha256 for each file. The working directories these were produced in
> (`eval_runs/`, `eval_results/`) are gitignored, so the archive is the durable copy. Per-episode
> screenshots (~166 MB) are packed separately alongside the adapter — see
> [`eval_archive/README.md`](eval_archive/README.md).

The pre-training baseline every fine-tune is compared against. Captured 2026-08-26 with
`make eval-online-mind2web`; raw artefacts archived at `eval_archive/online_mind2web/baseline/`
(produced in `eval_runs/baseline_qwen35_9b_online_mind2web_20260826T165014Z/`).

**Model:** `Qwen/Qwen3.5-9B` (post-trained checkpoint, no adapter) served by vLLM 0.28.0 on a
Modal L40S via `modal_serve.py`. Browser ran locally in `DockerSandbox` on a residential IP —
a datacenter IP changes the blocking rate and makes the number incomparable.

**Task set:** `osunlp/Online-Mind2Web` rev `eacad896a84d`, 50 tasks sampled level-stratified
from 300 (17 easy / 16 medium / 17 hard). `coord_space=normalized_1000`, temperature 0,
max_tokens 2048, per-task step cap `3x reference_length` clamped to [15, 50], `--parallel 3`.
Judge: `claude-sonnet-5` (final + 2 mid screenshots).

### Headline

| metric | value |
| --- | --- |
| verified success (scored episodes) | **8.8%** (3 / 34) |
| verified success (all episodes) | **6.0%** (3 / 50) |
| site-blocked | 28.0% (14 episodes) |
| unreachable | 4.0% (2 episodes) |
| parse-failure rate | 22.3% |
| canonical action form | 26.8% of 276 parsed actions |
| `done()` emission rate | 28.0% |
| `<think>` emission rate | 78.0% |
| steps / reference on success | 1.36 |

Two denominators are reported because they answer different questions. **8.8%** excludes
episodes where the site blocked automation or never loaded — the model never saw those pages,
so counting them measures the network. **6.0%** counts every task attempted, which is the
convention the Online-Mind2Web paper and the Fara-1.5 results table use. Quote 6.0% when
comparing to published work.

### Per level (scored episodes)

| level | n | scored | success | rate | blocked |
| --- | --- | --- | --- | --- | --- |
| easy | 17 | 11 | 2 | 18.2% | 29.4% |
| medium | 16 | 10 | 0 | 0.0% | 37.5% |
| hard | 17 | 13 | 1 | 7.7% | 17.6% |

Sampling is **equal allocation** across levels, not proportional to the full set (80/141/79).
Reweighted to the true mix the scored rate is ~6.9%, so the headline mildly overstates the
300-task estimate by over-sampling easy tasks.

### How it compares

Published Online-Mind2Web results, all scored with WebJudge/o4-mini:

| model | score |
| --- | --- |
| FaraGen1.5 Solver (GPT-5.4) | 83.4% |
| Fara-1.5-9B | 63.4% |
| GUI-Owl-1.5 (8B) | 48.6% |
| MolmoWeb (8B) | 35.3% |
| Fara-7B | 34.1% |
| baseline search agent | ~22% |
| **`Qwen3.5-9B`, no agent training (this run)** | **6.0%** |

Sitting an order of magnitude below every purpose-trained CUA is the expected result for a model
with no agent training, and it is the point of the measurement: it is the floor a fine-tune has
to clear.

**These numbers are not rankable against that table.** The judge is `claude-sonnet-5` on this
repo's capture rubric, not WebJudge/o4-mini, and the OM2W authors measure WebJudge itself running
6-10 points above human-rated success. The comparison is a plausibility check, not a leaderboard
position. Closing that gap requires running a published checkpoint through this harness.

### What the baseline actually fails at

The failure is not reasoning. `<think>` was emitted on 78% of steps and the reasoning was usually
correct — on the GOV.UK task it identified the right visa page and said why. Three things break:

- **Output format.** Only 26.8% of parsed actions were in canonical form, and 755 lines were
  unparseable against 276 parsed. The dominant deviations were omitted default kwargs (164),
  bare coordinates without `<ref>` (115) and a missing `action:` prefix (58). 22.3% of steps
  produced no executable action at all, and `parse_failure_abort` was the single most common
  outcome (16 of 50).
- **Stopping.** Only 28% of episodes emitted `done()`. The model rarely recognises completion.
- **The environment.** 32% of tasks (16/50) never reached the model — blocked or unreachable.

Grounding, notably, was **not** a failure mode once the coordinate space was set correctly:
0 out-of-viewport actions were refused and the driver error rate was 0.4%.

### Reproducing

```bash
modal deploy modal_serve.py                     # serves Qwen/Qwen3.5-9B on an L40S
make eval-online-mind2web \
  EVAL_ENDPOINT=<url>/v1 EVAL_MODEL=cua-student \
  EVAL_LABEL=before EVAL_LIMIT=50 EVAL_COORD_SPACE=normalized_1000
```

After fine-tuning, re-run with `EVAL_LABEL=after`, `EVAL_COORD_SPACE=viewport_abs` (a model
trained on this corpus answers in viewport pixels, not normalised units) and otherwise identical
flags, then `make eval-compare`. Read the harness-parity section of the comparison before
trusting the delta.

### Fine-tuned result (SFT v1)

`Qwen/Qwen3.5-9B` + LoRA r64/alpha128 on 5,477 training examples (616 val) from 456 trajectories,
3 epochs = 2,055 optimizer steps at effective batch 8, A100-80GB, ~7h35m. Adapter on the
`cua-outputs` Modal volume at `sft-v1/adapter`; per-epoch checkpoints at
`checkpoint-685` / `checkpoint-1370` / `checkpoint-2055`. Evaluated on the **same 50 tasks** as the
baseline, same judge, same step-cap policy; the only intended difference is `coord_space`
(`normalized_1000` -> `viewport_abs`, since a model trained on this corpus answers in viewport pixels).

| metric | baseline | fine-tuned | delta |
| --- | --- | --- | --- |
| verified success (scored), as measured | 8.8% | 29.4% | +20.6 pp |
| **verified success (scored), audited** | **8.8%** | **23.5%** | **+14.7 pp** |
| verified success (all 50), audited | 6.0% | 16.0% | +10.0 pp |
| parse-failure rate | 22.3% | 7.4% | -15.0 pp |
| canonical action form | 26.8% | **100.0%** | +73.2 pp |
| `done()` rate | 28.0% | 18.0% | **-10.0 pp** |
| steps / reference on success | 1.36x | 2.14x | worse |
| site-blocked | 28.0% | 28.0% | unchanged |

By difficulty (scored episodes):

| level | baseline | fine-tuned |
| --- | --- | --- |
| easy | 18.2% (2/11) | 36.4% (4/11) |
| medium | **0.0%** (0/10) | 30.0% (3/10) |
| hard | 7.7% (1/13) | 23.1% (3/13) |

**Quote the audited row.** Two of the ten verified successes are confirmed false positives, checked
against their final screenshots — see `docs/decisions/judge-consistency-audit.md`. The judge's
leniency is concentrated on `meandering` episodes, which only the fine-tuned model produces, so it
inflates the after side and not the baseline.

**What the model learned, decomposed.** Reporting these separately is the point of the extra columns:

- *Format: fully learned.* Canonical action form 26.8% -> 100%, parse failures 22.3% -> 7.4%.
- *Browsing: genuinely improved.* Medium tasks went 0% -> 30%, which format alone cannot explain.
- *Stopping: regressed.* `done()` fell 28% -> 18% and steps/reference rose 1.36x -> 2.14x; 4 of 10
  successes were graded meandering. The model acts more persistently but terminates worse.

The honest claim is therefore "this corpus taught the model to emit valid actions and to browse
better, at the cost of knowing when to stop" — not a single headline number.

### Grounding: ScreenSpot (SFT v1)

Pixel-level grounding — predict a click point from a screenshot + instruction, score whether it
lands inside the target bbox. Complements the agentic eval above: it isolates *pointing* from
*task completion*. 300 samples of `rootsautomation/ScreenSpot` `test`, both models served from one
vLLM container so hardware and sampling are identical.

| metric | before (base) | after (SFT v1) | delta |
| --- | ---: | ---: | ---: |
| **overall accuracy** | **89.0%** | **85.0%** | **-4.0 pp** |
| parse rate | 94.7% | 97.0% | +2.3 pp |
| type: icon | 80.3% (98/122) | 75.4% (92/122) | -4.9 pp |
| type: text | 94.9% (169/178) | 91.6% (163/178) | -3.3 pp |
| source: macOS | 87.0% (120/138) | 86.2% (119/138) | -0.8 pp |
| source: Windows | 90.7% (147/162) | 84.0% (136/162) | -6.7 pp |

Per-model coordinate spaces, as `eval_screenspot.py` requires: base in `normalized_1000`, LoRA in
`viewport_abs`. Scoring one in the other's convention is meaningless: in a separate 100-sample check the
same base-model predictions scored 85% under `normalized_1000` and 25% under `viewport_abs`.

**The regression is real, not a resizing artifact.** `viewport_abs` resizes screenshots to
1280x720, which distorts non-16:9 images, so the after-run was split by whether the resize actually
changes the aspect ratio: distorted samples scored **88.6%** (n=167), near-16:9 samples **86.3%**
(n=124). Distorted scored higher, so distortion does not explain the drop.

Paired per-sample: 244 both hit, **23 lost**, 11 gained, 22 neither — net -12 of 300, which is the
-4.0 pp exactly. A modest shift, not a collapse.

### What SFT actually traded

Both evaluations agree on the shape:

| axis | direction | evidence |
| --- | --- | --- |
| output format compliance | **up** | ScreenSpot parse 94.7% -> 97.0%; OM2W canonical form 26.8% -> 100% |
| agentic task success | **up** | OM2W 8.8% -> 23.5% (audited) |
| raw pointing accuracy | **down** | ScreenSpot 89.0% -> 85.0% |
| verbosity | down | mean response 532 -> 196 chars |

The corpus is 5,477 web-browsing steps at a fixed 1280x720 viewport; ScreenSpot is desktop and
mobile UI pointing at native resolutions. Gaining format discipline and task completion while losing
~4 points of general-purpose grounding is the expected narrow-SFT trade. State both halves.

Reproduce with the `openai` backend (added so the harness can evaluate any OpenAI-compatible
endpoint instead of shelling out to `swift infer`):

```bash
modal run modal_serve.py::warm                  # populate the weight cache on CPU first
modal deploy modal_serve.py                     # serves base + LoRA from one container
SCREENSPOT_ENDPOINT=<url>/v1 SCREENSPOT_API_KEY=<key> \
  python eval_screenspot.py run --models-config eval_models_served.yaml \
  --dataset rootsautomation/ScreenSpot --split test \
  --bbox-format xyxy --bbox-normalized --limit 300
```

`eval_models_served.yaml` carries the vLLM *served* names (`cua-student`, `cua-student-lora`) and
`backend: openai` per spec; `eval_models.yaml` is the local `swift infer` lane and takes a repo id
plus an adapters directory instead.

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
