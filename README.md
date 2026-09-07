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
| cua-grpo-v3 | all | canonical_form_rate | 100.0 % <sub>[99–100]</sub> | 284 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (grpo-v3 = v2 one image + viewport_abs, both inherited from its merged sft-v2 parent) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_grpo-v3/. Share of parsed actions spelled exactly as format_action_for_training emits. |
| cua-grpo-v3 | all | parse_failure_rate | 0.2 % <sub>[0–2]</sub> | 284 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (grpo-v3 = v2 one image + viewport_abs, both inherited from its merged sft-v2 parent) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_grpo-v3/. Denominator is model steps, not episodes, so no interval is filled. |
| cua-grpo-v3 | all-episodes | verified_success_rate | 6.7 % <sub>[4–10]</sub> | 284 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (grpo-v3 = v2 one image + viewport_abs, both inherited from its merged sft-v2 parent) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_grpo-v3/. |
| cua-grpo-v3 | easy | verified_success_rate | 32.4 % <sub>[20–49]</sub> | 37 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (grpo-v3 = v2 one image + viewport_abs, both inherited from its merged sft-v2 parent) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_grpo-v3/. 35 of 74 tasks at this level were blocked. |
| cua-grpo-v3 | hard | verified_success_rate | 9.5 % <sub>[4–22]</sub> | 42 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (grpo-v3 = v2 one image + viewport_abs, both inherited from its merged sft-v2 parent) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_grpo-v3/. 31 of 74 tasks at this level were blocked. |
| cua-grpo-v3 | medium | verified_success_rate | 4.8 % <sub>[2–13]</sub> | 63 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (grpo-v3 = v2 one image + viewport_abs, both inherited from its merged sft-v2 parent) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_grpo-v3/. 70 of 136 tasks at this level were blocked. |
| cua-grpo-v3 | scored | verified_success_rate | 13.4 % <sub>[9–20]</sub> | 142 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (grpo-v3 = v2 one image + viewport_abs, both inherited from its merged sft-v2 parent) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_grpo-v3/. |
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
| cua-sft-v2 | all | canonical_form_rate | 100.0 % <sub>[99–100]</sub> | 300 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v2 = v2 one image + viewport_abs; base = v3 + normalized_1000) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v2/. Share of parsed actions spelled exactly as format_action_for_training emits. |
| cua-sft-v2 | all | parse_failure_rate | 0.0 % <sub>[0–7]</sub> | 50 | done |  |
| cua-sft-v2 | all | parse_failure_rate | 0.0 % <sub>[0–1]</sub> | 300 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v2 = v2 one image + viewport_abs; base = v3 + normalized_1000) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v2/. Denominator is model steps, not episodes, so no interval is filled. |
| cua-sft-v2 | all-episodes | done_emission_rate | 0.3 | 50 | done |  |
| cua-sft-v2 | all-episodes | verified_success_rate | 8.0 % <sub>[3–19]</sub> | 50 | done |  |
| cua-sft-v2 | all-episodes | verified_success_rate | 8.0 % <sub>[5–12]</sub> | 300 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v2 = v2 one image + viewport_abs; base = v3 + normalized_1000) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v2/. |
| cua-sft-v2 | easy | verified_success_rate | 27.3 % <sub>[10–57]</sub> | 11 | done |  |
| cua-sft-v2 | easy | verified_success_rate | 47.4 % <sub>[32–63]</sub> | 38 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v2 = v2 one image + viewport_abs; base = v3 + normalized_1000) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v2/. 35 of 80 tasks at this level were blocked. |
| cua-sft-v2 | hard | verified_success_rate | 0.0 % <sub>[0–23]</sub> | 13 | done |  |
| cua-sft-v2 | hard | verified_success_rate | 6.4 % <sub>[2–17]</sub> | 47 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v2 = v2 one image + viewport_abs; base = v3 + normalized_1000) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v2/. 28 of 79 tasks at this level were blocked. |
| cua-sft-v2 | medium | verified_success_rate | 11.1 % <sub>[2–44]</sub> | 9 | done |  |
| cua-sft-v2 | medium | verified_success_rate | 4.9 % <sub>[2–13]</sub> | 61 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v2 = v2 one image + viewport_abs; base = v3 + normalized_1000) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v2/. 72 of 141 tasks at this level were blocked. |
| cua-sft-v2 | parsed-actions | canonical_form_rate | 100.0 % <sub>[99–100]</sub> | 644 | done |  |
| cua-sft-v2 | scored | verified_success_rate | 12.1 % <sub>[5–27]</sub> | 33 | done | sft-v2 on the archived 50-task subset, same night and same judge as om2w-base-v2-50. Success delta is inside noise at n=33; parse-failure and done-rate deltas are not. |
| cua-sft-v2 | scored | verified_success_rate | 16.4 % <sub>[11–23]</sub> | 146 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v2 = v2 one image + viewport_abs; base = v3 + normalized_1000) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v2/. |
| cua-sft-v3 | all | canonical_form_rate | 100.0 % <sub>[99–100]</sub> | 300 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v3 = v3 two-image + normalized_1000, the format it was trained on) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v3/. Share of parsed actions spelled exactly as format_action_for_training emits. |
| cua-sft-v3 | all | parse_failure_rate | 0.0 % <sub>[0–1]</sub> | 300 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v3 = v3 two-image + normalized_1000, the format it was trained on) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v3/. Denominator is model steps, not episodes, so no interval is filled. |
| cua-sft-v3 | all-episodes | verified_success_rate | 5.3 % <sub>[3–8]</sub> | 300 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v3 = v3 two-image + normalized_1000, the format it was trained on) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v3/. |
| cua-sft-v3 | easy | verified_success_rate | 23.7 % <sub>[13–39]</sub> | 38 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v3 = v3 two-image + normalized_1000, the format it was trained on) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v3/. 36 of 80 tasks at this level were blocked. |
| cua-sft-v3 | hard | verified_success_rate | 2.2 % <sub>[0–11]</sub> | 46 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v3 = v3 two-image + normalized_1000, the format it was trained on) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v3/. 28 of 79 tasks at this level were blocked. |
| cua-sft-v3 | medium | verified_success_rate | 10.0 % <sub>[5–20]</sub> | 60 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v3 = v3 two-image + normalized_1000, the format it was trained on) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v3/. 72 of 141 tasks at this level were blocked. |
| cua-sft-v3 | scored | verified_success_rate | 11.1 % <sub>[7–17]</sub> | 144 | done | 2026-09-05: the FULL 300-task set against the base arm's same run, same harness commit and same 2048-token budget, each arm prompted in the observation format it was TRAINED for (sft-v3 = v3 two-image + normalized_1000, the format it was trained on) — the per-arm principle coord_space already followed. Served from Modal. Scored denominator excludes site_blocked/unreachable/harness_error; the all-episode rate is registered alongside. Artifacts eval_runs/om2w300_sft-v3/. |
| qwen35-9b-base | all | parse_failure_rate | 22.3 % |  | done | Denominator is harness-internal; not a per-episode proportion. |
| qwen35-9b-base | all | parse_failure_rate | 20.6 % |  | done | Denominator is harness-internal; not a per-episode proportion. |
| qwen35-9b-base | all | parse_failure_rate | 23.6 % <sub>[19–30]</sub> | 230 | done |  |
| qwen35-9b-base | all | parse_failure_rate | 37.7 % <sub>[26–52]</sub> | 50 | done |  |
| qwen35-9b-base | all | parse_failure_rate | 12.8 % <sub>[9–17]</sub> | 300 | done | 2026-09-05: the FULL 300-task set, not a 34/50-task sample, with max_tokens 2048 (at the old 512 the base was truncated before it emitted an action) and a working post-hoc judge. Scored denominator excludes site_blocked/unreachable/harness_error, which is 143 of 300 from this egress; the all-episode rate is registered alongside for exactly that reason. Supersedes the n=34 and n=164 base rows, which came from different harness commits. Artifacts eval_runs/om2w300_base/. Denominator is model steps, not episodes, so no interval is filled. |
| qwen35-9b-base | all-episodes | done_emission_rate | 0.113 | 230 | done |  |
| qwen35-9b-base | all-episodes | done_emission_rate | 0.12 | 50 | done |  |
| qwen35-9b-base | all-episodes | done_rate | 22.0 % <sub>[13–35]</sub> | 50 | done | Episodes the policy ended with done(). |
| qwen35-9b-base | all-episodes | verified_success_rate | 6.0 % <sub>[2–16]</sub> | 50 | done | Denominator includes blocked and unreachable episodes. |
| qwen35-9b-base | all-episodes | verified_success_rate | 10.0 % <sub>[4–21]</sub> | 50 | done | Denominator includes blocked and unreachable sites. |
| qwen35-9b-base | all-episodes | verified_success_rate | 4.8 % <sub>[3–8]</sub> | 230 | done |  |
| qwen35-9b-base | all-episodes | verified_success_rate | 6.0 % <sub>[2–16]</sub> | 50 | done |  |
| qwen35-9b-base | all-episodes | verified_success_rate | 1.7 % <sub>[1–4]</sub> | 300 | done | 2026-09-05: the FULL 300-task set, not a 34/50-task sample, with max_tokens 2048 (at the old 512 the base was truncated before it emitted an action) and a working post-hoc judge. Scored denominator excludes site_blocked/unreachable/harness_error, which is 143 of 300 from this egress; the all-episode rate is registered alongside for exactly that reason. Supersedes the n=34 and n=164 base rows, which came from different harness commits. Artifacts eval_runs/om2w300_base/. |
| qwen35-9b-base | easy | verified_success_rate | 18.2 % <sub>[5–48]</sub> | 11 | done |  |
| qwen35-9b-base | easy | verified_success_rate | 36.4 % <sub>[15–65]</sub> | 11 | done | 4/11. |
| qwen35-9b-base | easy | verified_success_rate | 25.0 % <sub>[15–39]</sub> | 44 | done |  |
| qwen35-9b-base | easy | verified_success_rate | 27.3 % <sub>[10–57]</sub> | 11 | done |  |
| qwen35-9b-base | easy | verified_success_rate | 12.5 % <sub>[5–26]</sub> | 40 | done | 2026-09-05: the FULL 300-task set, not a 34/50-task sample, with max_tokens 2048 (at the old 512 the base was truncated before it emitted an action) and a working post-hoc judge. Scored denominator excludes site_blocked/unreachable/harness_error, which is 143 of 300 from this egress; the all-episode rate is registered alongside for exactly that reason. Supersedes the n=34 and n=164 base rows, which came from different harness commits. Artifacts eval_runs/om2w300_base/. 33 of 80 tasks at this level were blocked. |
| qwen35-9b-base | hard | verified_success_rate | 7.7 % <sub>[1–33]</sub> | 13 | done |  |
| qwen35-9b-base | hard | verified_success_rate | 7.7 % <sub>[1–33]</sub> | 13 | done | 1/13. |
| qwen35-9b-base | hard | verified_success_rate | 0.0 % <sub>[0–7]</sub> | 49 | done |  |
| qwen35-9b-base | hard | verified_success_rate | 0.0 % <sub>[0–23]</sub> | 13 | done |  |
| qwen35-9b-base | hard | verified_success_rate | 0.0 % <sub>[0–7]</sub> | 48 | done | 2026-09-05: the FULL 300-task set, not a 34/50-task sample, with max_tokens 2048 (at the old 512 the base was truncated before it emitted an action) and a working post-hoc judge. Scored denominator excludes site_blocked/unreachable/harness_error, which is 143 of 300 from this egress; the all-episode rate is registered alongside for exactly that reason. Supersedes the n=34 and n=164 base rows, which came from different harness commits. Artifacts eval_runs/om2w300_base/. 27 of 79 tasks at this level were blocked. |
| qwen35-9b-base | medium | verified_success_rate | 0.0 % <sub>[0–28]</sub> | 10 | done |  |
| qwen35-9b-base | medium | verified_success_rate | 0.0 % <sub>[0–28]</sub> | 10 | done | 0/10. |
| qwen35-9b-base | medium | verified_success_rate | 0.0 % <sub>[0–5]</sub> | 71 | done |  |
| qwen35-9b-base | medium | verified_success_rate | 0.0 % <sub>[0–28]</sub> | 10 | done |  |
| qwen35-9b-base | medium | verified_success_rate | 0.0 % <sub>[0–5]</sub> | 69 | done | 2026-09-05: the FULL 300-task set, not a 34/50-task sample, with max_tokens 2048 (at the old 512 the base was truncated before it emitted an action) and a working post-hoc judge. Scored denominator excludes site_blocked/unreachable/harness_error, which is 143 of 300 from this egress; the all-episode rate is registered alongside for exactly that reason. Supersedes the n=34 and n=164 base rows, which came from different harness commits. Artifacts eval_runs/om2w300_base/. 66 of 141 tasks at this level were blocked. |
| qwen35-9b-base | parsed-actions | canonical_form_rate | 26.8 % <sub>[22–32]</sub> | 276 | done | 74/276 parsed actions in canonical form. |
| qwen35-9b-base | parsed-actions | canonical_form_rate | 18.8 % <sub>[15–24]</sub> | 282 | done | 53/282 parsed actions in canonical form. |
| qwen35-9b-base | parsed-actions | canonical_form_rate | 29.7 % <sub>[27–32]</sub> | 1351 | done |  |
| qwen35-9b-base | parsed-actions | canonical_form_rate | 25.5 % <sub>[19–33]</sub> | 149 | done |  |
| qwen35-9b-base | scored | verified_success_rate | 8.8 % <sub>[3–23]</sub> | 34 | done | 3/34 scored episodes; 50 sampled, 14 site-blocked, 2 unreachable. |
| qwen35-9b-base | scored | verified_success_rate | 14.7 % <sub>[6–30]</sub> | 34 | done | 5/34 scored episodes. Criteria-first judge (judge_prompt_sha), all three policies the same night; not comparable to rows without a judge_prompt_sha. |
| qwen35-9b-base | scored | verified_success_rate | 6.7 % <sub>[4–12]</sub> | 164 | done | Base re-measured on the 230-task pinned set under the navigate/handoff-aware harness. Not comparable to pre-2026-08-31 rows: the action grammar and history_window changed. |
| qwen35-9b-base | scored | verified_success_rate | 8.8 % <sub>[3–23]</sub> | 34 | done | Same base run, restricted to the archived 50-task subset so sft-v2 has a like-for-like comparator. Base scored 0.1471 on these same 50 on 2026-08-27, so ~6pp is run-to-run noise. |
| qwen35-9b-base | scored | verified_success_rate | 3.2 % <sub>[1–7]</sub> | 157 | done | 2026-09-05: the FULL 300-task set, not a 34/50-task sample, with max_tokens 2048 (at the old 512 the base was truncated before it emitted an action) and a working post-hoc judge. Scored denominator excludes site_blocked/unreachable/harness_error, which is 143 of 300 from this egress; the all-episode rate is registered alongside for exactly that reason. Supersedes the n=34 and n=164 base rows, which came from different harness commits. Artifacts eval_runs/om2w300_base/. |
| qwen35-9b-base | successes | steps_over_reference_on_success | 1.92 | 5 | done | Mean steps / reference_length over verified successes. |
| cua-sft-v1-nologin | scored | verified_success_rate | _not measured_ | 34 | planned | Ablation that turns the post-authentication novelty claim into a result. |

#### `qa-bench-refapp` — catalog-v1@0f3d7f88e85a, catalog-v2@cb427058ac9e

| model_id | subset | metric | value | n | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| claude-sonnet-5 | 6-defect-subset | ddr | 100.0 % <sub>[61–100]</sub> | 6 | done | 6/6 via the full vetty agent in native browser-tool mode. |
| claude-sonnet-5 | 6-defect-subset | done_rate | 100.0 % <sub>[61–100]</sub> | 6 | done |  |
| claude-sonnet-5 | 6-defect-subset | execution_cost_per_case_usd | $1.17 | 6 | done | The phase the local model replaces. Provenance corrected: this figure is the reported execution cost per test case in the frontier-e1 report, 1.1692. |
| claude-sonnet-5 | class-action | ddr | 80.0 % <sub>[38–96]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| claude-sonnet-5 | class-data | ddr | 80.0 % <sub>[38–96]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| claude-sonnet-5 | class-labeling | ddr | 40.0 % <sub>[12–77]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| claude-sonnet-5 | class-navigation | ddr | 60.0 % <sub>[23–88]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| claude-sonnet-5 | class-silent_failure | ddr | 100.0 % <sub>[57–100]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| claude-sonnet-5 | class-state_dependent | ddr | 66.7 % <sub>[21–94]</sub> | 3 | done | Per-class detection over 3 defects; at this size a single detection moves the value by 33 points, so no class difference between arms is resolvable. |
| claude-sonnet-5 | class-validation | ddr | 40.0 % <sub>[12–77]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| claude-sonnet-5 | completed-runs-pinned-plan | phase_call_share | 98.2 % <sub>[98–98]</sub> | 8343 | done | Execution's share of model calls over 50 completed runs that reused a pinned plan, so exploration and planning did not run: 8195 of 8343 calls. Corroborates phase4-calls-executing without depending on one run. |
| claude-sonnet-5 | completed-runs-pinned-plan | phase_cost_share | 95.0 % |  | done | Execution's share of cost over the same 50 completed runs: 779.04 of 820.44 US dollars. |
| claude-sonnet-5 | cost-model | e2e_cost_per_case_usd | $1.27 | 87 | done | FRONTIER CONFIGURATION, every phase on claude-sonnet-5: execution 1.17 plus reporting 0.1025. Measured over results/frontier-e1: 7 runs, all completed, judge pass performed, reports delivered. Denominator is the 87 case-runs those 7 runs executed. |
| claude-sonnet-5 | cost-model | e2e_cost_per_case_usd | $0.59 | 105 | done | HYBRID CONFIGURATION, local execution plus a frontier verdict: accelerator 0.01509 plus verdict 0.575. Still a composition of two runs rather than one end-to-end run, as hybrid-cost-and-the-product-gap.md requires it to be stated. econ-grpo-v3 reports 0.594 per case for this sum within a single arm, which corroborates it. |
| claude-sonnet-5 | cost-model | e2e_saving_vs_frontier | 2.15 |  | done | Frontier 1.27 divided by hybrid 0.59. Execution alone falls by a factor of 78; the verdict now governs the total. |
| claude-sonnet-5 | cost-model | execution_cost_per_case_usd_e2e | 1.17 |  | done | FRONTIER CONFIGURATION = every phase on claude-sonnet-5, the reference the hybrid is compared against. Excludes the DDR match-judge, which is benchmark apparatus and has never entered any cost figure. Provenance corrected: this figure is the reported execution cost per test case in the frontier-e1 report, 1.1692. |
| claude-sonnet-5 | cost-model | explore_plan_cost_per_app_usd | $3.79 |  | done | Exploration 2.374 plus planning 1.420, paid once for an application. Measured over results/frontier-e1: 7 runs, all completed, judge pass performed, reports delivered. Denominator is the 87 case-runs those 7 runs executed. |
| claude-sonnet-5 | cost-model | reporting_cost_per_case_usd | $0.10 | 87 | done | Reporting phase, 8.92 US dollars over 87 case-runs. Measured over results/frontier-e1: 7 runs, all completed, judge pass performed, reports delivered. Denominator is the 87 case-runs those 7 runs executed. |
| claude-sonnet-5 | cost-model | verdict_cost_per_case_usd | $0.57 | 105 | done | Measured from the verdict cache of results/econ-grpo-v3, the only arm whose cache stores the usage of the call that produced each verdict: 91 verdicts, 50.30 US dollars, mean 0.5528. The report's own per-case figure over 105 cases is 0.575. An upper bound: the cached prompt still carries configuration overhead a deployed system would not send. |
| claude-sonnet-5 | economy-7run-105case | ddr | 66.7 % <sub>[50–80]</sub> | 33 | done | DDR strict over 33 catalogue defects. Arm 'sonnet'. VERIFIED 2026-09-05: re-judged under harness 428c062 and reproduced exactly (DDR 0.6667, precision 0.7708, FPR 0.0), so this arm is comparable to the 428c062 local arms despite being recorded at 409810f. The v3 executor changes touch only the CUA path, which the claude-sdk backbone does not import. Artifacts vetty results/econ-sonnet-428c062. DENOMINATOR CORRECTED 2026-09-06: n was the 105 case-runs, but DDR is the share of the 33 catalogue defects detected, so the interval was computed from the wrong denominator and was too narrow. Value unchanged; n and the interval now reflect 33. |
| claude-sonnet-5 | economy-7run-105case | error_verdict_rate | 0.0286 | 105 | done | Share of 105 case-runs the verdict tier could not judge - the execution-competence number. Arm 'sonnet'. VERIFIED 2026-09-05: re-judged under harness 428c062 and reproduced exactly (DDR 0.6667, precision 0.7708, FPR 0.0), so this arm is comparable to the 428c062 local arms despite being recorded at 409810f. The v3 executor changes touch only the CUA path, which the claude-sdk backbone does not import. Artifacts vetty results/econ-sonnet-428c062. |
| claude-sonnet-5 | economy-7run-105case | executor_cost_per_case_usd | $1.40 | 105 | done | Frontier executor cost per case (executing $139.348 + reporting $7.266 over 105 cases). Claude Agent SDK's own cost estimate on a Max subscription via the logged-in CLI - notional, not billed, and known to drift. Reported as an estimate, never as a measurement. VERIFIED 2026-09-05: re-judged under harness 428c062 and reproduced exactly (DDR 0.6667, precision 0.7708, FPR 0.0), so this arm is comparable to the 428c062 local arms despite being recorded at 409810f. The v3 executor changes touch only the CUA path, which the claude-sdk backbone does not import. Artifacts vetty results/econ-sonnet-428c062. |
| claude-sonnet-5 | economy-7run-105case | executor_input_tokens_per_case | 22 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. VERIFIED 2026-09-05: re-judged under harness 428c062 and reproduced exactly (DDR 0.6667, precision 0.7708, FPR 0.0), so this arm is comparable to the 428c062 local arms despite being recorded at 409810f. The v3 executor changes touch only the CUA path, which the claude-sdk backbone does not import. Artifacts vetty results/econ-sonnet-428c062. |
| claude-sonnet-5 | economy-7run-105case | executor_output_tokens_per_case | 3875 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. VERIFIED 2026-09-05: re-judged under harness 428c062 and reproduced exactly (DDR 0.6667, precision 0.7708, FPR 0.0), so this arm is comparable to the 428c062 local arms despite being recorded at 409810f. The v3 executor changes touch only the CUA path, which the claude-sdk backbone does not import. Artifacts vetty results/econ-sonnet-428c062. |
| claude-sonnet-5 | economy-7run-105case | precision_strict | 0.7708 | 48 | done | Share of filed bug reports that matched a real catalogue defect. Arm 'sonnet'. VERIFIED 2026-09-05: re-judged under harness 428c062 and reproduced exactly (DDR 0.6667, precision 0.7708, FPR 0.0), so this arm is comparable to the 428c062 local arms despite being recorded at 409810f. The v3 executor changes touch only the CUA path, which the claude-sdk backbone does not import. Artifacts vetty results/econ-sonnet-428c062. Denominator corrected to the number of filed bug reports, which is what precision divides by; the earlier value was the case-run count. |
| claude-sonnet-5 | paper-34run-33defect | bugs_per_clean_run | 3 | 1 | done | Three reports filed against the build carrying no seeded defect, over a single clean run. |
| claude-sonnet-5 | paper-34run-33defect | ddr | 81.8 % <sub>[66–91]</sub> | 33 | done | 27 of 33, strict equal to lenient because no report was judged a consequence. Full pass of 34 runs over the pinned 15-case plan, one clean plus 33 single-defect builds. |
| claude-sonnet-5 | paper-34run-33defect | precision_strict | 0.6164 | 73 | done | 45 of 73 filed reports match a seeded defect. |
| claude-sonnet-5 | phase-executing | phase_call_share | 92.5 % <sub>[90–95]</sub> | 402 | done | 92.5 % of 402 calls, 372 calls over 45 test cases. Measured on results/frontier-e1, the one recorded run that performed exploration, planning, execution and reporting to completion on the reference application. The run's own bug reports are not used: 7 were filed against a clean build. Only the distribution of work across phases is taken from it. |
| claude-sonnet-5 | phase-executing | phase_cost_share | 91.5 % |  | done | 52.31 of 57.14 US dollars. Measured on results/frontier-e1, the one recorded run that performed exploration, planning, execution and reporting to completion on the reference application. The run's own bug reports are not used: 7 were filed against a clean build. Only the distribution of work across phases is taken from it. |
| claude-sonnet-5 | phase-exploring | phase_call_share | 5.0 % <sub>[3–8]</sub> | 402 | done | 5.0 % of 402 calls, 20 calls. Measured on results/frontier-e1, the one recorded run that performed exploration, planning, execution and reporting to completion on the reference application. The run's own bug reports are not used: 7 were filed against a clean build. Only the distribution of work across phases is taken from it. |
| claude-sonnet-5 | phase-exploring | phase_cost_share | 4.2 % |  | done | 2.37 of 57.14 US dollars. Measured on results/frontier-e1, the one recorded run that performed exploration, planning, execution and reporting to completion on the reference application. The run's own bug reports are not used: 7 were filed against a clean build. Only the distribution of work across phases is taken from it. |
| claude-sonnet-5 | phase-planning | phase_call_share | 1.7 % <sub>[1–4]</sub> | 402 | done | 1.7 % of 402 calls, 7 calls. Measured on results/frontier-e1, the one recorded run that performed exploration, planning, execution and reporting to completion on the reference application. The run's own bug reports are not used: 7 were filed against a clean build. Only the distribution of work across phases is taken from it. |
| claude-sonnet-5 | phase-planning | phase_cost_share | 2.5 % |  | done | 1.42 of 57.14 US dollars. Measured on results/frontier-e1, the one recorded run that performed exploration, planning, execution and reporting to completion on the reference application. The run's own bug reports are not used: 7 were filed against a clean build. Only the distribution of work across phases is taken from it. |
| claude-sonnet-5 | phase-reporting | phase_call_share | 0.8 % <sub>[0–2]</sub> | 402 | done | 0.7 % of 402 calls, 3 calls, which matches the mean of 3 reporting calls a run over 50 completed runs. Measured on results/frontier-e1, the one recorded run that performed exploration, planning, execution and reporting to completion on the reference application. The run's own bug reports are not used: 7 were filed against a clean build. Only the distribution of work across phases is taken from it. |
| claude-sonnet-5 | phase-reporting | phase_cost_share | 1.8 % |  | done | 1.03 of 57.14 US dollars. Measured on results/frontier-e1, the one recorded run that performed exploration, planning, execution and reporting to completion on the reference application. The run's own bug reports are not used: 7 were filed against a clean build. Only the distribution of work across phases is taken from it. |
| claude-sonnet-5 | skin-v2 | pass_rate_retained | 84.6 % <sub>[58–96]</sub> | 13 | done | Agent arm, 11 of 13. skin_v2 changes ids, classes and CSS plus labels and field order. The two broken cases are TC-006 and TC-013. |
| claude-sonnet-5 | skin-v2-ids-only | pass_rate_retained | 100.0 % <sub>[77–100]</sub> | 13 | done | Agent arm, 13 test cases passing on clean, majority of 3 repeats per build. skin_v2_ids_only renames ids, classes and CSS while keeping labels and field order. Supersedes the planned e3-brittleness-frontier row. |
| cua-grpo-v3 | class-action | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-grpo-v3 | class-data | ddr | 40.0 % <sub>[12–77]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-grpo-v3 | class-labeling | ddr | 40.0 % <sub>[12–77]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-grpo-v3 | class-navigation | ddr | 40.0 % <sub>[12–77]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-grpo-v3 | class-silent_failure | ddr | 20.0 % <sub>[4–62]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-grpo-v3 | class-state_dependent | ddr | 33.3 % <sub>[6–79]</sub> | 3 | done | Per-class detection over 3 defects; at this size a single detection moves the value by 33 points, so no class difference between arms is resolvable. |
| cua-grpo-v3 | class-validation | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-grpo-v3 | economy-7run-105case | ddr | 24.2 % <sub>[13–41]</sub> | 33 | done | 2026-09-05, harness 428c062. Config identical to matched-sft-v2 (cua_observation=v2, viewport_abs, history window 20, 25-step cap): grpo-v3 is a LoRA over merged sft-v2 and inherits both, so only an identical config makes the two comparable. Artifacts vetty results/econ-grpo-v3. |
| cua-grpo-v3 | economy-7run-105case | error_verdict_rate | 0.5714 | 105 | done | 2026-09-05, harness 428c062. Config identical to matched-sft-v2 (cua_observation=v2, viewport_abs, history window 20, 25-step cap): grpo-v3 is a LoRA over merged sft-v2 and inherits both, so only an identical config makes the two comparable. Artifacts vetty results/econ-grpo-v3. |
| cua-grpo-v3 | economy-7run-105case | precision_strict | 1 | 11 | done | 2026-09-05, harness 428c062. Config identical to matched-sft-v2 (cua_observation=v2, viewport_abs, history window 20, 25-step cap): grpo-v3 is a LoRA over merged sft-v2 and inherits both, so only an identical config makes the two comparable. Artifacts vetty results/econ-grpo-v3. |
| cua-sft-v1 | 6-defect-subset | completion_tokens_per_case | 385 | 6 | done | ~6.4 model calls per case. |
| cua-sft-v1 | 6-defect-subset | ddr | 83.3 % <sub>[44–97]</sub> | 6 | done | 5/6. Misses delete_button_no_action (confabulates that the delete succeeded). |
| cua-sft-v1 | 6-defect-subset | done_rate | 100.0 % <sub>[61–100]</sub> | 6 | done | 2-12 steps per case. |
| cua-sft-v1 | economy-7run-105case | ddr | 27.3 % <sub>[15–44]</sub> | 33 | done | DDR strict over 33 catalogue defects, economy sweep. DENOMINATOR CORRECTED 2026-09-06: n was the 105 case-runs, but DDR is the share of the 33 catalogue defects detected, so the interval was computed from the wrong denominator and was too narrow. Value unchanged; n and the interval now reflect 33. |
| cua-sft-v1 | economy-7run-105case | error_verdict_rate | 0.6 | 105 | done | Share of the 105 case-runs the verdict tier could not judge - the execution-competence number. |
| cua-sft-v1 | economy-7run-105case | executor_input_tokens_per_case | 11920 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v1 | economy-7run-105case | executor_output_tokens_per_case | 517 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v1 | economy-7run-105case | executor_turns_per_case | 9 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v1 | economy-7run-105case | gpu_cost_per_case_usd | $0.01 | 105 | done | Derived, not self-reported: phase_wall_seconds.executing x Modal L40S $1.95/GPU-hour (benchmark/cost.py, wired into the report 2026-09-04). This is the local arm's REAL cost; its tokens are free because the model is self-hosted. |
| cua-sft-v1 | economy-7run-105case | verdict_cost_per_run_usd | $10.49 | 105 | done | Verdict-tier cost, NOT model cost: executor runs on our own GPU at $0. sft-v3 ran on vetty 428c062 and the others on 409810f, so the per-run cost is confounded by harness changes and is not comparable across arms. |
| cua-sft-v2 | class-action | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-sft-v2 | class-data | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-sft-v2 | class-labeling | ddr | 20.0 % <sub>[4–62]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-sft-v2 | class-navigation | ddr | 20.0 % <sub>[4–62]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-sft-v2 | class-silent_failure | ddr | 20.0 % <sub>[4–62]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-sft-v2 | class-state_dependent | ddr | 0.0 % <sub>[0–56]</sub> | 3 | done | Per-class detection over 3 defects; at this size a single detection moves the value by 33 points, so no class difference between arms is resolvable. |
| cua-sft-v2 | class-validation | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-sft-v2 | cost-model | execution_cost_per_case_usd_e2e | 0.015 |  | done | HYBRID CONFIGURATION = the proposed architecture: local model executes, frontier model draws the conclusions. Composed from a measured local execution cost (GPU seconds x L40S $1.95/h) and a measured frontier verdict cost - NOT one instrumented end-to-end run, because the shipped product routes its conclusion steps to the local model too. Must be re-measured end to end once LocalModelAgentAdapter takes a frontier model_override. See docs/decisions/hybrid-cost-and-the-product-gap.md |
| cua-sft-v2 | economy-7run-105case | ddr | 21.2 % <sub>[11–38]</sub> | 33 | done | DDR strict over 33 catalogue defects, economy sweep. DENOMINATOR CORRECTED 2026-09-06: n was the 105 case-runs, but DDR is the share of the 33 catalogue defects detected, so the interval was computed from the wrong denominator and was too narrow. Value unchanged; n and the interval now reflect 33. |
| cua-sft-v2 | economy-7run-105case | ddr | 9.1 % <sub>[3–24]</sub> | 33 | done | DDR strict over 33 catalogue defects. Arm 'matched-sft-v2'. DENOMINATOR CORRECTED 2026-09-06: n was the 105 case-runs, but DDR is the share of the 33 catalogue defects detected, so the interval was computed from the wrong denominator and was too narrow. Value unchanged; n and the interval now reflect 33. |
| cua-sft-v2 | economy-7run-105case | error_verdict_rate | 0.6286 | 105 | done | Share of the 105 case-runs the verdict tier could not judge - the execution-competence number. |
| cua-sft-v2 | economy-7run-105case | error_verdict_rate | 0.619 | 105 | done | Share of 105 case-runs the verdict tier could not judge - the execution-competence number. Arm 'matched-sft-v2'. |
| cua-sft-v2 | economy-7run-105case | executor_input_tokens_per_case | 12315 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v2 | economy-7run-105case | executor_input_tokens_per_case | 16541 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v2 | economy-7run-105case | executor_output_tokens_per_case | 538 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v2 | economy-7run-105case | executor_output_tokens_per_case | 708 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v2 | economy-7run-105case | executor_turns_per_case | 9.2 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v2 | economy-7run-105case | gpu_cost_per_case_usd | $0.02 | 105 | done | Derived, not self-reported: phase_wall_seconds.executing x Modal L40S $1.95/GPU-hour (benchmark/cost.py, wired into the report 2026-09-04). This is the local arm's REAL cost; its tokens are free because the model is self-hosted. |
| cua-sft-v2 | economy-7run-105case | gpu_cost_per_case_usd | $0.01 | 105 | done | Derived, not self-reported: phase_wall_seconds.executing x Modal L40S $1.95/GPU-hour (benchmark/cost.py, wired into the report 2026-09-04). This is the local arm's REAL cost; its tokens are free because the model is self-hosted. |
| cua-sft-v2 | economy-7run-105case | precision_strict | 1 | 3 | done | Share of filed bug reports that matched a real catalogue defect. Arm 'matched-sft-v2'. Denominator corrected to the number of filed bug reports, which is what precision divides by; the earlier value was the case-run count. |
| cua-sft-v2 | economy-7run-105case | verdict_cost_per_run_usd | $10.68 | 105 | done | Verdict-tier cost, NOT model cost: executor runs on our own GPU at $0. sft-v3 ran on vetty 428c062 and the others on 409810f, so the per-run cost is confounded by harness changes and is not comparable across arms. |
| cua-sft-v3 | class-action | ddr | 20.0 % <sub>[4–62]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-sft-v3 | class-data | ddr | 40.0 % <sub>[12–77]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-sft-v3 | class-labeling | ddr | 40.0 % <sub>[12–77]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-sft-v3 | class-navigation | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-sft-v3 | class-silent_failure | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-sft-v3 | class-state_dependent | ddr | 33.3 % <sub>[6–79]</sub> | 3 | done | Per-class detection over 3 defects; at this size a single detection moves the value by 33 points, so no class difference between arms is resolvable. |
| cua-sft-v3 | class-validation | ddr | 20.0 % <sub>[4–62]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| cua-sft-v3 | economy-7run-105case | ddr | 21.2 % <sub>[11–38]</sub> | 33 | done | DDR strict over 33 catalogue defects, economy sweep. DENOMINATOR CORRECTED 2026-09-06: n was the 105 case-runs, but DDR is the share of the 33 catalogue defects detected, so the interval was computed from the wrong denominator and was too narrow. Value unchanged; n and the interval now reflect 33. |
| cua-sft-v3 | economy-7run-105case | error_verdict_rate | 0.5429 | 105 | done | Share of the 105 case-runs the verdict tier could not judge - the execution-competence number. |
| cua-sft-v3 | economy-7run-105case | executor_input_tokens_per_case | 18797 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v3 | economy-7run-105case | executor_output_tokens_per_case | 746 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v3 | economy-7run-105case | executor_turns_per_case | 11.3 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| cua-sft-v3 | economy-7run-105case | gpu_cost_per_case_usd | $0.02 | 105 | done | Derived, not self-reported: phase_wall_seconds.executing x Modal L40S $1.95/GPU-hour (benchmark/cost.py, wired into the report 2026-09-04). This is the local arm's REAL cost; its tokens are free because the model is self-hosted. |
| cua-sft-v3 | economy-7run-105case | precision_strict | 1 | 7 | done | 7 of 7 filed reports match a seeded defect; no report filed against the clean build. |
| cua-sft-v3 | economy-7run-105case | verdict_cost_per_run_usd | $0.73 | 105 | done | Verdict-tier cost, NOT model cost: executor runs on our own GPU at $0. sft-v3 ran on vetty 428c062 and the others on 409810f, so the per-run cost is confounded by harness changes and is not comparable across arms. |
| qa-bench-refapp | all | defect_catalog_size | 33 |  | done | Catalogue version 2. Severity split: 17 major, 10 minor, 6 critical. Supersedes bench-defects at 12. |
| qa-bench-refapp | all | defect_classes | 7 |  | done | validation 5, action 5, navigation 5, labeling 5, data 5, silent_failure 5, state_dependent 3. Supersedes bench-classes at 6. |
| qa-bench-refapp | full-48-case-suite | defect_catalog_size | 12 |  | done |  |
| qa-bench-refapp | full-48-case-suite | defect_classes | 6 |  | done | validation, action, navigation, labeling, data, silent_failure. |
| qa-bench-refapp | full-48-case-suite | planned_test_cases | 48 |  | done | Test cases in the cached plan for the full suite. |
| qa-bench-refapp | full-48-case-suite | scripted_suite_size | 13 |  | done | Locator-based Playwright tests forming the incumbent-method arm. Provenance corrected: the clean run of results/thesis/scripted/refapp records 13 tests, 0 failures. |
| qwen35-9b-base | 6-defect-subset | completion_tokens_per_case | 4600 | 6 | done | Rambles without stopping; ~15 model calls per case. |
| qwen35-9b-base | 6-defect-subset | ddr | 50.0 % <sub>[19–81]</sub> | 6 | done | 3/6. The 3 caught are all statically visible in page text; the model never completes a task. |
| qwen35-9b-base | 6-defect-subset | done_rate | 0.0 % <sub>[0–39]</sub> | 6 | done | All 6 hit the 15-step cap; 1 crashed on a malformed keypress. |
| qwen35-9b-base | class-action | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| qwen35-9b-base | class-data | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| qwen35-9b-base | class-labeling | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| qwen35-9b-base | class-navigation | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| qwen35-9b-base | class-silent_failure | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| qwen35-9b-base | class-state_dependent | ddr | 0.0 % <sub>[0–56]</sub> | 3 | done | Per-class detection over 3 defects; at this size a single detection moves the value by 33 points, so no class difference between arms is resolvable. |
| qwen35-9b-base | class-validation | ddr | 0.0 % <sub>[0–43]</sub> | 5 | done | Per-class detection over 5 defects; at this size a single detection moves the value by 20 points, so no class difference between arms is resolvable. |
| qwen35-9b-base | economy-7run-105case | ddr | 0.0 % <sub>[0–10]</sub> | 33 | done | DDR strict over 33 catalogue defects, economy sweep. DENOMINATOR CORRECTED 2026-09-06: n was the 105 case-runs, but DDR is the share of the 33 catalogue defects detected, so the interval was computed from the wrong denominator and was too narrow. Value unchanged; n and the interval now reflect 33. |
| qwen35-9b-base | economy-7run-105case | ddr | 0.0 % <sub>[0–10]</sub> | 33 | done | DDR strict over 33 catalogue defects. Arm 'matched-base'. DENOMINATOR CORRECTED 2026-09-06: n was the 105 case-runs, but DDR is the share of the 33 catalogue defects detected, so the interval was computed from the wrong denominator and was too narrow. Value unchanged; n and the interval now reflect 33. |
| qwen35-9b-base | economy-7run-105case | error_verdict_rate | 0.7333 | 105 | done | Share of the 105 case-runs the verdict tier could not judge - the execution-competence number. |
| qwen35-9b-base | economy-7run-105case | error_verdict_rate | 0.6667 | 105 | done | Share of 105 case-runs the verdict tier could not judge - the execution-competence number. Arm 'matched-base'. |
| qwen35-9b-base | economy-7run-105case | executor_input_tokens_per_case | 19386 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| qwen35-9b-base | economy-7run-105case | executor_input_tokens_per_case | 37135 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| qwen35-9b-base | economy-7run-105case | executor_output_tokens_per_case | 3617 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| qwen35-9b-base | economy-7run-105case | executor_output_tokens_per_case | 9738 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| qwen35-9b-base | economy-7run-105case | executor_turns_per_case | 15 | 105 | done | Executor tokens per case, economy sweep (7 runs x 15 cases = 105). Local model, so tokens are compute not spend. |
| qwen35-9b-base | economy-7run-105case | gpu_cost_per_case_usd | $0.13 | 105 | done | Derived, not self-reported: phase_wall_seconds.executing x Modal L40S $1.95/GPU-hour (benchmark/cost.py, wired into the report 2026-09-04). This is the local arm's REAL cost; its tokens are free because the model is self-hosted. |
| qwen35-9b-base | economy-7run-105case | gpu_cost_per_case_usd | $0.03 | 105 | done | Derived, not self-reported: phase_wall_seconds.executing x Modal L40S $1.95/GPU-hour (benchmark/cost.py, wired into the report 2026-09-04). This is the local arm's REAL cost; its tokens are free because the model is self-hosted. |
| qwen35-9b-base | economy-7run-105case | precision_strict | 0 | 5 | done | Share of filed bug reports that matched a real catalogue defect. Arm 'matched-base'. Denominator corrected to the number of filed bug reports, which is what precision divides by; the earlier value was the case-run count. |
| qwen35-9b-base | economy-7run-105case | verdict_cost_per_run_usd | $1.32 | 105 | done | Verdict-tier cost, NOT model cost: executor runs on our own GPU at $0. sft-v3 ran on vetty 428c062 and the others on 409810f, so the per-run cost is confounded by harness changes and is not comparable across arms. |
| scripted-playwright | paper-34run-33defect | ddr | 51.5 % <sub>[35–68]</sub> | 33 | done | 17 of 33. A defect counts as detected when a test that passes on clean fails on that defect's build, so the value is executor-independent and identical in every arm of the paper pass. |
| scripted-playwright | skin-v2 | pass_rate_retained | 0.0 % <sub>[0–23]</sub> | 13 | done | Locator-bound suite, 0 of 13 on the full reskin as on the ids-only one. |
| scripted-playwright | skin-v2-ids-only | pass_rate_retained | 0.0 % <sub>[0–23]</sub> | 13 | done | Locator-bound suite, 0 of 13. Every failure is the same cause, a timeout waiting for a renamed selector. NOTE the 13 here counts Playwright tests, a different set from the agent arm's 13 test cases; the equal size is a coincidence. |
| claude-sonnet-5 | 6-defect-subset | ddr | _not measured_ | 6 | pending | All 6 defect runs plus clean completed; the judge pass has not been run so no DDR exists yet. |
| cua-sft-v1 | skin-v2 | pass_rate_retained | _not measured_ |  | planned | Needs an ids-only skin to separate DOM change from label change. |
| cua-sft-v1-nologin | 6-defect-subset | ddr | _not measured_ | 6 | planned | Same ablation measured in the QA harness. |
| claude-opus-5 | 6-defect-subset | verdict_cost_per_case_usd | $0.87 | 6 | superseded | Now dominates hybrid cost; the remaining lever is a cheaper verdict tier. No run on disk reproduces this figure. Its artifacts_path names results/smoke, which does not exist; the surviving runs of that date are the ones already retired as the crashed run. The verdict caches of matched-base, matched-sft-v2 and econ-sft-v3 store no cost fields at all, so every verdict they replayed was counted as free. Only econ-grpo-v3 recorded verdict usage. Superseded by qa-e2e-hybrid-verdict-v2. |
| claude-sonnet-5 | 6-defect-subset | cost_per_case_usd | $1.30 | 6 | superseded | Execution $1.17 + reporting $0.13. Plus a one-time explore+plan of ~$3.68/app. Earlier duplicate of qa-e2e-frontier-cost carrying the same unreproducible reporting term. |
| claude-sonnet-5 | 6-defect-subset | explore_plan_cost_per_app_usd | $3.68 |  | superseded | One-time per app, shared across all cases and mutants. No run on disk reproduces this figure. Its artifacts_path names results/smoke, which does not exist; the surviving runs of that date are the ones already retired as the crashed run. Superseded by qa-e2e-explore-plan-v2: exploration 2.374 plus planning 1.420 is 3.79, not 3.68. |
| claude-sonnet-5 | 6-defect-subset | reporting_cost_per_case_usd | $0.13 | 6 | superseded | No run on disk reproduces this figure. Its artifacts_path names results/smoke, which does not exist; the surviving runs of that date are the ones already retired as the crashed run. Superseded by qa-e2e-frontier-report-v2, computed from the run set that completed. |
| claude-sonnet-5 | cost-model | e2e_cost_per_case_usd | $1.30 |  | superseded | FRONTIER CONFIGURATION = every phase on claude-sonnet-5, the reference the hybrid is compared against. Excludes the DDR match-judge, which is benchmark apparatus and has never entered any cost figure. Sum of an execution term that holds and a reporting term that does not. Superseded by qa-e2e-frontier-cost-v2. |
| claude-sonnet-5 | cost-model | report_cost_per_case_usd | $0.13 |  | superseded | FRONTIER CONFIGURATION = every phase on claude-sonnet-5, the reference the hybrid is compared against. Excludes the DDR match-judge, which is benchmark apparatus and has never entered any cost figure. No run on disk reproduces this figure. Its artifacts_path names results/smoke, which does not exist; the surviving runs of that date are the ones already retired as the crashed run. Superseded by qa-e2e-frontier-report-v2. |
| claude-sonnet-5 | full-48-case-suite | bugs_per_clean_run | 9 | 1 | superseded | 4 of 9 describe interactions never performed (0 POST /settings in the access log); the rest are fixture behaviour, not agent noise. |
| claude-sonnet-5 | full-48-case-suite | cost_per_run_usd | $61.20 | 1 | superseded | Reported $61.20 vs $24.87 by the pricing table; the gap is unreconciled. |
| claude-sonnet-5 | full-48-case-suite | flakiness | 0.0 % | 1 | superseded | Not a measurement: flakiness needs >=2 runs and this had 1, so 0 was the only possible output. |
| claude-sonnet-5 | full-48-case-suite | post_login_share | 85.4 % <sub>[73–93]</sub> | 48 | superseded | Regex heuristic over test-case text rather than an observed auth transition. |
| claude-sonnet-5 | full-48-case-suite | ui_coverage_elements | 92.1 % | 1 | superseded | Same crashed run. |
| claude-sonnet-5 | full-48-case-suite | ui_coverage_views | 88.9 % | 1 | superseded | Single run that crashed in the reporter phase; judge pass never ran. |
| claude-sonnet-5 | phase-executing | phase_call_share | 94.3 % <sub>[92–96]</sub> | 511 | superseded | 482 of 511 calls. The figure quoted on the architecture slide. Taken from the single full-pipeline run of 2026-08-27 that crashed in the reporting phase, the same run whose coverage, flakiness, false-positive and cost rows were already retired. Superseded by the phase4-* rows, measured on a run that completed all four phases and produced a report. |
| claude-sonnet-5 | phase-executing | phase_cost_share | 92.5 % |  | superseded | Single full-pipeline run, which crashed during reporting, so the reporting share is understated and this is a slight overstatement. Corroborated independently by $1.17 of $1.30 per case in vetty#21. Taken from the single full-pipeline run of 2026-08-27 that crashed in the reporting phase, the same run whose coverage, flakiness, false-positive and cost rows were already retired. Superseded by the phase4-* rows, measured on a run that completed all four phases and produced a report. |
| claude-sonnet-5 | phase-exploring | phase_call_share | 3.5 % <sub>[2–5]</sub> | 511 | superseded | Taken from the single full-pipeline run of 2026-08-27 that crashed in the reporting phase, the same run whose coverage, flakiness, false-positive and cost rows were already retired. Superseded by the phase4-* rows, measured on a run that completed all four phases and produced a report. |
| claude-sonnet-5 | phase-exploring | phase_cost_share | 3.4 % |  | superseded | Taken from the single full-pipeline run of 2026-08-27 that crashed in the reporting phase, the same run whose coverage, flakiness, false-positive and cost rows were already retired. Superseded by the phase4-* rows, measured on a run that completed all four phases and produced a report. |
| claude-sonnet-5 | phase-planning | phase_call_share | 1.6 % <sub>[1–3]</sub> | 511 | superseded | Taken from the single full-pipeline run of 2026-08-27 that crashed in the reporting phase, the same run whose coverage, flakiness, false-positive and cost rows were already retired. Superseded by the phase4-* rows, measured on a run that completed all four phases and produced a report. |
| claude-sonnet-5 | phase-planning | phase_cost_share | 2.5 % |  | superseded | Taken from the single full-pipeline run of 2026-08-27 that crashed in the reporting phase, the same run whose coverage, flakiness, false-positive and cost rows were already retired. Superseded by the phase4-* rows, measured on a run that completed all four phases and produced a report. |
| claude-sonnet-5 | phase-reporting | phase_call_share | 0.6 % <sub>[0–2]</sub> | 511 | superseded | Taken from the single full-pipeline run of 2026-08-27 that crashed in the reporting phase, the same run whose coverage, flakiness, false-positive and cost rows were already retired. Superseded by the phase4-* rows, measured on a run that completed all four phases and produced a report. |
| claude-sonnet-5 | phase-reporting | phase_cost_share | 1.6 % |  | superseded | Understated: the run crashed in this phase after 3 calls. Taken from the single full-pipeline run of 2026-08-27 that crashed in the reporting phase, the same run whose coverage, flakiness, false-positive and cost rows were already retired. Superseded by the phase4-* rows, measured on a run that completed all four phases and produced a report. |
| claude-sonnet-5 | skin-v2 | pass_rate_retained | _not measured_ |  | superseded | Superseded by the e3-retained-* rows, which record the measurement this row planned. |
| cua-grpo-v2 | 6-defect-subset | ddr | _not measured_ | 6 | superseded | RL dropped for the paper after grpo-v3 showed no measurable gain (docs/decisions/grpo-v3-results.md); no GRPO checkpoint will be evaluated. |
| cua-sft-v1 | 6-defect-subset | cost_per_case_usd | $0.88 | 6 | superseded | Opus-5 verdict $0.87 + Modal L40S ~$0.01. The verdict now dominates; a cheaper verdict tier is the remaining lever. Earlier duplicate of qa-e2e-hybrid-cost carrying the same unreproducible verdict term. |
| cua-sft-v1 | 6-defect-subset | execution_cost_per_case_usd | $0.01 | 6 | superseded | Execution phase only: $1.17 -> ~$0.01, about 100x. Metered natively as of vetty#22. Rounded restatement of qa-e2e-hybrid-execution; the measured accelerator cost is 0.01509 per case. |
| cua-sft-v2 | cost-model | e2e_cost_per_case_usd | $0.89 |  | superseded | HYBRID CONFIGURATION = the proposed architecture: local model executes, frontier model draws the conclusions. Composed from a measured local execution cost (GPU seconds x L40S $1.95/h) and a measured frontier verdict cost - NOT one instrumented end-to-end run, because the shipped product routes its conclusion steps to the local model too. Must be re-measured end to end once LocalModelAgentAdapter takes a frontier model_override. See docs/decisions/hybrid-cost-and-the-product-gap.md Sum of an execution term that holds and a verdict term that does not. Superseded by qa-e2e-hybrid-cost-v2. |
| cua-sft-v2 | cost-model | e2e_saving_vs_frontier | 1.46 |  | superseded | Frontier $1.30 / hybrid $0.89. Execution alone falls ~78x; the frontier conclusion step now dominates. HYBRID CONFIGURATION = the proposed architecture: local model executes, frontier model draws the conclusions. Composed from a measured local execution cost (GPU seconds x L40S $1.95/h) and a measured frontier verdict cost - NOT one instrumented end-to-end run, because the shipped product routes its conclusion steps to the local model too. Must be re-measured end to end once LocalModelAgentAdapter takes a frontier model_override. See docs/decisions/hybrid-cost-and-the-product-gap.md Computed from the two superseded totals. Superseded by qa-e2e-saving-ratio-v2. |
| cua-sft-v2 | cost-model | report_cost_per_case_usd | $0.87 |  | superseded | HYBRID CONFIGURATION = the proposed architecture: local model executes, frontier model draws the conclusions. Composed from a measured local execution cost (GPU seconds x L40S $1.95/h) and a measured frontier verdict cost - NOT one instrumented end-to-end run, because the shipped product routes its conclusion steps to the local model too. Must be re-measured end to end once LocalModelAgentAdapter takes a frontier model_override. See docs/decisions/hybrid-cost-and-the-product-gap.md No run on disk reproduces this figure. Its artifacts_path names results/smoke, which does not exist; the surviving runs of that date are the ones already retired as the crashed run. The verdict caches of matched-base, matched-sft-v2 and econ-sft-v3 store no cost fields at all, so every verdict they replayed was counted as free. Only econ-grpo-v3 recorded verdict usage. Superseded by qa-e2e-hybrid-verdict-v2. |
| scripted-playwright | skin-v2 | pass_rate_retained | _not measured_ | 13 | superseded | Superseded by the e3-retained-* rows, which record the measurement this row planned. |

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
| claude-sonnet-5 | all | accuracy | 68.3 % <sub>[66–71]</sub> | 1272 | done | 2026-09-05: full 1272-sample split on NATIVE source images, asked for PIXEL coordinates and scored native_abs. Replaces ss-1272-sonnet-*, which asked for the 0-1000 normalized space Qwen emits natively; the model half-complied and half-reverted to pixels, so no single mapping scored it. Asked for its own convention it is unambiguous: 100% parse, 1 of 1272 answers outside the image, and the same answers read as normalized_1000 score 4.4% against 68.3% native. VERIFIED: not an API resize artefact. claude-sonnet-5 is on the high-resolution tier (long edge 2576px, 4784 visual tokens), which downsizes only 74 of the 1272 images, and rescaling answers out of that frame gives 67.8%, slightly worse than leaving them alone; rescaling out of the standard 1568px frame gives 17.1%, which rules that tier out. Misses are wrong-element, not imprecise: median gap to the target box is 8.7% of the image diagonal and only 13.7% land within 50px. The weakness is by SOURCE, not resolution: 90.7% windows and 89.5% macos against 59.9% android, 62.4% ios and 42.7% gitlab. Comparability: Sonnet 5 has no published grounding figure. Claude models DO appear on ScreenSpot-Pro (Opus 4.8 87.9%, 4.6 83.1%, 4.5 45.7%) where they beat Qwen3-VL's self-reported 61.8-62.0%, but those runs permit cropping, visual search and planners by the leaderboard's own FAQ, while this row is bare single-turn. The ordering here is a property of the protocol, not a general capability claim. See docs/decisions/why-a-9b-outgrounds-the-frontier-model.md. |
| claude-sonnet-5 | all | parse_rate | 100.0 % <sub>[100–100]</sub> | 1272 | done | 2026-09-05: full 1272-sample split on NATIVE source images, asked for PIXEL coordinates and scored native_abs. Replaces ss-1272-sonnet-*, which asked for the 0-1000 normalized space Qwen emits natively; the model half-complied and half-reverted to pixels, so no single mapping scored it. Asked for its own convention it is unambiguous: 100% parse, 1 of 1272 answers outside the image, and the same answers read as normalized_1000 score 4.4% against 68.3% native. VERIFIED: not an API resize artefact. claude-sonnet-5 is on the high-resolution tier (long edge 2576px, 4784 visual tokens), which downsizes only 74 of the 1272 images, and rescaling answers out of that frame gives 67.8%, slightly worse than leaving them alone; rescaling out of the standard 1568px frame gives 17.1%, which rules that tier out. Misses are wrong-element, not imprecise: median gap to the target box is 8.7% of the image diagonal and only 13.7% land within 50px. The weakness is by SOURCE, not resolution: 90.7% windows and 89.5% macos against 59.9% android, 62.4% ios and 42.7% gitlab. Comparability: Sonnet 5 has no published grounding figure. Claude models DO appear on ScreenSpot-Pro (Opus 4.8 87.9%, 4.6 83.1%, 4.5 45.7%) where they beat Qwen3-VL's self-reported 61.8-62.0%, but those runs permit cropping, visual search and planners by the leaderboard's own FAQ, while this row is bare single-turn. The ordering here is a property of the protocol, not a general capability claim. See docs/decisions/why-a-9b-outgrounds-the-frontier-model.md. |
| claude-sonnet-5 | icon | accuracy | 65.9 % <sub>[62–70]</sub> | 575 | done | 2026-09-05: full 1272-sample split on NATIVE source images, asked for PIXEL coordinates and scored native_abs. Replaces ss-1272-sonnet-*, which asked for the 0-1000 normalized space Qwen emits natively; the model half-complied and half-reverted to pixels, so no single mapping scored it. Asked for its own convention it is unambiguous: 100% parse, 1 of 1272 answers outside the image, and the same answers read as normalized_1000 score 4.4% against 68.3% native. VERIFIED: not an API resize artefact. claude-sonnet-5 is on the high-resolution tier (long edge 2576px, 4784 visual tokens), which downsizes only 74 of the 1272 images, and rescaling answers out of that frame gives 67.8%, slightly worse than leaving them alone; rescaling out of the standard 1568px frame gives 17.1%, which rules that tier out. Misses are wrong-element, not imprecise: median gap to the target box is 8.7% of the image diagonal and only 13.7% land within 50px. The weakness is by SOURCE, not resolution: 90.7% windows and 89.5% macos against 59.9% android, 62.4% ios and 42.7% gitlab. Comparability: Sonnet 5 has no published grounding figure. Claude models DO appear on ScreenSpot-Pro (Opus 4.8 87.9%, 4.6 83.1%, 4.5 45.7%) where they beat Qwen3-VL's self-reported 61.8-62.0%, but those runs permit cropping, visual search and planners by the leaderboard's own FAQ, while this row is bare single-turn. The ordering here is a property of the protocol, not a general capability claim. See docs/decisions/why-a-9b-outgrounds-the-frontier-model.md. |
| claude-sonnet-5 | text | accuracy | 70.3 % <sub>[67–74]</sub> | 697 | done | 2026-09-05: full 1272-sample split on NATIVE source images, asked for PIXEL coordinates and scored native_abs. Replaces ss-1272-sonnet-*, which asked for the 0-1000 normalized space Qwen emits natively; the model half-complied and half-reverted to pixels, so no single mapping scored it. Asked for its own convention it is unambiguous: 100% parse, 1 of 1272 answers outside the image, and the same answers read as normalized_1000 score 4.4% against 68.3% native. VERIFIED: not an API resize artefact. claude-sonnet-5 is on the high-resolution tier (long edge 2576px, 4784 visual tokens), which downsizes only 74 of the 1272 images, and rescaling answers out of that frame gives 67.8%, slightly worse than leaving them alone; rescaling out of the standard 1568px frame gives 17.1%, which rules that tier out. Misses are wrong-element, not imprecise: median gap to the target box is 8.7% of the image diagonal and only 13.7% land within 50px. The weakness is by SOURCE, not resolution: 90.7% windows and 89.5% macos against 59.9% android, 62.4% ios and 42.7% gitlab. Comparability: Sonnet 5 has no published grounding figure. Claude models DO appear on ScreenSpot-Pro (Opus 4.8 87.9%, 4.6 83.1%, 4.5 45.7%) where they beat Qwen3-VL's self-reported 61.8-62.0%, but those runs permit cropping, visual search and planners by the leaderboard's own FAQ, while this row is bare single-turn. The ordering here is a property of the protocol, not a general capability claim. See docs/decisions/why-a-9b-outgrounds-the-frontier-model.md. |
| cua-grpo-v2 | all | accuracy | 85.0 % <sub>[81–89]</sub> | 300 | done | coord_space=viewport_abs. Merged SFT + GRPO LoRA, viewport_abs. No grounding regression against sft-v1. |
| cua-grpo-v2 | all | parse_rate | 96.7 % <sub>[94–98]</sub> | 300 | done |  |
| cua-grpo-v2 | full-split | accuracy | 78.6 % <sub>[76–81]</sub> | 1272 | done | Full 1,272-sample test split (desktop + mobile + web); the desktop-300 rows are the comparable ones. |
| cua-grpo-v2 | icon | accuracy | 75.4 % <sub>[67–82]</sub> | 122 | done |  |
| cua-grpo-v2 | text | accuracy | 91.6 % <sub>[87–95]</sub> | 178 | done |  |
| cua-grpo-v3 | all | accuracy | 85.1 % <sub>[83–87]</sub> | 1272 | done | 2026-09-05: full 1272-sample split on NATIVE source images, read back as viewport_abs, which grpo-v3 inherits from its sft-v2 parent. Replaces ss-1272-grpo-v3-*, measured on images squashed to 1280x720. This retires the grounding gain previously claimed for GRPO-3: on squashed images grpo-v3 scored 0.8270 against sft-v2's 0.7626, a 6.4-point lead, but on the source images the two are 0.8506 and 0.8538, a 0.3-point difference with fully overlapping Wilson intervals. The lead measured robustness to a harness artefact, not grounding. GRPO-3 therefore has no measured advantage on any lane. |
| cua-grpo-v3 | all | parse_rate | 98.3 % <sub>[97–99]</sub> | 1272 | done | 2026-09-05: full 1272-sample split on NATIVE source images, read back as viewport_abs, which grpo-v3 inherits from its sft-v2 parent. Replaces ss-1272-grpo-v3-*, measured on images squashed to 1280x720. This retires the grounding gain previously claimed for GRPO-3: on squashed images grpo-v3 scored 0.8270 against sft-v2's 0.7626, a 6.4-point lead, but on the source images the two are 0.8506 and 0.8538, a 0.3-point difference with fully overlapping Wilson intervals. The lead measured robustness to a harness artefact, not grounding. GRPO-3 therefore has no measured advantage on any lane. |
| cua-grpo-v3 | icon | accuracy | 77.2 % <sub>[74–80]</sub> | 575 | done | 2026-09-05: full 1272-sample split on NATIVE source images, read back as viewport_abs, which grpo-v3 inherits from its sft-v2 parent. Replaces ss-1272-grpo-v3-*, measured on images squashed to 1280x720. This retires the grounding gain previously claimed for GRPO-3: on squashed images grpo-v3 scored 0.8270 against sft-v2's 0.7626, a 6.4-point lead, but on the source images the two are 0.8506 and 0.8538, a 0.3-point difference with fully overlapping Wilson intervals. The lead measured robustness to a harness artefact, not grounding. GRPO-3 therefore has no measured advantage on any lane. |
| cua-grpo-v3 | text | accuracy | 91.5 % <sub>[89–93]</sub> | 697 | done | 2026-09-05: full 1272-sample split on NATIVE source images, read back as viewport_abs, which grpo-v3 inherits from its sft-v2 parent. Replaces ss-1272-grpo-v3-*, measured on images squashed to 1280x720. This retires the grounding gain previously claimed for GRPO-3: on squashed images grpo-v3 scored 0.8270 against sft-v2's 0.7626, a 6.4-point lead, but on the source images the two are 0.8506 and 0.8538, a 0.3-point difference with fully overlapping Wilson intervals. The lead measured robustness to a harness artefact, not grounding. GRPO-3 therefore has no measured advantage on any lane. |
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
| cua-sft-v2 | all | accuracy | 85.4 % <sub>[83–87]</sub> | 1272 | done | 2026-09-05: full 1272-sample split on NATIVE source images, the same picture every other arm is shown; answers read back as viewport_abs, which is what sft-v2 was trained to emit. Replaces ss-1272-sft-v2-*, where prepare_image belonged to the coordinate space and so resized the input to 1280x720 for viewport_abs arms only: sft-v2 and grpo-v3 were scored on squashed images while base, sft-v3 and the frontier arm saw the source. Correcting that moves sft-v2 from 0.7626 to 0.8538. Zero answers fall outside the declared space, so the readback is not in question. |
| cua-sft-v2 | all | parse_rate | 98.4 % <sub>[98–99]</sub> | 1272 | done | 2026-09-05: full 1272-sample split on NATIVE source images, the same picture every other arm is shown; answers read back as viewport_abs, which is what sft-v2 was trained to emit. Replaces ss-1272-sft-v2-*, where prepare_image belonged to the coordinate space and so resized the input to 1280x720 for viewport_abs arms only: sft-v2 and grpo-v3 were scored on squashed images while base, sft-v3 and the frontier arm saw the source. Correcting that moves sft-v2 from 0.7626 to 0.8538. Zero answers fall outside the declared space, so the readback is not in question. |
| cua-sft-v2 | icon | accuracy | 77.4 % <sub>[74–81]</sub> | 575 | done | 2026-09-05: full 1272-sample split on NATIVE source images, the same picture every other arm is shown; answers read back as viewport_abs, which is what sft-v2 was trained to emit. Replaces ss-1272-sft-v2-*, where prepare_image belonged to the coordinate space and so resized the input to 1280x720 for viewport_abs arms only: sft-v2 and grpo-v3 were scored on squashed images while base, sft-v3 and the frontier arm saw the source. Correcting that moves sft-v2 from 0.7626 to 0.8538. Zero answers fall outside the declared space, so the readback is not in question. |
| cua-sft-v2 | text | accuracy | 92.0 % <sub>[90–94]</sub> | 697 | done | 2026-09-05: full 1272-sample split on NATIVE source images, the same picture every other arm is shown; answers read back as viewport_abs, which is what sft-v2 was trained to emit. Replaces ss-1272-sft-v2-*, where prepare_image belonged to the coordinate space and so resized the input to 1280x720 for viewport_abs arms only: sft-v2 and grpo-v3 were scored on squashed images while base, sft-v3 and the frontier arm saw the source. Correcting that moves sft-v2 from 0.7626 to 0.8538. Zero answers fall outside the declared space, so the readback is not in question. |
| cua-sft-v3 | all | accuracy | 91.6 % <sub>[90–93]</sub> | 1272 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. |
| cua-sft-v3 | all | parse_rate | 100.0 % <sub>[100–100]</sub> | 1272 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. |
| cua-sft-v3 | icon | accuracy | 88.0 % <sub>[85–90]</sub> | 575 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. |
| cua-sft-v3 | text | accuracy | 94.5 % <sub>[93–96]</sub> | 697 | done | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. |
| qwen35-9b-base | all | accuracy | 89.0 % <sub>[85–92]</sub> | 300 | done | coord_space=normalized_1000 |
| qwen35-9b-base | all | accuracy | 89.0 % <sub>[85–92]</sub> | 300 | done | coord_space=normalized_1000. Served stock base, normalized_1000, same night as the grpo-v2 rows. |
| qwen35-9b-base | all | accuracy | 72.2 % <sub>[70–75]</sub> | 1272 | done | 2026-09-05, parser-corrected. Same endpoint/harness/token budget as the sft-v2 and sft-v3 rows; this re-run additionally accepts the base model's own `click(<x, y>)` spelling, which the earlier floor row could not parse. Supersedes ss-1272-base-* as first written. Artifacts eval_results/base_parserfix/. |
| qwen35-9b-base | all | parse_rate | 94.7 % <sub>[92–97]</sub> | 300 | done |  |
| qwen35-9b-base | all | parse_rate | 94.3 % <sub>[91–96]</sub> | 300 | done |  |
| qwen35-9b-base | all | parse_rate | 86.2 % <sub>[84–88]</sub> | 1272 | done | 2026-09-05, parser-corrected. Same endpoint/harness/token budget as the sft-v2 and sft-v3 rows; this re-run additionally accepts the base model's own `click(<x, y>)` spelling, which the earlier floor row could not parse. Supersedes ss-1272-base-* as first written. Artifacts eval_results/base_parserfix/. |
| qwen35-9b-base | icon | accuracy | 80.3 % <sub>[72–86]</sub> | 122 | done |  |
| qwen35-9b-base | icon | accuracy | 80.3 % <sub>[72–86]</sub> | 122 | done |  |
| qwen35-9b-base | icon | accuracy | 73.2 % <sub>[69–77]</sub> | 575 | done | 2026-09-05, parser-corrected. Same endpoint/harness/token budget as the sft-v2 and sft-v3 rows; this re-run additionally accepts the base model's own `click(<x, y>)` spelling, which the earlier floor row could not parse. Supersedes ss-1272-base-* as first written. Artifacts eval_results/base_parserfix/. |
| qwen35-9b-base | text | accuracy | 94.9 % <sub>[91–97]</sub> | 178 | done |  |
| qwen35-9b-base | text | accuracy | 94.9 % <sub>[91–97]</sub> | 178 | done |  |
| qwen35-9b-base | text | accuracy | 71.5 % <sub>[68–75]</sub> | 697 | done | 2026-09-05, parser-corrected. Same endpoint/harness/token budget as the sft-v2 and sft-v3 rows; this re-run additionally accepts the base model's own `click(<x, y>)` spelling, which the earlier floor row could not parse. Supersedes ss-1272-base-* as first written. Artifacts eval_results/base_parserfix/. |
| claude-sonnet-5 | all | accuracy | 36.5 % <sub>[34–39]</sub> | 1272 | superseded | WITHDRAWN 2026-09-05 — not measurable under this protocol. claude-sonnet-5 answers in native pixels on some samples and the prompt's 0-1000 normalization on others: 36.5% scoring every answer as native, 19.4% scoring every answer as normalized, 51.7% if each answer is scored under whichever mapping suits it, and only 4.2% where both agree. 74% of answers have both coordinates <=1000 and are ambiguous on their face. The true value is bounded by [36.5, 51.7] and the harness cannot resolve it, because the model does not commit to the convention the prompt specifies. An earlier note on this row attributed the low score to vision-encoder downsampling; that was wrong. Accuracy correlates with source resolution only because the two conventions nearly coincide at 960x540 (scale 0.96) and diverge 2.56x at 2560x1440. Artifacts eval_results/sonnet_final/. Superseded by ss-1272n-sonnet-*: scored on images squashed to 1280x720, which only the viewport_abs arms were, so the row was not comparable with the rest of the table. |
| claude-sonnet-5 | all | accuracy | 20.1 % <sub>[18–22]</sub> | 1272 | superseded | 2026-09-05: superseded, first frontier attempt. Prompted for the 0-1000 normalized space and additionally hit the max_tokens cap. Withdrawn for the same reason as ss-1272-sonnet-*: the prompt measured convention translation, not grounding. Final value is ss-1272n-sonnet-acc. |
| claude-sonnet-5 | all | accuracy | 21.5 % <sub>[19–24]</sub> | 1272 | superseded | 2026-09-05: superseded, second frontier attempt with the parser widened and the token budget raised, still prompted for the 0-1000 space. Final value is ss-1272n-sonnet-acc. |
| claude-sonnet-5 | all | accuracy | 24.0 % <sub>[18–31]</sub> | 150 | superseded | 2026-09-05: superseded probe, n=150 under the 0-1000 prompt. Not representative: --limit head-samples in dataset order, so the subset is one data_source rather than a cross-section. |
| claude-sonnet-5 | all | accuracy | 90.8 % <sub>[84–95]</sub> | 120 | superseded | 2026-09-05: superseded probe, n=120, first run of the pixel-coordinate prompt. It confirmed the convention fix (100% parse, no out-of-bounds answers) but its 90.8% was reported as an accuracy estimate and should not have been: --limit head-samples in dataset order, so all 120 were windows screenshots, the single source this arm scores best on (90.7% on the full split). The full 1272-sample value is 68.3%, ss-1272n-sonnet-acc. |
| claude-sonnet-5 | all | parse_rate | 99.6 % <sub>[99–100]</sub> | 1272 | superseded | 2026-09-05: same 1272-sample split and 2048-token budget as the local arms, prompted with the identical grounding observation. Scored in NATIVE PIXELS, chosen by testing every candidate mapping against the recorded source dimensions: native 36.5%, normalized_1000 19.4%, and an Anthropic-downsample correction 11.6-24.1% — native wins and only 0.2% of predictions fall outside the image, so this is not a convention mismatch. Accuracy tracks source resolution (86.4% at 960x540, 12.7% at 2560x1440), consistent with vision-encoder downsampling removing the detail small UI targets need. NOT comparable to published Claude Computer Use figures: those drive Anthropic's native computer-use tool, while this asks for coordinates as text through a grammar built for a 9B student. Artifacts eval_results/sonnet_final/. Superseded by ss-1272n-sonnet-*: scored on images squashed to 1280x720, which only the viewport_abs arms were, so the row was not comparable with the rest of the table. |
| claude-sonnet-5 | all | parse_rate | 99.7 % <sub>[99–100]</sub> | 1272 | superseded | 2026-09-05: superseded, first frontier attempt. Prompted for the 0-1000 normalized space and additionally hit the max_tokens cap. Withdrawn for the same reason as ss-1272-sonnet-*: the prompt measured convention translation, not grounding. Final value is ss-1272n-sonnet-acc. |
| claude-sonnet-5 | all | parse_rate | 99.5 % <sub>[99–100]</sub> | 1272 | superseded | 2026-09-05: superseded, second frontier attempt with the parser widened and the token budget raised, still prompted for the 0-1000 space. Final value is ss-1272n-sonnet-acc. |
| claude-sonnet-5 | all | parse_rate | 99.3 % <sub>[96–100]</sub> | 150 | superseded | 2026-09-05: superseded probe, n=150 under the 0-1000 prompt. Not representative: --limit head-samples in dataset order, so the subset is one data_source rather than a cross-section. |
| claude-sonnet-5 | all | parse_rate | 100.0 % <sub>[97–100]</sub> | 120 | superseded | 2026-09-05: superseded probe, n=120, first run of the pixel-coordinate prompt. It confirmed the convention fix (100% parse, no out-of-bounds answers) but its 90.8% was reported as an accuracy estimate and should not have been: --limit head-samples in dataset order, so all 120 were windows screenshots, the single source this arm scores best on (90.7% on the full split). The full 1272-sample value is 68.3%, ss-1272n-sonnet-acc. |
| claude-sonnet-5 | icon | accuracy | 33.8 % <sub>[30–38]</sub> | 571 | superseded | WITHDRAWN 2026-09-05 — not measurable under this protocol. claude-sonnet-5 answers in native pixels on some samples and the prompt's 0-1000 normalization on others: 36.5% scoring every answer as native, 19.4% scoring every answer as normalized, 51.7% if each answer is scored under whichever mapping suits it, and only 4.2% where both agree. 74% of answers have both coordinates <=1000 and are ambiguous on their face. The true value is bounded by [36.5, 51.7] and the harness cannot resolve it, because the model does not commit to the convention the prompt specifies. An earlier note on this row attributed the low score to vision-encoder downsampling; that was wrong. Accuracy correlates with source resolution only because the two conventions nearly coincide at 960x540 (scale 0.96) and diverge 2.56x at 2560x1440. Artifacts eval_results/sonnet_final/. Superseded by ss-1272n-sonnet-*: scored on images squashed to 1280x720, which only the viewport_abs arms were, so the row was not comparable with the rest of the table. |
| claude-sonnet-5 | text | accuracy | 38.9 % <sub>[35–43]</sub> | 696 | superseded | WITHDRAWN 2026-09-05 — not measurable under this protocol. claude-sonnet-5 answers in native pixels on some samples and the prompt's 0-1000 normalization on others: 36.5% scoring every answer as native, 19.4% scoring every answer as normalized, 51.7% if each answer is scored under whichever mapping suits it, and only 4.2% where both agree. 74% of answers have both coordinates <=1000 and are ambiguous on their face. The true value is bounded by [36.5, 51.7] and the harness cannot resolve it, because the model does not commit to the convention the prompt specifies. An earlier note on this row attributed the low score to vision-encoder downsampling; that was wrong. Accuracy correlates with source resolution only because the two conventions nearly coincide at 960x540 (scale 0.96) and diverge 2.56x at 2560x1440. Artifacts eval_results/sonnet_final/. Superseded by ss-1272n-sonnet-*: scored on images squashed to 1280x720, which only the viewport_abs arms were, so the row was not comparable with the rest of the table. |
| cua-grpo-v3 | all | accuracy | 82.7 % <sub>[81–85]</sub> | 1272 | superseded | 2026-09-05: same 1272-sample split, harness and 2048-token budget as the base/sft-v2/sft-v3 rows, served from its own rental because grpo-v3 is a LoRA over the MERGED sft-v2 weights and cannot be stacked on the raw base. Scored in viewport_abs, which it inherits from sft-v2. Artifacts eval_results/rentalB/. Superseded by ss-1272n-grpo-v3-*: scored on images squashed to 1280x720, which only the viewport_abs arms were, so the row was not comparable with the rest of the table. |
| cua-grpo-v3 | all | parse_rate | 98.7 % <sub>[98–99]</sub> | 1272 | superseded | 2026-09-05: same 1272-sample split, harness and 2048-token budget as the base/sft-v2/sft-v3 rows, served from its own rental because grpo-v3 is a LoRA over the MERGED sft-v2 weights and cannot be stacked on the raw base. Scored in viewport_abs, which it inherits from sft-v2. Artifacts eval_results/rentalB/. Superseded by ss-1272n-grpo-v3-*: scored on images squashed to 1280x720, which only the viewport_abs arms were, so the row was not comparable with the rest of the table. |
| cua-grpo-v3 | icon | accuracy | 75.0 % <sub>[71–78]</sub> | 575 | superseded | 2026-09-05: same 1272-sample split, harness and 2048-token budget as the base/sft-v2/sft-v3 rows, served from its own rental because grpo-v3 is a LoRA over the MERGED sft-v2 weights and cannot be stacked on the raw base. Scored in viewport_abs, which it inherits from sft-v2. Artifacts eval_results/rentalB/. Superseded by ss-1272n-grpo-v3-*: scored on images squashed to 1280x720, which only the viewport_abs arms were, so the row was not comparable with the rest of the table. |
| cua-grpo-v3 | text | accuracy | 89.1 % <sub>[87–91]</sub> | 697 | superseded | 2026-09-05: same 1272-sample split, harness and 2048-token budget as the base/sft-v2/sft-v3 rows, served from its own rental because grpo-v3 is a LoRA over the MERGED sft-v2 weights and cannot be stacked on the raw base. Scored in viewport_abs, which it inherits from sft-v2. Artifacts eval_results/rentalB/. Superseded by ss-1272n-grpo-v3-*: scored on images squashed to 1280x720, which only the viewport_abs arms were, so the row was not comparable with the rest of the table. |
| cua-sft-v2 | all | accuracy | 76.3 % <sub>[74–79]</sub> | 1272 | superseded | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. Superseded by ss-1272n-sft-v2-*: scored on images squashed to 1280x720, which only the viewport_abs arms were, so the row was not comparable with the rest of the table. |
| cua-sft-v2 | all | accuracy | 76.3 % <sub>[74–79]</sub> | 1272 | superseded | 2026-09-04: superseded. Same max_tokens=512 run as ss-1272-cap512-base; this arm answers tersely and was nearly unaffected (76.34% here against 76.26% at 2048), which is what made the cap a one-sided penalty rather than a uniform one. |
| cua-sft-v2 | all | parse_rate | 91.8 % <sub>[90–93]</sub> | 1272 | superseded | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. Superseded by ss-1272n-sft-v2-*: scored on images squashed to 1280x720, which only the viewport_abs arms were, so the row was not comparable with the rest of the table. |
| cua-sft-v2 | all | parse_rate | 91.8 % <sub>[90–93]</sub> | 1272 | superseded | 2026-09-04: superseded. Same max_tokens=512 run as ss-1272-cap512-base; this arm answers tersely and was nearly unaffected (76.34% here against 76.26% at 2048), which is what made the cap a one-sided penalty rather than a uniform one. |
| cua-sft-v2 | icon | accuracy | 67.5 % <sub>[64–71]</sub> | 575 | superseded | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. Superseded by ss-1272n-sft-v2-*: scored on images squashed to 1280x720, which only the viewport_abs arms were, so the row was not comparable with the rest of the table. |
| cua-sft-v2 | text | accuracy | 83.5 % <sub>[81–86]</sub> | 697 | superseded | 2026-09-05 run: all three arms scored from ONE vLLM endpoint under one harness commit, on the FULL 1272-sample test split (earlier rows used --limit 300, head-sampled in dataset order). max_tokens 2048 identically across arms - at the previous hard-coded 512 the base model's reasoning was cut off before it emitted a click, which scored the cap rather than the model. Each arm scored in its own coord_space. Artifacts eval_results/rentalA_2048/. Superseded by ss-1272n-sft-v2-*: scored on images squashed to 1280x720, which only the viewport_abs arms were, so the row was not comparable with the rest of the table. |
| cua-sft-v3 | all | accuracy | 91.9 % <sub>[90–93]</sub> | 1272 | superseded | 2026-09-04: superseded. Same max_tokens=512 run; 91.90% here against 91.59% at 2048. |
| cua-sft-v3 | all | parse_rate | 100.0 % <sub>[100–100]</sub> | 1272 | superseded | 2026-09-04: superseded. Same max_tokens=512 run; 91.90% here against 91.59% at 2048. |
| qwen35-9b-base | all | accuracy | 54.1 % <sub>[51–57]</sub> | 1272 | superseded | 2026-09-04: superseded by ss-1272-base-*. Scored under the hard-coded max_tokens=512 in OpenAIPredictor, which ignored the configured value. The base model reasons before answering, so 462 of 1272 samples were cut off mid-thought and never emitted a click — 54.1% measures the cap, not the model. The terse fine-tuned arms in the same run were barely affected, so the cap penalised exactly one arm of the comparison. |
| qwen35-9b-base | all | parse_rate | 63.7 % <sub>[61–66]</sub> | 1272 | superseded | 2026-09-04: superseded by ss-1272-base-*. Scored under the hard-coded max_tokens=512 in OpenAIPredictor, which ignored the configured value. The base model reasons before answering, so 462 of 1272 samples were cut off mid-thought and never emitted a click — 54.1% measures the cap, not the model. The terse fine-tuned arms in the same run were barely affected, so the cap penalised exactly one arm of the comparison. |

### Figures cited from other people's studies

| claim | value | n | venue | year | reference | used_on |
| --- | --- | --- | --- | --- | --- | --- |
| Share of record/replay web test breakages caused by stale locators | 73.6 % | 1065 | ICST | 2016 | 10.1109/ICST.2016.16 | deck slide 3, thesis ch.1 |
| Legacy GUI tests on live sites that stopped passing within 18 months | 100.0 % | 120 | ASE | 2025 | 10.1109/ASE63991.2025.00251 | deck slide 3, thesis ch.1 |

_347 registry rows: 289 done, 1 pending, 3 planned, 54 superseded. Percentages carry a 95 % Wilson interval where n is known._

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
