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

The pre-training baseline every fine-tune is compared against. Captured 2026-08-26 with
`make eval-online-mind2web`; raw artefacts in `eval_runs/baseline_qwen35_9b_online_mind2web_20260826T165014Z/`.

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
