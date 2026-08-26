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
