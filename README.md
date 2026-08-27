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
