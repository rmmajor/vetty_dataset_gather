# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A data-gathering and training pipeline that distills GPT-5.4's computer-use (web browsing) ability into Qwen 3.5-9B via trajectory-based SFT. **Two independent data sources** produce the **same trajectory JSON format**, and a single converter/trainer (`train_qwen.py`) consumes either:

- **Path A — GPT-5.4 CUA distillation** (`record_trajectories_from_gpt_api.py`): drives GPT-5.4's `computer` tool through Mind2Web tasks in a Playwright browser, recording each screenshot + reasoning + action. ~$0.55/trajectory.
- **Path B — QA repo pipeline** (`repo_pipeline/`): finds real Playwright test repos on GitHub, runs them in Docker sandboxes with injected instrumentation that captures screenshots + pixel coordinates, then has Claude write the `instruction` and per-step `<think>` reasoning. ~$0.05–0.15/trajectory.

The trajectory schema is the integration contract tying all three pieces together — see `repo_pipeline/schemas.py` (`Trajectory`, `TrajectoryStep`) and the output-format notes below before changing any producer or the converter.

## Trajectory format (the contract)

Every trajectory is a directory `task_*/` (Path A: `task_mind2web_NNNN/`; Path B: `repo_<owner>_<repo>/task_*/`) containing `trajectory.json`, `metadata.json`, and `step_NNN_screenshot.png` files.

```json
{"task_id", "instruction", "start_url",
 "steps": [{"step", "screenshot", "thought", "actions": [...]}],
 "total_steps", "success"}
```

Action dicts are normalized across both paths: `{type:"click", x, y, button}`, `{type:"type", text}`, `{type:"keypress", keys:[...]}`, `{type:"scroll", x, y, scroll_x, scroll_y}`, `{type:"drag", path:[{x,y}...]}`, `{type:"move", x, y}` (hover), `{type:"wait"}`, and the terminal `{type:"done"}`. Viewport is fixed at **1280x720, deviceScaleFactor=1** everywhere.

`train_qwen.py convert` emits **one SFT example per step** (current screenshot + instruction + windowed text history → `<think>` + actions; whole-trajectory multi-turn examples blew the 4096 max_length and truncated long-horizon tails). Gating: `metadata.excluded_reason` always skips; agent trajectories require `success: true` (the verifier's verdict); legacy replay trajectories train only with `--include-replay`. Every convert prints a corpus health report — read it.

## Common commands

```bash
# Setup
make setup                 # Path A deps + playwright chromium
make setup-pipeline        # Path B deps + chromium
make build-sandbox         # build Docker sandbox image (required for stages 4-6)
make test-sandbox          # verify container lifecycle works

# Path A
python load_tasks.py --output tasks.jsonl [--limit N]
make record                # needs OPENAI_API_KEY; --resume is on by default

# Path B (registry-driven, see below)
make pipeline-discover     # stage 1, needs GITHUB_TOKEN
make pipeline-filter       # stage 2
make pipeline-score PIPELINE_LIMIT=50 PIPELINE_BUDGET=5   # stage 3, costs money
make pipeline-overnight PIPELINE_BUDGET=30                # stages 3-7 once + Telegram report
make pipeline-until-limit PIPELINE_LIMIT_PCT=75           # loop stages 3-7 until 75% of Claude usage limit
make pipeline-status
make test-driver           # scripted (no-LLM) smoke test of the agent-capture browser driver
python -m repo_pipeline.orchestrator --stage 3-7 --limit 10 --budget 5 --workers 5
python -m repo_pipeline.orchestrator --stage 6 --capture-mode replay   # legacy test-replay capture
python -m repo_pipeline.orchestrator --stage 7          # finalize supervision for captured repos
python -m repo_pipeline.orchestrator --status

# Convert + train (works on output of either path)
make convert               # trajectories/ -> training_data/sft_dataset.jsonl
make train                 # invokes `swift sft` (needs ms-swift on a GPU box)

# Eval grounding before/after SFT (ScreenSpot)
make install-eval          # eval deps (GPU box)
make eval-screenspot       # evals every model in eval_models.yaml, writes eval_results/
make test-eval             # harness smoke test (no model/GPU/network)
```

There is no test suite. The executable checks are `make test-sandbox` (container smoke test), `make test-driver` (scripted no-LLM browser-driver session asserting the trajectory contract), `make test-notifier` (Telegram bot), and `make test-eval` (ScreenSpot eval harness smoke test). Verify changes by running the relevant stage with a small `--limit`/`--budget`.

## Evaluation (`eval_screenspot.py`)

ScreenSpot grounding eval: predicted click point vs. target bbox. Mirrors `train_qwen.py`'s style and **imports `SYSTEM_PROMPT` from it** to guarantee train/eval prompt parity. Everything is registry/flag-driven so it works for other models: coordinate spaces (`COORD_SPACES`: `viewport_abs` default matches our 1280x720 training, plus `normalized_1000`/`native_abs`), inference backends (`PREDICTOR_BACKENDS`: `swift` shells out to `swift infer` like training shells out to `swift sft`; `mock` for tests), a configurable output parser (`DEFAULT_PARSE_PATTERNS`, tries `<ref>(x,y)</ref>` first), and a configurable dataset/bbox adapter. Evaluates N model specs from `eval_models.yaml` (default before/after) on one dataset. The `swift infer` flags/result schema and the ScreenSpot-v2 column/bbox conventions are assumptions that must be validated on the GPU box / against the real dataset — they are isolated behind flags and `SwiftPredictor`. Heavy deps (`datasets`) are imported lazily so `make test-eval` needs none.

## Path B architecture (`repo_pipeline/`)

A 7-stage quality funnel, cast wide then reject aggressively. `orchestrator.py` is the entry point; it sets up logging to `pipeline_data/run_logs/`, then dispatches stages. Capture (6) and supervision finalize (7) are split so a slow/flaky LLM pass can never block or fail a capture.

**Capture modes.** `schemas.CAPTURE_MODE` (env `PIPELINE_CAPTURE_MODE` / CLI `--capture-mode`, default **`agent`**) switches how stages 4-6 behave:
- **agent** (primary): the repo's app is booted inside the sandbox and a Claude teacher (`agent_capture.py`) pursues user tasks proposed by `task_proposer.py`, acting through an in-container Playwright HTTP driver (`instrumentation/browser_driver.js` + host wrapper `driver_session.py`) that records a screenshot + bbox-center coordinates per action. Every action tool requires a `thought` argument — per-step intent-level supervision is structural, not retrofitted. A post-run verifier judges the final screenshot against the goal; only verified trajectories get `success: true` (coherent partials are hindsight-relabeled to what they actually accomplished). Framework-independent: the repo's own tests are only feature hints for the proposer.
- **replay** (secondary/legacy): inject instrumentation into the repo's Playwright/Selenium suite and record what the tests do.

**Registry state machine is the spine.** All repo state lives in `pipeline_data/registry.db` (SQLite, WAL). `schemas.py` defines three dicts — `STAGE_TO_INPUT_STATUS`, `STAGE_TO_PASSED_STATUS`, `STAGE_TO_REJECTED_STATUS` — that map each stage to the `RepoStatus` it consumes / promotes to / rejects to:

```
discovered → filtered → scored → analyzed → validated → captured → completed
                ↓          ↓         ↓          ↓           ↓
          rejected_filter / _score / _analyze / _trial / _generate
                                                  (stage 7 "think": captured → completed, never rejects)
```

Each stage calls `registry.get_repos_for_stage(N)` (which selects by input status, ordered by a query-priority heuristic that deprioritizes demo/tutorial repos), then for each repo either `registry.update_status(..., PASSED)` or `registry.reject(...)`. Stages are **idempotent**: repos guard on `current_stage >= N` and re-running never reprocesses. To re-evaluate rejections use `registry.reset_rejected(stage=N)`. When adding a stage or status, extend the three mapping dicts rather than hardcoding stage numbers.

**Stage responsibilities:**
- **1 discover** (`stage1_discover.py`): GitHub API across ~50+ `SEARCH_QUERIES`. No LLM. Needs `GITHUB_TOKEN`.
- **2 filter** (`stage2_filter.py`): size/recency filters + fork & duplicate-description dedup. No LLM.
- **3 shallow_score**: fetches repo metadata via `gh` CLI, Claude scores 7 dimensions (`ShallowScores`), gate at `SCORE_THRESHOLD`.
- **4 deep_analyze**: static analysis in a `DockerSandbox`. Bootability (dev server / webServer / docker-compose) is the dominant score axis; in agent mode, public-URL-only suites with nothing bootable are hard-rejected, and framework support is irrelevant (replay mode still gates on `is_supported_profile`).
- **5 trial_run**: agent mode = **boot gate** (`boot_gate_single_repo`): clone, `npm install`, daemonize the app (`driver_session.start_app`), probe it through the browser driver, reject `NO_BOOT` / trivial UIs (`BOOT_GATE_MIN_ELEMENTS`). Replay mode = legacy instrumented POC test run.
- **6 capture** (`stage6_generate.py`): agent mode boots the app, proposes tasks, runs the teacher per task (fresh browser context each; one shared app server), verifies + dedups + caps per repo (`AGENT_MAX_KEPT_PER_REPO`, default 10), enriches metadata, marks **`captured`**. Resume-aware: re-running skips capture if valid trajectories already exist; partial sessions are copied out even on usage-limit aborts.
- **7 finalize** (`stage7_think.py`): agent trajectories get their raw teacher thoughts POLISHED into student voice (one text-only call per trajectory, `thought_source: teacher_raw → teacher`); replay trajectories get thoughts GENERATED from screenshots. Then marks `completed`. Decoupled, resumable, **never rejects**.

**Sandbox isolation** (`sandbox.py`): repo code (clone, npm/pip install, test runs) executes only inside an ephemeral, capability-dropped Docker container (`repo-pipeline-sandbox:latest`, built from `docker/Dockerfile.sandbox` on the Playwright base image). `DockerSandbox` is a context manager exposing `exec_command/read_file/write_file/copy_out`; it enforces a 30-min container and 5-min per-command timeout. `create_mcp_server_config` / `sandbox_tool_executor` expose these as tools for the agent.

**Instrumentation** (`instrumentation/`): `capture.py`'s `ActionRecorder` is the shared writer producing the trajectory format; `pw_python_conftest.py` (pytest-playwright) and `pw_ts_fixture.ts` (@playwright/test) hook Playwright actions to capture a screenshot + bounding-box center coordinate per action. These files are written into the cloned repo inside the sandbox by stages 5/6.

**Think/instruction generation** (`think_generator.py`, `stage6_generate.py`): Claude fills the empty `thought` fields and the `instruction` after capture — instrumentation records coords/screenshots only. Both generators are *intent-anchored*: the thought prompt receives the trajectory's goal + the screenshot + the history of prior thoughts; the instruction prompt receives the test function source (cached by instrumentation as `test_source.txt`) + the repo description. Outputs are trusted as written — no regex post-filters discard or retry the model's response. To regenerate over already-recorded trajectories, run `python -m repo_pipeline.regenerate_supervision --dry-run --sample N trajectories/` first to review old-vs-new in a Markdown report, then drop `--dry-run` to bulk-write.

## Claude Agent SDK usage

Path B calls Claude via `claude_agent_sdk.query()` returning `AsyncIterator[Message]`. Standard extraction pattern (used in every stage): iterate, take `message.result` from a `ResultMessage` where `message.subtype == "success"`. Options are `ClaudeAgentOptions(model, max_turns, max_budget_usd, permission_mode="bypassPermissions", allowed_tools, cwd)`. Models: the latest Sonnet-class model (currently `claude-sonnet-5`) across the pipeline (analysis, scoring, instruction generation, thought generation, task-type classification). Configured centrally in `repo_pipeline/llm_config.py` — `LATEST_MODEL_BY_CLASS` maps each class (opus/sonnet/haiku) to its latest ID, and every call site imports a named purpose (`MODEL_FOR_REASONING`, `MODEL_FOR_GROUNDED_GENERATION`, `MODEL_FOR_CLASSIFICATION`) rather than a model string.

## Budget & concurrency

LLM stages run with an `asyncio.Semaphore(workers)`; the orchestrator caps workers per stage (3 for analyze, 2 for trial/generate) since those hold a container each. Cost control is two-layered: `orchestrator.STAGE_COST_ESTIMATES` stops launching new stages once the run budget is spent, and each stage tracks its own `cost_estimate` against `budget_remaining`. `--budget` is a soft cap based on per-repo estimates, not metered spend.

`--overnight` and `--until-limit` both run through `orchestrator.drain_funnel` (the generalized stage-3→6 loop): overnight is one pass, until-limit loops until the usage threshold / budget / drained funnel. Every `query()` goes through `repo_pipeline/sdk_runner.run_query`, which feeds the SDK's `RateLimitEvent` into a process-global `rate_limit_tracker` (per-`rate_limit_type` utilization) and raises `UsageLimitReached` on a hard 429. That exception is re-raised (never caught-and-rejected) at every layer between `run_query` and each stage's `process_with_semaphore`, so hitting the usage limit leaves repos in their input status for the next run instead of rejecting them. `--until-limit PCT` sets `rate_limit_tracker.threshold = PCT/100`; with no usage events observed it degrades to "stop on hard 429".

## Environment variables

`OPENAI_API_KEY` (Path A), `GITHUB_TOKEN` + `ANTHROPIC_API_KEY` (Path B), `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` (optional, for overnight reports). `crontab.example` shows the nightly `--overnight` job.

Sandbox resource caps (per Docker container, `repo_pipeline/sandbox.py`) are env-tunable so a run can't exhaust host RAM: `SANDBOX_MEM_LIMIT` (default `2g`), `SANDBOX_MEMSWAP_LIMIT` (default = mem limit ⇒ no swap, hard cap), `SANDBOX_PIDS_LIMIT` (`1024`), `SANDBOX_CPUS` (`2`), `SANDBOX_TMP_SIZE` (RAM-backed `/tmp`, `1g`), `SANDBOX_SHM_SIZE` (`256m`). Peak ≈ `workers * (mem + tmpfs)`; keep it under the Docker VM memory. Raise `SANDBOX_MEM_LIMIT` if a legitimately heavy suite gets OOM-killed.

## Training

`config.yaml` holds hyperparameters from the blueprint (Qwen3.5-9B-Base, LoRA r64/α128, lr 5e-5, 3 epochs, max_length 4096, `freeze_vit: true`, `max_pixels` exported as `MAX_PIXELS`). `train_qwen.py train` maps these keys to `swift sft` CLI flags; CLI overrides (`--epochs/--lr/--batch-size`) win over the config.
