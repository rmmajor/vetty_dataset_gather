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

`train_qwen.py convert` emits **one SFT example per step** (current screenshot + instruction + windowed text history → `<think>` + actions; whole-trajectory multi-turn examples blew the 4096 max_length and truncated long-horizon tails). Gating: one shared function (`corpus_audit.trainability`) used by both convert and `make audit-corpus` — `metadata.excluded_reason` always skips; agent trajectories require `success: true` (the verifier's verdict); legacy replay trajectories train only with `--include-replay`. Verifier-graded meandering trajectories are handled by `--meandering-mode keep|subsample|exclude` (train lanes use `subsample`: head/tail steps + every Nth mid-wander step train, the rest stay history-only; `MEANDER_*` env knobs). Every convert prints a corpus health report AND enforces `corpus_health.json` thresholds (done coverage, teacher_raw=0, meandering example share, per-group share, action-share floors) — violations refuse the dataset; `--skip-health-gate` bypasses for experiments only.

Corpus data passes (dry-run by default, `APPLY=1` to write, all resumable): `make polish-pass` (stage-7 thought polish for trainable `teacher_raw` dirs outside the registry), `make relabel-recoverable` (audited hindsight relabel: propose goal from what the run accomplished, fresh `verify_trajectory` against the NEW instruction gates every write, provenance in `metadata.relabeled_from`), `make wall-demos` (trim blocked runs + synthesize terminal `handoff`/`done` wall supervision, original kept as `trajectory_prewall.json`), `make instruction-style` (rewrite procedure-flavored instructions into goals). `make capture-search` runs the search-goal capture lane (`public_sites_search.json`, `task_style: search` proposer block; sites must never appear in Online-Mind2Web's 300-task set).

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

# Student eval (did the checkpoint actually get better? — see "Student eval" below)
make eval-sources                                                     # registered task sources
make eval-online-mind2web EVAL_ENDPOINT=http://gpu:8000/v1 EVAL_MODEL=qwen EVAL_LABEL=before \
     EVAL_COORD_SPACE=normalized_1000                                 # stock model's convention
make eval-online-mind2web EVAL_ENDPOINT=http://gpu:8000/v1 EVAL_MODEL=qwen EVAL_LABEL=after \
     EVAL_ADAPTER=output/checkpoint-500 EVAL_COORD_SPACE=viewport_abs # fine-tuned convention
make eval-held-out EVAL_ENDPOINT=... EVAL_LABEL=before                # our own unseen tasks
make eval-compare EVAL_RUNS="eval_runs/before_... eval_runs/after_..."  # the delta table
make test-student-eval                                                # harness self-test
# Eval grounding before/after SFT (ScreenSpot)
make install-eval          # eval deps (GPU box)
make eval-screenspot       # evals every model in eval_models.yaml, writes eval_results/
make test-eval             # harness smoke test (no model/GPU/network)
```

Two eval lanes, deliberately complementary: `eval_screenspot.py` measures GROUNDING (one screenshot, one instruction, is the predicted point inside the target box), and `repo_pipeline/student_eval.py` measures whether the model can actually complete a task across many steps in a live browser. Grounding can be excellent while task success is near zero, which is exactly what the baseline shows.

There is no test suite. The executable checks are `make test-sandbox` (container smoke test), `make test-driver` (scripted no-LLM browser-driver session asserting the trajectory contract), `make test-eval` (ScreenSpot eval harness smoke test), `make test-student-eval` (episode-eval parser round-trip + mock replay episode + judge), and `make test-notifier` (Telegram bot). Verify changes by running the relevant stage with a small `--limit`/`--budget`.

## Evaluation (`eval_screenspot.py`)

ScreenSpot grounding eval: predicted click point vs. target bbox. Mirrors `train_qwen.py`'s style and **imports `SYSTEM_PROMPT` from it** to guarantee train/eval prompt parity (so `train_qwen.py` — and the `repo_pipeline.corpus_audit` it imports — must be present wherever the eval runs). Registry/flag-driven: `COORD_SPACES`, `PREDICTOR_BACKENDS` (`swift` shells out to `swift infer`; `mock` for tests), a configurable parser (`DEFAULT_PARSE_PATTERNS`, tries `<ref>(x,y)</ref>` first), and a configurable dataset/bbox adapter. Evaluates N specs from `eval_models.yaml`. Heavy deps (`datasets`) load lazily so `make test-eval` needs none. Validated end-to-end on a rented GPU (`swift infer` flags confirmed).

**Measured baseline:** `Qwen/Qwen3.5-9B` (instruct, no SFT) scores **85.0%** on `rootsautomation/ScreenSpot` `test` (n=100, 93% parse, text 96% / icon 86%). That is the bar SFT must beat.

**The scoring conventions are the whole ballgame — get one wrong and the eval reports a confident WRONG number instead of crashing.** All were verified against real model output:
- **`coord_space` is PER-MODEL** (`ModelSpec.coord_space`, set it in `eval_models.yaml`). A stock Qwen answers in `normalized_1000`; a checkpoint fine-tuned on this corpus answers in `viewport_abs` (the 1280x720 training viewport). The *same* base-model predictions score **85% as `normalized_1000` and 25% as `viewport_abs`** — so a before/after comparison scored with one global coord space is meaningless.
- **`--bbox-normalized` (default on)** — the default dataset stores bbox in 0-1, not pixels. Treating it as pixels shrinks every target to a sub-pixel box at the origin: **0% by construction, regardless of the model.**
- **`--bbox-format xyxy` (default)** — that mirror stores `(x1,y1,x2,y2)`, not `xywh`.
- The default dataset is `rootsautomation/ScreenSpot`, **not** `os-copilot/ScreenSpot-v2` (that repo is raw JSON + an image zip: no `test` split, no image column, `load_dataset` cannot read it). Re-check all three conventions against any other mirror you point `--dataset` at.

## Path B architecture (`repo_pipeline/`)

A 7-stage quality funnel, cast wide then reject aggressively. `orchestrator.py` is the entry point; it sets up logging to `pipeline_data/run_logs/`, then dispatches stages. Capture (6) and supervision finalize (7) are split so a slow/flaky LLM pass can never block or fail a capture.

**Capture modes.** `schemas.CAPTURE_MODE` (env `PIPELINE_CAPTURE_MODE` / CLI `--capture-mode`, default **`agent`**) switches how stages 4-6 behave:
- **agent** (primary): the repo's app is booted inside the sandbox and a Claude teacher (`agent_capture.py`) pursues user tasks proposed by `task_proposer.py`, acting through an in-container Playwright HTTP driver (`instrumentation/browser_driver.js` + host wrapper `driver_session.py`) that records a screenshot + bbox-center coordinates per action. Every action tool requires a `thought` argument — per-step intent-level supervision is structural, not retrofitted. A post-run verifier judges the final screenshot against the goal; only verified trajectories get `success: true`, judged ONLY against the original instruction — salvaging coherent partials is the separate audited `relabel_recoverable` pass (below), never the capture-time verifier. Framework-independent: the repo's own tests are only feature hints for the proposer.
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

Student eval: `HF_READ_TOKEN` (or `HF_TOKEN`) for the gated Online-Mind2Web dataset, and `EVAL_ENDPOINT` / `EVAL_MODEL` / `EVAL_API_KEY` / `EVAL_ADAPTER` as make-level defaults for the served model.

Sandbox resource caps (per Docker container, `repo_pipeline/sandbox.py`) are env-tunable so a run can't exhaust host RAM: `SANDBOX_MEM_LIMIT` (default `2g`), `SANDBOX_MEMSWAP_LIMIT` (default = mem limit ⇒ no swap, hard cap), `SANDBOX_PIDS_LIMIT` (`1024`), `SANDBOX_CPUS` (`2`), `SANDBOX_TMP_SIZE` (RAM-backed `/tmp`, `1g`), `SANDBOX_SHM_SIZE` (`256m`). Peak ≈ `workers * (mem + tmpfs)`; keep it under the Docker VM memory. Raise `SANDBOX_MEM_LIMIT` if a legitimately heavy suite gets OOM-killed.

## Student eval (`repo_pipeline/student_eval.py`)

The measurement side of the loop: run a served model through real browser episodes and report
whether a fine-tuned checkpoint beat the base model. A single number is useless — **the
deliverable is a delta**, so every run takes a `--label` (`before` / `after`), writes to
`eval_runs/<run_id>/{report.md,results.json,run.log,episodes/}`, and `compare` renders the
before/after table plus a **harness parity check** that flags any field the two runs disagree on.

- **Prompt parity is the point.** The per-step user message is built by the same rules as
  `train_qwen.convert_trajectory_to_examples` (same `SYSTEM_PROMPT`, `Task:` / `Previous actions:`
  / `What action should you take?`, same `HISTORY_WINDOW`). An eval that prompts differently
  measures the prompt.
- **Model remote, browser local.** `--endpoint` is any OpenAI-compatible `/chat/completions`
  (vLLM, `swift deploy`, a rented Vast.ai box); the browser is a `DockerSandbox` on the invoking
  machine so live sites see a residential IP. Never assume the two are co-located.
- **Task sources are a registry** (`repo_pipeline/eval_task_sources.py`): `TaskSource` +
  `TASK_SOURCES`, same idiom as `frameworks.PROFILES`. `online_mind2web` (gated HF dataset,
  300 tasks, columns `task_id` / `confirmed_task` / `website` / `reference_length` / `level`) and
  `held_out` (our own never-captured `public_sites.json` + `proposed_tasks.json` tasks, proven
  unseen by an on-disk check against the corpus). Adding an eval is a data entry.
- **HF gating.** The dataset is `gated: auto`. The token is resolved explicitly — `HF_TOKEN`,
  `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, then `HF_READ_TOKEN` from `.env` (this repo's
  name, which no HF library looks for), then `~/.cache/huggingface/token`. A 401/403 fails loudly
  naming the URL to accept terms at; there is never a stub fallback.
- **`--coord-space` is PER MODEL and never auto-detected.** A model fine-tuned here answers in
  `viewport_abs` (1280x720 pixels); a stock Qwen answers in `normalized_1000`. Emitted positions
  are scaled to viewport pixels before dispatch (scroll *deltas* are relative motion and are not
  scaled), and `trajectory.json` records the EXECUTED pixel coordinates while the raw model text
  stays in `raw_last_response`. Getting this wrong does not error — it silently produces a
  confident wrong number, so the harness warns at the first impossible coordinate and prints a
  `COORDINATE SPACE SUSPECT` banner in `report.md` when >25% of emitted coordinates fall outside
  the viewport under `viewport_abs`. Same reasoning as `eval_models.yaml`'s `coord_space`.
  In `compare`, a `coord_space` difference is an *expected, annotated* model-side difference, not
  a parity violation — a before/after pair is supposed to differ here.
- **The action grammar is tolerant by design, and canonical form is its own metric.** The base
  model has never been trained on our grammar — the system prompt only *describes* it — so
  penalising surface deviation measures "can it imitate an unfamiliar convention", not "can it
  browse". The parser therefore accepts any **unambiguous** spelling of a call and reports
  `canonical_form_rate` (share of parsed actions spelled exactly as `format_action_for_training`
  emits) plus an `action_form_deviations` breakdown, instead of charging deviation to the success
  rate. The fine-tuned model emitting the canonical form is a real improvement and gets reported
  in that column. Tolerance is a **parity key** (`action_grammar: tolerant|canonical`), so
  `compare` proves both sides were parsed identically; `--strict-action-grammar` accepts canonical
  only. Accepted (canonical first):

  | verb | accepted spellings |
  | --- | --- |
  | `click` | `click(<ref>(x, y)</ref>, button=left)` · `(x, y)` · `x, y` · trailing `left`/`right` · button omitted (defaults left) |
  | `double_click`, `move` | `<ref>(x, y)</ref>` · `(x, y)` · `x, y` |
  | `drag` | both endpoints in any coordinate spelling, separated by `->` or `,` |
  | `scroll` | `scrollX=`/`scrollY=`, `scroll_x=`/`scroll_y=`, `dx=`/`dy=`, positional deltas, or omitted (defaults `0, 500`) |
  | `type` | `type("text")` · `type('text')` |
  | `keypress` | `["Enter"]` · `"Enter"` · `Enter` · `Control, a` · `Control+a` |
  | `wait`, `done`, `screenshot` | `verb()` · `verb` |
  | `navigate` | `navigate("url")` · `navigate('url')` · `navigate(url="…")` · bare `https?://` token; executes via the driver's `navigate` kind |
  | `handoff`, `confirm` | `handoff(reason=login)` · bare/quoted reason · `confirm(propose=call(...), risk=x)`; terminal — episode ends with scored outcome `handoff` |
  | everywhere | optional `action:` prefix, bullets/backticks/whitespace stripped, unknown kwargs ignored, floats rounded |

  **Rejected as genuinely ambiguous** (not merely unfamiliar): `type(hello world)` and
  `type(a, b)` — an unquoted string cannot be told from an argument list; `scroll(500)` — one
  number reads equally as an x coordinate or a scrollY delta; `click(100)`, `drag((10, 20))`,
  `keypress()`, `navigate()`, scheme-less `navigate(example.com)` — incomplete. Loosening the *shape* of a call never loosens the rule that the line
  **is** a call: prose wrapped around a valid call is `malformed`, narration with no call is
  ignored.
- **Outcome categories, not one bucket.** `completed` / `handoff` / `step_cap` / `stuck` /
  `parse_failure_abort` / `timeout` are scored; `site_blocked` (CAPTCHA/WAF/interstitial, via
  `BLOCK_SIGNATURES`), `unreachable` and `harness_error` are environment failures, excluded from
  the success denominator and reported separately. The headline carries all three headline numbers.
- **Two abort rules, kept separate because they diagnose different things.** 3 consecutive steps
  with no parseable action → `parse_failure_abort` (the model is not speaking the action
  language); 5 consecutive steps that execute *zero* actions for any reason → `stuck` (it speaks
  the language but nothing lands). The parse rule fires first on a parse-failure run, which is the
  more specific diagnosis. Both are `--max-parse-failures` / `--max-unproductive-steps` and both
  are in the harness fingerprint. This replaces the driver's own loop guard, which eval disables.
- **Metrics split by `level`** (easy/medium/hard, as the source paper reports) and by domain;
  step efficiency is `steps / reference_length`, and the **step cap is derived per task**
  (`3x reference_length`, clamped to `[15, 50]`) because `reference_length` runs 1..36 — a flat
  cap truncates the hardest tasks. `--max-steps N` pins a flat cap; the effective cap is recorded
  per episode either way.
- **The judge is `agent_capture.verify_trajectory`** (the WebJudge analogue), reached through the
  `run_agent` chokepoint on the logged-in `claude` CLI — never `ANTHROPIC_API_KEY`. Screenshots per
  verdict are the dominant cost: `--judge-mid-screenshots` (default 2, plus the final) and the new
  `VERIFY_MID_SCREENSHOTS` env knob control it.
- **Comparability — our numbers are NOT rankable against published Online-Mind2Web figures.**
  Two deliberate differences. (1) *Denominator*: this harness excludes `site_blocked` /
  `unreachable` / `harness_error` episodes from the success denominator, because a CAPTCHA wall
  measures the network rather than the model. The Online-Mind2Web paper and the published results
  tables (e.g. Fara-1.5) do NOT do this — there an unreachable site is simply a failed task. Always
  report both denominators; `report.md` prints the all-episode rate next to the scored one for
  exactly this reason. (2) *Judge*: ours is the capture pipeline's `verify_trajectory` rubric on
  the latest Sonnet, not the paper's WebJudge/o4-mini, so absolute levels are not on the same
  scale. Both are fine for a **before/after delta measured through this same harness**, which is
  what the harness exists for; neither licenses putting our number in someone else's league table.
- **Eval episodes must never enter training.** They land under `eval_runs/`, outside the
  `trajectories/**` glob `convert` walks, and each carries `metadata.excluded_reason=eval_episode`.
- Driver support for a coordinate-only agent lives in `browser_driver.js` as the `move_xy`,
  `type_text` and `scroll_xy` `/act` kinds plus `POST /screenshot {format:"png"}`;
  `DRIVER_LOOP_GUARD_REPEATS=0` disables the capture-time loop guard so a stuck student is
  observed rather than refused.

## Training

`config.yaml` holds hyperparameters from the blueprint (Qwen3.5-9B-Base, LoRA r64/α128, lr 5e-5, 3 epochs, max_length 4096, `freeze_vit: true`, `max_pixels` exported as `MAX_PIXELS`). `train_qwen.py train` maps these keys to `swift sft` CLI flags; CLI overrides (`--epochs/--lr/--batch-size`) win over the config.

## Remote training (`remote_train/`, Vast.ai; provider-pluggable)

`make remote-train` runs the whole SFT on a rented cloud GPU instead of a hand-managed box: rent a GPU → ship `trajectories/` + the training code → run the **unchanged** `train_qwen.py convert` + `train_qwen.py train` on the box → pull the LoRA adapter back to `./output` → **destroy the instance**. No training logic is duplicated — the cloud layer only orchestrates.

- **Provider-pluggable by design.** `remote_train/base.py` defines a `ComputeProvider` interface and a `PROVIDERS` registry of factory functions (same idiom as `eval_screenspot.py`'s `PREDICTOR_BACKENDS`). Vast.ai (`remote_train/vastai.py`, shells out to the `vastai` CLI) is impl #1; a second provider is a new module implementing 8 methods + one line appending its factory to `PROVIDERS` — `lifecycle.py`/CLI/Makefile never change. Offer selection (geo/price/VRAM policy) is the pure shared `base.select_cheapest_allowed`.
- **Teardown is the #1 invariant** (never waste credits): `lifecycle.run_remote_training` destroys in `try/finally`, on SIGINT/SIGTERM/SIGHUP, via a `destroy` that polls until the box is **confirmed gone**, via a crash-safe `.remote_train_state.json` + the `cleanup` orphan sweep (by instance label), and via an independent max-runtime watchdog thread. `make remote-train-cleanup` force-destroys leftovers.
- **Constraints baked in:** offers in `exclude_countries` (default `RU, CN`) are never rented; every instance is labeled `roman.maior` (also the orphan-sweep key). Both configurable in `remote_train.yaml` / env.
- **Config:** `remote_train.yaml` (defaults < env < CLI), secrets via env only — `VAST_API_KEY` (required), `HF_TOKEN` (optional). Optional post-train flags (default off): `--eval` (PR #12's `eval_screenspot.py` before/after), `--push-hf`, `--notify` (reuses `repo_pipeline/notifier.py`).
- **Verify:** `make test-remote` (no-cost mock-provider self-test whose central assertion is "destroy ran in every exit path: success, exception, SIGINT"); `make remote-train-dry` (real API, prints the cheapest geo-allowed offer, provisions nothing). The exact `vastai` CLI flags / JSON fields are documented assumptions isolated in `vastai.py` — validate with the dry run before a real run.

## RL rollout harness (`rl/`)

Policy-agnostic rollouts of the student over the same sandboxed driver (`POST /observe` + `/act kind=sequence`): one recorded step per policy turn, the observation PNG's bytes ARE the recorded step screenshot, and the observation text is built by `train_qwen.build_step_user_text` — SFT input == RL input byte-for-byte. `rl.actions.decode_step` is the inverse of `format_action_for_training`. Rollouts land under `rl_runs/<run_id>/iter_NNN/<task>/kKK/task_*/` as full trajectory dirs (never `trajectories/`; `metadata.excluded_reason=rl_rollout`), so `verify_trajectory` / `corpus_audit` work on them unchanged. Reward: a task's programmatic `success_check` (`rl/checks.py` registry, authored in `rl/rl_tasks.json`) → `verify_trajectory` fallback (`--verifier-mode always|fallback|never`), composed by `rl/rewards.py` with weights in `rl/reward_config.json`. Reward v2 (2026-08-27): `success` is the judge's criteria-consistent verdict (`agent_capture.verdict_from_parsed`; see `docs/decisions/judge-consistency-audit.md`), discounted ×0.6 on a `meandering` grade; step efficiency slopes to 0 at 2× `expected_actions`; the done() bonus is paid only on success; a false done() and hitting either cap (`step_cap` env-side, `max_turns` trainer-side) are penalised. Train rows are filtered by `train_max_expected_actions` so `max_turns` can cover a wandering sample — otherwise every sample in a group hits the cap alike and the advantage is zero. The driver's loop guard is a service knob (`--loop-guard-repeats`, default `RL_LOOP_GUARD_REPEATS=5`; the capture-time 3 ended two thirds of RL episodes, the student eval disables it with 0). The suite (`make rl-tasks`) is built from the site registries minus everything SFT trained on; splits: `train` (idempotent, non-first task per site), `eval_seen_site_unseen_task`, `eval_unseen_site`, `eval_mutating`. Commands: `make test-rl-unit`, `make test-rl` (Docker, no LLM), `make rl-serve` (env service on 127.0.0.1:8900 for the GPU trainer's `Env` plugin), `make rl-rollout`, `make rl-eval` (E3 report with `--baseline` deltas). Design: `docs/superpowers/specs/2026-08-26-rl-harness-design.md`.

GPU side (`rl/trainer/`, `rl_train.py`, `rl_config.yaml`, `requirements-rl.txt`): ms-swift GRPO in vLLM server mode with a browser `Env`/scheduler plugin that calls the env service over a reverse SSH tunnel (`remote_train/tunnel.py`), one training record per turn with the episode reward broadcast; the SFT LoRA is merged into `Qwen/Qwen3.5-9B` before RL so the KL anchor is the SFT policy. `remote_train` gained `GrpoJob` (tunnel `-R 8900`/`-L 8001`, before/after `rl.eval` via a `PHASE`/`PHASE_ACK` handshake, periodic adapter rsync) and `ServeJob` (`make remote-serve`) under the same teardown guarantees. `make test-grpo` is the no-GPU self-test (mock provider + `mock://` env). Validated on Modal (2026-08-27, `make modal-grpo-smoke`): the `requirements-rl.txt` column, the merge, the ms-swift 4.5.2 plugin paths, vLLM 0.23 + the SFT LoRA target set, and one GRPO step with per-turn records. Two hard-won settings: server mode needs **two GPUs** (the trainer↔vLLM NCCL weight-sync communicator cannot share a device — `compute.rollout_devices`/`trainer_devices`), and `response_prefix: ""` (ms-swift's default forces an empty `<think>` block at generation, so the student could never emit its trained `<think>…</think>\naction:` form and every reward was 0). `--mode fallback` (decoupled batch GRPO, spec §3.4) is not implemented; ms-swift warns the multi-record scheduler path is slated for removal in a future release.

Modal is the primary GRPO compute (same volumes and serve flow as SFT v1): `make rl-serve` + `make rl-expose` (ngrok ingress, `RL_ENV_TOKEN` guards the service) on the Mac, then `make modal-grpo RL_ENV_URL=https://…` runs `modal_grpo.py` (mirrors `modal_train.py`: `cua-outputs` adapter in, `rl_train.py run` with evals off, merged weights cached on `cua-hf-cache`, checkpoints committed every 10 min, final adapter at `<run>/adapter`). Before/after numbers come from the existing flow: `CUA_ADAPTER_NAME=<run>/adapter modal deploy modal_serve.py` + `make eval-online-mind2web`, so GRPO deltas are comparable to the SFT v1 table. `make modal-grpo-smoke` is the first paid rung (merge + one step against the in-container `mock://` env). The Vast lane (`make remote-grpo`) remains the SSH-tunnel alternative.
