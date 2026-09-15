# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A data-gathering, training and evaluation pipeline that distills frontier computer-use (web browsing) ability into Qwen 3.5-9B via trajectory-based SFT, then GRPO. Three trajectory **producers** write one **trajectory JSON format**, and a single converter/trainer (`cua/sft/legacy_train_qwen.py`) consumes all of them:

- **Path A — GPT-5.4 CUA distillation** (`cua/path_a/recorder.py`): drives GPT-5.4's `computer` tool through Mind2Web tasks in a Playwright browser. ~$0.55/trajectory. Contributes only a handful of trainable trajectories today.
- **Path B — QA repo funnel** (`cua/pipeline/orchestrator.py`, stages 1-7): finds bootable web-app repos on GitHub, boots them in Docker sandboxes, and has a Claude teacher pursue proposed user tasks through a coordinate-recording browser driver. Only a fraction of a percent of discovered repos yield trajectories, so this lane is a minority of the corpus.
- **Public-site lanes** (`cua/pipeline/public_capture.py` and siblings): point the SAME teacher/driver/verifier machinery at curated registries of real public sites (login-free, search, login-gated via burner accounts, human-provisioned). This is the bulk of the trainable corpus (over 90% of it, roughly a third of that behind an auth wall). See "Public-site capture lanes".

The deliberate data philosophy is **real third-party sites over self-hosted seeded clones** (`docs/decisions/real-sites-vs-controlled-backends.md`): the target is deployment transfer, not benchmark reproducibility. Do not re-propose controlled backends as the primary training environment.

The trajectory schema is the integration contract tying producers, converter, evals and RL together — see `cua/pipeline/schemas.py` (`Trajectory`, `TrajectoryStep`) before changing any producer or consumer.

## Trajectory format (the contract)

Every trajectory is a directory `task_*/` (Path A: `task_mind2web_NNNN/`; Path B: `repo_<owner>_<repo>/task_*/`; public lanes: `site_<name>/task_*/`) containing `trajectory.json`, `metadata.json`, optional `capture_meta.json` (raw verifier verdict), and `step_NNN_screenshot.png` files.

```json
{"task_id", "instruction", "start_url",
 "steps": [{"step", "screenshot", "thought", "actions": [...]}],
 "total_steps", "success"}
```

Action dicts are normalized across all producers: `{type:"click", x, y, button}`, `{type:"type", text}`, `{type:"keypress", keys:[...]}`, `{type:"scroll", x, y, scroll_x, scroll_y}`, `{type:"drag", path:[{x,y}...]}`, `{type:"move", x, y}` (hover), `{type:"navigate", url}`, `{type:"wait"}`, the guarded-lane controls `{type:"confirm"}` / `{type:"handoff"}` / `{type:"approval_event"}`, and the terminal `{type:"done"}`. Viewport is fixed at **1280x720, deviceScaleFactor=1** everywhere. Credentials only ever appear as `<secret:...>` placeholders (driver-side substitution).

`instruction` is a terse user GOAL ("sign me up for an account on X"), never a step-by-step procedure; `thought` text describes intent, never a play-by-play of the action. Resources (credentials, inbox) go in the teacher policy, not the instruction.

`cua/sft/legacy_train_qwen.py convert` emits **one SFT example per step** in the **sft3 observation format** (`PROMPT_FORMAT_VERSION`): a 640x360 thumbnail of the PREVIOUS screen plus the current 1280x720 screenshot, the instruction, and a windowed text history (`HISTORY_WINDOW=20`) whose entries carry harness-computed effect annotations (` [no change]` when the screen stayed pixel-identical, ` -> /path` when the URL path changed) → `<think>` + actions. Coordinates are emitted in `normalized_1000` (the base model's native convention; `trajectory.json` stays in pixels, conversion lives only in `cua.sft.legacy_train_qwen`'s builders). Convert also adds a **grounding slice** (`--grounding-share`, default 0.25 of agent examples): click steps whose `capture_meta` element has role+name+bbox become `GROUNDING_SYSTEM_PROMPT` examples (`Task: Click the "Save" button.` → the click line, no `<think>`, small targets oversampled). Whole-trajectory multi-turn examples blew the 4096 max_length; a two-image step is ~2,020 tokens (1,100 image-pad). `build_step_observation` / `build_step_messages` / `history_entries_for_step` and `format_action_for_training` in `cua/sft/legacy_train_qwen.py` define the prompt and action grammar; the student eval and the RL harness reuse them byte-for-byte, with `tests/fixtures/prompt_golden.json` as the byte-level contract. Gating: one shared function (`corpus_audit.trainability`) used by convert, `make audit-corpus` and every corpus pass — `metadata.excluded_reason` always skips; agent trajectories require `success: true` (the verifier's verdict); legacy replay trajectories train only with `--include-replay`. Verifier-graded meandering trajectories are handled by `--meandering-mode keep|subsample|exclude` (train lanes use `subsample`: head/tail steps + every Nth mid-wander step train, the rest stay history-only; `MEANDER_*` env knobs); `--exclude-waste-steps` drops flagged waste clicks. Every convert prints a corpus health report AND enforces `corpus_health.json` thresholds (min trainable count, done coverage, teacher_raw=0, meandering example share, per-group share, action-share floors, grounding share) — violations refuse the dataset; `python -m cua.pipeline.corpus_audit` additionally gates `min_outcome_relabeled_share` (the `make outcome-relabel` pass must have run) and `min_done_evidence_share`; `--skip-health-gate` bypasses for experiments only.

## Environment, auth and Python

- Run Path B / evals / RL from the repo root with the pipeline venv: `.venv-pipeline/bin/python -m cua.pipeline.<module>` (the venv lives at the main checkout root, not inside worktrees).
- **Claude calls authenticate through the logged-in `claude` CLI (device OAuth), never `ANTHROPIC_API_KEY`.** `sdk_runner.CLAUDE_CLI_PATH` pins the SDK to the system `claude` binary (the SDK's bundled CLI has wedged on spawn); override with `CLAUDE_CLI_PATH`. The `ANTHROPIC_API_KEY` mentions in `README.md` / `crontab.example` are legacy.
- Other env: `OPENAI_API_KEY` (Path A), `GITHUB_TOKEN` (stage 1 + the stage-2 `gh` probe), `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` (optional reports), `HF_READ_TOKEN`/`HF_TOKEN` (gated Online-Mind2Web dataset), `VAST_API_KEY` (Vast.ai lanes), `RL_ENV_TOKEN` (env-service guard), `EVAL_ENDPOINT`/`EVAL_MODEL`/`EVAL_API_KEY`/`EVAL_ADAPTER` (make-level defaults for a served model), `SCREENSPOT_ENDPOINT`/`SCREENSPOT_API_KEY` (ScreenSpot `openai` backend).
- `credentials/` (burner registry, known accounts, operator PII file), `pipeline_data/`, `trajectories/`, `eval_runs/`, `rl_runs/`, `models/`, `output/` and `cua/pipeline/sites/public_sites_human.json` are gitignored. Local thesis notes under `docs/` are excluded via `.git/info/exclude`, never `.gitignore`. Feature work happens in a git worktree on its own branch + PR.
- Sandbox resource caps (per Docker container, `cua/pipeline/sandbox.py`): `SANDBOX_MEM_LIMIT` (`2g`), `SANDBOX_MEMSWAP_LIMIT` (= mem ⇒ no swap), `SANDBOX_PIDS_LIMIT`, `SANDBOX_CPUS`, `SANDBOX_TMP_SIZE` (RAM-backed `/tmp`), `SANDBOX_SHM_SIZE`, `SANDBOX_VM_RESERVE`. `safe_parallel_sandboxes` clamps parallelism so `workers × (mem + tmpfs)` stays under the Docker VM memory — exceeding it turns into a silent fast-then-1/hour collapse, not an error. The host itself is memory-bound (`docs/open-items.md` §1).

## Common commands

```bash
# Setup
make setup                 # Path A deps + playwright chromium
make setup-pipeline        # Path B deps + chromium
make build-sandbox         # Docker sandbox image (required for stages 4-6, capture lanes, evals, RL)
make build-sandbox-selenium  # amd64 Selenium image, replay mode only

# Path A
python -m cua.path_a.load_mind2web_tasks --output tasks.jsonl [--limit N]
make record                # needs OPENAI_API_KEY; --resume is on by default

# Path B funnel (registry-driven, see below)
make pipeline-discover     # stage 1, needs GITHUB_TOKEN
make pipeline-filter       # stage 2 (metadata + no-LLM bootable gate)
make pipeline-score PIPELINE_LIMIT=50 PIPELINE_BUDGET=5   # stage 3, costs money
make pipeline-overnight PIPELINE_BUDGET=30                # stages 3-7 once + Telegram report
make pipeline-until-limit PIPELINE_LIMIT_PCT=75           # loop stages 3-7 until 75% of Claude usage limit
make pipeline-think        # stage 7 only (run first on a fresh usage window; it starves otherwise)
make pipeline-status
python -m cua.pipeline.orchestrator --stage 3-7 --limit 10 --budget 5 --workers 5
python -m cua.pipeline.orchestrator --stage 6 --capture-mode replay   # legacy test-replay capture
python -m cua.pipeline.orchestrator --requeue-infra|--requeue-stale|--requeue-stage 4 --min-score 4|--prune-unfit [--dry-run]

# Public-site capture lanes (bulk of the corpus)
make capture-public [PUBLIC_SITES=a,b] [PUBLIC_MAX_TASKS=N]   # public_sites.json, login-free
make capture-search [SITES=a,b]                               # public_sites_search.json, search goals
python -m cua.pipeline.capture_logingated [--sites a,b] [--provisioning open_signup,email_verify] [--parallel K]
make provision-human SITE=<name>                              # headed host browser, human clears the wall once
make capture-human [SITES=a,b]                                # replay saved sessions from public_sites_human.json
python -m cua.pipeline.probe_capturable [--dry-run]          # HTTP WAF pre-filter → needs_human
python -m cua.pipeline.probe_sandbox <config.json> [--mark]  # render-probe through the real sandbox browser

# Corpus passes (dry-run by default, APPLY=1 to write, all resumable)
make audit-corpus          # mechanical gate over trainable trajectories, exit 1 on defects
make polish-pass           # stage-7 thought polish for teacher_raw dirs outside the registry (site_* lanes)
make relabel-recoverable   # audited hindsight relabel, fresh verifier verdict gates every write
make wall-demos            # trim blocked runs into terminal handoff/done wall supervision
make instruction-style     # rewrite procedure-flavored instructions into goals
python -m cua.pipeline.verify_pattern_scan trajectories/     # how much of the corpus demonstrates act→save→re-check
python -m cua.pipeline.regenerate_supervision --dry-run --sample 5 trajectories/
make dashboard             # read-only Streamlit corpus explorer, localhost:8501

# Convert + train
make convert               # trajectories/ -> training_data/sft_dataset.jsonl (waste steps dropped, meandering subsampled, health-gated)
make train                 # `swift sft` on a GPU box; CONFIG=config_sft_v2_conservative.yaml picks a variant
modal run cua/cloud/modal/serve.py::warm && modal run cua/cloud/modal/train.py --run-name sft-v2 --config config.yaml   # Modal SFT lane
make remote-train          # Vast.ai SFT lane (needs VAST_API_KEY), ALWAYS destroys the instance

# Serve + student eval (the deliverable is a before/after delta)
modal deploy cua/cloud/modal/serve.py                                  # base as `cua-student`; CUA_ADAPTER_NAME=<run>/adapter adds `cua-student-lora`
make eval-sources
make eval-online-mind2web EVAL_ENDPOINT=<url>/v1 EVAL_MODEL=cua-student EVAL_LABEL=before EVAL_COORD_SPACE=normalized_1000   # base and sft-v3+
make eval-online-mind2web EVAL_ENDPOINT=<url>/v1 EVAL_MODEL=cua-student-lora EVAL_LABEL=after EVAL_COORD_SPACE=viewport_abs \  # sft-v1/v2 only
     EVAL_ARGS="--tasks-file eval_archive/online_mind2web/tasks_pinned_v2.jsonl --parallel 3"
make eval-held-out EVAL_ENDPOINT=... EVAL_LABEL=before
make eval-compare EVAL_RUNS="eval_runs/before_... eval_runs/after_..."
make eval-screenspot       # local `swift infer` lane over eval_models.yaml; served lane: --backend openai + eval_models_served.yaml
make remote-eval EVAL_MODEL=Qwen/Qwen3.5-9B EVAL_LIMIT=100   # benchmark on a rented GPU, no training

# RL (GRPO)
make rl-tasks && make rl-serve && make rl-expose             # env service on the Mac + ngrok ingress
make modal-grpo-smoke && make modal-grpo RL_ENV_URL=https://...   # Modal GRPO (primary); make remote-grpo is the Vast lane
make rl-rollout / make rl-eval RL_BASELINE=report.json

# Experiment registry (every measured number lives here)
make registry              # fill Wilson CIs + re-render the README block from experiments/*.csv
make registry-check        # fails if the README block is stale
```

**Executable checks** (there is no pytest suite): `make test-sandbox` (container lifecycle), `make test-driver` (scripted no-LLM browser-driver session asserting the trajectory contract), `make test-verify-gate` (capture success gate never rejects a verified trajectory), `make test-eval` (ScreenSpot harness smoke), `make test-student-eval` (parser round-trip + mock episode + judge + browser-backend surface), `make test-rl-unit` (`python -m tests [filter]`, dependency-free runner), `make test-rl` (Docker-only RL smoke), `make test-remote` / `make test-grpo` (mock-provider lifecycle: destroy runs on every exit path), `make test-notifier`, `python -m cua.pipeline.auth_flow_smoke` (mail stack + masked-secret wiring, no LLM). Verify pipeline changes by running the relevant stage with a small `--limit`/`--budget`.

## Conventions the code enforces

- **Registries over special cases.** New behavior is a data entry in an existing registry, not a new branch: `frameworks.PROFILES`, `eval_task_sources.TASK_SOURCES`, `student_eval.BROWSER_BACKENDS` / `STUDENT_BACKENDS`, `cua.eval.screenspot.PREDICTOR_BACKENDS` / `COORD_SPACES`, `cua.cloud.vast.PROVIDERS` / `JOBS` / `BENCHMARKS`, `cua.rl.checks` / `cua.rl.policies` / `cua.rl.backends`, `safety_watchdog.ApprovalPolicy` (`WATCHDOG_POLICY`), `bootable_gate` signal sets, `schemas.STAGE_TO_*`. Hardcoded name checks are a smell.
- **Every measured number goes into `experiments/models.csv` + `experiments/results.csv`**, then `make registry` renders the README block — never edit that block by hand. `status` carries meaning: `done` is measured, `pending` = run finished but metric not computed, `planned` = worklist, `superseded` = must not be quoted (keep the row, say why in `notes`). `experiments/sources.csv` holds figures cited from other people's work.
- **Raw eval artefacts are committed under `eval_archive/`** (per-sample ScreenSpot records, Online-Mind2Web `results.json`/`report.md`/`run.log`, `manifest.json` with sha256s, and the pinned task files) because `eval_runs/`/`eval_results/` are gitignored. Per-episode screenshots (~166 MB) are packed under gitignored `models/` next to the adapter.
- **No regex post-filters on LLM output.** Fix the prompt instead of discard/retry-on-pattern; observability metrics are fine, content filters are not. Generator outputs are trusted as written.
- **Burner accounts only.** Login-gated capture never uses the operator's real accounts: disposable identity (`persona.py`), a readable throwaway inbox (`mail_service.py` for sandboxed apps, `mailtm.py` for real sites), an isolated browser context, a fail-closed registry under `credentials/` (`app_secrets.py`), and `operator_pii.py` so the operator's identity is never typed or shipped. Account creation and login ARE in-scope skills; the isolation is what makes them allowed.
- **Eval and RL episodes never enter training.** They land under `eval_runs/` / `rl_runs/` (outside the `trajectories/**` glob) with `metadata.excluded_reason=eval_episode` / `rl_rollout`. `saucedemo` is both a benchmark subject and in the SFT corpus, so never report numbers over it as unseen.
- **Never `nohup` a long run.** Long jobs run harness-tracked in the background under `caffeinate` so they can be stopped and survive sleep.

## Path B architecture (`cua/pipeline/`)

A 7-stage quality funnel, cast wide then reject aggressively. `orchestrator.py` is the entry point; it sets up logging to `pipeline_data/run_logs/`, then dispatches stages. Capture (6) and supervision finalize (7) are split so a slow/flaky LLM pass can never block or fail a capture.

**Capture modes.** `schemas.CAPTURE_MODE` (env `PIPELINE_CAPTURE_MODE` / CLI `--capture-mode`, default **`agent`**) switches how stages 4-6 behave:
- **agent** (primary): the repo's app is booted inside the sandbox and a Claude teacher (`agent_capture.py`) pursues user tasks proposed by `task_proposer.py`, acting through an in-container Playwright HTTP driver (`instrumentation/browser_driver.js` + host wrapper `driver_session.py`) that records a screenshot + bbox-center coordinates per action. Every action tool requires a `thought` argument — per-step intent-level supervision is structural, not retrofitted. A post-run verifier (`verify_trajectory`, `verdict_from_parsed`) judges final + sampled mid screenshots against the goal; only verified trajectories get `success: true`, judged ONLY against the original instruction — salvaging coherent partials is the separate audited `relabel_recoverable` pass, never the capture-time verifier. Framework-independent: the repo's own tests are only feature hints for the proposer.
- **replay** (secondary/legacy): inject instrumentation into the repo's Playwright/Selenium suite and record what the tests do; `think_generator.py` then GENERATES thoughts and instructions from screenshots + test source.

**Registry state machine is the spine.** All repo state lives in `pipeline_data/registry.db` (SQLite, WAL). `schemas.py` defines three dicts — `STAGE_TO_INPUT_STATUS`, `STAGE_TO_PASSED_STATUS`, `STAGE_TO_REJECTED_STATUS` — that map each stage to the `RepoStatus` it consumes / promotes to / rejects to:

```
discovered → filtered → scored → analyzed → validated → captured → completed
                ↓          ↓         ↓          ↓           ↓
          rejected_filter / _score / _analyze / _trial / _generate
                                                  (stage 7 "think": captured → completed, never rejects)
```

Each stage calls `registry.get_repos_for_stage(N)` (ordered by a query-priority heuristic that deprioritizes demo/tutorial repos). Per-repo work runs through `_processing.py`: the stage's core function returns normally (passed), raises `QualityReject` (terminal, never retried) or `RetryableFailure`/any exception (infra; retried `PIPELINE_MAX_RETRIES` times, then rejected with an infra `FailureCategory` so it stays re-queueable). Stages are **idempotent**: repos guard on `current_stage >= N` and re-running never reprocesses. Re-queue flags: `--requeue-infra` (infra/flaky rejections), `--requeue-stale` (rejections from code superseded per `schemas.STAGE_BEHAVIOR_CUTOFFS`), `--requeue-stage N --min-score X` (after lowering a threshold), `--prune-unfit` (re-judge past stage 4, reject toys/boilerplates; reversible), all honoring `--dry-run`. When adding a stage or status, extend the mapping dicts rather than hardcoding stage numbers.

**Stage responsibilities:**
- **1 discover** (`stage1_discover.py`): GitHub API across ~50+ `SEARCH_QUERIES`, down to `--star-floor`. No LLM.
- **2 filter** (`stage2_filter.py`): size/recency filters, fork & duplicate-description dedup, then the **no-LLM bootable-app gate** (`bootable_gate.py`: one `gh` probe of root tree + package.json, data-driven framework sets; hard rejects → `NOT_BOOTABLE`, ambiguous repos pass unless `BOOTABLE_GATE_AGGRESSIVE=1`). `cua/pipeline/tools/regate_backlog.py` re-gates the existing backlog; `cua/pipeline/tools/validate_gate.py` measures its precision/recall against completed repos.
- **3 shallow_score**: fetches repo metadata via `gh`, Claude scores 7 dimensions (`ShallowScores`), gate at `SCORE_THRESHOLD`.
- **4 deep_analyze**: static analysis in a `DockerSandbox`. Bootability is the dominant score axis; in agent mode a `good_fit` verdict plus est-trajectory / production-complexity backstops hard-veto toys, and framework support is irrelevant (replay mode still gates on `is_supported_profile`).
- **5 trial_run**: agent mode = **boot gate** (`boot_gate_single_repo`): clone, install (pnpm/npm), daemonize the app (`driver_session.start_app`, with `app_secrets.boot_env_secrets()` + mail env injected), probe it through the driver, reject `NO_BOOT` / trivial UIs (`BOOT_GATE_MIN_ELEMENTS`). Replay mode = legacy instrumented POC test run via `app_runner.wrap_test_command`.
- **6 capture** (`stage6_generate.py`): boots the app, proposes tasks, runs the teacher per task (fresh browser context each; one shared app server), verifies + dedups + caps per repo (`AGENT_MAX_KEPT_PER_REPO`, default 10), enriches metadata, marks **`captured`**. Resume-aware; partial sessions are copied out even on usage-limit aborts.
- **7 finalize** (`stage7_think.py`): agent trajectories get their raw teacher thoughts POLISHED into student voice (one text-only call per trajectory, `thought_source: teacher_raw → teacher`); replay trajectories get thoughts GENERATED. Then marks `completed`. Decoupled, resumable, **never rejects**. It runs LAST in a drain loop and is starved by the usage limit — run `make pipeline-think` on a fresh window to unstick captures.

**Sandbox isolation** (`sandbox.py`): repo code executes only inside an ephemeral, capability-dropped Docker container (`repo-pipeline-sandbox:latest`, built from `docker/Dockerfile.sandbox` on the Playwright base image, driver pre-baked at `/opt/driver`). `DockerSandbox` is a context manager exposing `exec_command/read_file/write_file/copy_out/start_background/wait_for_port/http_json`; it enforces a 30-min container and 5-min per-command timeout. `ShellSandbox` is the transport-agnostic base shared with `remote_sandbox.RemoteSandbox` (ssh/rsync to a rented box) so the driver behaves identically in a container and over ssh. If a run yields 0 with "No such image", check `docker images` and rebuild before trusting the number.

**Instrumentation** (`instrumentation/`): `browser_driver.js` is the live agent-mode recorder (JSON-over-HTTP `/observe`, `/act`, `/screenshot`; kinds include bbox-ref actions plus coordinate-only `move_xy` / `type_text` / `scroll_xy` for students; `DRIVER_LOOP_GUARD_REPEATS` is its stuck-loop guard). Replay mode writes trajectories from the three injected StepRecorders — `pw_python_conftest.py` (pytest-playwright), `pw_ts_fixture.ts` (@playwright/test), `selenium_python_conftest.py` — each writing `trajectory.json` + `capture_meta.json` + `metadata.json`. `capture.py`'s `ActionRecorder` is not imported anywhere (dead code).

## Public-site capture lanes

All lanes reuse `agent_capture.run_teacher_task` + `verify_trajectory` and write `trajectories/site_<name>/task_*/` with no registry row, so **stage 7 never visits them — `make polish-pass` is their finalize**. Capture attempts are recorded per task (`record_capture_attempt` / `is_capture_settled`, `CAPTURE_MAX_ATTEMPTS`) so re-runs resume. Site registries (`cua/pipeline/sites/public_sites*.json`, same schema: `name`, `start_url`, `tasks[{id, instruction, difficulty, expected_actions, action_kinds, auth_required}]`, optional `provisioning`, `task_style`, `skip_capture`):

| file | role |
| --- | --- |
| `public_sites.json` | original curated login-free set (also the `held_out` eval source's site pool) |
| `public_sites_actions_loginfree.json` / `_actions.json` / `_actions_indie.json` | action-UI sites (do/create/transact, never static browsing) |
| `public_sites_actions_logingated.json` | ~320 mainstream login-gated SaaS sites, `provisioning` hint per site |
| `public_sites_search.json` | search-goal lane (`task_style: search`); sites must never appear in Online-Mind2Web's 300 |
| `public_sites_guarded.json` | guarded-mode (confirm/handoff) tasks; `.example.json` is the template |
| `public_sites_human.json` | generated by `provision_human.py`, gitignored, `auth_mode=pre_session` |
| `public_sites_saucedemo.json` | login smoke target, leaks into both bench and corpus |

- **Login-gated, autonomous** (`capture_logingated.py`): mints a mail.tm burner, runs the first task in `discovered` auth mode (teacher registers/signs in, reads the verification mail in the browser), then reuses the captured `storage_state` via `pre_session` for the rest. Session + burner persist through `app_secrets`, so a site is registered once. A blocked signup abandons the site after one attempt and marks it `needs_human` in `pipeline_data/provisioning_status.json` (`provisioning_status.py`); success marks `registered`. Only works on sites without captcha / phone / disposable-domain blocks.
- **Human-provisioned** (`provision_human.py` → `make capture-human`): a headed HOST browser where a person clears the signup wall once; the saved session works from the sandbox because Docker egresses via the host IP. `--serve` exposes a localhost JSON control API for an assistant-driven loop.
- **Pre-filters:** `probe_capturable.py` (host HTTP, explicit WAF signatures only) and `probe_sandbox.py` (real sandbox browser, affordance counts, `--mark` writes `skip_capture`). Run probes when capture is not running.
- **Guarded mode / watchdog** (`safety_watchdog.py`): at a `confirm` step the teacher proposes instead of executes; an `ApprovalPolicy` (LLM default, static-rule and human drop-ins via `WATCHDOG_POLICY`) returns approve/deny, recorded as an observation step. `FINANCIAL_RISK` categories are never auto-approved on live sites.
- **Mail stack** (`mail_service.py`): one shared GreenMail (SMTP sink + IMAP, auto-creates mailboxes) + Roundcube stack on Docker network `vetty-mail`; sandboxed apps SMTP to `greenmail:3025`, the in-container browser reads `http://mail.test/`. Per-scenario addresses `qa-<run>-<scenario>@mail.test`; GreenMail strips `+tag` addressing, so identity must live in the base local-part.
- Known gotchas: session injection works for cookie/localStorage auth but IndexedDB-auth SPAs need Playwright ≥1.51 with `storageState({indexedDB:true})` (the pinned driver Playwright is 1.55); `cf_clearance` does not transfer into the container; cloud-email apps (Brevo etc.) cannot deliver to GreenMail, so use `known_account` entries instead.

Design docs: `docs/superpowers/specs/2026-06-28-login-gated-trajectories-design.md`, `docs/open-items.md` (living list of what is blocked and why).

## Claude Agent SDK usage

Path B calls Claude via `claude_agent_sdk.query()` returning `AsyncIterator[Message]`. There are exactly **two chokepoints**, both wrapped by `SDK_QUERY_TIMEOUT` (default 2400 s) so a wedged CLI subprocess cannot hang a run: `sdk_runner.run_query` (raw `query()` loop, rate-limit tracking, `harden_options` strips operator plugins from headless calls) and `agent_runner.run_agent` (adds one `agent_calls` ledger row per call with turns/cost/tokens, classifies non-success results as `RetryableFailure` when `raise_on_error`). Standard extraction: take `message.result` from a `ResultMessage` where `message.subtype == "success"`. Options are `ClaudeAgentOptions(model, max_turns, max_budget_usd, permission_mode="bypassPermissions", allowed_tools, cwd, mcp_servers)`; tiny `max_budget_usd` caps fail before the first tool call, so budgets have a floor. Models are configured centrally in `cua/pipeline/llm_config.py`: `LATEST_MODEL_BY_CLASS` maps each class to its latest ID (currently `claude-sonnet-5` for every purpose) and call sites import a named purpose (`MODEL_FOR_REASONING`, `MODEL_FOR_GROUNDED_GENERATION`, `MODEL_FOR_TEACHER_AGENT`, `MODEL_FOR_CLASSIFICATION`, `MODEL_FOR_SAFETY_JUDGE`), never a model string.

A "hung" LLM stage is usually one of two things: a saturated 5-hour usage window (every CLI call waits ~15 min at 0% CPU — time `claude -p pong` before debugging code) or Docker VM memory exhaustion (see sandbox caps above).

## Budget & concurrency

LLM stages run with an `asyncio.Semaphore(workers)`; the orchestrator caps workers per stage (3 for analyze, 2 for trial/generate) since those hold a container each. Cost control is two-layered: `orchestrator.STAGE_COST_ESTIMATES` stops launching new stages once the run budget is spent, and each stage checks its own `cost_estimate` against `budget_remaining` **inside** the semaphore (a gate before the semaphore races past on single-stage runs). `--budget` is a soft cap based on per-repo estimates, not metered spend; the `agent_calls` ledger is the metered record.

`--overnight` and `--until-limit` both run through `orchestrator.drain_funnel`: overnight is one pass, until-limit loops until the usage threshold / budget / drained funnel. `run_query` feeds the SDK's `RateLimitEvent` into a process-global `rate_limit_tracker` and raises `UsageLimitReached` on a hard 429. That exception is re-raised (never caught-and-rejected) at every layer, so hitting the usage limit leaves repos in their input status for the next run. `--until-limit PCT` sets `rate_limit_tracker.threshold = PCT/100`; with no usage events observed it degrades to "stop on hard 429".

## Evaluation: ScreenSpot grounding (`cua/eval/screenspot.py`)

Predicted click point vs. target bbox on `rootsautomation/ScreenSpot`. **Imports `SYSTEM_PROMPT` from `cua/sft/legacy_train_qwen.py`** to guarantee train/eval prompt parity (so `cua/sft/legacy_train_qwen.py` and the `cua.pipeline.corpus_audit` it imports must be present wherever the eval runs). `PREDICTOR_BACKENDS`: `swift` (shells out to `swift infer`, specs in `eval_models.yaml` with a repo id + `adapters` dir), `openai` (any OpenAI-compatible endpoint via `SCREENSPOT_ENDPOINT`, specs in `eval_models_served.yaml` naming vLLM *served* models), `mock` (tests). Heavy deps load lazily so `make test-eval` needs none.

**The scoring conventions are the whole ballgame — get one wrong and the eval reports a confident WRONG number instead of crashing.**
- **`coord_space` is PER-MODEL** (`ModelSpec.coord_space`). A stock Qwen and every sft-v3+ checkpoint answer in `normalized_1000`; sft-v1/v2 checkpoints answer in `viewport_abs` (1280x720). The *same* base-model predictions score **85% as `normalized_1000` and 25% as `viewport_abs`** — a before/after scored with one global coord space is meaningless.
- **`--bbox-normalized` (default on)** — the default dataset stores bbox in 0-1. Treating it as pixels yields **0% by construction**.
- **`--bbox-format xyxy` (default)** — that mirror stores `(x1,y1,x2,y2)`.
- The default dataset is `rootsautomation/ScreenSpot`, **not** `os-copilot/ScreenSpot-v2` (raw JSON + image zip; `load_dataset` cannot read it). Re-check all three conventions against any other mirror.

Grounding and task success are complementary and diverge: SFT v1 lost ~4 pp of ScreenSpot while gaining task success. Current numbers live in the experiment registry, not here.

## Student eval (`cua/eval/student.py`)

Runs a served model through real browser episodes and reports whether a checkpoint beat the base. **The deliverable is a delta**: every run takes `--label` (`before`/`after`), writes `eval_runs/<run_id>/{report.md,results.json,run.log,episodes/}`, and `compare` renders the before/after table plus a **harness parity check** (`harness_fingerprint`) that flags any field the two runs disagree on.

- **Prompt parity.** The per-step messages are built by the SAME functions convert uses (`cua.sft.legacy_train_qwen.build_step_observation` / `build_step_messages` / `history_entries_for_step`, same `SYSTEM_PROMPT`, same `HISTORY_WINDOW`, same effect annotations computed from consecutive PNG screenshots — JPEG is refused); `tests/fixtures/prompt_golden.json` is the byte-level contract every harness asserts against, and `prompt_format` is a fingerprint key. An eval that prompts differently measures the prompt.
- **Model remote, browser wherever egress needs it.** `--endpoint` is any OpenAI-compatible `/chat/completions` (Modal `cua/cloud/modal/serve.py`, vLLM, `swift deploy`). The browser is a `BROWSER_BACKENDS` entry: `docker` (local `DockerSandbox`, residential UA IP), `remote_ssh` (`--browser-host`), `vast_residential` (rents a Vast box whose public IP belongs to a consumer ISP; destroys on stop). Online-Mind2Web is US-centric: only 58% of its 300 tasks are reachable from this machine, a datacenter VPN trades geo-blocks for WAF-blocks 1:1, and only a residential US exit recovers a real share. **Egress is a parity key carrying the resolved ASN** — runs on different egress measure the network, not the model.
- **Pinned task files** (`--tasks-file`): `eval_archive/online_mind2web/tasks_pinned_50.jsonl` (the legacy 50, level-stratified), `tasks_pinned_v2.jsonl` (230 = legacy 50 + 180 stratified, sites hard-blocked from this IP excluded from the new 180; target ≥150 scored per side), `tasks_reachable_us_174.jsonl` (the US-reachable set). n≈34 cannot resolve any plausible delta (±8 pp).
- **Task sources are a registry** (`eval_task_sources.TASK_SOURCES`): `online_mind2web` (gated HF dataset, 300 tasks, columns `task_id`/`confirmed_task`/`website`/`reference_length`/`level`) and `held_out` (our own never-captured public-site tasks, proven unseen by an on-disk check against the corpus). Token resolution: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, then `HF_READ_TOKEN` from `.env`, then `~/.cache/huggingface/token`; a 401/403 fails loudly naming the URL to accept terms at, never a stub.
- **`--coord-space` is PER MODEL and never auto-detected** (default `normalized_1000`, which a stock Qwen and sft-v3+ answer in; sft-v1/v2 checkpoints need `viewport_abs`). Emitted positions are scaled to viewport pixels before dispatch (scroll deltas are not); `trajectory.json` records EXECUTED pixels, `raw_last_response` keeps the model text. Getting this wrong silently produces a confident wrong number, so the harness warns at the first impossible coordinate and prints a `COORDINATE SPACE SUSPECT` banner when >25% of coordinates fall outside the declared space (the viewport under `viewport_abs`, 0-1000 under `normalized_1000`). In `compare`, a `coord_space` difference is an expected, annotated model-side difference, not a parity violation.
- **The action grammar is tolerant by design, and canonical form is its own metric.** The parser accepts any **unambiguous** spelling and reports `canonical_form_rate` plus an `action_form_deviations` breakdown instead of charging deviation to success. Tolerance is a parity key (`action_grammar: tolerant|canonical`, `--strict-action-grammar`). Accepted (canonical first):

  | verb | accepted spellings |
  | --- | --- |
  | `click` | `click(<ref>(x, y)</ref>, button=left)` · `(x, y)` · `x, y` · trailing `left`/`right` · button omitted |
  | `double_click`, `move` | `<ref>(x, y)</ref>` · `(x, y)` · `x, y` |
  | `drag` | both endpoints in any coordinate spelling, separated by `->` or `,` |
  | `scroll` | `scrollX=`/`scrollY=`, `scroll_x=`/`scroll_y=`, `dx=`/`dy=`, positional deltas, or omitted (defaults `0, 500`) |
  | `type` | `type("text")` · `type('text')` |
  | `keypress` | `["Enter"]` · `"Enter"` · `Enter` · `Control, a` · `Control+a` |
  | `wait`, `done`, `screenshot` | `verb()` · `verb` |
  | `navigate` | `navigate("url")` · `navigate(url="…")` · bare `https?://` token; executes via the driver's `navigate` kind |
  | `handoff`, `confirm` | `handoff(reason=login)` · `confirm(propose=call(...), risk=x)`; terminal — scored outcome `handoff` |
  | everywhere | optional `action:` prefix, bullets/backticks stripped, unknown kwargs ignored, floats rounded |

  Rejected as genuinely ambiguous: `type(hello world)`, `type(a, b)`, `scroll(500)`, `click(100)`, `drag((10, 20))`, `keypress()`, `navigate()`, scheme-less `navigate(example.com)`. Prose wrapped around a valid call is `malformed`; narration with no call is ignored. The verb set is part of the fingerprint — pre-`navigate` runs compare as a different harness.
- **Outcome categories, not one bucket.** `completed` / `handoff` / `step_cap` / `stuck` / `parse_failure_abort` / `timeout` are scored; `site_blocked` (via `BLOCK_SIGNATURES`: `captcha` / `human_verification` / `cloudflare` / `waf_block` / `geo_block` / `site_error`), `unreachable` and `harness_error` are environment failures excluded from the success denominator and reported separately. A signature the set misses is scored as a MODEL failure, so the set is a parity key (`BLOCK_SIGNATURES_VERSION`) and every episode records `final_url` / `final_snapshot`.
- **Two abort rules**: 3 consecutive steps with no parseable action → `parse_failure_abort`; 5 consecutive steps executing zero actions → `stuck` (`--max-parse-failures` / `--max-unproductive-steps`, both in the fingerprint). The driver's loop guard is disabled (`--loop-guard-repeats 0`) so a stuck student is observed, not refused.
- **Metrics split by `level`** and domain; step efficiency is `steps / reference_length`; the step cap is per task (`3x reference_length`, clamped to `[15, 50]`; `--max-steps N` pins a flat cap).
- **The judge is `agent_capture.verify_trajectory`** (criteria-first, latest Sonnet, via `run_agent` on the logged-in CLI). `--judge-mid-screenshots` (default 2) / `VERIFY_MID_SCREENSHOTS` control cost. The judge is lenient on `meandering` episodes, which only fine-tuned models produce (`docs/decisions/judge-consistency-audit.md`): **quote audited numbers**, and read final screenshots for claimed successes.
- **Not rankable against published Online-Mind2Web figures**: we exclude blocked/unreachable from the denominator (the paper does not) and use a different judge. Always report both denominators; `report.md` prints both.

## Training

`config.yaml` holds the SFT recipe (Qwen3.5-9B, LoRA r64/α128, lr 5e-5, 3 epochs, max_length 4096, batch 2 × accum 4, `freeze_vit: true`, `packing: false` because the images lack flash-attention, `max_pixels` exported as `MAX_PIXELS`). Variants are configs, not forks: `config_sft_v2_conservative.yaml` (r16, 1 epoch, lr 1e-5). `cua/sft/legacy_train_qwen.py train` maps config keys to `swift sft` flags (ms-swift 4.x, `tuner_type`); CLI overrides (`--epochs/--lr/--batch-size`) win over the config.

**Serving a fine-tuned adapter needs `--response_prefix ''`** (ms-swift's default prefills an empty `<think>` block so no action line is ever emitted), and the base must be passed explicitly because `adapter_config.json` records a stale volume path. A GRPO adapter is trained over the MERGED SFT weights and must be served on those (`CUA_SERVE_MODEL`), never on the raw base.

**Modal lane (primary for SFT and GRPO).** `cua/cloud/modal/train.py` runs the unchanged `cua/sft/legacy_train_qwen.py convert` + `train` on an A100-80GB with volumes `cua-trajectories` (input), `cua-outputs` (adapters at `<run_name>/adapter`, per-epoch checkpoints committed every 10 min), `cua-hf-cache`; `--config` picks any `config*.yaml`, `--train-args` passes extras. `cua/cloud/modal/serve.py` serves the base as `cua-student` on an L40S via vLLM and, with `CUA_ADAPTER_NAME=<run>/adapter`, the LoRA as `cua-student-lora` from the same container; `modal run cua/cloud/modal/serve.py::warm` populates the weight cache on CPU first. Lanes are parameterised by env (`CUA_SERVE_APP`, `CUA_SERVE_MODEL`, `CUA_SERVE_GPU`, `CUA_TRAIN_GPU`, `CUA_RUN_NAME`), never forked. Cost checks must include volumes/caches on every workspace, not just running apps.

## Remote training (`cua/cloud/vast/`, Vast.ai; provider-pluggable)

`python -m cua.cloud.vast <job>` rents a GPU, ships code/data, runs the job, pulls results, and **destroys the instance**. No training logic is duplicated — the cloud layer only orchestrates.

- **Three registries.** `base.PROVIDERS` (Vast.ai in `vastai.py` shells out to the `vastai` CLI; `mock.py` for tests), `jobs.JOBS` (`train` = unchanged convert+train; `eval` = a `benchmarks.BENCHMARKS` entry, no training; `grpo` = `cua/rl/train.py` against the Mac env service over a tunnel; `serve` = student on 127.0.0.1:8001 through a tunnel), `benchmarks.BENCHMARKS` (files to ship, pip deps, command, result path, summarizer). Offer selection is the pure `base.select_cheapest_allowed` (geo/price/VRAM/compute-cap policy). `--provider` / `--benchmark` choices come from the registries.
- **Teardown is the #1 invariant**: `lifecycle.run_remote_training` destroys in `try/finally`, on SIGINT/SIGTERM/SIGHUP, via a `destroy` that polls until the box is confirmed gone, via a crash-safe `.remote_train_state.json` + the `cleanup` orphan sweep (by instance label), and via an independent max-runtime watchdog thread. A contract a failed create still allocated is destroyed too. **Teardown never outranks ownership**: the account is shared with other researchers AND with concurrent agent sessions, so `_owned_by_us` refuses any instance the provider reports under a foreign label, records carry the owning pid so a live sibling run's box is never swept, and taking every labelled box needs an explicit `cleanup --all`. A record with `pid: null` means a supervisor released the box (`python -m cua.cloud.vast release --instance ID`); `cleanup` reclaims released or dead-owner boxes only. `make remote-train-cleanup` reclaims boxes whose owning run is dead. Never destroy by id from a bare `vastai show instances`.
- **Constraints baked in:** offers in `exclude_countries` (default `RU, CN`) are never rented; every instance is labeled `roman.maior` (also the orphan-sweep key). The Vast account is SHARED with colleagues: never touch an instance not carrying that label. Config: `remote_train.yaml` (defaults < env < CLI), secrets via env only (`VAST_API_KEY`, optional `HF_TOKEN`). `--resume-from` continues an interrupted SFT run.
- **Verify:** `make test-remote` (mock provider; asserts destroy ran on success, exception and SIGINT) and `make remote-train-dry` (real API, cheapest allowed offer, provisions nothing). The exact `vastai` flags / JSON fields are documented assumptions isolated in `vastai.py`.

## RL rollout harness (`cua/rl/`)

Policy-agnostic rollouts of the student over the same sandboxed driver (`POST /observe` + `/act kind=sequence`): one recorded step per policy turn, the observation PNG's bytes ARE the recorded step screenshot, and the observation text is built by `cua.sft.legacy_train_qwen.build_step_user_text`. **The RL lane is frozen at the one-image v2 observation** (RL was dropped 2026-09-02 after three GRPO runs, see `docs/decisions/grpo-v3-results.md`); it must not be run against an sft-v3 policy without the two-image plugin work (ms-swift `_add_default_tags` would silently prepend a missing `<image>` into the system prompt). `cua.rl.actions.decode_step` is the inverse of `format_action_for_training`. Rollouts land under `rl_runs/<run_id>/iter_NNN/<task>/kKK/task_*/` as full trajectory dirs (`metadata.excluded_reason=rl_rollout`), so `verify_trajectory` / `corpus_audit` work on them unchanged.

Reward (v2, `cua/rl/rewards.py` + `cua/rl/reward_config.json`): a task's programmatic `success_check` (`cua/rl/checks.py` registry, authored in `cua/rl/rl_tasks.json`) → `verify_trajectory` fallback (`--verifier-mode always|fallback|never`); `success` is the judge's criteria-consistent verdict discounted ×0.6 on `meandering`; step efficiency slopes to 0 at 2× `expected_actions`; the done() bonus is paid only on success; a false done() and hitting either cap (`step_cap` env-side, `max_turns` trainer-side) are penalised. Train rows are filtered by `train_max_expected_actions` so `max_turns` can cover a wandering sample — otherwise every sample in a group hits the cap alike and the advantage is zero. The driver's loop guard is a service knob (`--loop-guard-repeats`, default `RL_LOOP_GUARD_REPEATS=5`; the capture-time 3 ended two thirds of RL episodes). The suite (`make rl-tasks`) is built from the site registries minus everything SFT trained on; splits: `train`, `eval_seen_site_unseen_task`, `eval_unseen_site`, `eval_mutating`. Design: `docs/superpowers/specs/2026-08-26-rl-harness-design.md`.

GPU side (`cua/rl/trainer/`, `cua/rl/train.py`, `rl_config.yaml`, `constraints/rl-modal.txt`): ms-swift GRPO in vLLM server mode with a browser `Env`/scheduler plugin (`cua/rl/trainer/swift_plugin.py`, paths verified against ms-swift 4.5.2) that calls the env service (`cua/rl/service.py`, loopback + `RL_ENV_TOKEN`) over an ingress, one training record per turn with the episode reward broadcast; the SFT LoRA is merged into the base before RL so the KL anchor is the SFT policy. Two hard-won settings: server mode needs **two GPUs** (trainer↔vLLM NCCL weight sync cannot share a device — `compute.rollout_devices`/`trainer_devices`), and `response_prefix: ""`. Flow on Modal: `make rl-serve` + `make rl-expose` (ngrok, `RL_ENV_TOKEN`) on the Mac, then `make modal-grpo RL_ENV_URL=https://…` (`cua-outputs` adapter in, checkpoints every 10 min, final adapter at `<run>/adapter`); `make modal-grpo-smoke` is the first paid rung against the in-container `mock://` env. Before/after numbers come through `cua/cloud/modal/serve.py` + `make eval-online-mind2web` so GRPO deltas are comparable to the SFT table. `--mode fallback` (decoupled batch GRPO) is not implemented. What would actually move the RL score is ranked in `docs/open-items.md` §8, and §9 explains why the verification loop must enter the SFT corpus first.

## Where the reasoning lives

- `docs/decisions/` — dated decision records: `judge-consistency-audit.md` (why audited numbers differ from measured), `grpo-v2-first-live-run.md`, `sft-v2-corpus-round.md` / `sft-v2-results.md`, `sft-v3-plan.md` (current direction: act→save→verify supervision, long routes, multi-login flows), `real-sites-vs-controlled-backends.md`, `open-questions.md`.
- `docs/sft-corpus-gaps.md` — ranked, measured corpus gaps that drive each corpus round.
- `docs/open-items.md` — living blocker list with evidence (host memory, unverified bucket, watchdog inconsistency, human-only queue, RL levers).
- `docs/superpowers/specs/` and `plans/` — design specs for the login-gated lane, funnel intake, RL harness.
- `README.md` — user-facing walkthrough plus the generated experiment-registry block and the measured before/after tables with reproduction commands.
