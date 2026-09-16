# Conventions

The rules this repository enforces, and the reasons they exist. `docs/architecture.md` explains how
the code is shaped; this file is what a change has to satisfy.

## Comments and code style

- **Every comment and docstring is one line.** No multi-line comment blocks, no explanatory
  paragraphs above a function. A comment earns its place only where the code cannot say the thing
  itself: non-obvious library behaviour, a consequence that is invisible at that spot, or a choice
  that looks wrong but is deliberate. A comment restating the line below it is deleted.
- **Rationale does not live in the source.** Measurements, design arguments and the history of a
  decision go in `docs/` — `architecture.md` for why a module is shaped as it is,
  `docs/decisions/` for a dated record — or in the commit message. That is why the modules carry
  one-line comments while `docs/architecture.md` is long.
- **Never reflow a comment you did not write.** If indentation shifts around it, keep the original
  wrapping, so the diff shows only the real change.
- Prefer expressive naming and structure over commenting.

`scripts/check_comments.py` reports multi-line comment runs and multi-line docstrings; `make lint`
runs it.

## Registries over special cases

New behaviour is a data entry in an existing registry, not a new branch. A hardcoded name check
(`if x.name == "Specific Thing":`) is a smell — the question to ask before writing one is whether a
second instance appearing tomorrow would be absorbed by a data or config change alone.

The registries that already exist:

| registry | adds |
| --- | --- |
| `cua.pipeline.orchestrator.STAGES` + `schemas.STAGE_TO_*` | a funnel stage and its statuses |
| `cua.pipeline.frameworks.PROFILES` | a replay-mode test framework |
| `cua.pipeline.bootable_gate` signal sets | a bootable/non-bootable signal |
| `cua.pipeline.teacher.tools` tool registries | a teacher action |
| `cua.pipeline.safety_watchdog.ApprovalPolicy` (`WATCHDOG_POLICY`) | a guarded-mode approval policy |
| `cua.pipeline.sites` registries (`public_sites*.json`) | a capture site or a whole lane worklist |
| `cua.trajectory.actions.ACTION_SPECS` | an action verb, everywhere at once |
| `cua.eval.task_sources.TASK_SOURCES` | an eval task source |
| `cua.eval.student.backends.BROWSER_BACKENDS` / `STUDENT_BACKENDS` | a browser or policy backend |
| `cua.eval.screenspot.predictors.PREDICTOR_BACKENDS`, `scoring.COORD_SPACES` | an inference backend or coordinate convention |
| `cua.cloud.vast.base.PROVIDERS`, `jobs.JOBS`, `benchmarks.BENCHMARKS` | a GPU provider, a remote job, a benchmark |
| `cua.rl.checks`, `cua.rl.policies`, `cua.rl.backends` | an RL success check, policy or environment backend |
| `cua.cloud.vast.lifecycle.SMOKE_CASES` | a case in the no-cost teardown self-test |

## Imports

- **No nested imports.** Imports live at the top of the module. The exceptions that survive are the
  ones where a top-level import would be wrong rather than merely inconvenient: a genuine import
  cycle, a heavy optional dependency that a code path never reaching it must not be made to
  install, or a dependency that exists only inside a container image while the module still has to
  import on the host that launches it.
- **Every surviving nested import carries a `# lazy: <reason>` comment** on the line above it,
  naming which of those it is. One marker covers a whole contiguous run of nested imports, so a
  sorted block needs one comment rather than one per line, and the reason is mandatory:
  `scripts/check_nested_imports.py` reports any nested import that has no marker above it.
- **No imports inside tests.** `tests/` modules import at the top level; a test that needs a
  dependency the runner may not have is skipped by the runner, not guarded by a local import.

All three convention checks enforce. `scripts/check_comments.py`, `scripts/check_env_reads.py` and
`scripts/check_nested_imports.py` each fail `make lint` on a single finding anywhere in the tree;
`--report-only` exists for surveying a change in progress, never for the gate.

## No regex post-filters on LLM output

Generator output is trusted as written. When a generator produces something wrong, the fix is the
prompt — never a discard-on-pattern or retry-on-pattern filter downstream. Observability metrics
over the output are fine; content filters are not. A filter hides the prompt defect and silently
changes the distribution the corpus is built from.

## Burner accounts only

Login-gated capture never touches the operator's real accounts. Account creation and login are
in-scope skills for the student; the isolation is what makes them allowed: a disposable identity
(`persona.py`), a readable throwaway inbox (`mailtm.py`, `mail_service.py`), an isolated browser
context, a fail-closed credential registry under gitignored `credentials/` (`app_secrets.py`), and
`operator_pii.py` so the operator's own identity is never typed into a page or shipped in a corpus.
Credentials appear in a trajectory only as `<secret:...>` placeholders.

## Evaluation and RL episodes never enter training

They land under `eval_runs/` and `rl_runs/` — outside the `trajectories/**` glob — and carry
`metadata.excluded_reason` of `eval_episode` or `rl_rollout`. One site, `saucedemo`, is both a
benchmark subject and part of the SFT corpus, so numbers over it are never reported as unseen.

## Never `nohup` a long local run

A long local job runs in a supervised background process under `caffeinate`, so it can be stopped
deliberately and survives the machine sleeping. A `nohup`'d job outlives the session that started
it and becomes impossible to stop cleanly; sleep has silently killed an unsupervised sweep, a serve
loop and their waiters.

## Measurement: the experiment registry

**Every measured number goes into `experiments/`, and nothing that is not in the registry is
quoted anywhere.** `experiments/models.csv` says what was trained, `experiments/results.csv` what
was measured (one row per model × eval × subset × metric × repeat), and `experiments/sources.csv`
holds figures cited from other people's work. `make registry` fills Wilson intervals and re-renders
the block in `README.md` between its markers; `make registry-check` fails if that block is stale.
**Never edit the rendered block by hand.**

`status` carries meaning a value alone cannot:

- `done` — measured.
- `pending` — the run finished but the metric has not been computed. The value column stays empty
  rather than being filled with a zero.
- `planned` — on the worklist. These rows make the registry double as the experiment plan.
- `superseded` — the number exists but must not be quoted. The row stays and `notes` says why, so a
  retired figure cannot quietly return to a slide.

`artifacts_path` names where the raw records live and `artifacts_availability` says whether a
reader can actually open them (`committed`, `external:vetty`, `local-only`) — see
`experiments/README.md`.

A registry row saying `done` is not by itself evidence: before quoting a number, check that the run
finished and that its `artifacts_path` resolves.

Raw evaluation artefacts are committed under `eval_archive/` — per-sample ScreenSpot records,
Online-Mind2Web `results.json` / `report.md` / `run.log`, the pinned task files, and a
`manifest.json` carrying a sha256 per file — because `eval_runs/` and `eval_results/` are
gitignored. Per-episode screenshots (~166 MB) are packed next to the adapter under gitignored
`models/`.

## Evaluation conventions that are traps

Each of these silently produces a confident WRONG number rather than crashing.
`docs/evaluation.md` has the full treatment; this is the checklist.

- **`coord_space` is per MODEL, never global.** A stock Qwen and every sft-v3+ checkpoint answer in
  `normalized_1000`; sft-v1 and sft-v2 checkpoints, and the GRPO adapters trained over their merged
  weights, answer in `viewport_abs`. The same base-model predictions score 85% as `normalized_1000`
  and 25% as `viewport_abs`.
- **The bbox convention belongs to the dataset mirror.** `rootsautomation/ScreenSpot` stores bbox
  as `xyxy` normalised to 0-1. Reading those fractions as pixels is 0% accuracy by construction.
- **The image every arm sees is a harness setting**, not a property of a coordinate space. Two arms
  shown different pictures are not comparable, whatever else matched.
- **Egress is part of the measurement.** Online-Mind2Web is US-centric; a datacenter VPN trades
  geo-blocks for WAF-blocks roughly 1:1. The student eval records the resolved ASN as a parity key,
  because two runs on different egress measure the network rather than the model.
- **Report both denominators.** Blocked and unreachable episodes are excluded from our scored rate
  and included in the all-episode rate; published Online-Mind2Web figures use the second. Our
  numbers are comparable to our own before/after, not to a leaderboard.
- **`saucedemo` is in both the benchmark and the corpus.** Never report it as unseen.

## Running the checks

There is no pytest suite. The checks are tiered by what they need.

**Tier 1 — no Docker, no LLM, no network, no GPU.** This is `make check`, and it is what a change
has to pass:

```bash
make check      # every no-LLM self-test, the regression fixtures, registry-check and lint
make lint       # ruff check + ruff format --check + the three convention checks
make fmt        # ruff format + the safe autofixes
```

`make check` fans out to exactly `test-rl-unit` (`python -m tests`, which needs the `pipeline` extra
but no Docker, LLM or GPU — pass a substring to filter, e.g. `python -m tests sites`), `test-eval`,
`test-verify-gate`, `test-qa-verify`, `test-remote`, `test-corpus-audit`, `test-instruction-style`,
`test-outcome-relabel`, `registry-check`, `conventions-check`, `fixtures-check` and `lint`.
`make test-student-eval` is deliberately NOT among them: its backend-surface check builds the
`docker` browser backend, so it needs a Docker daemon and belongs to Tier 2.
`make conventions-check` is `python -m cua.settings --check`: it fails
when the environment table below no longer matches the field metadata, so the table cannot go stale
the way a hand-maintained one would.

`make fixtures-check` compares today's behaviour against the stored fixtures under
`tests/fixtures/`. They pin prompt bytes, the action grammar round-trip, CLI surfaces and
`make -n` output, so an accidental behaviour change shows up as a diff instead of as a worse model.
Rebuild them deliberately with `make fixtures-build` (pass `--corpus` so the corpus-derived
fixtures are not silently kept stale), and say in the report which fixtures changed and why.

**Tier 2 — needs Docker.** `make test-sandbox` (container lifecycle), `make test-driver` (a scripted
no-LLM browser-driver session asserting the trajectory contract), `make test-auth-flow` (the mail
stack and masked-secret wiring), `make test-student-eval` (parser round-trip, mock episode, judge
and the browser-backend surface), `make test-rl` (the end-to-end RL smoke).

**Tier 3 — needs an LLM, a GPU or money.** Nothing here is part of a gate. Verify a pipeline change
by running the relevant stage with a small `--limit` and `--budget`; verify a cloud change with
`make remote-train-dry` (real API, rents nothing) before anything that bills.

## Environment variables

Every variable below is read exactly once, at import of `cua/settings.py`, into a frozen section
object. No other module reads the environment **except** the marked runtime reads: each carries a
`# env-read: <reason>` marker naming the variable and the reason, and
`scripts/check_env_reads.py` fails `make lint` on an unmarked read. Those variables are therefore
not in the table below. `# env-read:` and the `# lazy:` marker in the Imports section are the two
marker conventions in the tree, and both make the exception legible at the line that takes it.

The marked exceptions fall into four groups:

- **Code that runs where the `cua` package is not importable.** The Modal serve and merge images
  (`cua/cloud/modal/serve.py`, `merge.py`) are invoked in script mode, so the container re-imports
  the single file with no package on the path. The injected replay recorders
  (`instrumentation/pw_python_conftest.py`, `selenium_python_conftest.py`) execute inside the
  repository under test. The auth-flow smoke (`smoke/auth_flow.py`) sends a program string into the
  sandbox app process.
- **Child-process environment propagation.** A value set in this process so that an inherited
  subprocess sees it: the `swift sft` child (`cua/sft/swift_launch.py`), the `swift infer` child
  (`cua/eval/screenspot/predictors.py`), the GRPO trainer subprocess and the repo root prepended to
  its path because ms-swift loads `--external_plugins` as a top-level module (`cua/rl/train.py`),
  the Modal GRPO lane passing a switch through to `cua.rl.train` (`cua/cloud/modal/grpo.py`), and
  the regression-fixture subprocess, which is handed the real `PATH`/`HOME` and nothing else
  (`tests/regression_fixtures.py`).
- **Write-then-read.** `cua/cloud/modal/train.py` exports `MAX_PIXELS` because ms-swift's template
  reads it back out of the environment; there is no flag for it.
- **Test toggles.** `cua/rl/trainer/core.py` reads an in-process switch the GRPO smoke flips to
  simulate a verdictless service, and `tests/test_trainer_core.py` sets and unsets it as a pair so
  the toggle never leaks into another test.

Regenerate the table with `python -m cua.settings --table`.

| variable | default | settings field | used by |
| --- | --- | --- | --- |
| `SANDBOX_MEM_LIMIT` | `2g` | `SANDBOX.mem_limit` | cua.pipeline.sandbox |
| `SANDBOX_MEMSWAP_LIMIT` | `SANDBOX.mem_limit` | `SANDBOX.memswap_limit` | cua.pipeline.sandbox |
| `SANDBOX_PIDS_LIMIT` | `1024` | `SANDBOX.pids_limit` | cua.pipeline.sandbox |
| `SANDBOX_CPUS` | `2` | `SANDBOX.cpus` | cua.pipeline.sandbox |
| `SANDBOX_TMP_SIZE` | `1g` | `SANDBOX.tmp_size` | cua.pipeline.sandbox |
| `SANDBOX_SHM_SIZE` | `256m` | `SANDBOX.shm_size` | cua.pipeline.sandbox |
| `SANDBOX_VM_RESERVE` | `1.5g` | `SANDBOX.vm_reserve` | cua.pipeline.sandbox |
| `SANDBOX_APP_READY_TIMEOUT` | `60` | `SANDBOX.app_ready_timeout` | cua.pipeline.app_runner, cua.pipeline.driver_session |
| `SANDBOX_APP_BUILD_TIMEOUT` | `420` | `SANDBOX.app_build_timeout` | cua.pipeline.app_runner |
| `PIPELINE_MAX_RETRIES` | `3` | `PIPELINE.max_retries` | cua.pipeline._processing |
| `PIPELINE_RETRY_BACKOFF` | `5` | `PIPELINE.retry_backoff` | cua.pipeline._processing |
| `PIPELINE_CAPTURE_MODE` | `agent` | `PIPELINE.capture_mode` | cua.pipeline.schemas |
| `PIPELINE_AUTH_FLOW_CAPTURE` | `0` | `PIPELINE.auth_flow_capture` | cua.pipeline.stage6_generate |
| `PIPELINE_AUTH_DISCOVERED_RATIO` | `0.25` | `PIPELINE.auth_discovered_ratio` | cua.pipeline.stage6_generate |
| `PIPELINE_LANGUAGE` | — | `PIPELINE.language` | cua.pipeline.registry |
| `PIPELINE_CREDENTIALS_DIR` | `credentials` | `PIPELINE.credentials_dir` | cua.pipeline.app_secrets |
| `STAGE3_BUDGET_USD` | `2.0` | `PIPELINE.stage3_budget_usd` | cua.pipeline.stage3_shallow_score |
| `STAGE3_GATE_APP_MIN` | `3` | `PIPELINE.stage3_gate_app_min` | cua.pipeline.stage3_shallow_score, cua.pipeline.dashboard.pages.4_Rejections |
| `STAGE3_GATE_RUNNABILITY_MIN` | `2` | `PIPELINE.stage3_gate_runnability_min` | cua.pipeline.stage3_shallow_score, cua.pipeline.dashboard.pages.4_Rejections |
| `STAGE3_GATE_INTERACTION_MIN` | `2` | `PIPELINE.stage3_gate_interaction_min` | cua.pipeline.stage3_shallow_score, cua.pipeline.dashboard.pages.4_Rejections |
| `STAGE4_BUDGET_USD` | `5.0` | `PIPELINE.stage4_budget_usd` | cua.pipeline.stage4_deep_analyze |
| `STAGE4_TOP_K_PER_PASS` | `15` | `PIPELINE.stage4_top_k_per_pass` | cua.pipeline.stage4_deep_analyze |
| `STAGE4_DEEP_SCORE_THRESHOLD` | `4` | `PIPELINE.stage4_deep_score_threshold` | cua.pipeline.stage4_deep_analyze, cua.pipeline.dashboard.pages.4_Rejections |
| `STAGE4_MIN_EST_TRAJECTORIES` | `3` | `PIPELINE.stage4_min_est_trajectories` | cua.pipeline.stage4_deep_analyze |
| `STAGE4_TOY_COMPLEXITY_MAX` | `2` | `PIPELINE.stage4_toy_complexity_max` | cua.pipeline.stage4_deep_analyze |
| `STAGE4_WORKERS` | `3` | `PIPELINE.stage4_workers` | cua.pipeline.orchestrator |
| `STAGE5_WORKERS` | `2` | `PIPELINE.stage5_workers` | cua.pipeline.orchestrator |
| `STAGE6_WORKERS` | `2` | `PIPELINE.stage6_workers` | cua.pipeline.orchestrator |
| `BOOTABLE_GATE` | `1` | `PIPELINE.bootable_gate` | cua.pipeline.stage2_filter |
| `BOOTABLE_GATE_AGGRESSIVE` | `0` | `PIPELINE.bootable_gate_aggressive` | cua.pipeline.stage2_filter |
| `BOOTABLE_GATE_WORKERS` | `8` | `PIPELINE.bootable_gate_workers` | cua.pipeline.stage2_filter |
| `BOOT_GATE_MIN_ELEMENTS` | `5` | `PIPELINE.boot_gate_min_elements` | cua.pipeline.stage5_trial_run |
| `AGENT_TASKS_PER_REPO` | `8` | `PIPELINE.tasks_per_repo` | cua.pipeline.task_proposer |
| `TASK_PROPOSE_BUDGET_USD` | `2.0` | `PIPELINE.task_propose_budget_usd` | cua.pipeline.task_proposer |
| `INSTRUCTION_BUDGET_USD` | `0.30` | `PIPELINE.instruction_budget_usd` | cua.pipeline.stage6_generate |
| `TASK_TYPE_BUDGET_USD` | `0.30` | `PIPELINE.task_type_budget_usd` | cua.pipeline.stage6_generate |
| `THINK_BUDGET_USD` | `0.30` | `PIPELINE.think_budget_usd` | cua.pipeline.think_generator |
| `THINK_SESSION_BUDGET_USD` | `5.0` | `PIPELINE.think_session_budget_usd` | cua.pipeline.think_generator |
| `POLISH_BUDGET_USD` | `0.50` | `PIPELINE.polish_budget_usd` | cua.pipeline.think_generator |
| `QA_VOICE_BUDGET_USD` | `0.50` | `PIPELINE.qa_voice_budget_usd` | cua.pipeline.corpus.instruction_style |
| `OUTCOME_WINDOW_STEPS` | `12` | `PIPELINE.outcome_window_steps` | cua.pipeline.corpus.outcome_relabel |
| `OUTCOME_BUDGET_USD` | `1.00` | `PIPELINE.outcome_budget_usd` | cua.pipeline.corpus.outcome_relabel, cua.pipeline.smoke.outcome_relabel |
| `OUTCOME_IMAGE_TOKENS` | `350` | `PIPELINE.outcome_image_tokens` | cua.pipeline.corpus.outcome_relabel |
| `OUTCOME_PRICE_IN_PER_M` | `2.0` | `PIPELINE.outcome_price_in_per_m` | cua.pipeline.corpus.outcome_relabel |
| `OUTCOME_PRICE_OUT_PER_M` | `10.0` | `PIPELINE.outcome_price_out_per_m` | cua.pipeline.corpus.outcome_relabel |
| `OUTCOME_CALL_OVERHEAD_USD` | `0.08` | `PIPELINE.outcome_call_overhead_usd` | cua.pipeline.corpus.outcome_relabel |
| `PUBLIC_TASKS_PER_SITE` | `6` | `PIPELINE.public_tasks_per_site` | cua.pipeline.lanes.public |
| `DRIVER_PORT` | `8765` | `PIPELINE.driver_port` | cua.pipeline.driver_session, cua.rl.backends |
| `DRIVER_PLAYWRIGHT_CORE_VERSION` | `1.55.0` | `PIPELINE.driver_playwright_core_version` | cua.pipeline.driver_session |
| `DRIVER_READY_TIMEOUT` | `60` | `PIPELINE.driver_ready_timeout` | cua.pipeline.driver_session |
| `AGENT_MAX_ACTIONS` | `60` | `CAPTURE.max_actions_per_task` | cua.pipeline.teacher.run, cua.pipeline.teacher.tools |
| `AGENT_TASK_BUDGET_USD` | `25.0` | `CAPTURE.task_budget_usd` | cua.pipeline.teacher.run |
| `AGENT_MAX_TURNS` | `150` | `CAPTURE.max_turns` | cua.pipeline.teacher.run |
| `VERIFY_BUDGET_USD` | `0.30` | `CAPTURE.verify_budget_usd` | cua.pipeline.teacher.verify |
| `AGENT_RELABEL_MIN_STEPS` | `3` | `CAPTURE.relabel_min_steps` | cua.pipeline.corpus.relabel_recoverable |
| `AGENT_DEGENERATE_LOOP_FRACTION` | `0.6` | `CAPTURE.degenerate_loop_fraction` | cua.pipeline.teacher.verify |
| `AGENT_DEGENERATE_MIN_REPEATS` | `3` | `CAPTURE.degenerate_min_repeats` | cua.pipeline.teacher.verify |
| `AGENT_CONTAINER_TIMEOUT` | `10800` | `CAPTURE.container_timeout` | cua.pipeline.stage6_generate |
| `AGENT_TASK_TIME_RESERVE` | `900` | `CAPTURE.task_time_reserve` | cua.pipeline.stage6_generate |
| `AGENT_MAX_KEPT_PER_REPO` | `10` | `CAPTURE.max_kept_per_repo` | cua.pipeline.stage6_generate |
| `AGENT_MAX_CONSECUTIVE_FAILS` | `3` | `CAPTURE.max_consecutive_fails` | cua.pipeline.stage6_generate |
| `CAPTURE_MAX_ATTEMPTS` | `2` | `CAPTURE.max_attempts` | cua.pipeline.lanes.capture_lane, tests.test_capture_lane |
| `PUBLIC_CONTAINER_TIMEOUT` | `10800` | `CAPTURE.public_container_timeout` | cua.pipeline.lanes.public |
| `LOGINGATED_CONTAINER_TIMEOUT` | `5400` | `CAPTURE.logingated_container_timeout` | cua.pipeline.lanes.logingated |
| `REGISTER_CONTAINER_TIMEOUT` | `5400` | `CAPTURE.register_container_timeout` | cua.pipeline.lanes.register_real_site |
| `PROBE_CONTAINER_TIMEOUT` | `1800` | `CAPTURE.probe_container_timeout` | cua.pipeline.lanes.probe_sandbox |
| `WATCHDOG_POLICY` | `llm` | `CAPTURE.watchdog_policy` | cua.pipeline.safety_watchdog |
| `WATCHDOG_BUDGET_USD` | `0.20` | `CAPTURE.watchdog_budget_usd` | cua.pipeline.safety_watchdog |
| `PERSONA_EMAIL_DOMAIN` | `example.com` | `CAPTURE.persona_email_domain` | cua.pipeline.persona |
| `PERSONA_PHONE_PREFIX` | `+1202555` | `CAPTURE.persona_phone_prefix` | cua.pipeline.persona |
| `MAIL_HOST_SMTP_PORT` | `13025` | `CAPTURE.mail_host_smtp_port` | cua.pipeline.mail_service |
| `MAIL_HOST_IMAP_PORT` | `13143` | `CAPTURE.mail_host_imap_port` | cua.pipeline.mail_service |
| `MAIL_HOST_WEBMAIL_PORT` | `18080` | `CAPTURE.mail_host_webmail_port` | cua.pipeline.mail_service |
| `OPERATOR_PII` | `""` | `CAPTURE.operator_pii` | cua.pipeline.operator_pii |
| `OPERATOR_PII_FILE` | — | `CAPTURE.operator_pii_file` | cua.pipeline.operator_pii |
| `SDK_QUERY_TIMEOUT` | `2400` | `SDK.query_timeout_s` | cua.pipeline.agent_runner, cua.pipeline.sdk_runner |
| `CLAUDE_CLI_PATH` | — | `SDK.claude_cli_path` | cua.pipeline.sdk_runner |
| `PREV_IMAGE_SIZE` | `640x360` | `PROMPT.prev_image_size` | cua.prompt.effects, cua.prompt.format |
| `EFFECT_NO_CHANGE_MAX_FRACTION` | `0.0005` | `PROMPT.effect_no_change_max_fraction` | cua.prompt.effects |
| `EFFECT_PIXEL_DELTA` | `16` | `PROMPT.effect_pixel_delta` | cua.prompt.effects |
| `EFFECT_PATH_MAX_CHARS` | `40` | `PROMPT.effect_path_max_chars` | cua.prompt.effects |
| `WASTE_CLICK_RADIUS_PX` | `10` | `CORPUS.waste_click_radius_px` | cua.sft.corpus_filters |
| `WASTE_LOOKBACK_STEPS` | `4` | `CORPUS.waste_lookback_steps` | cua.sft.corpus_filters |
| `MEANDER_KEEP_HEAD` | `5` | `CORPUS.meander_keep_head` | cua.sft.corpus_filters |
| `MEANDER_KEEP_TAIL` | `5` | `CORPUS.meander_keep_tail` | cua.sft.corpus_filters |
| `MEANDER_MID_KEEP_EVERY` | `6` | `CORPUS.meander_mid_keep_every` | cua.sft.corpus_filters |
| `KEYPRESS_MAX_REPEAT` | `6` | `CORPUS.keypress_max_repeat` | cua.sft.corpus_filters |
| `GROUNDING_SMALL_OVERSAMPLE` | `3` | `GROUNDING.small_oversample` | cua.sft.grounding |
| `GROUNDING_PER_GROUP_CAP` | `40` | `GROUNDING.per_group_cap` | cua.sft.grounding |
| `GROUNDING_SEED` | `0` | `GROUNDING.seed` | cua.sft.grounding |
| `EVAL_RUNS_DIR` | `eval_runs` | `EVAL.runs_dir` | cua.eval.student.config |
| `EVAL_EPISODE_TIME_RESERVE` | `900` | `EVAL.episode_time_reserve` | cua.eval.student.backends |
| `EVAL_CONTAINER_TIMEOUT` | `10800` | `EVAL.container_timeout` | cua.eval.student.backends |
| `EVAL_MAX_PARSE_FAILURES` | `3` | `EVAL.max_parse_failures` | cua.eval.student.cli |
| `EVAL_MAX_UNPRODUCTIVE_STEPS` | `5` | `EVAL.max_unproductive_steps` | cua.eval.student.cli |
| `EVAL_COORD_SPACE_SUSPECT_RATE` | `0.25` | `EVAL.coord_space_suspect_rate` | cua.eval.student.fingerprint |
| `EVAL_ENDPOINT` | `""` | `EVAL.endpoint` | cua.eval.student.cli |
| `EVAL_MODEL` | `""` | `EVAL.model` | cua.eval.student.cli |
| `EVAL_ADAPTER` | `""` | `EVAL.adapter` | cua.eval.student.cli |
| `EVAL_API_KEY` | `""` | `EVAL.api_key` | cua.eval.student.cli |
| `EVAL_COORD_SPACE` | `normalized_1000` | `EVAL.coord_space` | cua.eval.student.cli, cua.eval.student.fingerprint |
| `EVAL_ENV_FILE` | `""` | `EVAL.env_file` | cua.eval.task_sources |
| `VERIFY_MID_SCREENSHOTS` | `2` | `EVAL.verify_mid_screenshots` | cua.pipeline.teacher.verify |
| `HF_ENDPOINT` | `https://huggingface.co` | `EVAL.hf_endpoint` | cua.eval.task_sources |
| `HF_TOKEN` | — | `EVAL.hf_token` | cua.cloud.vast.config |
| `HUGGING_FACE_HUB_TOKEN` | — | `EVAL.hugging_face_hub_token` | — |
| `HUGGINGFACE_HUB_TOKEN` | — | `EVAL.huggingface_hub_token` | — |
| `HF_READ_TOKEN` | — | `EVAL.hf_read_token` | — |
| `SCREENSPOT_IMAGE_SIZE` | `native` | `EVAL.screenspot_image_size` | cua.eval.screenspot.config |
| `SCREENSPOT_ENDPOINT` | `""` | `EVAL.screenspot_endpoint` | cua.eval.screenspot.predictors |
| `SCREENSPOT_API_KEY` | `""` | `EVAL.screenspot_api_key` | cua.eval.screenspot.predictors |
| `SCREENSPOT_WORKERS` | `8` | `EVAL.screenspot_workers` | cua.eval.screenspot.predictors |
| `SCREENSPOT_CLAUDE_WORKERS` | `4` | `EVAL.screenspot_claude_workers` | cua.eval.screenspot.predictors |
| `SCREENSPOT_CLAUDE_BUDGET_USD` | `0.30` | `EVAL.screenspot_claude_budget_usd` | cua.eval.screenspot.predictors |
| `CUA_TRAIN_GPU` | `A100-80GB` | `CLOUD.train_gpu` | cua.cloud.modal.train |
| `CUA_RUN_NAME` | `sft-v1` | `CLOUD.run_name` | cua.cloud.modal.train, tests.test_modal_common |
| `CUA_TRAIN_TIMEOUT` | `36000` | `CLOUD.train_timeout` | cua.cloud.modal.train |
| `CUA_TRAIN_COMMIT_EVERY_S` | `600` | `CLOUD.train_commit_every_s` | cua.cloud.modal.train |
| `CUA_GRPO_GPU` | `A100-80GB:2` | `CLOUD.grpo_gpu` | cua.cloud.modal.grpo |
| `CUA_GRPO_TIMEOUT` | `32400` | `CLOUD.grpo_timeout` | cua.cloud.modal.grpo |
| `CUA_GRPO_COMMIT_EVERY_S` | `600` | `CLOUD.grpo_commit_every_s` | cua.cloud.modal.grpo |
| `VAST_API_KEY` | — | `CLOUD.vast_api_key` | cua.cloud.vast.config |
| `REMOTE_TRAIN_STATE_FILE` | `.remote_train_state.json` | `CLOUD.remote_train_state_file` | cua.cloud.vast.state |
| `REMOTE_TRAIN_BAD_OFFERS` | `.remote_train_bad_offers.json` | `CLOUD.remote_train_bad_offers` | cua.cloud.vast.state |
| `REMOTE_TRAIN_EXCLUDE_COUNTRIES` | `""` | `CLOUD.remote_train_exclude_countries` | cua.cloud.vast.config |
| `REMOTE_TRAIN_INSTANCE_LABEL` | `""` | `CLOUD.remote_train_instance_label` | cua.cloud.vast.config |
| `RL_ENV_HOST` | `127.0.0.1` | `RL.env_host` | cua.rl.service |
| `RL_ENV_PORT` | `8900` | `RL.env_port` | cua.rl.service |
| `RL_ENV_TOKEN` | `""` | `RL.env_token` | cua.cloud.modal.grpo, cua.rl.service, cua.rl.trainer.swift_plugin |
| `RL_ENV_URL` | `http://127.0.0.1:8900` | `RL.env_url` | cua.rl.trainer.swift_plugin |
| `RL_SERVE_PORT` | `8001` | `RL.serve_port` | cua.rl.rollout |
| `RL_RUN_DIR` | `/workspace/rl_runs/current` | `RL.run_dir` | cua.rl.trainer.swift_plugin |
| `RL_RUN_ID` | `grpo` | `RL.run_id` | cua.rl.trainer.swift_plugin |
| `RL_RESET_TIMEOUT_S` | `45` | `RL.reset_timeout_s` | cua.rl.service |
| `RL_EPISODE_IDLE_S` | `600` | `RL.episode_idle_s` | cua.rl.service |
| `RL_CONTAINER_TIMEOUT` | `14400` | `RL.container_timeout` | cua.rl.backends |
| `RL_RECYCLE_RESERVE_S` | `1200` | `RL.recycle_reserve_s` | cua.rl.backends |
| `RL_DRIVERS_PER_SANDBOX` | `1` | `RL.drivers_per_sandbox` | cua.rl.backends |
| `RL_LOOP_GUARD_REPEATS` | `5` | `RL.loop_guard_repeats` | cua.rl.backends, cua.rl.rollout, cua.rl.service, tests.test_backends |
| `RL_RESET_WAIT_S` | `1800` | `RL.reset_wait_s` | cua.rl.trainer.core |
| `RL_STEP_WAIT_S` | `2400` | `RL.step_wait_s` | cua.rl.trainer.core |
| `RL_VERIFIER_CONCURRENCY` | `2` | `RL.verifier_concurrency` | cua.rl.judge |
| `RL_VERIFY_TIMEOUT_S` | `2700` | `RL.verify_timeout_s` | cua.rl.judge |
| `RL_VERIFY_RETRY_S` | `300` | `RL.verify_retry_s` | cua.rl.judge, tests.test_judge |
| `RL_VERIFY_WAIT_S` | `1800` | `RL.verify_wait_s` | cua.rl.judge |
| `RL_POLICY_BACKOFF_S` | `5` | `RL.policy_backoff_s` | cua.rl.policies |
| `RL_POLICY_WAIT_S` | `600` | `RL.policy_wait_s` | cua.rl.policies |
| `RL_POLICY_API_KEY` | `""` | `RL.policy_api_key` | cua.rl.policies |
| `CUDA_VISIBLE_DEVICES` | `0` | `RL.cuda_visible_devices` | cua.rl.train |
| `TELEGRAM_BOT_TOKEN` | `""` | `NOTIFY.telegram_bot_token` | cua.pipeline.notifier |
| `TELEGRAM_CHAT_ID` | `""` | `NOTIFY.telegram_chat_id` | cua.pipeline.notifier |
| `GITHUB_TOKEN` | `""` | `GITHUB.token` | cua.pipeline.orchestrator |
