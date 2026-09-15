# End-to-end RL (GRPO) harness for the Qwen3.5-9B computer-use student

**Date:** 2026-08-26
**Branch:** `feat/rl-harness` (off main after PR #24 + #25 land)
**Status:** Approved design; implementation plan in `docs/superpowers/plans/2026-08-26-rl-harness-phase1.md`

## Context

The repo has a 1,116-trajectory corpus (~586 verified, 504 trainable across 328 sites), an SFT converter/trainer (`cua/sft/legacy_train_qwen.py` → `swift sft`), a sandboxed browser env (`browser_driver.js` + `DriverClient`), a Claude verifier (`verify_trajectory`), ~2.4k authored tasks, and a rented-GPU lane (`cua/cloud/vast/`, PR #25). What does **not** exist on any branch: a way to run the student, an action decoder, an episode loop, a held-out task suite, an RL-rate reward, or any trainer other than `swift sft`. No SFT checkpoint has ever been produced (`output/` never existed). RL was an explicit non-goal of the login-gated spec, but its §10 "Phase-2 harness" + `docs/prompts/e3-student-eval-harness.md` describe most of the rollout side.

Goal: after SFT, run **GRPO** on the student using live browser rollouts, with a harness that doubles as the missing eval harness (base vs SFT vs RL deltas).

## Decisions locked with the user

| Decision | Choice |
|---|---|
| Trainer | GRPO (ms-swift `swift rlhf --rlhf_type grpo`, vLLM server mode, LoRA); decoupled batch-GRPO as the fallback on the same rollouts |
| First env lane | Live public sites: login-free + `pre_session` login-gated (burner accounts). Sandbox repo apps stay behind the same interface, deferred |
| Base branch | Merge PR #24 (login-gated) and PR #25 (remote-train) to main first, then `feat/rl-harness` off main |
| SFT checkpoint | None exists → the SFT run is Phase 2 of this plan (existing `make remote-train` lane) |
| Lane in v1 | **autonomous only** (the SFT corpus' login-gated captures are autonomous; guarded lane + watchdog-in-rollout deferred — the student input carries no lane signal, so mixing lanes would give conflicting rewards on similar screens) |

Hard conventions: one-line comments only; no nested imports; no imports inside tests; registries/config over name checks; **no `git commit`/merge/push without an explicit go-ahead per action**; cloud work keeps the existing teardown guarantees, RU/CN deny-list, `roman.maior` label; Claude calls only via the logged-in `claude` CLI (Mac).

## Architecture

```
Mac                                                      GPU box (Vast.ai, rented per run)
┌──────────────────────────────────────────┐             ┌──────────────────────────────────────┐
│ cua/rl/service.py  :8900  (env service)      │◄─ ssh -R ───│ swift rollout :8000 (vLLM + plugin)  │
│   WorkerPool → DockerSandbox × N         │             │   cua/rl/trainer/swift_plugin.py         │
│     browser_driver.js (×2 per sandbox)   │             │     BrowserEnv.reset/step → :8900    │
│   Episode: decode → sequence act → obs   │             │ swift rlhf --rlhf_type grpo          │
│   rewards: success_check | verifier      │             │   LoRA on merged(SFT) model          │
│   (Claude via `claude` CLI OAuth)        │             │ swift deploy :8001 (eval phases)     │
│ cua/rl/eval.py / cua/rl/rollout.py ── ssh -L ────┼────────────►│                                      │
│ rl_runs/<run_id>/…  (never trajectories/)│             │ /workspace/rl_output/latest  ─rsync─►│
└──────────────────────────────────────────┘             └──────────────────────────────────────┘
```

**The contract (both halves build to this):**
- `POST /reset {task_id, run_id, iter, sample_k, policy_tag?}` → `{episode_id, observation}`; `POST /step {episode_id, response_text}` → `{observation|null, reward|null, done, info}`; `POST /close {episode_id, reason?}` → `{reward, info}`; `GET /health`, `GET /tasks?split=`, `GET /episode/<id>`. Header `X-RL-Token`. Bind `127.0.0.1:8900`.
- `observation = {"system": cua.sft.legacy_train_qwen.SYSTEM_PROMPT, "text": build_step_user_text(...), "image_b64": <PNG>, "step_index": n}` — byte-identical to the SFT user turn, built by the same `cua/sft/legacy_train_qwen.py` functions. Never the driver's text snapshot.
- `response_text` = the policy's raw `<think>…</think>\naction: …` block; decoding/execution/recording is Mac-side.
- Reward = one scalar per episode (trajectory-level), computed Mac-side, broadcast to every turn record on the box.
- Every rollout is a full trajectory dir under `rl_runs/<run_id>/iter_NNN/<task_id>/kKK/task_<episode_id>/` so `verify_trajectory`, `corpus_audit.audit_trajectory` and `convert_trajectory_to_examples` work on it unchanged; `metadata.excluded_reason="rl_rollout"` and the path outside `trajectories/` keep it out of SFT.

## Phase 0 — land the base (needs explicit go-ahead for each merge)

1. Merge PR #24 `feat/login-gated-trajectories` → main. Conflicts only in `cua/pipeline/llm_config.py` (branch pins `claude-sonnet-4-6` literals + adds `MODEL_FOR_SAFETY_JUDGE`; keep main's `LATEST_MODEL_BY_CLASS` and add the judge purpose) and `cua/pipeline/stage6_generate.py`.
2. Merge PR #25 `feat/vastai-remote-training` → main. Overlaps #24 only on `.gitignore`, `Makefile`.
3. `feat/rl-harness` off main in a worktree. First task: write `docs/superpowers/specs/2026-08-26-rl-harness-design.md` from this plan (the repo's spec convention), then implement.

## Phase 1 — Mac-side rollout harness (`cua/rl/`, no GPU)

Layout (top-level package: it imports both `cua.sft.legacy_train_qwen` and `cua.pipeline.*`; `cua.pipeline` must not grow a training-time dependency):
```
cua/rl/  actions.py observation.py executor.py backends.py episode.py checks.py rewards.py
     tasks.py service.py policies.py rollout.py eval.py smoke.py rl_tasks.json reward_config.json
     tests/{__main__.py,test_actions.py,test_checks.py,test_rewards.py,test_tasks.py,test_observation.py}
     trainer/{__init__.py,core.py,swift_plugin.py}     # box-side (Phase 3)
```

### 1.1 Shared student-input builders — `cua/sft/legacy_train_qwen.py` (pure refactor, output must be byte-identical)
- Extract `build_step_user_text(instruction, action_history) -> str` (the inline `Task:` / windowed `Previous actions: [...]` / `What action should you take?` block in `convert_trajectory_to_examples`), `history_entries_for_step(step) -> list[str]` (`format_action_summary` per action + `[watchdog: <decision>]` for approval events, already in the convert loop), `build_step_messages(system, user_text, image_ref)`.
- Verify: run `convert_trajectory_to_examples` per dir on the parent-repo corpus before/after and diff the emitted user texts.

### 1.2 Action decoder — `cua/rl/actions.py` (inverse of `format_action_for_training`)
- `decode_step(text) -> StepDecode{thought, actions, errors, canonical, raw}`; `encode_step(thought, actions)` delegates to `format_action_for_training` (single encoder, never re-implemented).
- `ACTION_DECODERS` registry keyed by type; anchored regexes with shared `_REF = <ref>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)</ref>`; `type("…")` greedy to the last `")` (encoder does no escaping); `keypress` via `json.loads` (python-repr fallback → `canonical=False`); `drag` endpoints; `navigate`, `handoff(reason=)`, `confirm(propose=…, risk=…)` per the branch grammar; `<think>` must precede actions; stray text / unknown type / zero actions → errors.
- Tests: fixture list covering every action shape (incl. `type` with `"`, `(`, `)`, unicode, empty; multi-action steps; missing think; malformed) → `decode(encode(a)) == a` and `canonical`.

### 1.3 Driver additions — `cua/pipeline/instrumentation/browser_driver.js` + `driver_session.py`
- `POST /observe` → `{b64 PNG, url, title}`: `scrubDevOverlays` + `settle` + lossless screenshot **stashed as the next step's `step_NNN_screenshot.png`**; `recordStep` (which today screenshots inside `/act`) consumes the stash. Guarantees observation bytes == recorded PNG. `/screenshot` stays JPEG for the teacher.
- `/act` kind `sequence {thought, actions:[contract dicts]}`: records **one** step with the student's action dicts verbatim (multi-action steps are one SFT example → one recorded step), viewport-clamps, `loopGuard` on the serialized list, then executes in order via a `SEQUENCE_EXECUTORS` table: `click`/`double_click`/`move` → `page.mouse`; **`type` → select-all if the focused element is editable, then `keyboard.type(resolveSecrets(text))`** (ref-free; matches the teacher `type` kind's select-all-then-type semantics and Path A); `keypress` → `keyboard.press(keys.join('+'))`; `scroll` → move + `wheel`; `drag` → down/move/up; `wait`; `navigate` → goto + settle; `done`/`handoff`/`confirm` → markers only. Returns `{ok, step, url, terminal, executed, error_at?}`.
- `DRIVER_PORT` per instance (`start_driver(sandbox, port)`, log per port) so one sandbox hosts 2 drivers = 2 concurrent episodes. `DriverClient(sandbox, port)` gains `observe()`, `act_sequence(thought, actions)`.
- `make test-driver` (`cua/pipeline/driver_smoke.py`) extended with an `/observe` + `sequence` leg.

### 1.4 Executor, backends, episode — `cua/rl/executor.py`, `cua/rl/backends.py`, `cua/rl/episode.py`
- `executor.to_sequence_payload(decode, thought_fallback)`: drops legacy `screenshot`, normalizes keys via `cua.path_a.recorder.CUA_KEY_TO_PLAYWRIGHT`, bounds-checks against `corpus_audit.VIEWPORT`, substitutes `MISSING_THOUGHT_PLACEHOLDER` (driver requires non-empty thought); `control_actions()` finds the first `done|handoff|confirm|navigate`.
- `BACKENDS = {"public_site", "static_page", "repo_app"(stub)}` with `open/start_url/pre_episode/session_kwargs/post_episode`. `public_site` = sandbox + driver(s), no app boot (pattern: `public_capture.capture_site`). `resolve_preconditions(task)` FAIL-CLOSED reusing `public_capture._resolve_site_auth` semantics (`app_secrets.load_storage_state`, `is_registered_burner`); missing → task skipped.
- `RolloutWorker` owns one `DockerSandbox(container_timeout=RL_CONTAINER_TIMEOUT=4h)`, recycles when `remaining_seconds() < 20 min` or the driver stops answering; `WorkerPool(n = safe_parallel_sandboxes(requested))` (12.5 GB Docker VM → 3 sandboxes × 2 drivers = 6 episodes); `install_sigterm_handler()` at start.
- `Episode.reset/step/close`: `episode_task_id = f"{task_id}__i{iter:03d}__k{k:02d}"`; `task_meta` to `/session` carries `expected_actions` + `autonomy` so `verify_trajectory` picks its prompt and `efficiency_ratio` unchanged. `step()`: decode → parse failure = not recorded, re-observe, abort after 3 consecutive; `navigate` allowed only to `start_url` host ∪ `task.allowed_hosts`; loop-guard refusal → terminal (`RL_LOOP_GUARD_TERMINATES=1`); `done` → terminal; in v1 `handoff`/`confirm` → terminal `control_unexpected` (penalized). Terminal: run `success_check` while the context is alive → `finish(outcome, summary)` via `OUTCOME_BY_REASON` → `copy_out` → `metadata.json` (shape of `public_capture._verify_and_finalize` + `autonomy/auth_mode/preconditions`, `source_repo="rl:<task_id>"`, `excluded_reason="rl_rollout"`) + `episode.json` (raw texts, decodes, infos, timings, reward breakdown) → reward.

### 1.5 Reward — `cua/rl/checks.py`, `cua/rl/rewards.py`, `cua/rl/reward_config.json`
- `CHECKS` registry (v1 set): `url_matches`, `url_contains`, `title_contains`, `text_visible`, `text_absent`, `element_present{role?,name_regex}`, `element_value`, `cookie_present`, `done_emitted`, `all_of`/`any_of`; `ctx` built from `client.snapshot()` (page text + `[N] role "name"` lines) and lazy `storage_state()`. Result stored in `capture_meta["success_check"]`.
- `REWARD_COMPONENTS = {terminal_success, format_validity, efficiency, terminated_with_done}` + `PENALTIES = {parse_abort, loop_guard, step_cap, control_unexpected, navigate_blocked}`; weights per lane in `reward_config.json`; scalar clipped to `[-1, 1.5]`; breakdown persisted in `episode.json`.
- `terminal_success`: `success_check` if the task has one, else `verify_trajectory(dir, autonomy=lane)` (cached via `"verifier" in capture_meta`, `RL_VERIFIER_CONCURRENCY=2`, all Claude calls on one background asyncio thread so `rate_limit_tracker`'s single-loop assumption holds). `verifier_mode: always|fallback|never` (train: `fallback`, eval: `always`). `rate_limit_tracker.limit_reached()` → `reward: null, info.reward_status="deferred"`; `cua.rl.rollout --rescore` finishes later.

### 1.6 Task suite — `cua/rl/tasks.py`, `cua/rl/rl_tasks.json`
- `python -m cua.rl.tasks build|check|list`. `TASK_SOURCES` registry: `public_sites_actions_loginfree.json` (149/221 → `env:live, lane:autonomous, auth_mode:none`), `public_sites_actions_logingated.json` + `public_sites_human.json` (only sites with a registered burner + storage_state → `auth_mode:pre_session`, `preconditions{requires_session, session_ref, burner_account_id, env:"live"}`), `public_sites_saucedemo.json` (`env:test`).
- Exclusions asserted by `check`: every `public_sites.json` task; any task whose dir or normalized instruction (`public_capture._norm_instruction`) exists under `trajectories/`; `skip_capture` sites; `requires_upload`; auth tasks lacking preconditions.
- Fields: `task_id (<site>__<id>), site, start_url, instruction, difficulty, expected_actions, backend, env, lane, auth_mode, idempotent, split, preconditions, allowed_hosts, success_check?, throttle{min_interval_s}`. `idempotent` heuristic (mutating-verb regex list, `env==test`); train split = idempotent only. Deterministic splits: `sha1(site)%100<15` → `eval_unseen_site`; one task per remaining site → `eval_seen_site_unseen_task`; rest → `train`. `check` prints the suite sha256 (goes into run headers).
- Manual curation step: author `success_check` for the initial ~40 train tasks so training rewards rarely hit the Claude verifier.

### 1.7 Service, policies, rollout, eval — `cua/rl/service.py`, `cua/rl/policies.py`, `cua/rl/rollout.py`, `cua/rl/eval.py`
- `service.py`: stdlib `ThreadingHTTPServer`, one thread per driver slot (driver calls never interleave), verifier/watchdog on the async bridge, idle-episode reaper, `503 {retry_after}` when no slot. `python -m cua.rl.service --workers 3 --drivers-per-sandbox 2 --run-id <id>`.
- `POLICIES = {"openai": OpenAICompatiblePolicy(endpoint, model, temperature, max_tokens=512), "mock": MockReplayPolicy(traj_dir)}`; messages from `observation.to_openai_messages` (data-URI image).
- `cua.rl.rollout --policy openai --endpoint … --split train --k 4 --iter N --run-id …` (in-process pool, no HTTP) writes `rollouts.jsonl` (and, for the fallback trainer, `response_token_ids` + `old_logprobs` from vLLM `logprobs`).
- `cua.rl.eval --endpoint … [--baseline report.json] [--checkpoint-sha …]` (temperature 0, k=1, `max_steps 40`, `verifier_mode always`): per split verified success, check pass rate, parse-failure rate (turns), parse-abort rate (episodes), done() rate, mean steps on successes, loop-guard rate, per-domain table; header with endpoint model id, checkpoint sha, `git rev-parse HEAD`, suite sha; `report.md` + `report.json` under `eval_runs/<run_id>/`. This is the E3 harness.

### 1.8 Tests, smoke, wiring
- `tests/__main__.py` runs `test_*` functions (pytest-compatible, no dependency) → `make test-rl-unit`.
- `cua/rl/smoke.py` → `make test-rl` (Docker only, no LLM): `static_page` backend serving `driver_smoke.TEST_PAGE`; scripted mock policy with a multi-action step, a malformed line, scroll, keypress, `done()`; asserts trajectory contract, **observation PNG bytes == recorded step PNG**, **parity**: `convert_trajectory_to_examples` user texts == served `text` per step, `audit_trajectory` clean, `success_check.passed`, reward breakdown, `excluded_reason`; plus an in-process `cua.rl.service` round-trip.
- Makefile: `rl-tasks`, `rl-serve`, `rl-rollout`, `rl-eval` (= `make eval`), `test-rl-unit`, `test-rl`. `.gitignore`: `rl_runs/`, `eval_runs/`. CLAUDE.md section "RL rollout harness".
- Fix in passing: `stage6_generate._excluded` writes `excluded_reason` to `capture_meta.json` while `corpus_audit.is_trainable` / `cua.sft.legacy_train_qwen.convert_trajectories` read `metadata.json` — write both, read either.

## Phase 2 — SFT checkpoint + first live numbers (one short GPU rental)

1. `make convert` on the current corpus (login-gated converter; `training_data/` is stale from Jun 28) — read the health report.
2. `make remote-train` (existing `TrainJob`) → `output/` LoRA adapter; record `adapter_model.safetensors` sha.
3. New small job `python -m cua.cloud.vast serve --adapter output` (`ServeJob`: rent → `swift deploy --infer_backend vllm --adapters … :8001` → `-L` tunnel → block until Ctrl-C / `--for-seconds`; same teardown). Use it to run `make rl-eval` for base (`coord_space normalized_1000`, `--baseline`) and SFT, and to live-smoke the harness against the real 9B (parse rate, `<think>` shape, temperature>0 sampling) before paying for GRPO.

## Phase 3 — GRPO trainer (`cua/rl/trainer/`, `cua/rl/train.py`, `rl_config.yaml`, `requirements-rl.txt`, `remote_train` `GrpoJob`)

Two design decisions:
- **D1 Merge the SFT LoRA into the base before RL** (`swift export --adapters output --merge_lora true`, ~3 min, +19 GB). ms-swift's LoRA reference = adapter-off model; without the merge the KL anchor is the raw base (wrong coordinate space, no grammar). Merged model is also what vLLM serves and what "eval before" scores.
- **D2 One training record per turn**, not one multi-image sequence per episode (a 1280×720 screenshot ≈ 900 visual tokens; a 10-turn episode blows `max_length` exactly as it did for SFT). Each turn = `[system, user(<image>, text), assistant]`; the scheduler returns `List[RolloutOutput]` (one per turn, ms-swift's documented multi-record path) with the episode's terminal reward in every record's `rollout_infos['total_reward']`; `compute_advantages_dynamic` groups by prompt id.

### 3.1 `cua/rl/trainer/core.py` (pure, Mac-testable) and `cua/rl/trainer/swift_plugin.py` (box only)
- `ENV_CLIENTS = {"http": HttpEnvClient(base_url, timeout, retries, backoff), "mock": MockEnvClient}` keyed by URL scheme; `observation_to_user_message`, `split_episode(turns, total_reward)`, `EpisodeRecorder(run_dir)` (box-side `obs/*.png` + `episodes.jsonl`).
- `BrowserEnv(swift.rollout.gym_env.Env)`: `reset(config)` → env client `/reset`, writes the PNG, returns `(text, info, SYSTEM_PROMPT)` (`cua/sft/legacy_train_qwen.py` + `cua/pipeline/{__init__,corpus_audit}.py` ship to the box like `benchmarks.py` does); `step(messages)` posts the last assistant content; any error after retries → `done=True, reward=0, info.env_error` (an episode failure never fails the run).
- `BrowserScheduler(GYMScheduler)`: `on_trajectory_start` sets `req.images`; `step` **replaces** messages/images with the fresh single-image prompt (history is inside `text`); `run()` returns per-turn `RolloutOutput`s with `rollout_infos={total_reward, step_rewards, num_turns, turn, trajectory_uuid, images:[path]}`; `check_finished` on env done / `max_turns` / `finish_reason=="length"`. Registrations: `envs['browser_env']`, `multi_turns['browser_scheduler']`, ORM `orms['action_format']` (regex format guard, weight 0.2).

### 3.2 `rl_config.yaml` → `cua/rl/train.py` flag maps (mirrors `config.yaml` / `train_model.flag_map`)
Keys: `model: /workspace/merged`, `sft_adapter: output`, `template: qwen3_5`, lora r64/α128 all-linear, `freeze_vit`, `max_pixels 1003520`; rollout `rollout_port 8000`, `vllm_gpu_memory_utilization 0.35`, `vllm_max_model_len 4096`, `vllm_max_lora_rank 64`, `vllm_limit_mm_per_prompt {"image":1}`, `max_turns 15`; trainer `dataset rl_tasks.jsonl`, `learning_rate 1e-5`, `beta 0.01`, `epsilon 0.2`, `num_generations 4`, `per_device_train_batch_size 1`, `gradient_accumulation_steps 16`, `steps_per_generation 16`, `max_length 3072`, `max_completion_length 512`, `temperature 0.7`, `top_p 0.95`, `max_steps 30`, `save_steps 5`, `async_generate true`, `log_completions true`; plugin `external_plugins cua/rl/trainer/swift_plugin.py`, `gym_env browser_env`, `multi_turn_scheduler browser_scheduler`, `reward_funcs action_format`, `reward_weights "0.2 1.0"`, `env_url http://127.0.0.1:8900`, `env_retry_s 180`; `compute: {min_vram_gb 80, disk_gb 120, image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel, max_runtime_hours 8, eval_before true, eval_after true, serve_port 8001, env_port 8900}`.
- `cua/rl/train.py` subcommands: `print-commands` (Mac smoke), `make-dataset` (one `{"messages":[user placeholder], "env_config":{name, task_id, sample_idx}}` row per train task), `run` (box driver: merge → `EVAL_BEFORE_READY` phase with `swift deploy` → `TRAINING` = `swift rollout` background + `swift rlhf` foreground, copy newest `checkpoint-*` to `rl_output/latest` on every save → `EVAL_AFTER_READY` → `versions.json`), `--mode fallback`.
- `requirements-rl.txt` pins the ms-swift-recommended column: `torch==2.11.0 vllm==0.23.0 transformers==5.12.1 ms-swift[llm]==4.5.2 peft<0.21 trl<1.0 qwen-vl-utils flash-linear-attention datasets<4.8.5`.

### 3.3 `remote_train` integration
- `GrpoJob(RemoteJob)` + `JOBS["grpo"]`; `ServeJob` from Phase 2. `preflight` applies `rl_config.compute` onto the run config, requires `sft_adapter/adapter_config.json` + `rl_tasks.jsonl`, checks `GET env_url/health` (optionally autostarts `cua.rl.service`), writes `rl_runs/<run_id>/run_header.json` (adapter sha, base id, config sha, suite sha, git HEAD, offer, versions, eval before/after).
- `cua/cloud/vast/tunnel.py`: `SshTunnel(ssh_base, forwards=[("R", 8900), ("L", 8001)], keepalive)` with a supervisor thread (autossh semantics), `NullTunnel` for mock. New optional `ComputeProvider.tunnel()` (default null) implemented in `VastAIProvider` via `_ssh_base`; new `RemoteJob.resources()` context entered after `wait_ready`, exited in the same `finally` **before** `_guaranteed_destroy` (stop Mac workers → close tunnel → destroy). New optional `RemoteJob.on_poll()` (try/except, never bypasses teardown): handles the `PHASE` handshake (runs `cua.rl.eval` against :8001, acks) and rsyncs `rl_output/latest` every N polls so a killed run keeps the newest adapter.
- `__main__.py` `grpo` + `serve` subparsers; `run --smoke` extended to `grpo` with mock provider + `mock://` env (asserts teardown on happy/exception/KeyboardInterrupt, both swift commands present, phase handshake). Makefile: `remote-grpo`, `remote-serve`, `test-grpo`, `grpo-dataset`.
- Sizing/cost: 1×80 GB (trainer ~30 GB + vLLM at 0.35 ≈ 28 GB); fallback 2×48 GB with per-process `CUDA_VISIBLE_DEVICES`. ≈ 8–10 min/step (Mac rollout-bound, 16 episodes at ~6 concurrency) → 30-step run ≈ 6 GPU-h ≈ $7 at ~$1.1/h; hard stops = `max_runtime_hours 8`, `max_steps`, sentinel `FAILED`.

### 3.4 Fallback trainer (`--mode fallback`, only if the multi-record gym path fails on the box)
`vllm serve /workspace/merged --enable-lora --lora-modules policy=rl_output/latest` with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`; Mac `cua.rl.rollout` stores `response_token_ids` + `old_logprobs` per turn and group-normalized advantages in `batch.jsonl`; box `cua/rl/trainer/train_grpo_offline.py` (transformers + peft, ~200 lines): teacher-force stored ids, clipped surrogate + `β·KL` vs `disable_adapter()`, save adapter → `POST /v1/load_lora_adapter`. Same tunnel, lifecycle, phases, collection.

## Deferred (explicitly out of v1)
Guarded lane in rollouts (watchdog `ApprovalPolicy` gate, `handoff_login` session injection via a driver `resume_task`, confirm-precision/handoff-recall metrics; needs a lane signal in the student input → SFT re-run); `repo_app` backend beyond the stub; direct published-port driver transport (measure docker-exec latency first); dynamic sampling; ScreenSpot regression via `EvalJob` after each RL run (already possible, just run it).

## Verification ladder (cheapest first)
1. `make test-rl-unit` — decoder round-trips, checks, reward composition, task-suite disjointness.
2. `make test-driver` (extended) and `make test-rl` — Docker only: PNG-parity + SFT-input parity assertions, service round-trip.
3. `python -m cua.rl.tasks check` — suite disjoint from `trajectories/` and `public_sites.json`; sha printed.
4. `make test-grpo` — mock provider + `mock://` env: teardown on 3 exit paths, commands, phase handshake, plugin `core` self-test.
5. Phase 2: `make remote-train` → adapter; `make remote-serve` + `make rl-eval` for base and SFT (real 9B behind the tunnel, verifier live, parse-rate measured; also 1 deliberate `rate_limit_tracker.threshold=0` run to exercise deferred rewards).
6. ~$0.30 tiny box: smallest Qwen3.5 VL model, `mock://` env, `num_generations 2`, `max_steps 2` — pins resolve, vLLM+LoRA loads the family, plugin loads, `-R` tunnel works, one multi-record GRPO step, adapter rsynced, teardown verified.
7. ≤1 h 9B box checklist: merge → rollout loads merged model (LoRA target warning check); turn-1 prompt diffed against an SFT row (template/`<think>` prefix, token count); one episode via tunnel with `mock://` then the real env; one optimizer step; `swift deploy --adapters` answers with image; tunnel kill mid-episode → `env_error`, run continues; Ctrl-C → instance gone.
8. First real run (~$7): 4 tasks × G4, `max_steps 30`, eval before/after on ~20 held-out tasks; success = eval delta ≥ 0, parse-failure rate not up, reward curve rising, adapter collected.

## VALIDATE-ON-BOX (assumptions from ms-swift/vLLM docs; each is one checklist line in step 7)
V1 vLLM 0.23 LoRA covers the `all-linear` target set on Qwen3.5's GDN modules (else restrict `target_modules`); V2 image token count at `MAX_PIXELS` fits `max_model_len`; V3 `qwen3_5` template does not double the `<think>` prefix vs SFT targets (`--response_prefix ''` / tolerant decoder); V4 same template name as SFT; V5 `swift rollout --host 127.0.0.1`; V6 `List[RolloutOutput]` multi-record in server+async mode grouped by prompt id; V7 `rollout_infos.images` as path survives the JSON hop (else data-URI); V8 `--vllm_server_pass_dataset` delivers `env_config`; V9 `num_generations` = G distinct uuids per row; V10 `async_generate` with gym server mode (else serial); V11 dataset loader accepts extra columns; V12 vLLM chat `return_tokens_as_token_ids` (fallback only); V13 Vast sshd allows `-R`; V14 `swift deploy --adapters` applies LoRA under vLLM; V15 `reward_weights` ordering (gym column last); V16 `flash-linear-attention`/`causal-conv1d` needed for Qwen3.5 training speed.

## Risks
- **Verifier throughput** (shared 5 h Claude window) bounds step time → `success_check` on train tasks, `verifier_mode fallback`, deferred-reward path, `max_turns 15`.
- **Live-site load/bans**: rollouts hit sites harder than capture → per-site `throttle`, `max_episodes_per_hour`, idempotent-only train split; validate on 3–4 login-free sites first.
- **Format collapse / KL**: D1 anchor + `beta 0.01`, `action_format` ORM, parse-rate abort threshold 0.8.
- **Template/coordinate drift** between SFT and RL: V3/V4 are mandatory gates; format reward rejects out-of-viewport coords; eval scores each model in its own `coord_space`.
- **Framework API drift**: pinned `requirements-rl.txt` + image; `print-commands` fails fast; fallback trainer is `--mode fallback`, not a redesign.
- **`type` semantics** (select-all-then-type) on rich-text editors: confirm on one live editor; per-session `type_mode` knob.
