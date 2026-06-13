# Task: Build the student eval harness (E3) — measure whether fine-tuned Qwen actually got better at computer use

## Context

This repo distills computer-use ability into Qwen3.5-9B via SFT on verified browser trajectories. The data side is done: an agent-capture funnel (`repo_pipeline/`) produces verified trajectories, `train_qwen.py convert` emits per-step SFT examples, and `make train` runs LoRA fine-tuning via ms-swift. **What does not exist is any way to measure whether a trained checkpoint improved** — no held-out task suite, no runner that lets the student act, no success metric. Your job is to build that harness. Without it, every training run produces a checkpoint and a shrug.

Read `CLAUDE.md` first. Work on a branch off `feat/agent-capture` (or `main` if it's merged), commit incrementally, and follow the repo conventions: minimal comments, no nested imports, generic/config-driven design over special cases.

## What already exists — reuse, don't rebuild

- **In-container browser driver** (`repo_pipeline/instrumentation/browser_driver.js` + host wrapper `repo_pipeline/driver_session.py`): boots in a Docker sandbox, holds a 1280x720 Playwright page, executes actions, records trajectory steps. The student's emitted actions map 1:1 onto its `/act` kinds (`click_xy` for coordinate clicks, `press`, `scroll`, `drag_path`, `type` needs thought — see below). `make test-driver` is its no-LLM smoke test.
- **Sandbox lifecycle** (`repo_pipeline/sandbox.py`): `DockerSandbox` self-expires + auto-removes; `install_sigterm_handler()` for clean shutdown. App boot for repo apps: `driver_session.start_app`.
- **Verifier** (`repo_pipeline/agent_capture.py: verify_trajectory`): judges a trajectory dir (final + sampled mid screenshots vs the goal, efficiency grading). Reuse it as the eval judge — a Claude call via the existing `run_agent` chokepoint. IMPORTANT: all Claude calls authenticate via the logged-in `claude` CLI (OAuth) — never set ANTHROPIC_API_KEY; `sdk_runner.harden_options` (applied automatically) strips operator plugins from headless calls.
- **Eval task sources**:
  - `repo_pipeline/public_sites.json` — 8 public sites, 31 tasks. The trained corpus used these tasks, so for eval you must author NEW held-out tasks (same sites are acceptable for a "seen-site/unseen-task" split; also add 2–3 unseen sites for the strict split).
  - The group-wise val split in `train_qwen.py convert` prints held-out app groups (`Validation groups: ...`) — repos in val never trained; their `proposed_tasks.json` files contain ready-made unseen tasks for boot-able apps.
- **The student's exact I/O contract** (`train_qwen.py`): system prompt `SYSTEM_PROMPT` (defines the action space), user message = one screenshot + `Task: ...` + `Previous actions: [...]` (windowed, `HISTORY_WINDOW=10`) + `What action should you take?`; assistant output = `<think>...</think>` then one-or-more `action: ...` lines. Coordinates are absolute pixels in `<ref>(x, y)</ref>` on a 1280x720 screenshot. Terminal action is `action: done()`.

## What to build

`repo_pipeline/student_eval.py` (+ `eval_tasks.json` config + `make eval` target):

1. **Student client**: talk to an OpenAI-compatible chat endpoint (`--endpoint`, e.g. vLLM or `swift deploy` serving the LoRA checkpoint; also accept `--model`). Send the system prompt + per-step user message exactly as the training format (image as base64). Keep it provider-agnostic — config, not special cases.
2. **Action parser**: parse the assistant's `action: ...` lines back into driver calls. Grammar is exactly what `format_action_for_training` emits — write the parser against those strings and unit-test the round trip for every action type (`click`, `double_click`, `move`, `type`, `keypress`, `scroll`, `drag`, `wait`, `done`). Malformed output = a recorded `parse_failure` for that step (count it as a metric, abort the episode after 3 consecutive).
3. **Episode runner**: per task — boot the target (public site: just the driver; repo app: clone + `start_app`), open a driver session, then loop: screenshot (PNG from the recorded step path or the driver's `/screenshot` — note that endpoint returns JPEG for size; for the student input either is fine, stay consistent with training which used PNG) → student → parse → execute via `DriverClient.act` (use `click_xy`/`drag_path` since the student emits coordinates, `thought` = the student's `<think>` text) → append to history → until `done()`, step cap (default 40), or 3 consecutive failures.
4. **Judging**: after each episode, run `verify_trajectory` on the recorded dir. Episode score = verified success (bool) + efficiency grade + steps used.
5. **Metrics + report**: per split (seen-site/unseen-task, unseen-site, held-out-app) report: verified success rate, parse-failure rate, mean steps on successes, `done()` emission rate (did it stop at all), per-domain breakdown. Output a markdown report + JSON. Support `--baseline` runs (the un-finetuned model through the same harness) so every number is a delta, not an absolute.
6. **Eval task set** (`repo_pipeline/eval_tasks.json`): ~20–30 NEW tasks, never trained on: a held-out wave on the existing public sites + 2–3 new sites + tasks from val-group repo apps. Mark each with its split. Keep tasks non-destructive and final-state-verifiable (the verifier judges screenshots).

## Constraints and pitfalls (paid for in blood this session)

- Eval episodes write trajectory dirs — put them under `eval_runs/<run_id>/`, NOT `trajectories/` (they must never enter training; `convert` globs `trajectories/**`).
- The student emits multi-action steps (`click` + `type` + `keypress` in one turn) — execute all lines in order before the next screenshot.
- Verifier calls go through the rate-limit tracker; a saturated 5h Claude window makes calls wait minutes (time `claude -p pong` before debugging "hangs").
- Containers: always through `DockerSandbox` (self-expiring); never leave manual containers.
- Don't auto-retry blown per-call budgets; surface them.
- Determinism: temperature 0 for the student; log the checkpoint hash + dataset commit into the report header.

## Verification (do all before calling it done)

1. Unit: action-parser round-trip against `format_action_for_training` outputs for all 9 action types.
2. Dry run: `--student mock` mode that replays a recorded trajectory's actions — proves runner+driver+judging end-to-end with zero model calls; episode must verify successfully.
3. Live smoke: 2–3 eval tasks against ANY OpenAI-compatible model (even a small one — it will fail the tasks; you're verifying the harness, not the model): parse failures counted, episodes terminate, report renders.
4. `make eval` target documented in CLAUDE.md; report lands in `eval_runs/<run_id>/report.md`.

## Definition of success

A single command that takes a checkpoint endpoint and produces "verified task success: X% (baseline Y%) across N held-out tasks, parse-failure rate Z%" — trustworthy enough to decide whether the next training run helped, ran entirely on existing machinery, and touching nothing in the training data path.
