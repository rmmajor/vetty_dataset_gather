# Architecture

## Prompt format and corpus gates

Why the code in `cua/prompt/`, `cua/trajectory/` and `cua/sft/` is shaped the way it is. The
modules themselves carry only one-line comments; the reasoning, the measurements and the failures
that motivated each rule live here.

### One example per step, not one per trajectory

`cua.sft.convert.convert_trajectory_to_examples` emits **one training example per step**: the
current screenshot, a downscaled thumbnail of the previous screen, the instruction, a windowed text
history, and the `<think>` + action lines as the target.

A whole trajectory as one multi-turn example packs every screenshot (~300 visual tokens each) into
a single sequence — a 21-step trajectory is ~9k tokens against `max_length: 4096`, so the trainer
silently truncated exactly the long-horizon tails this dataset exists for. Per-step examples also
match inference conditions (one screenshot plus a text history) and supervise every step uniformly.
A two-image step is ~2,020 tokens, of which ~1,100 are image pad.

### The windowed history and its effect annotations

`HISTORY_WINDOW = 20` (`cua.prompt.format`). Long trajectories would otherwise bloat the prompt, and
at inference the runtime keeps a bounded history too. 20 covers the corpus's median 24-step auth
trajectories; the earlier value of 10 truncated 56.5% of steps.

Each history entry is annotated with what the NEXT screenshot and URL showed the action did
(`cua.prompt.effects`): ` [no change]` when the screen stayed pixel-identical within
`EFFECT_NO_CHANGE_MAX_FRACTION`, and ` -> /path` when the URL path changed. The student is told
about both annotations in `SYSTEM_PROMPT`, so the two must always be changed together.

### Steps that are dropped as TARGETS but kept in history

Every filter in `cua.sft.corpus_filters` removes a supervision target only. The step still flows
into `action_history`, so later steps' context — and their post-retry screenshots — stay coherent,
and the trajectory keeps all its steps on disk.

* **Waste clicks** (`--exclude-waste-steps`). A single-click step whose target lands within
  `WASTE_CLICK_RADIUS_PX` of a click in the previous `WASTE_LOOKBACK_STEPS` is a retry/backtrack —
  the "click didn't register, click again" pattern that survives the stuck-loop gate (which only
  drops >= 3 IDENTICAL consecutive actions). Only PURE single-click steps qualify: a step that also
  types, scrolls or keypresses does real work and is never waste, and scrolls (which legitimately
  repeat x,y with different deltas) are never considered. Geometry-based, like the loop and viewport
  gates, never thought-text.
* **Meandering subsample** (`--meandering-mode subsample`). The head keeps the setup, the tail keeps
  the goal approach and the `done()`, and the wander in between supervises every Nth step only — so
  their 40%+ step mass stops dominating the loss while the stop signal stays intact.
  `MEANDER_MID_KEEP_EVERY = 6` lands the meandering EXAMPLE share at roughly their TRAJECTORY share
  (~23%): represented, not dominant. Efficiency is a quality tag, not a gate — `success` already
  means the goal was met.
* **Absurd keypress chains.** A keypress action repeating one key beyond `KEYPRESS_MAX_REPEAT`
  (10-30-key ArrowDown chains) is dropped as a target: the post-chain screenshot no longer matches a
  sensible per-press supervision signal.
* **Synthesized `done`.** Agent trajectories recorded before the driver emitted `done` steps get a
  terminal `{type: "done"}` step appended when the teacher honestly reported completion and the final
  screenshot exists, with the teacher's own completion summary as its thought. Without a terminal
  step the model is never taught to STOP — it would act forever at inference.

### The split unit is the app, never the step

`cua.sft.convert.write_splits` shuffles and splits whole GROUPS (`repo_<owner>_<repo>` /
`site_<name>`, falling back to the task directory). Steps of one trajectory — and trajectories of one
app — share screenshots and layout, so splitting below app level leaks pixels into validation and
turns val loss into a memorization check.

### The grounding slice

Recorded clicks land on a named element's bbox centre, which makes a free ScreenSpot-style grounding
set on our own distribution (`cua.sft.grounding`). A click step qualifies when its `capture_meta`
element carries a role in `GROUNDING_ROLE_NOUNS`, a short name, a bbox inside the viewport, and the
recorded click is within `GROUNDING_CENTRE_TOLERANCE_PX` of that bbox's centre. Masked-password
placeholders, password labels and `<secret:` placeholders are never targets.

Sampling: at most `GROUNDING_DEDUPE_MAX` targets per (name, 8px cell) — one button clicked in ten
trajectories is one element; small (<= `GROUNDING_SMALL_TARGET_PX`) targets are oversampled with
rephrasings; a per-group cap keeps one busy app from owning the slice; and the ceiling is a share of
AGENT examples (`--grounding-share`, default 0.25). Under a cap the survival order is small-target
primaries, then their rephrasings, with large targets dropped first. Grounding examples join their
site's split group so validation stays leak-free.

### The two fail-closed gates

`cua.sft.health` refuses to write a dataset rather than shipping a bad one.

* **Secret / PII leak (exit 1).** Masked credentials are recorded as `<secret:NAME>` placeholders by
  the driver, so any literal high-entropy key or token in the corpus is a real leak. The patterns are
  deliberately scoped to strings that CANNOT be legitimate page content — the gate must never
  false-positive on normal UI text. Credit-card numbers are deliberately NOT matched: we never type
  real cards (payments are denied or handed off), so the only card-shaped strings in the corpus are
  page content (year dropdowns, UUIDs) and public test cards (4111…/4242…) used in test-env checkout
  tasks — all false positives. Arbitrary leaked passwords and usernames cannot be pattern-matched
  without false positives; the structural guarantee (the driver records placeholders, asserted by
  `driver_smoke`) is their primary defense. Operator identities come from `cua.pipeline.operator_pii`.
* **Corpus health (exit 2).** Thresholds in `corpus_health.json`: the regressions that produced
  sft-v1's gaps refuse to ship again. `--skip-health-gate` reports violations and writes anyway, for
  experiments only.

The corpus report printed before those gates is the at-a-glance health check that catches a
Booking.com-style regression at convert time instead of in a JSON file weeks later.

### Trainability is one shared function

`cua.pipeline.corpus_audit.trainability` is the ONE gate, used by convert, `make audit-corpus` and
every corpus pass, so they cannot drift. A trajectory being written by a concurrent capture parses as
truncated JSON; convert skips it rather than letting one half-written file kill a multi-hour training
run after the GPU is already up.

### Coordinates

`trajectory.json` stays in recorded PIXELS at a fixed 1280x720 viewport (`cua.trajectory.VIEWPORT`).
The conversion to the model's `normalized_1000` space lives only in `cua.prompt.coords`, applied when
an action is rendered into a prompt or a training target. Scroll deltas are pixel deltas and are
never converted.

### The SFT launcher

`cua.sft.swift_launch.train_model` maps config keys onto `swift sft` flags. Two behaviours worth
knowing: convert writes `sft_val.jsonl` beside the train file and the launcher picks it up
automatically — without that it was generated and discarded, so no `eval_loss` was ever computed and
overfitting was invisible; and when no validation split exists, `eval_strategy` is dropped rather
than left to make the trainer raise at startup. `max_pixels` is exported as the `MAX_PIXELS`
environment variable instead of being passed as a flag.

## Configuration

Every environment variable the package reads is resolved once, at import of `cua/settings.py`,
into frozen section objects (`SANDBOX`, `PIPELINE`, `CAPTURE`, `SDK`, `PROMPT`, `CORPUS`,
`GROUNDING`, `EVAL`, `CLOUD`, `RL`, `NOTIFY`, `GITHUB`). `docs/conventions.md` lists every name,
default and consumer, generated from the field metadata by `python -m cua.settings --table`. The
knobs whose *values* were chosen for a reason are explained below; the modules themselves now carry
only the reference, not the argument.

### Sandbox resource caps

With `workers=2`, peak demand is roughly `2 * (SANDBOX_MEM_LIMIT + SANDBOX_TMP_SIZE)`, which must
stay under the Docker VM's memory; the defaults target ~6g on an 8g VM. `SANDBOX_MEMSWAP_LIMIT`
defaults to `SANDBOX_MEM_LIMIT` so swap is disabled and the cap is hard — a runaway test is
OOM-killed inside the container instead of spilling into host RAM. The tmpfs is RAM-backed, hence
capped too. `SANDBOX_VM_RESERVE` is the headroom left for the Docker daemon and any co-running
containers when `safe_parallel_sandboxes` auto-sizes concurrency; raise it when a heavy
non-pipeline stack shares the VM.

### Funnel gates

`STAGE3_GATE_*` are per-dimension minimums on a 1-5 scale, env-overridable so they can be tuned
from the dashboard's score distributions without a code edit. `runnability` defaults to 2 (reject
only score-1 "requires paid external services"): a 2 means "needs some setup", and the sandbox —
which has outbound network — gets to actually try it at stage 5 rather than being killed on a
shallow guess. `STAGE4_TOP_K_PER_PASS` is an admission cap, not a rejection: stage 3 passes plenty
of mediocre apps, `get_repos_for_stage(4)` orders by UI richness, so capping each pass to the top K
spends the analysis budget on the richest candidates and lets toy apps pool at `scored`.
`STAGE4_DEEP_SCORE_THRESHOLD` gates promotion to stage 5 on a 0-10 score. The fit vetoes
(`STAGE4_MIN_EST_TRAJECTORIES`, `STAGE4_TOY_COMPLEXITY_MAX`) exist to reject toys cheaply at stage
4, before a stage-5 boot or a stage-6 capture is wasted discovering that the "app" was a unit-test
suite, a boilerplate or a login wall; set a floor to 0 to disable that rule.
`BOOT_GATE_MIN_ELEMENTS` is the point below which a booted homepage (blank page, build splash, bare
error page) offers an agent nothing to do.

The stage-2 bootable gate rejects test-suite and non-app repos before any LLM stage. It is on by
default; `BOOTABLE_GATE=0` falls back to the old metadata-only filter and a probe error never
rejects (the repo stays `discovered` to retry). It is conservative: only hard rejects drop here and
ambiguous "no framework" repos pass to the LLM unless `BOOTABLE_GATE_AGGRESSIVE=1`.

### Capture mode and auth flow

`PIPELINE_CAPTURE_MODE` chooses how stages 4-6 produce trajectories: `agent` (a Claude teacher
pursues proposed user goals in the repo's booted app, the browser driver records coordinates,
framework-independent) or `replay` (the legacy instrumented-test-suite path). It is the one setting
a CLI flag rebinds at runtime (`schemas.CAPTURE_MODE`), so every stage reads it through the module
attribute. `PIPELINE_AUTH_FLOW_CAPTURE` wires the shared GreenMail+Roundcube stack into each
capture sandbox and lets the proposer emit registration / email-verification / password-reset /
email-OTP tasks the teacher completes end to end in the browser; it is off by default so ordinary
repo capture needs no mail stack. `PIPELINE_AUTH_DISCOVERED_RATIO` is the fraction of
account-needing tasks captured as "discovered" login (start signed out with credentials available,
sign in only when the app blocks progress) instead of `pre_session`, keeping login an organic
minority skill rather than a forced prefix on every trajectory.

### Per-container and per-task time

`AGENT_CONTAINER_TIMEOUT` (and its eval twin `EVAL_CONTAINER_TIMEOUT`) are far above the 30-minute
container default because one container hosts a whole repo's agent session — app boot plus N
teacher tasks — and teacher turns stretch enormously under subscription rate-limit waits.
`AGENT_TASK_TIME_RESERVE` / `EVAL_EPISODE_TIME_RESERVE` stop a new task starting with less than
that much container time left, because a task cut off mid-flight records an incoherent tail.
`AGENT_MAX_KEPT_PER_REPO` is a diversity guard so one prolific app cannot dominate the corpus.
`AGENT_MAX_CONSECUTIVE_FAILS` aborts a repo whose backend never came up or whose auth wall blocks
every flow, instead of grinding through its whole task list minutes per doomed task; a productive
task resets the counter. `CAPTURE_MAX_ATTEMPTS` bounds re-capture of failed trajectories so a
permanently blocked task cannot burn a sandbox forever.

### LLM budgets

Every Claude Agent SDK `query()` carries a fixed overhead (~$0.08 for Sonnet-class models), so the
per-call budgets sit comfortably above that floor rather than being tuned to token cost; tune one
down only after verifying a smaller cap does not trip `error_max_budget_usd`.
`THINK_SESSION_BUDGET_USD` covers a whole `ClaudeSDKClient` session (all steps of one trajectory),
not a single call, and is generous on purpose: it guards against a runaway session, not per-step
cost. `QA_VOICE_BUDGET_USD` is higher than the rewrite budget because the qa_verify prompt carries
the run's thoughts. The `OUTCOME_*` price knobs are estimate inputs only — image tokens per
downscaled screen on the judge model, list prices per 1M tokens, and the SDK per-call floor.

### SDK and driver

`SDK_QUERY_TIMEOUT` is a hard wall-clock cap on a single SDK query: the SDK can spawn a CLI
subprocess that wedges (blocks in `waitpid` forever, producing no `ResultMessage`), and without the
cap one such wedge stalls an entire stage run — a verify call once hung a stage-6 run for 2h. The
default comfortably exceeds the longest legitimate call (a multi-turn teacher session, ~25 min
observed) while staying finite. `CLAUDE_CLI_PATH` overrides the CLI the SDK spawns; unset, the
system `claude` is preferred because the SDK's bundled CLI can lag the system install and has
wedged on spawn, and `None` falls back to the SDK's own discovery.
`DRIVER_PLAYWRIGHT_CORE_VERSION` is pinned to the sandbox image's Playwright base (the browsers in
`/ms-playwright` are that revision); 1.55 (>= 1.51) is what lets `storageState({indexedDB: true})`
carry IndexedDB-based auth for SPA apps whose session cookies alone do not authenticate a reused
context.

### Eval and RL waits

`EVAL_MAX_UNPRODUCTIVE_STEPS` is the number of steps in a row that execute nothing (refused,
errored or unparseable) before an episode is abandoned; the driver's own loop guard is off for
eval, so this is what stops a miscalibrated model burning a full step cap on refusals.
`EVAL_COORD_SPACE_SUSPECT_RATE` exists because a calibrated `viewport_abs` model essentially never
lands outside the viewport: crossing that share of emitted coordinates means the run measured the
wrong coordinate space, not the model. `RL_LOOP_GUARD_REPEATS` defaults to 5, not the capture-time
3, which ended two thirds of RL episodes; 5 lets a recoverable retry survive while a true loop
still terminates. `RL_RESET_TIMEOUT_S` sits below the ingress's ~100 s origin timeout (Cloudflare),
since a longer server-side wait is cut off by the tunnel before the 503 reaches the client.
`RL_STEP_WAIT_S` is generous because a terminal step can wait on the verifier through a usage-limit
pause and the retries are idempotent, so waiting beats a fabricated reward. The policy knobs are
sized for a scale-to-zero served endpoint: it drops connections and cold-starts, the request
carries no side effect so transient failures are retried, and `RL_POLICY_WAIT_S` is a deadline, not
a retry count. `CUA_GRPO_GPU` defaults to two GPUs because ms-swift server mode syncs weights over
NCCL, which cannot host the trainer and the rollout engine on one device.
`CUA_TRAIN_COMMIT_EVERY_S` exists because without periodic commits the per-epoch checkpoints only
reach the volume when the whole run ends, so a crash costs every epoch and an earlier epoch cannot
be served while training continues.

### Identity and mail

`PERSONA_EMAIL_DOMAIN` and `PERSONA_PHONE_PREFIX` default to RFC 2606's reserved `example.com` and
NANPA's reserved 555-01xx range: neither can reach a real person.
