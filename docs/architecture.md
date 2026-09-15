# Architecture

A data-gathering, training and evaluation pipeline that distils frontier computer-use (web
browsing) ability into Qwen 3.5-9B by step-level supervised fine-tuning. Three trajectory
**producers** write one trajectory format, and a single converter/trainer (`cua/sft/`) consumes all
of them. A GRPO lane exists but was dropped after three runs.

The deliberate data philosophy is **real third-party sites over self-hosted seeded clones**
(`docs/decisions/real-sites-vs-controlled-backends.md`): the target is deployment transfer, not
benchmark reproducibility.

This document explains how the code is shaped and why. Commands live in `README.md`; the style and
process rules live in `docs/conventions.md`; the two evaluation harnesses live in
`docs/evaluation.md`. Dated records under `docs/decisions/` were written before the package
refactor and still name the pre-refactor module layout; their reasoning stands, their paths do not.

## The three trajectory producers

- **Path A — frontier CUA distillation** (`cua/path_a/recorder.py`). Drives GPT-5.4's `computer`
  tool through Mind2Web tasks in a Playwright browser, ~$0.55 per trajectory. Contributes a handful
  of trainable trajectories today. That tool is screenshot-only, so element semantics stay null on
  every step while url, screenshot and timing signals are still recorded — which is why the
  dashboard's Path A views show a different set of fields from Path B's.
- **Path B — the repo funnel** (`cua/pipeline/orchestrator.py`, stages 1-7). Finds bootable web-app
  repositories on GitHub, boots them in Docker sandboxes, and has a Claude teacher pursue proposed
  user tasks through a coordinate-recording browser driver. Only a fraction of a percent of
  discovered repositories yield trajectories, so this lane is a minority of the corpus.
- **Public-site capture lanes** (`cua/pipeline/lanes/`, all sharing `lanes/capture_lane.py`). Point
  the same teacher, driver and verifier at curated registries of real public sites: login-free,
  search goals, login-gated behind burner accounts, and human-provisioned. This is the bulk of the
  trainable corpus, roughly a third of it behind an authentication wall.

The trajectory schema is the integration contract tying producers, converter, evaluation and RL
together. Read `cua/trajectory/schema.py` and `cua/pipeline/schemas.py` before changing any producer
or consumer.

## The trajectory format (the integration contract)

Every trajectory is a directory `task_*/` — Path A writes `task_mind2web_NNNN/`, Path B writes
`repo_<owner>_<repo>/task_*/`, the public lanes write `site_<name>/task_*/` — containing
`trajectory.json`, `metadata.json`, an optional `capture_meta.json` (the raw verifier verdict plus
the recorded click targets), and `step_NNN_screenshot.png` files.

```json
{"task_id", "instruction", "start_url",
 "steps": [{"step", "screenshot", "thought", "actions": [...]}],
 "total_steps", "success"}
```

Action dicts are normalised across every producer: `{type:"click", x, y, button}`,
`{type:"type", text}`, `{type:"keypress", keys:[...]}`, `{type:"scroll", x, y, scroll_x, scroll_y}`,
`{type:"drag", path:[{x,y}...]}`, `{type:"move", x, y}` (hover), `{type:"navigate", url}`,
`{type:"wait"}`, the guarded-lane controls `{type:"confirm"}` / `{type:"handoff"}` /
`{type:"approval_event"}`, and the terminal `{type:"done"}`. The viewport is fixed at
**1280x720, deviceScaleFactor 1** everywhere (`cua.trajectory.VIEWPORT`). Credentials only ever
appear as `<secret:...>` placeholders; the driver substitutes the real value at dispatch time and
never records it.

Two supervision rules are enforced by every producer and every corpus pass, because violating them
is the most damaging failure mode this dataset has:

- `instruction` is a terse user GOAL ("sign me up for an account on X"), never a step-by-step
  procedure. The agent works out the route. Resources — credentials, an inbox — belong in the
  teacher policy, not in the instruction.
- `thought` text describes intent, never a play-by-play of the action being taken.

`cua/trajectory/io.py` is the one reader/writer: `TrajectoryDir.load` lazily reads the three JSON
files, `save` writes them with the exact dump options every earlier writer used, and
`iter_trajectory_dirs` finds every directory layout above. Nothing else opens a `trajectory.json`.
Those dump options are pinned precisely because every corpus writer has always used them, so a pass
that rewrites one file leaves the rest of the corpus byte-comparable; a step recording no screenshot
resolves to the directory itself, which is what every caller already assumed. `cua.trajectory` is
kept stdlib-only so that no consumer of the recording resolution inherits pydantic.

The schema carries three deliberate softnesses. A watchdog verdict
(`{kind: "approval", decision, rationale, by}`) is an observation step, which "Guarded mode and the
safety watchdog" explains. `TrajectoryMetadata`'s two free-form fields are untyped on
purpose, because both come straight from unvalidated LLM JSON and capture must record what it got
rather than reject it — and the login-gated lane's fields (capture policy, session provenance,
terminal risk class, the replay manifest convert ignores) are additive for the same reason. And a
capture must never die on one odd generated value, so when validation fails the writer degrades to
the pre-schema write-through and names the offending field rather than losing the trajectory.

## Repository layout

| package | holds |
| --- | --- |
| `cua/settings.py` | every environment variable the package reads, resolved once at import into frozen section objects |
| `cua/trajectory/` | the trajectory schema, the action grammar (`ACTION_SPECS`), and the one on-disk reader/writer |
| `cua/prompt/` | the observation format: `format.py` (prompts and history), `effects.py` (screen/URL effect annotations), `coords.py` (pixel ↔ `normalized_1000`) |
| `cua/sft/` | convert (`convert.py`, `corpus_filters.py`, `grounding.py`, `health.py`) and train (`recipe.py`, `swift_launch.py`) |
| `cua/path_a/` | the Mind2Web task loader and the frontier-CUA recorder |
| `cua/pipeline/` | the repo funnel, the teacher, the capture lanes, the sandboxes and the corpus passes |
| `cua/student/` | the shared student stack — observation building and action execution — used by both the eval episodes and the RL rollouts |
| `cua/eval/` | `screenspot/` (grounding), `student/` (browser episodes), `task_sources.py` (the eval task registry), `blocks.py` (what counts as a site blocking automation) |
| `cua/rl/` | the GRPO rollout harness and its trainer plugin (frozen, see below) |
| `cua/cloud/` | `modal/` and `vast/`, which orchestrate rented GPUs and hold no training logic |
| `tests/` | the test runner (`python -m tests`: no Docker, LLM or GPU, but it needs the `pipeline` extra) plus the behaviour-regression fixtures |
| `scripts/` | the repo's own convention checks (comments, nested imports, environment reads) |

`docker/` holds the sandbox images, `experiments/` the measurement registry, `eval_archive/` the
committed raw evaluation artefacts, and `constraints/` the resolved GPU dependency column.

## Path B: the repo funnel (`cua/pipeline/`)

A 7-stage quality funnel: cast the widest net, then reject aggressively. `orchestrator.py` is the
entry point; it sets up logging to `pipeline_data/run_logs/`, then dispatches stages from the
`STAGES` registry. Capture (6) and supervision finalize (7) are split so that a slow or flaky LLM
pass can never block or fail a capture.

### Capture modes

`schemas.CAPTURE_MODE` (environment `PIPELINE_CAPTURE_MODE`, CLI `--capture-mode`, default
**`agent`**) switches how stages 4-6 behave. It is the one setting a CLI flag rebinds at runtime, so
every stage reads it through the module attribute rather than caching it.

- **agent** (primary). The repository's app is booted inside the sandbox and a Claude teacher
  (`teacher/`: `policy.py` prompts, `tools.py` tool registries, `run.py` the loop, `verify.py` the
  judge) pursues user tasks proposed by `task_proposer.py`, acting through an in-container
  Playwright HTTP driver (`instrumentation/browser_driver.js` plus the host wrapper
  `driver_session.py`) that records a screenshot and bbox-centre coordinates per action. Every
  action tool requires a `thought` argument, so per-step intent-level supervision is structural
  rather than retrofitted. A post-run verifier (`teacher.verify_trajectory`, described under "The
  teacher and the verifier") decides `success: true`, and salvaging a coherent partial is the
  separate audited `corpus/relabel_recoverable.py` pass, never the capture-time verifier. The lane
  is framework-independent: the repository's own tests are only feature hints for the proposer.
- **replay** (secondary, legacy). Instrumentation is injected into the repository's
  Playwright/Selenium suite and records what the tests do; `think_generator.py` then GENERATES
  thoughts and instructions from screenshots plus test source. Its trajectories train only under
  `--include-replay`.

### The registry state machine is the spine

All repository state lives in `pipeline_data/registry.db` (SQLite, WAL). `schemas.py` defines three
dicts — `STAGE_TO_INPUT_STATUS`, `STAGE_TO_PASSED_STATUS`, `STAGE_TO_REJECTED_STATUS` — mapping each
stage to the `RepoStatus` it consumes, promotes to and rejects to:

```
discovered → filtered → scored → analyzed → validated → captured → completed
                ↓          ↓         ↓          ↓           ↓
          rejected_filter / _score / _analyze / _trial / _generate
                                                  (stage 7 "think": captured → completed, never rejects)
```

Each stage calls `registry.get_repos_for_stage(N)`, and the ordering changes at stage 4.
`_priority_ordered` (stages below 4) sorts by query priority — which deprioritises demo and
tutorial repositories — and then round-robins by domain, so scoring sees a varied spread instead of
the top N of one type. `_drain_ordered` (stage 4 and up) sorts by `deep_score` first, with NULL
read as 0 at stage 4's own input where the richness term then dominates, so drag-and-drop, editor
and data-grid apps reach capture ahead of plain click-and-type ones.

Per-repo work runs through `_processing.py`, and the outcome is signalled by what the stage's
`core_fn(registry, repo)` raises rather than by the stage writing rejections itself: returning
normally means passed (the stage has already set the passed status and may return a trajectory
count), `QualityReject` means the repository genuinely failed a gate and is terminal, and
`RetryableFailure` or any other exception means infrastructure or flakiness, retried
`PIPELINE_MAX_RETRIES` times and then rejected with an infrastructure `FailureCategory` so the row
stays distinguishable and re-queueable. Centralising it gives every stage identical retry
behaviour, single-point registry writes and inline reject/retry logging for free. A retry keeps the
repository's stage-input status and moves only `retry_count`, so a crash mid-stage still selects it
next run.

`FailureCategory` is orthogonal to `RepoStatus`: the status says WHERE a repository died, the
category says WHY, which is what separates a quality decision from an infrastructure or flaky
failure. Both are stored on the row alongside the free-text `filter_reason`, with the gate scores
behind the decision persisted next to them, so the dashboard can split genuine rejections from
recoverable noise and show exactly how many repositories a threshold change would recover.

Stages are **idempotent**: repositories guard on `current_stage >= N`, so re-running never
reprocesses. Re-queue flags cover the four reasons a rejection goes stale — `--requeue-infra`
(the `NON_QUALITY_CATEGORIES`, which are not auto-retried but become recoverable once a config
changes, a higher `max_turns` for instance), `--requeue-stale` (rejections recorded before a stage's
entry in `schemas.STAGE_BEHAVIOR_CUTOFFS`, the per-stage instant at which its behaviour last
changed), `--requeue-stage N --min-score X` (which reads `STAGE_SCORE_COLUMN` to target exactly the
rejections a lowered threshold would now accept), and `--prune-unfit` (re-judging repositories past
stage 4 from their STORED analysis, with no re-clone, so no further stage-5 or stage-6 compute is
spent on a toy; reversible with `--requeue-stage 4 --min-score 0`) — all honouring `--dry-run`,
which counts without writing.

Adding a stage or a status means extending the mapping dicts and the `STAGES` registry, never
hardcoding a stage number in `run_stage`. Two entries carry decisions rather than plumbing:
`STAGE_TO_REJECTED_STATUS` has no entry for stage 7, because a captured repository whose thoughts
are unfinished stays `captured` and is resumed, never thrown away; and `DeepAnalysis.good_fit`
defaults to True so analyses recorded before the fit verdict existed still parse, where True means
only that the holistic layer did not veto — the deterministic `fit_veto` rules still apply on top.
`TrialResults.affordances` carries the boot gate's UI-sophistication profile (draggables, sliders,
canvases, rich-text editors, counted by the driver's `/affordances`), which feeds the task
proposer's quotas and the dashboard's sophistication view.

A crashed stage starves everything downstream, so `orchestrator.log_stage_error` records one
loudly: a stage-4 `NameError` once ran a whole night looking like "0 repos pending". Under
`--workers 10` the stage loggers would otherwise interleave lines from many repositories, so
`logging_utils.repo_logger` prefixes every line with `[stage N][owner/repo]` and `StageProgress`
emits a running `done/total` after each repository, which is what makes a long run legible while it
is still going. Its counters are mutated from per-repo coroutines without locks, which is safe
because asyncio runs them on one thread and no counter update awaits mid-update.

### Stage responsibilities

| stage | module | what it does |
| --- | --- | --- |
| 1 discover | `stage1_discover.py` | GitHub API across the ~50 queries in `github_queries.json`, down to `--star-floor`. No LLM. |
| 2 filter | `stage2_filter.py` | size/recency filters, fork and duplicate-description dedup, then the no-LLM bootable-app gate |
| 3 shallow score | `stage3_shallow_score.py` | fetches repository metadata via `gh`; Claude scores 7 dimensions (`ShallowScores`), gated at `SCORE_THRESHOLD` |
| 4 deep analyze | `stage4_deep_analyze.py` | static analysis inside a `DockerSandbox`; bootability dominates the score |
| 5 trial run | `stage5_trial_run.py` | agent mode: the boot gate. Replay mode: the legacy instrumented test run. |
| 6 capture | `stage6_generate.py` | boots the app, proposes tasks, runs the teacher per task, verifies, dedups, caps per repo → `captured` |
| 7 finalize | `stage7_think.py` | polishes raw teacher thoughts into student voice (agent) or generates them (replay) → `completed` |

The **stage-2 bootable gate** (`bootable_gate.py`) is one `gh` probe of the root tree and
`package.json` against data-driven framework sets. Hard rejects drop to `NOT_BOOTABLE`; ambiguous
"no framework" repositories pass to the LLM unless `BOOTABLE_GATE_AGGRESSIVE=1`. It exists because
an LLM stage is the expensive one, and a test-suite or non-app repository is cheaply recognisable
without it. `classify_bootable` is a pure function so it is testable in isolation, `probe_repo` makes
the one network call through the `gh` CLI (reusing its auth exactly as stage 3's metadata fetch
does), and every signal set is module-level data, so a new framework, config file or compose name is
a data edit. Only two reasons are confident enough to be in `HARD_REJECT_REASONS` — clearly not a
Node app, and clearly a test suite. The ambiguous remainder ("package.json without a framework", "no
bootable-app signal") is a known false-positive class: jQuery, Bootstrap and vanilla web UIs the
probe cannot tell from non-apps, so borderline cases lean ACCEPT and the LLM stages judge quality on
the much smaller survivor set. Monorepo markers are accepted conservatively for the same reason: in
a JS monorepo the bootable app lives in a workspace subdirectory and the root `package.json` often
carries no framework dependency at all.

`cua/pipeline/tools/gate_tools.py` is that gate's offline CLI. `regate` re-probes the `filtered`
and `scored` buckets — the repositories that would otherwise burn LLM scoring and analysis — in
chunks at low concurrency, committing rejects per chunk so progress survives an interruption and a
re-run is idempotent. `validate` is the gate's empirical proof: agent-mode proven-good repositories
set the precision floor and must ALL accept, a sample of LLM-rejected backlog measures recall, and
the saved LLM dollars are projected from both; legacy replay suites rejecting here is the agent-mode
pivot working, and the report shows them for honesty. Both open the registry `mode=ro`, so a report
can never write where `RepoRegistry`'s own constructor would (WAL pragma, schema, migrations), and a
probe error reports `accept=None` rather than being counted as either verdict.

**Stage 3** gates only the signals that actually predict usable web-UI trajectories: it is an app
rather than a library, it can run in the sandbox, and its tests do more than navigate and assert.
`production_complexity` is deliberately NOT gated, because it conflates "real app" with "complex
app" and would reject simple-but-ideal apps such as SauceDemo.

**Stage 4** scores bootability as the dominant axis, because agent capture needs the repository's
own app serving inside the sandbox and a dev server, `webServer` block or docker-compose file is
exactly what `start_app` can boot; a suite that only targets a public URL has nothing for the agent
to drive and is hard-gated, and one needing an external backend that cannot be provided is
penalised. On top of the score it hard-vetoes toys in agent mode: the model's holistic `good_fit`
verdict is the main gate — it is what catches boilerplates, course repositories and login-walled
apps that the raw numbers rate highly — with estimated-trajectory and production-complexity
backstops as a cheap deterministic floor. The veto runs BEFORE the numeric threshold, so an obvious
toy is rejected without a stage-5 boot. The same `fit_veto` predicate guards the stage going forward
and the retroactive `--prune-unfit` cleanup, so a toy is rejected by identical logic whether it is
analysed today or was analysed before the veto existed; every rule reads a threshold and none
hardcodes a repository name. For repositories analysed before the verdict existed, `judge_fit`
renders the same judgment text-only over the stored analysis, with no clone and no sandbox, and a
parse failure answers "fit" so a flaky call never wrongly prunes. Framework support is irrelevant in
agent mode, which never runs the tests; replay mode still gates on `is_supported_profile`.

**Stage 5** in agent mode is a boot gate: clone, install (pnpm or npm), daemonise the app through
`driver_session.start_app` with `app_secrets.boot_env_secrets()` and the mail environment injected,
probe it through the driver, and reject `NO_BOOT` or a trivial UI (`BOOT_GATE_MIN_ELEMENTS`). It
gates on exactly what capture needs and nothing else, so the entire legacy `no_capture` failure
class — instrumentation hooks that never fire — cannot occur here at all. Repositories whose tests
only target public URLs are rejected, because real-site trajectories are the public-site lanes' job.
In replay mode `pass_rate` informs ranking only and is never the gate, because a test that performs
actions and then fails an assertion still produced a valid trajectory; the tail of the test output
is persisted so that "why pass_rate=0" (app needed, install failed, instrumentation broke) stays
diagnosable after the ephemeral container is gone.

**Stage 6** runs a fresh browser context per task against one shared app server, then verifies,
dedups and caps per repository (`AGENT_MAX_KEPT_PER_REPO`, default 10) before enriching metadata.
Dedup and cap run among the VERIFIED trajectories only: identical action shapes are dropped and the
longest survive, because trajectory length is the scarce resource. It is resume-aware at TASK
granularity — `_capture_agent` skips already-recorded tasks and reuses the persisted proposal, where
replay mode resumes only per repository — and partial sessions are copied out even when a
usage-limit abort ends the run. Its auth decisions are deterministic so that a resumed run picks the
same mode: `_use_discovered` splits account-needing tasks between `discovered` (start signed out
but with credentials, sign in only when the app blocks progress — the realistic "hit the wall, then
log in" pattern) and `pre_session` (start already signed in on the captured session), in the
`PIPELINE_AUTH_DISCOVERED_RATIO` proportion, which keeps login from being skewed onto every
trajectory. Only a CLAIMED-complete login mints the reusable session, so a half-finished one cannot
leave later `pre_session` tasks starting broken, and a repository with a `known_account` skips
registration tasks entirely, since they could neither be completed nor be relabelled into something
else.

**Stage 7** is decoupled, resumable and never rejects. Agent trajectories arrive carrying the
teacher's raw tool-aware reasoning in every step and are POLISHED into clean student-voice thoughts
by one text-only call per trajectory — no screenshots are needed, because the reasoning is already
grounded — which flips `metadata.thought_source` from `teacher_raw` to `teacher`; replay
trajectories arrive with empty thoughts and have them GENERATED step by step from the screenshots. A
repository is marked `completed` once every trajectory is finalized, and steps with a missing
screenshot cannot be generated so they do not block completion. If some remain — a transient SDK
stall, a rate limit — the repository simply stays `captured` and is picked up next run, because the
trajectories themselves are already good. It runs LAST in the drain loop, which means the Claude
usage limit starves it: run it alone on a fresh usage window to unstick `captured` repositories.

### Proposing the tasks

`task_proposer.py` is one Claude call per repository, grounded in what the app actually is: the
homepage screenshot and interactive-element snapshot from the live app, the README, and the
repository's test sources as feature hints. Tests document what an app can do, but they are hints
here, never the generator. Its output is the dataset's difficulty and diversity lever, so the prompt
demands genuine user goals rather than QA assertions, skews long-horizon (most tasks needing eight
or more actions), and mixes action kinds with typing and forms mandatory and scroll or drag wherever
the UI affords them — the affordance counts from the driver map to hard quota lines, so a new
sophistication signal is an entry in that table plus a counter in `/affordances`, never new prompt
code. Optional blocks specialise it without forking the prompt: the auth-flow block lifts the "no
email verification" restriction and has the proposer CLASSIFY each task's auth need, so login
appears organically in a minority of trajectories; the search block steers toward the constrained
info-seeking journeys the action-UI corpus lacks; and the known-account block forbids registration,
verification and invite tasks outright, because that app has a working pre-verified login and NO
readable inbox, so those tasks could never be completed. An unparseable response is a retryable
condition rather than a content filter, an unusable app comes back as `usable=False` with an empty
task list so the repository is rejected without spending a teacher call, and `register`/`login`
tasks are sorted FIRST because they mint the account whose captured session the later tasks reuse.

### The teacher and the verifier

`teacher/policy.py` renders the lane's policy block, which states the teacher's RESOURCES and stop
condition and never a procedure, and always closes with the disposable identity. Realistic email
handling IS the skill being captured, so waiting, refreshing and re-requesting a slow disposable
inbox are recorded browser actions rather than harness steps.

`teacher/tools.py` holds the tool registries. An action is refused when the task has ended, the
budget is spent, or the text carries operator identity. The terminal `done` step — a final
screenshot plus the agent's own goal-is-met observation — is the supervision that teaches WHEN to
stop, and it is guarded on done-recording rather than on `completed`, because a watchdog DENY has
already set `completed` and that trajectory still needs its terminal stop supervision. Guarded mode
adds the pause-and-ask and takeover tools; every other autonomy drives straight through, so repo
captures stay byte-for-byte unchanged. `teacher/run.py` captures the authenticated session BEFORE
`finish` closes the context, which is what makes a register or login trajectory's session reusable,
and `finalize_partial` puts the trajectory and final screenshot on disk even when the teacher ran
out of turns or budget without calling `task_complete`.

`teacher/verify.py` is the judge. Recorded success requires the verdict AND every `criteria_met`
entry, so a verdict cannot outrun reasoning that names an unmet criterion. Hindsight
RECLASSIFICATION is forbidden: success is judged ONLY against the original instruction, never
against a sub-goal the agent reached instead, and a goal the verifier confirmed was met is never
overturned by step count — only a degenerate recording, which by construction accomplished nothing,
is, and that degeneracy check is far stricter than the converter's stuck-loop defect check so that a
real run which retried one control four times survives it. The judge sees the final screenshot plus
evenly sampled mid-trajectory frames, so it grades the process and not only the end state. Whether a
register or login task left a reusable authenticated session is a separate question answered from
the verifier's own evidence, not from whether the bundled goal completed.

### Generating thoughts for replay trajectories

`think_generator.py` is the replay-mode half of supervision: the `<think>` blocks it writes teach
the student the inner monologue of an agent executing a GUI task, each thought being first-person,
pre-action reasoning toward the goal and grounded in the screenshot the agent sees. For step N the
model receives the current screenshot, the trajectory's goal and starting URL in the system preamble
so the goal stays salient, the thoughts and actions already produced for steps 1..N-1 so the chain
stays consistent and can refer back to progress, a natural-language description of the action about
to be performed, and a private grounding hint (role, name, bbox of the target) the prompt explicitly
forbids from leaking into the output, because those are mechanical details rather than part of the
monologue. Generation is sequential within a trajectory — step N's prompt depends on the thought
just written for N-1 — and concurrency happens at the trajectory level in the orchestrator. There
are no regex post-filters: a response that does not read like a thought is a reason to tighten the
prompt, not to silently retry, and a genuine SDK failure returns an empty thought that the
dashboard's observability metrics surface. Resume is handled by seeding: the session is given the
full prefix of already-done steps on its first generated turn and only the gaps thereafter, so on a
fresh trajectory the seed is empty after step 1 and later turns stay lean and cache-cheap.

### Sandbox isolation

Repository code executes only inside an ephemeral, capability-dropped Docker container
(`repo-pipeline-sandbox:latest`, built from `docker/Dockerfile.sandbox` on the Playwright base
image, with the driver pre-baked at `/opt/driver`). `DockerSandbox` is a context manager exposing
`exec_command` / `read_file` / `write_file` / `copy_out` / `start_background` / `wait_for_port` /
`http_json`, and enforces a 30-minute container and 5-minute per-command timeout.

"Capability-dropped" means `cap_drop=ALL` plus a narrow `cap_add` that grants back ONLY the
file-ownership and permission capabilities a dependency install needs: npm, pnpm and yarn extract
tarballs preserving ownership (`fchown`) and run postinstall scripts that chmod and chown. Without
them the install dies with `EPERM: operation not permitted, fchown`, `node_modules` stays empty, the
dev binary reads as "not found", and a perfectly good app is scored `NO_BOOT`. The dangerous
capabilities (SYS_ADMIN, NET_ADMIN, SYS_PTRACE, MKNOD) stay dropped, and no-new-privileges,
ephemerality, the resource caps and the absence of host bind-mounts carry the rest of the isolation.

The container is **self-expiring**: its main process sleeps exactly the container timeout and then
exits, and Docker removes it. Host-side `stop()` in a `finally` block is the normal path, but it
never runs when the pipeline process is SIGKILLed or crashes, so the self-expiry is what guarantees
no orphan outlives its timeout even with a dead owner. `install_sigterm_handler` covers the middle
case: `pkill` sends SIGTERM, which by default terminates Python WITHOUT running `finally` blocks and
leaks every live sandbox, so it is converted into `SystemExit` and the normal cleanup paths run on a
graceful restart. Hard kills fall back to the self-expiry.

Two details of the transport are worth knowing because both were found through a failure.
docker-py's `exec_run` has no timeout, so the per-command limit is enforced inside the container
with coreutils `timeout` (SIGTERM at the limit, SIGKILL ten seconds later, exit code 124); without
it a hung command — a stuck browser, typically — blocks until the container's own 30 minutes are up.
And `ShellSandbox.http_json` sends its payload base64-encoded through the exec shell, so arbitrary
text (quotes, newlines, unicode in agent thoughts) survives in a single exec; above
`HTTP_INLINE_MAX` (~100 kB of base64, first hit by a large `pre_session` `storage_state`) it is
staged into the container as a file through the Docker archive API instead, or the kernel's argv
limit rejects the command outright. That staging file MUST live on the overlay filesystem
(`/workspace`), never in `/tmp`: `/tmp` is a tmpfs mount and `put_archive` extracts into the
overlay layer, so a `/tmp` target lands in the overlay's hidden `/tmp`, invisible to the exec'd
curl, which then exits 26 with "error reading file".

`ShellSandbox.start_background` exists because each exec runs in an independent shell, so a plainly
backgrounded process dies with it; `setsid` plus full stdio detachment re-parents the process to
PID 1, where it survives until the sandbox is destroyed. That is what keeps the agent-capture app
server and the browser driver alive across many separate exec calls. `DockerSandbox.list_files`
prunes `node_modules` and `.git` before matching, because an installed repository holds thousands of
vendored `package.json` files that would spend the `head -500` cap and silently hide the project's
own. The sandbox MCP server used by the analysis stages runs each blocking docker-py call in an
executor, so concurrent stage workers do not stall the event loop.

`ShellSandbox` is the transport-agnostic base shared with `remote_sandbox.RemoteSandbox` (ssh and
rsync to a rented box), so the driver behaves identically in a container and over ssh. A backend
supplies only `exec_command`, `put_bytes`, `copy_out`, `start`, `stop` and `remaining_seconds`;
everything else on the class is transport-independent. `RemoteSandbox` deliberately does NOT nest
Docker — a rented box is already a container, so the driver runs natively against the Playwright
image's browsers — and the image ships those browsers without the npm package, so the driver's
`playwright-core` import only resolves after the setup command runs.

That remote transport exists because **egress is an evaluation parameter, not a property of whoever
runs the harness**. Online-Mind2Web is US-centric, and measured from this machine's Ukrainian
residential IP only 58% of its 300 tasks are reachable; the rest are geo- or WAF-blocked before the
model acts at all. `VastResidentialSandbox` adds the rental on top, picking an offer whose public IP
belongs to a consumer ISP, booting it and destroying it on stop with the same teardown invariant as
the training lane. A datacenter exit is blocked about as hard as a foreign one — measured on an M247
VPN endpoint, a datacenter exit recovered 1 of 24 blocked sites where a residential one recovered 9
— and `public_ipaddr` is exposed on a Vast offer BEFORE renting, so the ASN can be checked for free.

Container memory, pids, CPUs and tmpfs are capped by the `SANDBOX_*` knobs, and
`safe_parallel_sandboxes` clamps parallelism so that `workers × (mem + tmpfs)` stays under the
Docker VM's memory while leaving `SANDBOX_VM_RESERVE` for the daemon and any co-running stack.
Exceeding it does not raise: it degrades into a silent fast-then-one-per-hour collapse as the VM
thrashes (3 × 3g on a 7.7g VM is the case that forced the clamp). When the VM's memory cannot be
read the clamp returns the requested value unchanged, so it can never make a working setup worse.
If a run yields zero with "No such image", check `docker images` and rebuild before trusting the
number.

### Getting a drivable browser and a running app into the sandbox

`driver_session.py` owns everything between a bare sandbox and an agent-drivable browser.
`install_driver` writes the driver in and ensures `playwright-core` exists — a no-op against the
baked `/opt/driver` layer of the current image, an npm install pinned to the image's browser
revision on an older one; browsers are never downloaded, because the image already ships them under
`/ms-playwright`. `start_driver` daemonises it and waits for its port, one process per port because
RL runs several per sandbox. `boot_driver` / `boot_driver_sync` are the install + start + client
triple every lane uses, and `DriverClient` is a thin typed wrapper over the driver's JSON-over-HTTP
endpoints. Stage 5's boot-gate probe and stage 6's capture both go through it.

`start_app` boots the repository's own app daemonised, which is the difference from replay mode:
`app_runner.wrap_test_command` scopes the app to a single shell, while agent capture needs it alive
across many independent exec calls. Three decisions inside it account for most boot outcomes.

- **The install runs with the app's OWN package manager in its own directory.** A blind root
  `npm install` is a no-op for a monorepo subdirectory app and outright fails for pnpm, yarn and
  workspace repositories, whose dev binary then never lands on PATH; that is the single dominant
  boot failure, and it surfaces at start as `<tool>: not found` in `APP_LOG`, which is what the boot
  gate reports. pnpm and yarn are installed globally through npm rather than corepack, whose shim
  throws `DefaultVersion` on this image and then shadows any fallback, and the permissive major line
  is pinned (pnpm 8, yarn 1) because pnpm 10+ puts dependency build scripts behind an interactive
  `approve-builds` prompt and gates `pnpm run` on a deps-status check, both fatal unattended.
- **App detection runs with `for_agent=True`.** Agent capture has no Playwright run to honour a
  `webServer` config, so `package.json` start scripts are used even when such a block exists —
  those blocks almost always wrap the same script anyway. The search covers the repository root plus
  monorepo subdirectories (`apps/*`, `packages/*`, `examples/*` to depth 3, excluding
  `node_modules`), and the install itself runs at the workspace ROOT for a monorepo package so the
  app's binaries land on PATH.
- **The dev-server environment serves the app, not developer tooling.** CRA and webpack splash a
  full-screen ESLint wall over the page on lint findings — a CRLF/LF mismatch on checkout is enough
  — which would land in every recorded screenshot, so the lint plugin and overlay are disabled at
  the source (the browser driver additionally scrubs any overlay that still appears). `BROWSER=none`
  stops CRA opening a browser, an empty `CI` keeps warnings from being promoted to errors,
  `HOST=0.0.0.0` covers servers that honour `$HOST`, and `NODE_OPTIONS=--dns-result-order=ipv4first`
  exists because Vite and Next bind the hostname `localhost`, which Node resolves to IPv6 `::1`
  first on this image while the boot probe and the browser driver connect over IPv4 — so a perfectly
  served app reads as `NO_BOOT`.

Port detection is one in-container shell loop parsing `/proc/net/tcp{,6}` rather than a per-second
`docker exec`: the inferred `preferred` port is tried first each round, and any other listening port
answering 2xx/3xx is accepted, so an app on a port nobody guessed (webpack-dev-server's 8080, a
custom server's anything) is found instead of being read as `NO_BOOT`. The driver's own port is
excluded. Because a whole repository's tasks share one long-lived app server that can crash
mid-session, `app_is_up` is a cheap in-container curl probe the capture loop uses to restart the app
before the next task rather than failing every remaining one with `CONNECTION_REFUSED`.

`app_secrets.boot_env_secrets` supplies the dummy auth values (`JWT_SECRET`, `*_SECRET`,
`NEXTAUTH_URL`) most Next.js and Express auth apps read at startup and throw without. In a
disposable single-run sandbox a fixed dummy is safe because it protects nothing real, and it is
fixed rather than random so that a captured `pre_session` cookie stays valid on resume; it is merged
into the app's environment alongside the mail configuration.

The session itself carries the sensitive material. `DriverClient.session` takes `auth_config` (the
`pre_session` `storage_state` the driver injects at context creation), `secrets` (the session-scoped
map the driver substitutes into `<secret:NAME>` placeholders at type time, never recorded) and
`mail_config` (`{url, address, password}`, which pre-authenticates the scenario webmail so the agent
finds a ready inbox when it navigates there). All three are set once at session setup rather than
per `/act`, to keep the secret-transport surface minimal, and all three travel base64-encoded
through `http_json` — never as plaintext on a command line, never logged.

### Instrumentation

`instrumentation/browser_driver.js` is the live agent-mode recorder, speaking JSON over HTTP on
`/observe`, `/act` and `/screenshot`. Its action kinds include bbox-reference actions for the
teacher plus the coordinate-only `move_xy` / `type_text` / `scroll_xy` a student emits;
`DRIVER_LOOP_GUARD_REPEATS` is its stuck-loop guard.

Replay mode writes trajectories from three injected StepRecorders — `pw_python_conftest.py`
(pytest-playwright), `pw_ts_fixture.ts` (`@playwright/test`) and `selenium_python_conftest.py` —
each writing `trajectory.json`, `capture_meta.json` and `metadata.json`. Those three files run
inside the repository under test, where the `cua` package is not importable, which is why they keep
their own marked environment reads instead of going through `cua/settings.py`.

Injection is driven entirely by the resolved `FrameworkProfile` and is shared by stages 5 and 6.
The TypeScript fixture re-exports the whole `@playwright/test` module as a drop-in superset and
overrides the `page` fixture, so a single import swap makes instrumentation apply to every test with
no per-test edits. For Python, pytest only auto-discovers a file named exactly `conftest.py`, so the
recorder is appended to the repository's top-level `conftest.py` (preserving its own fixtures) or
one is created. Each recorder captures, BEFORE every interaction, a screenshot plus the target
element's bounding box plus a DOM probe (tag, role, accessible name, attributes, visibility) plus
page state (url, scroll, console errors); `capture_meta.json` holds those full per-step diagnostics
keyed by step number while `trajectory.json` keeps the minimal training contract. The test
function's own source is persisted next to the trajectory so the instruction generator can infer
intent from more than a function name — best-effort, with an explicit prompt fallback for the older
trajectories that have none.

The Selenium recorder differs because Selenium repositories build their own WebDriver rather than
receiving a fixture, so it patches `WebElement`, `Select` and `WebDriver` at import time and hangs
naming and saving off global pytest hooks, which captures `unittest.TestCase`-style tests as well as
plain functions. Coordinates come from `getBoundingClientRect` (viewport-relative CSS pixels, the
same convention as Playwright's `bounding_box`) and the viewport is locked to 1280x720 at
`deviceScaleFactor` 1 through CDP, so the recorded coordinates align with the screenshot pixels. An
action with no usable on-screen coordinate is skipped entirely rather than recorded as a (0,0) step
— recording those was the main cause of Selenium's low capture yield — and the wrapped methods never
raise, because instrumentation must not break the test it is watching.

`frameworks.py` is what keeps all of this out of the stages: rather than `if framework ==
"playwright"` branches, every framework-specific decision (how to install dependencies, which
instrumentation file to inject, how to invoke the runner, which action names count as interactive)
lives in a `FrameworkProfile` resolved from the `(test_framework, framework_language)` pair recorded
during deep analysis. Adding Cypress would be an entry plus its instrumentation file and no stage
edit; the trajectory output contract is identical across profiles. `resolve_profile` falls back to
playwright-ts for unknown values, preserving the historical default, while `is_supported_profile`
answers False for genuinely unsupported combinations (Java Playwright, non-Python Selenium) so
stage 4 can reject them instead of burning a sandbox on a run with no instrumentation hook. The
Selenium image is a separate linux/amd64 build (`make build-sandbox-selenium`), because Chrome and
chromedriver for Linux exist only for x86_64 and therefore run under emulation on Apple Silicon,
while the Playwright image is arch-native.

In replay mode the app itself is started by `app_runner.wrap_test_command`, which starts, waits,
tests and stops inside ONE compound command, because the sandbox runs each command in an
independent shell and a dev server backgrounded in one `exec_command` is dead by the next.
Repositories whose Playwright config already declares a `webServer` are left alone, since Playwright
manages that lifecycle itself. The module exists because most E2E suites assume something is already
serving on localhost: run with nothing serving they fail at `pass_rate 0` and capture nothing, which
was the dominant `no_capture` cause. Its scope is deliberately Node apps with a root `package.json`
start script — Python app servers and stacks needing a database or credentials still fail
`no_capture`, now with a persisted log saying why.

## Public-site capture lanes

The bulk of the corpus. Every lane is a thin shell over `lanes/capture_lane.py` (driver boot,
per-task capture and copy-out, verify and finalize, attempt bookkeeping, secret-leak check) and
resolves its sites through `sites/registry.py` (`load_sites`, `find_site`, `resolve_named_sites`,
`known_site_configs`, documented in `cua/pipeline/sites/schema.md`). They reuse
`teacher.run_teacher_task` and `teacher.verify_trajectory` and write `trajectories/site_<name>/task_*/`
with no registry row — so **stage 7 never visits them, and `make polish-pass` is their finalize**.
Capture attempts are recorded per task (`record_capture_attempt` / `is_capture_settled`, bounded by
`CAPTURE_MAX_ATTEMPTS`) so that a re-run resumes instead of restarting.

### Site registries

All of `cua/pipeline/sites/public_sites*.json` share one schema, so adding a lane or a site is a
data edit rather than a new loader.

| file | role |
| --- | --- |
| `public_sites.json` | the original curated login-free set; also the `held_out` eval source's site pool |
| `public_sites_actions_loginfree.json` / `_actions.json` / `_actions_indie.json` / `_actions_frontier.json` | action-UI sites (do, create, transact — never static browsing) |
| `public_sites_actions_logingated.json` | ~320 mainstream login-gated SaaS sites, each with a `provisioning` hint |
| `public_sites_actions_ua.json` | 87 Ukrainian sites / 411 Ukrainian-language goals, each carrying a recorded `probe` verdict (`docs/ua-coverage.md`) |
| `public_sites_search.json` | the search-goal lane (`task_style: search`); its sites must never appear in Online-Mind2Web's 300 |
| `public_sites_qa_verify.json` | the act → save → re-check lane (`task_style: qa_verify`) |
| `public_sites_guarded.json` | guarded-mode confirm/handoff tasks; gitignored, with `public_sites_guarded.example.json` as the template |
| `public_sites_human.json` | written by `lanes/provision_human.py`, gitignored, every site `auth_mode: pre_session` |
| `public_sites_saucedemo.json` | the login smoke target, which leaks into both the benchmark and the corpus |

### The lanes

- **Login-free and search** (`lanes/public.py`). The lane exists because funnel-sourced trajectories
  all live on localhost apps and rich bootable repositories are about 2% of GitHub, so this points
  the same machinery at curated, stable, bot-tolerant public sites with hand-authored hard tasks —
  cookie banners, dense layouts, real search ranking, map canvases — at the same cost per
  trajectory. Tasks are non-destructive by curation policy: browse, search, navigate and
  configure-view only. It reads any registry, optionally lets the proposer fill in tasks for a site
  that ships `tasks: []` (the context probe carries the site's session, so a login-gated site is
  proposed from its signed-in state rather than from its login wall), and honours the four auth keys
  fail-closed: a site that needs authentication without a vetted burner, or without its saved
  session, is skipped before a sandbox starts, so a real account is never touched. With a persisted
  proposal, or with proposing off, the whole task set is known without the driver, so an
  already-recorded site never starts a sandbox at all.
- **Login-gated, autonomous** (`lanes/logingated.py`). Per site it mints a real disposable mailbox
  and runs the site's tasks easy-first. The first task runs in `discovered` auth mode with the
  masked burner credentials plus the webmail inbox, so the teacher signs in — creating the account
  first when the site has none — and then completes the task; its authenticated `storage_state` is
  captured and REUSED via `pre_session` for the remaining tasks, so there is no repeated login.
  Session and burner persist through `app_secrets`, so a later run skips registration entirely: a
  site is registered once, not once per run. Because that first task also signs up, it gets an extra
  action budget, or a goal-completing run would be mis-graded as meandering. What counts as success
  there is whether a REUSABLE AUTHENTICATED SESSION was left, not whether the task's own goal
  completed, since a slow email-verify login often establishes the session and then exhausts its
  budget. If no session can be established — captcha, phone, disposable-domain rejection — the site
  is abandoned after that single attempt and marked `needs_human`, so a gated-beyond-reach site
  costs one trajectory rather than many; a site with prior autonomous success stays in this lane,
  because a human would hit the same flaky email and a transient failure is the wrong reason to
  route it to the human queue. Trajectories captured on a previous run but never verified are
  re-verified rather than being skipped by resume as "valid" and never becoming trainable.
- **Human-provisioned** (`lanes/provision_human.py`). Some sites guard signup with an interactive
  wall — captcha, phone, headless-fingerprint detection — that the sandbox agent cannot clear, and
  this is their on-ramp: a headed HOST browser in which a person clears the wall ONCE by hand, after
  which the saved session works from the headless sandbox because Docker Desktop egresses via the
  host's public IP, so the sandbox replays the session from the SAME IP the human logged in from.
  The only thing the human cleared was a one-time challenge; the durable session is portable. The
  identity is a disposable mail.tm burner minted once per batch and reused across sites, so the
  operator autofills the same credentials throughout while each site still gets its OWN
  `storage_state` under its OWN registered burner reference, with the inbox open in a second window
  so a verification link is one click away. Playwright's default Chromium is the automation-branded
  "Chrome for Testing" launched with `--enable-automation` and the AutomationControlled blink
  feature, which set `navigator.webdriver` and the other tells Turnstile blocks even when a human
  solves the challenge, so the real system Chrome is preferred and those flags are stripped — this
  stops the browser being fingerprinted as a bot, it does not solve anything. On a real site a plain
  visit already sets cookies, so "the dump is non-empty" never proves anyone signed in; only a change
  against the pre-login baseline does. `--serve` exposes a localhost JSON control API so the
  per-site open/save loop can be driven programmatically while the operator only ever types into the
  browser window in front of them.
- **Pre-filters.** `lanes/probe_capturable.py` probes over host HTTP against explicit WAF
  signatures only (Cloudflare challenge, Akamai, Imperva, PerimeterX, DataDome), using the same
  egress IP as the sandbox so IP-level blocks show up too, and it is conservative by design: it
  rejects only on an unambiguous block or challenge signature and leaves anything uncertain —
  transient network errors, Turnstile-only-on-submit pages that still render — to the more accurate
  capture attempt. `lanes/probe_sandbox.py` answers the different question that probe cannot:
  rome2rio passes the HTTP probe yet serves the container a Turnstile challenge, and sbb.ch passes
  it yet renders a blank page inside Chromium, each costing two or three teacher runs to discover.
  It opens each start URL in a real sandbox driver session, reads the snapshot and the affordance
  counts, and `--mark` writes `skip_capture` plus `skip_reason` back into the registry — seconds per
  site, no LLM. Run probes when capture is not running: each probe holds a sandbox, and total
  sandbox memory is the host's binding constraint.
- **One-off registration** (`lanes/register_real_site.py`) takes its site on the command line rather
  than from a registry. It mints a disposable mail.tm mailbox and has the teacher create an account
  using that address (typed masked), walk to the mail.tm web inbox, sign in, open the confirmation
  message and follow its link — all visually, recorded as one trajectory. This is the deployment
  skill SauceDemo's login could not prove. The site password IS the mailbox password, one masked
  secret for both logins, and the real verification mail is also confirmed out of band through the
  mail.tm API purely as a harness sanity check; the agent still reads it in the browser.

`provisioning_status.py` is the small store behind all of this: a generic site-name-to-status map
recording whether a site's burner account could be created autonomously. The autonomous lane marks
`registered` on success and `needs_human` when registration is blocked; the human lane consumes the
`needs_human` queue, and later autonomous runs skip those sites instead of re-failing them for cost.
Adding a status or a consumer is a data change. Its write is synchronous and called from the event
loop thread, so concurrent capture coroutines never truly overlap the read-modify-write.

### Guarded mode and the safety watchdog

At a `confirm` step the teacher proposes a consequential action INSTEAD of executing it, and
`safety_watchdog.py` judges that proposal as the LLM implementation of an out-of-model safety
service — the same shape as Gemini's computer-use safety service — returning approve or deny. On
approve the teacher executes and capture records the outcome; on deny the trajectory terminates at
the confirm. The verdict is recorded as an observation step with no SFT target of its own, threaded
into the student's history, so the student learns to execute only after approval, and the same
policy is reusable as the deployment-time safety gate. A refused confirm recorded no step, so no
watchdog call is spent and no orphan event step is emitted — a verdict would have nothing to attach
to.

`ApprovalPolicy` is pluggable: `LLMApprovalPolicy` is the default, with `StaticRuleApprovalPolicy`
(deterministic, no LLM) and `HumanApprovalPolicy` as drop-ins selected by `WATCHDOG_POLICY`, so
swapping the authority is configuration rather than code. Every implementation must fail closed —
deny — on any internal error, and `HumanApprovalPolicy` is a placeholder that fails closed until it
is wired to a real prompt channel, so it can never silently auto-approve. The `FINANCIAL_RISK`
categories move real money or cannot be cleanly undone, so they are never auto-approved on a LIVE
site; inside a controlled ephemeral test environment (`env="test"`) nothing is real and they are
allowed, and adding a category hard-denies it on live without touching the model prompt.

### Burner identities and the mail stack

Login-gated capture never uses the operator's real accounts. The isolation is what makes account
creation and login in-scope skills rather than forbidden ones: a disposable identity (`persona.py`,
defaulting to RFC 2606 and NANPA reserved ranges), a readable throwaway inbox (`mailtm.py` for real
sites, `mail_service.py` for sandboxed apps), an isolated browser context, a fail-closed registry
under gitignored `credentials/` (`app_secrets.py`), and `operator_pii.py` so the operator's own
identity can never be typed into a page or shipped in a corpus.

The point of the mail stack is that the agent does registration, email verification, password reset
and email-OTP login entirely through the browser, as ONE screenshot trajectory: it registers on the
app, NAVIGATES to a real webmail UI, opens the message, clicks the link or reads the code, and
returns. `mail_service.py` runs one shared GreenMail (an SMTP sink plus IMAP server that
AUTO-CREATES a mailbox for any address on first delivery, with auth disabled) and Roundcube stack on
the `vetty-mail` Docker network: sandboxed apps send to `greenmail:3025` and the in-container
browser reads `http://mail.test/`. Roundcube matters because it is a realistic webmail UI, so the
inbox skill the student learns transfers to Gmail or Outlook at deployment. One shared stack rather
than one per sandbox keeps memory flat, which is what the Docker VM ceiling demands. Addresses are
per scenario, `qa-<run>-<scenario>@mail.test`, isolated with zero provisioning — GreenMail strips
`+tag` addressing and pools it into the base mailbox, so identity has to live in the base
local-part. The SMTP environment handed to the app is keyed broadly (Django `EMAIL_*`, Laravel
`MAIL_*`, Symfony `MAILER_DSN`, Node `SMTP_*`/`MAIL_*`) so common frameworks pick it up with no
per-app special-casing, and the scenario password is derived deterministically — so a resumed run
reuses it — while satisfying the usual sign-up complexity rules and only ever reaching a trajectory
as `<secret:password>`.

GreenMail only catches mail from sandboxed apps whose SMTP was redirected. A real public site sends
its verification mail wherever the address resolves, so `mailtm.py` mints a genuine throwaway
mailbox with a web UI the agent opens in the browser: the API creates the box and confirms out of
band that a message arrived, but the agent still reads it visually, which is the skill being
captured. mail.tm throttles bursts, account and token creation especially, so 429 and 503 back off
and retry while everything else re-raises, with a separate connection-level retry for transient
network and DNS blips that would otherwise fail a whole site instantly during a brief outage.

`app_secrets.py` is the fail-closed registry underneath. `is_registered_burner` answers False for
anything missing, malformed or not-yet-registered, so a guarded capture refuses to run against a
real site without a vetted disposable identity — it must never be relaxed to "assume registered",
because the whole safety story rests on it. `register_burner` records `provisioning_mode`
(`human_provisioned` or `generated`) and the generated first and last names, so a signup form's name
fields stay consistent on reuse; a real personal account must never be registered here.
`save_storage_state` refuses an empty dump, so a mis-run where the human never actually signed in
fails loudly instead of persisting a logged-out session. `known_account` is the optional gitignored
registry of pre-existing test logins: demo apps routinely ship or advertise one, and when an app has
one, capture signs in with it instead of registering a fresh address — the only practical path for
apps whose mail leaves the sandbox to a cloud provider, where a generated mailbox would never
receive anything. Those credentials reach the driver as masked `<secret:...>` values and are never
recorded.

Known limits, each found the hard way: session injection works for cookie and localStorage
authentication but IndexedDB-auth SPAs need Playwright ≥ 1.51 with `storageState({indexedDB: true})`
(the pinned driver Playwright is 1.55); `cf_clearance` does not transfer into the container; and
cloud-email apps cannot deliver to GreenMail at all, so those need `known_account` entries instead.

## Calling Claude: the SDK paths

Every Claude call goes through `claude_agent_sdk.query()`, which returns an
`AsyncIterator[Message]`; the standard extraction is `message.result` from a `ResultMessage` whose
`subtype == "success"`. Authentication is the logged-in `claude` CLI's device OAuth, never an API
key, and `sdk_runner.CLAUDE_CLI_PATH` pins the SDK to the system binary (see "SDK and driver" under
Configuration for why).

There are **three** SDK paths, two of them wrapped by `SDK_QUERY_TIMEOUT` so a wedged CLI
subprocess cannot hang a run:

1. `sdk_runner.run_query` — the raw `query()` loop, rate-limit tracking, and `harden_options`, which
   strips operator plugins from headless calls. Without `setting_sources=[]` the spawned CLI loads
   the OPERATOR'S user-level plugins and skills into every call's system prompt: measured at +20k
   cache-creation tokens per call after a plugin install, enough to make a bare one-word call cost
   $0.19, which silently tripled the fixed cost of every query and blew through per-call budgets.
   Pipeline prompts are self-contained and must not inherit the operator's environment.
2. `agent_runner.run_agent` — wraps `run_query` and adds one `agent_calls` ledger row per call with
   turns, cost and tokens, classifying a non-success result as `RetryableFailure` when
   `raise_on_error`. Before it existed each stage hand-rolled the same `async for message in
   query(...)` loop and threw that telemetry away. Stages whose agent call is critical (scoring,
   deep analysis) set `raise_on_error=True`; stages with a graceful per-item fallback leave it False
   and use whatever text came back.
3. `think_generator._session_think_turn` — outside both, and therefore outside that timeout. It
   drives a persistent `ClaudeSDKClient` session because the one-shot `query()` the other two wrap
   cannot carry a conversation across turns, and the session exists precisely so the fixed system
   context is cache-READ on turns 2+ rather than cache-CREATED per step. It feeds the rate-limit and
   cost trackers itself.

**A failure that is not a rate limit never propagates.** A timeout, an exhausted budget, a 5xx or a
max-turns stop leaves the result empty and the caller's existing fallback and parse logic handles
it; `fallback_to_assistant_text` additionally returns the last assistant text block when the success
result was empty, which stage 3's scoring path relies on, and `run_agent`'s
`capture_assistant_fallback` keeps the last assistant message so an answer arriving in the final
turn rather than as a tool-free `ResultMessage.result` is still recovered. A wedged CLI subprocess
is abandoned with no `ResultMessage` and therefore no telemetry row, returning whatever partial text
exists: for the teacher that is a soft stop, because its own `finally` block has already finalized
the partial trajectory, and for verify and think the caller falls back exactly as it does for an
empty result. Budget is INFORMATIONAL ONLY and must never reject a repository or discard a
trajectory — the SDK surfaces a per-call cap as a "Reached maximum budget" exception, treated here
as a soft stop — because runaway protection comes from `max_turns` and the per-task action cap, not
from dollars.

`RateLimitTracker` tracks utilization per `rate_limit_type`, because the CLI interleaves events for
different windows (`five_hour` and `seven_day`) and a single "latest" slot would flip-flop between
windows at different utilizations; `limit_reached` fires when ANY tracked window crosses the
threshold, or when a hard 429 marked the limit rejected. Its methods are synchronous with no
`await`, so they run atomically under the one asyncio event loop and concurrent workers need no
locking. `CostTracker` is the matching process-global sum of every `ResultMessage.total_cost_usd`,
which is the metered spend the orchestrator's budget pacing reads (see Budget and concurrency).

Options are `ClaudeAgentOptions(model, max_turns, max_budget_usd, permission_mode, allowed_tools,
cwd, mcp_servers)`. A tiny `max_budget_usd` fails before the first tool call, so every budget has a
floor. Models are configured centrally in `cua/pipeline/llm_config.py`: `LATEST_MODEL_BY_CLASS` maps
each class to its latest ID and call sites import a named purpose (`MODEL_FOR_REASONING`,
`MODEL_FOR_GROUNDED_GENERATION`, `MODEL_FOR_TEACHER_AGENT`, `MODEL_FOR_CLASSIFICATION`,
`MODEL_FOR_SAFETY_JUDGE`), never a model string. The purposes capture intent rather than capability,
so if a future re-split sends classification to a cheaper model again, one constant changes here and
every classifier follows — no grep across stage files, and no model ID anywhere else in
`cua/pipeline/`.

A "hung" LLM stage is almost always one of two things: a saturated 5-hour usage window, where every
CLI call waits at 0% CPU (time a one-word `claude -p` call before debugging code), or Docker VM
memory exhaustion.

## Budget and concurrency

LLM stages run under an `asyncio.Semaphore(workers)`, and the orchestrator caps workers per stage
(3 for analyze, 2 for trial and generate) because those hold a container each. `--workers` is first
capped by the stage's own ceiling and then clamped against Docker VM memory for exactly those
container-holding stages, so a generous flag cannot re-create the over-subscription collapse.

Cost control is two-layered, and the second layer exists because the first one raced.
`orchestrator.budget_available` stops launching new stages once `--budget` is covered by METERED
spend — `cost_tracker.total_usd`, the sum of every `ResultMessage.total_cost_usd` since the drain
started — and each stage additionally checks its own `cost_estimate` against the remaining metered
budget **inside** the semaphore, because a gate placed before it is evaluated by every gathered
coroutine while `cost_estimate` is still ~0: they all pass, queue past the gate, and the cap never
fires. Hitting either limit leaves repositories in their input status rather than rejecting them.

`orchestrator.STAGE_COST_ESTIMATES` survives only to pre-size batches. Pacing `--budget` by those
per-repo guesses drifted badly — a 200-repo stage-3 pass "spent" $30 of estimate while the
subscription window sat at 10%, stopping runs that had barely started — so the budget is paced by
actual spend and the estimates only decide how much work to launch. The `agent_calls` ledger is the
per-call record behind the same numbers.

`--overnight` and `--until-limit` both run through `orchestrator.drain_funnel`: overnight is one
pass, until-limit loops until the usage threshold, the budget or a drained funnel. `run_query` feeds
the SDK's `RateLimitEvent` into a process-global `rate_limit_tracker` and raises `UsageLimitReached`
on a hard 429. That exception is re-raised — never caught and turned into a rejection — at every
layer, so hitting the usage limit leaves repositories in their input status for the next run.
`--until-limit PCT` sets the tracker's threshold; with no usage events observed it degrades to "stop
on a hard 429".

## The dashboard

`cua/pipeline/dashboard/` is a read-only Streamlit explorer over the corpus and the registry, and
its design rule is that it **describes and never scores**: every chart is a plain distribution,
count or scatter, with no thresholds, no green/red target bands and no composite quality number.
The few constants in `metrics.py` are descriptive bands used to LABEL parts of a histogram, never to
gate. Each chart function takes a possibly-filtered DataFrame and returns a
`plotly.graph_objects.Figure`, with empty or missing-data input caught up front and routed to a
small grey "no data" placeholder instead of raising.

`loader.py` walks `trajectories/` in both the flat `task_*/` and nested `repo_*/task_*/` layouts and
tolerantly loads the three contract files from each task directory, returning two long-form pandas
frames plus human-readable warnings for anything skipped or defaulted; every load is wrapped,
because the dashboard must never crash on one malformed trajectory. `trajectory.json` is the
post-stage-6 source of truth for the instruction — it can be generated after the test ran — so the
trajectory's copy wins where an older `metadata.json` carries an empty one. The registry loaders are
equally defensive: they read the funnel state read-only, never migrate, and guarantee the columns
the Rejections page needs even against a database predating the failure-taxonomy migration.
`_state.py` is kept separate from `loader.py`, `charts.py` and `metrics.py` so those three stay free
of `streamlit` imports and remain exercisable from a plain REPL.

The landing page answers "what does the corpus look like overall?" with raw distributions, and the
per-path breakdowns live on subpages, because the two paths carry different evidence: Path A
trajectories have no in-sandbox DOM probe, so `bbox`, `role`, `name`, `tag`, `dt_ms` and
`dom_changed` are unset on every step, while Path B is where `capture_meta.json` lights up every
element field and the Elements tab means anything. The Inspector draws a bbox as a green rectangle
when one exists and otherwise a small red dot at the click point. Rejections splits score
distributions passed against rejected with the gate threshold drawn in, so the operator can see how
many repositories a threshold change would recover; older rejections only have that data after
`python -m cua.pipeline.backfill_scores` runs, while new ones populate automatically. Tokens splits
usage into output, uncached input, cache-read and cache-write, stacked cache-below-billed-output —
the cache columns are absent on calls recorded before the cache-token telemetry fix and read as 0,
so historical totals understate true input. Yield's "reached stage 5" means a repository that either
cleared the trial or was rejected at it, sliced by framework, which makes it the headline health
metric for whether app-start and selection are working; its `no_capture` table shows the persisted
tail of each failed test run.

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
  `cua.pipeline.smoke.driver`) is their primary defense. Operator identities come from `cua.pipeline.operator_pii`.
* **Corpus health (exit 2).** Thresholds in `corpus_health.json`: the regressions that produced
  sft-v1's gaps refuse to ship again. `--skip-health-gate` reports violations and writes anyway, for
  experiments only.

The corpus report printed before those gates is the at-a-glance health check that catches a
Booking.com-style regression at convert time instead of in a JSON file weeks later.

### Trainability is one shared function

`cua.pipeline.corpus.audit.trainability` is the ONE gate, used by convert, `make audit-corpus` and
every corpus pass, so they cannot drift. A trajectory being written by a concurrent capture parses as
truncated JSON; convert skips it rather than letting one half-written file kill a multi-hour training
run after the GPU is already up.

`metadata.excluded_reason` always skips; agent trajectories require the verifier's `success: true`;
legacy replay trajectories train only with `--include-replay`. Running the gate as a standalone
audit (`python -m cua.pipeline.corpus.audit`) additionally enforces `min_outcome_relabeled_share`
(the outcome-relabel pass must have run) and `min_done_evidence_share`.

### The corpus passes

`cua/pipeline/corpus/` holds the passes that improve an already-captured corpus, all built on one
`pass_base.py` (dry-run by default, `--apply` to write, resumable, one shared trainability gate):
`polish.py` (the stage-7 thought polish for `teacher_raw` directories outside the registry flow),
`outcome_relabel.py`, `relabel_recoverable.py` (an audited hindsight relabel where a fresh verifier
verdict gates every write), `wall_supervision.py`, `instruction_style.py`,
`verify_pattern_scan.py` and `regenerate_supervision.py`.

The package `__init__` is deliberately empty of sibling imports and `audit.py` is deliberately
pydantic-free, because the Modal and Vast train lanes install no pydantic and reach the module
through `cua.sft.convert`; pulling a pass in behind the audit would break both. Resumability is a
property of the data rather than a checkpoint file: `polish.needs_polish` skips a trajectory whose
`thought_source` has already flipped to `teacher`, and `PassResult` splits the status the tally
counts (`note`) from whatever the pass's own report row needs (`detail`).

Each pass carries one non-obvious rule. `audit.action_signature` yields an empty signature for a
step made only of control actions, which the run-length check treats as a break, so repeated pauses
never read as a loop. `outcome_relabel.relabel_window` runs with every built-in tool removed,
because a tool call would consume the single turn and the SDK would raise `error_max_turns`.
`relabel_recoverable` proposes a goal for what a partial run actually finished and then re-judges it
with the SAME verifier, keeping it only on a fresh success — and its prompt is written against the
observed failure mode, a "goal" that reads as a browsing procedure ("navigate to the compress
page"). `wall_supervision.trim_to_wall` drops the old verdict's efficiency grade, because that
verdict graded the UNTRIMMED run and carrying it over would subsample a 4-step demo as meandering.
`regenerate_supervision` walks Path B nested and Path A flat directories only, deliberately not the
corpus-wide walk, because it has no prompt for the `site_*` lanes, and it fills every
`step['thought']` in step order so the prompt for step N sees the thoughts just written for 1..N-1.

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
environment variable instead of being passed as a flag, and it is not alone: `ENV_KEYS` lists the
config keys that are environment rather than flags, while everything unmapped is forwarded verbatim
as `--key value` in config order, so a recipe can reach a `swift` flag the map has never heard of.

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

`BOOTABLE_GATE=0` turns the stage-2 gate off and falls back to the old metadata-only filter; with it
on, a probe error never rejects (the repository stays `discovered` to retry).

### Capture mode and auth flow

`PIPELINE_CAPTURE_MODE` chooses between the two capture modes described above, and is the one
setting a CLI flag rebinds at runtime. `PIPELINE_AUTH_FLOW_CAPTURE` wires the shared mail stack into each
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

## Training

`config.yaml` holds the SFT recipe: Qwen3.5-9B, LoRA r64/α128 all-linear, lr 5e-5, 2 epochs,
max_length 4096, batch 2 × accumulation 4, `freeze_vit: true`, `packing: false` (the image inputs
lack a flash-attention implementation), and `max_pixels` exported as `MAX_PIXELS`. Variants are configs,
not forks: `config_sft_v2_conservative.yaml` is the r16 / 1-epoch / lr 1e-5 arm. `python -m cua.sft
train` maps config keys onto `swift sft` flags through `cua/sft/swift_launch.py` (ms-swift 4.x);
CLI overrides (`--epochs`, `--lr`, `--batch-size`) win over the config, and `cua/sft/recipe.py` is
the one shared convert+train recipe the cloud lanes call.

**Serving a fine-tuned adapter needs `--response_prefix ''`** — ms-swift's default prefills an empty
`<think>` block, after which no action line is ever emitted — and the base must be passed
explicitly, because `adapter_config.json` records a stale volume path. A GRPO adapter is trained
over the MERGED SFT weights and must be served on those, never on the raw base; `cua/cloud/modal/merge.py`
is what produces the merge.

## Rented GPUs

Two lanes rent GPUs: Modal (`cua/cloud/modal/`, primary for SFT and GRPO) and Vast.ai
(`cua/cloud/vast/`, provider-pluggable). Neither holds any training logic — both run the unchanged
`python -m cua.sft` / `python -m cua.rl.train` entry points, so a recipe change is never duplicated
in the cloud layer.

### The Modal lanes and their shared module

`cua/cloud/modal/common.py` holds what the lanes genuinely share: the `cua-trajectories`,
`cua-outputs` and `cua-hf-cache` volume handles, the mount points, `with_package` (the env plus the
first-party file layers that must close an image, because `add_local_*` admits no pip step after
it), and `commit_periodically`, which commits a volume from a daemon thread so per-epoch
checkpoints survive a crashed or preempted run instead of landing only when the run ends. The
`cua/` layer's `ignore=` list is `PACKAGE_IGNORE`, derived from the same source as the Vast upload
filter below, so neither lane bakes or ships a gitignored file.

`train.py` and `grpo.py` import it; `serve.py` and `merge.py` deliberately do not. Modal is invoked
in *script* mode (`modal run cua/cloud/modal/serve.py`), where the file is imported as a top-level
module, so Modal mounts only that single file
(`modal/_utils/function_utils.py:get_entrypoint_mount`, `FunctionInfoType.FILE`) and the container
re-imports it with no `cua` package on the path. `train.py` and `grpo.py` are importable there only
because their images mount `cua/` at `/workspace` with `PYTHONPATH` set; the vLLM serve image and
the merge image do not, so those two read their configuration from the environment (each read is
marked) and keep their own volume handles. That is also why every env value they resolve at deploy
time is baked into the image: the container re-import would otherwise resolve it differently.

The CUDA and torch pins differ per lane on purpose: `train.py` is CUDA 12.4.1 + torch 2.6.0
(the stack ms-swift 4.x was proven on, with torchvision 0.21.0 as its paired release), `grpo.py` is
CUDA 12.8.1 because the vLLM/torch column resolved in `constraints/rl-modal.txt` needs it,
`serve.py` is CUDA 12.4.1 + vLLM 0.28.0, and `merge.py` is a plain `debian_slim` with ms-swift
4.5.2 and unpinned torch, since it only runs `swift export` once per adapter. A GRPO adapter is
trained over MERGED SFT weights and cannot be stacked on the raw base, which is what `merge.py`
exists for.

`train.py` also carries `encode_check`, a CPU-only function that tokenizes one two-image sft3 step
and one grounding example through ms-swift's own template and reports their image-pad and label
counts. It is the cheap way to answer "does this prompt actually fit `max_length`?" against the real
tokenizer rather than an estimate, without renting a GPU.

### Teardown is five layers, not one

`cua/cloud/vast/lifecycle.py` rents a GPU, waits for SSH, uploads, launches the job detached under
a `STATUS` sentinel, polls, pulls results and ALWAYS destroys. The guarantee is defence in depth,
because a leaked instance bills until someone notices: a `try/finally` around the whole run;
SIGINT/SIGTERM/SIGHUP traps; a `destroy` that polls until the provider confirms the box is gone; a
crash-safe state file (`state.py`) plus the `cleanup` orphan sweep for hard kills; and an
independent max-runtime watchdog thread. Teardown never outranks ownership: the account is shared
with colleagues and with concurrent agent sessions, so `_owned_by_us` refuses any instance the
provider reports under a foreign label, and a record whose owning pid is still alive belongs to a
sibling run and is left alone.

Two failure modes are only visible in the details. The ssh client can fail to return for a detached
child even with `setsid` and all three descriptors redirected, so its exit code proves nothing and
the `STATUS` file the job writes is the only evidence it started. And a rejected Vast create can
still allocate a billing contract (observed 2026-09-01: `success=False` alongside
`new_contract=49552760`), which is why `provision` destroys the contract before raising — nothing
later in the lifecycle can, since the caller never received an instance.

Two more details make the teardown guarantee survive contact with reality. A detached remote job
outlives its supervisor, so `keep_on_signal` trades a possible orphan for hours of paid work rather
than killing a running training run on a stray signal, and `state.owner_alive` treats an UNKNOWN
owner as alive for the same reason: a leaked box is reclaimable later, while a destroyed live job is
not. `pid: null` is the explicit released sentinel a supervisor writes to hand a box to `cleanup`.
Selection is deterministic, which means a naive retry re-rents the host that just failed the
workload, so `rent_with_retry` blocklists a failed host before trying again.

Offer selection is the pure `base.select_cheapest_allowed`, configured in `remote_train.yaml`
(defaults < environment < CLI): a VRAM floor, a compute-capability floor, a CUDA-version floor, a
price cap, a host-reliability floor, and an `exclude_countries` deny-list. `max_cuda` is the
opposite bound, the host driver's maximum supported CUDA with 0 disabling the filter, because an
image built on newer CUDA fails at engine initialisation on an older driver. Every instance carries
`cfg.instance_label` (default `cua-remote-train`), which is also the orphan-sweep key.

The provider adapter isolates Vast's own quirks. Its search grammar treats `gpu_ram` as GIGABYTES in
a QUERY but megabytes in the response JSON — verified by probe: `gpu_ram >= 70` returns only 81920MB
cards — and the SSH key is attached to THIS instance rather than the account, because a team-scoped
API key cannot register an account-wide one, with the onstart hook as the fallback. `tunnel.py` is
the other half of the transport: an SSH resource that exposes the local env service to the box with
`-R` and brings the box's serve port back with `-L`, supervised the way autossh would.

### What is uploaded to a rented box

`jobs.upload_package` ships the whole installable package rather than a hand-listed manifest,
because a manifest goes stale the moment someone adds an import and the failure lands 45 minutes
and one rental later as an ImportError on the box. The box is shared hardware, so the upload is
filtered instead, and both lanes read that filter from one place, `cua/cloud/ignore.py`:
`gitignored_excludes` asks git for every ignored path under `cua/`
(`git ls-files --others --ignored --exclude-standard cua/`, bounded by a timeout) and turns each
into an rsync `--exclude`, anchored with a leading `/` because the transfer root is the `cua`
directory itself; `modal_ignore_patterns` re-shapes the same list as dockerignore-style globs for
`add_local_dir(ignore=…)`. Bytecode is excluded by pattern (`__pycache__`, `*.pyc`) rather than
enumerated, and the two gitignored site registries are excluded unconditionally — a git listing
names only files that exist right now, while `public_sites_human.json` is written by the
provisioning lane afterwards. That is what keeps `public_sites_human.json` /
`public_sites_guarded.json` — burner-account and guarded-task data — out of a rented machine and out
of an image layer alike. `download` is unfiltered: results come back whole.

The remote commands carry two hard-won preambles. The CUDA images' python is externally managed
(PEP 668) and ships no `ensurepip`, so pip needs `--break-system-packages` and `uv` builds the venv
over the image's site-packages, keeping its torch/CUDA while sidestepping Debian's unremovable
apt-installed pins. And the plain HuggingFace downloader pulls the 19GB base at ~0.5MB/s on a
rented box — hours of paid GPU before the first step — while `hf_transfer` is now silently ignored
by `huggingface_hub`, so the acceleration knob is `HF_XET_HIGH_PERFORMANCE`. The train job installs
Modal's proven train stack explicitly and never `.[pipeline]`, which clobbered torch and left
transformers half-imported.

### The CLI and the no-cost self-test

`python -m cua.cloud.vast <job>` (installed as `cua-remote`) exposes `train`, `run` (an alias whose
`--smoke` is the self-test), `eval`, `grpo`, `serve`, `search` (a dry run that prints the cheapest
allowed offer and rents nothing), `cleanup` (`--all` also takes boxes a sibling run may be using)
and `release` (hands a recorded box to `cleanup` by writing the pid-null sentinel). `--provider` and
`--benchmark` choices come straight from the `PROVIDERS` / `BENCHMARKS` registries — importing the
package is what registers every provider, which is what populates the flag — so a new provider or
eval appears in the CLI with no change there. `jobs.JOBS` is the third registry: `train`
(the unchanged convert + train), `eval` (a `benchmarks.BENCHMARKS` entry, no training), `grpo`
(`cua/rl/train.py` against the local env service over a tunnel) and `serve` (the student on
127.0.0.1:8001 through a tunnel).

A `BENCHMARKS` entry pins the properties of its dataset mirror, because getting one wrong reports a
confident wrong number rather than crashing, and it does NOT pin `coord_space`, which belongs to the
model and is written per run by the eval job. Its `harness_files` preserve each DECLARED relative
path, so an imported package lands on the box as a package rather than as loose files. The GRPO job
adds two guards learned from paid runs: `preflight` asks the env service for the task it is about to
train on, because a task the service does not serve 404s on reset and the trainer plugin turns that
into a fabricated reward of 0.0 — better caught before renting — and `on_poll` pulls checkpoints
from `rl_output/` rather than `latest/`, since `latest` only appears after swift exits, so every
mid-run checkpoint used to die with the box.

`lifecycle.SMOKE_CASES` is that self-test: each case is one `_smoke_*` function taking the shared
scratch environment, and adding one is a function plus a registry entry. Its central assertions are
teardown (every job type destroys its instance on the happy, exception and SIGINT paths), the
ownership gate, the pid-based sparing of live siblings, and that the package upload carries its
excludes. It rents nothing and runs as `make test-remote`, against the in-memory provider in
`mock.py`, which supplies fake offers, an instance that answers SSH, and a record of every destroy,
upload and tunnel for the assertions to read.

## Evaluation

Two harnesses, documented in full in `docs/evaluation.md`:

- `cua/eval/screenspot/` scores **grounding** — a predicted click point against a target bounding
  box on `rootsautomation/ScreenSpot`. It isolates pointing from task completion.
- `cua/eval/student/` runs a served model through **real browser episodes** and reports whether a
  checkpoint beat the base. The deliverable is always a before/after delta plus a harness-parity
  check.

Both import their prompt builders from `cua/prompt/format.py`, so an evaluation cannot silently
measure a different prompt than training used; `tests/fixtures/prompt_golden.json` is the
byte-level contract convert, both harnesses and the RL rollouts all assert against.

Evaluation and RL episodes never enter training: they land under `eval_runs/` and `rl_runs/`, both
outside the `trajectories/**` glob, and carry `metadata.excluded_reason` of `eval_episode` or
`rl_rollout`.

## RL rollout harness (frozen)

`cua/rl/` performs policy-agnostic rollouts of the student over the same sandboxed driver
(`POST /observe` plus `/act kind=sequence`): one recorded step per policy turn, the observation
PNG's bytes ARE the recorded step screenshot, and the observation text is built by
`cua.prompt.format.build_step_user_text`. `cua.rl.actions.decode_step` is the inverse of
`format_action_for_training`. Rollouts land under `rl_runs/<run_id>/iter_NNN/<task>/kKK/task_*/` as
full trajectory directories, so `verify_trajectory` and the corpus audit work on them unchanged.

**The lane is frozen at the one-image v2 observation.** RL was dropped on 2026-09-02 after three
GRPO runs (`docs/decisions/grpo-v3-results.md`), and it must not be run against an sft-v3 policy
without the two-image plugin work: ms-swift's `_add_default_tags` would silently prepend a missing
`<image>` into the system prompt.

Reward v2 (`cua/rl/rewards.py` plus `cua/rl/reward_config.json`) resolves a task's programmatic
`success_check` (the `cua/rl/checks.py` registry, authored in `cua/rl/rl_tasks.json`) and falls back
to `verify_trajectory` (`--verifier-mode always|fallback|never`). `success` is the judge's
criteria-consistent verdict discounted ×0.6 on `meandering`; step efficiency slopes to 0 at twice
`expected_actions`; the `done()` bonus is paid only on success; and a false `done()` or hitting
either cap (`step_cap` env-side, `max_turns` trainer-side) is penalised. Train rows are filtered by
`train_max_expected_actions` so that `max_turns` can cover a wandering sample — otherwise every
sample in a group hits the cap alike and the advantage is exactly zero.

The task suite is built from the site registries minus everything SFT trained on, with splits
`train`, `eval_seen_site_unseen_task`, `eval_unseen_site` and `eval_mutating`; the frontier sweep
pool is the frontier registry plus the base suite's short train tasks, all forced into `train`.
`cua/rl/smoke.py` is the Docker-only end-to-end check of the whole harness — a static page and a
scripted policy driven through `Episode` and the HTTP service, with no LLM.

Nothing in a batch may be killed by one bad episode. A precondition refusal skips its task and a
dead sandbox is a harness error, and neither ends a batch of hundreds; `backends.reset` tolerates a
sandbox that is gone entirely (OOM-killed, auto-removed) and rebuilds rather than raising; and
`episode._observe` retries the driver's title and screenshot reads, which fail transiently while a
navigation is still committing. A CAPTCHA or WAF wall is the environment refusing rather than the
policy failing, so the analysis drops those samples outright. `judge.py` resolves terminal success
as the programmatic check first with the Claude verifier as fallback or audit, all SDK calls on one
background loop, and it clears the rate-limit tracker's sticky rejected flag before judging — a
stale limit would otherwise defer every verdict for the rest of the run, and the call itself is the
proof. `analysis.py` reads `rollouts.jsonl` when a run finished and the `episode.json` files it
already wrote otherwise, so a run stopped early is still analysable; `HARNESS_ENDINGS` marks the
endings that measure the harness rather than the policy, because counting them as failures would
push a task out of its band for free, and `BNORM_FLOOR` is the kill signal — a GRPO adapter whose
LoRA-B norm stays under that share of the SFT adapter's after about ten steps has learned nothing.

The env service and the trainer plugin are joined by an HTTP contract with several sharp edges. The
trainer's GRPO samples of one task all carry the same `sample_idx`, so the service numbers them
itself and `k` is what keeps their trajectory directories apart. A step that raised AFTER acting
must end the episode there, because a retried request would execute the action a second time.
`EnvClient` retries every 5xx (the service's own 503, the tunnel's 502 or 524) and treats a 4xx as
fatal; a comma-separated `RL_ENV_URL` is an ordered set of ingresses to the SAME service, rotated
through so one dead tunnel does not end the run; and queueing for a slot is the normal case at
generation start, so a reset waits far longer than a step and carries a `request_id` that makes a
cut-off retry land on the same episode. `MockEnvClient` is the offline stand-in, returning a fixed
PNG and reward 1.0 only when the final response decodes. On the plugin side, `BrowserEnv`'s
round-trips (slot waits, the verifier on a terminal step) must not block the rollout server's loop
or every concurrent episode stalls with them; an unlabelled transport blip is recorded on the env
too, because ms-swift sums step rewards into `total_reward` and an unmarked blip would become a 0.0
that outranks a genuine failure in its group. `BrowserScheduler.run` makes each turn a fresh
single-image prompt and returns one `RolloutOutput` per turn carrying the episode reward; the env
only sees a close for a trainer-side cut, so that close is named and its reward read back, because
the reward function treats `max_turns` like its own step cap. Env unavailability stays 0.0 but is
labelled, so the analysis can tell it from a policy failure.

On-box orchestration is merge, then deploy and `EVAL_BEFORE`, then rollout and rlhf, then deploy
latest and `EVAL_AFTER`, with the phases signalled through `PHASE` / `PHASE_ACK` files. The merge
passes `--exist_ok`, because without it a merge killed mid-write poisons every retry: `swift export`
refuses the half-written directory. A deploy target given as a dict serves several checkpoints from
one rental, each under its own name, while a string keeps the canonical single-adapter name.

On the GPU side (`cua/rl/trainer/`, `cua/rl/train.py`, `rl_config.yaml`, `constraints/rl-modal.txt`)
ms-swift runs GRPO in vLLM server mode with a browser `Env`/scheduler plugin
(`cua/rl/trainer/swift_plugin.py`, verified against ms-swift 4.5.2) that calls the env service
(`cua/rl/service.py`, loopback plus `RL_ENV_TOKEN`) over an ingress, writing one training record per
turn with the episode reward broadcast. The SFT LoRA is merged into the base before RL so the KL
anchor is the SFT policy. Two hard-won settings: server mode needs **two GPUs**, because the
trainer↔vLLM NCCL weight sync cannot share a device, and `response_prefix: ""`.
`--mode fallback` (decoupled batch GRPO) is not implemented. What would actually move the RL score
is ranked in `docs/open-items.md`.

## What was deliberately left alone

- **The three replay StepRecorders** (`instrumentation/pw_python_conftest.py`, `pw_ts_fixture.ts`,
  `selenium_python_conftest.py`) are untouched by the package refactor. They execute inside the
  repository under test, where the `cua` package is not importable, so they keep their own
  environment reads behind a marker and their own copies of what they need.
- **The RL lane** is frozen as described above rather than being brought forward to the two-image
  observation.
- **Dated records under `docs/decisions/`** keep the module paths that were true when they were
  written. They are evidence of what was decided and when, not living documentation.

## Where the reasoning lives

`docs/decisions/` holds dated decision records:

| record | subject |
| --- | --- |
| `real-sites-vs-controlled-backends.md` | why the corpus captures on real third-party sites |
| `open-questions.md` | running list of decisions surfaced and parked |
| `judge-consistency-audit.md` | why audited numbers differ from measured ones |
| `grpo-v2-first-live-run.md` | the first live GRPO before/after, and what it proved |
| `grpo-v3-pipeline-fixes.md` | why GRPO v2 was a dead run and what changed |
| `grpo-v3-results.md` / `grpo-v3-run-results.md` | the third and last GRPO round: alive, and flat |
| `sft-v2-corpus-round.md` / `sft-v2-results.md` | the corpus-gap round and its measurements |
| `sft-v3-plan.md` | what the third round was for |
| `sft-v3-corpus-round.md` | its design and pre-registered bands |
| `sft-v3-results.md` / `sft-v3-postmortem.md` | what it measured, and the gap against the prediction |
| `why-a-9b-outgrounds-the-frontier-model.md` | why the local arms sit above the frontier arm on ScreenSpot |
| `hybrid-cost-and-the-product-gap.md` | what counts as cost, and what the architecture does not yet ship |

Alongside them:

- `docs/design/` — design specs written before implementation: the real-site auth capture lane, the
  quality-first funnel intake, the login-gated trajectory lane, the post-gather SFT worklist, and
  the RL harness. These are living documents and carry current module paths.
- `docs/runbooks/` — step-by-step operating procedures: the GRPO runbook and the SFT v3 evaluation
  runbook.
- `docs/sft-corpus-gaps.md` — ranked, measured corpus gaps that drive each corpus round.
- `docs/pr-sft-v3-measurement-round.md` — the write-up of the round that audited both evaluation
  harnesses and re-measured grounding on a comparable basis.
- `docs/open-items.md` — the living blocker list, with evidence.
- `docs/ua-coverage.md` — the Ukrainian-web registry and what its probe pass measured.
- `docs/conventions.md` — the style, registry and measurement rules the code enforces.
- `docs/evaluation.md` — the two evaluation harnesses and their scoring traps.
- `README.md` — the user-facing walkthrough, the generated experiment-registry block, and the
  measured before/after tables with reproduction commands.
