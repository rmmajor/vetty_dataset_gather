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
