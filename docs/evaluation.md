# Evaluation

Two harnesses, measuring two different things. ScreenSpot scores **pointing**: given a screenshot
and an instruction, does the predicted click land inside the target box. The student eval scores
**task completion**: does a served model finish a real browser task. They are complementary and
they diverge — SFT v1 lost grounding while gaining task success — so neither substitutes for the
other.

Both import their prompt builders from `cua/prompt/format.py`, the same functions
`python -m cua.sft convert` uses, and `tests/fixtures/prompt_golden.json` is the byte-level contract
all three assert against. An eval that prompts differently measures the prompt.

Measured numbers are not repeated here. They live in the experiment registry
(`experiments/*.csv`, rendered into `README.md`), because a number quoted in two places drifts.

## ScreenSpot grounding

`cua/eval/screenspot/` measures **grounding** accuracy: given a screenshot and an instruction, the
model predicts a click point and the harness scores whether that point lands inside the target
bounding box. It exists to compare one model before and after SFT, and stays reusable for arbitrary
models through a coordinate-space registry, a configurable output parser, a pluggable inference
backend and a configurable dataset/bbox adapter.

The prompt is training's grounding slice verbatim — `GROUNDING_SYSTEM_PROMPT` plus the one-image
sft3 observation, built by `cua.prompt.format` — so an eval that prompts differently would be
measuring the prompt rather than the model.

### The scoring conventions are the whole ballgame

Get one of these wrong and the eval does not crash; it reports a confident WRONG number.
All of them were verified against real model output rather than assumed.

**`coord_space` is per MODEL, never global** (`ModelSpec.coord_space`). Different checkpoints answer
in different coordinate conventions: a stock Qwen and every sft-v3+ checkpoint emit
`normalized_1000` (0-1000 of the image), while the sft-v1/v2 checkpoints emit `viewport_abs`,
absolute pixels of the fixed 1280x720 training viewport. `native_abs` reads the answer as pixels of
the source image, for other models. Measured: the SAME base-model predictions score **85% as
`normalized_1000` and 25% as `viewport_abs`** — a before/after scored with one global
`--coord-space` silently mis-scores one side of the comparison. A frontier arm has to be asked for
pixels and scored `native_abs`: asked for 0-1000 it half-complies, and no single mapping scores it
(36.5% withdrawn against 68.3%). A coordinate space maps points in
both directions and NEVER touches the image: when it resized, declaring `viewport_abs` squashed the
input, and the two arms were then scored on different pictures.

**The image every arm sees is a harness setting, not a property of its coordinate space.**
`SCREENSPOT_IMAGE_SIZE` (default `native`) is recorded in every artifact as `prepared_image_size`.
Until 2026-09-06 the `viewport_abs` path resized inputs to 1280x720 on its own, so those arms were
scored on squashed screenshots while the others saw the source image; correcting it moved sft-v2
from 76.3% to 85.4%. Set it once, for every arm, and read it back out of the artifact.

**`--max-tokens` (default 2048).** At 512 a model that reasons before answering is cut off before
it emits a click, which scores the cap rather than the model — and it penalises only the verbose
arm, manufacturing an improvement on the terse one. A cap truncating one arm of a comparison is
the most dangerous defect in this harness; read raw responses before believing a low score.

**`--bbox-normalized` (default on).** The default dataset stores bbox as fractions of the image.
Treating those fractions as pixels shrinks every target to a sub-pixel box at the origin, which is
0% accuracy by construction, regardless of the model.

**`--bbox-format xyxy` (default).** The default mirror stores `(x1, y1, x2, y2)`; `xywh` describes a
different box from the same four numbers.

**The dataset is `rootsautomation/ScreenSpot`, not `os-copilot/ScreenSpot-v2`.** That second repo is
raw JSON plus an image zip, with no `test` split and no image column, so `load_dataset()` cannot
read it at all. The mirror above ships real parquet with a 1272-row `test` split and a proper image
column, and its annotation convention (bbox as `xyxy` normalized to 0-1) is what the defaults
encode. Re-check all three conventions against any other mirror.

`cua/cloud/vast/benchmarks.py` pins the two bbox flags on the rented-box lane for the same reason,
and writes `coord_space` per spec rather than passing a global one.

### Running it

End-to-end evaluation needs a GPU and a real checkpoint. `--smoke-test` exercises every non-model
component (prompt parity, parsing, coordinate mapping, hit-testing, aggregation, reporting) against
a mock predictor and synthetic samples, requiring no model, GPU or network — it is what
`make test-eval` runs.

```bash
make test-eval                                     # no model, GPU or network
make eval-screenspot EVAL_MODELS=eval_models.yaml  # local `swift infer` lane
python -m cua.eval.screenspot run --models-config eval_models_served.yaml --backend openai
```

`eval_models.yaml` drives the local `swift infer` lane (a repo id plus a local `adapters`
directory); `eval_models_served.yaml` drives the served lane, where `model` is the vLLM *served*
name. Both carry an explicit per-spec `coord_space`.

### Module map

| module | holds |
| --- | --- |
| `config.py` | `EvalConfig`, `ModelSpec`, the dataset/bbox/viewport defaults, the parse patterns |
| `dataset.py` | `Sample`, `normalize_bbox`, the lazy `datasets` loader |
| `scoring.py` | `COORD_SPACES`, `parse_point`, `point_in_bbox`, `evaluate`, `aggregate` |
| `predictors.py` | the grounding prompt builders and the `PREDICTOR_BACKENDS` registry |
| `report.py` | the per-spec results file, `comparison.json`, the side-by-side table |
| `smoke.py` | the no-model self-test |
| `cli.py` / `__main__.py` | the `run` subcommand |

`PREDICTOR_BACKENDS`: `swift` shells out to the `swift infer` CLI (mirroring how
`cua.sft.swift_launch` shells out to `swift sft`), so it needs no in-process model imports;
`openai` posts to any OpenAI-compatible `/chat/completions` through a stdlib `urllib` client, so the
harness adds no HTTP dependency to the rented box's `.[eval]` install; `claude` reaches the
frontier arm through the logged-in `claude` CLI, never an API key, and grounding is the one lane
where that comparison is apples-to-apples (single turn, one screenshot, one click point, no
scaffold or tool-use advantage to confound it); `mock` returns canned text for tests. The Claude
arm's budget cap is a runaway guard, not a throttle — every SDK call carries a full-agent cost
floor, so a cap near the true per-call cost returns `""` instead of an answer.

## Student eval: real browser episodes

`cua/eval/student/` runs a served model through real browser episodes and reports whether a
checkpoint beat the base. **The deliverable is a delta, never a single number**: every run takes
`--label` (`before` / `after`), writes `eval_runs/<run_id>/{report.md,results.json,run.log,episodes/}`,
and `compare` renders the before/after table plus a **harness-parity check** that flags any
fingerprint field the two runs disagree on.

```bash
python -m cua.eval.student run --endpoint <url>/v1 --model <served-name> --label before ...
python -m cua.eval.student judge <run_dir>        # score a --no-judge run afterwards, no GPU
python -m cua.eval.student compare <run_a> <run_b>
python -m cua.eval.student sources                # list the registered task sources
make test-student-eval                            # parser round-trip + mock episode + judge + backend surface (Docker)
```

### Module map

| module | holds |
| --- | --- |
| `config.py` | the run configuration and the per-episode record every metric, report and resume reads |
| `backends.py` | `BROWSER_BACKENDS` and `STUDENT_BACKENDS`: where the browser runs and where the model is served |
| `grammar.py` | the one tolerant definition of the student action grammar |
| `episode.py` | one task start to finish: model coordinates into driver calls, plus the two abort rules |
| `run.py` | run orchestration: load the task set, drive an episode per task, judge it, write the artefacts |
| `metrics.py` | the success denominator, the grammar counters, the per-level and per-domain breakdowns |
| `fingerprint.py` | everything that must match for two runs to be comparable, and the wrong-space diagnosis |
| `report.py` | `report.md` and the before/after comparison with its parity check |

Two pieces live outside the package because the RL rollouts need them too: prompt construction and
action execution are the shared student stack in `cua/student/`, and `cua.eval.blocks` holds
`BLOCK_SIGNATURES` — what counts as the site blocking automation.

### Prompt parity

The per-step messages are built by the SAME functions convert uses —
`cua.prompt.format.build_step_observation`, `build_step_messages`, `history_entries_for_step` — with
the same `SYSTEM_PROMPT`, the same `HISTORY_WINDOW`, and the same effect annotations computed from
consecutive PNG screenshots (JPEG is refused, because its compression makes the pixel-identity test
meaningless). `prompt_format` is a fingerprint key.

### Model remote, browser wherever egress needs it

`--endpoint` is any OpenAI-compatible `/chat/completions` — the Modal serve app, a local vLLM, or
`swift deploy`. The browser is a `BROWSER_BACKENDS` entry: `docker` (a local `DockerSandbox`, so the
egress is this machine's), `remote_ssh` (`--browser-host`), or `vast_residential`, which rents a box
whose public IP belongs to a consumer ISP and destroys it on stop.

This matters more than it sounds. Online-Mind2Web is US-centric: only 58% of its 300 tasks are
reachable from a European residential IP, a datacenter VPN trades geo-blocks for WAF-blocks roughly
1:1, and only a residential US exit recovers a real share. **Egress is a parity key carrying the
resolved ASN** — two runs on different egress measure the network, not the model.

### Task sources and pinned task files

Task sources are a registry (`cua.eval.task_sources.TASK_SOURCES`): `online_mind2web` (the gated HF
dataset, 300 tasks, columns `task_id` / `confirmed_task` / `website` / `reference_length` / `level`)
and `held_out` (this project's own never-captured public-site tasks, proven unseen by an on-disk
check against the corpus). Token resolution goes `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`,
`HUGGINGFACE_HUB_TOKEN`, then `HF_READ_TOKEN` from `.env`, then `~/.cache/huggingface/token`; a
401 or 403 fails loudly naming the URL to accept terms at, and never falls back to a stub.

`--tasks-file` pins the exact sample, which is what makes a before/after comparable at all:

| file | n | what |
| --- | ---: | --- |
| `eval_archive/online_mind2web/tasks_pinned_50.jsonl` | 50 | the legacy level-stratified sample |
| `eval_archive/online_mind2web/tasks_pinned_v2.jsonl` | 230 | the 50 above plus 180 stratified, sites hard-blocked from this IP excluded |
| `eval_archive/online_mind2web/tasks_reachable_us_174.jsonl` | 174 | every task whose site loads from a US residential egress |

n≈34 cannot resolve any plausible delta (±8 pp), which is why the sample was widened twice.

### Coordinate space is per model and never auto-detected

`--coord-space` defaults to `normalized_1000`, which a stock Qwen and every sft-v3+ checkpoint
answer in; sft-v1 and sft-v2 checkpoints need `viewport_abs`. Emitted positions are scaled to
viewport pixels before dispatch (scroll deltas are not), `trajectory.json` records the EXECUTED
pixels, and `raw_last_response` keeps the model's own text.

Getting this wrong silently produces a confident wrong number, so the harness warns at the first
impossible coordinate and prints a `COORDINATE SPACE SUSPECT` banner once more than
`EVAL_COORD_SPACE_SUSPECT_RATE` of coordinates fall outside the declared space. In `compare`, a
`coord_space` difference is an expected, annotated model-side difference, not a parity violation.
`--observation` is the same kind of per-checkpoint setting: `v3` (the two-image default) for the
base and sft-v3+, `v2` for sft-v1/v2 and the GRPO adapters over them.

### The action grammar is tolerant by design, and canonical form is its own metric

The parser accepts any **unambiguous** spelling and reports `canonical_form_rate` plus an
`action_form_deviations` breakdown, instead of charging a spelling deviation to task success.
Tolerance is itself a parity key (`action_grammar: tolerant|canonical`, `--strict-action-grammar`).
Accepted spellings, canonical first:

| verb | accepted spellings |
| --- | --- |
| `click` | `click(<ref>(x, y)</ref>, button=left)` · `(x, y)` · `x, y` · trailing `left`/`right` · button omitted |
| `double_click`, `move` | `<ref>(x, y)</ref>` · `(x, y)` · `x, y` |
| `drag` | both endpoints in any coordinate spelling, separated by `->` or `,` |
| `scroll` | `scrollX=`/`scrollY=`, `scroll_x=`/`scroll_y=`, `dx=`/`dy=`, positional deltas, or omitted (defaults `0, 500`) |
| `type` | `type("text")` · `type('text')` |
| `keypress` | `["Enter"]` · `"Enter"` · `Enter` · `Control, a` · `Control+a` |
| `wait`, `done`, `screenshot` | `verb()` · `verb` |
| `navigate` | `navigate("url")` · `navigate(url="…")` · a bare `https?://` token |
| `handoff`, `confirm` | `handoff(reason=login)` · `confirm(propose=call(...), risk=x)`; terminal, scored outcome `handoff` |
| everywhere | an optional `action:` prefix, bullets and backticks stripped, unknown kwargs ignored, floats rounded |

Rejected as genuinely ambiguous: `type(hello world)`, `type(a, b)`, `scroll(500)`, `click(100)`,
`drag((10, 20))`, `keypress()`, `navigate()`, and a scheme-less `navigate(example.com)`. Prose
wrapped around a valid call is `malformed`; narration with no call at all is ignored. The verb set
is part of the fingerprint, so a run recorded before a verb existed compares as a different harness.

### Outcome categories, not one bucket

`completed`, `handoff`, `step_cap`, `stuck`, `parse_failure_abort` and `timeout` are scored.
`site_blocked` (via `cua.eval.blocks.BLOCK_SIGNATURES`: captcha, human verification, Cloudflare, WAF block,
geo block, site error), `unreachable` and `harness_error` are environment failures, excluded from
the success denominator and reported separately.

A signature the set misses is scored as a MODEL failure, which is why the set is a parity key
(`BLOCK_SIGNATURES_VERSION`) and every episode records `final_url` and `final_snapshot`.

Two abort rules end a run that is going nowhere: 3 consecutive steps with no parseable action gives
`parse_failure_abort`, and 5 consecutive steps executing zero actions gives `stuck`
(`--max-parse-failures` / `--max-unproductive-steps`, both in the fingerprint). The driver's own
loop guard is disabled for eval (`--loop-guard-repeats 0`) so that a stuck student is observed
rather than refused.

Metrics split by `level` and by domain. Step efficiency is `steps / reference_length`, and the step
cap is per task — `3x reference_length` clamped to `[15, 50]`, with `--max-steps N` pinning a flat
cap instead.

### The judge

The judge is `cua.pipeline.teacher.verify_trajectory`, criteria-first, on the logged-in CLI through
`run_agent`. `--judge-mid-screenshots` (default 2, `VERIFY_MID_SCREENSHOTS`) controls cost.

The judge is lenient on `meandering` episodes, which only fine-tuned models produce, so its
leniency inflates the after side and not the baseline
(`docs/decisions/judge-consistency-audit.md`). **Quote audited numbers**, and read the final
screenshot of every claimed success before reporting it.

### Not rankable against published figures

Blocked and unreachable episodes are excluded from our scored denominator; the Online-Mind2Web
paper does not exclude them, and it uses a different judge. Always report both denominators —
`report.md` prints both — and treat a comparison against published results as a plausibility check
rather than a leaderboard position. Closing that gap would mean running a published checkpoint
through this harness.
