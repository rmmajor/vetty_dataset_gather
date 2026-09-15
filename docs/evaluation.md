# Evaluation

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

**`coord_space` is per MODEL, never global** (`ModelSpec.coord_space`). Different checkpoints answer
in different coordinate conventions: a stock Qwen and every sft-v3+ checkpoint emit
`normalized_1000` (0-1000 of the image), while the sft-v1/v2 checkpoints emit `viewport_abs`,
absolute pixels of the fixed 1280x720 training viewport. `native_abs` reads the answer as pixels of
the source image, for other models. Measured: the SAME base-model predictions score **85% as
`normalized_1000` and 25% as `viewport_abs`** — a before/after scored with one global
`--coord-space` silently mis-scores one side of the comparison. A coordinate space maps points in
both directions and NEVER touches the image: when it resized, declaring `viewport_abs` squashed the
input, and the two arms were then scored on different pictures.

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
