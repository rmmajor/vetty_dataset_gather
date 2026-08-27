# Experiment registry

Two CSVs are the single source of truth for every training run and every measured number
across both repositories (this one and `vetty`). The root `README.md` renders them; never
edit that rendered block by hand.

    make registry          # fill intervals, re-render the README block
    make registry-check    # CI guard

## `models.csv` — what was trained

| column | meaning |
| --- | --- |
| `model_id` | stable key referenced by `results.csv` |
| `kind` | `base`, `sft`, `grpo`, `frontier` (external API model), `tool` (non-model comparison arm) |
| `dataset_version` | corpus the model was trained on; empty for base and frontier rows |
| `dataset_size` | training examples, not trajectories |
| `hyperparams_ref` | path to the config that defines the run, rather than inlined values |
| `train_commit` | commit of this repo at training time |

## `results.csv` — what was measured

One row per model × eval × subset × metric × repeat. Long format, so a new eval or dataset
version adds rows and never reshapes the table.

| column | meaning |
| --- | --- |
| `eval_version` | dataset revision or benchmark catalog hash; two rows are comparable only if this matches |
| `subset` | slice within the eval (`all`, `icon`, `easy`, `6-defect-subset`, ...) |
| `n` | sample size behind the value, used to compute the interval |
| `ci_low` / `ci_high` | 95 % Wilson interval, filled by `--fill-ci`; never hand-written |
| `judge_model` / `judge_prompt_sha` | provenance for any LLM-adjudicated metric |
| `harness_commit` / `config_hash` | provenance for the run itself |
| `status` | `done`, `pending`, `running`, `planned`, `failed`, `superseded` |

`status` carries meaning that a value alone cannot:

- **`pending`** — the run finished but the metric has not been computed (for example, a
  benchmark run whose judge pass has not been executed). The value column stays empty
  rather than being filled with a zero.
- **`superseded`** — the number exists but must not be quoted. Keep the row and say why in
  `notes`, so a retired figure cannot quietly return to a slide.
- **`planned`** — on the worklist. These rows make the registry double as the experiment plan.

## Adding a proportion metric

Put the metric name in `PROPORTION_METRICS` in `render_registry.py`. That is what makes the
renderer show it as a percentage and give it an interval; a metric missing from the set is
rendered as a plain number and silently gets no interval.
