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
| `artifacts_path` | where the run's raw records were written |
| `artifacts_availability` | whether a reader of this repository can actually open them |
| `status` | `done`, `pending`, `running`, `planned`, `failed`, `superseded` |

`artifacts_availability` exists because most `artifacts_path` values point at gitignored working
directories, so a path alone does not say whether the evidence survives a fresh clone:

- **`committed`** — the records are in this repository, under `eval_archive/` or a tracked `docs/`
  file. Reproducible by anyone who clones it.
- **`external:vetty`** — the run was recorded in the companion `vetty` repository; the number is
  auditable there, not here.
- **`local-only`** — the records are in a gitignored working directory (`eval_runs/`,
  `eval_results/`, `results/`, `rl_runs/`, `output/`, `trajectories/`) or a local-only note. They
  exist on the machine that ran the eval and nowhere else, so quoting the row outside this project
  means quoting an unauditable number. A bare `results/*` path on a `qa-bench-refapp` row names the
  companion `vetty` repository's results directory, on the machine that ran the benchmark; it is
  classified `local-only` here because nothing about it is auditable from this repository.
- empty — no artefacts were recorded for the row (typically a `planned` row).

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
