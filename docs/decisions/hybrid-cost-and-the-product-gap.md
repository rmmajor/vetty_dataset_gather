# What counts as cost, and the gap between the architecture and the shipped product (2026-09-05)

## The principle

**Count what a user pays to run the tool. Do not count what only we pay to grade it.**

A cost figure in this project answers one question: *if someone adopted this system, what would a test
case cost them?* Every step that a user's run must perform is in. Every step that exists only so we can
score ourselves is out.

Applying it:

| step | in the cost? | why |
| --- | --- | --- |
| explore, plan | **in** | a user's run performs them (amortised once per application) |
| execute | **in** | a user's run performs it |
| conclude — pass/fail plus the bug report | **in** | this is the deliverable the user reads |
| match the bug report against the *injected* defect | **out** | only we know what defect was injected |

Two consequences worth stating because they are easy to get wrong:

- **Grading is not a phase of the product.** `benchmark/judge.py` compares a reported bug against the
  seeded defect catalogue to compute DDR. Nobody outside this project can run it — a real user has no
  catalogue. It is tracked in `judgements.jsonl`, never in `phase_costs` or `total_cost_usd`.
- **Concluding IS part of the product, even though our code for it lives in the benchmark package.**
  `benchmark/verdicts.py` never sees the injected defect; it reads the trace and the final page and
  writes the user-facing verdict and bug report. Where a function happens to live is not the test —
  what the run needs is.

The test to apply to any future number: **would a customer's run still have to do this?** If yes it is
cost; if no it is apparatus. Whichever way a figure falls, say which side it is on when reporting it.

## The number this yields today

End-to-end cost of one test case, USD. The hybrid column is the architecture this work proposes:
the local model executes, a frontier model draws the conclusions.

| item | frontier (Sonnet) | hybrid (SFT-2 + frontier verdict) |
| --- | --- | --- |
| execution | 1.17 | **0.015** |
| report / verdict | 0.13 | 0.87 |
| **total per test case** | **1.30** | **0.89** |
| exploration + planning, once per application | 3.68 | 3.68 |

**Hybrid is ~1.5x cheaper end to end.** Execution itself falls ~78x (1.17 -> 0.015), but the frontier
conclusion step, a small term while execution dominated, now dominates in its turn.

The execution figure is **derived, not self-reported**: `phase_wall_seconds.executing` x the Modal L40S
rate ($1.95/GPU-hour), via `benchmark/cost.py`, which was dead code until it was wired into the report
on 2026-09-04.

## What must NOT be in the table

`benchmark/judge.py` matches a reported bug against the *injected* defect catalogue to compute DDR.
That is measurement apparatus — nobody outside this project runs it. It costs ~$0.45/run and is
tracked in `judgements.jsonl`, **never** in `phase_costs` or `total_cost_usd`, so no cost figure in the
thesis has ever included it. Keep it that way.

The **verdict** (`benchmark/verdicts.py`) is a different call and IS product: it never sees the injected
defect, it reads the trace and the final page and produces the user-facing pass/fail plus the bug
report. Under this architecture the hybrid genuinely pays for it.

## The gap: the product does not implement the architecture

The architecture is *local executes, frontier concludes*. The benchmark implements it — local CUA loop,
then `produce_cua_verdict` on Sonnet. **The shipped product does not.**

In `src/vetty/infrastructure/agent/local_model_agent_adapter.py` every phase routes to the local model,
including `_convert_text_to_structured_output`, which is what turns a run into verdicts and bug reports:

```python
"model": model_override or self.settings.local_model_name,   # _run_structured_output_call
```

There is no frontier call after the execution loop. The local model judges itself — the exact failure
mode ch3.md documents, where a small model asserts a deletion that never happened.

This is not a regression introduced by the measurement work. `git log --diff-filter=A` dates the file to
`785c45f` (2026-03-13, *"feat: local models integration (untested)"*), and line 436 of that first commit
already routes structuring to `local_model_name`.

Consequence for the corpus of runs: **no local-backbone run on disk has ever produced a report**
(`report= False` on every record), because they all ran ``--record-only`` and handed off to the bench
verdict. There is no example of a locally-written report to inspect, and the product's reporting phase
has never been exercised for a local arm.

## Actions

1. **Report the hybrid configuration** (local execute + frontier verdict) as the architecture's cost,
   which is what the table above does. Do not report the all-local product path as if it were the
   proposal — it is cheaper (~$0.015/case, no frontier call at all) but it ships the self-report failure.
2. **Measure the hybrid end to end.** The product needs a frontier model for the conclusion steps before
   this is a measurement rather than a composition of two runs. The hook already exists:
   `_run_structured_output_call` accepts `model_override` and nothing passes it for this purpose.
3. Until (2) lands, state plainly that the hybrid figure composes a measured local execution cost with a
   measured frontier verdict cost, rather than one instrumented end-to-end run.

## Instrumentation fixed while establishing this

- `benchmark/verdicts.py` recorded `input_tokens=0, output_tokens=0` next to a cost copied from the
  Claude CLI, so `pricing.yaml` could never check it. Real token counts are now stored; the verdict
  derives to **$0.247/case** against the CLI's **$0.618**, a consistent ~2.5x CLI inflation also seen on
  `econ-sonnet` ($157.95 reported vs $63.18 token-derived). Both arms inflate by the same factor, so the
  ratio the thesis argues is unaffected.
- The verdict cache replayed `0.0` for every hit, so a partially-cached sweep reported a fraction of what
  judging costs. It now stores and replays the usage that produced it.
- The verdict call did not set `setting_sources=[]`, so every judgement loaded the operator's CLAUDE.md,
  skills and plugins — measured at 111k cache-write tokens per call against a ~1.5k prompt. That both
  inflated the cost and contaminated the judge with local configuration. Fixed.
