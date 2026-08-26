# Why this dataset captures on real sites, not controlled backends

*Rationale note (raised 2026-06-28). Captured for reuse when writing up the work — this is the
core data-philosophy argument behind the login-gated capture design.*

## Thesis

For a computer-use agent (CUA) meant to be **deployed** against the web people actually use,
training trajectories must be captured on the **real third-party sites** the agent will face at
deployment (Gmail, Amazon, LinkedIn, real SaaS dashboards, real booking flows). Capturing on
self-hosted, seeded, deterministically-resettable clones of those apps optimizes the wrong
objective and should not be the primary training environment.

## The two framings, and why they diverge

There are two distinct objectives that look similar but pull apart:

1. **Benchmark reproducibility** — you want a fixed, resettable world so that a score is
   comparable across runs, models, and papers. Self-hosting a seeded app with deterministic reset
   is the standard move *because the metric demands determinism*. Essentially every reproducible
   agent benchmark (WebArena, VisualWebArena, AppWorld, tau-bench, WorkArena, ST-WebAgentBench)
   provisions auth and state this way.

2. **Deployment usefulness** — you want an agent that completes real tasks for a user on the live
   web. The metric is task success in the wild, on sites that drift, re-skin, A/B-test, and gate
   behind real auth.

The mistake is importing (1)'s environment choice into the *training data* for (2). The benchmark
community self-hosts to make **scoring** reproducible — that is an **evaluation-time** property.
It says nothing about what training distribution produces a useful deployed policy.

## Why controlled backends don't transfer

A CUA grounds its actions in **pixels + DOM layout + interaction affordances** of a specific UI.
Train it only on a self-hosted clone (Saleor instead of Amazon, Roundcube instead of Gmail) and it
learns *that clone's* layout, spacing, control vocabulary, and flow — not the real target's. The
visual/interaction distribution at deployment is the real site, which the model never saw. The
distribution shift is exactly the part that matters for grounding. "Trained only on controlled
backends" is therefore a recipe for an agent that benchmarks well and deploys poorly.

The transferable signal we actually want — multi-step authenticated tasks, form-filling,
validation-error recovery, pausing before irreversible actions — is best learned **on the real
surfaces** where it will be exercised, not on stand-ins that share none of the real pixels.

## The reproducibility counterargument, and why it doesn't bind training

The strongest case for controlled backends is **drift**: live sites change, so captured
trajectories rot (e.g. ~47% of Mind2Web tasks have been invalidated by site changes over time),
and programmatic state-verification ("did the order actually get placed?") is only clean when you
own the backend.

Both are real, but both are **eval-time** concerns, separable from training:

- **Drift degrades *labels over time*, not the *pixels at capture time*.** A trajectory recorded on
  the real site on day _t_ is a faithful sample of the real deployment distribution at _t_. It
  remains valid supervision for grounding even after the site later changes; we are not re-running
  it. Staleness is managed by recapturing, not by switching to a synthetic distribution.
- **Verification doesn't require owning the backend.** Success can be judged from the final
  screenshot/DOM by a verifier model (which is what this pipeline does), decoupling "was the goal
  achieved" from "do I control the database."
- **Reproducible scoring is an evaluation harness's job.** If we want a stable leaderboard number,
  build a held-out eval on controlled mirrors *for measurement* — without contaminating the
  *training* distribution. Optimizing the training set for the convenience of the scoreboard is the
  inversion to avoid.

## Implication for this dataset

- **Primary capture environment = real third-party sites**, including login-gated ones — that is
  where day-to-day value lives and where grounding must transfer.
- **Login is implemented to mirror shipping products, not to manufacture a benchmark world**:
  trajectories carry the real-site auth context (session inheritance / `handoff` at login walls),
  mask credentials as `<secret:NAME>` placeholders so nothing leaks into weights, and pause via
  `confirm` before the single irreversible step — the behavior deployed CUAs actually exhibit.
- **Controlled/self-hosted apps, if used at all, belong in the *eval* harness** for reproducible
  measurement, never as the main training distribution.

## One honest caveat

This is not "never touch a controlled environment." Fully-autonomous, commit-everything behavior
(the QA-agent use case) genuinely needs a safe sandbox to *execute* irreversible actions during
capture. The point is narrower and load-bearing: **the bulk training distribution must be the real
deployment distribution.** Sandboxes are for the slice of behavior that cannot be safely recorded
on a live site — not the default world.
