# Quality-First Funnel Intake — Design

**Date:** 2026-06-14
**Status:** Design approved; pending implementation plan
**Scope:** Path B repo pipeline (`repo_pipeline/`), stages 1–2 + a one-time backlog re-gate

## Problem

The funnel is "cast wide, reject aggressively." In practice it floods with repos that are
destined to be rejected, and most of the rejection happens *after* expensive LLM stages have
already run on them. Measured from the `agent_calls` cost ledger (project-to-date, ~$660 total):

| purpose | calls | cost | note |
|---|---|---|---|
| `teacher_task` (stage 6, the real data) | 556 | $309 | only ~19 repos ever yielded trajectories |
| `shallow_score` (stage 3, LLM) | 4,181 | $184 | 2,732 (65%) rejected *at this stage* |
| `deep_analyze` (stage 4, LLM+sandbox) | 525 | $91 | dominant reject reason: *"not a runnable interactive web app"* |

~$275 of LLM spend went into scoring/analyzing repos that were non-bootable test suites from the
moment stage 1 admitted them. The dominant stage-4 rejection — "not a runnable interactive web
app" — is knowable from repo metadata + a single file (`package.json`) **without** an LLM or a
clone.

### Root cause

Discovery (stage 1) was designed for **replay** mode: find repos that *contain Playwright tests*
to instrument. The pipeline now runs in **agent** mode: a teacher drives a *booted app*, and the
repo's own tests are only hints. So the thing we actually want is "a bootable interactive web
app," but stage 1 queries for "a repo that has a test file." That mismatch admits thousands of
un-bootable test suites (tutorials, practice repos, suites targeting *external* sites like
saucedemo.com) that aren't unmasked until stage 4.

## Goal

Maximize the **quality and efficiency** of what enters the funnel, not quantity. Reject
non-bootable / test-suite-only repos *before any LLM spend*. Target operating point: a
few-hundred high-quality trajectories, sourced from repos that are actually bootable interactive
action-UIs.

Non-goal: maximizing repo count. Non-goal: booting non-Node app servers (already out of scope in
`app_runner.detect_app_start`).

## Architecture

Four components. Stages 1–2 are the only no-LLM stages, so all new gating lives there.

### Component 1 — Discovery realignment (`stage1_discover.py`)

Pivot queries from "has Playwright tests" to "is a bootable interactive app":

- **Remove** the broad generic queries (`playwright language:TypeScript|JavaScript|Python`,
  `playwright test … stars:>N`). These are the flood source.
- **Keep + expand** the targeted self-hostable-app queries (the cal.com / n8n / directus block).
- **Add topic-driven app queries**, config-listed, each gated by stars + language:
  `topic:self-hosted`, `topic:low-code`, `topic:cms`, `topic:headless-cms`, `topic:crm`,
  `topic:erp`, `topic:dashboard`, `topic:admin-panel`, `topic:project-management`, `topic:kanban`,
  `topic:e-commerce`, `topic:web-app`. Topics surface *apps*, not test suites.
- **Curated seed list** — `repo_pipeline/curated_apps.json`: a list of `owner/name` of known-good
  bootable apps, injected directly as `discovered` (bypassing search). Bootstrap from (a) the ~19
  repos that already produced trajectories (proven-good) and (b) a hand-picked self-hostable set.
  The file is data; it is extended over time.

### Component 2 — Bootable-app gate (`stage2_filter.py`, no-LLM)

Stage 2 keeps its existing metadata filters (size / recency / fork / duplicate-description), then
runs a deterministic **file-probe** on survivors before promoting to `filtered`.

**Probe (per repo):** one GitHub GraphQL call fetches
- the root tree entries (names + types),
- the root `package.json` blob text,
- (v1.1) for monorepos — `apps/*` and `packages/*` one level deep when the root `package.json`
  declares `workspaces`.

**Verdict (deterministic, config-driven):**

ACCEPT when any of:
1. a `package.json` has a run script (`dev`/`start`/`serve`/`preview`) **and** a frontend-framework
   dependency (`react`, `vue`, `svelte`, `@angular/core`, `next`, `nuxt`, `vite`, `astro`,
   `@remix-run/react`, `solid-js`, `preact`, `gatsby`, `@sveltejs/kit`, …);
2. a `docker-compose` / `compose` file is present (self-hostable app);
3. a known app-config file (`next.config.*`, `nuxt.config.*`, `vite.config.*`, `svelte.config.*`,
   `astro.config.*`, …) is present **and** a run script exists.

REJECT → `rejected_filter` (new `FailureCategory.NOT_BOOTABLE`) when:
1. no `package.json`, no compose file, and no recognized app-structure dir (`src`/`app`/`apps`/
   `client`/`frontend`/`web`/`packages`) → not an in-scope Node web app;
2. a `package.json` exists but has **no frontend-framework dependency** (the decisive signal),
   especially when its only relevant deps are **e2e test tooling** (`@playwright/test`,
   `playwright`, `cypress`, `selenium-webdriver`, `@wdio/cli`, `puppeteer`) → a test suite, not an
   app. (Note: `jest`/`vitest`/`@testing-library/*` alone do NOT mark a repo as a test suite —
   real apps carry unit-test deps. The decisive criterion is *absence of a frontend framework*.)

Nuanced cases still fall through to the LLM stages (3/4) — but on a much smaller, pre-qualified
set. The gate only removes high-confidence junk.

### Component 3 — Backlog re-gate (one-time)

Run the same probe over the existing `filtered` (9,336) and `scored` (1,675) repos. Reject the
non-bootable ones (→ `rejected_filter` with `NOT_BOOTABLE`), keep plausible apps. Implemented as a
`--regate` orchestrator mode (or a one-off script using `registry` + the shared probe function).
This stops the flood from ever reaching LLM scoring again and reclaims the genuine apps.

### Component 4 — Config (per "generalize, don't special-case")

All knobs are data, not `if name ==` checks — a single module-level config (or JSON) drives the
probe and discovery:

```
FRONTEND_FRAMEWORKS  = [...]   # accept signal
RUN_SCRIPTS          = ["dev", "start", "serve", "preview"]
E2E_TEST_TOOLING     = [...]   # corroborates "test suite"
APP_CONFIG_FILES     = [...]   # accept signal
COMPOSE_FILES        = [...]   # accept signal
APP_STRUCTURE_DIRS   = [...]   # presence test
DISCOVERY_TOPICS     = [...]   # stage-1 query generation
```

Adding a framework, topic, or config file = a data edit, no logic change. The probe verdict is one
pure rule-driven function: `classify_bootable(root_entries, package_jsons, compose_present) ->
(accept: bool, reason: str)`, unit-testable in isolation from any network call.

## Data flow

```
stage1 discover  → discovered      (targeted queries + curated seed; no generic flood)
stage2 filter    → filtered        (metadata filters → bootable-app probe → ACCEPT)
                 → rejected_filter  (metadata fail OR probe REJECT: NOT_BOOTABLE)
stage3 shallow_score → scored      (LLM, now on pre-qualified input)
... unchanged downstream ...

backlog re-gate (one-time): {filtered, scored} → probe → keep | reject NOT_BOOTABLE
```

## Error handling

- **Probe network/GraphQL failure, repo gone, no default branch:** treat as a *retryable infra*
  failure (consistent with the pipeline's "infra failures never reject" rule) — leave the repo at
  its input status for the next run. Do NOT reject on a transient fetch error.
- **`package.json` present but unparseable JSON:** treat as a missing `package.json` for signal
  purposes (fall through to the other accept/reject signals).
- **Rate limiting:** the probe is ~1 GraphQL point/repo; throttle and respect GitHub's rate-limit
  headers. The backlog re-gate (≈11k calls) runs over a few throttled hours, one-time. Batchable
  (multiple repos per GraphQL document) as an optimization, not required for v1.

## Testing / validation

There is no unit-test suite in this repo; validation is empirical and must run *before* the gate
is allowed to reject anything:

1. **Precision floor (zero false negatives on proven-good):** run the probe in dry-run over the
   ~19 repos that already produced verified trajectories. **All must ACCEPT.** Any rejection is a
   gate bug to fix before rollout.
2. **Recall check (catches junk):** run the probe in dry-run over a sample of current
   `rejected_score` / `rejected_analyze` repos. The large majority should REJECT, confirming the
   gate would have caught them at stage 2.
3. **Backlog dry-run report:** before applying, print accept/reject counts + sample reasons over
   the full backlog so the kept/purged split is reviewable.
4. Smoke: `make test-driver` / a small `--stage 1-2 --limit N` run still completes.

## Risks & mitigations

- **False negatives (good app rejected).** Non-Node apps (Django/Flask/Go/Rails) lack a JS
  `package.json` — but agent-mode's `detect_app_start` already declares non-Node app servers out
  of scope, so rejecting them is *correct today*. Monorepos handled by the workspaces deep-probe.
  Bar is "keep plausible," not "hard purge." The precision-floor test (validation #1) is the
  guardrail.
- **Server-rendered Node apps with no frontend-framework dep** (e.g. Express + EJS/Pug, a Node app
  serving HTML templates) would fail accept-signal #1 and, lacking a compose/app-config file, be
  rejected — a genuine false-negative class. Mitigations: (a) the precision-floor test catches it
  if any proven-good repo is shaped this way; (b) add an accept signal for `RUN_SCRIPTS` +
  server-framework dep (`express`/`fastify`/`koa`/`nestjs`) + a template-engine dep
  (`ejs`/`pug`/`handlebars`) — a one-line config addition once validation #1 shows it is needed;
  (c) the curated seed list backstops any individually-known case.
- **GitHub API cost.** ~1 GraphQL point/repo; backlog is one-time and throttleable.
- **Lost LLM nuance.** Preserved — the LLM stages remain, just on pre-qualified input. The gate
  only removes high-confidence junk; borderline calls still reach the LLM.

## Out of scope

- Non-Node bootable apps (Python/Go/Rails) — out of scope until `detect_app_start` supports them.
- Multi-service (frontend+backend) boot — separate effort (surfaced by the TREK case).
- Changes to stages 3–7 logic (they simply receive higher-quality input).

## Expected effect

`shallow_score` + `deep_analyze` spend drops sharply (the test-suite/non-app majority never reaches
an LLM). Spend concentrates on `teacher_task` — the actual trajectory data. Discovery yields fewer
but higher-hit-rate candidates.
