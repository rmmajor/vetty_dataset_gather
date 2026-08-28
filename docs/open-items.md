# Open items — login-gated capture

Living list. Written 2026-08-27 after the overnight capture session that merged PR #24.
Each item records what it is, why it matters, and the evidence behind it, so it is actionable
later without reconstructing the reasoning.

## 1. The capture host is out of memory (blocks everything else)

Capture cannot run unattended on this machine. Six consecutive runs were killed by macOS memory
pressure after 7, 31, 34, 59 and 10 minutes; every death occurred with free RAM under 25 MB.

Measured: 18 GB physical, **14.1 GB already resident with zero containers**, swap 13.9 GB of
15.4 GB used, ~9.8 GB compressed. A capture container adds ~1.7 GB, which is enough to trigger
the kill each time.

Already done and insufficient: per-sandbox cap 2g → 1500m, tmpfs 384m → 256m, concurrency 3 → 1,
`SANDBOX_VM_RESERVE` → 3g.

Two things checked and ruled out: no system sleep events, and the Docker VM holds only 1.35 GB
resident (its 11.67 GB is an allocation ceiling, not committed memory — lowering that slider
helps far less than it appears).

**Fix:** free ~3 GB of genuinely resident memory on the host — idle Claude sessions (~2.1 GB
across 9 processes), PyCharm, Chrome, Discord. Restart with
`scratchpad/keepalive.sh`, which retries the whole wave sequence up to 24 times. Note that a
supervisor script does NOT survive the kill: it is the parent of the heavy process tree and dies
with it.

## 2. 506 trajectories sit in `unverified`

The convert report skips them, but the bucket mixes two very different things: trajectories where
the agent genuinely failed, and trajectories that were captured but never got a verifier verdict
(a killed wave, an aborted run). The second kind is recoverable for the cost of one verifier call.

`scratchpad/finalize_orphans.py` does exactly this and rescued 20 of 29 on its first run. It only
covers `site_*` dirs — `repo_*` captures need stage 6's own metadata path, and running the
public-site finalizer over them would write `source_repo: public:<name>` and corrupt provenance.

**Fix:** split the bucket before quoting any yield figure; finalize the orphan half.

## 3. The watchdog contradicts itself on the same risk class

Identical action — invite `qa-teammate@web-library.net` — three sites, three verdicts:

| site | verdict | rationale |
|---|---|---|
| baserow | deny | "external communication to a **potentially real** recipient" |
| leantime | approve | "the designated QA teammate burner account, **not a real** person" |
| toodledo | approve | "a **controlled QA inbox** (not a real person)" |

The watchdog is given `{goal, screenshot, proposed_action, risk_category, autonomy, auth_mode,
env}` but is never told which recipients are burner-controlled, so it guesses. The corpus is
skewed 22 approve / 5 deny as a result, and — worse for training — labels the same situation both
ways on the exact behaviour the dataset exists to teach.

**Fix:** pass the burner registry (or a "these addresses are ours" set) into `ApprovalPolicy`, then
re-capture the ~8 invite tasks. A genuine deny then needs a task targeting something actually
external.

## 4. `cf_clearance` does not transfer into the container

Tested, not assumed: a Cloudflare-cleared session saved from the host browser was re-challenged
immediately inside the sandbox (new Ray ID per attempt). `browser_driver.js:710` builds the context
as `{viewport, deviceScaleFactor}` and **never sets a User-Agent**, so the container's UA differs
from the real Chrome the cookie was issued to, and Cloudflare binds `cf_clearance` to UA + IP.

Setting the UA is two lines but probably insufficient on its own — Cloudflare also fingerprints TLS
and headless signals. Worth one experiment because it would unlock 36 Ukrainian sites (173 tasks)
that currently require human provisioning.

## 5. Trajectories do not record which teacher model produced them

PR #24 merged on 2026-08-27 moved the pipeline from `claude-sonnet-4-6` to `claude-sonnet-5`
(`LATEST_MODEL_BY_CLASS`). Everything captured before that merge used 4.6; everything after uses 5.
`metadata.json` has no teacher-model field, so the only way to tell them apart is file mtime versus
the merge timestamp.

**Fix:** record `MODEL_FOR_TEACHER_AGENT` in `metadata.json` at finalize time.

## 6. Deferred work, with reasons

- **14 repo-lane PII re-captures** — `repo_mauriceboe_TREK` (7), `ai-qa-orchestrator` (2),
  `uptime-kuma`, `SSM`, plus 3 `auth_required` in `e2e-nextjs-playwright`. Needs a registry status
  reset to `validated` and a stage-6 pass per repo: clone, npm install, boot. Roughly an hour of
  container time for 14 trajectories.
- **~364-task re-capture sweep** — previously-failed tasks that resume now re-opens. Wave 6 of the
  overnight script runs it with `CAPTURE_MAX_ATTEMPTS=1`; most are known captcha/OAuth walls.
- **Disposable-email domain** — see `superpowers/specs/2026-06-28-post-gather-sft-todo.md`.
  A cheap domain with catch-all forwarding unblocks teable, scribbles, outline-wiki and likely a
  share of the mainstream set, while keeping the burner-only invariant intact.
- **`provision_human.SOURCE_CONFIGS`** is still a hardcoded 2-entry list and only *warns* on an
  unknown site name; it should adopt `public_capture.known_site_configs` / `resolve_named_sites`
  so the human lane cannot silently drop a named site either.
- **`stage6_generate._agent_capture_complete`** (`:169`, `:339`) still gates on
  `_is_valid_trajectory`, so the repo lane retains the "a failed task counts as recorded" bug that
  was fixed for the public lanes. Left alone deliberately: that lane has its own
  `AGENT_MAX_CONSECUTIVE_FAILS` semantics and re-capture there reboots a whole repo sandbox.
- **`think_generator.render_action_nl`** has no `confirm`/`handoff` branch, so the polish prompt
  describes guarded steps to the model as the bare words `confirm` / `handoff`. Two lines; worth
  doing before the guarded lane scales further.
- **`task_proposer.propose_tasks`** is not policy-aware (spec §5 wants `autonomy`/`auth_mode`).
  Harmless while guarded runs use `--no-propose`; wrong if proposing is ever enabled there.

## 7. Human-only queue

36 Ukrainian sites / 173 tasks behind anti-bot walls, led by rozetka (12 tasks), auto.ria (9),
silpo (8), novaposhta (8), dom.ria (7). Cleared only by a person in a real browser — see item 4 for
the one experiment that could shrink this list. A further 11 UA sites are phone-gated and are
permanently out of scope under the burner-only invariant.

## 8. RL / GRPO — what would move the score

Written 2026-08-28 after the first live GRPO run (`docs/decisions/grpo-v2-first-live-run.md`:
sft-v1 9.4% → grpo-v2 11.8% verified on the archived 50-task Online-Mind2Web sample, inside
noise). The harness is done; the ceiling is what it is fed. Ranked by expected effect per dollar.

1. **Train on the student's frontier.** 8 of the 12 train tasks were all-fail groups and gave no
   advantage; the run learned from 3–4 tasks. Before the next run, roll the SFT policy k=8 over a
   pool of ~60 short, idempotent tasks on sites outside the eval splits (the proposer can author
   them) and keep tasks with 20–70% success. ~$3 of serve and judge calls.
2. **Parse-failure aborts are the largest recoverable loss** on Online-Mind2Web — 13/50 tasks for
   SFT and GRPO alike: on 403/blocked/error pages the student narrates and never emits an
   `action:` line. Cheapest fix is SFT data (a few dozen error-page examples answered with
   `done()` / `navigate`; the teacher never met the walls the student hits). Second: put
   blocked-page tasks in the RL pool so the −0.5 parse penalty is exercised; the loop guard ended
   every such episode first this time.
3. **Group size 8, rollout temperature 1.0.** At 10–30% success a group of 4 is often
   zero-variance; 8 samples roughly doubles the share of groups with gradient at the same
   16-episode step cost. Config only (`num_generations`, `temperature`).
4. **`success_check`s for the frontier tasks** (`rl/checks.py` registry) so rewards stop
   depending on the judge's throughput, which is what caps steps per hour; the loop guard is
   already a service knob defaulting to 5.
5. **40–60 steps** (~$15–25 at the observed ~$0.9/step) once 1–4 are in place; leave β 0.01 and
   lr 1e-5 alone until a curve exists.
6. **Measure with enough episodes.** n≈34 scored at ~10% success is ±8 pp; a real delta needs
   ~150 scored tasks per side (~1.5 h, ~$5 per side). Serve the after-adapter on the merged SFT
   base (`CUA_SERVE_MODEL`), never on raw `Qwen/Qwen3.5-9B`.

Outside RL but bounding the ceiling: 28% of the eval sample is site-blocked from the sandbox IP
(excluded from the score, a hard cap on the all-episode rate), and the judge is an LLM reading
three screenshots — the right instrument for a before/after under one rubric, not for absolute
claims (`docs/decisions/judge-consistency-audit.md`).
