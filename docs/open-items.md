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

## 8. RL / GRPO — what would move the score (closed 2026-09-02 by §10)

Item 1 below was done for `grpo-v3` and the full 50-step run it was meant to de-risk (item 3) went ahead; the score did not move, see §10.

Rewritten 2026-08-31. The `grpo-v2` run is now understood as a **dead run**, not a weak one: its
LoRA-B norm is 1.9% of sft-v1's, because 10 of its 12 task groups had no reward contrast and 20% of
its episodes trained on a fabricated `0.0`. Everything in the "pipeline" block below is fixed and
proven (`docs/decisions/grpo-v3-pipeline-fixes.md`); what remains is data and spend.

**Done (2026-08-31, branch `feat/grpo-v3`).** Verdictless episodes can no longer train
(`require_reward`); a judge-free run refuses to start unless every task has a `success_check`;
milestone partial credit orders failed groups; the false-done inversion is fixed; oracles can see
the page's rendered text (`/page_text`, since `/snapshot` shows only headings and control names);
`navigate` is legal in the eval harness again (it is in the trained grammar and was scored a parse
failure); the RL decoder tolerates a prefilled `<think>`, which had been discarding every thought and
marking every turn unclean; G=8, temperature 1.0, `dynamic_sample` + `max_resample_times`,
`epsilon_high`, `max_turns 32`, 50 steps; `rl/analysis.py` gives per-task pass@k, reward histograms,
zero-variance-group share and the LoRA-B dead-run detector.

**Ranked, still open:**

1. **Finish frontier selection and keep it fresh.** `repo_pipeline/public_sites_actions_frontier.json`
   holds 51 oracle-checked tasks in two tiers (4–12 and 10–16 expected actions). A k=1 pilot on
   merged sft-v1 scored 51.5% pool-wide with zero deferred rewards, so the pool straddles the
   frontier; the k=8 sweep picks the 20–80% band. Task difficulty moves with the policy, so re-sweep
   the selection (k=4, ~$2) against any new checkpoint before training on it.
2. **Parse-failure aborts remain the largest recoverable benchmark loss** — 13/50 Online-Mind2Web
   tasks for SFT and GRPO alike. Two independent fixes now exist: the harness one (navigate, above,
   free) and the data one (error/blocked-page examples in the SFT corpus — see
   `docs/sft-corpus-gaps.md` gap #1). RL cannot fix it: on a wall every action scores alike, so there
   is no gradient toward "say `done()` properly".
3. **Prove learning before paying for it.** The remaining unproven link is that real grouped rewards
   move the weights — the Modal smoke proved mechanics against a mock env only. A ~10-step run on
   selected tasks, watching ms-swift's `frac_reward_zero_std` and the LoRA-B ratio, costs ~$5 and
   de-risks the full run. Kill any run whose B-norm is under 10% of the SFT adapter's after ~10 steps.
4. **Budget is now the binding constraint, not the harness.** Remaining free credits are roughly
   $15 (bombaleylo2) + $4 (mknssh) after the sweep; a 50-step run plus a 150-task before/after eval
   does not fit. Either shorten the run, or accept that the next honest before/after needs funding
   beyond the free tier. Never bill beyond the free credits.
5. **Measure with enough episodes.** n≈34 scored at ~10% success is ±8 pp; a real delta needs ~150
   scored tasks per side (~1.5 h, ~$5 per side). Serve any GRPO adapter on the **merged SFT** weights
   (`CUA_SERVE_MODEL`), never on raw `Qwen/Qwen3.5-9B` — its `base_model_name_or_path` is the merge.

Outside RL but bounding the ceiling: 28% of the eval sample is site-blocked from the sandbox IP
(excluded from the score, a hard cap on the all-episode rate), and the judge is an LLM reading
three screenshots — the right instrument for a before/after under one rubric, not for absolute
claims (`docs/decisions/judge-consistency-audit.md`).

## 9. The verification loop is missing from the corpus (closed as an RL gate 2026-09-02 by §10)

The corpus gap itself stands and is the subject of `docs/decisions/sft-v3-plan.md`; what is
closed is "do this before RL", since there is no RL run to gate.

Written 2026-09-01 from the QA-benchmark measurement of `sft-v1` plus a scan of our own corpus:
only **16.1%** of done() steps follow any verification thought, and only 11.6% of trajectories
re-open a page to check what they changed (`python -m repo_pipeline.verify_pattern_scan
trajectories`). The `false_done` penalty in `rl/reward_config.json` cannot teach a behaviour the
SFT distribution never demonstrates, so item 8 is gated on this. Full reasoning, tranche sizing and
success bar: `docs/decisions/sft-v3-plan.md`.

## 10. RL dropped (2026-09-02)

`grpo-v3` (`docs/decisions/grpo-v3-results.md`) ran with every §8 fix that was config or data:
oracle-only rewards, a measured frontier band, G=8 at temperature 1.0, 50 steps, 752 episodes.
The trainer had reward contrast at every step and the adapter moved (‖B‖ 0.5557, 3.1% of sft-v2),
and train-pool success went 59.4% → 50.0% between the first and last 160 episodes while the
false-done share rose 27.5% → 38.8%. The within-group spread on live sites is mostly rate limits,
CAPTCHAs, timing and sampling luck, and 800 episodes cannot average it out; a run that could
resolve a real gain needs ≥3k episodes, per-turn credit, the parity prompt, `false_done ≥
step_cap` and a pool at the benchmark's distribution, which is outside the remaining budget.

Decision: RL is out of the paper; v1/v2/v3 are reported as a diagnosed negative result; no
rollout enters SFT (`excluded_reason: rl_rollout`, under `rl_runs/`). §8 and §9's RL gate are
closed by this; the §9 corpus gap continues in `docs/decisions/sft-v3-plan.md`. Registry:
`cua-grpo-v3` in `experiments/models.csv`, `grpo-v3-*` rows in `experiments/results.csv`,
`qa-ddr-grpo` superseded.
