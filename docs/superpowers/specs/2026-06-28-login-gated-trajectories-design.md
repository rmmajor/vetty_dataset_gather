# Login-Gated Trajectories — Design Spec

**Date:** 2026-06-28
**Branch:** `feat/login-gated-trajectories`
**Status:** Approved design, pending implementation plan

## 1. Motivation

The ultimate goal is to train the **best open-source computer-use agent (CUA)** — one that is
*useful to real users*, not merely autonomous on a benchmark. The most valuable web tasks
(booking, posting, buying, configuring a SaaS dashboard, sending) live **behind a login**, and
today the pipeline cannot touch them: the site registry has **174 auth-tagged tasks across 140
sites**, but `auth` is purely a *veto reason* — stage 4 rejects "functionality entirely behind a
third-party login." There is no credential mechanism anywhere in the codebase.

### The research finding that shapes this design

A survey of production CUAs (OpenAI Operator / ChatGPT agent, Anthropic Claude for Chrome,
Google Project Mariner / Gemini computer-use, Perplexity Comet) shows one uniform pattern:

> **"The agent drives, but the human owns the irreversible and the secret."**
> Every vendor pauses for explicit confirmation before the same small set of consequential
> actions (purchase/payment, send, publish, accept ToS/cookies, delete, financial), and hands
> off ("takeover mode") for login, payment entry, CAPTCHA, and 2FA — the agent never types
> passwords or sees secrets. Gemini's API formalizes it as an *out-of-model, inference-time
> safety service* that assesses each proposed action and may return `require_confirmation`.

The decisive gap: a parallel audit of the open CUA **training datasets** (WebArena,
VisualWebArena, Mind2Web / Multimodal-Mind2Web, WebVoyager, GAIA, AndroidWorld/Control, WebLINX)
found that **none of them teach confirmation/handoff.** They are all built for full autonomy and
terminate with `stop`/`ANSWER`/`status`; login is handled by pre-authenticated cookies,
exclusion, or app choice. The only precedent for "ask the user before acting" as a *labeled*
behavior is ST-WebAgentBench (ICML 2025) — and that is an evaluation benchmark, not training
data.

**Therefore:** a dataset that teaches *pause-and-confirm* and *handoff-for-login* as first-class
behaviors is exactly "helpful to users, not synthetic" — it makes a CUA deployable rather than
benchmark-autonomous, and it simultaneously solves our safety, reproducibility, and credential
problems (we never actually commit, never actually pay, and the agent never types a real
password).

## 2. Goals / Non-goals

**Goals**
- Produce login-gated trajectories in the existing trajectory format, consumable by
  `cua/sft/legacy_train_qwen.py convert` with no special-casing.
- Teach two behaviors under **one agent**: a *guarded* assistant (pause/confirm, handoff for
  secrets, never commit irreversible) and an *autonomous* QA agent (given creds in a test env,
  do everything end-to-end including login and commit).
- Make confirmation/handoff **first-class, learned behavior**, adjudicated by a watchdog model.
- Keep the published dataset **secret-free** (masked credentials) and **reproducible** (a
  precondition manifest per trajectory).

**Non-goals (enabled, but out of scope for this spec)**
- The online **evaluation / deployment harness** that replays a trajectory's precondition
  manifest to run the trained agent live. We *record* the manifest now so this is possible
  later; we do not build the runner here.
- Reinforcement learning / online rollout. SFT only.
- Non-web (desktop/mobile) computer use.

## 3. Locked design decisions

These were each decided explicitly during brainstorming:

1. **Environment:** live third-party sites for the *guarded* lane; controlled/self-hosted test
   environments (sandbox + seeded creds) for the *autonomous QA* lane.
2. **Boundary behavior:** confirm/handoff + reversible round-trips where safe (guarded); full
   execution incl. login and commit (autonomous, test env only).
3. **One agent, two modes:** a typed `autonomy` policy field (`guarded` | `autonomous`), not two
   forked agents. The model learns to switch on this authorization signal — that single learned
   distinction is the differentiator.
4. **Credentials:** driver-level **masked secret injection** — the trajectory records a
   placeholder (`<secret:NAME>`), the driver substitutes the real value at execution time.
   Plaintext-in-prompt is rejected (leaks secrets into training text, teaches memorization).
   Secrets and saved sessions are **never committed**.
5. **`confirm` is a lane-aware HITL interrupt, not inherently terminal.** Whether the recorded
   trajectory proceeds past a `confirm` is decided **per-action by a watchdog model** at capture
   time; it terminates only when proceeding would mean real-world harm.
6. **The watchdog verdict is recorded as a first-class event** that flows into the next step's
   history, so the student learns *conditional* completion ("execute only after approval"),
   not "always execute right after confirming."
7. **SFT is offline.** Training needs no live browser, cookies, or preconditions — the
   observation (screenshot) already encodes the post-login state. Preconditions matter only at
   *capture* and at *eval/deploy*, bridged by a recorded **precondition manifest**.
8. **Build order:** shared core first, then both lanes drop in cleanly.

## 4. The shared core — the contract

Actions are free-form dicts inside `TrajectoryStep.actions`, so new action *types* are additive;
the work is in the producers (teacher tools), the converter switch, and the verifier.

### 4.1 New action types

```jsonc
// Agent yields control — models production "takeover mode".
{"type": "handoff", "reason": "login|payment|captcha|2fa|consent", "thought": "..."}

// Agent reaches the irreversible step and emits the PROPOSED action + why,
// INSTEAD of executing it — models production "confirm before consequential".
{"type": "confirm",
 "proposed_action": {"type": "click", "x": 880, "y": 540},
 "risk_category": "purchase|payment|send|publish|delete|consent|account_create|...",
 "thought": "..."}
```

`handoff` is a mid-trajectory pause. `confirm` is the proposal of a sensitive action; whether the
trajectory continues past it is decided by the watchdog (§4.3).

### 4.2 Masked credentials

Credential entry records a placeholder; the driver substitutes the real value at execution time.

```jsonc
{"type": "type", "text": "<secret:username>"}   // recorded
// driver substitutes the real value from an out-of-band secret map before keystrokes
```

The training corpus only ever contains placeholders. A convert-time scan asserts no literal
secret leaked (fail-closed). Pattern follows browser-use `sensitive_data` / Nova Act
`keyboard.type`.

### 4.3 The watchdog / `ApprovalPolicy`

A pluggable approval authority — the LLM implementation of an out-of-model safety service. On a
`confirm`, it sees `{goal, screenshot, proposed_action, risk_category, autonomy, auth_mode, env}`
and returns `{decision: approve|deny, rationale, risk_category}`:

- **approve** — genuinely harmless actions: free registration (no card), reversible actions,
  **publish via a reversible round-trip (post → delete)** — acceptable because capture runs only
  on burner accounts (§8), so brief exposure is on a disposable identity that is immediately
  cleaned up — and anything in a controlled test env. → record the verdict event, execute the
  real action, capture the outcome (and, for round-trips, the cleanup).
- **deny** — payment / credit-card / real money movement / sending to real recipients /
  irreversible-or-non-undoable actions on live. → record the verdict event, **terminate**; never
  execute.

`ApprovalPolicy` is an interface; the LLM watchdog is the default implementation, with
static-rule and human-in-the-loop implementations as drop-ins (per "generalize, don't
special-case"). Configured in `llm_config.py` as `MODEL_FOR_SAFETY_JUDGE` = `claude-sonnet-4-6`
(consistent with the rest of the pipeline). The same watchdog is reusable as the deployment-time
safety service — it is not a data-generation-only trick.

### 4.4 The approval event

The watchdog verdict is recorded as a first-class observation on the post-`confirm` step so it
enters the student's `action_history` window:

```jsonc
// TrajectoryStep gains an optional `event` field
{"step": N+1, "screenshot": "...", "event": {"kind": "approval",
  "decision": "approve|deny", "rationale": "...", "by": "watchdog"},
 "thought": "...", "actions": [...]}
```

The student therefore learns: emit `confirm` → see the verdict in context → execute (on approve)
or stop (on deny). This yields **balanced supervision** — both approved→execute and denied→stop
examples for the pause behavior.

### 4.5 Metadata additions (all optional / additive → backward-compatible)

`TrajectoryMetadata` gains:

- `autonomy: "guarded" | "autonomous"`
- `auth_mode: "pre_session" | "handoff_login" | "none"`
- `risk_category: str` (the trajectory's terminal risk class, if any)
- `preconditions`: the **replay manifest** —
  ```jsonc
  {"requires_session": true, "session_ref": "site_<name>/storage_state.ref",
   "seed_state": "...", "start_url": "...", "env": "live|test"}
  ```
  The SFT converter ignores it; an eval/deploy harness reads it to reconstruct the conditions
  under which the recorded actions are valid.

### 4.6 Auth provisioning sub-modes

Production splits here, so we support both behind `auth_mode`:

- **`pre_session`** — inject a saved `storage_state` (cookies) so capture starts logged-in; the
  agent inherits the session and never logs in (Mariner / Comet / Claude-for-Chrome behavior).
  `handoff` appears only for *fresh* mid-flow secrets.
- **`handoff_login`** — start unauthenticated; the agent navigates to the wall, emits
  `handoff(login)`, the session is injected out-of-band, capture continues (Operator /
  ChatGPT-agent behavior). Teaches the explicit login-handoff.

## 5. Components & where each change lands

| Component | File | Change |
|---|---|---|
| Action contract | `cua/pipeline/schemas.py` | `TrajectoryMetadata` gains `autonomy`, `auth_mode`, `risk_category`, `preconditions`; `TrajectoryStep` gains optional `event`. Additive/optional. |
| Teacher tools | `cua/pipeline/agent_capture.py` (`build_driver_mcp_server`) | Add `handoff` and `confirm` tools (each requires `thought`). On `confirm`, call the watchdog, record the verdict event, branch proceed/terminate. |
| Watchdog | new `cua/pipeline/safety_watchdog.py` + `cua/pipeline/llm_config.py` | `ApprovalPolicy` interface; LLM impl (`MODEL_FOR_SAFETY_JUDGE`); static-rule + human impls as drop-ins. |
| Driver | `cua/pipeline/driver_session.py`, `cua/pipeline/instrumentation/browser_driver.js` | (a) `<secret:NAME>` → real-value substitution at execution (placeholder stays recorded); (b) load `storage_state` for `pre_session`. |
| Proposer | `cua/pipeline/task_proposer.py` | Accept `autonomy`/`auth_mode`; propose policy-aware tasks + success criteria (guarded: "reach the confirm/handoff point"). |
| Verifier | `cua/pipeline/agent_capture.py` (`verify_trajectory`) | Policy-aware success: guarded = correctly paused/handed-off at the right step; autonomous = task actually completed. |
| Converter | `cua/sft/legacy_train_qwen.py` (`format_action_for_training`, history builder, `corpus_audit`) | Render `handoff`/`confirm`; thread the approval `event` into `action_history`; mark confirm-terminal trajectories trainable (not "stuck"); secret-leak scan. |
| Credentials | new gitignored secret store | Per-site burner creds + `storage_state` files. Never committed. |

## 6. The two lanes (data flow)

Both lanes emit the **same trajectory format** and share the same teacher / driver / watchdog /
converter. They differ only in environment and the watchdog's risk bar — the intended
generalization.

### 6.1 Guarded / live lane (`cua/pipeline/public_capture.py`)

Pick an auth-tagged site → `auth_mode` decides start state (`pre_session`: inject saved session,
agent inherits login; `handoff_login`: start fresh, agent emits `handoff(login)`, session
injected out-of-band) → agent pursues goal → at a sensitive step emits `confirm` → **watchdog**
approves (free signup / reversible) → execute + capture outcome, or denies (payment /
irreversible) → terminal. Reversible round-trips clean up after themselves. Teaches **deployable
assistant behavior.**

### 6.2 Autonomous / QA lane (reuses sandbox agent-capture, `cua/pipeline/stage6_generate.py` path)

Boot the app in a sandbox → inject **masked credentials** → agent performs login + full task
including commit (safe — the container resets) → capture outcome. The watchdog still runs, but
its bar is set by `env=test` (it approves more, since nothing is real). Teaches **QA automation
behavior.**

## 7. Training & deployment semantics

- **SFT is offline.** A training example is `(screenshot + instruction + windowed history) →
  (<think> + actions)`. No browser, cookies, or preconditions at train time; the logged-in state
  is already baked into the screenshot pixels. The model learns the observation→action mapping.
- **Graceful degradation at deployment.** Because we *train* both `auth_mode` behaviors, a
  trained agent dropped into a not-logged-in browser sees a login wall it was trained on → emits
  `handoff(login)`; dropped into a pre-authenticated session → proceeds. Deployment need not be
  pixel-perfect.
- **Reproducibility via the manifest.** The precondition manifest (§4.5) is the replay recipe an
  eval/deploy harness uses to reconstruct "the same conditions under which these actions are
  valid." Recording it now is what turns the corpus from *trainable* into *trainable and
  reproducible*.

## 8. Safety, secrets, reproducibility

- **Burner accounts only — hard invariant.** The dataset is *never* gathered with the operator's
  real/personal accounts. Capture runs in a dedicated, isolated browser profile with **no access
  to the operator's real sessions** — we never import a personal browser profile or real cookies.
  The secret store and every injected `storage_state` contain *only* purpose-created
  burner-account credentials, tracked in a burner registry; capture **refuses to run against any
  session/credential not registered as a burner** (fail-closed). This invariant is what makes the
  publish round-trips of §4.3 acceptable.
- **Secrets:** secret store + `storage_state` files gitignored; placeholders in all recorded
  text; a convert-time scan asserts no literal secret leaked (fail-closed).
- **Operational:** no real PII, blocked categories (banking / financial) refused outright —
  matching every production CUA.
- **Reproducibility:** the guarded lane never mutates real account state (confirm-terminal on
  irreversible); the autonomous lane is reset-able. Both are offline-stable for SFT.
- **ToS:** the accepted cost of the "live sites" choice; mitigated by burner accounts, low rate,
  and watchdog-enforced non-destructiveness.

## 9. Testing & build order

**Build order (shared core first):**

1. **Shared core** — schema fields + teacher tools (`handoff`/`confirm`) + watchdog
   (`ApprovalPolicy` + LLM impl) + driver (masked secrets + `storage_state`) + converter
   (rendering + approval-event history + secret scan) + policy-aware verifier. Validated by an
   extended **`make test-driver`** scripted (no-LLM) run that asserts `handoff`, `confirm`,
   masked-secret substitution, and the approval event all satisfy the trajectory contract.
2. **Guarded / live lane** — `public_capture.py` auth_mode + per-site credential/session config
   + confirm-terminal on irreversible. Validated with a small `--limit` E2E on a safe site.
3. **Autonomous / QA lane** — sandbox agent-capture + credential injection + full transactional.
   Validated with a small `--limit` E2E on a test-env app.

**Pilot scope:** shared core + ~5 guarded live sites from the 140 auth-tagged (safe categories:
free signup, reversible actions) + ~2 autonomous test-env apps, before scaling.

## 10. Future phases (post-harvest)

Deliberately deferred until login trajectories have been harvested. This is a rough outline to
resume from, **not** a finished design — a detailed spec will follow.

### Phase 2 — Evaluation / deployment harness

**Purpose.** Take the trained model and *run it live* against a target, replaying a trajectory's
precondition manifest (§4.5), to measure real task success — and, later, to deploy.

**Inputs.** A trained checkpoint; a task + its precondition manifest (`auth_mode`, `session_ref`,
`seed_state`, `start_url`, `env`).

**Sketch.**
1. *Reconstruct preconditions* — load the burner `storage_state` (`pre_session`) or be ready to
   service a `handoff_login`; navigate to `start_url`; apply any `seed_state`.
2. *Rollout loop* — screenshot → model predicts `<think>` + action → execute via the **same
   driver** → repeat. The **watchdog** runs as the live safety service, gating sensitive actions
   exactly as at capture (confirm → approve/deny).
3. *Scoring* — a verifier (reuse the `verify_trajectory` judge) checks goal achievement: guarded
   success = correctly reached confirm/handoff at the right step; autonomous success = task
   completed.

**Reuses everything from Phase 1** — driver, watchdog, verifier, manifest — so the marginal cost
is the rollout loop + scoring, not new infrastructure.

**Candidate metrics.** task success rate; confirm-precision (paused at the right steps?);
handoff-recall (handed off at login walls?); over-/under-confirmation rate.

This closes the loop from *trainable* → *deployable*. Out of scope for the current spec.

### Resolved during review (folded into the spec above)

- Watchdog model + cost: `MODEL_FOR_SAFETY_JUDGE` = `claude-sonnet-4-6` (§4.3).
- `publish`-class actions: **approvable** via reversible round-trip on burner accounts (§4.3,
  §8) — not always-denied.
