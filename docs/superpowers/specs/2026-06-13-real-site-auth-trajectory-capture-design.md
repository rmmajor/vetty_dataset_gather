# Real-site auth-gated trajectory capture — design

**Date:** 2026-06-13
**Status:** Draft for review
**Branch (impl):** to be done in an isolated worktree on `feat/auth-trajectory-capture` (this doc commits to `feat/agent-capture`)

## Purpose

Teach the distilled CUA model (Qwen3.5-9B ← GPT-5.4) to perform **authentication as a
necessary step of a real task** — register an account, verify an email, log in, then do the
thing that was gated behind the login. This is a core CUA skill the current corpus lacks: the
public-capture lane is explicitly non-destructive/login-free, and the funnel rejects apps that
need a backend/credentials (`stage4_deep_analyze.py:57`, `app_runner.py:13`), so auth coverage
today is incidental.

The target is **real third-party sites**, not practice sites or our own sandboxed apps — the
model must generalize over real-web auth UIs (cookie/consent walls, varied form layouts,
email-verification steps, OAuth buttons). The browser still executes through the existing
driver-in-container runtime (or headed-on-host for attended runs); "real" refers to the *target*,
not the execution environment.

## Locked decisions (from brainstorming, 2026-06-13)

1. **Email infra:** own catch-all domain (`cua+<uuid>@ourdomain`), robust against
   disposable-domain blocklists. Not a throwaway-mail API.
2. **CAPTCHA:** both modes — human solves when an operator is attached; otherwise record a
   terminal `blocked` state and skip. Requires a harness pause/resume signal.
3. **Egress:** datacenter IP first (headed + stealth); residential proxy is a wired-but-unused
   config knob added later only for targets that prove to need it.
4. **Attended browser location:** headed on the host (Mac) for attended auth runs;
   in-container headless for unattended runs.
5. **MVP target:** Open Library (`openlibrary.org`). Non-profit/automation-tolerant, plain
   email+password registration with email verification, historically no aggressive CAPTCHA,
   genuine auth-gated reversible task. Fallback if its register flow is CAPTCHA-walled:
   Letterboxd / Trakt.tv (film watchlist) or MyAnimeList.

## Scope

**In scope**
- Register → verify-email → login → perform an auth-gated, **non-transactional** task.
- A generic, config-driven auth capability (no per-site code): auth is data on a site entry
  plus a small set of new generic tools.
- Secret-safe trajectories (passwords never stored verbatim).
- A one-site MVP that produces a convertible, audit-clean trajectory before any list-scaling.

**Out of scope (explicit guardrails)**
- **No completed real purchases / bookings** (real charge or real reservation). Auth-gated
  tasks end at the authenticated state + a read/configure action (view bookings, add-to-cart
  without checkout, set a preference, search while logged in).
- **No CAPTCHA-solving-as-a-service / no bot-defense evasion for mass abuse.** Human solves at
  low volume, or we record the block and move on.
- No sourcing the 300-site curated list until the MVP proves the flow (the list becomes a
  downstream phase that validates the full register→login flow per candidate).

## Components

Designed generic-over-special-case per the repo's CLAUDE.md rules: auth is JSON config keyed by
site + a strategy/provider registry, never hardcoded site checks.

### 1. `InboxProvider` (new) — `repo_pipeline/inbox.py`
Interface: `get_message(alias, since, match=...) -> Message | None`, `extract_code(msg)`,
`extract_link(msg)`. Concrete impl reads **one catch-all mailbox** (IMAP or Gmail API) and
filters by the `To:` alias, so a single mailbox backs unlimited per-task aliases. Backend
selection is config; credentials come from env (never committed).

Exposed to the teacher as a `check_email(alias)` tool. The harness (not the model) does the IMAP
fetch + code/link extraction and returns the result as a tool message; the model then types the
code or navigates the link, keeping it as genuine CUA grounding.

### 2. Identity generator (new) — `repo_pipeline/identity.py`
Per task: `cua+<uuid>@ourdomain` (uuid4 minted at task start on the host), a generated password,
and synthetic profile fields (name/username/DOB) for registration forms. The minted identity is
recorded in the task state so a re-run can reuse or regenerate it deterministically.

### 3. Auth task schema (changed) — `public_sites.json` + `schemas.py`
Site/task entries gain an optional `auth` block:
```json
{"auth": {"mode": "register|login",
          "identity_strategy": "fresh_alias|account_pool",
          "expected_end_state": "free-text goal for the verifier"}}
```
plus a new `auth` task type (the `auth` category already exists in `registry.py:104`).
- `register` tasks → `fresh_alias` (unique per run; fixes "email already taken").
- `login` tasks → `account_pool`: accounts registered once and stored in an env-vault file
  (`pipeline_data/account_pool.json`, gitignored), reused for repeatable login-gated tasks.
  This splits the one-time register skill from the repeatable login skill.

### 4. Secret handling (changed) — `agent_capture.py` type tool
Add `secret: true` to the type-tool args. When set, `trajectory.json` stores the text as
`<password>` (placeholder) and the windowed text history masks it; the masked screenshot +
action sequence still teach "type into the password field, click sign in." Throwaway email may
appear verbatim (disposable). A convert-time assertion fails the build if any non-placeholder
value matches a known pool password (defense in depth).

### 5. CAPTCHA / blocked handling (new)
- **Detection:** heuristic in the driver — presence of known challenge selectors
  (`iframe[src*="hcaptcha"]`, `recaptcha`, Turnstile, "verify you are human" text) — plus a
  `report_blocked(reason)` tool the teacher can call.
- **Attended (operator attached):** emit a pause signal + operator notification (reuse the
  Telegram notifier), wait on a resume signal (file/socket), operator solves in the headed host
  browser, run resumes. Bounded wait with timeout → degrade to blocked.
- **Unattended:** record a terminal `blocked` outcome (no success). This is itself useful CUA
  behavior — recognizing a wall and stopping.
- **Verifier** learns to judge `blocked` distinctly from success/failure.

### 6. Stealth driver (changed) — `browser_driver.js` / `driver_session.py`
Headed + stealth for auth runs (playwright-extra stealth or hand-rolled fingerprint patches,
real UA/viewport, persistent context). `--proxy` knob wired through `driver_session.start_app`
but unused until a target needs it. Attended runs launch the driver **headed on the host**;
unattended runs keep the in-container headless path. Both speak the same driver HTTP protocol so
the teacher/verifier are unchanged.

### 7. Capture lane (changed) — `public_capture.py`
Auth-enabled site entries flow through the same `capture_site` machinery. The non-destructive
curation comment in `public_sites.json` is amended: register/login permitted, real
purchases/bookings still forbidden. Verifier judges a **stable authenticated end-state**
(dashboard reached / account menu shows the name / item on shelf).

## Data flow

```
host: generate identity (alias, password, profile)
  → driver navigates to site
  → teacher fills + submits registration form  [secret:true on password]
  → [CAPTCHA branch: attended solve | blocked-skip]
  → teacher calls check_email(alias)
       → harness IMAP-fetches latest msg to alias, extracts code/link, returns as tool result
  → teacher enters code / navigates verification link
  → logged-in state reached
  → teacher performs the auth-gated task (add to shelf / view account / configure)
  → verifier checks expected_end_state on final screenshot → success | blocked | fail
  → trajectory copied out, thoughts polished, metadata written (capture_source=agent)
```

## MVP acceptance criteria

1. One Open Library trajectory: `register → verify-email → login → add a book to "Want to Read"`,
   ending with the book visible on the user's shelf (or username in the header).
2. `trajectory.json` has `success: true` (verifier verdict), no verbatim password
   (placeholder only), `capture_source == "agent"`.
3. `train_qwen.py convert` counts it; `make audit-corpus` stays at **0 failing**.
4. The `check_email` loop demonstrably read a real verification message from the catch-all
   mailbox.
5. Thoughts describe **user intent** ("I need an account to save books, so I'm registering"),
   not tooling — per the intent-level supervision rule.

## Ops prerequisites (operator-only, cannot be automated here)

- Own a domain + catch-all forwarding (ImprovMX→Gmail ≈ free, or Cloudflare Email Routing).
- Provide the harness the mailbox creds / Gmail-API token via env (never committed):
  `AUTH_INBOX_DOMAIN`, `AUTH_INBOX_IMAP_HOST/USER/PASS` (or Gmail OAuth token path).
- Be reachable to solve CAPTCHAs for attended runs.

## Phased rollout

- **Phase 0 (MVP):** Open Library, attended, datacenter IP. Prove the five acceptance criteria.
- **Phase 1:** harden (unattended skip-mode, account pool for login tasks, secret-leak convert
  assertion, blocked-state verifier).
- **Phase 2:** source the curated real-site list — each candidate validated through the full
  register→verify→login flow, not just reachability. This is where the original "300 sites" ask
  lands, now correctly gated behind a working capability and a much stricter validation bar.

## Risks / open questions

- **Bot defense on datacenter IP** may block register/login on many targets → Phase 2 likely
  needs residential proxies for a real list; the MVP target was chosen to avoid this.
- **Email verification timing:** some sites let you act before verifying; some hard-gate. The
  `check_email` loop must poll with a bounded timeout and handle "no email arrived."
- **Two execution paths** (headed-host vs in-container) is maintenance surface; mitigated by a
  shared driver HTTP protocol.
- **ToS variance per site** — Phase 2 list curation must screen ToS, not just feasibility.
- **Account-pool hygiene:** pool file must be gitignored and treated as secrets.
```
