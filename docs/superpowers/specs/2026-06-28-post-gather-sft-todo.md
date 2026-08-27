# Post-Gather → SFT TODO

**Date:** 2026-06-28
**Companion to:** `2026-06-28-login-gated-trajectories-design.md`

Checklist for going from "login-gated trajectories gathered" → "SFT run launched". This is
**out of scope** for the login-gated implementation itself; it's the next milestone after the
trajectories exist.

## 1. Verify shared-core data handling landed (gates)
- Converter renders `handoff`, `confirm` (proposed_action + risk_category), masked `type`
  placeholder; approval `event` threaded into `action_history`. Lock the exact training-format
  token rendering for the new actions.
- `is_trainable` / convert gating treats confirm-terminal guarded trajectories as valid success
  (not stuck/incomplete); verifier sets `success` correctly per `autonomy` policy.
- Secret-leak scan green (convert fails closed on any literal resolved secret).

## 2. Finalize supervision for the new step kinds
- Think-polish handles confirm/handoff intent ("reached payment → pause and ask"; "hit login
  wall → hand off") — intent-level, watchdog rationale can seed the confirm thought.
- Run the polish/finalize pass over the login-gated trajectories.

## 3. Assemble + balance the combined corpus
- Decide mix ratios across Path A (GPT-5.4 CUA), login-free Path B (~340), new login-gated
  (guarded + autonomous); handoff/confirm/auth_mode represented but not dominant.
- Cross-lane dedup (`dedup_hash`); review action-distribution + length-bucket health.

## 4. Final convert + pre-train gates
- `make convert` → `training_data/sft_dataset.jsonl`; read the corpus health report
  (0 bad_json/schema/missing_img; sane action histogram; trainable count).
- Confirm `max_length: 4096` isn't truncating with the new confirm/approval-event tokens.

## 5. Training environment + config (infra prereqs)
- Provision GPU box with ms-swift (`make train` → `swift sft`); Qwen3.5-9B-Base weights +
  tokenizer; export `MAX_PIXELS`.
- Sanity-check `config.yaml` (LoRA r64/α128, lr 5e-5, 3 epochs, `freeze_vit: true`).

## 6. Minimum confidence gate
- Hold out a validation split; manual spot-check of confirm/handoff/masked-secret examples
  (does the model see approval before execute? are secrets masked?).
- Real task-success eval = the Phase 2 eval/deploy harness (separate future phase, see the
  design doc §10).

## Disposable-email wall (deferred — 2026-08-26)

A growing set of sites reject mail.tm addresses at signup outright ("This email domain has been
banned due to policy violations"): teable, scribbles, outline-wiki confirmed, and it is expected to
hit a meaningful share of the 63 mainstream targets (etsy/ebay/canva/figma class). Neither lane can
pass it — the autonomous lane has nothing else to offer, and a human with the same burner hits the
identical wall, so it is NOT a human-provisioning problem.

Fix: own a cheap domain with catch-all forwarding and mint burners on it instead of mail.tm. Keeps
the burner-only invariant fully intact (still disposable, still never the operator's identity) while
presenting a non-disposable domain. Blocks the whole class at once; touches `mailtm.py`'s mailbox
minting and nothing else, since the rest of the pipeline only ever sees `{address, password}`.

Related: `provisioning_status` gained an `email_blocked` class to keep both lanes off these sites.
