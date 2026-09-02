# GRPO runbook — what to do when a new SFT checkpoint lands

Every step below is a single command. Costs are Modal serve/GPU time; the judge is never used for
training rewards, so a run does not consume the Claude usage window.

## 0. Prerequisites

**The Vast.ai account is SHARED with other university researchers.** Never destroy or stop an
instance whose label is not `roman.maior`; use the lane's own `list_owned("roman.maior")` filter, and
treat `vastai show instances` as read-only. Destroying a colleague's running instance has already
happened once and cost them their work.

**A box being ours is not permission to destroy it either.** Ownership is per RUN: a labelled
instance may belong to a concurrent session, another worktree or a background job, and from the
outside that is indistinguishable from an orphan. Teardown goes through `remote_train` — never
`vastai destroy` by hand — where `_owned_by_us` refuses foreign labels and the state records carry
the owning pid so live siblings are skipped. `make remote-train-cleanup` reclaims only boxes whose
owning run is dead; `cleanup --all` takes every labelled box and is correct only when you have
confirmed no sibling run is live.

**Every `modal` command in this runbook runs with `MODAL_PROFILE=majorchannelboost`** — the free-credit
workspace. Never `bombaleylo2`, `roman-maior-mknssh-2025` or `roman-yu-major`; those hold the owner's own
work or are past their free credit. Pin it explicitly per command: the active profile changes underneath a
session. If a volume, secret or cached weight is missing there, create it there rather than switching.

- The new SFT LoRA adapter on the `cua-outputs` volume, e.g. `sft-v2/adapter`.
- `EVAL_API_KEY` in `.env` (the served endpoint's key), `MODAL_PROFILE` pinned per command.
- The task pool: `rl/rl_tasks_candidates.json` (179 oracle-checked candidates, 35 hosts).

## 1. Re-check contamination against the grown corpus (free)

```
python -m rl.tasks build-candidates --trajectories /Users/rmajor/unik/maga/dataset_gather/trajectories --max-expected 18
```

`excluded_trained` must be reported and the printed problem list empty. The corpus grows between runs,
so a task that was disjoint last week may now be in the SFT set.

## 2. Serve the new SFT policy (~$2/h while busy)

Serve the BASE model with the new adapter attached — equivalent to the merged weights and skips the
merge (ScreenSpot measured base+LoRA 85.3 vs merged 85.3):

```
CUA_ADAPTER_NAME=sft-v2/adapter modal deploy modal_serve.py
```

Then WARM it before any sweep: a scale-to-zero endpoint answers nothing for its first ~4 minutes, and
a cold start at sweep time reads as a dead policy. Poll `/v1/models` until it returns.

## 3. Sweep the pool to find the frontier band (~2-3 h, ~$5)

```
python -m rl.rollout --policy openai --endpoint <serve>/v1 --model cua-student-lora \
  --temperature 1.0 --tasks rl/rl_tasks_candidates.json --split train \
  --verifier-mode never --k 8 --workers 3 --drivers-per-sandbox 2 --max-steps 36 \
  --run-id sweep-sft-v2
```

Oracle-scored, no judge. `SANDBOX_MEM_LIMIT=3g SANDBOX_TMP_SIZE=256m` — two browser contexts per
sandbox exceed the 2g default and get OOM-killed. Sample-major ordering means a run stopped early
still yields the same sample count per task.

## 4. Select the band (free)

```
python -m rl.analysis rollouts rl_runs/sweep-sft-v2          # histogram, zero-variance share, harness failures
python -m rl.analysis select-frontier --rollouts rl_runs/sweep-sft-v2 --min 0.2 --max 0.8 \
  --out rl/frontier_selection.json
```

Before trusting a cluster of 0.0 tasks, look at one episode's `final_screenshot.png` — a site behind a
CAPTCHA or returning an error scores 0 for reasons that have nothing to do with the policy. Blocked and
crashed episodes are already excluded from the denominator; whole sites that are walled should be added
to `_config.exclude_sites` in `repo_pipeline/public_sites_actions_frontier.json`.

If fewer than ~15 tasks land in the band, widen to `--min 0.1 --max 0.9` rather than training on seven.

## 5. Train (~$0.9/step; 50 steps ~ $18)

Env service on the Mac, exposed through an ingress, then the run:

```
RL_ENV_TOKEN=<token> python -m rl.service --tasks rl/rl_tasks_candidates.json --split train \
  --workers 3 --drivers-per-sandbox 2 --verifier-mode never --max-steps 36 --run-id grpo-v3
ngrok http 127.0.0.1:8900
modal run modal_grpo.py --run-name grpo-v3 --env-url <ingress> --env-token <token> \
  --tasks-file rl/rl_tasks_candidates.json --task-ids-file rl/frontier_selection.json --max-steps 50
```

The service refuses to start if any served task lacks a `success_check`, and a verdictless episode
aborts the run rather than training a fabricated zero.

## 6. THE KILL CHECK — at the first checkpoint, before paying for the rest

`save_steps: 5`, so ~15 minutes in there is a `checkpoint-5`. Pull it and compare against the SFT
adapter the run started from:

```
modal volume get cua-outputs grpo-v3/rl_output/.../checkpoint-5/adapter_model.safetensors <local>
python -m rl.analysis bnorm <local> --ref <the sft-v2 adapter_model.safetensors>
```

`B_ratio` under 0.10 sets `dead_run: true` — the weights did not move and no further steps will help.
Kill the run. grpo-v2 measured 1.9% and was allowed to finish anyway; this check exists so that never
repeats. Watch ms-swift's `frac_reward_zero_std` in the log for the same signal live: it is the share
of groups with no reward contrast, and a healthy band should keep it low.

## 7. Eval

Same protocol both sides or the comparison is meaningless: same task sample, same judge, same harness
commit, each model scored in its own `coord_space` (a tuned checkpoint answers in `viewport_abs`).
Serve a GRPO adapter on the **merged SFT weights** — its `base_model_name_or_path` is the merge, never
raw `Qwen/Qwen3.5-9B`. Report the scored and all-episode rates side by side; blocked sites are excluded
from the former, which is why our numbers are comparable only to our own before/after.
