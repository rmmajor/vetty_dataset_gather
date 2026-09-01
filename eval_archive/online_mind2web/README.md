# Pinned Online-Mind2Web task sets

Tracked deliberately (`.gitignore` exempts this directory): a before/after is meaningless unless
both sides ran the same tasks, so every subset a reported number came from lives here.

| file | n | what it is |
| --- | --- | --- |
| `tasks_pinned_50.jsonl` | 50 | the judge2 triple's sample (base / sft-v1 / grpo-v2, 2026-08-27) and sft-v2's comparator. Kept verbatim for continuity. |
| `tasks_pinned_v2.jsonl` | 230 | the 50 above + 180 level-stratified, excluding sites blocked from a Ukrainian IP in all three judge2 runs. |
| `tasks_reachable_us_174.jsonl` | 174 | every task whose site actually loads from a **US residential** egress (2026-09-01). |

## Why the 174 set exists

Online-Mind2Web is US-centric and this machine egresses from Lviv, so a large share of the
benchmark is geo- or WAF-blocked before the model acts. Measured against the 24 sites lost in the
sft-v2 run, three egresses recover very differently:

| egress | ASN | usable |
| --- | --- | --- |
| local | residential UA | 0 / 24 |
| ClearVPN Dallas | AS9009 M247 (datacenter) | 1 / 24 |
| Vast NYC | AS701 Verizon (consumer) | 9 / 23 |

A datacenter VPN trades geo-blocks for WAF-blocks roughly 1:1; only a residential ASN recovers a
real share. Probing all 147 distinct start URLs from a US residential box then gave **174 of 300
tasks reachable (58.0%)**, level mix intact (easy 50 / medium 75 / hard 49 against 80 / 141 / 79).

`tasks_pinned_v2.jsonl` excluded 70 tasks as permanently blocked; 25 of those are reachable from a
US exit, which is why this set is not a subset of it.

Reproduce with `python -m repo_pipeline.student_eval ... --browser-backend vast_residential`, or
re-probe with `repo_pipeline/remote_sandbox.py`'s `residential_offer_ids` plus a render probe.

## Two caveats

A homepage that loads does not guarantee deeper pages stay unblocked during an episode, and every
rental is a different residential IP — so this is a better starting set, not a guarantee. The
harness excludes blocked episodes from the success denominator either way; what this set buys is
**statistical power**, not a higher score.
