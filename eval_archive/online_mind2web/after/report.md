# Student eval — `after`

**verified task success: 29.4% across 50 online_mind2web tasks, parse-failure rate 7.4%, site-blocked 28.0%**

Success is over the SCORED episodes (10 verified of 34); site-blocked / unreachable / harness-error episodes are excluded from that denominator and reported separately, because folding bot-blocking into the success rate measures the network, not the model. Over all 50 episodes the rate is 20.0%.

## Run

| field | value |
| --- | --- |
| run id | `after_online_mind2web_20260827T063651Z` |
| label | `after` |
| started (UTC) | 2026-08-27T06:36:51.602340+00:00 |
| finished (UTC) | 2026-08-27T07:01:29.984998+00:00 |
| endpoint | `https://roman-yu-major--cua-serve-server.us-east.modal.direct/v1` |
| model | `cua-student-lora` |
| adapter / checkpoint | `(none declared)` |
| task source | `online_mind2web` — 300 tasks across live third-party websites (Online-Mind2Web, COLM 2025) |
| dataset | `osunlp/Online-Mind2Web` |
| dataset revision | `eacad896a84dc5b65e29b0b06e4699ab0544d701` |
| coordinate space | `viewport_abs` |
| step cap | 3.0x reference_length, clamped to [15, 50] (ungraded tasks: 30) |
| abort rules | 5 unproductive steps / 3 parse failures |
| task subset | level-stratified 50 |
| temperature / max tokens | 0.0 / 2048 |
| observation format | png @ 1280x720 |
| judge | claude-sonnet-5 (final + 2 mid screenshots) |
| history window | 10 |
| driver loop guard | disabled |
| browser | local DockerSandbox on the invoking host |
| repo commit | `3b340b5c1634-dirty` |

<details><summary>task provenance</summary>

```json
{
  "dataset": "osunlp/Online-Mind2Web",
  "split": "test",
  "revision": "eacad896a84dc5b65e29b0b06e4699ab0544d701",
  "file": "Online_Mind2Web.json",
  "rows_available": 300,
  "distinct_sites": 145,
  "level_distribution": {
    "easy": 80,
    "hard": 79,
    "medium": 141
  },
  "reference_length": {
    "mean": 8.4,
    "max": 36
  },
  "citation": "Xue et al., An Illusion of Progress? (arXiv:2504.01382), CC-BY-4.0"
}
```

</details>

## Metrics

| metric | value |
| --- | --- |
| episodes | 50 |
| scored episodes | 34 |
| judge deferred (usage limit) | 0 |
| verified success (scored) | 29.4% |
| verified success (all episodes) | 20.0% |
| verified AND non-meandering | 17.6% |
| site-blocked | 28.0% |
| parse-failure rate (steps) | 7.4% |
| malformed action lines | 42 |
| canonical action form | 100.0% (584/584 actions) |
| form deviations | none |
| done() emission rate | 18.0% |
| <think> emission rate (steps) | 100.0% |
| driver-refused action rate | 0.0% |
| emitted coords outside the viewport | 0/418 (0.0%) |
| out-of-viewport actions refused | 0 |
| episodes abandoned as stuck | 0 |
| mean steps on success | 17.7 |
| mean steps / reference_length on success | 2.14x |
| mean steps (all) | 11.42 |
| episodes that hit the step cap | 12 |

`verified` is the judge's verdict on the ORIGINAL goal; the efficiency grade is reported beside it rather than folded into it. `steps / reference_length` compares against the benchmark's own human step count, which is a more defensible efficiency denominator than our raw step counts.

`canonical action form` is the share of parsed actions spelled exactly as `format_action_for_training` emits them. The parser accepts any UNAMBIGUOUS spelling of a call (optional `action:`, bare or `<ref>`-wrapped coordinates, `wait` with or without `()`, omitted keyword arguments, single or double quotes, bare key names) and rejects genuinely ambiguous ones. A base model has only seen the grammar described in the system prompt, so charging surface deviation to its success rate would measure imitation of an unfamiliar convention rather than browsing. The deviation is reported here instead, and the tolerance is a harness parity key so both sides of a delta are parsed identically.

## Per level

| level | n | scored | verified | success rate | steps/ref | blocked |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 17 | 11 | 4 | 36.4% | 1.88x | 5 |
| medium | 16 | 10 | 3 | 30.0% | 2.36x | 6 |
| hard | 17 | 13 | 3 | 23.1% | 2.27x | 3 |

## Outcomes

| outcome | episodes |
| --- | --- |
| site_blocked | 14 |
| parse_failure_abort | 13 |
| step_cap | 12 |
| completed | 9 |
| unreachable | 2 |

## Per-domain

| domain | n | scored | verified | success rate | blocked |
| --- | --- | --- | --- | --- | --- |
| www.google.com | 3 | 3 | 1 | 33.3% | 0 |
| cookpad.com | 2 | 2 | 1 | 50.0% | 0 |
| www.bbb.org | 2 | 0 | 0 | 0.0% | 2 |
| www.healthgrades.com | 2 | 2 | 0 | 0.0% | 0 |
| www.stubhub.com | 2 | 2 | 0 | 0.0% | 0 |
| www.drugs.com | 2 | 0 | 0 | 0.0% | 2 |
| www.bestbuy.com | 2 | 0 | 0 | 0.0% | 2 |
| www.qatarairways.com | 1 | 0 | 0 | 0.0% | 1 |
| www.gov.uk | 1 | 1 | 1 | 100.0% | 0 |
| careers.walmart.com | 1 | 1 | 0 | 0.0% | 0 |
| www.petfinder.com | 1 | 1 | 0 | 0.0% | 0 |
| www.cboe.com | 1 | 0 | 0 | 0.0% | 1 |
| www.carmax.com | 1 | 0 | 0 | 0.0% | 0 |
| www.flightaware.com | 1 | 1 | 0 | 0.0% | 0 |
| umich.edu | 1 | 1 | 0 | 0.0% | 0 |
| www.cbp.gov | 1 | 1 | 0 | 0.0% | 0 |
| www.coolmath4kids.com | 1 | 1 | 0 | 0.0% | 0 |
| us.trip.com | 1 | 1 | 1 | 100.0% | 0 |
| www.theweathernetwork.com | 1 | 1 | 1 | 100.0% | 0 |
| www.dillards.com | 1 | 0 | 0 | 0.0% | 1 |
| www.mayoclinic.org | 1 | 0 | 0 | 0.0% | 1 |
| phet.colorado.edu | 1 | 1 | 0 | 0.0% | 0 |
| www.carnival.com | 1 | 1 | 1 | 100.0% | 0 |
| www.recreation.gov | 1 | 1 | 1 | 100.0% | 0 |
| www.americashealthrankings.org | 1 | 1 | 1 | 100.0% | 0 |
| www.adoptapet.com | 1 | 1 | 0 | 0.0% | 0 |
| www.thumbtack.com | 1 | 1 | 0 | 0.0% | 0 |
| www.accuweather.com | 1 | 0 | 0 | 0.0% | 0 |
| www.kaggle.com | 1 | 1 | 0 | 0.0% | 0 |
| www.birkenstock.com | 1 | 1 | 0 | 0.0% | 0 |
| www.stanford.edu | 1 | 1 | 0 | 0.0% | 0 |
| ohiomeansjobs.ohio.gov | 1 | 1 | 0 | 0.0% | 0 |
| www.kbb.com | 1 | 0 | 0 | 0.0% | 1 |
| raisingcanes.com | 1 | 1 | 0 | 0.0% | 0 |
| www.disney.com | 1 | 0 | 0 | 0.0% | 1 |
| www.webmd.com | 1 | 1 | 1 | 100.0% | 0 |
| ohio.gov | 1 | 1 | 0 | 0.0% | 0 |
| www.dmv.virginia.gov | 1 | 0 | 0 | 0.0% | 1 |
| www.apartments.com | 1 | 0 | 0 | 0.0% | 1 |
| www.ca.gov | 1 | 1 | 1 | 100.0% | 0 |
| us.megabus.com | 1 | 1 | 0 | 0.0% | 0 |
| www.youtube.com | 1 | 1 | 0 | 0.0% | 0 |

## Episodes

| task | level | outcome | verified | efficiency | steps | cap | ref | parse fails | driver errors | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `005be9dd91c95669d6ddde9ae667125c` | easy | site_blocked | no | - | 0 | 15 | 4 | 0 | 0 | waf_block |
| `0170ca95038b05fa58d463fe627ac605` | hard | step_cap | yes | meandering | 36 | 36 | 12 | 0 | 0 | The final state shows the Health and Care Worker visa page on GOV.UK, which does answer the goal for a healthcare worker |
| `0059adc6b12a3822305deb68929b2de8` | medium | parse_failure_abort | no | meandering | 9 | 24 | 8 | 3 | 0 | All three screenshots are blank/white with no visible content, and the run consisted of one click followed by four repea |
| `0632e496d37badee0350dad358f047c5` | easy | step_cap | no | meandering | 15 | 15 | 4 | 1 | 0 | No recipe results were ever reached (final state still shows the 'no recipes found' empty page), and instead of adjustin |
| `0b2623e9fa5cea997f76490bcbc5220f_070826` | hard | parse_failure_abort | no | meandering | 12 | 39 | 13 | 3 | 0 | The site returned a persistent multilingual server error page for the entire run, and the agent only ever issued 'wait'  |
| `01abae9608f2d8752a83e08f136f720c` | medium | site_blocked | no | - | 0 | 18 | 6 | 0 | 0 | waf_block |
| `16200f51d63f0a47a58fa17acd49e368` | easy | completed | yes | efficient | 3 | 15 | 3 | 0 | 0 | The final screenshot shows a Cookpad recipe page for eggplant and mushroom stir-fry (なすとキノコのバジル炒め), with eggplant and mu |
| `0b51b4fa0295ae80ccd176ebdad6fff6` | hard | unreachable | no | - | 0 | 39 | 13 | 0 | 0 | page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://www.carmax.com/ Call log:   - navigating to "https://www.carmax.com/", waiting until "domcontentloaded"  |
| `046138801a05ddf56ad94e8672942496_070826` | medium | step_cap | no | meandering | 18 | 18 | 6 | 0 | 0 | The final screenshot is still the same 'Top' topic list (sorted by Year, not all-time) with no thread opened, identical  |
| `180ed2ec377ef3a4af9035a21522091a` | easy | step_cap | no | meandering | 15 | 15 | 5 | 0 | 0 | All three screenshots show the umich.edu domain stuck behind a Cloudflare bot-verification challenge, never reaching any |
| `0b838cd54f826c59c71f600c56b89a11_070826` | hard | site_blocked | no | - | 0 | 33 | 11 | 0 | 0 | waf_block |
| `05483c50cc9b04c8ac44c574758fb2bd` | medium | site_blocked | no | - | 0 | 24 | 8 | 0 | 0 | waf_block |
| `1fc28d91d25ccd1c6ba268101326a654` | easy | completed | yes | efficient | 3 | 15 | 3 | 0 | 0 | Final screenshot shows a Bitcoin/USD chart on Google Finance with the 5D range selector active, matching the goal exactl |
| `0e5536aaad9d3462b06cf725e6ed535a` | hard | parse_failure_abort | no | meandering | 27 | 33 | 11 | 3 | 0 | Final state is the generic 'For U.S. Citizens' travel landing page, not any wait-times page; mid-run screenshot even sho |
| `070c907d34a4ce71dfdbea38f9c5d4d8` | medium | parse_failure_abort | no | meandering | 6 | 21 | 7 | 3 | 0 | The final state is a Healthgrades geo-block error page ('This page is not available in your area'), showing no dentist s |
| `20a460a8fe1971b84411c5b1e6ac4186` | easy | parse_failure_abort | no | meandering | 6 | 15 | 3 | 3 | 0 | Final screenshot shows StubHub's bot-detection block page, not a Las Vegas theatre events list, so no event was selected |
| `1223b07536a87e0170ff87cbbebd1d3c` | hard | step_cap | no | meandering | 50 | 50 | 24 | 0 | 0 | Final screenshot shows the same untouched quiz setup page (0-10, 10 questions, 30 sec) as the first mid-run screenshot,  |
| `07bdc595306729a028ba06cc7451a80a` | medium | completed | yes | acceptable | 12 | 21 | 7 | 0 | 0 | Final screenshot shows a Shanghai→Beijing high-speed search with the 15:21→21:10 result expanded and bookable, satisfyin |
| `2207bb4f21786690cfed20b37253fb8b` | easy | completed | yes | acceptable | 5 | 15 | 2 | 0 | 0 | Final screenshot shows Calgary, AB weather page with Wind clearly displayed (6 km/h S, gust 9 km/h), satisfying the goal |
| `199be0b54a436daee74247971fc684ee_070826` | hard | site_blocked | no | - | 0 | 50 | 20 | 0 | 0 | waf_block |
| `07ec4a12cba8090e2dc524d558ac7675` | medium | site_blocked | no | - | 0 | 18 | 6 | 0 | 0 | waf_block |
| `23204728192da9f73197a613d9681c18` | easy | site_blocked | no | - | 0 | 15 | 3 | 0 | 0 | waf_block |
| `1ab384fb3a791edfb410213cc6b82151_110325` | hard | step_cap | no | meandering | 33 | 33 | 11 | 0 | 0 | Final screenshot shows the sim's About/description page for 'Build a Nucleus' rather than the simulation running with a  |
| `0a0fa834ce41b5297c6474293383759d_110325` | medium | completed | yes | acceptable | 19 | 24 | 8 | 0 | 0 | Final screenshot shows an 8-Day Mexican Riviera from Los Angeles cruise (correct duration/origin) with the 'What's inclu |
| `246d654fab7c31d9651007e39e75f74f` | easy | step_cap | yes | meandering | 15 | 15 | 5 | 0 | 0 | Final screenshot shows Alpine Ridge's Ratings & Reviews tab with 196 reviews, but sorted by 'Most Recent' showing a 4-st |
| `1b867afecf072cb877ebfa4069263746` | hard | completed | yes | meandering | 41 | 45 | 15 | 0 | 0 | Final screenshot shows the Unemployment - Women Trends chart with Michigan plotted, but the legend/lines don't clearly s |
| `0a54069a0ef542e571d1fee7f39c93d5` | medium | parse_failure_abort | no | meandering | 13 | 18 | 6 | 5 | 0 | All 8 steps were passive 'wait' actions and the final screenshot shows a persistent 403 CloudFront geo-block, never reac |
| `2532fd402d3c741b79894e6ff2269f53` | easy | parse_failure_abort | no | meandering | 3 | 15 | 3 | 3 | 0 | The site blocked the request as a geo-restricted country, no search was performed and no electrician results were ever s |
| `1bc154377120ec15b18dbabdba49c741_051526` | hard | parse_failure_abort | no | meandering | 5 | 50 | 24 | 3 | 0 | The run stalled after only 3 actions (typing 'BTS') and hit a StubHub bot-detection block page, never reaching search re |
| `0e42c3a73f2aece1f854e0ba55b7c8b0` | medium | step_cap | no | meandering | 21 | 21 | 7 | 0 | 0 | The agent never reached a gas station listing's review panel to sort by lowest rating; instead it got stuck re-searching |
| `26810ed9c123a62992e3eed31db3c5ee` | easy | unreachable | no | - | 0 | 15 | 4 | 0 | 0 | page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://www.accuweather.com/ Call log:   - navigating to "https://www.accuweather.com/", waiting until "domcontentloaded"  |
| `1c3b747ae12ccee895745f82e3f2ef8a` | hard | step_cap | no | meandering | 33 | 33 | 11 | 0 | 0 | The final screenshot shows an unrelated user profile page (Tom Aindow), not the Code tab sorted by votes for the highest |
| `11857213ca01510f12813740afd59918_110325` | medium | completed | no | meandering | 20 | 36 | 12 | 0 | 0 | The agent filtered by EU sizes 37 and 39 (roughly US 5-6.5, not the requested US men's 10-10.5) and added a Boston clog  |
| `26a0e5c21c145dd8448aa92f35bec5ea` | easy | parse_failure_abort | no | meandering | 3 | 15 | 3 | 3 | 0 | No actions were executed at all — the run aborted with a parse failure before making any progress, and the final screens |
| `27fa3ac20745d3d35e89fae157f63069` | hard | parse_failure_abort | no | meandering | 8 | 33 | 11 | 3 | 0 | The agent never reached graduate chemistry course schedules for winter 2023 afternoons; screenshots show it stuck on a g |
| `11abb668c751dd56bb41f296a8bb3a13` | medium | site_blocked | no | - | 0 | 30 | 10 | 0 | 0 | waf_block |
| `271b36efd4346721b5542488ff997042` | easy | site_blocked | no | - | 0 | 15 | 4 | 0 | 0 | waf_block |
| `28e7574e7bd6d14f36d2988a5ef2bd23` | hard | parse_failure_abort | no | meandering | 3 | 36 | 12 | 3 | 0 | Zero actions were executed and the final screenshot shows a 404 error page on an unrelated site (OhioJobs/Innovate Ohio) |
| `123e8c2fc453f55fadd1d0b9aaf94df4` | medium | site_blocked | no | - | 0 | 24 | 8 | 0 | 0 | waf_block |
| `29526b17a32485742b5ab63507e99417` | easy | site_blocked | no | - | 0 | 15 | 3 | 0 | 0 | waf_block |
| `33bd2cdcea4fcc42a09a8a1e4e5841c6_110325` | hard | step_cap | no | meandering | 50 | 50 | 20 | 0 | 0 | The site returned a persistent error page ('this site is temporarily unavailable') across all three screenshots, and the |
| `157f4a79d55e8fa3fd55ba772ba40fbc_070826` | medium | site_blocked | no | - | 0 | 27 | 9 | 0 | 0 | waf_block |
| `29b7372d5a3884a2ba831af2d117af3c` | easy | step_cap | no | meandering | 15 | 15 | 3 | 0 | 0 | Final state shows the MSFT overview/chart page with a news summary snippet, but no article/news item was opened as requi |
| `3443e9c3151fef19a3c3a45eb2c13640_070826` | hard | completed | yes | acceptable | 13 | 36 | 12 | 0 | 0 | The final screenshot shows the WebMD ovulation calculator with results including Approximate Ovulation (June 15) and Pre |
| `15be05973fba714e490cd9c884e4f072` | medium | parse_failure_abort | no | meandering | 15 | 18 | 6 | 3 | 0 | All three screenshots (including the final state) show a 404 error page, so the agent never reached any content about At |
| `34992feb69eb8e788faa06868b365c49` | easy | site_blocked | no | - | 0 | 15 | 5 | 0 | 0 | waf_block |
| `367d843c640637745e8fafa741cca13b` | hard | site_blocked | no | - | 0 | 45 | 15 | 0 | 0 | waf_block |
| `1aeca99e6a60b0e3aefb3ef212bdce79` | medium | step_cap | yes | meandering | 30 | 30 | 10 | 0 | 0 | Final screenshot shows CalCareers job search results filtered/showing San Diego County full-time positions with visible  |
| `34ccd15a8ea8fd3895af83f5ccf62369` | easy | parse_failure_abort | no | meandering | 3 | 15 | 3 | 3 | 0 | No actions were executed and the run aborted with a parse failure; the final screenshot shows only a CloudFront 403 erro |
| `3ef64f34eae59c9fac7ee9a4f18b4a0c` | hard | completed | no | acceptable | 11 | 48 | 16 | 0 | 0 | The final screenshot shows a video opened on standard YouTube (Ukrainian UI, regular login prompt), not YouTube Kids, an |

---

Episodes are recorded under this run directory and are excluded from training by construction (outside `trajectories/`, and each carries `metadata.excluded_reason=eval_episode`).
