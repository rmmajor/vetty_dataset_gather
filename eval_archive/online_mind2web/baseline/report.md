# Student eval — `baseline_qwen35_9b`

**verified task success: 8.8% across 50 online_mind2web tasks, parse-failure rate 22.3%, site-blocked 28.0%**

Success is over the SCORED episodes (3 verified of 34); site-blocked / unreachable / harness-error episodes are excluded from that denominator and reported separately, because folding bot-blocking into the success rate measures the network, not the model. Over all 50 episodes the rate is 6.0%.

## Run

| field | value |
| --- | --- |
| run id | `baseline_qwen35_9b_online_mind2web_20260826T165014Z` |
| label | `baseline_qwen35_9b` |
| started (UTC) | 2026-08-26T16:50:14.951097+00:00 |
| finished (UTC) | 2026-08-26T17:40:08.065576+00:00 |
| endpoint | `https://<modal-workspace>--cua-serve-server.us-east.modal.direct/v1` |
| model | `cua-student` |
| adapter / checkpoint | `(none declared)` |
| task source | `online_mind2web` — 300 tasks across live third-party websites (Online-Mind2Web, COLM 2025) |
| dataset | `osunlp/Online-Mind2Web` |
| dataset revision | `eacad896a84dc5b65e29b0b06e4699ab0544d701` |
| coordinate space | `normalized_1000` |
| step cap | 3.0x reference_length, clamped to [15, 50] (ungraded tasks: 30) |
| abort rules | 5 unproductive steps / 3 parse failures |
| task subset | level-stratified 50 |
| temperature / max tokens | 0.0 / 2048 |
| observation format | png @ 1280x720 |
| judge | claude-sonnet-5 (final + 2 mid screenshots) |
| history window | 10 |
| driver loop guard | disabled |
| browser | local DockerSandbox on the invoking host |
| repo commit | `89603f852203` |

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
| verified success (scored) | 8.8% |
| verified success (all episodes) | 6.0% |
| verified AND non-meandering | 8.8% |
| site-blocked | 28.0% |
| parse-failure rate (steps) | 22.3% |
| malformed action lines | 755 |
| canonical action form | 26.8% (74/276 actions) |
| form deviations | default_kwargs 164, bare_coords 115, missing_prefix 58, missing_parens 7, other 2, bare_keys 1 |
| done() emission rate | 28.0% |
| <think> emission rate (steps) | 78.0% |
| driver-refused action rate | 0.4% |
| emitted coords outside the viewport | 45/211 (21.3%) |
| out-of-viewport actions refused | 0 |
| episodes abandoned as stuck | 0 |
| mean steps on success | 5.67 |
| mean steps / reference_length on success | 1.36x |
| mean steps (all) | 6.54 |
| episodes that hit the step cap | 4 |

`verified` is the judge's verdict on the ORIGINAL goal; the efficiency grade is reported beside it rather than folded into it. `steps / reference_length` compares against the benchmark's own human step count, which is a more defensible efficiency denominator than our raw step counts.

`canonical action form` is the share of parsed actions spelled exactly as `format_action_for_training` emits them. The parser accepts any UNAMBIGUOUS spelling of a call (optional `action:`, bare or `<ref>`-wrapped coordinates, `wait` with or without `()`, omitted keyword arguments, single or double quotes, bare key names) and rejects genuinely ambiguous ones. A base model has only seen the grammar described in the system prompt, so charging surface deviation to its success rate would measure imitation of an unfamiliar convention rather than browsing. The deviation is reported here instead, and the tolerance is a harness parity key so both sides of a delta are parsed identically.

## Per level

| level | n | scored | verified | success rate | steps/ref | blocked |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 17 | 11 | 2 | 18.2% | 1.67x | 5 |
| medium | 16 | 10 | 0 | 0.0% | 0.0x | 6 |
| hard | 17 | 13 | 1 | 7.7% | 0.75x | 3 |

## Outcomes

| outcome | episodes |
| --- | --- |
| parse_failure_abort | 16 |
| site_blocked | 14 |
| completed | 14 |
| step_cap | 4 |
| unreachable | 2 |

## Per-domain

| domain | n | scored | verified | success rate | blocked |
| --- | --- | --- | --- | --- | --- |
| www.google.com | 3 | 3 | 1 | 33.3% | 0 |
| cookpad.com | 2 | 2 | 0 | 0.0% | 0 |
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
| us.trip.com | 1 | 1 | 0 | 0.0% | 0 |
| www.theweathernetwork.com | 1 | 1 | 1 | 100.0% | 0 |
| www.dillards.com | 1 | 0 | 0 | 0.0% | 1 |
| www.mayoclinic.org | 1 | 0 | 0 | 0.0% | 1 |
| phet.colorado.edu | 1 | 1 | 0 | 0.0% | 0 |
| www.carnival.com | 1 | 1 | 0 | 0.0% | 0 |
| www.recreation.gov | 1 | 1 | 0 | 0.0% | 0 |
| www.americashealthrankings.org | 1 | 1 | 0 | 0.0% | 0 |
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
| www.webmd.com | 1 | 1 | 0 | 0.0% | 0 |
| ohio.gov | 1 | 1 | 0 | 0.0% | 0 |
| www.dmv.virginia.gov | 1 | 0 | 0 | 0.0% | 1 |
| www.apartments.com | 1 | 0 | 0 | 0.0% | 1 |
| www.ca.gov | 1 | 1 | 0 | 0.0% | 0 |
| us.megabus.com | 1 | 1 | 0 | 0.0% | 0 |
| www.youtube.com | 1 | 1 | 0 | 0.0% | 0 |

## Episodes

| task | level | outcome | verified | efficiency | steps | cap | ref | parse fails | driver errors | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `005be9dd91c95669d6ddde9ae667125c` | easy | site_blocked | no | - | 0 | 15 | 4 | 0 | 0 | waf_block |
| `0170ca95038b05fa58d463fe627ac605` | hard | parse_failure_abort | yes | acceptable | 9 | 36 | 12 | 3 | 0 | The final screenshot shows the GOV.UK Health and Care Worker visa page with eligibility details, which directly answers  |
| `0059adc6b12a3822305deb68929b2de8` | medium | completed | no | meandering | 15 | 24 | 8 | 0 | 0 | All three screenshots show an identical WAF/proxy rejection page ('The requested URL was rejected'), meaning the site wa |
| `0632e496d37badee0350dad358f047c5` | easy | parse_failure_abort | no | meandering | 10 | 15 | 4 | 3 | 0 | The final state is still an empty search-results page ('レシピがみつかりませんか？') with no recipe browsed, so the goal of browsing  |
| `0b2623e9fa5cea997f76490bcbc5220f_070826` | hard | parse_failure_abort | no | meandering | 16 | 39 | 13 | 5 | 0 | The site returned a persistent multi-language server error page throughout the entire run, and the agent repeatedly type |
| `01abae9608f2d8752a83e08f136f720c` | medium | site_blocked | no | - | 0 | 18 | 6 | 0 | 0 | waf_block |
| `16200f51d63f0a47a58fa17acd49e368` | easy | completed | no | - | 6 | 15 | 3 | 0 | 1 | verifier unavailable |
| `0b51b4fa0295ae80ccd176ebdad6fff6` | hard | unreachable | no | - | 0 | 39 | 13 | 0 | 0 | page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://www.carmax.com/ Call log:   - navigating to "https://www.carmax.com/", waiting until "domcontentloaded"  |
| `046138801a05ddf56ad94e8672942496_070826` | medium | parse_failure_abort | no | meandering | 4 | 18 | 6 | 3 | 0 | The run only clicked once, opening a products dropdown menu unrelated to community discussions, and aborted with a parse |
| `180ed2ec377ef3a4af9035a21522091a` | easy | completed | no | meandering | 2 | 15 | 5 | 0 | 0 | The final screenshot shows only a Cloudflare bot-verification challenge page, not any UM-Dearborn giving/donation page,  |
| `0b838cd54f826c59c71f600c56b89a11_070826` | hard | site_blocked | no | - | 0 | 33 | 11 | 0 | 0 | waf_block |
| `05483c50cc9b04c8ac44c574758fb2bd` | medium | site_blocked | no | - | 0 | 24 | 8 | 0 | 0 | waf_block |
| `1fc28d91d25ccd1c6ba268101326a654` | easy | completed | yes | efficient | 4 | 15 | 3 | 0 | 0 | The final screenshot shows the BTC/USD chart on Google Finance with the 5D range button selected, correctly displaying a |
| `0e5536aaad9d3462b06cf725e6ed535a` | hard | parse_failure_abort | no | meandering | 5 | 33 | 11 | 3 | 0 | The final screenshot shows only the generic CBP 'For U.S. Citizens' landing page, not the wait-time statistics page for  |
| `070c907d34a4ce71dfdbea38f9c5d4d8` | medium | completed | no | meandering | 7 | 21 | 7 | 2 | 0 | The final screenshot shows a Healthgrades geo-block error page ('Oops... not available in your area'), not a pediatric d |
| `20a460a8fe1971b84411c5b1e6ac4186` | easy | step_cap | no | meandering | 15 | 15 | 3 | 2 | 0 | All three screenshots (start, mid-run, final) show the identical Lviv-location state with the location dropdown open and |
| `1223b07536a87e0170ff87cbbebd1d3c` | hard | parse_failure_abort | no | meandering | 15 | 50 | 24 | 5 | 0 | The agent never got past the quiz setup screen (Numbers Covered still shows 0-10, not 11-12) across all three screenshot |
| `07bdc595306729a028ba06cc7451a80a` | medium | completed | no | acceptable | 11 | 21 | 7 | 2 | 0 | The final screenshot only shows a search results list with a 06:31 departure train and its Book button visible, but no t |
| `2207bb4f21786690cfed20b37253fb8b` | easy | completed | yes | efficient | 4 | 15 | 2 | 0 | 0 | Final screenshot shows Calgary, AB current weather with Wind 9 km/h SW clearly visible, matching the goal. Four steps (c |
| `199be0b54a436daee74247971fc684ee_070826` | hard | site_blocked | no | - | 0 | 50 | 20 | 0 | 0 | waf_block |
| `07ec4a12cba8090e2dc524d558ac7675` | medium | site_blocked | no | - | 0 | 18 | 6 | 0 | 0 | waf_block |
| `23204728192da9f73197a613d9681c18` | easy | site_blocked | no | - | 0 | 15 | 3 | 0 | 0 | waf_block |
| `1ab384fb3a791edfb410213cc6b82151_110325` | hard | parse_failure_abort | no | meandering | 7 | 33 | 11 | 3 | 0 | The final screenshot shows only the generic PhET simulations browse page, with no Beryllium proton-decay simulation open |
| `0a0fa834ce41b5297c6474293383759d_110325` | medium | parse_failure_abort | no | meandering | 6 | 24 | 8 | 4 | 0 | The final screenshot shows only the Carnival homepage with no search performed, no cruise located, and no onboard activi |
| `246d654fab7c31d9651007e39e75f74f` | easy | parse_failure_abort | no | meandering | 13 | 15 | 5 | 3 | 0 | The final state shows an availability/reservation grid, not any 5-star review content — the agent never reached or opene |
| `1b867afecf072cb877ebfa4069263746` | hard | parse_failure_abort | no | - | 27 | 45 | 15 | 3 | 0 | verifier unavailable |
| `0a54069a0ef542e571d1fee7f39c93d5` | medium | completed | no | meandering | 1 | 18 | 6 | 0 | 0 | The agent performed zero actions and the final screenshot shows a 403 CloudFront geo-block error, not any dog-listing co |
| `2532fd402d3c741b79894e6ff2269f53` | easy | completed | no | meandering | 3 | 15 | 3 | 2 | 0 | The site returned a geo-blocked error page ('coming from a country which we do not currently support') and no electricia |
| `1bc154377120ec15b18dbabdba49c741_051526` | hard | completed | no | meandering | 4 | 50 | 24 | 0 | 0 | The run ends on a StubHub bot-detection block page with no search results, seats, or checkout progress ever reached; the |
| `0e42c3a73f2aece1f854e0ba55b7c8b0` | medium | parse_failure_abort | no | meandering | 8 | 21 | 7 | 3 | 0 | The run aborted on a parse failure after only a search and one click; the final screenshot shows a gas station with rati |
| `26810ed9c123a62992e3eed31db3c5ee` | easy | unreachable | no | - | 0 | 15 | 4 | 0 | 0 | page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://www.accuweather.com/ Call log:   - navigating to "https://www.accuweather.com/", waiting until "domcontentloaded"  |
| `1c3b747ae12ccee895745f82e3f2ef8a` | hard | parse_failure_abort | no | meandering | 16 | 33 | 11 | 3 | 0 | The agent ended on Titanic's Leaderboard tab showing submission scores, not the Code tab showing notebook votes, and Tit |
| `11857213ca01510f12813740afd59918_110325` | medium | completed | no | meandering | 8 | 36 | 12 | 2 | 0 | Final screenshot shows the 'Naples Wrapped' suede loafer/moccasin product page, not a Birkenstock clog, with no size sel |
| `26a0e5c21c145dd8448aa92f35bec5ea` | easy | step_cap | no | meandering | 15 | 15 | 3 | 3 | 0 | All three screenshots show the same Healthgrades geo-block 'Oops' error page across the entire run, so no optometrist li |
| `27fa3ac20745d3d35e89fae157f63069` | hard | parse_failure_abort | no | meandering | 8 | 33 | 11 | 3 | 0 | The final screenshot shows a generic unfiltered course search results page with no subject, term, day, or time filters a |
| `11abb668c751dd56bb41f296a8bb3a13` | medium | site_blocked | no | - | 0 | 30 | 10 | 0 | 0 | waf_block |
| `271b36efd4346721b5542488ff997042` | easy | site_blocked | no | - | 0 | 15 | 4 | 0 | 0 | waf_block |
| `28e7574e7bd6d14f36d2988a5ef2bd23` | hard | parse_failure_abort | no | meandering | 4 | 36 | 12 | 3 | 0 | The single click led to a 404 error page unrelated to any job search progress, and the agent aborted with a parse failur |
| `123e8c2fc453f55fadd1d0b9aaf94df4` | medium | site_blocked | no | - | 0 | 24 | 8 | 0 | 0 | waf_block |
| `29526b17a32485742b5ab63507e99417` | easy | site_blocked | no | - | 0 | 15 | 3 | 0 | 0 | waf_block |
| `33bd2cdcea4fcc42a09a8a1e4e5841c6_110325` | hard | parse_failure_abort | no | meandering | 17 | 50 | 20 | 3 | 0 | All three screenshots show the Raising Cane's site returning an error page ('this site is temporarily unavailable'); the |
| `157f4a79d55e8fa3fd55ba772ba40fbc_070826` | medium | site_blocked | no | - | 0 | 27 | 9 | 0 | 0 | waf_block |
| `29b7372d5a3884a2ba831af2d117af3c` | easy | step_cap | no | meandering | 15 | 15 | 3 | 5 | 0 | Final state is a Motley Fool email-capture popup, not Google Finance news for Microsoft; the agent wandered off-site aft |
| `3443e9c3151fef19a3c3a45eb2c13640_070826` | hard | parse_failure_abort | no | meandering | 10 | 36 | 12 | 3 | 0 | The agent reached the Ovulation Calculator page (date-picker visible) but the run ended in a parse_failure_abort with no |
| `15be05973fba714e490cd9c884e4f072` | medium | step_cap | no | meandering | 18 | 18 | 6 | 0 | 0 | All three screenshots (steps ~7, ~13, and final) show the same 404 error page, showing no forward progress was ever made |
| `34992feb69eb8e788faa06868b365c49` | easy | site_blocked | no | - | 0 | 15 | 5 | 0 | 0 | waf_block |
| `367d843c640637745e8fafa741cca13b` | hard | site_blocked | no | - | 0 | 45 | 15 | 0 | 0 | waf_block |
| `1aeca99e6a60b0e3aefb3ef212bdce79` | medium | completed | no | meandering | 15 | 30 | 10 | 0 | 0 | Only a search query was entered on CA.gov and results shown (with $2,000+ min salary, not $4,000+); the final screenshot |
| `34ccd15a8ea8fd3895af83f5ccf62369` | easy | completed | no | meandering | 3 | 15 | 3 | 2 | 0 | The agent performed zero actions and the final screenshot shows a CloudFront 403 error page, not any information about l |
| `3ef64f34eae59c9fac7ee9a4f18b4a0c` | hard | completed | no | meandering | 6 | 48 | 16 | 0 | 0 | The final state shows regular YouTube (not YouTube Kids) displaying search results for the typed query, with no course v |

---

Episodes are recorded under this run directory and are excluded from training by construction (outside `trajectories/`, and each carries `metadata.excluded_reason=eval_episode`).
