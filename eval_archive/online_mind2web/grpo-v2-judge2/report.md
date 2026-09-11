# Student eval — `grpo-v2-judge2`

**verified task success: 11.8% across 50 online_mind2web tasks, parse-failure rate 8.7%, site-blocked 28.0%**

Success is over the SCORED episodes (4 verified of 34); site-blocked / unreachable / harness-error episodes are excluded from that denominator and reported separately, because folding bot-blocking into the success rate measures the network, not the model. Over all 50 episodes the rate is 8.0%.

## Run

| field | value |
| --- | --- |
| run id | `grpo-v2-judge2_online_mind2web_20260827T223307Z` |
| label | `grpo-v2-judge2` |
| started (UTC) | 2026-08-27T22:33:07.908100+00:00 |
| finished (UTC) | 2026-08-27T22:54:05.356817+00:00 |
| endpoint | `https://roman-maior-mknssh-2025--cua-serve-server.us-east.modal.direct/v1` |
| model | `cua-student-lora` |
| adapter / checkpoint | `(none declared)` |
| task source | `online_mind2web` — 300 tasks across live third-party websites (Online-Mind2Web, COLM 2025) |
| dataset | `https://huggingface.co/datasets/osunlp/Online-Mind2Web` |
| dataset revision | `(n/a)` |
| coordinate space | `viewport_abs` |
| step cap | 3.0x reference_length, clamped to [15, 50] (ungraded tasks: 30) |
| abort rules | 5 unproductive steps / 3 parse failures |
| task subset | full set |
| temperature / max tokens | 0.0 / 2048 |
| observation format | png @ 1280x720 |
| judge | claude-sonnet-5 (final + 2 mid screenshots) |
| history window | 10 |
| driver loop guard | disabled |
| browser | local DockerSandbox on the invoking host |
| repo commit | `076ef7ca96d6-dirty` |

<details><summary>task provenance</summary>

```json
{
  "tasks_file": "omw_sft_v1_tasks.jsonl",
  "rows_available": 50
}
```

</details>

## Metrics

| metric | value |
| --- | --- |
| episodes | 50 |
| scored episodes | 34 |
| judge deferred (usage limit) | 0 |
| verified success (scored) | 11.8% |
| verified success (all episodes) | 8.0% |
| verified AND non-meandering | 11.8% |
| site-blocked | 28.0% |
| parse-failure rate (steps) | 8.7% |
| malformed action lines | 43 |
| canonical action form | 100.0% (518/518 actions) |
| form deviations | none |
| done() emission rate | 22.0% |
| <think> emission rate (steps) | 100.0% |
| driver-refused action rate | 0.0% |
| emitted coords outside the viewport | 0/325 (0.0%) |
| out-of-viewport actions refused | 0 |
| episodes abandoned as stuck | 0 |
| mean steps on success | 4.5 |
| mean steps / reference_length on success | 1.12x |
| mean steps (all) | 9.86 |
| episodes that hit the step cap | 10 |

`verified` is the judge's verdict on the ORIGINAL goal after its own criteria checklist is enforced (`judged` is the raw claim); the efficiency grade is reported beside it rather than folded into it. `steps / reference_length` compares against the benchmark's own human step count, which is a more defensible efficiency denominator than our raw step counts.

`canonical action form` is the share of parsed actions spelled exactly as `format_action_for_training` emits them. The parser accepts any UNAMBIGUOUS spelling of a call (optional `action:`, bare or `<ref>`-wrapped coordinates, `wait` with or without `()`, omitted keyword arguments, single or double quotes, bare key names) and rejects genuinely ambiguous ones. A base model has only seen the grammar described in the system prompt, so charging surface deviation to its success rate would measure imitation of an unfamiliar convention rather than browsing. The deviation is reported here instead, and the tolerance is a harness parity key so both sides of a delta are parsed identically.

## Per level

| level | n | scored | verified | success rate | steps/ref | blocked |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 17 | 11 | 3 | 27.3% | 1.28x | 5 |
| medium | 16 | 10 | 0 | 0.0% | 0.0x | 6 |
| hard | 17 | 13 | 1 | 7.7% | 0.67x | 3 |

## Outcomes

| outcome | episodes |
| --- | --- |
| site_blocked | 14 |
| parse_failure_abort | 13 |
| completed | 11 |
| step_cap | 10 |
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
| `0170ca95038b05fa58d463fe627ac605` | hard | completed | yes | efficient | 8 | 36 | 12 | 0 | 0 | The agent searched GOV.UK, opened the Health and Care Worker visa results, and the final screenshot shows the eligibilit |
| `0059adc6b12a3822305deb68929b2de8` | medium | parse_failure_abort | no | meandering | 11 | 24 | 8 | 5 | 0 | All three screenshots are blank white frames with no visible content, and the agent's own log shows only one click follo |
| `0632e496d37badee0350dad358f047c5` | easy | step_cap | no | meandering | 15 | 15 | 4 | 0 | 0 | The agent repeatedly typed and re-typed fragmented, garbled search queries mixing English and Japanese terms into an acc |
| `0b2623e9fa5cea997f76490bcbc5220f_070826` | hard | parse_failure_abort | no | meandering | 14 | 39 | 13 | 3 | 0 | The final screenshot shows only a Nestle multi-language error page ('We're sorry, there seems to be a problem with our w |
| `01abae9608f2d8752a83e08f136f720c` | medium | site_blocked | no | - | 0 | 18 | 6 | 0 | 0 | waf_block |
| `16200f51d63f0a47a58fa17acd49e368` | easy | completed | yes | acceptable | 3 | 15 | 3 | 0 | 0 | The final screenshot shows a Cookpad recipe page for '茄子と舞茸のとろとろ味噌煮' (eggplant and maitake mushroom miso stew), with the |
| `0b51b4fa0295ae80ccd176ebdad6fff6` | hard | unreachable | no | - | 0 | 39 | 13 | 0 | 0 | page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://www.carmax.com/ Call log:   - navigating to "https://www.carmax.com/", waiting until "domcontentloaded"  |
| `046138801a05ddf56ad94e8672942496_070826` | medium | parse_failure_abort | no | meandering | 5 | 18 | 6 | 3 | 0 | The final screenshot shows only the FlightAware homepage with a cookie consent banner still visible; the agent never rea |
| `180ed2ec377ef3a4af9035a21522091a` | easy | step_cap | no | meandering | 15 | 15 | 5 | 0 | 0 | The final screenshot shows only a Cloudflare 'verifying you are human' security check on umich.edu; the agent never got  |
| `0b838cd54f826c59c71f600c56b89a11_070826` | hard | site_blocked | no | - | 0 | 33 | 11 | 0 | 0 | waf_block |
| `05483c50cc9b04c8ac44c574758fb2bd` | medium | site_blocked | no | - | 0 | 24 | 8 | 0 | 0 | waf_block |
| `1fc28d91d25ccd1c6ba268101326a654` | easy | completed | yes | efficient | 4 | 15 | 3 | 0 | 0 | Final screenshot shows the BTC/USD page on Google Finance with the 5D range button selected and a corresponding 5-day pr |
| `0e5536aaad9d3462b06cf725e6ed535a` | hard | completed | no | acceptable | 33 | 33 | 11 | 0 | 0 | The final screenshot shows an RDU AWT wait-times table with average figures, but the visible rows are dated 2025-03-08/0 |
| `070c907d34a4ce71dfdbea38f9c5d4d8` | medium | parse_failure_abort | no | meandering | 6 | 21 | 7 | 3 | 0 | The final screenshot shows a Healthgrades 'Oops' geo-restriction error page stating the content is unavailable in the us |
| `20a460a8fe1971b84411c5b1e6ac4186` | easy | parse_failure_abort | no | meandering | 8 | 15 | 3 | 3 | 0 | The final screenshot shows a StubHub bot-detection block page ('Access temporarily restricted'), not any theatre event l |
| `1223b07536a87e0170ff87cbbebd1d3c` | hard | completed | no | meandering | 21 | 50 | 24 | 0 | 0 | The final screenshot shows the quiz still on Question 1 of 10 with Numbers Covered set to '0-10' (not 11-12) and a timer |
| `07bdc595306729a028ba06cc7451a80a` | medium | completed | no | meandering | 13 | 21 | 7 | 0 | 0 | The final screenshot only shows the search results list with Book/Select buttons still unclicked; no ticket was actually |
| `2207bb4f21786690cfed20b37253fb8b` | easy | completed | yes | acceptable | 3 | 15 | 2 | 0 | 0 | The final screenshot shows the Weather Network's Calgary, AB current conditions page with wind speed clearly shown as 28 |
| `199be0b54a436daee74247971fc684ee_070826` | hard | site_blocked | no | - | 0 | 50 | 20 | 0 | 0 | waf_block |
| `07ec4a12cba8090e2dc524d558ac7675` | medium | site_blocked | no | - | 0 | 18 | 6 | 0 | 0 | waf_block |
| `23204728192da9f73197a613d9681c18` | easy | site_blocked | no | - | 0 | 15 | 3 | 0 | 0 | waf_block |
| `1ab384fb3a791edfb410213cc6b82151_110325` | hard | parse_failure_abort | no | meandering | 12 | 33 | 11 | 3 | 0 | The trajectory ends on the 'Generator' sim's About page (an electromagnetism sim, unrelated to nuclear decay), not any d |
| `0a0fa834ce41b5297c6474293383759d_110325` | medium | step_cap | no | meandering | 24 | 24 | 8 | 0 | 0 | The final screenshot shows a Carnival 'Transpacific' filtered search with 2 results departing from San Francisco and Syd |
| `246d654fab7c31d9651007e39e75f74f` | easy | step_cap | no | meandering | 15 | 15 | 5 | 0 | 0 | The final screenshot shows the Guest Reviews panel for Alpine Ridge open with the sort dropdown displaying 'Most Recent' |
| `1b867afecf072cb877ebfa4069263746` | hard | completed | no | acceptable | 3 | 45 | 15 | 0 | 0 | The final screenshot shows the state comparison page for Illinois and Michigan with the 'Unemployment - Women Trends' se |
| `0a54069a0ef542e571d1fee7f39c93d5` | medium | step_cap | no | meandering | 18 | 18 | 6 | 0 | 0 | All 18 recorded actions were 'wait' and every screenshot, including the final one, shows a CloudFront 403 error page blo |
| `2532fd402d3c741b79894e6ff2269f53` | easy | parse_failure_abort | no | meandering | 3 | 15 | 3 | 3 | 0 | The run performed zero actions and the final screenshot shows a Thumbtack geo-block error page, not any electrician sear |
| `1bc154377120ec15b18dbabdba49c741_051526` | hard | completed | no | meandering | 3 | 50 | 24 | 0 | 0 | The agent only clicked, typed 'BTS', and waited before declaring done; the final screenshot shows StubHub's bot-detectio |
| `0e42c3a73f2aece1f854e0ba55b7c8b0` | medium | parse_failure_abort | no | meandering | 18 | 21 | 7 | 5 | 0 | The agent never reached a gas station results page; instead it got stuck on a Google account sign-in screen, repeatedly  |
| `26810ed9c123a62992e3eed31db3c5ee` | easy | unreachable | no | - | 0 | 15 | 4 | 0 | 0 | page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://www.accuweather.com/ Call log:   - navigating to "https://www.accuweather.com/", waiting until "domcontentloaded"  |
| `1c3b747ae12ccee895745f82e3f2ef8a` | hard | step_cap | no | meandering | 33 | 33 | 11 | 0 | 0 | The agent never sorted or compared competitions by prize amount; it ended up in the 'all-new-titanic2' competition's Cod |
| `11857213ca01510f12813740afd59918_110325` | medium | completed | no | meandering | 21 | 36 | 12 | 0 | 0 | The final screenshot shows a 'Naples Wrapped' loafer/moccasin style (not a clog) added to cart in size EU 35 / UK 2.5, w |
| `26a0e5c21c145dd8448aa92f35bec5ea` | easy | parse_failure_abort | no | meandering | 3 | 15 | 3 | 3 | 0 | No actions were executed and the final screenshot shows a Healthgrades geo-block error page ('This page is not available |
| `27fa3ac20745d3d35e89fae157f63069` | hard | step_cap | no | meandering | 33 | 33 | 11 | 0 | 0 | The final screenshot is just the fresh homepage search with an empty search box, no filters applied, and no results show |
| `11abb668c751dd56bb41f296a8bb3a13` | medium | site_blocked | no | - | 0 | 30 | 10 | 0 | 0 | waf_block |
| `271b36efd4346721b5542488ff997042` | easy | site_blocked | no | - | 0 | 15 | 4 | 0 | 0 | waf_block |
| `28e7574e7bd6d14f36d2988a5ef2bd23` | hard | parse_failure_abort | no | meandering | 3 | 36 | 12 | 3 | 0 | The run recorded zero actions and ended on a 404 'Page not found' error page unrelated to any job search interface, show |
| `123e8c2fc453f55fadd1d0b9aaf94df4` | medium | site_blocked | no | - | 0 | 24 | 8 | 0 | 0 | waf_block |
| `29526b17a32485742b5ab63507e99417` | easy | site_blocked | no | - | 0 | 15 | 3 | 0 | 0 | waf_block |
| `33bd2cdcea4fcc42a09a8a1e4e5841c6_110325` | hard | step_cap | no | meandering | 50 | 50 | 20 | 0 | 0 | The site returned an error page ('temporarily unavailable') across all three screenshots, so the agent never reached the |
| `157f4a79d55e8fa3fd55ba772ba40fbc_070826` | medium | site_blocked | no | - | 0 | 27 | 9 | 0 | 0 | waf_block |
| `29b7372d5a3884a2ba831af2d117af3c` | easy | parse_failure_abort | no | meandering | 6 | 15 | 3 | 3 | 0 | The agent searched Microsoft on Google Finance but then navigated to a Google Sign-in page and typed an invalid email in |
| `3443e9c3151fef19a3c3a45eb2c13640_070826` | hard | step_cap | no | meandering | 36 | 36 | 12 | 0 | 0 | All three screenshots show the identical untouched calculator form (June 2026, no date selected, Calculate not pressed), |
| `15be05973fba714e490cd9c884e4f072` | medium | parse_failure_abort | no | meandering | 15 | 18 | 6 | 3 | 0 | The final screenshot shows a 404 'Page not found' error, meaning the agent's navigation led to a broken link rather than |
| `34992feb69eb8e788faa06868b365c49` | easy | site_blocked | no | - | 0 | 15 | 5 | 0 | 0 | waf_block |
| `367d843c640637745e8fafa741cca13b` | hard | site_blocked | no | - | 0 | 45 | 15 | 0 | 0 | waf_block |
| `1aeca99e6a60b0e3aefb3ef212bdce79` | medium | step_cap | no | meandering | 30 | 30 | 10 | 0 | 0 | The final screenshot shows an empty Advanced Job Search filter form (Location, Work Schedule, Min. Salary, keyword all b |
| `34ccd15a8ea8fd3895af83f5ccf62369` | easy | parse_failure_abort | no | meandering | 3 | 15 | 3 | 3 | 0 | No actions were executed and the final screenshot shows only a CloudFront 403 error page, so no information about lost i |
| `3ef64f34eae59c9fac7ee9a4f18b4a0c` | hard | completed | no | meandering | 5 | 48 | 16 | 0 | 0 | The final screenshot shows regular YouTube (with a 'Try YouTube Kids' banner, meaning it's not actually on Kids) still o |

---

Episodes are recorded under this run directory and are excluded from training by construction (outside `trajectories/`, and each carries `metadata.excluded_reason=eval_episode`).
