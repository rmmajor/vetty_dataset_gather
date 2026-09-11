# Student eval — `sft-v1-judge2`

**verified task success: 9.4% across 50 online_mind2web tasks, parse-failure rate 8.0%, site-blocked 28.0%**

Success is over the SCORED episodes (3 verified of 32); site-blocked / unreachable / harness-error episodes are excluded from that denominator and reported separately, because folding bot-blocking into the success rate measures the network, not the model. Over all 50 episodes the rate is 6.0%.

## Run

| field | value |
| --- | --- |
| run id | `sft-v1-judge2_online_mind2web_20260827T213815Z` |
| label | `sft-v1-judge2` |
| started (UTC) | 2026-08-27T21:50:32.797154+00:00 |
| finished (UTC) | 2026-08-27T22:05:49.810686+00:00 |
| endpoint | `https://bombaleylo2--cua-serve-server.us-east.modal.direct/v1` |
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
| scored episodes | 32 |
| judge deferred (usage limit) | 0 |
| verified success (scored) | 9.4% |
| verified success (all episodes) | 6.0% |
| verified AND non-meandering | 9.4% |
| site-blocked | 28.0% |
| parse-failure rate (steps) | 8.0% |
| malformed action lines | 40 |
| canonical action form | 100.0% (509/509 actions) |
| form deviations | none |
| done() emission rate | 16.0% |
| <think> emission rate (steps) | 100.0% |
| driver-refused action rate | 0.0% |
| emitted coords outside the viewport | 0/337 (0.0%) |
| out-of-viewport actions refused | 0 |
| episodes abandoned as stuck | 0 |
| mean steps on success | 3.33 |
| mean steps / reference_length on success | 1.33x |
| mean steps (all) | 9.98 |
| episodes that hit the step cap | 11 |

`verified` is the judge's verdict on the ORIGINAL goal after its own criteria checklist is enforced (`judged` is the raw claim); the efficiency grade is reported beside it rather than folded into it. `steps / reference_length` compares against the benchmark's own human step count, which is a more defensible efficiency denominator than our raw step counts.

`canonical action form` is the share of parsed actions spelled exactly as `format_action_for_training` emits them. The parser accepts any UNAMBIGUOUS spelling of a call (optional `action:`, bare or `<ref>`-wrapped coordinates, `wait` with or without `()`, omitted keyword arguments, single or double quotes, bare key names) and rejects genuinely ambiguous ones. A base model has only seen the grammar described in the system prompt, so charging surface deviation to its success rate would measure imitation of an unfamiliar convention rather than browsing. The deviation is reported here instead, and the tolerance is a harness parity key so both sides of a delta are parsed identically.

## Per level

| level | n | scored | verified | success rate | steps/ref | blocked |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 17 | 11 | 3 | 27.3% | 1.33x | 5 |
| medium | 16 | 9 | 0 | 0.0% | 0.0x | 6 |
| hard | 17 | 12 | 0 | 0.0% | 0.0x | 3 |

## Outcomes

| outcome | episodes |
| --- | --- |
| site_blocked | 14 |
| parse_failure_abort | 13 |
| step_cap | 11 |
| completed | 8 |
| unreachable | 2 |
| harness_error | 2 |

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
| www.gov.uk | 1 | 1 | 0 | 0.0% | 0 |
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
| www.carnival.com | 1 | 0 | 0 | 0.0% | 0 |
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
| www.webmd.com | 1 | 0 | 0 | 0.0% | 0 |
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
| `0170ca95038b05fa58d463fe627ac605` | hard | step_cap | no | meandering | 36 | 36 | 12 | 0 | 0 | The final screenshot shows only a cookie consent banner and the top of a generic 'Skilled Worker visa' page with no visi |
| `0059adc6b12a3822305deb68929b2de8` | medium | parse_failure_abort | no | meandering | 3 | 24 | 8 | 3 | 0 | The run executed zero actions and the final screenshot shows only a URL-rejected error page, so no progress toward findi |
| `0632e496d37badee0350dad358f047c5` | easy | step_cap | no | meandering | 15 | 15 | 4 | 0 | 0 | The agent repeatedly typed and re-typed search queries (English then many Japanese variants) that all got concatenated i |
| `0b2623e9fa5cea997f76490bcbc5220f_070826` | hard | parse_failure_abort | no | meandering | 10 | 39 | 13 | 4 | 0 | All screenshots show a site error page (Nestle 'We're sorry, there seems to be a problem with our website' maintenance m |
| `01abae9608f2d8752a83e08f136f720c` | medium | site_blocked | no | - | 0 | 18 | 6 | 0 | 0 | waf_block |
| `16200f51d63f0a47a58fa17acd49e368` | easy | completed | yes | acceptable | 3 | 15 | 3 | 0 | 0 | The final screenshot shows a recipe titled 'なすとキノコのバジル炒め' (eggplant and mushroom basil stir-fry) with なす listed in ingre |
| `0b51b4fa0295ae80ccd176ebdad6fff6` | hard | unreachable | no | - | 0 | 39 | 13 | 0 | 0 | page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://www.carmax.com/ Call log:   - navigating to "https://www.carmax.com/", waiting until "domcontentloaded"  |
| `046138801a05ddf56ad94e8672942496_070826` | medium | step_cap | no | meandering | 18 | 18 | 6 | 0 | 0 | The final screenshot still shows the 'Top' topics list filtered to the past year (Aug 2025–Aug 2026) rather than all tim |
| `180ed2ec377ef3a4af9035a21522091a` | easy | step_cap | no | meandering | 15 | 15 | 5 | 0 | 0 | All three screenshots show only a Cloudflare 'Verifying you are human' security check on umich.edu; the agent never got  |
| `0b838cd54f826c59c71f600c56b89a11_070826` | hard | site_blocked | no | - | 0 | 33 | 11 | 0 | 0 | waf_block |
| `05483c50cc9b04c8ac44c574758fb2bd` | medium | site_blocked | no | - | 0 | 24 | 8 | 0 | 0 | waf_block |
| `1fc28d91d25ccd1c6ba268101326a654` | easy | completed | yes | efficient | 3 | 15 | 3 | 0 | 0 | The final screenshot shows the Bitcoin/USD price chart on Google Finance with the 5D range selector actively highlighted |
| `0e5536aaad9d3462b06cf725e6ed535a` | hard | step_cap | no | meandering | 33 | 33 | 11 | 0 | 0 | The final screenshot shows only the generic CBP 'Advisories and Wait Times' landing page with general advisory text, not |
| `070c907d34a4ce71dfdbea38f9c5d4d8` | medium | parse_failure_abort | no | meandering | 5 | 21 | 7 | 3 | 0 | The agent only clicked and waited before Healthgrades blocked access with a region-restriction error page; no search was |
| `20a460a8fe1971b84411c5b1e6ac4186` | easy | parse_failure_abort | no | meandering | 6 | 15 | 3 | 3 | 0 | The final screenshot shows a StubHub access-restricted/bot-detection page, not a Las Vegas theatre events listing, so ne |
| `1223b07536a87e0170ff87cbbebd1d3c` | hard | completed | no | meandering | 33 | 50 | 24 | 0 | 0 | The final screenshot is identical to the initial setup screen (Numbers Covered still 0-10, Time per Question still 30 Se |
| `07bdc595306729a028ba06cc7451a80a` | medium | step_cap | no | meandering | 21 | 21 | 7 | 0 | 0 | The final screenshot shows the search form still set to Beijing (Any) to Beijing (Any), identical to the pre-run state;  |
| `2207bb4f21786690cfed20b37253fb8b` | easy | completed | yes | acceptable | 4 | 15 | 2 | 0 | 0 | The final screenshot shows Calgary, AB weather with 'Today's Conditions' including Wind 19 km/h, which directly answers  |
| `199be0b54a436daee74247971fc684ee_070826` | hard | site_blocked | no | - | 0 | 50 | 20 | 0 | 0 | waf_block |
| `07ec4a12cba8090e2dc524d558ac7675` | medium | site_blocked | no | - | 0 | 18 | 6 | 0 | 0 | waf_block |
| `23204728192da9f73197a613d9681c18` | easy | site_blocked | no | - | 0 | 15 | 3 | 0 | 0 | waf_block |
| `1ab384fb3a791edfb410213cc6b82151_110325` | hard | parse_failure_abort | no | meandering | 14 | 33 | 11 | 3 | 0 | The final screenshot only shows the PhET simulations browse/filter listing page (Physics category with search results li |
| `0a0fa834ce41b5297c6474293383759d_110325` | medium | harness_error | no | - | 9 | 24 | 8 | 0 | 0 | screenshot failed: page.screenshot: Timeout 30000ms exceeded. Call log:   - taking page screenshot   - waiting for fonts to load...  |
| `246d654fab7c31d9651007e39e75f74f` | easy | step_cap | no | meandering | 15 | 15 | 5 | 0 | 0 | The final screenshot shows the Alpine Ridge campground detail page with its rating (4.5, 196 reviews) visible, but no re |
| `1b867afecf072cb877ebfa4069263746` | hard | parse_failure_abort | no | meandering | 37 | 45 | 15 | 3 | 0 | The final screenshot shows a 'Drive Alone to Work - Female' measure page for the United States, not an unemployment figu |
| `0a54069a0ef542e571d1fee7f39c93d5` | medium | step_cap | no | meandering | 18 | 18 | 6 | 0 | 0 | All three screenshots show a CloudFront 403 error blocking access by country; the agent never reached any usable page an |
| `2532fd402d3c741b79894e6ff2269f53` | easy | parse_failure_abort | no | meandering | 3 | 15 | 3 | 3 | 0 | The agent never took any actions; the final screenshot shows a Thumbtack geo-block error page stating the request is com |
| `1bc154377120ec15b18dbabdba49c741_051526` | hard | completed | no | meandering | 3 | 50 | 24 | 0 | 0 | The final screenshot shows a StubHub bot-detection block page ('Access temporarily restricted'), meaning the agent never |
| `0e42c3a73f2aece1f854e0ba55b7c8b0` | medium | parse_failure_abort | no | meandering | 11 | 21 | 7 | 3 | 0 | The agent found only gas stations rated 2.8-3.5 (none above 4.0) and then veered into a Google account sign-in flow, end |
| `26810ed9c123a62992e3eed31db3c5ee` | easy | unreachable | no | - | 0 | 15 | 4 | 0 | 0 | page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://www.accuweather.com/ Call log:   - navigating to "https://www.accuweather.com/", waiting until "domcontentloaded"  |
| `1c3b747ae12ccee895745f82e3f2ef8a` | hard | step_cap | no | meandering | 33 | 33 | 11 | 0 | 0 | The agent correctly navigated to ARC Prize 2026 - ARC-AGI-3 ($850,000, the top prize among active competitions) but the  |
| `11857213ca01510f12813740afd59918_110325` | medium | completed | no | meandering | 23 | 36 | 12 | 0 | 0 | The final screenshot shows 'Naples Wrapped' added to cart, which is a wrapped moccasin/loafer style, not a clog, in size |
| `26a0e5c21c145dd8448aa92f35bec5ea` | easy | parse_failure_abort | no | meandering | 3 | 15 | 3 | 3 | 0 | The run recorded zero actions and the final screenshot shows a Healthgrades geo-block error page, with no optometrist li |
| `27fa3ac20745d3d35e89fae157f63069` | hard | parse_failure_abort | no | meandering | 8 | 33 | 11 | 3 | 0 | The final screenshot shows Stanford's general academic calendar page for 2026-2027, not a course schedule search, and it |
| `11abb668c751dd56bb41f296a8bb3a13` | medium | site_blocked | no | - | 0 | 30 | 10 | 0 | 0 | waf_block |
| `271b36efd4346721b5542488ff997042` | easy | site_blocked | no | - | 0 | 15 | 4 | 0 | 0 | waf_block |
| `28e7574e7bd6d14f36d2988a5ef2bd23` | hard | parse_failure_abort | no | meandering | 3 | 36 | 12 | 3 | 0 | No actions were executed (0 steps, parse_failure_abort) and the final screenshot shows a 404 'Page not found' error on a |
| `123e8c2fc453f55fadd1d0b9aaf94df4` | medium | site_blocked | no | - | 0 | 24 | 8 | 0 | 0 | waf_block |
| `29526b17a32485742b5ab63507e99417` | easy | site_blocked | no | - | 0 | 15 | 3 | 0 | 0 | waf_block |
| `33bd2cdcea4fcc42a09a8a1e4e5841c6_110325` | hard | step_cap | no | meandering | 50 | 50 | 20 | 0 | 0 | The site returned a persistent error page ('site is temporarily unavailable') across all three screenshots, so no items  |
| `157f4a79d55e8fa3fd55ba772ba40fbc_070826` | medium | site_blocked | no | - | 0 | 27 | 9 | 0 | 0 | waf_block |
| `29b7372d5a3884a2ba831af2d117af3c` | easy | completed | no | acceptable | 3 | 15 | 3 | 0 | 0 | The final screenshot shows the MSFT:NASDAQ page with a 'News stories' list visible, but the agent only scrolled to revea |
| `3443e9c3151fef19a3c3a45eb2c13640_070826` | hard | harness_error | no | - | 0 | 36 | 12 | 0 | 0 | screenshot failed: page.screenshot: Timeout 30000ms exceeded. Call log:   - taking page screenshot   - waiting for fonts to load...  |
| `15be05973fba714e490cd9c884e4f072` | medium | parse_failure_abort | no | meandering | 15 | 18 | 6 | 3 | 0 | All three screenshots show a 404 Page Not Found error on OhioGov's site, with no license information ever reached; the a |
| `34992feb69eb8e788faa06868b365c49` | easy | site_blocked | no | - | 0 | 15 | 5 | 0 | 0 | waf_block |
| `367d843c640637745e8fafa741cca13b` | hard | site_blocked | no | - | 0 | 45 | 15 | 0 | 0 | waf_block |
| `1aeca99e6a60b0e3aefb3ef212bdce79` | medium | step_cap | no | meandering | 30 | 30 | 10 | 0 | 0 | The final screenshot shows an empty CalCareers Advanced Filters form with all fields (Job Categories, Location, Min. Sal |
| `34ccd15a8ea8fd3895af83f5ccf62369` | easy | parse_failure_abort | no | meandering | 3 | 15 | 3 | 3 | 0 | No actions were executed and the final screenshot shows only a CloudFront 403 error page, meaning the site was never rea |
| `3ef64f34eae59c9fac7ee9a4f18b4a0c` | hard | completed | no | acceptable | 13 | 48 | 16 | 0 | 0 | The final screenshot shows regular youtube.com (Ukrainian interface, 'Увійти' sign-in option, ad-supported), not YouTube |

---

Episodes are recorded under this run directory and are excluded from training by construction (outside `trajectories/`, and each carries `metadata.excluded_reason=eval_episode`).
