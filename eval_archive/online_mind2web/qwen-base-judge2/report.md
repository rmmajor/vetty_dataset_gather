# Student eval — `qwen-base-judge2`

**verified task success: 14.7% across 50 online_mind2web tasks, parse-failure rate 20.6%, site-blocked 28.0%**

Success is over the SCORED episodes (5 verified of 34); site-blocked / unreachable / harness-error episodes are excluded from that denominator and reported separately, because folding bot-blocking into the success rate measures the network, not the model. Over all 50 episodes the rate is 10.0%.

## Run

| field | value |
| --- | --- |
| run id | `qwen-base-judge2_online_mind2web_20260827T230942Z` |
| label | `qwen-base-judge2` |
| started (UTC) | 2026-08-27T23:09:42.440365+00:00 |
| finished (UTC) | 2026-08-27T23:55:39.948655+00:00 |
| endpoint | `https://<modal-workspace>--cua-serve-server.us-east.modal.direct/v1` |
| model | `cua-student` |
| adapter / checkpoint | `(none declared)` |
| task source | `online_mind2web` — 300 tasks across live third-party websites (Online-Mind2Web, COLM 2025) |
| dataset | `https://huggingface.co/datasets/osunlp/Online-Mind2Web` |
| dataset revision | `(n/a)` |
| coordinate space | `normalized_1000` |
| step cap | 3.0x reference_length, clamped to [15, 50] (ungraded tasks: 30) |
| abort rules | 5 unproductive steps / 3 parse failures |
| task subset | full set |
| temperature / max tokens | 0.0 / 2048 |
| observation format | png @ 1280x720 |
| judge | claude-sonnet-5 (final + 2 mid screenshots) |
| history window | 10 |
| driver loop guard | disabled |
| browser | local DockerSandbox on the invoking host |
| repo commit | `02526c20082e-dirty` |

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
| verified success (scored) | 14.7% |
| verified success (all episodes) | 10.0% |
| verified AND non-meandering | 11.8% |
| site-blocked | 28.0% |
| parse-failure rate (steps) | 20.6% |
| malformed action lines | 446 |
| canonical action form | 18.8% (53/282 actions) |
| form deviations | default_kwargs 200, bare_coords 105, missing_prefix 52, missing_parens 6, other 2 |
| done() emission rate | 22.0% |
| <think> emission rate (steps) | 86.1% |
| driver-refused action rate | 0.0% |
| emitted coords outside the viewport | 54/234 (23.1%) |
| out-of-viewport actions refused | 0 |
| episodes abandoned as stuck | 0 |
| mean steps on success | 10.0 |
| mean steps / reference_length on success | 1.92x |
| mean steps (all) | 6.88 |
| episodes that hit the step cap | 4 |

`verified` is the judge's verdict on the ORIGINAL goal after its own criteria checklist is enforced (`judged` is the raw claim); the efficiency grade is reported beside it rather than folded into it. `steps / reference_length` compares against the benchmark's own human step count, which is a more defensible efficiency denominator than our raw step counts.

`canonical action form` is the share of parsed actions spelled exactly as `format_action_for_training` emits them. The parser accepts any UNAMBIGUOUS spelling of a call (optional `action:`, bare or `<ref>`-wrapped coordinates, `wait` with or without `()`, omitted keyword arguments, single or double quotes, bare key names) and rejects genuinely ambiguous ones. A base model has only seen the grammar described in the system prompt, so charging surface deviation to its success rate would measure imitation of an unfamiliar convention rather than browsing. The deviation is reported here instead, and the tolerance is a harness parity key so both sides of a delta are parsed identically.

## Per level

| level | n | scored | verified | success rate | steps/ref | blocked |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 17 | 11 | 4 | 36.4% | 1.92x | 5 |
| medium | 16 | 10 | 0 | 0.0% | 0.0x | 6 |
| hard | 17 | 13 | 1 | 7.7% | 1.93x | 3 |

## Outcomes

| outcome | episodes |
| --- | --- |
| parse_failure_abort | 19 |
| site_blocked | 14 |
| completed | 11 |
| step_cap | 4 |
| unreachable | 2 |

## Per-domain

| domain | n | scored | verified | success rate | blocked |
| --- | --- | --- | --- | --- | --- |
| www.google.com | 3 | 3 | 2 | 66.7% | 0 |
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
| www.carnival.com | 1 | 1 | 0 | 0.0% | 0 |
| www.recreation.gov | 1 | 1 | 0 | 0.0% | 0 |
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
| `0170ca95038b05fa58d463fe627ac605` | hard | completed | no | acceptable | 7 | 36 | 12 | 1 | 0 | The agent reached the gov.uk Health and Care Worker visa overview page, which is the right resource, but the final scree |
| `0059adc6b12a3822305deb68929b2de8` | medium | parse_failure_abort | no | meandering | 6 | 24 | 8 | 3 | 0 | The final screenshot shows a proxy/firewall rejection error page ('The requested URL was rejected'), not a job listing s |
| `0632e496d37badee0350dad358f047c5` | easy | completed | no | acceptable | 4 | 15 | 4 | 0 | 0 | The agent typed the search query but it was truncated ('witho') and the site returned a 'no recipes found' page, so no a |
| `0b2623e9fa5cea997f76490bcbc5220f_070826` | hard | parse_failure_abort | no | meandering | 3 | 39 | 13 | 3 | 0 | The agent executed zero actions and the final screenshot shows a generic Petfinder/Nestle server error page, not any sea |
| `01abae9608f2d8752a83e08f136f720c` | medium | site_blocked | no | - | 0 | 18 | 6 | 0 | 0 | waf_block |
| `16200f51d63f0a47a58fa17acd49e368` | easy | completed | yes | acceptable | 5 | 15 | 3 | 0 | 0 | The final screenshot shows a recipe (茄子と椎茸と油揚げのお味噌汁) whose title and ingredient list include eggplant (茄子) and mushroom/ |
| `0b51b4fa0295ae80ccd176ebdad6fff6` | hard | unreachable | no | - | 0 | 39 | 13 | 0 | 0 | page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://www.carmax.com/ Call log:   - navigating to "https://www.carmax.com/", waiting until "domcontentloaded"  |
| `046138801a05ddf56ad94e8672942496_070826` | medium | step_cap | no | meandering | 18 | 18 | 6 | 5 | 0 | The final screenshot shows a Cloudflare 'Performing security verification' bot-check page on login.flightaware.com, not  |
| `180ed2ec377ef3a4af9035a21522091a` | easy | step_cap | no | meandering | 15 | 15 | 5 | 0 | 0 | All three screenshots show only a Cloudflare 'verifying you are human' security check page for umich.edu; the agent neve |
| `0b838cd54f826c59c71f600c56b89a11_070826` | hard | site_blocked | no | - | 0 | 33 | 11 | 0 | 0 | waf_block |
| `05483c50cc9b04c8ac44c574758fb2bd` | medium | site_blocked | no | - | 0 | 24 | 8 | 0 | 0 | waf_block |
| `1fc28d91d25ccd1c6ba268101326a654` | easy | completed | yes | acceptable | 5 | 15 | 3 | 0 | 0 | The final screenshot shows Google Finance's BTC/USD page with the 5D tab actively selected and the chart plotting the la |
| `0e5536aaad9d3462b06cf725e6ed535a` | hard | parse_failure_abort | no | meandering | 10 | 33 | 11 | 3 | 0 | The final screenshot shows a generic Mobile Passport Control page mentioning 'shorter wait times' in bullet text, not an |
| `070c907d34a4ce71dfdbea38f9c5d4d8` | medium | parse_failure_abort | no | meandering | 3 | 21 | 7 | 3 | 0 | No actions were executed and the final screenshot shows a Healthgrades geo-restriction error page ('not available in you |
| `20a460a8fe1971b84411c5b1e6ac4186` | easy | completed | no | meandering | 5 | 15 | 3 | 0 | 0 | The final screenshot shows location still set to Lviv (not Las Vegas) despite the agent typing and pressing Enter, and w |
| `1223b07536a87e0170ff87cbbebd1d3c` | hard | parse_failure_abort | no | meandering | 19 | 50 | 24 | 5 | 0 | The final screenshot shows the initial quiz configuration screen still set to default options (0-10, 10 questions, 30 se |
| `07bdc595306729a028ba06cc7451a80a` | medium | completed | no | acceptable | 8 | 21 | 7 | 1 | 0 | The final screenshot only shows the high-speed train search results list for Shanghai to Beijing with several options be |
| `2207bb4f21786690cfed20b37253fb8b` | easy | completed | yes | efficient | 4 | 15 | 2 | 0 | 0 | The final screenshot shows The Weather Network page for Calgary, AB with Today's Conditions listing Wind 28 km/h S, sati |
| `199be0b54a436daee74247971fc684ee_070826` | hard | site_blocked | no | - | 0 | 50 | 20 | 0 | 0 | waf_block |
| `07ec4a12cba8090e2dc524d558ac7675` | medium | site_blocked | no | - | 0 | 18 | 6 | 0 | 0 | waf_block |
| `23204728192da9f73197a613d9681c18` | easy | site_blocked | no | - | 0 | 15 | 3 | 0 | 0 | waf_block |
| `1ab384fb3a791edfb410213cc6b82151_110325` | hard | parse_failure_abort | no | meandering | 13 | 33 | 11 | 6 | 0 | The final screenshot shows the Build a Nucleus decay screen reset to 0 protons and 0 neutrons with no symbol set, indica |
| `0a0fa834ce41b5297c6474293383759d_110325` | medium | parse_failure_abort | no | meandering | 5 | 24 | 8 | 3 | 0 | The final screenshot only shows the Carnival homepage with no search performed, no filters applied, and no cruise or onb |
| `246d654fab7c31d9651007e39e75f74f` | easy | step_cap | no | meandering | 15 | 15 | 5 | 0 | 0 | The final screenshot shows the sort dropdown open with 'Sort by: Most Recent' still highlighted/selected and 'Sort by: M |
| `1b867afecf072cb877ebfa4069263746` | hard | completed | yes | meandering | 29 | 45 | 15 | 0 | 0 | The final screenshot shows a line chart titled with Illinois and Michigan selectors, plotting 'Percentage of female civi |
| `0a54069a0ef542e571d1fee7f39c93d5` | medium | parse_failure_abort | no | meandering | 3 | 18 | 6 | 3 | 0 | No actions were executed at all; the final screenshot shows only a CloudFront 403 error page blocking access, with no si |
| `2532fd402d3c741b79894e6ff2269f53` | easy | parse_failure_abort | no | meandering | 3 | 15 | 3 | 3 | 0 | The run produced zero actions and the final screenshot shows a Thumbtack geo-block error page, not any electrician searc |
| `1bc154377120ec15b18dbabdba49c741_051526` | hard | completed | no | acceptable | 9 | 50 | 24 | 0 | 0 | The final screenshot shows StubHub's own bot-detection block page ('Access temporarily restricted'), meaning the site cu |
| `0e42c3a73f2aece1f854e0ba55b7c8b0` | medium | parse_failure_abort | no | meandering | 13 | 21 | 7 | 3 | 0 | The final screenshot shows 'Gas star gas station' located in Woodland Park, NJ (not Manhattan, NY), and there is no evid |
| `26810ed9c123a62992e3eed31db3c5ee` | easy | unreachable | no | - | 0 | 15 | 4 | 0 | 0 | page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://www.accuweather.com/ Call log:   - navigating to "https://www.accuweather.com/", waiting until "domcontentloaded"  |
| `1c3b747ae12ccee895745f82e3f2ef8a` | hard | parse_failure_abort | no | meandering | 12 | 33 | 11 | 3 | 0 | The agent browsed the competitions list and ended up on the Titanic competition's Leaderboard tab, not the Code tab, and |
| `11857213ca01510f12813740afd59918_110325` | medium | parse_failure_abort | no | acceptable | 10 | 36 | 12 | 3 | 0 | The final screenshot shows a brown men's Birkenstock clog product page, but size is still 'Bitte wählen' (unselected) an |
| `26a0e5c21c145dd8448aa92f35bec5ea` | easy | completed | no | meandering | 1 | 15 | 3 | 0 | 0 | The agent took no real actions and the final screenshot shows a Healthgrades geo-restriction error page, not any optomet |
| `27fa3ac20745d3d35e89fae157f63069` | hard | parse_failure_abort | no | meandering | 19 | 33 | 11 | 5 | 0 | The final screenshot shows an Aeronautics and Astronautics course listing page (AA100, AA101), not chemistry courses, an |
| `11abb668c751dd56bb41f296a8bb3a13` | medium | site_blocked | no | - | 0 | 30 | 10 | 0 | 0 | waf_block |
| `271b36efd4346721b5542488ff997042` | easy | site_blocked | no | - | 0 | 15 | 4 | 0 | 0 | waf_block |
| `28e7574e7bd6d14f36d2988a5ef2bd23` | hard | parse_failure_abort | no | meandering | 8 | 36 | 12 | 3 | 0 | The final screenshot shows a broken 404 'Page not found' error on an unrelated OhioJobs site, indicating the agent navig |
| `123e8c2fc453f55fadd1d0b9aaf94df4` | medium | site_blocked | no | - | 0 | 24 | 8 | 0 | 0 | waf_block |
| `29526b17a32485742b5ab63507e99417` | easy | site_blocked | no | - | 0 | 15 | 3 | 0 | 0 | waf_block |
| `33bd2cdcea4fcc42a09a8a1e4e5841c6_110325` | hard | parse_failure_abort | no | meandering | 16 | 50 | 20 | 3 | 0 | All three screenshots show the same Raising Cane's site error page ('temporarily unavailable'), meaning the site never l |
| `157f4a79d55e8fa3fd55ba772ba40fbc_070826` | medium | site_blocked | no | - | 0 | 27 | 9 | 0 | 0 | waf_block |
| `29b7372d5a3884a2ba831af2d117af3c` | easy | completed | yes | acceptable | 7 | 15 | 3 | 0 | 0 | Agent searched Microsoft, landed on the MSFT:NASDAQ Google Finance page, scrolled to the News stories section, and click |
| `3443e9c3151fef19a3c3a45eb2c13640_070826` | hard | step_cap | no | meandering | 36 | 36 | 12 | 0 | 0 | The final screenshot is identical to the mid-run screenshot from step 25, showing the calculator page still on 'August 2 |
| `15be05973fba714e490cd9c884e4f072` | medium | parse_failure_abort | no | meandering | 16 | 18 | 6 | 3 | 0 | The final screenshot shows a 404 'Page not found' error on OhioGov, meaning the agent navigated to a broken link and nev |
| `34992feb69eb8e788faa06868b365c49` | easy | site_blocked | no | - | 0 | 15 | 5 | 0 | 0 | waf_block |
| `367d843c640637745e8fafa741cca13b` | hard | site_blocked | no | - | 0 | 45 | 15 | 0 | 0 | waf_block |
| `1aeca99e6a60b0e3aefb3ef212bdce79` | medium | parse_failure_abort | no | meandering | 11 | 30 | 10 | 3 | 0 | The agent typed the query into a generic CA.gov site search box, which returned irrelevant PDF results (Trinity County,  |
| `34ccd15a8ea8fd3895af83f5ccf62369` | easy | parse_failure_abort | no | meandering | 3 | 15 | 3 | 3 | 0 | The run recorded zero actions and the final screenshot shows only a CloudFront 403 error page, meaning the site was neve |
| `3ef64f34eae59c9fac7ee9a4f18b4a0c` | hard | parse_failure_abort | no | meandering | 3 | 48 | 16 | 3 | 0 | The final screenshot shows the regular YouTube homepage (not YouTube Kids) with no video opened and zero recorded action |

---

Episodes are recorded under this run directory and are excluded from training by construction (outside `trajectories/`, and each carries `metadata.excluded_reason=eval_episode`).
