# Ukrainian web coverage

`cua/pipeline/sites/public_sites_actions_ua.json` — a curated registry of Ukrainian websites and
Ukrainian-language user tasks for computer-use-agent trajectory capture.

It exists to back a specific novelty claim: **no open CUA dataset has meaningful Ukrainian web
coverage.** Mind2Web, WebArena, WebVoyager, GAIA and WebLINX are built on English-language (and
for WebArena, synthetic) sites; none of them carries `.ua` hosts, Ukrainian-language instructions,
or the interaction conventions of the Ukrainian market (Nova Poshta branch pickup, «Власний
Рахунок» loyalty pricing, phone-first registration, OLX haggling by message).

The file uses the same schema as `public_sites_actions_logingated.json` and
`public_sites_actions_indie.json`, so it loads through `cua.pipeline.sites.registry.load_sites` —
the one loader every capture lane shares — with no code changes.

## Headline numbers

| | |
|---|---|
| Sites | **87** |
| Tasks | **411** |
| Categories | 15 groups / 28 category tags |
| Tasks completable anonymously | 330 |
| Tasks requiring an account | 81 |
| Difficulty | 21 easy / 250 medium / 140 hard |
| Reachable headless (autonomous lane) | 68 |
| Anti-bot walled headless (skip-tagged) | 19 |

Every instruction is a terse Ukrainian user **goal**, never a procedure — «набрати продуктів для
борщу в Сільпо», not «натисни пошук, введи буряк». The agent works out the route.

## Coverage by category

| Category | Sites | Tasks | Login-gated tasks |
|---|---|---|---|
| Marketplaces & general retail | 5 | 39 | 10 |
| Electronics, home & DIY | 5 | 27 | 1 |
| Fashion, beauty & sport | 7 | 33 | 2 |
| Grocery & food delivery | 11 | 53 | 2 |
| Logistics & parcels | 3 | 18 | 4 |
| Classifieds (C2C, auto, realty) | 6 | 40 | 20 |
| Jobs & services | 4 | 21 | 7 |
| Travel, transport & events | 10 | 44 | 6 |
| Pharmacy & healthcare | 6 | 28 | 4 |
| Books & publishing | 3 | 14 | 2 |
| Education | 6 | 23 | 11 |
| Government services & open data | 6 | 20 | 4 |
| Telecom & mobility | 5 | 18 | 2 |
| Ukrainian SaaS & web infrastructure | 7 | 24 | 5 |
| Portals & weather | 3 | 9 | 1 |
| **Total** | **87** | **411** | **81** |

## Login posture

`provisioning` classifies how hard it is to obtain an account. It is a curated hint, not a probed
fact — the probe measured reachability and anti-bot posture only.

| provisioning | Sites | Meaning for capture |
|---|---|---|
| `email_verify` | 46 | reachable by the autonomous lane's disposable-inbox self-registration |
| `phone` | 41 | needs a Ukrainian SMS number — both capture lanes skip these for signup; their `auth_required: false` tasks still capture anonymously |

81 of 411 tasks are marked `auth_required`. The other
330 finish without an account — searching, filtering, comparing, configuring a cart,
reaching a checkout or booking form — which is what keeps the majority of this registry capturable
today on `phone`-gated sites.

## Probe results

Every host was probed live on 2026-08-26, twice: once over HTTP with a browser-like User-Agent
(reusing the WAF/challenge signatures in `cua/pipeline/lanes/probe_capturable.py`), then again in the
Playwright Chromium the capture lane actually drives. The browser probe is the one that counts —
several hosts that answered plain `curl` happily still refuse a real automated browser, and one
(`decathlon.ua`) did the opposite.

**68 of 87 sites render for headless Chromium** and are swept by default.

**19 sites sit behind an anti-bot wall for headless Chromium.** They are kept in the registry
with `skip_capture: true` and a `skip_reason`, so the default sweep passes them over but
`--sites <name>` still routes them to the human-provisioned lane. A single headed re-probe showed
15 of them render normally in a real browser window:

| Site | Signature | HTTP | Real browser window |
|---|---|---|---|
| aromateque.com.ua | `cf_chl_opt` | 403 | renders (662 elements) |
| www.atbmarket.com | `cf_chl_opt` | 403 | renders (521 elements) |
| www.blablacar.com.ua | `captcha-delivery.com` | 200 | not retested |
| book-ye.com.ua | `cf_chl_opt` | 403 | renders (3067 elements) |
| busfor.ua | `captcha-delivery.com` | 403 | not retested |
| comfy.ua | `request unsuccessful. incapsula` | 200 | not retested |
| eva.ua | `cf_chl_opt` | 403 | renders (2158 elements) |
| intertop.ua | `cf_chl_opt` | 403 | renders (92 elements) |
| ua.jooble.org | `cf_chl_opt` | 403 | renders (182 elements) |
| kasta.ua | `cf_chl_opt` | 403 | renders (509 elements) |
| liki24.com | `attention required! \| cloudflare` | 403 | renders (1766 elements) |
| lun.ua | `cf_chl_opt` | 403 | not retested |
| opendatabot.ua | `cf_chl_opt` | 403 | renders (102 elements) |
| rozetka.com.ua | `cf_chl_opt` | 403 | renders (491 elements) |
| silpo.ua | `cf_chl_opt` | 403 | renders (1307 elements) |
| tabletki.ua | `attention required! \| cloudflare` | 403 | renders (234 elements) |
| booking.uz.gov.ua | `cf_chl_opt` | 403 | renders (89 elements) |
| www.watsons.ua | `you don't have permission to access` | 403 | renders (751 elements) |
| www.yakaboo.ua | `attention required! \| cloudflare` | 403 | renders (671 elements) |

The practical read: the Ukrainian web is unusually Cloudflare-heavy at the top of the market, and
the flagship retailers (Rozetka, Silpo, ATB, EVA, Watsons, Kasta, Yakaboo) are reachable for
trajectory capture **only** through the human-provisioned lane. That is a capture-cost finding, not
a coverage gap — the sites and their tasks are in the registry either way.

## Probed and rejected

Dropped from the registry entirely, with what the probe returned:

| Host | Category | Why dropped |
|---|---|---|
| eldorado.com.ua | electronics | connection timed out after 30 s (curl and Chromium alike) |
| rocket.ua | food delivery | ERR_CONNECTION_REFUSED on :443 |
| meest.ua | logistics | DNS does not resolve; successor meest-group.com renders an empty SPA shell |
| e-data.gov.ua | open data | DNS does not resolve; successor spending.gov.ua answers Akamai `Access Denied` |
| svitmebliv.ua | furniture | DNS does not resolve |
| vsi.ua | classifieds | DNS does not resolve |
| ua.farmasi.online | cosmetics | DNS does not resolve |
| aviakasa.com | air tickets | domain parked and listed for sale |
| ek.ua (E-Katalog) | price comparison | HTTP 200 but `E-Katalog — Under Maintenance`, 1 interactive element |
| citysites.ua | city portal | TLS failure — unable to get local issuer certificate |
| answear.ua | fashion | HTTP 403 `ANSWEAR - Site Maintenance` |
| rst.ua | auto classifieds | HTTP 403, 7 interactive elements |
| uz.gov.ua | railway | HTTP 500 `The URL you requested has been blocked` (booking.uz.gov.ua used instead) |
| preply.com/uk/ | tutoring | HTTP 404 on the Ukrainian path |
| tracking.novaposhta.ua | logistics | renders only 10 interactive elements (novaposhta.ua used instead) |
| brocard.ua | perfumery | Cloudflare challenge in BOTH headless and a real browser window |
| mon.gov.ua | government | Cloudflare challenge in BOTH headless and a real browser window |
| robota.ua | jobs | Cloudflare challenge in BOTH headless and a real browser window |

## Excluded by policy

| Excluded | Reason |
|---|---|
| privat24.ua, monobank.ua, oschadbank.ua and every other bank | never probed — the project's safety spec refuses the banking category outright |
| easypay.ua | probed live (HTTP 200) but dropped: a payments aggregator sits inside the excluded financial category |
| minfin.com.ua | financial portal — same exclusion |

No banking or financial-institution host appears in the registry; a check for bank-domain
substrings over every `start_url` returns empty.

## Excluded by curation

Reachable, but deliberately left out:

| Excluded | Reason |
|---|---|
| pravda.com.ua, tsn.ua, liga.net, nv.ua, suspilne regional sites, chytomo.com, dou.ua | reachable, but read-only news browsing; this corpus curates action/task-completion UIs. Only ukr.net, suspilne.media and sinoptik.ua survive, and only with genuinely action-shaped tasks (personalise a feed, switch region, add a city) |
| metro.zakaz.ua, auchan.zakaz.ua | same zakaz.ua storefront engine as novus.zakaz.ua and zakaz.ua, both already in the registry — duplicate UI, no new interaction shape |
| bigl.ua | Prom's own marketplace front end — duplicate of prom.ua |
| citrus.ua, podorozhnyk.ua, rieltor.ua, nashformat.ua, hotels24.ua, tickets.ua, poizdato.net, prozorro.sale, kmu.gov.ua, e-dem.ua, ecozagroza.gov.ua, dream.gov.ua, zno.osvita.ua, ukraine.com.ua | reachable, but each duplicates an interaction shape already covered or offers too thin a task surface to be worth authoring depth on |

## Reproducing the probe

```bash
python -m cua.pipeline.lanes.probe_capturable --config cua/pipeline/sites/public_sites_actions_ua.json \
    --provisioning email_verify,phone --dry-run
```

That reruns the HTTP-level check against this registry and reports which hosts have since changed
posture. The headless-browser verdicts recorded in each site's `probe` block came from driving the
same Playwright Chromium the capture lane uses, at 1280x720 with a `uk-UA` locale.
