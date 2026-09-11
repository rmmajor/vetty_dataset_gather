# Student eval — comparison

| label | verified success | all-episode success | parse-failure | canonical form | done() rate | mean steps (success) | steps/ref | site-blocked | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_qwen35_9b` | 8.8% | 6.0% | 22.3% | 26.8% | 28.0% | 5.67 | 1.36x | 28.0% | 50 |
| `after` | 29.4% | 20.0% | 7.4% | 100.0% | 18.0% | 17.7 | 2.14x | 28.0% | 50 |

## By level

| label | easy | medium | hard |
| --- | --- | --- | --- |
| `baseline_qwen35_9b` | 18.2% (2/11) | 0.0% (0/10) | 7.7% (1/13) |
| `after` | 36.4% (4/11) | 30.0% (3/10) | 23.1% (3/13) |

## Delta vs `baseline_qwen35_9b`

| label | verified success | parse-failure | canonical form | done() rate | easy success | medium success | hard success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `after` | +20.6 pp | -15.0 pp | +73.2 pp | -10.0 pp | +18.2 pp | +30.0 pp | +15.4 pp |

## Harness parity

All compared runs share the same task subset, step-cap and abort policy, prompt settings and judge configuration.

Expected model-side differences (not parity violations — each model is measured in its own convention):

- `coord_space`: `baseline_qwen35_9b`='normalized_1000', `after`='viewport_abs'

Repo commits: {"baseline_qwen35_9b": "89603f852203", "after": "3b340b5c1634-dirty"}

## Per-domain success rate

| domain | `baseline_qwen35_9b` | `after` |
| --- | --- | --- |
| careers.walmart.com | 0% (0/1) | 0% (0/1) |
| cookpad.com | 0% (0/2) | 50% (1/2) |
| ohio.gov | 0% (0/1) | 0% (0/1) |
| ohiomeansjobs.ohio.gov | 0% (0/1) | 0% (0/1) |
| phet.colorado.edu | 0% (0/1) | 0% (0/1) |
| raisingcanes.com | 0% (0/1) | 0% (0/1) |
| umich.edu | 0% (0/1) | 0% (0/1) |
| us.megabus.com | 0% (0/1) | 0% (0/1) |
| us.trip.com | 0% (0/1) | 100% (1/1) |
| www.accuweather.com | 0% (0/0) | 0% (0/0) |
| www.adoptapet.com | 0% (0/1) | 0% (0/1) |
| www.americashealthrankings.org | 0% (0/1) | 100% (1/1) |
| www.apartments.com | 0% (0/0) | 0% (0/0) |
| www.bbb.org | 0% (0/0) | 0% (0/0) |
| www.bestbuy.com | 0% (0/0) | 0% (0/0) |
| www.birkenstock.com | 0% (0/1) | 0% (0/1) |
| www.ca.gov | 0% (0/1) | 100% (1/1) |
| www.carmax.com | 0% (0/0) | 0% (0/0) |
| www.carnival.com | 0% (0/1) | 100% (1/1) |
| www.cboe.com | 0% (0/0) | 0% (0/0) |
| www.cbp.gov | 0% (0/1) | 0% (0/1) |
| www.coolmath4kids.com | 0% (0/1) | 0% (0/1) |
| www.dillards.com | 0% (0/0) | 0% (0/0) |
| www.disney.com | 0% (0/0) | 0% (0/0) |
| www.dmv.virginia.gov | 0% (0/0) | 0% (0/0) |
| www.drugs.com | 0% (0/0) | 0% (0/0) |
| www.flightaware.com | 0% (0/1) | 0% (0/1) |
| www.google.com | 33% (1/3) | 33% (1/3) |
| www.gov.uk | 100% (1/1) | 100% (1/1) |
| www.healthgrades.com | 0% (0/2) | 0% (0/2) |
| www.kaggle.com | 0% (0/1) | 0% (0/1) |
| www.kbb.com | 0% (0/0) | 0% (0/0) |
| www.mayoclinic.org | 0% (0/0) | 0% (0/0) |
| www.petfinder.com | 0% (0/1) | 0% (0/1) |
| www.qatarairways.com | 0% (0/0) | 0% (0/0) |
| www.recreation.gov | 0% (0/1) | 100% (1/1) |
| www.stanford.edu | 0% (0/1) | 0% (0/1) |
| www.stubhub.com | 0% (0/2) | 0% (0/2) |
| www.theweathernetwork.com | 100% (1/1) | 100% (1/1) |
| www.thumbtack.com | 0% (0/1) | 0% (0/1) |
| www.webmd.com | 0% (0/1) | 100% (1/1) |
| www.youtube.com | 0% (0/1) | 0% (0/1) |


wrote <tmp>/comparison.md
