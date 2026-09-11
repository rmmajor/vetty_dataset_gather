import argparse
import json

from datasets import load_dataset

WEBSITE_URL_MAP = {
    "aa": "https://www.aa.com",
    "agoda": "https://www.agoda.com",
    "airbnb": "https://www.airbnb.com",
    "amazon": "https://www.amazon.com",
    "amctheatres": "https://www.amctheatres.com",
    "amtrak": "https://www.amtrak.com",
    "apple": "https://www.apple.com",
    "boardgamegeek": "https://www.boardgamegeek.com",
    "bookdepository": "https://www.bookdepository.com",
    "booking": "https://www.booking.com",
    "budget": "https://www.budget.com",
    "cabelas": "https://www.cabelas.com",
    "cargurus": "https://www.cargurus.com",
    "carmax": "https://www.carmax.com",
    "carnival": "https://www.carnival.com",
    "cvs": "https://www.cvs.com",
    "delta": "https://www.delta.com",
    "discogs": "https://www.discogs.com",
    "ebay": "https://www.ebay.com",
    "enterprise": "https://www.enterprise.com",
    "espn": "https://www.espn.com",
    "eventbrite": "https://www.eventbrite.com",
    "expedia": "https://www.expedia.com",
    "exploretock": "https://www.exploretock.com",
    "flightaware": "https://www.flightaware.com",
    "foxsports": "https://www.foxsports.com",
    "gamestop": "https://www.gamestop.com",
    "ign": "https://www.ign.com",
    "ikea": "https://www.ikea.com",
    "imdb": "https://www.imdb.com",
    "instacart": "https://www.instacart.com",
    "jetblue": "https://www.jetblue.com",
    "kayak": "https://www.kayak.com",
    "kbb": "https://www.kbb.com",
    "koa": "https://www.koa.com",
    "kohls": "https://www.kohls.com",
    "last.fm": "https://www.last.fm",
    "marriott": "https://www.marriott.com",
    "mbta": "https://www.mbta.com",
    "menards": "https://www.menards.com",
    "new.mta.info": "https://new.mta.info",
    "newegg": "https://www.newegg.com",
    "nfl": "https://www.nfl.com",
    "nps.gov": "https://www.nps.gov",
    "nyc": "https://www.nyc.gov",
    "parking": "https://www.parking.com",
    "qatarairways": "https://www.qatarairways.com",
    "redbox": "https://www.redbox.com",
    "rei": "https://www.rei.com",
    "rentalcars": "https://www.rentalcars.com",
    "resy": "https://www.resy.com",
    "rottentomatoes": "https://www.rottentomatoes.com",
    "ryanair": "https://www.ryanair.com",
    "seatgeek": "https://www.seatgeek.com",
    "sixflags": "https://www.sixflags.com",
    "soundcloud": "https://www.soundcloud.com",
    "sports.yahoo": "https://sports.yahoo.com",
    "spothero": "https://www.spothero.com",
    "store.steampowered": "https://store.steampowered.com",
    "target": "https://www.target.com",
    "tesla": "https://www.tesla.com",
    "thetrainline": "https://www.thetrainline.com",
    "ticketcenter": "https://www.ticketcenter.com",
    "travelzoo": "https://www.travelzoo.com",
    "tvguide": "https://www.tvguide.com",
    "ultimate-guitar": "https://www.ultimate-guitar.com",
    "underarmour": "https://www.underarmour.com",
    "uniqlo": "https://www.uniqlo.com",
    "united": "https://www.united.com",
    "us.megabus": "https://us.megabus.com",
    "viator": "https://www.viator.com",
    "yellowpages": "https://www.yellowpages.com",
    "yelp": "https://www.yelp.com",
}


def guess_url_from_website_name(website_name):
    cleaned = website_name.lower().replace(" ", "").replace("'", "").replace("&", "and")
    cleaned = cleaned.replace(".", "").replace("+", "plus")
    return f"https://www.{cleaned}.com"


def load_and_convert_tasks(output_path, limit=None):
    print("Loading Mind2Web dataset from HuggingFace...")
    dataset = load_dataset("osunlp/Mind2Web", split="train")
    print(f"Loaded {len(dataset)} entries from Mind2Web train split")

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        print(f"Limited to {len(dataset)} entries")

    tasks = []
    unmapped_websites = set()

    for idx, entry in enumerate(dataset):
        task_id = f"mind2web_{idx:04d}"
        website = entry["website"]
        instruction = entry["confirmed_task"]
        domain = entry["domain"]
        subdomain = entry.get("subdomain", "")
        annotation_id = entry.get("annotation_id", "")
        action_reprs = entry.get("action_reprs", [])

        if website in WEBSITE_URL_MAP:
            start_url = WEBSITE_URL_MAP[website]
        else:
            start_url = guess_url_from_website_name(website)
            unmapped_websites.add(website)

        task = {
            "task_id": task_id,
            "instruction": instruction,
            "start_url": start_url,
            "domain": domain,
            "subdomain": subdomain,
            "website": website,
            "annotation_id": annotation_id,
            "action_reprs": action_reprs,
        }
        tasks.append(task)

    with open(output_path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    print(f"Wrote {len(tasks)} tasks to {output_path}")

    if unmapped_websites:
        print(f"\nWARNING: {len(unmapped_websites)} websites had no URL mapping (guessed URL):")
        for w in sorted(unmapped_websites):
            guessed = guess_url_from_website_name(w)
            print(f"  {w} -> {guessed}")
        print("Review these URLs and add correct mappings to WEBSITE_URL_MAP if needed.")

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Load Mind2Web tasks and convert to recording format")
    parser.add_argument("--output", type=str, default="tasks.jsonl", help="Output JSONL file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks (for testing)")
    args = parser.parse_args()

    load_and_convert_tasks(args.output, args.limit)


if __name__ == "__main__":
    main()
