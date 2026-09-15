"""Render-probe candidate sites through the ACTUAL sandbox browser, before any teacher run.

`probe_capturable` fetches over HTTP from the host, which answers a different question: rome2rio
passes it yet serves the container a Cloudflare Turnstile challenge, and sbb.ch passes it yet
renders a blank page inside Chromium. Both facts cost 2-3 teacher runs each to discover. This
probe opens each start_url in a sandbox driver session, reads the snapshot and the affordance
counts, and reports the sites worth capturing — seconds per site, no LLM.

  python -m cua.pipeline.lanes.probe_sandbox cua/pipeline/sites/public_sites_search.json
  python -m cua.pipeline.lanes.probe_sandbox --urls https://a.test https://b.test
  python -m cua.pipeline.lanes.probe_sandbox <config> --mark   # write skip_capture into the config

Run it when capture is NOT running: each probe holds a sandbox, and total sandbox memory is the
host's binding constraint (see safe_parallel_sandboxes).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from cua.eval.student.blocks import detect_block
from cua.pipeline.driver_session import boot_driver_sync
from cua.pipeline.instrumentation import RESULTS_DIR
from cua.pipeline.sandbox import DockerSandbox
from cua.settings import CAPTURE

logger = logging.getLogger(__name__)

# A usable page shows at least this many interactive elements in the driver's snapshot.
MIN_ELEMENTS = 5


def probe_site(client, sandbox, name: str, start_url: str) -> dict:
    try:
        probe = client.session("render_probe", start_url)
    except Exception as e:
        return {"name": name, "ok": False, "reason": f"session failed: {str(e)[:120]}"}
    snapshot = probe.get("snapshot", "")
    url = probe.get("url", start_url)
    try:
        affordances = client.affordances()
    except Exception:
        affordances = {}
    finally:
        try:
            client.finish("probe", "")
            sandbox.exec_command(f"rm -rf {RESULTS_DIR}/task_render_probe")
        except Exception:
            pass

    blocked = detect_block(url, snapshot)
    elements = sum(v for v in affordances.values() if isinstance(v, int))
    if blocked:
        return {"name": name, "ok": False, "reason": f"blocked in sandbox: {blocked}", "elements": elements}
    if len(snapshot.strip()) < 40 or elements < MIN_ELEMENTS:
        return {
            "name": name,
            "ok": False,
            "reason": f"renders empty in sandbox ({elements} interactive elements)",
            "elements": elements,
        }
    return {"name": name, "ok": True, "reason": f"{elements} interactive elements", "elements": elements}


def run_probes(sites: list[tuple], image: str = "") -> list[dict]:
    results = []
    kwargs = {"container_timeout": CAPTURE.probe_container_timeout}
    if image:
        kwargs["image"] = image
    with DockerSandbox(**kwargs) as sandbox:
        sandbox.start("probe/render")
        client = boot_driver_sync(sandbox, {"DRIVER_LOOP_GUARD_REPEATS": "0"})
        for name, start_url in sites:
            result = probe_site(client, sandbox, name, start_url)
            results.append(result)
            logger.info(f"{'OK  ' if result['ok'] else 'DEAD'} {name:24s} {result['reason']}")
    return results


def mark_config(config_path: Path, results: list[dict]):
    config = json.loads(config_path.read_text())
    dead = {r["name"]: r["reason"] for r in results if not r["ok"]}
    marked = 0
    for site in config.get("sites", []):
        if site["name"] in dead and not site.get("skip_capture"):
            site["skip_capture"] = True
            site["skip_reason"] = f"sandbox render probe: {dead[site['name']]}"
            marked += 1
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    logger.info(f"marked {marked} sites skip_capture in {config_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, nargs="?", help="a public_sites*.json to probe")
    parser.add_argument("--urls", nargs="*", default=[], help="probe bare URLs instead")
    parser.add_argument("--mark", action="store_true", help="write skip_capture for dead sites back into the config")
    parser.add_argument("--sandbox-image", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.urls:
        sites = [(u.split("//")[-1].split("/")[0], u) for u in args.urls]
    elif args.config:
        config = json.loads(args.config.read_text())
        sites = [(s["name"], s["start_url"]) for s in config.get("sites", []) if not s.get("skip_capture")]
    else:
        parser.error("pass a config path or --urls")

    results = run_probes(sites, args.sandbox_image)
    alive = [r for r in results if r["ok"]]
    print(f"\n{len(alive)}/{len(results)} sites render in the sandbox")
    for r in results:
        if not r["ok"]:
            print(f"  DEAD {r['name']}: {r['reason']}")
    if args.mark and args.config:
        mark_config(args.config, results)
    sys.exit(0)


if __name__ == "__main__":
    main()
