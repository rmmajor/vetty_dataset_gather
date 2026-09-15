"""Register-then-verify-email capture on one REAL public site, done end-to-end in the browser as one trajectory."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from cua.pipeline import mailtm
from cua.pipeline.driver_session import boot_driver, run_blocking
from cua.pipeline.lanes.capture_lane import capture_task, finalize_tasks
from cua.pipeline.registry import RepoRegistry
from cua.pipeline.sandbox import DockerSandbox, install_sigterm_handler
from cua.settings import CAPTURE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("register_real_site")

DEFAULT_REGISTER_SITE = ("mstdn-party", "https://mstdn.party/auth/sign_up")
DEFAULT_RESET_SITE = ("arena", "https://www.are.na/sign_up")


def _default_instruction(site_label: str) -> str:
    """A terse user GOAL, the way a real person would prompt: the agent works the process out itself."""
    return f"Sign me up for a new account on {site_label}."


def _reset_instruction(site_label: str) -> str:
    """The reset goal, again as intent only — the resources reach the agent through the teacher policy."""
    return f"I forgot my password for {site_label}. Reset it and get me signed back in."


def _site_label(signup_url: str) -> str:
    host = signup_url.split("/")[2] if "://" in signup_url else signup_url
    return host[4:] if host.startswith("www.") else host


async def _run_tasks(name: str, signup_url: str, tasks: list[dict], secrets: dict, site_dir: Path, registry) -> dict:
    """Run every task on one real site through a single sandbox, sharing one disposable mailbox."""
    sandbox = DockerSandbox(container_timeout=CAPTURE.register_container_timeout)
    states: dict = {}
    try:
        await run_blocking(sandbox.start, f"public/{name}")
        client = await boot_driver(sandbox)
        for task in tasks:
            states[task["id"]] = await capture_task(
                client,
                task,
                signup_url,
                sandbox=sandbox,
                site_dir=site_dir,
                label=name,
                registry=registry,
                full_name=f"site:{name}",
                autonomy="autonomous",
                auth_mode="login",
                secrets=secrets,
                webmail_url=mailtm.WEBMAIL_URL,
                webmail_preauthed=False,
            )
    finally:
        await run_blocking(sandbox.stop)
    return states


async def capture(name: str, signup_url: str, instruction: str) -> dict:
    """One register+verify trajectory on a real site, plus the out-of-band mail and leak checks."""
    mailbox = mailtm.create_mailbox()
    # One password for both the site account and the mail.tm inbox login -> a single masked secret.
    secrets = {"email": mailbox["address"], "password": mailbox["password"]}
    site = {"name": name, "start_url": signup_url, "domain": "public_web"}
    task = {"id": f"register-{name}", "instruction": instruction, "difficulty": "hard", "instruction_source": "curated"}
    site_dir = Path("trajectories") / f"site_{name}"

    registry = RepoRegistry("pipeline_data/registry.db")
    logger.info(f"[{name}] mailbox={mailbox['address']} — running register+verify teacher")
    states = await _run_tasks(name, signup_url, [task], secrets, site_dir, registry)

    # Out-of-band harness check: did the site actually deliver verification mail to the box?
    got_mail = mailtm.wait_for_message(mailbox["token"], timeout=10)
    logger.info(f"[{name}] verification mail in box (out-of-band): {got_mail.get('subject') if got_mail else 'NONE'}")

    results = await finalize_tasks([task], site, site_dir, states, registry, secrets=secrets, require_valid=True)
    registry.close()
    if task["id"] not in results:
        logger.warning(f"[{name}] no valid trajectory recorded")
        return {"name": name, "captured": False}
    verdict = results[task["id"]]["verdict"]
    logger.info(f"[{name}] verifier reasoning: {verdict.get('reasoning', '')[:160]}")
    return {
        "name": name,
        "captured": True,
        "verified": verdict.get("final_success"),
        "mail_received": bool(got_mail),
        "password_leaked": results[task["id"]]["leaked"],
    }


async def capture_register_then_reset(name: str, signup_url: str) -> dict:
    """Register+verify then a full forgot-password reset on one real site, sharing one disposable mailbox."""
    mailbox = mailtm.create_mailbox()
    # Three masked secrets; the instructions say which is which and the driver substitutes at type time.
    secrets = {"email": mailbox["address"], "password": mailbox["password"], "new_password": mailtm._random_password()}
    site = {"name": name, "start_url": signup_url, "domain": "public_web"}
    label = _site_label(signup_url)
    tasks = [
        {
            "id": f"register-{name}",
            "instruction": _default_instruction(label),
            "difficulty": "hard",
            "instruction_source": "curated",
        },
        {
            "id": f"reset-{name}",
            "instruction": _reset_instruction(label),
            "difficulty": "hard",
            "instruction_source": "curated",
        },
    ]
    site_dir = Path("trajectories") / f"site_{name}"
    registry = RepoRegistry("pipeline_data/registry.db")
    logger.info(f"[{name}] mailbox={mailbox['address']} — register then reset")
    states = await _run_tasks(name, signup_url, tasks, secrets, site_dir, registry)

    finalized = await finalize_tasks(tasks, site, site_dir, states, registry, secrets=secrets, require_valid=True)
    registry.close()
    results = {}
    for task in tasks:
        outcome = finalized.get(task["id"])
        if outcome is None:
            results[task["id"]] = {"captured": False}
            continue
        results[task["id"]] = {
            "captured": True,
            "verified": outcome["verdict"].get("final_success"),
            "password_leaked": outcome["leaked"],
        }
    return {"name": name, "tasks": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous register / password-reset capture on a real public site")
    parser.add_argument("name", nargs="?", default=None, help="Site name for trajectories/site_<name>")
    parser.add_argument("signup_url", nargs="?", default=None, help="The signup page URL")
    parser.add_argument("instruction", nargs="?", default=None, help="Override the generated user goal (register only)")
    parser.add_argument(
        "--reset", action="store_true", help="Register + verify, then a full forgot-password reset (also: `reset ...`)"
    )
    return parser


def normalise_argv(argv: list[str]) -> list[str]:
    """Accept the historical `reset <name> <url>` verb as well as `--reset`, so one parser serves both."""
    return ["--reset", *argv[1:]] if argv[:1] == ["reset"] else argv


def resolve_args(argv: list[str]):
    """The parsed arguments with each lane's own positional defaults filled in."""
    args = build_parser().parse_args(normalise_argv(argv))
    fallback = DEFAULT_RESET_SITE if args.reset else DEFAULT_REGISTER_SITE
    args.name = args.name or fallback[0]
    args.signup_url = args.signup_url or fallback[1]
    return args


async def main():
    args = resolve_args(sys.argv[1:])
    install_sigterm_handler()
    if args.reset:
        result = await capture_register_then_reset(args.name, args.signup_url)
    else:
        instruction = args.instruction or _default_instruction(_site_label(args.signup_url))
        result = await capture(args.name, args.signup_url, instruction)
    print("RESULT:", json.dumps(result))


if __name__ == "__main__":
    asyncio.run(main())
