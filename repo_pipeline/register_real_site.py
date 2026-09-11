"""Autonomous REGISTER -> verify-email capture on a REAL public site, end-to-end in the browser.

Mints a disposable mail.tm mailbox, then lets the Claude teacher: create an account on a real site
using that address (typed masked), walk to the mail.tm web inbox, sign in, open the confirmation
message, and follow its link — all visually, recorded as one trajectory. This is the deployment
skill saucedemo's login could not prove.

The site account password is the mailbox password (one masked secret reused for both the site and
the inbox login). Real verification mail is also confirmed out-of-band via the mail.tm API purely
as a harness sanity check; the agent still reads it in the browser.

Run: python -m repo_pipeline.register_real_site <name> <signup_url> ["instruction"]
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from repo_pipeline import driver_session, mailtm
from repo_pipeline.agent_capture import run_teacher_task
from repo_pipeline.instrumentation import RESULTS_DIR
from repo_pipeline.public_capture import _verify_and_finalize
from repo_pipeline.registry import RepoRegistry
from repo_pipeline.sandbox import DockerSandbox, install_sigterm_handler
from repo_pipeline.stage6_generate import _is_valid_trajectory, _safe_task_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("register_real_site")

CONTAINER_TIMEOUT = 5400


async def _run(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


# Recorded instructions are TERSE USER GOALS — how a real person would prompt the agent. The agent
# works out the process (find the form, fill it, confirm email if the site requires it) from the
# resources handed to it via the teacher policy (it has credentials + an email inbox), NOT from
# step-by-step text in the instruction. The site name fills in so the goal reads naturally.
def _default_instruction(site_label: str) -> str:
    return f"Sign me up for a new account on {site_label}."


def _reset_instruction(site_label: str) -> str:
    return f"I forgot my password for {site_label}. Reset it and get me signed back in."


def _site_label(signup_url: str) -> str:
    host = signup_url.split("/")[2] if "://" in signup_url else signup_url
    return host[4:] if host.startswith("www.") else host


async def capture(name: str, signup_url: str, instruction: str) -> dict:
    mailbox = mailtm.create_mailbox()
    # One password for both the site account and the mail.tm inbox login -> a single masked secret.
    secrets = {"email": mailbox["address"], "password": mailbox["password"]}
    site = {"name": name, "start_url": signup_url, "domain": "public_web"}
    task = {"id": f"register-{name}", "instruction": instruction, "difficulty": "hard", "instruction_source": "curated"}
    site_dir = Path("trajectories") / f"site_{name}"

    registry = RepoRegistry("pipeline_data/registry.db")
    sandbox = DockerSandbox(container_timeout=CONTAINER_TIMEOUT)
    state = None
    try:
        await _run(sandbox.start, f"public/{name}")
        await _run(driver_session.install_driver, sandbox)
        await _run(driver_session.start_driver, sandbox)
        client = driver_session.DriverClient(sandbox)

        logger.info(f"[{name}] mailbox={mailbox['address']} — running register+verify teacher")
        state = await run_teacher_task(
            client,
            task,
            signup_url,
            registry=registry,
            full_name=f"site:{name}",
            autonomy="autonomous",
            auth_mode="login",
            secrets=secrets,
            webmail_url=mailtm.WEBMAIL_URL,
            webmail_preauthed=False,
        )
        logger.info(f"[{name}] teacher: {state['actions']} actions, outcome={state['agent_outcome']}")
        site_dir.mkdir(parents=True, exist_ok=True)
        await _run(sandbox.copy_out, RESULTS_DIR, str(site_dir))
    finally:
        await _run(sandbox.stop)

    # Out-of-band harness check: did the site actually deliver verification mail to the box?
    got_mail = mailtm.wait_for_message(mailbox["token"], timeout=10)
    logger.info(f"[{name}] verification mail in box (out-of-band): {got_mail.get('subject') if got_mail else 'NONE'}")

    traj_dir = site_dir / f"task_{_safe_task_id(task['id'])}"
    if not _is_valid_trajectory(traj_dir):
        logger.warning(f"[{name}] no valid trajectory recorded")
        registry.close()
        return {"name": name, "captured": False}
    verdict = await _verify_and_finalize(traj_dir, site, task, state, registry)
    registry.close()
    logger.info(
        f"[{name}] verified={verdict.get('final_success')} ({verdict.get('efficiency')}) "
        f"— {verdict.get('reasoning', '')[:160]}"
    )
    # Leak check: the masked password must never appear literally in the recording.
    blob = (traj_dir / "trajectory.json").read_text()
    leaked = mailbox["password"] in blob
    logger.info(f"[{name}] password leaked literally: {leaked}  (must be False)")
    return {
        "name": name,
        "captured": True,
        "verified": verdict.get("final_success"),
        "mail_received": bool(got_mail),
        "password_leaked": leaked,
    }


async def capture_register_then_reset(name: str, signup_url: str) -> dict:
    """Two trajectories on a real site sharing one disposable mailbox: first register+verify (mints
    the account), then a full forgot-password reset (request -> read reset mail in the browser -> set
    a new password -> sign in). Proves the password-reset flow the user called out, on a real site.

    Three masked secrets: email, password (mailbox/email-account login), new_password (the reset
    value). The instructions tell the agent which is which; the driver substitutes at type time."""
    mailbox = mailtm.create_mailbox()
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
    sandbox = DockerSandbox(container_timeout=CONTAINER_TIMEOUT)
    states: dict = {}
    try:
        await _run(sandbox.start, f"public/{name}")
        await _run(driver_session.install_driver, sandbox)
        await _run(driver_session.start_driver, sandbox)
        client = driver_session.DriverClient(sandbox)
        logger.info(f"[{name}] mailbox={mailbox['address']} — register then reset")
        for task in tasks:
            states[task["id"]] = await run_teacher_task(
                client,
                task,
                signup_url,
                registry=registry,
                full_name=f"site:{name}",
                autonomy="autonomous",
                auth_mode="login",
                secrets=secrets,
                webmail_url=mailtm.WEBMAIL_URL,
                webmail_preauthed=False,
            )
            logger.info(
                f"[{name}] task '{task['id']}': {states[task['id']]['actions']} actions, "
                f"outcome={states[task['id']]['agent_outcome']}"
            )
            site_dir.mkdir(parents=True, exist_ok=True)
            await _run(sandbox.copy_out, RESULTS_DIR, str(site_dir))
    finally:
        await _run(sandbox.stop)

    results = {}
    for task in tasks:
        traj_dir = site_dir / f"task_{_safe_task_id(task['id'])}"
        if not _is_valid_trajectory(traj_dir):
            results[task["id"]] = {"captured": False}
            continue
        verdict = await _verify_and_finalize(traj_dir, site, task, states[task["id"]], registry)
        leaked = any(v in (traj_dir / "trajectory.json").read_text() for v in secrets.values())
        results[task["id"]] = {"captured": True, "verified": verdict.get("final_success"), "password_leaked": leaked}
        logger.info(
            f"[{name}] {task['id']}: verified={verdict.get('final_success')} "
            f"({verdict.get('efficiency')}) leaked={leaked}"
        )
    registry.close()
    return {"name": name, "tasks": results}


async def main():
    install_sigterm_handler()
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        name = sys.argv[2] if len(sys.argv) > 2 else "arena"
        signup_url = sys.argv[3] if len(sys.argv) > 3 else "https://www.are.na/sign_up"
        r = await capture_register_then_reset(name, signup_url)
        print("RESULT:", json.dumps(r))
        return
    name = sys.argv[1] if len(sys.argv) > 1 else "mstdn-party"
    signup_url = sys.argv[2] if len(sys.argv) > 2 else "https://mstdn.party/auth/sign_up"
    instruction = sys.argv[3] if len(sys.argv) > 3 else _default_instruction(_site_label(signup_url))
    r = await capture(name, signup_url, instruction)
    print("RESULT:", json.dumps(r))


if __name__ == "__main__":
    asyncio.run(main())
