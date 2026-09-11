"""Shared, isolated webmail for end-to-end auth-flow capture (registration, email
verification, password reset, email-OTP login).

The agent does these flows entirely through the browser, as one screenshot trajectory: it
registers on the booted app, then NAVIGATES to a real webmail UI, opens the verification /
reset mail, clicks the link or reads the code, and returns to the app. To make that possible
in isolation we run ONE mail stack per pipeline host, reused by every sandbox:

  * GreenMail  — an SMTP sink + IMAP server that AUTO-CREATES a mailbox for any address on
    first delivery/login (auth disabled). Each capture scenario gets a unique base address
    `qa-<run>-<scenario>@mail.test`; GreenMail isolates them with zero provisioning. NOTE:
    GreenMail strips `+tag` plus-addressing (pools it into the base mailbox), so scenario
    identity must live in the base local-part — never a `+tag`.
  * Roundcube  — a realistic webmail UI (so the inbox the student learns transfers to real
    Gmail/Outlook at deployment), pointed at GreenMail over IMAP/SMTP.

Both live on a user-defined Docker network (`vetty-mail`); each sandbox attaches to it so the
booted app can SMTP to `greenmail:3025` and the in-container browser can reach the webmail at
`http://mail.test/`. One shared stack (not per-sandbox) keeps memory flat — important given the
Docker-VM ceiling that has wedged parallel capture before.

Entry points: `ensure_mail_stack()` (idempotent bring-up), `mail_env_for_app()` (the SMTP env
to inject when booting an app), `scenario_address()` (mint an isolated mailbox address),
`teardown_mail_stack()`. Run `python -m repo_pipeline.mail_service` to smoke it standalone.
"""

import hashlib
import logging
import re
import socket
import time
import urllib.request
from dataclasses import dataclass, field

import docker

logger = logging.getLogger(__name__)

MAIL_NETWORK = "vetty-mail"
WEBMAIL_HOST = "mail.test"  # network alias the in-sandbox browser navigates to
MAIL_SERVER_ALIAS = "greenmail"  # network alias the app's SMTP + Roundcube target

GREENMAIL_IMAGE = "greenmail/standalone:2.1.0"
ROUNDCUBE_IMAGE = "roundcube/roundcubemail:latest"
GREENMAIL_CONTAINER = "vetty-greenmail"
ROUNDCUBE_CONTAINER = "vetty-roundcube"

SMTP_PORT = 3025  # in-network ports (GreenMail "test.all" defaults)
IMAP_PORT = 3143
WEBMAIL_PORT = 80

# Host-published ports for readiness polling + host-side mail probes (the in-sandbox path uses
# the network DNS above, not these). Only one shared stack per host, so fixed ports are safe.
HOST_SMTP_PORT = int(__import__("os").environ.get("MAIL_HOST_SMTP_PORT", "13025"))
HOST_IMAP_PORT = int(__import__("os").environ.get("MAIL_HOST_IMAP_PORT", "13143"))
HOST_WEBMAIL_PORT = int(__import__("os").environ.get("MAIL_HOST_WEBMAIL_PORT", "18080"))

GREENMAIL_OPTS = (
    "-Dgreenmail.setup.test.all -Dgreenmail.auth.disabled -Dgreenmail.users.login=email -Dgreenmail.hostname=0.0.0.0"
)
READY_TIMEOUT = 90

# SMTP env injected when booting an app, keyed broadly so the common frameworks pick it up
# without per-app special-casing (Django EMAIL_*, Laravel MAIL_*, Symfony MAILER_DSN, Node
# SMTP_*/MAIL_*). GreenMail accepts any credentials (auth disabled) over plaintext on :3025.
_SMTP_ENV_TEMPLATE = {
    "SMTP_HOST": "{host}",
    "SMTP_PORT": "{port}",
    "SMTP_USER": "qa",
    "SMTP_PASSWORD": "qa",
    "SMTP_PASS": "qa",
    "SMTP_SECURE": "false",
    "SMTP_TLS": "false",
    "MAIL_HOST": "{host}",
    "MAIL_PORT": "{port}",
    "MAIL_MAILER": "smtp",
    "MAIL_USERNAME": "qa",
    "MAIL_PASSWORD": "qa",
    "MAIL_ENCRYPTION": "",
    "EMAIL_HOST": "{host}",
    "EMAIL_PORT": "{port}",
    "EMAIL_HOST_USER": "qa",
    "EMAIL_HOST_PASSWORD": "qa",
    "EMAIL_USE_TLS": "false",
    "EMAIL_USE_SSL": "false",
    "MAILER_DSN": "smtp://{host}:{port}",
}


@dataclass(frozen=True)
class MailStack:
    network: str = MAIL_NETWORK
    webmail_url: str = f"http://{WEBMAIL_HOST}/"
    smtp_host: str = MAIL_SERVER_ALIAS
    smtp_port: int = SMTP_PORT
    imap_host: str = MAIL_SERVER_ALIAS
    imap_port: int = IMAP_PORT
    host_smtp_port: int = HOST_SMTP_PORT
    host_imap_port: int = HOST_IMAP_PORT
    host_webmail_port: int = HOST_WEBMAIL_PORT
    aliases: tuple = field(default=(WEBMAIL_HOST, MAIL_SERVER_ALIAS))


def _ensure_network(client) -> None:
    try:
        client.networks.get(MAIL_NETWORK)
    except docker.errors.NotFound:
        client.networks.create(MAIL_NETWORK, driver="bridge")
        logger.info(f"created docker network {MAIL_NETWORK}")


def _connect_alias(client, container, aliases) -> None:
    """Attach `container` to MAIL_NETWORK with the given DNS aliases (idempotent)."""
    net = client.networks.get(MAIL_NETWORK)
    net.reload()
    attached = {c["Name"] for c in (net.attrs.get("Containers") or {}).values()}
    if container.name in attached:
        return
    net.connect(container, aliases=list(aliases))


def _ensure_container(client, name, image, *, ports, environment, command=None) -> "docker.models.containers.Container":
    """Get-or-run a stably-named container. Reuses a running one, starts a stopped one, creates
    if absent — so ensure_mail_stack() is idempotent across pipeline runs."""
    try:
        c = client.containers.get(name)
        if c.status != "running":
            c.start()
        return c
    except docker.errors.NotFound:
        pass
    return client.containers.run(
        image,
        name=name,
        detach=True,
        ports=ports,
        environment=environment,
        command=command,
        labels={"vetty-mail": "1"},
        restart_policy={"Name": "unless-stopped"},
    )


def _smtp_banner_ok(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=4) as s:
            s.settimeout(4)
            return s.recv(64).startswith(b"220")
    except OSError:
        return False


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_mail_stack(client=None, wait: bool = True) -> MailStack:
    """Bring up (or reuse) the shared GreenMail + Roundcube stack and return its connection info."""
    client = client or docker.from_env()
    _ensure_network(client)

    greenmail = _ensure_container(
        client,
        GREENMAIL_CONTAINER,
        GREENMAIL_IMAGE,
        ports={f"{SMTP_PORT}/tcp": HOST_SMTP_PORT, f"{IMAP_PORT}/tcp": HOST_IMAP_PORT},
        environment={"GREENMAIL_OPTS": GREENMAIL_OPTS},
    )
    _connect_alias(client, greenmail, (MAIL_SERVER_ALIAS,))

    roundcube = _ensure_container(
        client,
        ROUNDCUBE_CONTAINER,
        ROUNDCUBE_IMAGE,
        ports={f"{WEBMAIL_PORT}/tcp": HOST_WEBMAIL_PORT},
        environment={
            "ROUNDCUBEMAIL_DEFAULT_HOST": MAIL_SERVER_ALIAS,
            "ROUNDCUBEMAIL_DEFAULT_PORT": str(IMAP_PORT),
            "ROUNDCUBEMAIL_SMTP_SERVER": MAIL_SERVER_ALIAS,
            "ROUNDCUBEMAIL_SMTP_PORT": str(SMTP_PORT),
            "ROUNDCUBEMAIL_DB_TYPE": "sqlite",
        },
    )
    _connect_alias(client, roundcube, (WEBMAIL_HOST,))

    if wait:
        deadline = time.time() + READY_TIMEOUT
        smtp_ready = web_ready = False
        while time.time() < deadline:
            smtp_ready = smtp_ready or _smtp_banner_ok("127.0.0.1", HOST_SMTP_PORT)
            web_ready = web_ready or _http_ok(f"http://127.0.0.1:{HOST_WEBMAIL_PORT}/")
            if smtp_ready and web_ready:
                break
            time.sleep(2)
        if not (smtp_ready and web_ready):
            raise RuntimeError(f"mail stack not ready within {READY_TIMEOUT}s (smtp={smtp_ready}, webmail={web_ready})")
        logger.info("mail stack ready (greenmail SMTP + roundcube HTTP)")

    return MailStack()


def attach_sandbox(client, container) -> None:
    """Join a sandbox container to MAIL_NETWORK so it resolves `greenmail` and `mail.test`."""
    _ensure_network(client)
    net = client.networks.get(MAIL_NETWORK)
    net.connect(container)


def mail_env_for_app(stack: MailStack | None = None) -> dict:
    """SMTP environment to inject when booting an app so its outbound mail lands in GreenMail."""
    stack = stack or MailStack()
    return {k: v.format(host=stack.smtp_host, port=stack.smtp_port) for k, v in _SMTP_ENV_TEMPLATE.items()}


def scenario_address(run_id: str, scenario: str) -> str:
    """A unique, isolated mailbox address for one capture scenario. Base local-part only — no
    `+tag` (GreenMail pools plus-addresses into the base, which would cross-contaminate)."""
    slug = re.sub(r"[^a-z0-9]+", "-", f"{run_id}-{scenario}".lower()).strip("-")[:48]
    return f"qa-{slug}@{WEBMAIL_HOST}"


def scenario_password(run_id: str, scenario: str) -> str:
    """A deterministic per-scenario password that satisfies common sign-up complexity rules
    (upper + lower + digit + symbol). Deterministic so resume reuses the same value; only ever
    reaches the trajectory as the `<secret:password>` placeholder."""
    h = hashlib.sha256(f"{run_id}/{scenario}".encode()).hexdigest()
    return f"Qa!{h[:10]}9"


def teardown_mail_stack(client=None) -> None:
    client = client or docker.from_env()
    for name in (ROUNDCUBE_CONTAINER, GREENMAIL_CONTAINER):
        try:
            client.containers.get(name).remove(force=True)
        except docker.errors.NotFound:
            pass
    try:
        client.networks.get(MAIL_NETWORK).remove()
    except docker.errors.NotFound:
        pass


if __name__ == "__main__":
    import smtplib
    from email.mime.text import MIMEText

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    stack = ensure_mail_stack()
    addr = scenario_address("smoke", "signup")
    msg = MIMEText("Verify: https://app.test/verify?token=SMOKE123")
    msg["Subject"], msg["From"], msg["To"] = "Confirm your email", "noreply@app.test", addr
    s = smtplib.SMTP("127.0.0.1", stack.host_smtp_port, timeout=10)
    s.sendmail("noreply@app.test", [addr], msg.as_string())
    s.quit()
    logger.info(f"sent verification mail to {addr}")
    logger.info(f"webmail (in-sandbox): {stack.webmail_url}  | host: http://127.0.0.1:{stack.host_webmail_port}/")
    logger.info(
        f"app SMTP env target: {stack.smtp_host}:{stack.smtp_port}  | env keys: {sorted(mail_env_for_app(stack))}"
    )
