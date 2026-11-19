"""Scripted (no-LLM) end-to-end check of auth-flow capture: register, verify by emailed code, reset, and reuse."""

import imaplib
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from cua.pipeline import driver_session, mail_service
from cua.pipeline.instrumentation import RESULTS_DIR
from cua.pipeline.sandbox import DockerSandbox
from cua.pipeline.smoke._harness import run_checks

# The six-digit code and reset link the stub app mails, read over IMAP because no LLM reads them off a screen.
CODE_PATTERN = re.compile(r"\b(\d{6})\b")
RESET_LINK_PATTERN = re.compile(r"http://127\.0\.0\.1:3000/reset\?token=[0-9a-f]+")

APP_PORT = 3000
APP_SERVER = '''
import os, random, smtplib, urllib.parse
from email.mime.text import MIMEText
from http.server import BaseHTTPRequestHandler, HTTPServer

CODES = {}
TOKENS = {}
SESSIONS = set()
SMTP_HOST = os.environ.get("SMTP_HOST", "greenmail")  # Runs inside the sandbox app process
SMTP_PORT = int(os.environ.get("SMTP_PORT", "3025"))  # Runs inside the sandbox app process

SIGNUP = """<!doctype html><html><head><title>Demo App</title></head><body>
<h1>Create your account</h1>
<form method="post" action="/signup">
<input id="email" name="email" type="email" placeholder="Email address">
<button type="submit">Sign up</button></form></body></html>"""

SENT = """<!doctype html><html><head><title>Check your email</title></head><body>
<h1>Almost there</h1><p>We emailed a 6-digit verification code to your inbox.</p>
<p>When you have it, open the verification page to finish.</p>
<a href="/verify" id="goverify">Enter your code</a></body></html>"""

VERIFY = """<!doctype html><html><head><title>Verify</title></head><body>
<h1>Verify your email</h1>
<form method="post" action="/verify">
<input id="vemail" name="email" type="email" placeholder="Email address">
<input id="code" name="code" placeholder="Verification code">
<button type="submit">Verify</button></form></body></html>"""

OK = """<!doctype html><html><head><title>Verified</title></head><body>
<h1 id="ok">Account verified!</h1><p>Welcome to the app.</p></body></html>"""

BAD = """<!doctype html><html><head><title>Verify</title></head><body>
<h1 id="bad">That code is not right</h1><a href="/verify">Try again</a></body></html>"""

FORGOT = """<!doctype html><html><head><title>Reset password</title></head><body>
<h1>Reset your password</h1>
<form method="post" action="/forgot">
<input id="femail" name="email" type="email" placeholder="Email address">
<button type="submit">Send reset link</button></form></body></html>"""

RESET_SENT = """<!doctype html><html><head><title>Reset link sent</title></head><body>
<h1>Reset link sent</h1><p>Check your email for a link to reset your password.</p></body></html>"""

RESET_OK = """<!doctype html><html><head><title>Reset</title></head><body>
<h1 id="resetok">Password reset link verified</h1><p>You can now set a new password.</p></body></html>"""

RESET_BAD = """<!doctype html><html><head><title>Reset</title></head><body>
<h1 id="resetbad">Invalid or expired reset link</h1></body></html>"""

LOGIN = """<!doctype html><html><head><title>Sign in</title></head><body>
<h1>Sign in</h1>
<form method="post" action="/login">
<input id="lemail" name="email" type="email" placeholder="Email address">
<input id="lpass" name="password" type="password" placeholder="Password">
<button type="submit">Sign in</button></form></body></html>"""

DASHBOARD = """<!doctype html><html><head><title>Dashboard</title></head><body>
<h1 id="dash">Your dashboard</h1><p>Signed in. This is the protected account area.</p>
<a href="/account">Account</a></body></html>"""

SIGNED_OUT = """<!doctype html><html><head><title>Account</title></head><body>
<h1 id="signedout">Please sign in to view your account</h1>
<a href="/login">Sign in</a></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, html, status=200, cookie=None):
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _signed_in(self):
        raw = self.headers.get("Cookie", "")
        cookies = dict(c.strip().split("=", 1) for c in raw.split(";") if "=" in c)
        return cookies.get("sid") in SESSIONS

    def do_GET(self):
        if self.path.startswith("/verify"):
            self._send(VERIFY)
        elif self.path.startswith("/forgot"):
            self._send(FORGOT)
        elif self.path.startswith("/reset"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            token = (q.get("token") or [""])[0]
            self._send(RESET_OK if token and token in TOKENS.values() else RESET_BAD)
        elif self.path.startswith("/login"):
            self._send(LOGIN)
        elif self.path.startswith("/account"):
            self._send(DASHBOARD if self._signed_in() else SIGNED_OUT)
        else:
            self._send(SIGNUP)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        form = urllib.parse.parse_qs(self.rfile.read(n).decode())
        if self.path == "/signup":
            email = form.get("email", [""])[0]
            code = "%06d" % random.randint(0, 999999)
            CODES[email] = code
            msg = MIMEText("Your verification code is %s. Enter it to finish signing up." % code)
            msg["Subject"] = "Your verification code"
            msg["From"] = "noreply@demo.test"
            msg["To"] = email
            try:
                s = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
                s.sendmail("noreply@demo.test", [email], msg.as_string())
                s.quit()
            except Exception as e:
                self._send("<h1>mail error</h1><pre>%s</pre>" % e, 500)
                return
            self._send(SENT)
        elif self.path == "/login":
            email = form.get("email", [""])[0]
            password = form.get("password", [""])[0]
            if email and password:
                sid = "%016x" % random.randrange(16 ** 16)
                SESSIONS.add(sid)
                self._send(DASHBOARD, cookie="sid=%s; Path=/" % sid)
            else:
                self._send(LOGIN, status=400)
        elif self.path == "/verify":
            email = form.get("email", [""])[0]
            code = form.get("code", [""])[0]
            self._send(OK if CODES.get(email) == code else BAD)
        elif self.path == "/forgot":
            email = form.get("email", [""])[0]
            token = "%08x" % random.randrange(16 ** 8)
            TOKENS[email] = token
            link = "http://127.0.0.1:3000/reset?token=%s" % token
            html = ('<p>We received a request to reset your password.</p>'
                    '<p><a href="%s">Choose a new password</a></p>'
                    '<p>Or open this link: %s</p>' % (link, link))
            msg = MIMEText(html, "html")
            msg["Subject"] = "Reset your password"
            msg["From"] = "noreply@demo.test"
            msg["To"] = email
            try:
                s = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
                s.sendmail("noreply@demo.test", [email], msg.as_string())
                s.quit()
            except Exception as e:
                self._send("<h1>mail error</h1><pre>%s</pre>" % e, 500)
                return
            self._send(RESET_SENT)
        else:
            self._send("not found", 404)

    def log_message(self, *a):
        pass


HTTPServer(("0.0.0.0", 3000), H).serve_forever()
'''


def _ref_for(snapshot: str, needle: str, role: str = None) -> int:
    for line in snapshot.splitlines():
        if not line.startswith("[") or needle.lower() not in line.lower():
            continue
        if role and f"] {role} " not in line:
            continue
        return int(line[1 : line.index("]")])
    raise AssertionError(f"no {role or 'element'} matching {needle!r} in snapshot:\n{snapshot[:800]}")


def _read_latest_mail(stack, address: str, pattern: re.Pattern, what: str) -> str:
    """Poll the mailbox over IMAP for `pattern` in the newest message, where the LLM teacher would read a screen."""
    for _ in range(20):
        try:
            mailbox = imaplib.IMAP4("127.0.0.1", stack.host_imap_port)
            mailbox.login(address, "qa")
            mailbox.select("INBOX")
            ids = mailbox.search(None, "ALL")[1][0].split()
            if ids:
                raw = mailbox.fetch(ids[-1], "(RFC822)")[1][0][1].decode("utf-8", "replace")
                mailbox.logout()
                match = pattern.search(raw)
                if match:
                    # Group 0 is the whole match, so a pattern with one capture group returns that group instead.
                    return match.group(pattern.groups)
            mailbox.logout()
        except Exception:
            pass
        time.sleep(1)
    raise AssertionError(f"{what} never arrived in the mailbox")


class AuthFlowSmoke:
    """The mail stack, sandbox, stub app and driver the checks below share, in the order they run."""

    def __init__(self):
        self.stack = None
        self.sandbox = None
        self.client = None
        self.app_url = f"http://127.0.0.1:{APP_PORT}/"
        self.address = ""
        self.reset_address = ""
        self.login_address = ""
        self.snapshot = ""
        self.verify_snapshot = ""
        self.captured_state = None

    def start(self) -> list[str]:
        """Boot the mail stack, the sandboxed stub app and the driver the whole flow runs through."""
        self.stack = mail_service.ensure_mail_stack()
        self.sandbox = DockerSandbox()
        self.sandbox.start("auth-flow/smoke")
        mail_service.attach_sandbox(self.sandbox.client, self.sandbox.container)
        self.sandbox.exec_command("mkdir -p /workspace/app")
        self.sandbox.write_file("/workspace/app/server.py", APP_SERVER)
        env = mail_service.mail_env_for_app(self.stack)
        env_prefix = " ".join(f"{k}='{v}'" for k, v in env.items())
        self.sandbox.start_background(f"cd /workspace/app && {env_prefix} python3 server.py", "/workspace/app.log")
        if not self.sandbox.wait_for_port(APP_PORT, 30):
            return ["stub app did not start"]
        driver_session.install_driver(self.sandbox)
        driver_session.start_driver(self.sandbox)
        self.client = driver_session.DriverClient(self.sandbox)
        return []

    def stop(self) -> None:
        if self.sandbox is not None:
            self.sandbox.stop()

    def mail_config(self, address: str) -> dict:
        return {"url": self.stack.webmail_url, "address": address, "password": "qa"}

    def trajectory(self, task_id: str) -> dict:
        """The trajectory the driver recorded for one task, copied out of the container."""
        with tempfile.TemporaryDirectory(prefix="auth_flow_smoke_") as tmp:
            self.sandbox.copy_out(RESULTS_DIR, tmp)
            return json.loads((Path(tmp) / f"task_{task_id}" / "trajectory.json").read_text())


SESSION = AuthFlowSmoke()


def check_boot_app_and_driver() -> list[str]:
    return SESSION.start()


def check_signup_sends_code() -> list[str]:
    """Register with the masked placeholder: the driver types the real disposable address."""
    SESSION.address = mail_service.scenario_address("smoke", "signup")
    session = SESSION.client.session(
        "auth_flow_smoke",
        SESSION.app_url,
        instruction="Register a new account and verify the email address.",
        secrets={"email": SESSION.address, "password": "Passw0rd!"},
        mail_config=SESSION.mail_config(SESSION.address),
    )
    SESSION.snapshot = session["snapshot"]
    SESSION.client.act(
        {
            "kind": "type",
            "ref": _ref_for(SESSION.snapshot, "Email address"),
            "text": "<secret:email>",
            "thought": "I'm creating an account, so I enter my email address to sign up.",
        }
    )
    result = SESSION.client.act(
        {
            "kind": "click",
            "ref": _ref_for(SESSION.snapshot, "Sign up"),
            "thought": "With my email entered, I click Sign up to create the account.",
        }
    )
    snapshot = result["snapshot"].lower()
    if "verification code" not in snapshot and "almost there" not in snapshot:
        return [f"signup did not reach the check-your-email page: {result['snapshot'][:400]}"]
    return []


def check_webmail_shows_code() -> list[str]:
    """Walking to the webmail in the browser is a RECORDED navigate step, and the new mail shows on that fresh load."""
    SESSION.client.act(
        {
            "kind": "navigate",
            "url": SESSION.stack.webmail_url,
            "thought": "The app emailed me a code, so I open my inbox to read it.",
        }
    )
    SESSION.client.act({"kind": "wait", "seconds": 2, "thought": "Letting the inbox refresh so the new message shows."})
    inbox_snapshot = SESSION.client.snapshot()["snapshot"]
    if "verification code" not in inbox_snapshot.lower():
        return [f"verification email not visible in the Roundcube inbox: {inbox_snapshot[:600]}"]
    return []


def check_code_verifies_account() -> list[str]:
    """The code is read out-of-band here; the LLM teacher reads it from the screenshot."""
    code = _read_latest_mail(SESSION.stack, SESSION.address, CODE_PATTERN, "verification code")
    SESSION.client.act(
        {
            "kind": "navigate",
            "url": f"http://127.0.0.1:{APP_PORT}/verify",
            "thought": "I have the code now, so I go back to the app to finish verifying.",
        }
    )
    SESSION.verify_snapshot = SESSION.client.snapshot()["snapshot"]
    SESSION.client.act(
        {
            "kind": "type",
            "ref": _ref_for(SESSION.verify_snapshot, "Email address"),
            "text": "<secret:email>",
            "thought": "I re-enter my email on the verification form.",
        }
    )
    SESSION.client.act(
        {
            "kind": "type",
            "ref": _ref_for(SESSION.verify_snapshot, "Verification code"),
            "text": code,
            "thought": "I type the verification code from my email.",
        }
    )
    done = SESSION.client.act(
        {
            "kind": "click",
            "ref": _ref_for(SESSION.verify_snapshot, "Verify"),
            "thought": "I submit the code to confirm my email and finish signing up.",
        }
    )
    SESSION.client.act({"kind": "done", "thought": "The page confirms my account is verified — the goal is met."})
    SESSION.client.finish("completed", "Registered, read the emailed code in the webmail, and verified the account.")
    if "account verified" not in done["snapshot"].lower():
        return [f"verification did not succeed: {done['snapshot'][:400]}"]
    return []


def check_signup_trajectory() -> list[str]:
    failures = []
    traj = SESSION.trajectory("auth_flow_smoke")
    blob = json.dumps(traj)
    if SESSION.address in blob:
        failures.append("LEAK: the real disposable address appears in the trajectory")
    if "<secret:email>" not in blob:
        failures.append("masked email placeholder was not recorded")
    navs = [
        a
        for s in traj["steps"]
        for a in s["actions"]
        if a.get("type") == "navigate" and a.get("url") == SESSION.stack.webmail_url
    ]
    if not navs:
        failures.append("the webmail navigation was not recorded as a navigate step")
    for step in traj["steps"]:
        for action in step["actions"]:
            if "x" in action and not (0 <= action["x"] <= 1280 and 0 <= action["y"] <= 720):
                failures.append(f"bad coords: {action}")
    return failures


def check_reset_link_requested() -> list[str]:
    """Scenario 2 opens on the forgot-password form with its own burner address."""
    SESSION.reset_address = mail_service.scenario_address("smoke", "reset")
    session = SESSION.client.session(
        "auth_flow_reset",
        f"http://127.0.0.1:{APP_PORT}/forgot",
        instruction="Reset your password using the link emailed to you.",
        secrets={"email": SESSION.reset_address},
        mail_config=SESSION.mail_config(SESSION.reset_address),
    )
    SESSION.snapshot = session["snapshot"]
    SESSION.client.act(
        {
            "kind": "type",
            "ref": _ref_for(SESSION.snapshot, "Email address"),
            "text": "<secret:email>",
            "thought": "I forgot my password, so I enter my email to request a reset link.",
        }
    )
    sent = SESSION.client.act(
        {
            "kind": "click",
            "ref": _ref_for(SESSION.snapshot, "Send reset link"),
            "thought": "I submit the form to have the reset link emailed to me.",
        }
    )
    if "reset link sent" not in sent["snapshot"].lower():
        return [f"reset request did not confirm: {sent['snapshot'][:300]}"]
    return []


def check_reset_link_in_message_iframe() -> list[str]:
    """Roundcube renders the message body in an iframe, so surfacing its link proves cross-iframe snapshotting."""
    SESSION.client.act(
        {
            "kind": "navigate",
            "url": SESSION.stack.webmail_url,
            "thought": "I open my inbox to find the password-reset email.",
        }
    )
    SESSION.client.act({"kind": "wait", "seconds": 2, "thought": "Letting the inbox load the new message."})
    inbox_snapshot = SESSION.client.snapshot()["snapshot"]
    SESSION.client.act(
        {
            "kind": "click",
            "ref": _ref_for(inbox_snapshot, "Reset your password", role="link"),
            "thought": "I open the password-reset email to read the reset link.",
        }
    )
    message_snapshot = ""
    for _ in range(8):
        SESSION.client.wait(1)
        message_snapshot = SESSION.client.snapshot()["snapshot"]
        if "choose a new password" in message_snapshot.lower():
            break
    if "choose a new password" not in message_snapshot.lower():
        return [f"reset link not surfaced from the message iframe: {message_snapshot[:500]}"]
    return []


def check_reset_link_verifies() -> list[str]:
    """Webmail clients sandbox body links against navigation, so the agent reads the link and navigates to it."""
    reset_url = _read_latest_mail(SESSION.stack, SESSION.reset_address, RESET_LINK_PATTERN, "reset link")
    done = SESSION.client.act(
        {
            "kind": "navigate",
            "url": reset_url,
            "thought": "I follow the reset link from the email to set a new password.",
        }
    )
    SESSION.client.act({"kind": "done", "thought": "The reset link is confirmed valid — the goal is met."})
    SESSION.client.finish("completed", "Requested a reset, opened the email, and followed the reset link.")
    if "password reset link verified" not in done["snapshot"].lower():
        return [f"following the reset link did not verify: {done['snapshot'][:400]}"]
    return []


def check_reset_trajectory() -> list[str]:
    if SESSION.reset_address in json.dumps(SESSION.trajectory("auth_flow_reset")):
        return ["LEAK: reset address appears in the trajectory"]
    return []


def check_login_captures_session() -> list[str]:
    """Scenario 3 signs in and captures the storage_state that pre_session tasks start from."""
    failures = []
    SESSION.login_address = mail_service.scenario_address("smoke", "account")
    session = SESSION.client.session(
        "auth_login_capture",
        f"http://127.0.0.1:{APP_PORT}/login",
        instruction="Sign in to the account.",
        secrets={"email": SESSION.login_address, "password": "Passw0rd!"},
    )
    SESSION.snapshot = session["snapshot"]
    SESSION.client.act(
        {
            "kind": "type",
            "ref": _ref_for(SESSION.snapshot, "Email address"),
            "text": "<secret:email>",
            "thought": "I enter my email to sign in.",
        }
    )
    SESSION.client.act(
        {
            "kind": "type",
            "ref": _ref_for(SESSION.snapshot, "Password"),
            "text": "<secret:password>",
            "thought": "I enter my password to sign in.",
        }
    )
    dash = SESSION.client.act(
        {
            "kind": "click",
            "ref": _ref_for(SESSION.snapshot, "Sign in"),
            "thought": "I submit the sign-in form to access my account.",
        }
    )
    if "your dashboard" not in dash["snapshot"].lower():
        failures.append(f"login did not reach the dashboard: {dash['snapshot'][:300]}")
    SESSION.captured_state = SESSION.client.storage_state()
    if not SESSION.captured_state.get("cookies"):
        failures.append("no cookies captured in the storage_state")
    SESSION.client.act({"kind": "done", "thought": "I'm signed in and on my dashboard — the goal is met."})
    SESSION.client.finish("completed", "Signed in to the account.")
    return failures


def check_fresh_session_is_signed_out() -> list[str]:
    """The control: without an injected session /account really gates, so the reuse check below proves something."""
    control = SESSION.client.session(
        "auth_no_session", f"http://127.0.0.1:{APP_PORT}/account", instruction="Open the account page."
    )
    SESSION.client.finish("partial", "Not signed in.")
    if "please sign in" not in control["snapshot"].lower():
        return [f"a fresh session should be signed out at /account: {control['snapshot'][:300]}"]
    return []


def check_pre_session_reuse() -> list[str]:
    """Injecting the captured session loads the protected dashboard with no login step, as a pre_session task starts."""
    reused = SESSION.client.session(
        "auth_pre_session",
        f"http://127.0.0.1:{APP_PORT}/account",
        instruction="Open the account page.",
        auth_config={"auth_mode": "pre_session", "storage_state": SESSION.captured_state},
    )
    SESSION.client.act({"kind": "done", "thought": "I'm already signed in, so the account page is shown directly."})
    SESSION.client.finish("completed", "Reached the account area via the reused session, no login needed.")
    if "your dashboard" not in reused["snapshot"].lower():
        return [f"pre_session injection did not start signed in: {reused['snapshot'][:300]}"]
    return []


def check_login_trajectory() -> list[str]:
    failures = []
    blob = json.dumps(SESSION.trajectory("auth_login_capture"))
    if SESSION.login_address in blob:
        failures.append("LEAK: the real address appears in the login trajectory")
    if "Passw0rd!" in blob:
        failures.append("LEAK: the real password appears in the login trajectory")
    if "<secret:password>" not in blob:
        failures.append("masked password placeholder was not recorded")
    return failures


CHECKS = [
    check_boot_app_and_driver,
    check_signup_sends_code,
    check_webmail_shows_code,
    check_code_verifies_account,
    check_signup_trajectory,
    check_reset_link_requested,
    check_reset_link_in_message_iframe,
    check_reset_link_verifies,
    check_reset_trajectory,
    check_login_captures_session,
    check_fresh_session_is_signed_out,
    check_pre_session_reuse,
    check_login_trajectory,
]


if __name__ == "__main__":
    try:
        sys.exit(run_checks(CHECKS, "Auth-flow smoke"))
    finally:
        SESSION.stop()
