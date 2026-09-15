"""Scripted (no-LLM) end-to-end check of auth-flow capture: register on an app, walk to the
webmail in the browser, read the verification code, return to the app, verify.

Proves the mail-stack wiring + driver additions without an LLM:
  * a sandbox attaches to the shared `vetty-mail` network (mail_service.attach_sandbox);
  * an app booted with the injected SMTP env delivers its verification mail to GreenMail;
  * the driver pre-logs the scenario mailbox into Roundcube (mail_config) so the inbox is ready;
  * the agent's NAVIGATE to the webmail is a recorded `{type:"navigate"}` step;
  * the masked `<secret:email>` is what lands in the trajectory — never the real address.

The verification CODE is read here over IMAP (no LLM to read it from the screenshot); the LLM
teacher reading the code from the inbox screenshot is what the full agent E2E exercises. Link-based
(click-the-link) verification additionally needs iframe-aware snapshotting (separate task).

Run: python -m cua.pipeline.auth_flow_smoke
"""

import imaplib
import json
import logging
import re
import sys
import tempfile
from pathlib import Path

from cua.pipeline import driver_session, mail_service
from cua.pipeline.instrumentation import RESULTS_DIR
from cua.pipeline.sandbox import DockerSandbox

logger = logging.getLogger(__name__)

APP_PORT = 3000
APP_SERVER = '''
import os, random, smtplib, urllib.parse
from email.mime.text import MIMEText
from http.server import BaseHTTPRequestHandler, HTTPServer

CODES = {}
TOKENS = {}
SESSIONS = set()
SMTP_HOST = os.environ.get("SMTP_HOST", "greenmail")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "3025"))

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


def _read_reset_url_via_imap(stack, address: str) -> str:
    import re as _re

    for _ in range(20):
        try:
            M = imaplib.IMAP4("127.0.0.1", stack.host_imap_port)
            M.login(address, "qa")
            M.select("INBOX")
            ids = M.search(None, "ALL")[1][0].split()
            if ids:
                raw = M.fetch(ids[-1], "(RFC822)")[1][0][1].decode("utf-8", "replace")
                M.logout()
                m = _re.search(r"http://127\.0\.0\.1:3000/reset\?token=[0-9a-f]+", raw)
                if m:
                    return m.group(0)
            M.logout()
        except Exception:
            pass
        import time

        time.sleep(1)
    raise AssertionError("reset link never arrived in the mailbox")


def _read_code_via_imap(stack, address: str) -> str:
    for _ in range(20):
        try:
            M = imaplib.IMAP4("127.0.0.1", stack.host_imap_port)
            M.login(address, "qa")
            M.select("INBOX")
            ids = M.search(None, "ALL")[1][0].split()
            if ids:
                raw = M.fetch(ids[-1], "(RFC822)")[1][0][1].decode("utf-8", "replace")
                M.logout()
                m = re.search(r"\b(\d{6})\b", raw)
                if m:
                    return m.group(1)
            M.logout()
        except Exception:
            pass
        import time

        time.sleep(1)
    raise AssertionError("verification code never arrived in the mailbox")


def run_smoke() -> bool:
    stack = mail_service.ensure_mail_stack()
    sandbox = DockerSandbox()
    try:
        sandbox.start("auth-flow/smoke")
        mail_service.attach_sandbox(sandbox.client, sandbox.container)

        sandbox.exec_command("mkdir -p /workspace/app")
        sandbox.write_file("/workspace/app/server.py", APP_SERVER)
        env = mail_service.mail_env_for_app(stack)
        env_prefix = " ".join(f"{k}='{v}'" for k, v in env.items())
        sandbox.start_background(f"cd /workspace/app && {env_prefix} python3 server.py", "/workspace/app.log")
        assert sandbox.wait_for_port(APP_PORT, 30), "stub app did not start"

        driver_session.install_driver(sandbox)
        driver_session.start_driver(sandbox)
        client = driver_session.DriverClient(sandbox)

        address = mail_service.scenario_address("smoke", "signup")
        app_url = f"http://127.0.0.1:{APP_PORT}/"
        session = client.session(
            "auth_flow_smoke",
            app_url,
            instruction="Register a new account and verify the email address.",
            secrets={"email": address, "password": "Passw0rd!"},
            mail_config={"url": stack.webmail_url, "address": address, "password": "qa"},
        )
        snapshot = session["snapshot"]

        # 1. Register with the masked email placeholder (driver types the real disposable address).
        client.act(
            {
                "kind": "type",
                "ref": _ref_for(snapshot, "Email address"),
                "text": "<secret:email>",
                "thought": "I'm creating an account, so I enter my email address to sign up.",
            }
        )
        result = client.act(
            {
                "kind": "click",
                "ref": _ref_for(snapshot, "Sign up"),
                "thought": "With my email entered, I click Sign up to create the account.",
            }
        )
        assert "verification code" in result["snapshot"].lower() or "almost there" in result["snapshot"].lower(), (
            f"signup did not reach the check-your-email page: {result['snapshot'][:400]}"
        )

        # 2. Walk to the webmail in the browser — a RECORDED navigate step.
        inbox = client.act(
            {
                "kind": "navigate",
                "url": stack.webmail_url,
                "thought": "The app emailed me a code, so I open my inbox to read it.",
            }
        )
        # Pre-login means the inbox is already shown; the new mail appears on this fresh load.
        client.act({"kind": "wait", "seconds": 2, "thought": "Letting the inbox refresh so the new message shows."})
        inbox_snap = client.snapshot()["snapshot"]
        assert "verification code" in inbox_snap.lower(), (
            f"verification email not visible in the Roundcube inbox: {inbox_snap[:600]}"
        )

        # 3. Read the code (out-of-band here; the LLM teacher reads it from the screenshot), return
        #    to the app's verify page, and enter it.
        code = _read_code_via_imap(stack, address)
        verify_url = f"http://127.0.0.1:{APP_PORT}/verify"
        client.act(
            {
                "kind": "navigate",
                "url": verify_url,
                "thought": "I have the code now, so I go back to the app to finish verifying.",
            }
        )
        vsnap = client.snapshot()["snapshot"]
        client.act(
            {
                "kind": "type",
                "ref": _ref_for(vsnap, "Email address"),
                "text": "<secret:email>",
                "thought": "I re-enter my email on the verification form.",
            }
        )
        client.act(
            {
                "kind": "type",
                "ref": _ref_for(vsnap, "Verification code"),
                "text": code,
                "thought": "I type the verification code from my email.",
            }
        )
        done = client.act(
            {
                "kind": "click",
                "ref": _ref_for(vsnap, "Verify"),
                "thought": "I submit the code to confirm my email and finish signing up.",
            }
        )
        assert "account verified" in done["snapshot"].lower(), f"verification did not succeed: {done['snapshot'][:400]}"
        client.act({"kind": "done", "thought": "The page confirms my account is verified — the goal is met."})
        client.finish("completed", "Registered, read the emailed code in the webmail, and verified the account.")

        # 4. Inspect the recorded trajectory.
        with tempfile.TemporaryDirectory() as tmp:
            sandbox.copy_out(RESULTS_DIR, tmp)
            traj = json.loads((Path(tmp) / "task_auth_flow_smoke" / "trajectory.json").read_text())
            blob = json.dumps(traj)

            assert address not in blob, "LEAK: the real disposable address appears in the trajectory"
            assert "<secret:email>" in blob, "masked email placeholder was not recorded"
            navs = [
                a
                for s in traj["steps"]
                for a in s["actions"]
                if a.get("type") == "navigate" and a.get("url") == stack.webmail_url
            ]
            assert navs, "the webmail navigation was not recorded as a navigate step"
            for step in traj["steps"]:
                for a in step["actions"]:
                    if "x" in a:
                        assert 0 <= a["x"] <= 1280 and 0 <= a["y"] <= 720, f"bad coords: {a}"

        # ---- Scenario 2: LINK-based password reset (clicks the link INSIDE the email body, which
        # Roundcube renders in an iframe — exercises cross-iframe snapshot + ref resolution). ----
        addr2 = mail_service.scenario_address("smoke", "reset")
        session2 = client.session(
            "auth_flow_reset",
            f"http://127.0.0.1:{APP_PORT}/forgot",
            instruction="Reset your password using the link emailed to you.",
            secrets={"email": addr2},
            mail_config={"url": stack.webmail_url, "address": addr2, "password": "qa"},
        )
        fsnap = session2["snapshot"]
        client.act(
            {
                "kind": "type",
                "ref": _ref_for(fsnap, "Email address"),
                "text": "<secret:email>",
                "thought": "I forgot my password, so I enter my email to request a reset link.",
            }
        )
        sent = client.act(
            {
                "kind": "click",
                "ref": _ref_for(fsnap, "Send reset link"),
                "thought": "I submit the form to have the reset link emailed to me.",
            }
        )
        assert "reset link sent" in sent["snapshot"].lower(), f"reset request did not confirm: {sent['snapshot'][:300]}"

        client.act(
            {
                "kind": "navigate",
                "url": stack.webmail_url,
                "thought": "I open my inbox to find the password-reset email.",
            }
        )
        client.act({"kind": "wait", "seconds": 2, "thought": "Letting the inbox load the new message."})
        rsnap = client.snapshot()["snapshot"]
        client.act(
            {
                "kind": "click",
                "ref": _ref_for(rsnap, "Reset your password", role="link"),
                "thought": "I open the password-reset email to read the reset link.",
            }
        )
        # The body renders async into an iframe; poll (non-recorded) until its link surfaces — this is
        # the cross-iframe-traversal proof (the link lives inside Roundcube's message-body iframe).
        msnap = ""
        for _ in range(8):
            client.wait(1)
            msnap = client.snapshot()["snapshot"]
            if "choose a new password" in msnap.lower():
                break
        assert "choose a new password" in msnap.lower(), (
            f"reset link not surfaced from the message iframe: {msnap[:500]}"
        )

        # Follow the link by NAVIGATING to it (the robust real-world pattern: webmail clients sandbox
        # message-body links against navigation, so the agent reads the link and goes to it). The URL
        # is read here over IMAP; the LLM teacher reads it from the rendered message.
        reset_url = _read_reset_url_via_imap(stack, addr2)
        done = client.act(
            {
                "kind": "navigate",
                "url": reset_url,
                "thought": "I follow the reset link from the email to set a new password.",
            }
        )
        assert "password reset link verified" in done["snapshot"].lower(), (
            f"following the reset link did not verify: {done['snapshot'][:400]}"
        )
        client.act({"kind": "done", "thought": "The reset link is confirmed valid — the goal is met."})
        client.finish("completed", "Requested a reset, opened the email, and followed the reset link.")

        with tempfile.TemporaryDirectory() as tmp:
            sandbox.copy_out(RESULTS_DIR, tmp)
            rtraj = json.loads((Path(tmp) / "task_auth_flow_reset" / "trajectory.json").read_text())
            assert addr2 not in json.dumps(rtraj), "LEAK: reset address appears in the trajectory"

        # ---- Scenario 3: capture an authenticated session and REUSE it (pre_session). Proves the
        # driver's /storage_state capture + storageState injection — the mechanism that lets most
        # "account" tasks start ALREADY signed in instead of replaying a login on every trajectory. ----
        addr3 = mail_service.scenario_address("smoke", "account")
        login_session = client.session(
            "auth_login_capture",
            f"http://127.0.0.1:{APP_PORT}/login",
            instruction="Sign in to the account.",
            secrets={"email": addr3, "password": "Passw0rd!"},
        )
        lsnap = login_session["snapshot"]
        client.act(
            {
                "kind": "type",
                "ref": _ref_for(lsnap, "Email address"),
                "text": "<secret:email>",
                "thought": "I enter my email to sign in.",
            }
        )
        client.act(
            {
                "kind": "type",
                "ref": _ref_for(lsnap, "Password"),
                "text": "<secret:password>",
                "thought": "I enter my password to sign in.",
            }
        )
        dash = client.act(
            {
                "kind": "click",
                "ref": _ref_for(lsnap, "Sign in"),
                "thought": "I submit the sign-in form to access my account.",
            }
        )
        assert "your dashboard" in dash["snapshot"].lower(), (
            f"login did not reach the dashboard: {dash['snapshot'][:300]}"
        )
        captured = client.storage_state()
        assert captured.get("cookies"), "no cookies captured in the storage_state"
        client.act({"kind": "done", "thought": "I'm signed in and on my dashboard — the goal is met."})
        client.finish("completed", "Signed in to the account.")

        # Control: a FRESH session (no injected state) at /account is signed OUT — confirms the
        # protected page really gates on the session, so the reuse test below proves something.
        control = client.session(
            "auth_no_session", f"http://127.0.0.1:{APP_PORT}/account", instruction="Open the account page."
        )
        assert "please sign in" in control["snapshot"].lower(), (
            f"a fresh session should be signed out at /account: {control['snapshot'][:300]}"
        )
        client.finish("partial", "Not signed in.")

        # Reuse: inject the captured session — /account loads the protected dashboard WITHOUT any
        # login step, exactly as a pre_session "account" task would start.
        reused = client.session(
            "auth_pre_session",
            f"http://127.0.0.1:{APP_PORT}/account",
            instruction="Open the account page.",
            auth_config={"auth_mode": "pre_session", "storage_state": captured},
        )
        assert "your dashboard" in reused["snapshot"].lower(), (
            f"pre_session injection did not start signed in: {reused['snapshot'][:300]}"
        )
        client.act({"kind": "done", "thought": "I'm already signed in, so the account page is shown directly."})
        client.finish("completed", "Reached the account area via the reused session, no login needed.")

        with tempfile.TemporaryDirectory() as tmp:
            sandbox.copy_out(RESULTS_DIR, tmp)
            ltraj = json.loads((Path(tmp) / "task_auth_login_capture" / "trajectory.json").read_text())
            lblob = json.dumps(ltraj)
            assert addr3 not in lblob, "LEAK: the real address appears in the login trajectory"
            assert "Passw0rd!" not in lblob, "LEAK: the real password appears in the login trajectory"
            assert "<secret:password>" in lblob, "masked password placeholder was not recorded"

        logger.info("Auth-flow smoke PASSED (code flow + link-in-iframe reset + session capture/reuse)")
        return True
    finally:
        sandbox.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ok = False
    try:
        ok = run_smoke()
    except AssertionError as e:
        logger.error(f"Auth-flow smoke FAILED: {e}")
    except Exception as e:
        logger.exception(f"Auth-flow smoke ERRORED: {e}")
    sys.exit(0 if ok else 1)
