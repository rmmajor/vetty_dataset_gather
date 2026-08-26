"""Who the operator is — so no capture types their identity and no dataset ships it — configured, never hardcoded."""
import os
import re
import subprocess
from pathlib import Path

# Sources, merged in order: inline $OPERATOR_PII, $OPERATOR_PII_FILE, then the machine's git user.email.
PII_FILE = Path(os.environ.get("OPERATOR_PII_FILE")
                or Path(__file__).resolve().parent.parent / "credentials" / "operator_pii.txt")

_PATTERNS = None


def configured_identities() -> list[str]:
    """Operator-identity literals (emails, names, phone numbers) from env, file, then git user.email."""
    blobs = [os.environ.get("OPERATOR_PII", "")]
    if PII_FILE.exists():
        blobs.append(PII_FILE.read_text())
    try:
        blobs.append(subprocess.run(["git", "config", "--get", "user.email"], capture_output=True,
                                    text=True, timeout=10).stdout)
    except (OSError, subprocess.SubprocessError):
        pass
    values = []
    seen = set()
    for blob in blobs:
        for entry in re.split(r"[,;\n]", blob):
            value = entry.split("#", 1)[0].strip()
            if value and value.lower() not in seen:
                seen.add(value.lower())
                values.append(value)
    return values


def mask(value: str) -> str:
    """A recognizable but redacted rendering, so a leak report never reprints the PII in full."""
    local, at, domain = value.partition("@")
    return f"{local[:3]}***{at}{domain}"


def patterns() -> list[tuple[str, re.Pattern]]:
    """Matchers for the configured identities; an email also matches its +tag and case variants."""
    global _PATTERNS
    if _PATTERNS is None:
        _PATTERNS = []
        for value in configured_identities():
            local, at, domain = value.partition("@")
            expr = (re.escape(local) + r"(\+[^@\s]*)?@" + re.escape(domain)) if at \
                else r"\b" + re.escape(value) + r"\b"
            _PATTERNS.append((f"operator_pii({mask(value)})", re.compile(expr, re.IGNORECASE)))
    return _PATTERNS


def scan_text(text: str) -> list[str]:
    """The masked labels of every configured identity appearing in `text` — empty when none is configured."""
    return [name for name, pattern in patterns() if pattern.search(text)]
