"""The disposable QA identity every capture hands its teacher, so it never invents (or recalls) a real one."""

import os
import secrets

_FIRST_NAMES = [
    "Emma",
    "Liam",
    "Olivia",
    "Noah",
    "Ava",
    "Ethan",
    "Sophia",
    "Mason",
    "Isabella",
    "Lucas",
    "Mia",
    "Oliver",
    "Amelia",
    "Elijah",
    "Charlotte",
    "James",
    "Harper",
    "Benjamin",
    "Evelyn",
    "Henry",
    "Chloe",
    "Daniel",
    "Grace",
    "Owen",
    "Nora",
]
_LAST_NAMES = [
    "Smith",
    "Johnson",
    "Miller",
    "Davis",
    "Garcia",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Martin",
    "Lee",
    "Walker",
    "Hall",
    "Allen",
    "Young",
    "King",
    "Wright",
    "Scott",
    "Bennett",
    "Reed",
    "Cooper",
    "Bailey",
    "Rivera",
    "Hughes",
]

# RFC 2606 reserves example.com and NANPA reserves 555-01xx: neither can reach a real person.
PERSONA_EMAIL_DOMAIN = os.environ.get("PERSONA_EMAIL_DOMAIN", "example.com")
PERSONA_PHONE_PREFIX = os.environ.get("PERSONA_PHONE_PREFIX", "+1202555")

# Masked like real burner credentials, so a lane with a real inbox overrides the fallback in place.
MASKED_FIELDS = ("email", "phone")

_DEFAULT_PERSONA: dict | None = None


def random_name() -> tuple[str, str]:
    """A plausible disposable first/last name for signup forms — never the operator's real identity."""
    return secrets.choice(_FIRST_NAMES), secrets.choice(_LAST_NAMES)


def disposable_persona() -> dict:
    """A fresh fictional identity: display fields typed literally, MASKED_FIELDS via <secret:...>."""
    first, last = random_name()
    tag = 1000 + secrets.randbelow(9000)
    handle = f"{first.lower()}.{last.lower()}{tag}"
    return {
        "first_name": first,
        "last_name": last,
        "full_name": f"{first} {last}",
        "username": handle,
        "email": f"{handle}@{PERSONA_EMAIL_DOMAIN}",
        "phone": f"{PERSONA_PHONE_PREFIX}01{secrets.randbelow(100):02d}",
    }


def default_persona() -> dict:
    """One persona per process, minted on first use — every task of a run shares an identity."""
    global _DEFAULT_PERSONA
    if _DEFAULT_PERSONA is None:
        _DEFAULT_PERSONA = disposable_persona()
    return _DEFAULT_PERSONA


def persona_secrets(persona: dict) -> dict:
    """The persona fields the driver substitutes at type time, so they record as placeholders."""
    return {field: persona[field] for field in MASKED_FIELDS if persona.get(field)}
