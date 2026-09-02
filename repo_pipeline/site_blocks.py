"""Anti-bot / WAF wall signatures — one registry, shared by the eval harness and the RL episode loop."""
import re


BLOCK_SIGNATURES: list = [
    ("captcha", re.compile(r"\b(re)?captcha\b|\bhcaptcha\b|press *(&|and) *hold", re.IGNORECASE)),
    ("human_verification", re.compile(
        r"verify (that )?you('re| are)( a)? human|are you a robot|i'm not a robot|"
        r"confirm you are (a )?human|prove you('re| are) (not a robot|human)", re.IGNORECASE)),
    ("cloudflare", re.compile(
        r"checking your browser before accessing|cloudflare ray id|"
        r"enable javascript and cookies to continue|attention required! \| cloudflare", re.IGNORECASE)),
    ("waf_block", re.compile(
        r"access denied|403 forbidden|you (have been|are) blocked|request unsuccessful|"
        r"pardon our interruption|unusual traffic from your computer|"
        r"your request has been blocked|bot detection", re.IGNORECASE)),
    # Sustained RL traffic earns rate limits long before it earns a CAPTCHA, and an
    # undetected limit turns a whole task all-fail while looking like a policy failure.
    ("rate_limited", re.compile(
        r"\b429\b|too many requests|rate limit(ed|s)?|slow down|"
        r"try again (later|in a few)|temporarily unavailable|"
        r"you (have )?(exceeded|reached) .{0,20}limit", re.IGNORECASE)),
]


def detect_block(url: str, snapshot: str) -> str | None:
    """Name of the first blocking signature visible in the current observation, else None."""
    haystack = f"{url or ''}\n{snapshot or ''}"
    for name, pattern in BLOCK_SIGNATURES:
        if pattern.search(haystack):
            return name
    return None
