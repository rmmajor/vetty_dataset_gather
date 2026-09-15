"""The corpus passes' shared argparse surface: one declaration per flag, added in whatever order a pass's help already prints it."""

import argparse
from pathlib import Path

DEFAULT_CORPUS = Path("trajectories/")

# Stdlib-only on purpose: `audit` builds its parser from here and must import no SDK and no pydantic.
SHARED_ARGUMENTS = {
    "corpus": (("corpus",), {"type": Path, "nargs": "?", "default": DEFAULT_CORPUS}),
    "apply": (("--apply",), {"action": "store_true"}),
    "dry_run": (("--dry-run",), {"action": "store_true"}),
    "workers": (("--workers",), {"type": int, "default": 3}),
    "limit": (("--limit",), {"type": int, "default": None}),
}


def add_shared(parser, *names, **overrides):
    """Add the named shared arguments in the given order; `workers={"default": 2}` retunes one without redeclaring it."""
    for name in names:
        flags, kwargs = SHARED_ARGUMENTS[name]
        parser.add_argument(*flags, **{**kwargs, **overrides.get(name, {})})
    return parser


def build_parser(description, *names, **overrides):
    """A pass's parser: the shared arguments it names, in that order, before it adds its own flags."""
    return add_shared(argparse.ArgumentParser(description=description), *names, **overrides)
