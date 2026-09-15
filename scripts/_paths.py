"""The file walk, the skip list and the argument surface every repository convention check shares."""

import argparse
import os
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# ruff's own built-in excludes, which `extend-exclude` in pyproject.toml adds to rather than replaces.
RUFF_DEFAULT_EXCLUDES = {"__pycache__", "build", "dist", "node_modules", "site-packages", "venv"}


def skip_dirs() -> set[str]:
    """Directory names no check descends into: ruff's defaults plus every plain-name `extend-exclude` entry."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excluded = config.get("tool", {}).get("ruff", {}).get("extend-exclude", [])
    return RUFF_DEFAULT_EXCLUDES | {name for name in excluded if "*" not in name and "/" not in name}


SKIP_DIRS = skip_dirs()


def python_files(paths):
    """Every .py file under the given files or directories, with generated, vendored and hidden trees pruned."""
    found = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
            continue
        for directory, subdirs, filenames in os.walk(path):
            subdirs[:] = sorted(name for name in subdirs if name not in SKIP_DIRS and not name.startswith("."))
            found.extend(Path(directory) / name for name in filenames if name.endswith(".py"))
    return sorted(found)


def build_parser(description: str) -> argparse.ArgumentParser:
    """The flags every convention check shares: what to scan, whether to fail, and how much to print."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("paths", nargs="+", help="Files or directories to scan")
    parser.add_argument("--report-only", action="store_true", help="Print findings and exit 0")
    parser.add_argument("--quiet", action="store_true", help="Print the summary line alone while the run passes")
    return parser


def add_pending(parser: argparse.ArgumentParser) -> None:
    """Add `--pending`, the temporary list of paths whose findings are reported but do not fail the run."""
    parser.add_argument("--pending", nargs="*", default=[], help="Paths whose findings do not fail the run")


def is_pending(path, pending) -> bool:
    """True when `path` is one of the pending paths, or sits under one of them."""
    text = str(path)
    return any(text == entry or text.startswith(entry.rstrip("/") + "/") for entry in pending)


def report(name: str, lines_by_path: dict, scanned: int, args) -> int:
    """Print one check's findings, honouring `--quiet` and `--report-only`, and return its exit status."""
    pending = getattr(args, "pending", [])
    blocking = sum(len(lines) for path, lines in lines_by_path.items() if not is_pending(path, pending))
    total = sum(len(lines) for lines in lines_by_path.values())
    if not args.quiet or (blocking and not args.report_only):
        for lines in lines_by_path.values():
            print("\n".join(lines))
    deferred = f", {total - blocking} in pending paths" if total != blocking else ""
    print(f"{name}: {total} finding(s) across {scanned} file(s){deferred}")
    return 1 if blocking and not args.report_only else 0
