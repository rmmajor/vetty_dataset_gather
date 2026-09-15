"""Report raw environment reads in Python sources outside the files allowed to own configuration."""

import argparse
import os
import sys
from pathlib import Path

SKIP_DIRS = {"eval_archive", "experiments", "docs", ".superpowers", "__pycache__", "build", "dist", "node_modules"}
# `environ.get` and `").environ` also catch the `__import__("os").environ` spelling, which hides from the first three.
PATTERNS = ("os.environ", "os.getenv", "environ[", "environ.get", '").environ', "getenv(")
MARKER = "# env-read:"
FIRST_PARTY_IMPORT = ("from cua ", "from cua.", "import cua")


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


def is_marked(line) -> bool:
    """True when this line carries a `# env-read: <reason>` marker with a non-empty reason."""
    head, _, reason = line.partition(MARKER)
    return head != line and bool(reason.strip())


def file_findings(path):
    """Every (line, pattern, source) where this file reads the environment without a `# env-read: <reason>` marker."""
    found = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        # The marker sits on the read itself or, when a trailing comment would make the formatter reflow it, above.
        if is_marked(line) or (number >= 2 and is_marked(lines[number - 2])):
            continue
        for pattern in PATTERNS:
            if pattern in line:
                found.append((number, pattern, line.strip()))
                break
    return found


def allowed_file_findings(path):
    """A configuration-owning file may read the environment, but importing the package it configures would cycle."""
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.startswith(FIRST_PARTY_IMPORT):
            found.append((number, "first-party import", line.strip()))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Report environment reads outside the allowed configuration modules")
    parser.add_argument("paths", nargs="+", help="Files or directories to scan")
    parser.add_argument("--allow", nargs="*", default=[], help="Files allowed to read the environment")
    parser.add_argument("--report-only", action="store_true", help="Print findings and exit 0")
    args = parser.parse_args(argv)
    allowed = {Path(entry).resolve() for entry in args.allow}
    # This file spells out the patterns it hunts for, so it would always report itself.
    own_source = Path(__file__).resolve()
    total = 0
    for path in python_files(args.paths):
        if path.resolve() == own_source:
            continue
        findings = allowed_file_findings(path) if path.resolve() in allowed else file_findings(path)
        for number, pattern, source in findings:
            print(f"{path}:{number}: {pattern}: {source}")
        total += len(findings)
    print(f"check_env_reads: {total} finding(s) outside {len(allowed)} allowed file(s)")
    return 1 if total and not args.report_only else 0


if __name__ == "__main__":
    sys.exit(main())
