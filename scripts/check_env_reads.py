"""Report raw environment reads in Python sources outside the files allowed to own configuration."""

import sys
from pathlib import Path

from _paths import build_parser, python_files, report

# `environ.get` and `").environ` also catch the `__import__("os").environ` spelling, which hides from the first three.
PATTERNS = ("os.environ", "os.getenv", "environ[", "environ.get", '").environ', "getenv(")
MARKER = "# env-read:"
FIRST_PARTY_IMPORT = ("from cua ", "from cua.", "import cua")


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
    parser = build_parser("Report environment reads outside the allowed configuration modules")
    parser.add_argument("--allow", nargs="*", default=[], help="Files allowed to read the environment")
    args = parser.parse_args(argv)
    allowed = {Path(entry).resolve() for entry in args.allow}
    # This file spells out the patterns it hunts for, so it would always report itself.
    own_source = Path(__file__).resolve()
    files = python_files(args.paths)
    lines_by_path = {}
    for path in files:
        if path.resolve() == own_source:
            continue
        findings = allowed_file_findings(path) if path.resolve() in allowed else file_findings(path)
        if findings:
            lines_by_path[path] = [f"{path}:{number}: {pattern}: {source}" for number, pattern, source in findings]
    return report("check_env_reads", lines_by_path, len(files), args)


if __name__ == "__main__":
    sys.exit(main())
