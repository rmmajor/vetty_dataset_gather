"""Report imports nested inside a function or class body, except those marked by a `# lazy:` comment above them."""

import argparse
import ast
import os
import sys
from pathlib import Path

SKIP_DIRS = {"eval_archive", "experiments", "docs", ".superpowers", "__pycache__", "build", "dist", "node_modules"}
NESTING_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
LAZY_MARKER = "# lazy:"


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


def nested_import_nodes(node, nested=False):
    """Every import node below this one that sits inside a function or class body."""
    found = []
    for child in ast.iter_child_nodes(node):
        if nested and isinstance(child, (ast.Import, ast.ImportFrom)):
            found.append(child)
        found.extend(nested_import_nodes(child, nested or isinstance(child, NESTING_NODES)))
    return found


def file_findings(path):
    """Every (line, source) nested import in this file that carries no `# lazy:` marker."""
    lines = path.read_text(encoding="utf-8").splitlines()
    findings = []
    for node in nested_import_nodes(ast.parse("\n".join(lines))):
        marker = lines[node.lineno - 2].strip() if node.lineno >= 2 else ""
        if not marker.startswith(LAZY_MARKER):
            findings.append((node.lineno, lines[node.lineno - 1].strip()))
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Report nested imports that carry no `# lazy:` marker")
    parser.add_argument("paths", nargs="+", help="Files or directories to scan")
    parser.add_argument("--report-only", action="store_true", help="Print findings and exit 0")
    args = parser.parse_args(argv)
    total = 0
    for path in python_files(args.paths):
        try:
            findings = file_findings(path)
        except (SyntaxError, UnicodeDecodeError) as error:
            print(f"{path}:1: unreadable ({error})")
            continue
        for number, source in findings:
            print(f"{path}:{number}: nested import: {source}")
        total += len(findings)
    print(f"check_nested_imports: {total} unmarked nested import(s)")
    return 1 if total and not args.report_only else 0


if __name__ == "__main__":
    sys.exit(main())
