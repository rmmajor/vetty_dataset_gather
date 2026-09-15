"""Report imports nested inside a function or class body, except those marked by a `# lazy:` comment above them."""

import ast
import sys

from _paths import add_pending, build_parser, python_files, report

NESTING_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
LAZY_MARKER = "# lazy:"


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
    nodes = nested_import_nodes(ast.parse("\n".join(lines)))
    owner = {number: node for node in nodes for number in range(node.lineno, node.end_lineno + 1)}
    findings = []
    for node in nodes:
        # One marker covers the whole contiguous run of nested imports below it, so a sorted block needs one comment.
        probe = node.lineno - 1
        while probe >= 1 and probe in owner:
            probe = owner[probe].lineno - 1
        marker = lines[probe - 1].strip() if probe >= 1 else ""
        if not marker.startswith(LAZY_MARKER):
            findings.append((node.lineno, lines[node.lineno - 1].strip()))
    return findings


def main(argv=None) -> int:
    parser = build_parser("Report nested imports that carry no `# lazy:` marker")
    add_pending(parser)
    args = parser.parse_args(argv)
    files = python_files(args.paths)
    lines_by_path = {}
    for path in files:
        try:
            findings = file_findings(path)
        except (SyntaxError, UnicodeDecodeError) as error:
            lines_by_path[path] = [f"{path}:1: unreadable ({error})"]
            continue
        if findings:
            lines_by_path[path] = [f"{path}:{number}: nested import: {source}" for number, source in findings]
    return report("check_nested_imports", lines_by_path, len(files), args)


if __name__ == "__main__":
    sys.exit(main())
