"""Report multi-line docstrings, runs of consecutive comment lines and banner comments in Python sources."""

import argparse
import ast
import io
import os
import sys
import tokenize
from pathlib import Path

SKIP_DIRS = {"eval_archive", "experiments", "docs", ".superpowers", "__pycache__", "build", "dist", "node_modules"}
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
BANNER_CHARS = set("-=")


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


def docstring_findings(tree):
    """Every docstring spanning more than one line, as (line, kind) pairs."""
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, DOCSTRING_OWNERS) or not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant):
            continue
        if isinstance(first.value.value, str) and first.end_lineno > first.lineno:
            findings.append((first.lineno, "multi-line docstring"))
    return findings


def block_findings(lines):
    """The first line of every run of two or more consecutive own-line comments."""
    findings = []
    run = []
    for line in [*lines, None]:
        if run and line == run[-1] + 1:
            run.append(line)
            continue
        if len(run) >= 2:
            findings.append((run[0], f"comment block ({len(run)} lines)"))
        run = [] if line is None else [line]
    return findings


def comment_findings(source):
    """Banner comments plus the start of every run of two or more own-line comments, as (line, kind) pairs."""
    findings = []
    own_line = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT or (token.start[0] == 1 and token.string.startswith("#!")):
            continue
        text = token.string.lstrip("#").strip()
        if text and set(text) <= BANNER_CHARS:
            findings.append((token.start[0], "banner comment"))
        if token.line.lstrip().startswith("#"):
            own_line.append(token.start[0])
    return findings + block_findings(own_line)


def file_findings(path):
    """Every comment or docstring finding in one file, sorted by line."""
    source = path.read_text(encoding="utf-8")
    return sorted(docstring_findings(ast.parse(source)) + comment_findings(source))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Report multi-line docstrings, comment blocks and banner comments")
    parser.add_argument("paths", nargs="+", help="Files or directories to scan")
    parser.add_argument("--enforce", action="store_true", help="Exit 1 when anything is reported")
    args = parser.parse_args(argv)
    files = python_files(args.paths)
    total = 0
    for path in files:
        try:
            findings = file_findings(path)
        except (SyntaxError, UnicodeDecodeError, tokenize.TokenError) as error:
            print(f"{path}:1: unreadable ({error})")
            continue
        for line, kind in findings:
            print(f"{path}:{line}: {kind}")
        if findings:
            print(f"{path}: {len(findings)} finding(s)")
        total += len(findings)
    print(f"check_comments: {total} finding(s) across {len(files)} file(s)")
    return 1 if args.enforce and total else 0


if __name__ == "__main__":
    sys.exit(main())
