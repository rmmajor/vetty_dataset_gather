"""Report multi-line docstrings, runs of consecutive comment lines and banner comments in Python sources."""

import ast
import io
import sys
import tokenize

from _paths import add_pending, build_parser, python_files, report

DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
BANNER_CHARS = set("-=")
BANNER_FENCE = 3


def is_banner(text: str) -> bool:
    """A rule made only of `-`/`=`, or a title fenced by at least three of them at both ends."""
    if not text:
        return False
    leading = len(text) - len(text.lstrip("-="))
    trailing = len(text) - len(text.rstrip("-="))
    return set(text) <= BANNER_CHARS or (leading >= BANNER_FENCE and trailing >= BANNER_FENCE)


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
        if is_banner(text):
            findings.append((token.start[0], "banner comment"))
        if token.line.lstrip().startswith("#"):
            own_line.append(token.start[0])
    return findings + block_findings(own_line)


def file_findings(path):
    """Every comment or docstring finding in one file, sorted by line."""
    source = path.read_text(encoding="utf-8")
    return sorted(docstring_findings(ast.parse(source)) + comment_findings(source))


def main(argv=None) -> int:
    parser = build_parser("Report multi-line docstrings, comment blocks and banner comments")
    add_pending(parser)
    args = parser.parse_args(argv)
    files = python_files(args.paths)
    lines_by_path = {}
    for path in files:
        try:
            findings = file_findings(path)
        except (SyntaxError, UnicodeDecodeError, tokenize.TokenError) as error:
            lines_by_path[path] = [f"{path}:1: unreadable ({error})"]
            continue
        if findings:
            lines_by_path[path] = [f"{path}:{line}: {kind}" for line, kind in findings]
    return report("check_comments", lines_by_path, len(files), args)


if __name__ == "__main__":
    sys.exit(main())
