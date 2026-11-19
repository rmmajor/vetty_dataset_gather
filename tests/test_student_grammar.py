import ast
from pathlib import Path

from cua.eval.student import grammar
from cua.eval.student.grammar import ACTION_BUILDERS, ACTION_VERBS, KNOWN_KWARGS, parse_student_output

GRAMMAR_SOURCE = Path(grammar.__file__)
# The spellings docs/evaluation.md calls genuinely ambiguous: guessing at one is worse than refusing it.
REJECTED = (
    "action: type(hello world)",
    "action: type(a, b)",
    "action: scroll(500)",
    "action: click(100)",
    "action: drag((10, 20))",
    "action: keypress()",
    "action: navigate()",
    "action: navigate(example.com)",
)


def _registry_keys(name: str) -> list:
    """The key literals of a module-level dict assignment, as written: a duplicate survives here, not in the dict."""
    tree = ast.parse(GRAMMAR_SOURCE.read_text())
    for node in tree.body:
        targets = getattr(node, "targets", []) or ([node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and isinstance(node.value, ast.Dict):
            return [key.value for key in node.value.keys if isinstance(key, ast.Constant)]
    raise AssertionError(f"{name} is not a module-level dict literal in {GRAMMAR_SOURCE}")


def test_action_builders_has_no_repeated_key_literal():
    written = _registry_keys("ACTION_BUILDERS")
    assert len(written) == len(set(written)), f"repeated key literal in ACTION_BUILDERS: {written}"
    assert written == list(ACTION_BUILDERS)


def test_every_verb_has_exactly_one_builder():
    assert set(ACTION_BUILDERS) == set(ACTION_VERBS)
    assert set(KNOWN_KWARGS) <= set(ACTION_VERBS)


def test_rejected_spellings_stay_rejected():
    for text in REJECTED:
        parsed = parse_student_output(text)
        assert parsed.parse_failed and not parsed.actions, text
