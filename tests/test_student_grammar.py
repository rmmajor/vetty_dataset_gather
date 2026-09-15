import ast
import dataclasses
import json
from pathlib import Path

from cua.eval.student import grammar
from cua.eval.student.grammar import ACTION_BUILDERS, ACTION_VERBS, KNOWN_KWARGS, parse_student_output

GRAMMAR_SOURCE = Path(grammar.__file__)
ROUNDTRIP = json.loads((Path(__file__).resolve().parent / "fixtures" / "action_roundtrip.json").read_text())


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


def test_every_recorded_spelling_parses_as_the_fixture_records_it():
    spellings = ROUNDTRIP["spellings"]
    assert len(spellings) >= 58, f"the grammar table shrank to {len(spellings)} spellings"
    for entry in spellings:
        parsed = parse_student_output(entry["text"])
        actual = {**dataclasses.asdict(parsed), "parse_failed": parsed.parse_failed}
        assert json.loads(json.dumps(actual)) == entry["parse_student_output"], entry["text"]


def test_rejected_spellings_stay_rejected():
    ambiguous = [e["text"] for e in ROUNDTRIP["spellings"] if e["parse_student_output"]["parse_failed"]]
    assert "action: type(hello world)" in ambiguous and "action: navigate(example.com)" in ambiguous
    for text in ambiguous:
        assert not parse_student_output(text).actions, text
