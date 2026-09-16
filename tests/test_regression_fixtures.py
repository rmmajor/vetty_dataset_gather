from tests.regression_fixtures import DEPENDENCY_NOTE_PREFIX, dependency_note, fixture_matches

MISSING_OPTIONAL = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'openai'\n[exit 1]\n"
MISSING_FIRST_PARTY = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'cua.prompt'\n[exit 1]"
HELP = "options:\n  -h, --help  show this help message and exit\n"
BROKEN = "Traceback (most recent call last):\nSyntaxError: invalid syntax\n[exit 1]\n"


def test_only_a_third_party_import_becomes_a_note():
    assert dependency_note(MISSING_OPTIONAL) == f"{DEPENDENCY_NOTE_PREFIX}openai\n"
    assert dependency_note(MISSING_FIRST_PARTY) is None
    assert dependency_note(HELP) is None


def test_a_note_matches_a_clean_capture_in_either_direction():
    note = dependency_note(MISSING_OPTIONAL)
    assert fixture_matches(note, HELP) and fixture_matches(HELP, note)
    assert fixture_matches(note, note)


def test_a_note_never_matches_a_real_break():
    note = dependency_note(MISSING_OPTIONAL)
    assert not fixture_matches(note, BROKEN)
    assert not fixture_matches(note, f"{DEPENDENCY_NOTE_PREFIX}datasets\n")
    assert not fixture_matches(HELP, BROKEN)
