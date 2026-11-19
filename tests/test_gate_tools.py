"""The merged bootable-gate CLI: both subcommands parse, the constants are defined once, no database is opened."""

import contextlib
import io
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from cua.pipeline.tools import gate_tools

MODULE_PATH = Path(gate_tools.__file__)
SOURCE = MODULE_PATH.read_text()

# Both proven-good outcomes only print on failure, so the report's error paths are exercised as well.
PROBE_OUTCOMES = {
    "acme/agentapp": (True, "bootable app: react + dev script"),
    "acme/brokenapp": (False, "test-suite only (e2e tooling, no app framework)"),
    "acme/flakyapp": (None, "gh probe failed: HTTP 502"),
    "acme/legacysuite": (False, "test-suite only (e2e tooling, no app framework)"),
    "junk/one": (False, "no package.json / compose / app structure"),
    "junk/two": (True, "ambiguous: no framework detected"),
    "junk/three": (None, "gh probe failed: HTTP 404"),
    "junk/four": (False, "no package.json / compose / app structure"),
}
REPO_ROWS = [
    ("acme/agentapp", "completed", 4),
    ("acme/brokenapp", "completed", 2),
    ("acme/flakyapp", "completed", 1),
    ("acme/legacysuite", "completed", 3),
    ("junk/one", "rejected_score", 0),
    ("junk/two", "rejected_score", 0),
    ("junk/three", "rejected_score", 0),
    ("junk/four", "rejected_analyze", 0),
]
AGENT_CAPTURED = ("acme/agentapp", "acme/brokenapp", "acme/flakyapp")


def build_fake_registry(root: Path) -> Path:
    db_path = root / "registry.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE repos (full_name TEXT, status TEXT, trajectory_count INTEGER)")
    connection.executemany("INSERT INTO repos VALUES (?, ?, ?)", REPO_ROWS)
    connection.commit()
    connection.close()
    return db_path


def build_fake_corpus(root: Path) -> Path:
    corpus = root / "trajectories"
    for name in AGENT_CAPTURED:
        task_dir = corpus / f"repo_{name.replace('/', '_')}" / "task_001"
        task_dir.mkdir(parents=True)
        (task_dir / "metadata.json").write_text('{"capture_source": "agent"}')
    return corpus


def run_validate(argv) -> str:
    """Drive the real `validate` command with the gate replaced by a fixed accept/reject/error mix."""
    original = gate_tools.probe_all
    gate_tools.probe_all = lambda names, workers=gate_tools.PROBE_WORKERS: [(n, *PROBE_OUTCOMES[n]) for n in names]
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            assert gate_tools.main(argv) == 0
    finally:
        gate_tools.probe_all = original
    return buffer.getvalue()


def test_both_subcommands_print_help():
    for argv in (["--help"], ["validate", "--help"], ["regate", "--help"]):
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                gate_tools.main(argv)
        except SystemExit as exit_code:
            assert exit_code.code == 0
        assert "usage:" in buffer.getvalue()


def test_validate_defaults_match_the_original_script():
    args = gate_tools.build_parser().parse_args(["validate"])
    assert args.sample == 150
    assert args.db == gate_tools.DEFAULT_DB
    assert args.trajectories == gate_tools.DEFAULT_TRAJECTORIES
    assert args.func is gate_tools.command_validate


def test_regate_defaults_match_the_original_script():
    args = gate_tools.build_parser().parse_args(["regate"])
    assert args.sample == 400
    assert args.apply is False
    assert args.aggressive is False
    assert args.func is gate_tools.command_regate


def test_positional_sample_and_flags_still_parse():
    args = gate_tools.build_parser().parse_args(["regate", "50", "--apply", "--aggressive"])
    assert (args.sample, args.apply, args.aggressive) == (50, True, True)
    assert gate_tools.build_parser().parse_args(["validate", "7"]).sample == 7


def test_a_missing_subcommand_is_an_error():
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            gate_tools.build_parser().parse_args([])
    except SystemExit as exit_code:
        assert exit_code.code != 0
    else:
        raise AssertionError("a bare invocation must not parse")


def test_shared_constants_are_defined_once():
    for name in ("SCORE_COST", "ANALYZE_COST", "SAMPLE_SEED", "DEFAULT_DB"):
        assert SOURCE.count(f"\n{name} = ") == 1
    assert gate_tools.SAMPLE_SEED == 1781409272
    assert gate_tools.SCORE_COST == 184.19 / 4181
    assert gate_tools.ANALYZE_COST == 91.50 / 525


def test_the_hard_reject_rule_is_the_only_drop_rule():
    assert gate_tools.drops("test-suite only (e2e tooling, no app framework)", aggressive=False) is True
    assert gate_tools.drops("no framework detected", aggressive=False) is False
    assert gate_tools.drops("no framework detected", aggressive=True) is True


def test_validate_reports_every_bucket_end_to_end():
    """The whole `validate` path over a known mix, so the report's 3-tuple unpacking cannot regress."""
    with tempfile.TemporaryDirectory(prefix="gate_tools_validate_") as tmp:
        root = Path(tmp)
        db_path = build_fake_registry(root)
        corpus = build_fake_corpus(root)
        out = run_validate(["--db", str(db_path), "validate", "3", "--trajectories", str(corpus)])

    assert "AGENT-MODE PROVEN-GOOD (must all ACCEPT): 3 repos" in out
    assert "ACCEPT=1  REJECT=1  probe_error=1" in out
    assert "!! FALSE NEGATIVES (gate would wrongly reject agent-mode proven-good repos):" in out
    assert "acme/brokenapp: test-suite only (e2e tooling, no app framework)" in out
    assert "probe error: acme/flakyapp: gh probe failed: HTTP 502" in out
    assert "PRECISION FLOOR PASS" not in out

    assert "LEGACY REPLAY proven-good (rejection is intended): 1 repos" in out
    assert "legacy rejected (test suites, no bootable app — intended exclusion): ['acme/legacysuite']" in out

    # 3 rejected_score + SAMPLE // 3 = 1 rejected_analyze; 2 of the 3 gradable ones reject.
    assert "LLM-REJECTED BACKLOG SAMPLE (should mostly REJECT): 4 repos" in out
    assert "gate REJECT rate on junk: 67%  (2/3 gradable)" in out
    assert "junk/two: ambiguous: no framework detected" in out
    assert "Of 3 rejected_score repos, gate would pre-reject ~67% = ~2 repos" in out


def test_validate_passes_the_precision_floor_when_nothing_is_rejected():
    with tempfile.TemporaryDirectory(prefix="gate_tools_floor_") as tmp:
        root = Path(tmp)
        db_path = build_fake_registry(root)
        corpus = root / "trajectories"
        task_dir = corpus / "repo_acme_agentapp" / "task_001"
        task_dir.mkdir(parents=True)
        (task_dir / "metadata.json").write_text('{"capture_source": "agent"}')
        out = run_validate(["--db", str(db_path), "validate", "1", "--trajectories", str(corpus)])

    assert "AGENT-MODE PROVEN-GOOD (must all ACCEPT): 1 repos" in out
    assert "PRECISION FLOOR PASS: zero false negatives on agent-mode proven-good" in out
    assert "FALSE NEGATIVES" not in out


def test_validate_never_writes_to_the_registry():
    """A report must not migrate or WAL-ify the operator's database, so its connection is opened mode=ro."""
    with tempfile.TemporaryDirectory(prefix="gate_tools_ro_") as tmp:
        db_path = build_fake_registry(Path(tmp))
        connection = gate_tools.open_readonly(str(db_path))
        assert connection.execute("SELECT COUNT(*) AS n FROM repos").fetchone()["n"] == len(REPO_ROWS)
        try:
            connection.execute("DELETE FROM repos")
        except sqlite3.OperationalError as error:
            assert "readonly" in str(error)
        else:
            raise AssertionError("open_readonly returned a writable connection")
        connection.close()


def test_importing_the_module_opens_no_database():
    """The import sweep runs anywhere, so a module-level sqlite3.connect would break it outside a checkout."""
    probe = (
        "import sqlite3;"
        "opened=[];"
        "real=sqlite3.connect;"
        "sqlite3.connect=lambda *a, **k: opened.append(a) or real(*a, **k);"
        "import cua.pipeline.tools.gate_tools;"
        "print(opened)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(MODULE_PATH.parents[3]),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
