import asyncio
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from cua.pipeline.corpus.cli import DEFAULT_CORPUS, add_shared, build_parser
from cua.pipeline.corpus.pass_base import USAGE_LIMIT_MESSAGE, CorpusPass, Outcome, tally
from cua.pipeline.sdk_runner import UsageLimitReached
from cua.trajectory.io import TrajectoryDir


class ToyPass(CorpusPass):
    """Marks every dir whose instruction still needs marking, so a second run over the same corpus selects nothing."""

    name = "toy"
    workers = 2

    def __init__(self, limit_reached_at=None):
        self.limit_reached_at = limit_reached_at
        self.applied = []

    def select(self, recorded):
        return recorded.metadata.get("toy_marked") is not True

    async def apply(self, recorded):
        self.applied.append(recorded.path.name)
        if self.limit_reached_at is not None and len(self.applied) >= self.limit_reached_at:
            raise UsageLimitReached("429")
        recorded.metadata["toy_marked"] = True
        recorded.save_metadata()
        return Outcome(changed=True, note="marked")

    def summary(self, corpus, candidates):
        return f"{len(candidates)} toys need marking"


def write_corpus(root, names, marked=()):
    for name in names:
        traj_dir = root / "site_toy" / f"task_{name}"
        traj_dir.mkdir(parents=True)
        (traj_dir / "trajectory.json").write_text(json.dumps({"task_id": name, "instruction": name, "steps": []}))
        metadata = {"capture_source": "agent"}
        if name in marked:
            metadata["toy_marked"] = True
        (traj_dir / "metadata.json").write_text(json.dumps(metadata))
    return root


def run_pass(corpus_pass, corpus, **kwargs):
    """The pass under asyncio with stdout captured, so a dry run's listing can be asserted on."""
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        counts = asyncio.run(corpus_pass.run(corpus, **kwargs))
    return counts, stdout.getvalue()


def test_dry_run_lists_candidates_and_writes_nothing():
    with tempfile.TemporaryDirectory(prefix="corpus_pass_dry_") as tmp:
        corpus = write_corpus(Path(tmp) / "trajectories", ["a", "b", "c"], marked=["b"])
        before = {path: path.read_bytes() for path in corpus.glob("**/*.json")}

        counts, printed = run_pass(ToyPass(), corpus, apply=False)

        assert counts == {}, counts
        assert printed.splitlines() == [
            "site_toy/task_a",
            "site_toy/task_c",
            "2 toys need marking",
        ], printed
        assert {path: path.read_bytes() for path in corpus.glob("**/*.json")} == before


def test_apply_changes_only_selected_dirs():
    with tempfile.TemporaryDirectory(prefix="corpus_pass_apply_") as tmp:
        corpus = write_corpus(Path(tmp) / "trajectories", ["a", "b", "c"], marked=["b"])
        unselected = corpus / "site_toy" / "task_b" / "metadata.json"
        before = unselected.read_bytes()

        counts, _ = run_pass(ToyPass(), corpus)

        assert counts == {"marked": 2}, counts
        assert unselected.read_bytes() == before, "the unselected dir's metadata.json was rewritten"
        marked = {
            path.parent.name: TrajectoryDir.load(path.parent).metadata.get("toy_marked")
            for path in corpus.glob("**/metadata.json")
        }
        assert marked == {"task_a": True, "task_b": True, "task_c": True}, marked


def test_limit_caps_the_candidates():
    with tempfile.TemporaryDirectory(prefix="corpus_pass_limit_") as tmp:
        corpus = write_corpus(Path(tmp) / "trajectories", ["a", "b", "c"])

        counts, _ = run_pass(ToyPass(), corpus, limit=2)

        assert counts == {"marked": 2}, counts
        unmarked = [recorded.path.name for recorded in ToyPass().candidates(corpus)]
        assert unmarked == ["task_c"], unmarked


def test_second_run_resumes_and_selects_nothing():
    with tempfile.TemporaryDirectory(prefix="corpus_pass_resume_") as tmp:
        corpus = write_corpus(Path(tmp) / "trajectories", ["a", "b"])
        run_pass(ToyPass(), corpus)

        second = ToyPass()
        counts, _ = run_pass(second, corpus)

        assert counts == {} and second.applied == [], (counts, second.applied)


def test_usage_limit_stops_cleanly_and_keeps_the_rest_selectable():
    with tempfile.TemporaryDirectory(prefix="corpus_pass_limit_reached_") as tmp:
        corpus = write_corpus(Path(tmp) / "trajectories", ["a", "b", "c", "d"])
        corpus_pass = ToyPass(limit_reached_at=2)

        counts, _ = run_pass(corpus_pass, corpus, workers=1)

        assert counts == {"marked": 1}, counts
        assert len(ToyPass().candidates(corpus)) == 3, "only the first dir may have been marked"
        assert USAGE_LIMIT_MESSAGE.startswith("usage limit reached"), USAGE_LIMIT_MESSAGE


def test_tally_counts_notes_in_completion_order():
    outcomes = [Outcome(True, "relabeled"), Outcome(False, "failed"), Outcome(True, "relabeled")]

    assert list(tally(outcomes).items()) == [("relabeled", 2), ("failed", 1)]


def test_shared_parser_keeps_each_pass_flag_order_and_defaults():
    parser = build_parser("toy", "corpus", "dry_run", "apply", apply={"help": "Write"})
    add_shared(parser, "workers", "limit", workers={"default": 2})
    args = parser.parse_args([])

    assert [action.dest for action in parser._actions] == ["help", "corpus", "dry_run", "apply", "workers", "limit"]
    assert (args.corpus, args.dry_run, args.apply, args.workers, args.limit) == (DEFAULT_CORPUS, False, False, 2, None)
    assert parser.parse_args(["some/corpus", "--apply", "--limit", "5"]).corpus == Path("some/corpus")
