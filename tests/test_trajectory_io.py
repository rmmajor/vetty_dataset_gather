import json
import logging
import tempfile
from pathlib import Path

from cua.trajectory.io import TrajectoryDir, iter_trajectory_dirs, load_json, write_json
from cua.trajectory.schema import metadata_from

# The key set cua.pipeline.lanes.capture_lane.verify_and_finalize writes for every site_* capture.
SITE_CAPTURE_KEYS = [
    "task_id",
    "instruction",
    "start_url",
    "total_steps",
    "success",
    "website",
    "domain",
    "framework",
    "capture_source",
    "instruction_source",
    "thought_source",
    "task_type",
    "difficulty",
    "efficiency",
    "efficiency_ratio",
    "relabeled",
    "viewport",
    "source_repo",
    "autonomy",
    "auth_mode",
    "task_style",
    "risk_category",
    "preconditions",
]


# The three directory layouts the corpus carries: a public-site lane, the repo funnel and Path A.
CORPUS_LAYOUTS = (
    "site_synth/task_0001_search",
    "site_synth/task_0002_settings",
    "repo_synth_app/task_0001_board",
    "repo_synth_app/task_0002_login",
    "task_mind2web_0001",
)


def write_corpus(root: Path) -> Path:
    """One trajectory directory per layout, each carrying the single file the walk keys on."""
    for relative in CORPUS_LAYOUTS:
        directory = root / relative
        directory.mkdir(parents=True)
        write_json(directory / "trajectory.json", {"task_id": directory.name, "steps": []})
    return root


def test_walk_finds_every_layout_and_skips_dirs_without_a_trajectory():
    with tempfile.TemporaryDirectory(prefix="traj_io_walk_") as tmp:
        root = write_corpus(Path(tmp) / "trajectories")
        (root / "site_synth" / "task_9999_empty").mkdir(parents=True)
        (root / "repo_synth_app" / "not_a_task").mkdir(parents=True)
        found = list(iter_trajectory_dirs(root))

        assert found == sorted(found), found
        relative = {str(d.relative_to(root)) for d in found}
        assert relative == set(CORPUS_LAYOUTS), relative
        assert "site_synth/task_9999_empty" not in relative
        assert not any(d.name == "not_a_task" for d in found)


def test_load_json_on_missing_invalid_and_non_object():
    with tempfile.TemporaryDirectory(prefix="traj_io_load_") as tmp:
        root = Path(tmp)
        assert load_json(root / "absent.json") == {}
        (root / "broken.json").write_text("{not json")
        assert load_json(root / "broken.json") == {}
        (root / "list.json").write_text("[1, 2, 3]")
        assert load_json(root / "list.json") == {}
        (root / "null.json").write_text("null")
        assert load_json(root / "null.json") == {}
        (root / "good.json").write_text('{"a": 1}')
        assert load_json(root / "good.json") == {"a": 1}
        assert load_json(root) == {}


def test_save_is_byte_identical_to_the_dump_options_every_writer_used():
    payload = {"task_id": "t", "thought": "ünïcødé ✓", "steps": [{"step": 1, "actions": [{"type": "done"}]}]}
    with tempfile.TemporaryDirectory(prefix="traj_io_save_") as tmp:
        traj_dir = Path(tmp) / "site_x" / "task_1"
        traj_dir.mkdir(parents=True)
        expected = json.dumps(payload, indent=2)

        recorded = TrajectoryDir.load(traj_dir)
        recorded.save(trajectory=payload, metadata=payload, capture_meta=payload)
        for name in ("trajectory.json", "metadata.json", "capture_meta.json"):
            assert (traj_dir / name).read_text() == expected, name

        write_json(traj_dir / "hand.json", payload)
        assert (traj_dir / "hand.json").read_bytes() == (traj_dir / "trajectory.json").read_bytes()


def test_lazy_load_mutate_and_save_round_trips_through_disk():
    with tempfile.TemporaryDirectory(prefix="traj_io_trip_") as tmp:
        traj_dir = Path(tmp) / "task_1"
        traj_dir.mkdir(parents=True)
        write_json(traj_dir / "trajectory.json", {"steps": [{"step": 1, "screenshot": "step_001.png"}]})

        recorded = TrajectoryDir.load(traj_dir)
        assert recorded.metadata == {} and recorded.capture_meta == {}
        assert recorded.screenshot_path(recorded.trajectory["steps"][0]) == traj_dir / "step_001.png"
        assert recorded.screenshot_path({}) == traj_dir

        recorded.trajectory["success"] = True
        recorded.save_trajectory()
        assert load_json(traj_dir / "trajectory.json")["success"] is True
        assert TrajectoryDir.load(traj_dir).trajectory["success"] is True


def test_save_only_writes_the_files_it_is_given():
    with tempfile.TemporaryDirectory(prefix="traj_io_partial_") as tmp:
        traj_dir = Path(tmp) / "task_1"
        traj_dir.mkdir(parents=True)
        write_json(traj_dir / "trajectory.json", {"task_id": "t"})

        TrajectoryDir.load(traj_dir).save(metadata={"task_id": "t"})
        assert (traj_dir / "metadata.json").exists()
        assert not (traj_dir / "capture_meta.json").exists()


def test_metadata_from_keeps_the_capture_key_set_order_and_values():
    built = metadata_from(
        task_id="task_paste",
        instruction="Share an encrypted note.",
        start_url="https://0bin.net/",
        total_steps=7,
        success=True,
        website="0bin.net",
        domain="public_web",
        framework="agent",
        capture_source="agent",
        instruction_source="curated",
        thought_source="teacher_raw",
        task_type="other",
        difficulty="medium",
        efficiency=None,
        efficiency_ratio=None,
        relabeled=False,
        viewport={"width": 1280, "height": 720},
        source_repo="public:0bin-net",
        autonomy="autonomous",
        auth_mode="none",
        task_style="",
        risk_category="",
        preconditions={"requires_session": False, "session_ref": "", "start_url": "https://0bin.net/", "env": "live"},
    )

    assert list(built) == SITE_CAPTURE_KEYS, list(built)
    assert built["efficiency"] is None and built["efficiency_ratio"] is None
    assert built["viewport"] == {"width": 1280, "height": 720}
    assert built["preconditions"]["env"] == "live"
    assert built["success"] is True and built["relabeled"] is False


def test_metadata_from_validates_a_partial_update_without_the_identity_fields():
    built = metadata_from(total_steps=3, n_navigations=1, valid_bbox_ratio=0.5, efficiency="acceptable")

    assert list(built) == ["total_steps", "n_navigations", "valid_bbox_ratio", "efficiency"]
    assert "task_id" not in built and "instruction" not in built and "start_url" not in built
    assert built["valid_bbox_ratio"] == 0.5 and isinstance(built["valid_bbox_ratio"], float)


class _CapturedWarnings(logging.Handler):
    """Collects the messages cua.trajectory.schema logs while a test runs."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _write_metadata(**fields) -> tuple[str, list[str]]:
    """(bytes written to metadata.json, warnings logged) for one metadata_from + write_json round."""
    captured = _CapturedWarnings()
    schema_logger = logging.getLogger("cua.trajectory.schema")
    schema_logger.addHandler(captured)
    try:
        with tempfile.TemporaryDirectory(prefix="traj_io_guard_") as tmp:
            path = Path(tmp) / "metadata.json"
            write_json(path, metadata_from(**fields))
            return path.read_text(), captured.messages
    finally:
        schema_logger.removeHandler(captured)


def test_an_odd_generated_value_is_written_through_instead_of_crashing_capture():
    # efficiency/difficulty come from unvalidated LLM JSON, so the schema accepts them as-is — no warning, no crash.
    odd_efficiency = {"task_id": "t", "efficiency": {"grade": "weird"}, "difficulty": ["hard"]}
    written, warnings = _write_metadata(**odd_efficiency)
    assert written == json.dumps(odd_efficiency, indent=2), written
    assert warnings == [], warnings

    # A typed tier-2 field given a list still degrades to the pre-schema write-through, but says so.
    odd_domain = {"task_id": "t", "domain": ["a", "b"], "total_steps": 3}
    written, warnings = _write_metadata(**odd_domain)
    assert written == json.dumps(odd_domain, indent=2), written
    assert len(warnings) == 1 and "domain" in warnings[0], warnings


def test_metadata_from_names_an_unknown_field_instead_of_dropping_it():
    try:
        metadata_from(task_id="t", wall_demo=True)
    except ValueError as exc:
        assert "wall_demo" in str(exc) and "TrajectoryMetadata" in str(exc), exc
        return
    raise AssertionError("metadata_from silently accepted a field the schema does not declare")
