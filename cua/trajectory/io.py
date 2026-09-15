"""The one corpus walk, the one tolerant JSON loader and the one read/write handle for a recorded `task_*/` directory."""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

TRAJECTORY_FILE = "trajectory.json"
METADATA_FILE = "metadata.json"
CAPTURE_META_FILE = "capture_meta.json"
# Every corpus writer has always used these, so a pass that rewrites one file leaves the rest of the corpus byte-comparable.
JSON_DUMP_KWARGS = {"indent": 2}


def iter_trajectory_dirs(root: Path) -> Iterator[Path]:
    """Every recorded trajectory directory under `root`, in sorted path order, whatever lane wrote it."""
    for path in sorted(root.glob("**/task_*")):
        if path.is_dir() and (path / TRAJECTORY_FILE).exists():
            yield path


def load_json(path: Path) -> dict:
    """The JSON object at `path`, or `{}` when it is missing, unparseable or not an object."""
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_json(path: Path, payload: dict) -> None:
    """Write one contract file with the corpus-wide dump options."""
    path.write_text(json.dumps(payload, **JSON_DUMP_KWARGS))


@dataclass
class TrajectoryDir:
    """Handle on one `task_*/` directory: its three contract files, loaded on first use and written back unchanged in shape."""

    path: Path
    _files: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Path) -> "TrajectoryDir":
        return cls(path)

    def _cached(self, name: str) -> dict:
        if name not in self._files:
            self._files[name] = load_json(self.path / name)
        return self._files[name]

    @property
    def trajectory(self) -> dict:
        return self._cached(TRAJECTORY_FILE)

    @property
    def metadata(self) -> dict:
        return self._cached(METADATA_FILE)

    @property
    def capture_meta(self) -> dict:
        return self._cached(CAPTURE_META_FILE)

    def screenshot_path(self, step: dict) -> Path:
        """A step's screenshot; a step recording none resolves to the directory itself, which is what every caller assumed."""
        return self.path / step.get("screenshot", "")

    def is_trainable(self) -> bool:
        # lazy: corpus_audit imports this module for the walk, so its gate can only be reached at call time.
        from cua.pipeline.corpus_audit import trainability

        return trainability(self.trajectory, self.metadata, self.capture_meta, self.path.name) is None

    def _save(self, name: str, payload: dict | None) -> None:
        if payload is not None:
            self._files[name] = payload
        write_json(self.path / name, self._cached(name))

    def save_trajectory(self, trajectory: dict | None = None) -> None:
        self._save(TRAJECTORY_FILE, trajectory)

    def save_metadata(self, metadata: dict | None = None) -> None:
        self._save(METADATA_FILE, metadata)

    def save_capture_meta(self, capture_meta: dict | None = None) -> None:
        self._save(CAPTURE_META_FILE, capture_meta)

    def save(self, trajectory: dict | None = None, metadata: dict | None = None, capture_meta: dict | None = None):
        """Write back only the files this call names, so a pass never rewrites one it did not touch."""
        for name, payload in (
            (TRAJECTORY_FILE, trajectory),
            (METADATA_FILE, metadata),
            (CAPTURE_META_FILE, capture_meta),
        ):
            if payload is not None:
                self._save(name, payload)
