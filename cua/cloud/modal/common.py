"""Volumes, mounts, image layers and the periodic volume commit the Modal train and GRPO lanes share."""

import pathlib
import threading

import modal

from cua.cloud.ignore import modal_ignore_patterns

DATA_MOUNT = "/data"
OUT_MOUNT = "/outputs"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
CODE_ROOT = "/workspace"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

trajectories = modal.Volume.from_name("cua-trajectories", create_if_missing=True)
outputs = modal.Volume.from_name("cua-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("cua-hf-cache", create_if_missing=True)

# Xet ignores the hub/ cache these volumes already hold and re-pulls 18GB, so the old transfer path stays on.
BASE_ENV = {"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": CODE_ROOT}
PROJECT_FILES = ("pyproject.toml",)
PROJECT_DIRS = ("constraints",)
# Resolved once per import, from the same gitignore source the Vast upload filter uses.
PACKAGE_IGNORE = modal_ignore_patterns()


def configs():
    """Every config*.yaml in the repo, so --config picks one per run without an image rebuild."""
    return sorted(path.name for path in REPO_ROOT.glob("config*.yaml"))


def with_package(image, env=None, files=()):
    """Bake the shared env plus the first-party tree; add_local_* closes an image, so this is always the last step."""
    image = image.env({**BASE_ENV, **(env or {})})
    for local in (*PROJECT_FILES, *files):
        image = image.add_local_file(str(local), f"{CODE_ROOT}/{pathlib.Path(local).name}")
    for name in PROJECT_DIRS:
        image = image.add_local_dir(name, f"{CODE_ROOT}/{name}")
    # Reached through PYTHONPATH, not an editable install: no pip step can follow this layer.
    return image.add_local_dir("cua", f"{CODE_ROOT}/cua", ignore=PACKAGE_IGNORE)


def commit_periodically(volume, every_s):
    """Commit `volume` from a daemon thread so a killed run keeps its checkpoints; set the returned event to stop."""
    stop = threading.Event()

    def loop():
        while not stop.wait(every_s):
            volume.commit()

    threading.Thread(target=loop, daemon=True, name="modal-volume-commit").start()
    return stop
