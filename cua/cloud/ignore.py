"""What must never leave this repo with the code: one gitignore-derived exclusion list, shaped for rsync (Vast) or for a Modal image layer."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Bytecode is machine-local, and enumerating it would put a hundred single-file patterns on the rsync line.
ALWAYS_EXCLUDE = ("__pycache__", "*.pyc")
# Applied even when git lists nothing: the human registry is written by the provisioning lane after any listing.
STATIC_UPLOAD_EXCLUDES = ("/pipeline/sites/public_sites_human.json", "/pipeline/sites/public_sites_guarded.json")
_GIT_TIMEOUT_S = 30


def gitignored_excludes(root="cua", run=subprocess.run):
    """rsync excludes for `root/`: bytecode, the known secret-bearing registries, and every other gitignored path under it."""
    argv = ["git", "ls-files", "--others", "--ignored", "--exclude-standard", f"{root}/"]
    baseline = (*ALWAYS_EXCLUDE, *STATIC_UPLOAD_EXCLUDES)
    try:
        listed = run(argv, capture_output=True, text=True, check=True, timeout=_GIT_TIMEOUT_S).stdout.splitlines()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"git could not list ignored files ({e}); using the static upload excludes")
        return baseline
    ignored = {line.strip() for line in listed if line.strip()}
    covered = {p for p in ignored if p.endswith(".pyc") or "__pycache__" in p}
    # Anchored at the transfer root, which is the `root` directory itself, never the repo.
    derived = sorted("/" + p.removeprefix(f"{root}/") for p in ignored - covered)
    return (*baseline, *(p for p in derived if p not in baseline))


def modal_ignore_patterns(root="cua", run=subprocess.run):
    """The same exclusions as dockerignore-style globs, for `modal.Image.add_local_dir(ignore=…)`."""
    return [f"**/{Path(pattern).name}" for pattern in gitignored_excludes(root, run)]
