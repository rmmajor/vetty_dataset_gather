import logging
from pathlib import Path

from repo_pipeline.frameworks import FrameworkProfile

logger = logging.getLogger(__name__)

INSTRUMENTATION_DIR = Path(__file__).parent
REPO_DIR = "/workspace/repo"
RESULTS_DIR = "/workspace/results"

INJECT_MARKER = "# --- injected trajectory instrumentation ---"


def setup_repo_instrumentation(sandbox, profile: FrameworkProfile):
    """Install dependencies and inject the framework's instrumentation into the cloned
    repo so that running its tests captures trajectories to /workspace/results.

    Driven entirely by `profile` (see repo_pipeline.frameworks). Shared by stage 5
    (trial) and stage 6 (generate).
    """
    for command in profile.install_commands:
        sandbox.exec_command(command.format(repo_dir=REPO_DIR), timeout=300)

    if profile.injection_kind == "python_conftest":
        _inject_python_conftest(sandbox, profile.instrumentation_filename)
    elif profile.injection_kind == "ts_fixture":
        fixture_content = INSTRUMENTATION_DIR.joinpath(profile.instrumentation_filename).read_text()
        sandbox.write_file(f"{REPO_DIR}/instrumented-fixture.ts", fixture_content)
    else:
        raise ValueError(f"Unknown injection_kind: {profile.injection_kind}")

    sandbox.exec_command(f"mkdir -p {RESULTS_DIR}")


def _inject_python_conftest(sandbox, conftest_filename: str):
    """Make the instrumentation load under pytest. pytest only auto-discovers files
    named exactly conftest.py, so append the instrumentation to the repo's top-level
    conftest.py (preserving its own fixtures), or create one if none exists.
    """
    conftest_content = INSTRUMENTATION_DIR.joinpath(conftest_filename).read_text()

    existing_conftests = sandbox.list_files("conftest.py", REPO_DIR)
    if existing_conftests:
        conftest_path = min(existing_conftests, key=lambda p: p.count("/"))
        try:
            existing = sandbox.read_file(conftest_path)
        except FileNotFoundError:
            existing = ""
        if INJECT_MARKER in existing:
            return
        merged = f"{existing.rstrip()}\n\n\n{INJECT_MARKER}\n{conftest_content}"
        sandbox.write_file(conftest_path, merged)
        logger.info(f"Appended instrumentation to existing {conftest_path}")
    else:
        test_files = sandbox.list_files("test_*.py", REPO_DIR)
        if test_files:
            test_dir = str(Path(test_files[0]).parent)
        else:
            test_dir = f"{REPO_DIR}/tests"
            sandbox.exec_command(f"mkdir -p {test_dir}")
        sandbox.write_file(f"{test_dir}/conftest.py", conftest_content)
        logger.info(f"Created {test_dir}/conftest.py with instrumentation")
