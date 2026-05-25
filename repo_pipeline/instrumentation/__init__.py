import logging
from pathlib import Path

logger = logging.getLogger(__name__)

INSTRUMENTATION_DIR = Path(__file__).parent
REPO_DIR = "/workspace/repo"
RESULTS_DIR = "/workspace/results"

INJECT_MARKER = "# --- injected trajectory instrumentation ---"


def setup_repo_instrumentation(sandbox, framework_language: str):
    """Install dependencies and inject Playwright instrumentation into the cloned
    repo so that running its tests captures trajectories to /workspace/results.

    Shared by stage 5 (trial) and stage 6 (generate).
    """
    if framework_language == "python":
        sandbox.exec_command(f"cd {REPO_DIR} && pip3 install -r requirements.txt --break-system-packages 2>&1 || true", timeout=300)
        sandbox.exec_command(f"cd {REPO_DIR} && pip3 install pytest-playwright pytest-timeout --break-system-packages 2>&1", timeout=120)
        # Repos pin their own Playwright version; fetch the matching browser so it
        # doesn't mismatch the one baked into the base image.
        sandbox.exec_command(f"cd {REPO_DIR} && python3 -m playwright install chromium 2>&1 || true", timeout=300)
        _inject_python_conftest(sandbox)
    else:
        sandbox.exec_command(f"cd {REPO_DIR} && npm install 2>&1 || true", timeout=300)
        sandbox.exec_command(f"cd {REPO_DIR} && npx playwright install chromium 2>&1 || true", timeout=120)
        fixture_content = INSTRUMENTATION_DIR.joinpath("pw_ts_fixture.ts").read_text()
        sandbox.write_file(f"{REPO_DIR}/instrumented-fixture.ts", fixture_content)

    sandbox.exec_command(f"mkdir -p {RESULTS_DIR}")


def _inject_python_conftest(sandbox):
    """Make the instrumentation load under pytest. pytest only auto-discovers files
    named exactly conftest.py, so append the instrumentation to the repo's top-level
    conftest.py (preserving its own fixtures), or create one if none exists.
    """
    conftest_content = INSTRUMENTATION_DIR.joinpath("pw_python_conftest.py").read_text()

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
