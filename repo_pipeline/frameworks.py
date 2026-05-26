"""Test-framework strategy registry.

The pipeline supports more than one browser-test framework. Rather than scatter
`if framework == "playwright"` branches across the stages, every framework-specific
decision (how to install deps, which instrumentation file to inject, how to invoke the
test runner, which action names count as interactive) lives in a `FrameworkProfile`.
Stages resolve a profile from the `(test_framework, framework_language)` pair recorded
during deep analysis and drive their behaviour from it. Adding a framework (e.g.
Cypress) is a new entry here plus its instrumentation file — no stage edits.

The trajectory output contract is identical across profiles; only the capture mechanism
differs (see the matching file in `instrumentation/`).
"""
from dataclasses import dataclass

# The Playwright image is built arch-native (fast on Apple Silicon). The Selenium image
# is linux/amd64-only — Chrome/chromedriver for Linux exist only for x86_64 — so it runs
# natively on x86_64 hosts and under emulation on arm64.
SANDBOX_IMAGE = "repo-pipeline-sandbox:latest"
SANDBOX_IMAGE_SELENIUM = "repo-pipeline-sandbox-selenium:latest"

# Shared pytest invocation: Selenium-Python and Playwright-Python both run under pytest,
# so they share the same command shape. {repo_name}/{test_name_prefix} are filled by the
# stage; <test_file> is a placeholder the agent substitutes for the chosen file.
_PYTEST_SINGLE = (
    'cd /workspace/repo && TRAJECTORY_DIR=/workspace/results SOURCE_REPO="{repo_name}" '
    'TEST_NAME_PREFIX="{test_name_prefix}" xvfb-run -a python3 -m pytest <test_file> '
    "-v --timeout=60 2>&1 | head -100"
)
_PYTEST_SUITE = (
    'cd /workspace/repo && TRAJECTORY_DIR=/workspace/results SOURCE_REPO="{repo_name}" '
    'TEST_NAME_PREFIX="{test_name_prefix}" xvfb-run -a python3 -m pytest --timeout=120 '
    "-v 2>&1 | tail -50"
)
_PW_TS_SINGLE = (
    'cd /workspace/repo && TRAJECTORY_DIR=/workspace/results SOURCE_REPO="{repo_name}" '
    'TEST_NAME_PREFIX="{test_name_prefix}" xvfb-run -a npx playwright test <test_file> '
    "--reporter=list 2>&1 | head -100"
)
_PW_TS_SUITE = (
    'cd /workspace/repo && TRAJECTORY_DIR=/workspace/results SOURCE_REPO="{repo_name}" '
    'TEST_NAME_PREFIX="{test_name_prefix}" xvfb-run -a npx playwright test '
    "--reporter=list 2>&1 | tail -50"
)


@dataclass(frozen=True)
class FrameworkProfile:
    name: str                          # registry key, e.g. "selenium-python"
    test_framework: str                # "playwright" | "selenium"
    language: str                      # "python" | "typescript" | "javascript"
    install_commands: tuple            # shell commands, .format(repo_dir=...), run in order
    injection_kind: str                # "python_conftest" | "ts_fixture"
    instrumentation_filename: str      # file in instrumentation/ to inject
    test_command_single: str           # stage 5, .format(repo_name=, test_name_prefix=)
    test_command_suite: str            # stage 6, .format(repo_name=, test_name_prefix=)
    setup_note: str                    # framework-specific setup hint for the agent prompt
    action_vocabulary: tuple           # method names that count as interactive actions
    sandbox_image: str                 # Docker image used for stages 5/6
    sandbox_platform: str | None       # forced platform (e.g. "linux/amd64") or None for native


PROFILES: dict[str, FrameworkProfile] = {
    "playwright-python": FrameworkProfile(
        name="playwright-python",
        test_framework="playwright",
        language="python",
        install_commands=(
            "cd {repo_dir} && pip3 install -r requirements.txt --break-system-packages 2>&1 || true",
            "cd {repo_dir} && pip3 install pytest-playwright pytest-timeout --break-system-packages 2>&1",
            # Repos pin their own Playwright version; fetch the matching browser so it
            # doesn't mismatch the one baked into the base image.
            "cd {repo_dir} && python3 -m playwright install chromium 2>&1 || true",
        ),
        injection_kind="python_conftest",
        instrumentation_filename="pw_python_conftest.py",
        test_command_single=_PYTEST_SINGLE,
        test_command_suite=_PYTEST_SUITE,
        setup_note="A conftest.py with instrumentation has been placed in the test directory; it sets the viewport itself, so no config changes are needed.",
        action_vocabulary=(
            "click", "fill", "type", "press", "selectOption", "select_option",
            "check", "uncheck", "dblclick", "dragTo", "drag_and_drop",
        ),
        sandbox_image=SANDBOX_IMAGE,
        sandbox_platform=None,
    ),
    "playwright-ts": FrameworkProfile(
        name="playwright-ts",
        test_framework="playwright",
        language="typescript",
        install_commands=(
            "cd {repo_dir} && npm install 2>&1 || true",
            "cd {repo_dir} && npx playwright install chromium 2>&1 || true",
        ),
        injection_kind="ts_fixture",
        instrumentation_filename="pw_ts_fixture.ts",
        test_command_single=_PW_TS_SINGLE,
        test_command_suite=_PW_TS_SUITE,
        setup_note="A fixture file has been placed at /workspace/repo/instrumented-fixture.ts. Modify playwright.config.ts to set the viewport to 1280x720.",
        action_vocabulary=(
            "click", "fill", "type", "press", "selectOption", "select_option",
            "check", "uncheck", "dblclick", "dragTo", "drag_and_drop",
        ),
        sandbox_image=SANDBOX_IMAGE,
        sandbox_platform=None,
    ),
    "selenium-python": FrameworkProfile(
        name="selenium-python",
        test_framework="selenium",
        language="python",
        install_commands=(
            "cd {repo_dir} && pip3 install -r requirements.txt --break-system-packages 2>&1 || true",
            "cd {repo_dir} && pip3 install -e . --break-system-packages 2>&1 || true",
            "cd {repo_dir} && pip3 install selenium pytest pytest-timeout --break-system-packages 2>&1",
        ),
        injection_kind="python_conftest",
        instrumentation_filename="selenium_python_conftest.py",
        test_command_single=_PYTEST_SINGLE,
        test_command_suite=_PYTEST_SUITE,
        setup_note="A conftest.py with Selenium instrumentation has been placed in the test directory. It locks the viewport to 1280x720 via CDP and captures coordinates from getBoundingClientRect, so no config changes are needed. Chrome and a matching chromedriver are already on PATH in the image.",
        action_vocabulary=(
            "click", "send_keys", "clear", "submit",
            "select_by_value", "select_by_visible_text", "select_by_index",
        ),
        sandbox_image=SANDBOX_IMAGE_SELENIUM,
        sandbox_platform="linux/amd64",
    ),
}

_DEFAULT_PROFILE = PROFILES["playwright-ts"]


def resolve_profile(test_framework: str | None, framework_language: str | None) -> FrameworkProfile:
    """Map the deep-analysis (test_framework, framework_language) pair to a profile.

    Unknown / missing values fall back to playwright-ts, preserving the historical
    default before Selenium support existed.
    """
    fw = (test_framework or "").strip().lower()
    lang = (framework_language or "").strip().lower()

    if fw == "selenium" and lang == "python":
        return PROFILES["selenium-python"]
    if lang == "python":
        return PROFILES["playwright-python"]
    return _DEFAULT_PROFILE


def all_interactive_actions() -> set[str]:
    """Union of every profile's interactive action vocabulary, lowercased.

    Used by stage 4 deep scoring to count interactive actions regardless of framework.
    """
    actions: set[str] = set()
    for profile in PROFILES.values():
        actions.update(a.lower() for a in profile.action_vocabulary)
    return actions
