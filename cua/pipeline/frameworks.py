"""Test-framework strategy registry: every framework-specific capture decision lives in a `FrameworkProfile`."""

from dataclasses import dataclass

# The Selenium image is linux/amd64-only, because Chrome and chromedriver for Linux exist only for x86_64.
SANDBOX_IMAGE = "repo-pipeline-sandbox:latest"
SANDBOX_IMAGE_SELENIUM = "repo-pipeline-sandbox-selenium:latest"

# `<test_file>` is a placeholder the agent substitutes; the FRAMEWORK vars let the in-sandbox recorder self-describe.
_ENV_PREFIX = (
    'cd /workspace/repo && TRAJECTORY_DIR=/workspace/results SOURCE_REPO="{repo_name}" '
    'TEST_NAME_PREFIX="{test_name_prefix}" FRAMEWORK="{framework}" '
    'FRAMEWORK_LANGUAGE="{framework_language}" '
)
_PYTEST_SINGLE = _ENV_PREFIX + "xvfb-run -a python3 -m pytest <test_file> -v --timeout=60 2>&1 | head -100"
_PYTEST_SUITE = _ENV_PREFIX + "xvfb-run -a python3 -m pytest --timeout=120 -v 2>&1 | tail -50"
_PW_TS_SINGLE = _ENV_PREFIX + "xvfb-run -a npx playwright test <test_file> --reporter=list 2>&1 | head -100"
_PW_TS_SUITE = _ENV_PREFIX + "xvfb-run -a npx playwright test --reporter=list 2>&1 | tail -50"


@dataclass(frozen=True)
class FrameworkProfile:
    name: str  # registry key, e.g. "selenium-python"
    test_framework: str  # "playwright" | "selenium"
    language: str  # "python" | "typescript" | "javascript"
    install_commands: tuple  # shell commands, .format(repo_dir=...), run in order
    injection_kind: str  # "python_conftest" | "ts_fixture"
    instrumentation_filename: str  # file in instrumentation/ to inject
    test_command_single: str  # stage 5, .format(repo_name=, test_name_prefix=)
    test_command_suite: str  # stage 6, .format(repo_name=, test_name_prefix=)
    setup_note: str  # framework-specific setup hint for the agent prompt
    action_vocabulary: tuple  # method names that count as interactive actions
    sandbox_image: str  # Docker image used for stages 5/6
    sandbox_platform: str | None  # forced platform (e.g. "linux/amd64") or None for native
    test_globs: tuple  # filename globs to locate test files (stage 5 trial)


PROFILES: dict[str, FrameworkProfile] = {
    "playwright-python": FrameworkProfile(
        name="playwright-python",
        test_framework="playwright",
        language="python",
        install_commands=(
            "cd {repo_dir} && pip3 install -r requirements.txt --break-system-packages 2>&1 || true",
            "cd {repo_dir} && pip3 install pytest-playwright pytest-timeout --break-system-packages 2>&1",
            # Repos pin their own Playwright version, so fetch the matching browser instead of the image's baked one.
            "cd {repo_dir} && python3 -m playwright install chromium 2>&1 || true",
        ),
        injection_kind="python_conftest",
        instrumentation_filename="pw_python_conftest.py",
        test_globs=("test_*.py", "*_test.py"),
        test_command_single=_PYTEST_SINGLE,
        test_command_suite=_PYTEST_SUITE,
        setup_note="A conftest.py with instrumentation has been placed in the test directory; "
        "it sets the viewport itself, so no config changes are needed.",
        action_vocabulary=(
            "click",
            "fill",
            "type",
            "press",
            "selectOption",
            "select_option",
            "check",
            "uncheck",
            "dblclick",
            "dragTo",
            "drag_and_drop",
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
        test_globs=("*.spec.ts", "*.test.ts", "*.spec.js", "*.test.js"),
        test_command_single=_PW_TS_SINGLE,
        test_command_suite=_PW_TS_SUITE,
        setup_note="Instrumentation is already fully wired: every test file's `@playwright/test` import has been "
        "redirected to /workspace/repo/instrumented-fixture, which overrides the `page` fixture and locks the "
        "1280x720 viewport itself. Do NOT edit imports or playwright.config — just pick a promising test file "
        "and run it.",
        action_vocabulary=(
            "click",
            "fill",
            "type",
            "press",
            "selectOption",
            "select_option",
            "check",
            "uncheck",
            "dblclick",
            "dragTo",
            "drag_and_drop",
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
        test_globs=("test_*.py", "*_test.py"),
        test_command_single=_PYTEST_SINGLE,
        test_command_suite=_PYTEST_SUITE,
        setup_note="A conftest.py with Selenium instrumentation has been placed in the test directory. It locks the "
        "viewport to 1280x720 via CDP and captures coordinates from getBoundingClientRect, so no config changes are "
        "needed. Chrome and a matching chromedriver are already on PATH in the image.",
        action_vocabulary=(
            "click",
            "send_keys",
            "clear",
            "submit",
            "select_by_value",
            "select_by_visible_text",
            "select_by_index",
        ),
        sandbox_image=SANDBOX_IMAGE_SELENIUM,
        sandbox_platform="linux/amd64",
    ),
}

_DEFAULT_PROFILE = PROFILES["playwright-ts"]


def resolve_profile(test_framework: str | None, framework_language: str | None) -> FrameworkProfile:
    """The profile for a (test_framework, framework_language) pair, falling back to playwright-ts."""
    fw = (test_framework or "").strip().lower()
    lang = (framework_language or "").strip().lower()

    if fw == "selenium" and lang == "python":
        return PROFILES["selenium-python"]
    if lang == "python":
        return PROFILES["playwright-python"]
    return _DEFAULT_PROFILE


# The playwright-ts profile also handles JavaScript, so javascript is listed though its `language` is typescript.
SUPPORTED_COMBOS = frozenset(
    {
        ("playwright", "python"),
        ("playwright", "typescript"),
        ("playwright", "javascript"),
        ("selenium", "python"),
    }
)


def is_supported_profile(test_framework: str | None, framework_language: str | None) -> bool:
    """True only for a (framework, language) pair with a real capture hook, so stage 4 can reject the rest."""
    fw = (test_framework or "").strip().lower()
    lang = (framework_language or "").strip().lower()
    return (fw, lang) in SUPPORTED_COMBOS


def all_interactive_actions() -> set[str]:
    """Union of every profile's interactive action vocabulary, lowercased."""
    actions: set[str] = set()
    for profile in PROFILES.values():
        actions.update(a.lower() for a in profile.action_vocabulary)
    return actions
