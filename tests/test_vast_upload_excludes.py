import subprocess

from cua.cloud.ignore import (
    ALWAYS_EXCLUDE,
    STATIC_UPLOAD_EXCLUDES,
    gitignored_excludes,
    modal_ignore_patterns,
)
from cua.cloud.vast import vastai
from cua.cloud.vast.base import Instance
from cua.cloud.vast.config import RemoteTrainConfig
from cua.cloud.vast.jobs import upload_package
from cua.cloud.vast.mock import MockProvider

LISTING = (
    "cua/__pycache__/settings.cpython-314.pyc\n"
    "cua/pipeline/__pycache__/sandbox.cpython-314.pyc\n"
    "cua/pipeline/sites/public_sites_human.json\n"
    "cua/pipeline/sites/public_sites_guarded.json\n"
)
SITE_PATTERNS = {"/pipeline/sites/public_sites_human.json", "/pipeline/sites/public_sites_guarded.json"}


class FakeCompleted:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, "", returncode


class FakeRunner:
    def __init__(self, stdout="", error=None):
        self.stdout, self.error, self.calls = stdout, error, []
        self.TimeoutExpired = subprocess.TimeoutExpired

    def run(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.error is not None:
            raise self.error
        return FakeCompleted(self.stdout)

    def __call__(self, argv, **kwargs):
        return self.run(argv, **kwargs)


def test_gitignored_excludes_anchors_paths_and_collapses_bytecode():
    runner = FakeRunner(LISTING)
    patterns = gitignored_excludes(run=runner)
    argv = runner.calls[0][0]
    assert argv == ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "cua/"]
    assert runner.calls[0][1] == {"capture_output": True, "text": True, "check": True, "timeout": 30}
    assert patterns[: len(ALWAYS_EXCLUDE)] == ALWAYS_EXCLUDE
    derived = patterns[len(ALWAYS_EXCLUDE) :]
    assert set(derived) == SITE_PATTERNS and len(derived) == len(set(derived))
    # The transfer root is the cua dir itself, so a derived pattern is anchored there, never repo-relative.
    assert all(p.startswith("/") and not p.startswith("/cua") for p in derived)
    assert not any("__pycache__" in p or p.endswith(".pyc") for p in derived)


def test_gitignored_excludes_falls_back_to_the_static_list_without_git():
    errors = (OSError("no git"), subprocess.CalledProcessError(128, "git"), subprocess.TimeoutExpired("git", 30))
    for error in errors:
        patterns = gitignored_excludes(run=FakeRunner(error=error))
        assert patterns == (*ALWAYS_EXCLUDE, *STATIC_UPLOAD_EXCLUDES)
        assert set(STATIC_UPLOAD_EXCLUDES) == SITE_PATTERNS
    # The registries are excluded even when git lists nothing: the human one is written after any listing.
    assert set(STATIC_UPLOAD_EXCLUDES) <= set(gitignored_excludes(run=FakeRunner("")))


def test_the_git_listing_is_bounded_by_a_timeout():
    runner = FakeRunner(LISTING)
    gitignored_excludes(run=runner)
    assert runner.calls[0][1]["timeout"] == 30


def test_modal_ignore_patterns_cover_both_site_registries_as_globs():
    patterns = modal_ignore_patterns(run=FakeRunner(LISTING))
    assert patterns[:2] == ["**/__pycache__", "**/*.pyc"]
    assert {"**/public_sites_human.json", "**/public_sites_guarded.json"} <= set(patterns)
    assert all(p.startswith("**/") and "/" not in p[3:] for p in patterns)
    # The image layer must be filtered even where git cannot answer, exactly like the rsync upload.
    assert {"**/public_sites_human.json", "**/public_sites_guarded.json"} <= set(
        modal_ignore_patterns(run=FakeRunner(error=OSError("no git")))
    )


def test_upload_package_hands_the_excludes_to_the_provider():
    cfg = RemoteTrainConfig()
    provider = MockProvider(cfg)
    instance = Instance(id="i-1", provider="mock", label=cfg.instance_label)
    upload_package(provider, instance, cfg)
    package = [u for u in provider.uploads if u["local"] == "cua/"]
    assert len(package) == 1 and package[0]["remote"] == f"{cfg.remote_workdir}/cua/"
    assert set(ALWAYS_EXCLUDE) <= set(package[0]["exclude"])
    others = [u for u in provider.uploads if u["local"] != "cua/"]
    assert others and all(u["exclude"] == () for u in others)


def test_rsync_argv_carries_one_exclude_flag_per_pattern():
    argv = vastai.rsync_argv("ssh -p 22", "cua/", "root@h:/workspace/cua/", ("__pycache__", "/pipeline/sites/x.json"))
    assert argv[:4] == ["rsync", "-az", "-e", "ssh -p 22"]
    assert argv[4:6] == ["--exclude=__pycache__", "--exclude=/pipeline/sites/x.json"]
    assert argv[-2:] == ["cua/", "root@h:/workspace/cua/"]
    assert "--exclude" not in vastai.rsync_argv("ssh", "a/", "root@h:/b/")


def test_vastai_upload_excludes_while_download_is_untouched():
    cfg = RemoteTrainConfig()
    provider = vastai.VastAIProvider(cfg)
    instance = Instance(id="1", provider="vastai", label=cfg.instance_label, ssh_host="h", ssh_port=2222)
    runner = FakeRunner()
    saved = vastai.subprocess
    vastai.subprocess = runner
    try:
        provider.upload(instance, "cua/", "/workspace/cua/", exclude=("__pycache__", "/x.json"))
        provider.upload(instance, "trajectories/", "/workspace/trajectories/")
        provider.download(instance, "/workspace/output/", "output/")
    finally:
        vastai.subprocess = saved
    package, corpus, pull = (call[0] for call in runner.calls)
    assert package[4:6] == ["--exclude=__pycache__", "--exclude=/x.json"]
    assert package[-2:] == ["cua/", "root@h:/workspace/cua/"]
    assert "--exclude=__pycache__" not in corpus and corpus[-2:] == ["trajectories/", "root@h:/workspace/trajectories/"]
    assert not [a for a in pull if a.startswith("--exclude")]
    assert pull[-2:] == ["root@h:/workspace/output/", "output/"]
