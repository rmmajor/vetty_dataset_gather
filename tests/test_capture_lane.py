"""The shared capture lane: site filtering, the secret-leak invariant, resume bookkeeping and recording validity."""

import json
import tempfile
from pathlib import Path

from cua.pipeline.lanes.capture_lane import (
    ATTEMPTS_FILE,
    LEAK_EXEMPT_KEYS,
    SecretLeak,
    capture_attempts,
    is_capture_exhausted,
    is_capture_settled,
    is_valid_trajectory,
    needs_verification,
    record_capture_attempt,
    secret_leak_check,
)
from cua.pipeline.sites.registry import load_sites, resolve_named_sites
from cua.settings import CAPTURE

REGISTRY = {
    "_comment": "fixture",
    "sites": [
        {"name": "alpha", "start_url": "https://alpha.test/", "provisioning": "open_signup", "tasks": []},
        {"name": "beta", "start_url": "https://beta.test/", "provisioning": "phone", "tasks": []},
        {
            "name": "gamma",
            "start_url": "https://gamma.test/",
            "provisioning": "email_verify",
            "skip_capture": True,
            "skip_reason": "renders empty",
            "tasks": [],
        },
        {"name": "delta", "start_url": "https://delta.test/", "tasks": []},
    ],
}


def _registry_file(work: Path) -> Path:
    path = work / "public_sites_fixture.json"
    path.write_text(json.dumps(REGISTRY))
    return path


def _recording(traj_dir: Path, *, actions=None, screenshots=True, instruction="Do the thing") -> Path:
    traj_dir.mkdir(parents=True, exist_ok=True)
    steps = [{"step": 1, "screenshot": "step_001_screenshot.png", "thought": "go", "actions": actions or []}]
    traj_dir.joinpath("trajectory.json").write_text(
        json.dumps({"task_id": "t", "instruction": instruction, "start_url": "https://x.test/", "steps": steps})
    )
    if screenshots:
        traj_dir.joinpath("step_001_screenshot.png").write_bytes(b"png")
    return traj_dir


def test_load_sites_reproduces_the_public_lanes_skip_capture_filter():
    with tempfile.TemporaryDirectory() as tmp:
        config = _registry_file(Path(tmp))
        assert [s["name"] for s in load_sites(config)] == ["alpha", "beta", "delta"]
        assert [s["name"] for s in load_sites(config, include_skipped=True)] == ["alpha", "beta", "gamma", "delta"]


def test_load_sites_reproduces_the_login_gated_lanes_provisioning_filter():
    with tempfile.TemporaryDirectory() as tmp:
        config = _registry_file(Path(tmp))
        autonomous = load_sites(config, allow_provisioning={"open_signup", "email_verify"}, include_skipped=True)
        assert [s["name"] for s in autonomous] == ["alpha", "gamma"]
        # A site with no `provisioning` key is never in an allow-list, exactly as the old lane behaved.
        assert [s["name"] for s in load_sites(config, allow_provisioning={"phone"}, include_skipped=True)] == ["beta"]
        assert load_sites(config, allow_provisioning=set(), include_skipped=True) == []


def test_load_sites_with_explicit_names_bypasses_both_filters_and_raises_on_an_unknown_one():
    with tempfile.TemporaryDirectory() as tmp:
        config = _registry_file(Path(tmp))
        # `only` searches the given config first, then every shipped one, and ignores skip_capture.
        assert [s["name"] for s in load_sites(config, only=["gamma", "beta"])] == ["gamma", "beta"]
        assert [s["name"] for s in load_sites(config, only=["saucedemo"])] == ["saucedemo"]
        try:
            load_sites(config, only=["definitely-not-a-site"])
            raise AssertionError("an unknown site name must raise, never be silently dropped")
        except ValueError as e:
            assert "definitely-not-a-site" in str(e)
    assert [s["name"] for s in resolve_named_sites(["saucedemo"])] == ["saucedemo"]


def test_secret_leak_check_covers_every_secret_except_the_exempt_address():
    with tempfile.TemporaryDirectory() as tmp:
        clean = _recording(Path(tmp) / "clean", instruction="Sign in as <secret:email>")
        secret_leak_check(clean, {"email": "burner@mail.test", "password": "hunter2"})
        secret_leak_check(clean, None)
        secret_leak_check(clean, {})

        leaky = _recording(Path(tmp) / "leaky", instruction="Sign in with hunter2")
        try:
            secret_leak_check(leaky, {"email": "burner@mail.test", "password": "hunter2"})
            raise AssertionError("a literal password must raise")
        except SecretLeak as e:
            assert "password" in str(e)

        # The burner address is masked at TYPE time but the teacher names it in thought text, so it is exempt.
        assert "email" in LEAK_EXEMPT_KEYS
        email_only = _recording(Path(tmp) / "email", instruction="I signed up as burner@mail.test")
        secret_leak_check(email_only, {"email": "burner@mail.test", "password": "hunter2"})

        # Every other value the lanes mask is still checked, e.g. the reset lane's new_password.
        reset = _recording(Path(tmp) / "reset", instruction="My new password is Tr0ub4dor")
        try:
            secret_leak_check(reset, {"email": "burner@mail.test", "new_password": "Tr0ub4dor"})
            raise AssertionError("a literal non-exempt secret must raise")
        except SecretLeak as e:
            assert "new_password" in str(e)


def test_is_valid_trajectory_needs_real_coordinates_and_every_screenshot():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        assert not is_valid_trajectory(work / "missing")
        assert not is_valid_trajectory(_recording(work / "no_actions"))
        assert not is_valid_trajectory(_recording(work / "origin", actions=[{"type": "click", "x": 0, "y": 0}]))
        assert is_valid_trajectory(_recording(work / "good", actions=[{"type": "click", "x": 12, "y": 40}]))
        assert not is_valid_trajectory(
            _recording(work / "no_png", actions=[{"type": "click", "x": 12, "y": 40}], screenshots=False)
        )
        broken = work / "broken"
        broken.mkdir()
        broken.joinpath("trajectory.json").write_text("{not json")
        assert not is_valid_trajectory(broken)


def test_attempt_bookkeeping_round_trips_and_settles_a_failed_recording():
    with tempfile.TemporaryDirectory() as tmp:
        traj_dir = _recording(Path(tmp) / "site_x" / "task_y", actions=[{"type": "click", "x": 5, "y": 5}])
        assert capture_attempts(traj_dir) == 0
        assert record_capture_attempt(traj_dir, "failed") == 1
        assert json.loads((traj_dir / ATTEMPTS_FILE).read_text()) == {"attempts": 1, "last_outcome": "failed"}
        assert capture_attempts(traj_dir) == 1

        # No verifier verdict yet: an orphan resume must FINALIZE rather than re-capture, so it counts as settled.
        (traj_dir / "capture_meta.json").write_text(json.dumps({"task_meta": {}}))
        assert is_capture_settled(traj_dir)
        assert needs_verification(traj_dir)
        assert not is_capture_exhausted(traj_dir)

        (traj_dir / "capture_meta.json").write_text(json.dumps({"verifier": {"final_success": False}}))
        assert not needs_verification(traj_dir)
        assert not is_capture_settled(traj_dir)
        while capture_attempts(traj_dir) < CAPTURE.max_attempts:
            record_capture_attempt(traj_dir, "failed")
        assert is_capture_exhausted(traj_dir)
        assert is_capture_settled(traj_dir)


def test_a_verified_success_settles_regardless_of_attempts():
    with tempfile.TemporaryDirectory() as tmp:
        traj_dir = _recording(Path(tmp) / "task_ok", actions=[{"type": "click", "x": 5, "y": 5}])
        traj = json.loads((traj_dir / "trajectory.json").read_text())
        traj["success"] = True
        (traj_dir / "trajectory.json").write_text(json.dumps(traj))
        (traj_dir / "capture_meta.json").write_text(json.dumps({"verifier": {"final_success": True}}))
        assert is_capture_settled(traj_dir)
        assert not is_capture_exhausted(traj_dir)
