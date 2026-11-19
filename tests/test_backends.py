from cua.rl.backends import BACKENDS, StaticPageBackend, WorkerPool, resolve_preconditions
from cua.settings import RL


def test_pool_passes_the_loop_guard_to_every_driver():
    pool = WorkerPool(
        StaticPageBackend("<html></html>"), requested_workers=1, drivers_per_sandbox=2, loop_guard_repeats=7
    )
    assert pool.loop_guard_repeats == 7 and all(
        w.driver_env == {"DRIVER_LOOP_GUARD_REPEATS": "7"} for w in pool.workers
    )
    assert (
        WorkerPool(StaticPageBackend("<html></html>"), requested_workers=1).loop_guard_repeats == RL.loop_guard_repeats
    )


def test_no_session_needed_is_empty_kwargs():
    assert resolve_preconditions({"task_id": "a", "preconditions": {}}) == {}


def test_unregistered_burner_fails_closed():
    task = {
        "task_id": "a",
        "lane": "autonomous",
        "preconditions": {"requires_session": True, "burner_account_id": "nope", "session_ref": "nope"},
    }
    assert resolve_preconditions(task) is None


def test_static_backend_serves_local_url_and_needs_no_auth():
    backend = StaticPageBackend("<html></html>")
    assert backend.start_url({"start_url": "ignored"}) == "http://127.0.0.1:3000/"
    assert backend.session_kwargs({"preconditions": {"requires_session": True}}) == {}
    assert set(BACKENDS) == {"public_site", "static_page"}
