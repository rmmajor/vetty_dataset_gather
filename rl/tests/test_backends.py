from rl.backends import BACKENDS, StaticPageBackend, resolve_preconditions


def test_no_session_needed_is_empty_kwargs():
    assert resolve_preconditions({"task_id": "a", "preconditions": {}}) == {}


def test_unregistered_burner_fails_closed():
    task = {"task_id": "a", "lane": "autonomous", "preconditions": {"requires_session": True, "burner_account_id": "nope", "session_ref": "nope"}}
    assert resolve_preconditions(task) is None


def test_static_backend_serves_local_url_and_needs_no_auth():
    backend = StaticPageBackend("<html></html>")
    assert backend.start_url({"start_url": "ignored"}) == "http://127.0.0.1:3000/"
    assert backend.session_kwargs({"preconditions": {"requires_session": True}}) == {}
    assert set(BACKENDS) == {"public_site", "static_page"}
