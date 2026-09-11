from rl.checks import CHECKS, build_context, evaluate

SNAPSHOT = {
    "url": "https://example.com/cart?id=7",
    "title": "Your Cart — Example",
    "snapshot": '## Page text\nOrder confirmed. Thanks!\n## Interactive elements (in viewport)\n[1] button "Checkout"\n[2] textbox "Coupon" value="SAVE10"\n[3] checkbox "Gift wrap" (checked)\n',
}


def _ctx(terminal="done"):
    return build_context(SNAPSHOT, lambda: {"cookies": [{"name": "session_id"}]}, {"terminal_reason": terminal})


def test_context_parses_elements():
    ctx = _ctx()
    assert ctx["elements"][1] == {"ref": 2, "role": "textbox", "name": "Coupon", "extra": 'value="SAVE10"'}


def test_each_check():
    ctx = _ctx()
    assert evaluate({"url_matches": r"/cart\?id=\d+"}, ctx)["passed"]
    assert evaluate({"url_contains": "/cart"}, ctx)["passed"]
    assert evaluate({"title_contains": "your cart"}, ctx)["passed"]
    assert evaluate({"text_visible": "order confirmed"}, ctx)["passed"]
    assert not evaluate({"text_absent": "order confirmed"}, ctx)["passed"]
    assert evaluate({"element_present": {"role": "button", "name_regex": "^check"}}, ctx)["passed"]
    assert evaluate({"element_absent": {"name_regex": "Delete"}}, ctx)["passed"]
    assert evaluate({"element_value": {"name_regex": "Coupon", "equals": "SAVE10"}}, ctx)["passed"]
    assert evaluate({"cookie_present": "^session"}, ctx)["passed"]
    assert (
        evaluate({"done_emitted": True}, ctx)["passed"]
        and not evaluate({"done_emitted": True}, _ctx("step_cap"))["passed"]
    )
    assert evaluate({"terminal_reason_in": ["done", "handoff"]}, ctx)["passed"]


def test_composition_and_unknown_check():
    ctx = _ctx()
    spec = {"all_of": [{"text_visible": "thanks"}, {"any_of": [{"url_contains": "nope"}, {"url_contains": "cart"}]}]}
    assert evaluate(spec, ctx) == {"passed": True, "results": {"all_of": True}}
    assert evaluate({"bogus": 1}, ctx) == {"passed": False, "results": {"bogus": False}}
    assert set(CHECKS) >= {"url_matches", "text_visible", "element_present", "all_of", "any_of"}


def test_page_text_is_searchable_by_text_visible():
    ctx = build_context(SNAPSHOT, lambda: {}, {"terminal_reason": "done"}, "BMI = 22.9 kg/m2")
    assert CHECKS["text_visible"]("22.9", ctx) and not CHECKS["text_visible"]("31.4", ctx)
    snapshot_only = build_context(SNAPSHOT, lambda: {}, {"terminal_reason": "done"})
    assert not CHECKS["text_visible"]("22.9", snapshot_only)


def test_page_text_visible_cannot_be_satisfied_by_typing_the_answer():
    typed = build_context(
        {"snapshot": '[1] textbox "Search" value="Quoth the Raven"', "url": "https://x/search", "title": "Search"},
        lambda: {},
        {"terminal_reason": "done"},
        "Search results for your query",
    )
    assert CHECKS["text_visible"]("Quoth the Raven", typed)
    assert not CHECKS["page_text_visible"]("Quoth the Raven", typed)
    rendered = build_context(
        {"snapshot": '[1] link "Home"', "url": "https://x/poem", "title": "The Raven"},
        lambda: {},
        {"terminal_reason": "done"},
        "Quoth the Raven, Nevermore",
    )
    assert CHECKS["page_text_visible"]("Quoth the Raven", rendered)
