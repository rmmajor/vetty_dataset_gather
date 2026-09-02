"""Programmatic success checks evaluated against the live page at episode end (registry keyed by check name)."""
import re

_ELEMENT_RE = re.compile(r'^\[(\d+)\] (\S+) "([^"]*)"[ \t]*(.*)$', re.MULTILINE)


def build_context(snapshot, storage_state_loader, record, page_text=""):
    snapshot_text = snapshot.get("snapshot", "")
    elements = [{"ref": int(m.group(1)), "role": m.group(2), "name": m.group(3), "extra": m.group(4).strip()}
                for m in _ELEMENT_RE.finditer(snapshot_text)]
    # text_visible must see a computed answer wherever it rendered, so it searches the page's own text too.
    return {"url": snapshot.get("url", ""), "title": snapshot.get("title", ""),
            "text": f"{snapshot_text}\n{page_text}" if page_text else snapshot_text, "elements": elements,
            # Rendered text only: ctx["text"] also carries element values, which an agent can satisfy by TYPING the answer.
            "page_text": page_text,
            "storage_state": storage_state_loader, "terminal_reason": record.get("terminal_reason", "")}


def _element_matches(element, spec):
    role_ok = not spec.get("role") or element["role"] == spec["role"]
    name_ok = not spec.get("name_regex") or re.search(spec["name_regex"], element["name"], re.I)
    return bool(role_ok and name_ok)


def _cookie_present(spec, ctx):
    state = ctx["storage_state"]() or {}
    return any(re.search(spec, c.get("name", "")) for c in state.get("cookies", []))


CHECKS = {
    "url_matches": lambda spec, ctx: bool(re.search(spec, ctx["url"])),
    "url_contains": lambda spec, ctx: spec in ctx["url"],
    "title_contains": lambda spec, ctx: spec.lower() in ctx["title"].lower(),
    "text_visible": lambda spec, ctx: spec.lower() in ctx["text"].lower(),
    "page_text_visible": lambda spec, ctx: spec.lower() in ctx.get("page_text", "").lower(),
    "text_absent": lambda spec, ctx: spec.lower() not in ctx["text"].lower(),
    "element_present": lambda spec, ctx: any(_element_matches(e, spec) for e in ctx["elements"]),
    "element_absent": lambda spec, ctx: not any(_element_matches(e, spec) for e in ctx["elements"]),
    "element_value": lambda spec, ctx: any(_element_matches(e, spec) and f'value="{spec["equals"]}"' in e["extra"] for e in ctx["elements"]),
    "cookie_present": _cookie_present,
    "done_emitted": lambda spec, ctx: (ctx["terminal_reason"] == "done") == bool(spec),
    "terminal_reason_in": lambda spec, ctx: ctx["terminal_reason"] in spec,
    "all_of": lambda spec, ctx: all(evaluate(s, ctx)["passed"] for s in spec),
    "any_of": lambda spec, ctx: any(evaluate(s, ctx)["passed"] for s in spec),
}


def evaluate(spec, ctx):
    """spec = {check_name: arg, ...}; several keys are an implicit all_of."""
    results = {}
    for name, arg in spec.items():
        check = CHECKS.get(name)
        results[name] = bool(check(arg, ctx)) if check else False
    return {"passed": bool(results) and all(results.values()), "results": results}
