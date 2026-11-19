"""Run-level metrics: the success denominator, the grammar counters and the per-level / per-domain breakdowns."""

from cua.eval.student.config import UNSCORED_OUTCOMES, EpisodeRecord


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _mean(values: list) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _is_scored(record: EpisodeRecord) -> bool:
    """An episode counts toward the success rate only if it both ran and was judged."""
    return record.outcome not in UNSCORED_OUTCOMES and not record.judge_deferred


def _group_metrics(records: list[EpisodeRecord], key) -> dict:
    """Success / blocking / step-efficiency broken out by any grouping key (level, domain)."""
    groups: dict = {}
    for r in records:
        bucket = groups.setdefault(
            key(r), {"n": 0, "scored": 0, "success": 0, "blocked": 0, "_success_ratios": [], "_success_steps": []}
        )
        bucket["n"] += 1
        if r.outcome == "site_blocked":
            bucket["blocked"] += 1
        if _is_scored(r):
            bucket["scored"] += 1
            bucket["success"] += int(r.verified_success)
            if r.verified_success:
                bucket["_success_steps"].append(r.model_steps)
                if r.reference_length:
                    bucket["_success_ratios"].append(r.model_steps / r.reference_length)
    for bucket in groups.values():
        bucket["success_rate"] = _rate(bucket["success"], bucket["scored"])
        bucket["blocked_rate"] = _rate(bucket["blocked"], bucket["n"])
        bucket["mean_steps_on_success"] = _mean(bucket.pop("_success_steps"))
        bucket["mean_steps_over_reference"] = _mean(bucket.pop("_success_ratios"))
    return groups


def _success_metrics(records: list[EpisodeRecord]) -> dict:
    """Both denominators: environment failures leave the scored one, so bot-blocking never reads as a model failure."""
    scored = [r for r in records if _is_scored(r)]
    successes = [r for r in scored if r.verified_success]
    blocked = [r for r in records if r.outcome == "site_blocked"]
    return {
        "episodes": len(records),
        "scored_episodes": len(scored),
        "judge_deferred_episodes": sum(1 for r in records if r.judge_deferred),
        "successes": len(successes),
        "verified_success_rate": _rate(len(successes), len(scored)),
        "verified_success_rate_all_episodes": _rate(len(successes), len(records)),
        "efficient_success_rate": _rate(sum(1 for r in successes if r.efficiency != "meandering"), len(scored)),
        "site_blocked_rate": _rate(len(blocked), len(records)),
    }


def _grammar_metrics(records: list[EpisodeRecord]) -> dict:
    """How the model spelled its actions, reported instead of charged to its success rate."""
    model_steps = sum(r.model_steps for r in records)
    actions_parsed = sum(r.actions_parsed for r in records)
    actions_canonical = sum(r.actions_canonical for r in records)
    form_deviations: dict = {}
    for record in records:
        for deviation, count in (record.form_deviations or {}).items():
            form_deviations[deviation] = form_deviations.get(deviation, 0) + count
    return {
        "parse_failure_rate": _rate(sum(r.parse_failure_steps for r in records), model_steps),
        "malformed_action_lines": sum(r.malformed_lines for r in records),
        "actions_parsed": actions_parsed,
        "actions_canonical": actions_canonical,
        "canonical_form_rate": _rate(actions_canonical, actions_parsed),
        "action_form_deviations": dict(sorted(form_deviations.items(), key=lambda kv: -kv[1])),
    }


def _behaviour_metrics(records: list[EpisodeRecord]) -> dict:
    """Terminal and reasoning behaviour: whether the model declares done, navigates, hands off and thinks."""
    return {
        "done_rate": _rate(sum(1 for r in records if r.done_emitted), len(records)),
        "navigate_actions": sum(r.navigations for r in records),
        "handoff_episodes": sum(1 for r in records if r.handoff_kind),
        "think_rate": _rate(sum(r.think_steps for r in records), sum(r.model_steps for r in records)),
    }


def _driver_metrics(records: list[EpisodeRecord]) -> dict:
    """What the driver refused and how much of that was the declared coordinate space being wrong."""
    emitted = sum(r.emitted_coords for r in records)
    out_of_space = sum(r.emitted_out_of_viewport for r in records)
    return {
        "driver_error_rate": _rate(
            sum(r.driver_errors for r in records), sum(r.actions_executed + r.driver_errors for r in records)
        ),
        "out_of_bounds_actions": sum(r.out_of_bounds for r in records),
        "emitted_coords": emitted,
        "emitted_out_of_viewport": out_of_space,
        "emitted_out_of_viewport_rate": _rate(out_of_space, emitted),
        "stuck_episodes": sum(1 for r in records if r.outcome == "stuck"),
    }


def _efficiency_metrics(records: list[EpisodeRecord]) -> dict:
    """Step efficiency against the benchmark's own human step count, a more defensible denominator than raw steps."""
    successes = [r for r in records if _is_scored(r) and r.verified_success]
    model_steps = sum(r.model_steps for r in records)
    return {
        "mean_steps_on_success": _mean([r.model_steps for r in successes]),
        "mean_steps_over_reference_on_success": _mean(
            [r.model_steps / r.reference_length for r in successes if r.reference_length]
        ),
        "mean_steps_all": round(model_steps / len(records), 2) if records else 0.0,
        "step_cap_hits": sum(1 for r in records if r.outcome == "step_cap"),
        "total_model_steps": model_steps,
    }


def _breakdown_metrics(records: list[EpisodeRecord]) -> dict:
    outcomes: dict = {}
    for r in records:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
    return {
        "outcomes": outcomes,
        "by_level": _group_metrics(records, lambda r: r.level or "ungraded"),
        "by_domain": _group_metrics(records, lambda r: r.domain or r.website or "unknown"),
    }


def compute_metrics(records: list[EpisodeRecord]) -> dict:
    return {
        **_success_metrics(records),
        **_grammar_metrics(records),
        **_behaviour_metrics(records),
        **_driver_metrics(records),
        **_efficiency_metrics(records),
        **_breakdown_metrics(records),
    }
