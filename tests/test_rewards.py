from cua.rl.rewards import CAP_REASONS, PENALTIES, REWARD_COMPONENTS, REWARD_MAX, compute_reward, load_reward_config

CONFIG = load_reward_config()
LANE = CONFIG["lanes"]["autonomous"]
TASK = {"lane": "autonomous", "expected_actions": 10}


def _record(**kw):
    base = {
        "success": True,
        "turns": [{"clean": True}] * 4,
        "ui_steps": 8,
        "terminal_reason": "done",
        "navigate_blocked": 0,
        "efficiency_grade": "efficient",
    }
    return {**base, **kw}


def _reward(**kw):
    return compute_reward(_record(**kw), TASK, CONFIG)[0]


def test_config_has_both_lanes_and_all_names():
    for lane in ("autonomous", "guarded"):
        assert set(CONFIG["lanes"][lane]["weights"]) == set(REWARD_COMPONENTS)
        assert set(CONFIG["lanes"][lane]["penalties"]) == set(PENALTIES)
        assert set(CONFIG["lanes"][lane]["success_by_efficiency_grade"]) == {"efficient", "acceptable", "meandering"}


def test_perfect_episode_scores_full_weights():
    reward, breakdown = compute_reward(_record(), TASK, CONFIG)
    payable = sum(weight for name, weight in LANE["weights"].items() if name != "progress")
    assert reward == round(payable, 4) and reward <= REWARD_MAX
    assert breakdown["components"] == {
        "terminal_success": 1.0,
        "format_validity": 1.0,
        "efficiency": 1.0,
        "terminated_with_done": 1.0,
        "progress": 0.0,
    }


def test_failed_parse_abort_is_negative():
    reward, breakdown = compute_reward(
        _record(success=False, terminal_reason="parse_abort", turns=[{"clean": False}] * 3), TASK, CONFIG
    )
    assert reward < 0 and breakdown["penalties"]["parse_abort"] == 1.0


def test_efficiency_slopes_to_zero_at_twice_expected_and_needs_success():
    efficiency = REWARD_COMPONENTS["efficiency"]
    assert efficiency(_record(ui_steps=10), TASK, LANE) == 1.0 and efficiency(_record(ui_steps=3), TASK, LANE) == 1.0
    assert efficiency(_record(ui_steps=15), TASK, LANE) == 0.5 and efficiency(_record(ui_steps=20), TASK, LANE) == 0.0
    assert efficiency(_record(success=False), TASK, LANE) == 0.0


def test_meandering_success_is_discounted_not_vetoed():
    meandering = _reward(efficiency_grade="meandering", ui_steps=22)
    assert 0 < meandering < _reward(efficiency_grade="acceptable", ui_steps=15) < _reward()
    assert meandering > _reward(success=False, terminal_reason="step_cap", ui_steps=28)
    assert REWARD_COMPONENTS["terminal_success"](_record(efficiency_grade=None), TASK, LANE) == 1.0


def test_under_stopping_and_false_done_are_both_penalised():
    capped_success = _reward(terminal_reason="max_turns", ui_steps=28, efficiency_grade="meandering")
    assert capped_success < _reward(efficiency_grade="meandering", ui_steps=22)
    assert "max_turns" in CAP_REASONS and PENALTIES["step_cap"](_record(terminal_reason="max_turns"), TASK, LANE) == 1.0
    assert (
        PENALTIES["false_done"](_record(success=False), TASK, LANE) == 1.0
        and PENALTIES["false_done"](_record(), TASK, LANE) == 0.0
    )
    assert _reward(success=False) >= _reward(success=False, terminal_reason="step_cap")
    assert REWARD_COMPONENTS["terminated_with_done"](_record(success=False), TASK, LANE) == 0.0


def test_unknown_success_counts_as_zero():
    reward, _ = compute_reward(_record(success=None, terminal_reason="step_cap"), TASK, CONFIG)
    assert reward < LANE["weights"]["terminal_success"]


MILESTONE_TASK = {
    **TASK,
    "milestones": [
        {"name": "reached", "check": {"url_contains": "x"}, "weight": 1.0},
        {"name": "filled", "check": {"text_visible": "y"}, "weight": 1.0},
    ],
}


def test_milestones_order_failed_groups():
    none_hit = compute_reward(
        _record(success=False, terminal_reason="step_cap", milestones={"reached": False, "filled": False}),
        MILESTONE_TASK,
        CONFIG,
    )[0]
    one_hit = compute_reward(
        _record(success=False, terminal_reason="step_cap", milestones={"reached": True, "filled": False}),
        MILESTONE_TASK,
        CONFIG,
    )[0]
    both_hit = compute_reward(
        _record(success=False, terminal_reason="step_cap", milestones={"reached": True, "filled": True}),
        MILESTONE_TASK,
        CONFIG,
    )[0]
    assert none_hit < one_hit < both_hit


def test_milestones_never_pay_on_success():
    plain = compute_reward(_record(), MILESTONE_TASK, CONFIG)[0]
    with_hits = compute_reward(_record(milestones={"reached": True, "filled": True}), MILESTONE_TASK, CONFIG)[0]
    assert plain == with_hits


def test_running_to_the_cap_is_the_worst_failure_so_stopping_is_never_discouraged():
    """Cheapest-failure-is-the-cap taught the policy never to terminate; the validation run showed it."""
    capped = _reward(success=False, terminal_reason="step_cap")
    looped = _reward(success=False, terminal_reason="loop_guard")
    falsely_done = _reward(success=False, terminal_reason="done")
    assert looped > falsely_done > capped
