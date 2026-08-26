from rl.rewards import PENALTIES, REWARD_COMPONENTS, REWARD_MAX, compute_reward, load_reward_config

CONFIG = load_reward_config()
TASK = {"lane": "autonomous", "expected_actions": 10}


def _record(**kw):
    base = {"success": True, "turns": [{"clean": True}] * 4, "ui_steps": 8, "terminal_reason": "done", "navigate_blocked": 0}
    return {**base, **kw}


def test_config_has_both_lanes_and_all_names():
    for lane in ("autonomous", "guarded"):
        assert set(CONFIG["lanes"][lane]["weights"]) == set(REWARD_COMPONENTS)
        assert set(CONFIG["lanes"][lane]["penalties"]) == set(PENALTIES)


def test_perfect_episode_scores_full_weights():
    reward, breakdown = compute_reward(_record(), TASK, CONFIG)
    w = CONFIG["lanes"]["autonomous"]["weights"]
    assert reward == round(sum(w.values()), 4) and reward <= REWARD_MAX
    assert breakdown["components"] == {"terminal_success": 1.0, "format_validity": 1.0, "efficiency": 1.0, "terminated_with_done": 1.0}


def test_failed_parse_abort_is_negative():
    reward, breakdown = compute_reward(_record(success=False, terminal_reason="parse_abort", turns=[{"clean": False}] * 3), TASK, CONFIG)
    assert reward < 0 and breakdown["penalties"]["parse_abort"] == 1.0


def test_efficiency_needs_success_and_caps_at_one():
    assert REWARD_COMPONENTS["efficiency"](_record(ui_steps=20), TASK) == 0.5
    assert REWARD_COMPONENTS["efficiency"](_record(success=False), TASK) == 0.0
    assert REWARD_COMPONENTS["efficiency"](_record(ui_steps=3), TASK) == 1.0


def test_unknown_success_counts_as_zero():
    reward, _ = compute_reward(_record(success=None, terminal_reason="step_cap"), TASK, CONFIG)
    assert reward < CONFIG["lanes"]["autonomous"]["weights"]["terminal_success"]
