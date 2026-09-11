from rl.eval import metrics_by_split, per_domain, render_report

RECORDS = [
    {
        "split": "eval_unseen_site",
        "site": "a",
        "success": True,
        "terminal_reason": "done",
        "turns": [{"clean": True}, {"clean": True}],
        "steps": 2,
        "check": {"passed": True},
        "reward_status": "final",
    },
    {
        "split": "eval_unseen_site",
        "site": "b",
        "success": False,
        "terminal_reason": "parse_abort",
        "turns": [{"clean": False}, {"clean": False}],
        "steps": 0,
        "check": None,
        "reward_status": "final",
    },
    {
        "split": "train",
        "site": "a",
        "success": True,
        "terminal_reason": "done",
        "turns": [{"clean": True}],
        "steps": 4,
        "check": None,
        "reward_status": "final",
    },
]


def test_metrics_by_split():
    m = metrics_by_split(RECORDS)["eval_unseen_site"]
    assert (
        m["n"] == 2 and m["verified_success"] == 0.5 and m["parse_failure_rate"] == 0.5 and m["parse_abort_rate"] == 0.5
    )
    assert m["done_rate"] == 0.5 and m["mean_steps_success"] == 2.0 and m["check_pass_rate"] == 1.0


def test_per_domain_and_report_with_baseline_delta():
    domains = per_domain(RECORDS)
    assert domains["a"] == {"n": 2, "success_rate": 1.0}
    report = render_report(
        {"run_id": "r", "model": "m"},
        metrics_by_split(RECORDS),
        domains,
        baseline={"metrics": {"eval_unseen_site": {"verified_success": 0.25}}},
    )
    assert "| eval_unseen_site |" in report and "+0.25" in report and "run_id: r" in report
