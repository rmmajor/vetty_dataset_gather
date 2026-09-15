import json
import tempfile
from pathlib import Path

from cua.rl.tasks import build_candidates, build_suite, check_suite, is_idempotent, select_tasks, split_for

SRC = {"path": None, "backend": "public_site", "env": "live", "lane": "autonomous", "auth_mode": "none"}


def _sites(n_sites=6, tasks_per_site=3):
    return {
        "sites": [
            {
                "name": f"site{i}",
                "start_url": f"https://site{i}.example/",
                "domain": "d",
                "tasks": [
                    {
                        "id": f"t{j}",
                        "instruction": f"Open the settings page and change the theme to dark {i}-{j}",
                        "difficulty": "hard",
                        "expected_actions": 12,
                        "action_kinds": ["click", "scroll"],
                        "auth_required": False,
                    }
                    for j in range(tasks_per_site)
                ],
            }
            for i in range(n_sites)
        ]
    }


def _write(tmp, name, data):
    p = Path(tmp) / name
    p.write_text(json.dumps(data))
    return p


def test_idempotent_heuristic():
    assert is_idempotent({"instruction": "Sort the table by price and open the first row", "action_kinds": ["click"]})
    assert not is_idempotent({"instruction": "Publish the draft post", "action_kinds": ["click"]})
    assert not is_idempotent({"instruction": "Fill the form", "action_kinds": ["type"], "requires_upload": True})
    assert is_idempotent({"instruction": "Delete everything", "action_kinds": ["click"], "idempotent": True})


def test_split_is_deterministic_and_covers_labels():
    assert split_for("site0", 0, True, 100) == "eval_unseen_site"
    assert split_for("site0", 0, True, 0) == "eval_seen_site_unseen_task"
    assert split_for("site0", 1, True, 0) == "train"
    assert split_for("site0", 1, False, 0) == "eval_mutating"


def test_build_excludes_trained_tasks_and_checks_disjointness():
    with tempfile.TemporaryDirectory() as tmp:
        sites = _sites()
        src = {**SRC, "path": _write(tmp, "src.json", sites)}
        seen = _write(
            tmp,
            "seen.json",
            {
                "sites": [
                    {
                        "name": "site1",
                        "tasks": [{"id": "x", "instruction": sites["sites"][1]["tasks"][0]["instruction"]}],
                    }
                ]
            },
        )
        traj = Path(tmp) / "trajectories" / "site_site2" / "task_t1"
        traj.mkdir(parents=True)
        (traj / "trajectory.json").write_text(json.dumps({"instruction": "irrelevant", "success": True, "steps": []}))
        (traj / "metadata.json").write_text(json.dumps({"capture_source": "agent"}))
        failed = Path(tmp) / "trajectories" / "site_site3" / "task_t2"
        failed.mkdir(parents=True)
        (failed / "trajectory.json").write_text(
            json.dumps({"instruction": "irrelevant", "success": False, "steps": []})
        )
        (failed / "metadata.json").write_text(json.dumps({"capture_source": "agent"}))
        suite = build_suite([src], Path(tmp) / "trajectories", seen, unseen_site_pct=0)
        ids = {t["task_id"] for t in suite["tasks"]}
        assert "site1__t0" not in ids and "site2__t1" not in ids and "site0__t0" in ids
        assert "site3__t2" in ids and [t for t in suite["tasks"] if t["task_id"] == "site3__t2"][0]["split"] == "train"
        assert suite["excluded_trained"] == 2
        assert len(suite["suite_sha256"]) == 64 and check_suite(suite, Path(tmp) / "trajectories", seen) == []
        train = select_tasks(suite, split="train")
        assert train and all(t["split"] == "train" and t["idempotent"] for t in train)
        assert select_tasks(suite, task_ids=["site0__t1"])[0]["allowed_hosts"] == ["site0.example"]


def test_build_candidates_merges_forces_train_and_excludes():
    with tempfile.TemporaryDirectory() as tmp:
        frontier = {
            "_config": {"exclude_sites": ["banned"], "exclude_domains": ["evil.example"]},
            "sites": [
                {
                    "name": "toolsite",
                    "start_url": "https://toolsite.example/",
                    "domain": "d",
                    "tasks": [
                        {
                            "id": "f1",
                            "instruction": "Convert 5 miles to km ending with the value shown",
                            "expected_actions": 5,
                            "idempotent": True,
                            "success_check": {"text_visible": "8.04"},
                            "milestones": [{"name": "entered", "check": {"text_visible": "5"}, "weight": 1.0}],
                        },
                        {
                            "id": "f2",
                            "instruction": "Look up the weekday of 2000-01-01 ending with it shown",
                            "expected_actions": 6,
                            "idempotent": True,
                            "success_check": {"text_visible": "Saturday"},
                        },
                    ],
                },
                {
                    "name": "banned",
                    "start_url": "https://banned.example/",
                    "domain": "d",
                    "tasks": [
                        {
                            "id": "x",
                            "instruction": "Open the page ending shown",
                            "expected_actions": 4,
                            "idempotent": True,
                            "success_check": {"url_contains": "p"},
                        }
                    ],
                },
                {
                    "name": "evilfront",
                    "start_url": "https://www.evil.example/app",
                    "domain": "d",
                    "tasks": [
                        {
                            "id": "y",
                            "instruction": "Open the app page ending shown",
                            "expected_actions": 4,
                            "idempotent": True,
                            "success_check": {"url_contains": "app"},
                        }
                    ],
                },
            ],
        }
        frontier_path = _write(tmp, "frontier.json", frontier)
        base = {
            "suite_sha256": "s",
            "tasks": [
                {
                    "task_id": "old__short",
                    "site": "old",
                    "start_url": "https://old.example/",
                    "instruction": "Do the short one until done",
                    "difficulty": "medium",
                    "expected_actions": 10,
                    "action_kinds": [],
                    "backend": "public_site",
                    "env": "live",
                    "lane": "autonomous",
                    "auth_mode": "none",
                    "idempotent": True,
                    "split": "train",
                    "preconditions": {},
                    "allowed_hosts": ["old.example"],
                    "success_check": {"url_contains": "done"},
                    "milestones": None,
                    "throttle": {"min_interval_s": 30},
                    "source": "b",
                },
                {
                    "task_id": "old__unoracled",
                    "site": "old",
                    "start_url": "https://old.example/",
                    "instruction": "Do the unoracled one until done",
                    "difficulty": "medium",
                    "expected_actions": 8,
                    "action_kinds": [],
                    "backend": "public_site",
                    "env": "live",
                    "lane": "autonomous",
                    "auth_mode": "none",
                    "idempotent": True,
                    "split": "train",
                    "preconditions": {},
                    "allowed_hosts": ["old.example"],
                    "success_check": None,
                    "milestones": None,
                    "throttle": {"min_interval_s": 30},
                    "source": "b",
                },
                {
                    "task_id": "old__long",
                    "site": "old",
                    "start_url": "https://old.example/",
                    "instruction": "Do the long one until done",
                    "difficulty": "hard",
                    "expected_actions": 30,
                    "action_kinds": [],
                    "backend": "public_site",
                    "env": "live",
                    "lane": "autonomous",
                    "auth_mode": "none",
                    "idempotent": True,
                    "split": "train",
                    "preconditions": {},
                    "allowed_hosts": ["old.example"],
                    "success_check": None,
                    "milestones": None,
                    "throttle": {"min_interval_s": 30},
                    "source": "b",
                },
            ],
        }
        base_path = _write(tmp, "base.json", base)
        suite = build_candidates(frontier_path, base_path, Path(tmp) / "none", None, max_expected=12)
        ids = [t["task_id"] for t in suite["tasks"]]
        assert ids == ["old__short", "toolsite__f1", "toolsite__f2"]
        assert suite["dropped_unoracled"] == ["old__unoracled"]
        assert all(t["split"] == "train" for t in suite["tasks"])
        by_id = {t["task_id"]: t for t in suite["tasks"]}
        assert by_id["toolsite__f1"]["milestones"][0]["name"] == "entered"
        assert not check_suite(suite, Path(tmp) / "none", None)
