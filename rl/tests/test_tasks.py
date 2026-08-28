import json
import tempfile
from pathlib import Path

from rl.tasks import build_suite, check_suite, is_idempotent, select_tasks, split_for

SRC = {"path": None, "backend": "public_site", "env": "live", "lane": "autonomous", "auth_mode": "none"}


def _sites(n_sites=6, tasks_per_site=3):
    return {"sites": [{"name": f"site{i}", "start_url": f"https://site{i}.example/", "domain": "d", "tasks": [
        {"id": f"t{j}", "instruction": f"Open the settings page and change the theme to dark {i}-{j}", "difficulty": "hard",
         "expected_actions": 12, "action_kinds": ["click", "scroll"], "auth_required": False} for j in range(tasks_per_site)]}
        for i in range(n_sites)]}


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
        seen = _write(tmp, "seen.json", {"sites": [{"name": "site1", "tasks": [{"id": "x", "instruction": sites["sites"][1]["tasks"][0]["instruction"]}]}]})
        traj = Path(tmp) / "trajectories" / "site_site2" / "task_t1"
        traj.mkdir(parents=True)
        (traj / "trajectory.json").write_text(json.dumps({"instruction": "irrelevant", "success": True, "steps": []}))
        (traj / "metadata.json").write_text(json.dumps({"capture_source": "agent"}))
        failed = Path(tmp) / "trajectories" / "site_site3" / "task_t2"
        failed.mkdir(parents=True)
        (failed / "trajectory.json").write_text(json.dumps({"instruction": "irrelevant", "success": False, "steps": []}))
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
