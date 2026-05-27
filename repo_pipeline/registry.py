import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from repo_pipeline.schemas import (
    STAGE_TO_INPUT_STATUS,
    STAGE_TO_PASSED_STATUS,
    STAGE_TO_REJECTED_STATUS,
    RepoStatus,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS repos (
    full_name       TEXT PRIMARY KEY,
    html_url        TEXT,
    stars           INTEGER DEFAULT 0,
    last_pushed     TEXT,
    size_kb         INTEGER DEFAULT 0,
    language        TEXT,
    archived        BOOLEAN DEFAULT 0,
    fork            BOOLEAN DEFAULT 0,
    parent_repo     TEXT,
    description     TEXT,
    topics          TEXT DEFAULT '[]',
    default_branch  TEXT DEFAULT 'main',

    status          TEXT NOT NULL,
    current_stage   INTEGER DEFAULT 0,
    added_at        TEXT,
    updated_at      TEXT,
    source_query    TEXT,

    filter_reason   TEXT,

    shallow_scores  TEXT,
    shallow_total   INTEGER,

    deep_analysis   TEXT,
    deep_score      INTEGER,

    trial_results   TEXT,
    trial_score     REAL,

    trajectory_count INTEGER DEFAULT 0,
    trajectory_dir   TEXT,

    is_demo         BOOLEAN DEFAULT 0,
    domain          TEXT
);

CREATE INDEX IF NOT EXISTS idx_repos_status ON repos(status);
CREATE INDEX IF NOT EXISTS idx_repos_current_stage ON repos(current_stage);
"""


class RepoRegistry:
    def __init__(self, db_path="pipeline_data/registry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA_SQL)
        self._migrate()
        self.conn.commit()

    DEMO_KEYWORDS = ["demo", "sample", "practice", "example", "boilerplate", "starter", "awesome", "todo"]

    # Keyword → domain classification (first match wins), checked against description + topics.
    DOMAIN_RULES = [
        ("ecommerce", ("ecommerce", "e-commerce", "shop", "cart", "checkout", "store", "payment", "shopping")),
        ("cms_admin", ("cms", "wordpress", "strapi", "headless", "admin panel", "content management")),
        ("forums_social", ("forum", "social", "discourse", "community", "chat", "messaging")),
        ("auth", ("auth", "login", "signup", "sign-in", "oauth", "sso", "identity")),
        ("web_apps", ("dashboard", "crud", "saas", "portal", "management", "booking", "invoice", "analytics")),
    ]

    def _migrate(self):
        columns = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(repos)").fetchall()
        }
        if "parent_repo" not in columns:
            self.conn.execute("ALTER TABLE repos ADD COLUMN parent_repo TEXT")
        if "is_demo" not in columns:
            self.conn.execute("ALTER TABLE repos ADD COLUMN is_demo BOOLEAN DEFAULT 0")
            self._backfill_is_demo()
        if "domain" not in columns:
            self.conn.execute("ALTER TABLE repos ADD COLUMN domain TEXT")
            self._backfill_domain()

    @staticmethod
    def _check_is_demo(source_query: str) -> bool:
        source_lower = (source_query or "").lower()
        return any(kw in source_lower for kw in RepoRegistry.DEMO_KEYWORDS)

    def _backfill_is_demo(self):
        rows = self.conn.execute("SELECT full_name, source_query FROM repos").fetchall()
        for row in rows:
            is_demo = self._check_is_demo(row["source_query"])
            if is_demo:
                self.conn.execute(
                    "UPDATE repos SET is_demo = 1 WHERE full_name = ?",
                    (row["full_name"],),
                )

    @staticmethod
    def classify_domain(description: str, topics=None) -> str:
        if isinstance(topics, str):
            try:
                topics = json.loads(topics)
            except (json.JSONDecodeError, TypeError):
                topics = []
        text = ((description or "") + " " + " ".join(topics or [])).lower()
        for domain, keywords in RepoRegistry.DOMAIN_RULES:
            if any(kw in text for kw in keywords):
                return domain
        return "other"

    def _backfill_domain(self):
        rows = self.conn.execute("SELECT full_name, description, topics FROM repos").fetchall()
        for row in rows:
            self.conn.execute(
                "UPDATE repos SET domain = ? WHERE full_name = ?",
                (self.classify_domain(row["description"], row["topics"]), row["full_name"]),
            )

    def close(self):
        self.conn.close()

    def add_candidate(self, repo_data: dict) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        topics = repo_data.get("topics", [])
        if isinstance(topics, list):
            topics = json.dumps(topics)

        is_demo = self._check_is_demo(repo_data.get("source_query", ""))
        domain = self.classify_domain(repo_data.get("description"), repo_data.get("topics"))

        try:
            cursor = self.conn.execute(
                """INSERT OR IGNORE INTO repos
                   (full_name, html_url, stars, last_pushed, size_kb, language,
                    archived, fork, parent_repo, description, topics, default_branch,
                    status, current_stage, added_at, updated_at, source_query, is_demo, domain)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)""",
                (
                    repo_data["full_name"],
                    repo_data.get("html_url", ""),
                    repo_data.get("stars", 0),
                    repo_data.get("last_pushed"),
                    repo_data.get("size_kb", 0),
                    repo_data.get("language"),
                    repo_data.get("archived", False),
                    repo_data.get("fork", False),
                    repo_data.get("parent_repo"),
                    repo_data.get("description"),
                    topics,
                    repo_data.get("default_branch", "main"),
                    RepoStatus.DISCOVERED.value,
                    now,
                    now,
                    repo_data.get("source_query"),
                    is_demo,
                    domain,
                ),
            )
            self.conn.commit()
            # rowcount is per-statement (1 if inserted, 0 if the full_name already
            # existed and INSERT OR IGNORE skipped it). conn.total_changes is cumulative
            # across the connection and would report True forever after the first insert.
            return cursor.rowcount > 0
        except sqlite3.IntegrityError:
            return False

    def get_repos_for_stage(self, stage: int, limit=None) -> list[dict]:
        required_status = STAGE_TO_INPUT_STATUS.get(stage)
        if required_status is None and stage == 1:
            return []

        query = """SELECT *,
            CASE
                WHEN source_query LIKE '%stars:>%' THEN 1
                WHEN source_query LIKE '%stars:%' THEN 2
                WHEN source_query LIKE 'playwright test %' AND source_query NOT LIKE '%demo%'
                     AND source_query NOT LIKE '%sample%' AND source_query NOT LIKE '%practice%'
                     AND source_query NOT LIKE '%example%' AND source_query NOT LIKE '%todo%' THEN 3
                WHEN source_query LIKE '%data grid%' OR source_query LIKE '%rich text%'
                     OR source_query LIKE '%kanban%' OR source_query LIKE '%drag drop%'
                     OR source_query LIKE '%wizard%' OR source_query LIKE '%workflow%'
                     OR source_query LIKE '%spreadsheet%' OR source_query LIKE '%calendar%' THEN 3
                WHEN source_query LIKE '%checkout%' OR source_query LIKE '%invoice%'
                     OR source_query LIKE '%management%' OR source_query LIKE '%dashboard%'
                     OR source_query LIKE '%crud%' OR source_query LIKE '%booking%' THEN 4
                WHEN source_query LIKE '%filename:%' OR source_query LIKE 'topic:%' THEN 5
                WHEN source_query LIKE '%demo%' OR source_query LIKE '%sample%'
                     OR source_query LIKE '%practice%' OR source_query LIKE '%example%'
                     OR source_query LIKE '%boilerplate%' OR source_query LIKE '%starter%'
                     OR source_query LIKE '%awesome%' OR source_query LIKE '%todo%' THEN 7
                ELSE 6
            END as query_priority
            FROM repos WHERE status = ?"""
        params = [required_status.value]

        # Optional language filter (e.g. run only Python repos through a stage) — set via
        # PIPELINE_LANGUAGE env var, matching the codebase's env-config convention.
        language = os.environ.get("PIPELINE_LANGUAGE")
        if language:
            query += " AND language = ?"
            params.append(language)

        if stage >= 4:
            # Favor real products (higher production_complexity, stored inside the shallow_scores
            # JSON) over generic practice repos, then by overall shallow score.
            query += (
                " ORDER BY CAST(json_extract(shallow_scores, '$.production_complexity') AS INTEGER) DESC,"
                " COALESCE(shallow_total, 0) DESC, stars DESC"
            )
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            return [dict(row) for row in self.conn.execute(query, params).fetchall()]

        # Stages <4: order by priority, then balance across domains so scoring sees a varied
        # spread of repo types instead of top-N of a single one.
        query += " ORDER BY is_demo ASC, query_priority ASC, stars DESC"
        rows = [dict(row) for row in self.conn.execute(query, params).fetchall()]
        if not limit or len(rows) <= limit:
            return rows
        return self._round_robin_by_domain(rows, limit)

    @staticmethod
    def _round_robin_by_domain(repos: list[dict], limit: int) -> list[dict]:
        buckets: dict[str, list[dict]] = {}
        for repo in repos:  # repos arrive already in priority order; preserve it within each domain
            buckets.setdefault(repo.get("domain") or "other", []).append(repo)
        bucket_lists = list(buckets.values())
        selected: list[dict] = []
        while len(selected) < limit and any(bucket_lists):
            for bucket in bucket_lists:
                if bucket:
                    selected.append(bucket.pop(0))
                    if len(selected) >= limit:
                        break
        return selected

    def update_status(self, full_name: str, status: str, stage: int, **extra_fields):
        now = datetime.now(timezone.utc).isoformat()

        set_parts = ["status = ?", "current_stage = ?", "updated_at = ?"]
        params = [status, stage, now]

        field_map = {
            "filter_reason": "filter_reason",
            "shallow_scores": "shallow_scores",
            "shallow_total": "shallow_total",
            "deep_analysis": "deep_analysis",
            "deep_score": "deep_score",
            "trial_results": "trial_results",
            "trial_score": "trial_score",
            "trajectory_count": "trajectory_count",
            "trajectory_dir": "trajectory_dir",
        }

        for key, column in field_map.items():
            if key in extra_fields:
                value = extra_fields[key]
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                set_parts.append(f"{column} = ?")
                params.append(value)

        params.append(full_name)
        query = f"UPDATE repos SET {', '.join(set_parts)} WHERE full_name = ?"
        self.conn.execute(query, params)
        self.conn.commit()

    def reject(self, full_name: str, stage: int, reason: str):
        rejected_status = STAGE_TO_REJECTED_STATUS.get(stage)
        if rejected_status is None:
            raise ValueError(f"No rejection status for stage {stage}")
        self.update_status(
            full_name,
            rejected_status.value,
            stage,
            filter_reason=reason,
        )

    def get_repo(self, full_name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM repos WHERE full_name = ?", (full_name,)
        ).fetchone()
        if row:
            return dict(row)
        return None

    def get_stats(self) -> dict:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) as count FROM repos GROUP BY status"
        ).fetchall()
        return {row["status"]: row["count"] for row in rows}

    def get_domain_distribution(self) -> dict:
        rows = self.conn.execute(
            "SELECT COALESCE(domain, 'other') AS domain, COUNT(*) AS count FROM repos "
            "WHERE status NOT LIKE 'rejected_%' GROUP BY domain ORDER BY count DESC"
        ).fetchall()
        return {row["domain"]: row["count"] for row in rows}

    def get_rejection_reasons(self, limit=10) -> list[tuple[str, int]]:
        rows = self.conn.execute(
            """SELECT filter_reason, COUNT(*) as count
               FROM repos
               WHERE filter_reason IS NOT NULL AND filter_reason != ''
               GROUP BY filter_reason
               ORDER BY count DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [(row["filter_reason"], row["count"]) for row in rows]

    def get_pending_counts(self) -> dict:
        result = {}
        for stage, status in STAGE_TO_INPUT_STATUS.items():
            row = self.conn.execute(
                "SELECT COUNT(*) as count FROM repos WHERE status = ?",
                (status.value,),
            ).fetchone()
            result[stage] = row["count"]
        return result

    def reset_rejected(self, stage: int = None):
        if stage is not None:
            rejected_status = STAGE_TO_REJECTED_STATUS.get(stage)
            if rejected_status is None:
                raise ValueError(f"No rejection status for stage {stage}")
            input_status = STAGE_TO_INPUT_STATUS.get(stage)
            previous_stage = stage - 1
            self.conn.execute(
                "UPDATE repos SET status = ?, current_stage = ?, filter_reason = NULL WHERE status = ?",
                (input_status.value, previous_stage, rejected_status.value),
            )
        else:
            for s in sorted(STAGE_TO_REJECTED_STATUS.keys()):
                self.reset_rejected(stage=s)
        self.conn.commit()

    def has_repo(self, full_name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM repos WHERE full_name = ?", (full_name,)
        ).fetchone()
        return row is not None

    def get_duplicate_descriptions(self) -> dict[str, list[str]]:
        rows = self.conn.execute(
            """SELECT description, GROUP_CONCAT(full_name) as repos, COUNT(*) as cnt
               FROM repos
               WHERE description IS NOT NULL AND description != ''
                 AND length(description) > 20
               GROUP BY description
               HAVING cnt > 1
               ORDER BY cnt DESC"""
        ).fetchall()
        return {
            row["description"]: row["repos"].split(",")
            for row in rows
        }

    def count_total_trajectories(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(trajectory_count), 0) as total FROM repos WHERE status = ?",
            (RepoStatus.COMPLETED.value,),
        ).fetchone()
        return row["total"]
