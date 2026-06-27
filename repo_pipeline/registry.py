import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from repo_pipeline.schemas import (
    MAX_RETRIES,
    RETRYABLE_CATEGORIES,
    STAGE_TO_INPUT_STATUS,
    STAGE_TO_PASSED_STATUS,
    STAGE_TO_REJECTED_STATUS,
    FailureCategory,
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
    failure_category TEXT,
    retry_count     INTEGER DEFAULT 0,

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

-- One row per Claude Agent SDK call, across every agent-driven stage. Lets the dashboard show
-- turn / cost / token distributions per stage so max_turns and budgets can be tuned from data.
CREATE TABLE IF NOT EXISTS agent_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     TEXT,
    stage         INTEGER,
    purpose       TEXT,
    model         TEXT,
    num_turns     INTEGER,
    cost_usd      REAL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cache_read_tokens     INTEGER,
    cache_creation_tokens INTEGER,
    subtype       TEXT,
    is_error      BOOLEAN,
    duration_ms   INTEGER,
    ts            TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_calls_stage ON agent_calls(stage);
CREATE INDEX IF NOT EXISTS idx_agent_calls_full_name ON agent_calls(full_name);
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
        if "failure_category" not in columns:
            self.conn.execute("ALTER TABLE repos ADD COLUMN failure_category TEXT")
        if "retry_count" not in columns:
            self.conn.execute("ALTER TABLE repos ADD COLUMN retry_count INTEGER DEFAULT 0")

        call_columns = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(agent_calls)").fetchall()
        }
        if "cache_read_tokens" not in call_columns:
            self.conn.execute("ALTER TABLE agent_calls ADD COLUMN cache_read_tokens INTEGER")
        if "cache_creation_tokens" not in call_columns:
            self.conn.execute("ALTER TABLE agent_calls ADD COLUMN cache_creation_tokens INTEGER")

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
                WHEN source_query LIKE 'topic:self-hosted%' OR source_query LIKE 'topic:selfhosted%'
                     OR source_query LIKE 'topic:kanban%' OR source_query LIKE 'topic:webapp%'
                     OR source_query LIKE 'topic:dashboard%' OR source_query LIKE 'topic:admin-dashboard%'
                     OR source_query LIKE 'topic:project-management%' OR source_query LIKE 'topic:crm%'
                     OR source_query LIKE 'topic:inventory%' OR source_query LIKE 'topic:notes-app%'
                     OR source_query LIKE 'topic:todo topic:fullstack%' THEN 1
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
            # Drain order: bootability first (deep_score, set in stage 4 — NULL→0 for stage 4's
            # own input, where the richness term then dominates), then UI sophistication. The
            # richness term double-weights ui_difficulty + interaction_richness so apps with
            # drag-and-drop / editors / data grids reach agent capture before plain click+type
            # apps — selection, not rejection: plain apps still drain, just later.
            query += (
                " ORDER BY COALESCE(deep_score, 0) DESC,"
                " (2 * COALESCE(CAST(json_extract(shallow_scores, '$.ui_difficulty') AS INTEGER), 0)"
                "  + 2 * COALESCE(CAST(json_extract(shallow_scores, '$.interaction_richness') AS INTEGER), 0)"
                "  + COALESCE(CAST(json_extract(shallow_scores, '$.production_complexity') AS INTEGER), 0)"
                "  + COALESCE(CAST(json_extract(shallow_scores, '$.multi_page_flows') AS INTEGER), 0)) DESC,"
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
            "failure_category": "failure_category",
            "retry_count": "retry_count",
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

    def reject(self, full_name: str, stage: int, reason: str, category=None, **extra_fields):
        """Terminal rejection. ``category`` (a FailureCategory) records *why* it failed, distinct
        from the stage-scoped status. ``**extra_fields`` (e.g. shallow_scores, deep_score,
        trial_results, retry_count) are persisted so the scores behind a rejection are no longer
        thrown away — the dashboard needs them to tune thresholds."""
        rejected_status = STAGE_TO_REJECTED_STATUS.get(stage)
        if rejected_status is None:
            raise ValueError(f"No rejection status for stage {stage}")
        fields = dict(extra_fields)
        fields["filter_reason"] = reason
        if category is not None:
            fields["failure_category"] = category.value if isinstance(category, FailureCategory) else category
        self.update_status(full_name, rejected_status.value, stage, **fields)

    def record_retry(self, full_name: str, stage: int, category, reason: str, attempt: int):
        """Note a transient/infra failure that will be retried rather than rejected. Keeps the
        repo at its stage-input status (so a re-run still selects it after a crash) and bumps
        retry_count — does NOT move it to a rejected_* bucket."""
        input_status = STAGE_TO_INPUT_STATUS.get(stage)
        if input_status is None:
            raise ValueError(f"No input status for stage {stage}")
        cat = category.value if isinstance(category, FailureCategory) else category
        self.update_status(
            full_name,
            input_status.value,
            stage - 1,
            filter_reason=reason,
            failure_category=cat,
            retry_count=attempt,
        )

    def record_agent_call(self, full_name: str, stage: int, purpose: str, model: str,
                          num_turns: int, cost_usd, usage, subtype: str, is_error: bool,
                          duration_ms: int):
        usage = usage or {}
        self.conn.execute(
            """INSERT INTO agent_calls
               (full_name, stage, purpose, model, num_turns, cost_usd, input_tokens,
                output_tokens, cache_read_tokens, cache_creation_tokens, subtype, is_error,
                duration_ms, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                full_name, stage, purpose, model, num_turns, cost_usd,
                usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                usage.get("cache_read_input_tokens", 0),
                usage.get("cache_creation_input_tokens", 0),
                subtype, is_error, duration_ms,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def get_repo(self, full_name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM repos WHERE full_name = ?", (full_name,)
        ).fetchone()
        if row:
            return dict(row)
        return None

    def get_repos_by_status(self, statuses: list[str]) -> list[dict]:
        """All repos whose status is in ``statuses``. Generic selector for cross-stage operations
        (e.g. the unfit-prune cleanup, which re-judges everything past stage 4 still awaiting an
        expensive capture) that don't fit the single-input-status get_repos_for_stage shape."""
        if not statuses:
            return []
        placeholders = ",".join("?" * len(statuses))
        rows = self.conn.execute(
            f"SELECT * FROM repos WHERE status IN ({placeholders})", statuses
        ).fetchall()
        return [dict(row) for row in rows]

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

    def get_rejection_reasons(self, limit=10, status: str = None) -> list[tuple[str, int]]:
        """Top rejection reasons, aggregated across all stages by default.

        Pass ``status`` (e.g. ``RepoStatus.REJECTED_SCORE.value``) to restrict to one
        stage's rejection bucket — used by --stage runs to show what just got rejected
        at that stage instead of the all-time, all-stages aggregate.
        """
        where = "filter_reason IS NOT NULL AND filter_reason != ''"
        params: list = []
        if status:
            where += " AND status = ?"
            params.append(status)
        params.append(limit)
        rows = self.conn.execute(
            f"SELECT filter_reason, COUNT(*) as count FROM repos WHERE {where} "
            f"GROUP BY filter_reason ORDER BY count DESC LIMIT ?",
            params,
        ).fetchall()
        return [(row["filter_reason"], row["count"]) for row in rows]

    def get_failure_category_distribution(self) -> list[tuple[str, int]]:
        """(failure_category, count) across all rejected repos, most common first — the
        quality-vs-infra split that the opaque single bucket used to hide."""
        rows = self.conn.execute(
            "SELECT COALESCE(failure_category, '(uncategorized)') AS cat, COUNT(*) AS count "
            "FROM repos WHERE status LIKE 'rejected_%' GROUP BY cat ORDER BY count DESC"
        ).fetchall()
        return [(row["cat"], row["count"]) for row in rows]

    def get_pending_counts(self) -> dict:
        result = {}
        for stage, status in STAGE_TO_INPUT_STATUS.items():
            row = self.conn.execute(
                "SELECT COUNT(*) as count FROM repos WHERE status = ?",
                (status.value,),
            ).fetchone()
            result[stage] = row["count"]
        return result

    # Stage -> the column holding its gate score, so a re-queue can target rejections that a
    # lowered threshold would now accept (e.g. reset stage-4 rejects with deep_score >= 4).
    STAGE_SCORE_COLUMN = {3: "shallow_total", 4: "deep_score", 5: "trial_score"}

    def reset_rejected(self, stage: int = None, before: str = None, category_in=None,
                       min_score: float = None, dry_run: bool = False) -> int:
        """Re-queue rejected repos back to a stage's input status so the (now-fixed) stage retries
        them. Returns the number of repos affected.

        ``before`` (ISO timestamp) limits to rejections recorded before that instant — used to
        re-queue stale rejections from superseded code. ``category_in`` (FailureCategory iterable)
        limits to specific failure categories — used to re-queue infra/flaky failures while leaving
        genuine quality rejections alone. ``min_score`` keeps only rejections whose stored gate
        score (per STAGE_SCORE_COLUMN) is >= the value — used after lowering a threshold to re-queue
        exactly the now-passing repos (requires ``stage``). ``dry_run`` counts without modifying.
        """
        if min_score is not None and stage is None:
            raise ValueError("min_score requires an explicit stage")
        stages = [stage] if stage is not None else sorted(STAGE_TO_REJECTED_STATUS.keys())
        cats = None
        if category_in:
            cats = [c.value if isinstance(c, FailureCategory) else c for c in category_in]

        total = 0
        for s in stages:
            rejected_status = STAGE_TO_REJECTED_STATUS.get(s)
            input_status = STAGE_TO_INPUT_STATUS.get(s)
            if rejected_status is None or input_status is None:
                raise ValueError(f"No rejection/input status for stage {s}")

            where = ["status = ?"]
            params: list = [rejected_status.value]
            if before:
                where.append("updated_at < ?")
                params.append(before)
            if cats:
                where.append(f"failure_category IN ({','.join('?' * len(cats))})")
                params.extend(cats)
            if min_score is not None:
                score_col = self.STAGE_SCORE_COLUMN.get(s)
                if score_col is None:
                    raise ValueError(f"No score column for stage {s}")
                where.append(f"{score_col} >= ?")
                params.append(min_score)
            clause = " AND ".join(where)

            if dry_run:
                row = self.conn.execute(
                    f"SELECT COUNT(*) AS c FROM repos WHERE {clause}", params
                ).fetchone()
                total += row["c"]
                continue

            cursor = self.conn.execute(
                f"UPDATE repos SET status = ?, current_stage = ?, filter_reason = NULL, "
                f"failure_category = NULL, retry_count = 0 WHERE {clause}",
                [input_status.value, s - 1] + params,
            )
            total += cursor.rowcount
        if not dry_run:
            self.conn.commit()
        return total

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

    def count_total_trajectories(self, status: str = None) -> int:
        """Sum trajectory_count over repos in a status (default: completed = finalized/trainable).
        Pass RepoStatus.CAPTURED.value to count trajectories captured but pending stage-7 finalize —
        these exist on disk and verified but are not yet counted as 'generated'."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(trajectory_count), 0) as total FROM repos WHERE status = ?",
            (status or RepoStatus.COMPLETED.value,),
        ).fetchone()
        return row["total"]
