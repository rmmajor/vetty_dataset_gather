CREATE INDEX idx_agent_calls_full_name ON agent_calls(full_name);
CREATE INDEX idx_agent_calls_stage ON agent_calls(stage);
CREATE INDEX idx_repos_current_stage ON repos(current_stage);
CREATE INDEX idx_repos_status ON repos(status);
CREATE TABLE agent_calls (
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
CREATE TABLE repos (
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
CREATE TABLE sqlite_sequence(name,seq);
