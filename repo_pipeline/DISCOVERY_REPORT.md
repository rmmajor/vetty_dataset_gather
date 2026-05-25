# Stage 1–2: Repository Discovery and Filtering

## Objective

Collect the largest possible pool of GitHub repositories containing Playwright browser tests suitable for trajectory extraction. The pipeline favors recall over precision at this stage — aggressive quality filtering happens in later stages (3–6) where Claude inspects actual code.

## Stage 1: Discovery

### Method

Automated search via the GitHub Search API (PyGithub). Each query returns up to 1,000 results sorted by stars. Repos are inserted into a SQLite registry with `INSERT OR IGNORE` — duplicates across queries are silently deduplicated by primary key (`owner/repo`).

### Search Queries

62 queries across 7 categories:

| Category | Queries | Example | Yield (top) |
|----------|---------|---------|-------------|
| Framework + star tier | 8 | `playwright e2e test language:TypeScript stars:>100` | 99 |
| Domain-specific | 11 | `playwright ecommerce checkout test` | 129 |
| Target application | 14 | `playwright test magento`, `playwright test shopify` | — |
| Curated / example | 9 | `playwright demo tests`, `awesome playwright` | 975 |
| App pattern | 12 | `playwright test todo app`, `playwright test drag drop` | — |
| Dependency-based | 4 | `"@playwright/test" filename:package.json` | — |
| Topic / tag | 4 | `topic:playwright topic:e2e-testing` | 284 |

### Discovery Results

| Metric | Value |
|--------|-------|
| Total search hits | 6,141 |
| Unique repos after cross-query dedup | 5,197 |
| Duplicate rate across queries | 15.4% |
| Queries executed | 62 (49 yielded unique results) |
| Runtime | ~72 minutes (rate-limited at 30 req/min) |

### Top-yielding queries

| Query | Repos found |
|-------|-------------|
| `playwright demo tests` | 975 |
| `playwright automation framework` | 880 |
| `playwright practice test` | 798 |
| `playwright example tests` | 493 |
| `playwright sample tests` | 363 |

## Stage 2: Metadata Filtering

### Design Rationale

Stage 2 applies only filters that can be evaluated from GitHub API metadata alone (no cloning, no LLM). The goal is to remove repos that are provably unusable while retaining everything that might contain valid Playwright tests — including archived repos, repos with zero stars, repos in non-JS/Python languages (Playwright has Java and C# bindings), and repos not updated since 2023.

Traditional heuristics like star count, recency, and language were deliberately excluded:
- **Stars**: A zero-star repo can have a complete Playwright test suite. 69.5% of the filtered pool has 0 stars.
- **Archived**: Archived repos still contain runnable tests. 53 archived repos are included.
- **Last pushed**: Old tests still work. 255 repos last pushed before 2023 are included.
- **Language**: Playwright supports TypeScript, JavaScript, Python, Java, and C# officially. The Claude agent in stage 4 can install dependencies for any language.

### Filters Applied

| Filter | Rationale | Rejected |
|--------|-----------|----------|
| Empty repo (`size_kb == 0`) | GitHub reports 0 KB for repos with no committed files. Cannot contain tests. | 13 |
| Too large (`size_kb > 500,000`) | Repos over 500 MB are typically monorepos (e.g., Chromium, VS Code) where Playwright tests are a negligible fraction buried in a massive codebase. Cloning and searching would waste compute. | 11 |
| Duplicate description | Repos with identical descriptions longer than 30 characters, keeping the highest-starred variant. Catches tutorials re-uploaded without forking. | 49 |
| Fork with parent in registry | If GitHub marks a repo as a fork and the parent repo is already in the registry, the fork is redundant. (Not yet effective — `parent_repo` field requires re-discovery to populate.) | 0 |

### Filtering Results

| Metric | Value |
|--------|-------|
| Input (discovered) | 5,197 |
| Passed | 5,124 (98.6%) |
| Rejected | 73 (1.4%) |

### Rejection Breakdown

| Reason | Count | % of rejected |
|--------|-------|---------------|
| Duplicate description (clone) | 49 | 67.1% |
| Empty repo | 13 | 17.8% |
| Too large (>500 MB) | 11 | 15.1% |

## Candidate Pool Profile

### Language Distribution

| Language | Repos | % |
|----------|-------|---|
| TypeScript | 2,228 | 43.5% |
| JavaScript | 1,109 | 21.7% |
| Python | 486 | 9.5% |
| (not set) | 464 | 9.1% |
| HTML | 430 | 8.4% |
| Java | 188 | 3.7% |
| C# | 103 | 2.0% |
| Other (Shell, Vue, CSS, PHP, Go, Rust, Kotlin, RobotFramework) | 116 | 2.3% |

### Star Distribution

| Stars | Repos | Cumulative % |
|-------|-------|-------------|
| 0 | 3,563 | 69.5% |
| 1–4 | 1,088 | 90.8% |
| 5–9 | 111 | 92.9% |
| 10–49 | 212 | 97.1% |
| 50–99 | 63 | 98.3% |
| 100–499 | 54 | 99.4% |
| 500–999 | 15 | 99.7% |
| 1,000–4,999 | 10 | 99.9% |
| 5,000+ | 8 | 100% |

- Median stars: 0
- Mean stars: 49.1
- Max stars: 85,202

### Repository Size Distribution

| Size range | Repos | % |
|------------|-------|---|
| 1–10 KB | 1,036 | 20.2% |
| 10–100 KB | 1,768 | 34.5% |
| 100 KB–1 MB | 1,003 | 19.6% |
| 1–10 MB | 848 | 16.6% |
| 10–100 MB | 413 | 8.1% |
| 100–500 MB | 56 | 1.1% |

### Domain Distribution (keyword-based estimate)

| Domain | Repos | % |
|--------|-------|---|
| Other / general | 4,376 | 85.4% |
| Web apps (dashboard, CRUD, forms) | 309 | 6.0% |
| E-commerce (shop, cart, checkout) | 204 | 4.0% |
| Auth (login, signup, OAuth) | 200 | 3.9% |
| CMS / admin (WordPress, Strapi) | 34 | 0.7% |
| Forums / social | 1 | <0.1% |

### Other Characteristics

| Metric | Value |
|--------|-------|
| Archived repos in pool | 53 |
| Repos last pushed before 2023 | 255 |
| Repos marked as forks | 0* |

*Fork detection requires `repo.parent` API call which was added after initial discovery. Will be populated on subsequent discovery runs.

## What Happens Next

The 5,124 candidates proceed to Stage 3 (Claude Sonnet shallow scoring) where an LLM inspects the file tree, dependency files, and sample test code to score each repo on framework clarity, test count, interaction richness, domain relevance, runnability, and action diversity. This is where repos without actual Playwright tests get rejected.