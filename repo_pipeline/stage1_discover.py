import logging
import time
from datetime import date, timedelta

from github import Github, GithubException, RateLimitExceededException

from repo_pipeline.registry import RepoRegistry

logger = logging.getLogger(__name__)

# GitHub's Search API returns at most 1000 results per query, so a broad query (sorted by
# stars) only ever surfaces its top 1000 and the long tail is invisible. To scrape a query
# exhaustively we partition it into sub-queries that each return <1000 results: first by
# `stars:` buckets (down to a floor), then — for any bucket still over the cap — recursively
# by `created:` date (immutable, arbitrarily splittable). See iter_windows().
PER_QUERY_CAP = 1000
STAR_THRESHOLDS = [5, 10, 20, 50, 100, 200, 500, 1000]
GITHUB_EPOCH = date(2008, 1, 1)
DEFAULT_STAR_FLOOR = 5

SEARCH_QUERIES = [
    # ---- Real product test suites (large projects using Playwright for CI) ----
    "playwright e2e test language:TypeScript stars:>500",
    "playwright e2e test language:TypeScript stars:100..500",
    "playwright e2e test language:TypeScript stars:20..100",
    "playwright e2e test language:Python stars:>50",
    "playwright e2e test language:Python stars:10..50",
    "playwright test language:TypeScript stars:>50",
    "playwright test language:Python stars:>10",
    "playwright test language:JavaScript stars:>50",
    "playwright test language:JavaScript stars:10..100",

    # ---- Named production applications ----
    "playwright test grafana",
    "playwright test supabase",
    "playwright test cal.com",
    "playwright test appsmith",
    "playwright test nocodb",
    "playwright test tooljet",
    "playwright test budibase",
    "playwright test strapi",
    "playwright test ghost",
    "playwright test directus",
    "playwright test payload cms",
    "playwright test medusa",
    "playwright test saleor",
    "playwright test vendure",
    "playwright test bagisto",
    "playwright test erpnext",
    "playwright test odoo",
    "playwright test twenty crm",
    "playwright test plane project",
    "playwright test huly",
    "playwright test outline wiki",
    "playwright test documenso",
    "playwright test formbricks",
    "playwright test typebot",
    "playwright test n8n",
    "playwright test windmill",
    "playwright test activepieces",
    "playwright test teable",
    "playwright test undb",
    "playwright test baserow",
    "playwright test grist",
    "playwright test mattermost",
    "playwright test rocket.chat",
    "playwright test element matrix",
    "playwright test chatwoot",
    "playwright test zammad",
    "playwright test freescout",
    "playwright test rallly",
    "playwright test cal booking",
    "playwright test gitlab",
    "playwright test gitea",
    "playwright test redmine",
    "playwright test taiga",
    "playwright test wekan",
    "playwright test discourse",
    "playwright test flarum",
    "playwright test lemmy",
    "playwright test postiz",
    "playwright test wordpress admin",
    "playwright test magento",
    "playwright test woocommerce",
    "playwright test shopify",
    "playwright test prestashop",
    "playwright test nextcloud",
    "playwright test immich",
    "playwright test paperless",

    # ---- Complex UI pattern queries ----
    "playwright test data grid inline edit",
    "playwright test rich text editor tiptap",
    "playwright test rich text editor prosemirror",
    "playwright test kanban board",
    "playwright test drag drop sortable",
    "playwright test multi-step wizard form",
    "playwright test nested navigation sidebar",
    "playwright test tree view",
    "playwright test file manager upload",
    "playwright test calendar scheduling",
    "playwright test table pagination filter sort",
    "playwright test dashboard charts widgets",
    "playwright test spreadsheet cells",
    "playwright test code editor monaco",
    "playwright test markdown editor preview",
    "playwright test image cropper",
    "playwright test color picker",
    "playwright test autocomplete combobox",
    "playwright test date range picker",
    "playwright test modal dialog nested",
    "playwright test tabs accordion",
    "playwright test virtual scroll infinite",
    "playwright test permission role management",
    "playwright test workflow builder",
    "playwright test form builder",
    "playwright test email template editor",
    "playwright test notification center",
    "playwright test command palette",
    "playwright test keyboard shortcuts",
    "playwright test context menu right click",
    "playwright test undo redo",

    # ---- Domain-specific (real business flows) ----
    "playwright ecommerce checkout test",
    "playwright shopping cart test",
    "playwright login authentication test",
    "playwright form submission test",
    "playwright cms admin test",
    "playwright dashboard test",
    "playwright crud test",
    "playwright search filter test",
    "playwright booking reservation test",
    "playwright inventory management test",
    "playwright invoice billing test",
    "playwright user management test",
    "playwright settings configuration test",
    "playwright onboarding wizard test",
    "playwright import export csv test",
    "playwright multi-tenant test",
    "playwright webhook integration test",
    "playwright api key management test",
    "playwright audit log test",

    # ---- Dependency-based (code search — finds real projects) ----
    '"@playwright/test" filename:package.json',
    "playwright filename:requirements.txt",
    "pytest-playwright filename:requirements.txt",
    '"@playwright/test" filename:playwright.config',

    # ---- Topic-based ----
    "topic:playwright topic:e2e-testing",
    "topic:playwright topic:test-automation",
    "topic:playwright-tests",
    "topic:browser-testing playwright",

    # ---- Self-hosted apps with Playwright CI ----
    "playwright ci self-hosted stars:>50",
    "playwright e2e open source app stars:>100",
    "playwright integration test SaaS",
    '"npx playwright test" filename:.github/workflows',
]

# Selenium is a second trajectory source. The query set mirrors the Playwright one
# category-for-category, but every query is constrained to language:Python because the
# Selenium instrumentation (conftest monkey-patching) only works for Python/pytest —
# non-Python Selenium repos would just be rejected at stage 4. Framework detection
# (Playwright vs Selenium) happens later in stage 4.
SELENIUM_SEARCH_QUERIES = [
    # ---- Real product test suites (Python star tiers) ----
    "selenium e2e test language:Python stars:>500",
    "selenium e2e test language:Python stars:100..500",
    "selenium e2e test language:Python stars:20..100",
    "selenium e2e test language:Python stars:>50",
    "selenium e2e test language:Python stars:10..50",
    "selenium test language:Python stars:>50",
    "selenium test language:Python stars:>10",
    "selenium webdriver test language:Python stars:>50",
    "pytest selenium language:Python stars:>10",

    # ---- Named production applications ----
    "selenium test grafana language:Python",
    "selenium test supabase language:Python",
    "selenium test cal.com language:Python",
    "selenium test appsmith language:Python",
    "selenium test nocodb language:Python",
    "selenium test tooljet language:Python",
    "selenium test budibase language:Python",
    "selenium test strapi language:Python",
    "selenium test ghost language:Python",
    "selenium test directus language:Python",
    "selenium test payload cms language:Python",
    "selenium test medusa language:Python",
    "selenium test saleor language:Python",
    "selenium test vendure language:Python",
    "selenium test bagisto language:Python",
    "selenium test erpnext language:Python",
    "selenium test odoo language:Python",
    "selenium test twenty crm language:Python",
    "selenium test plane project language:Python",
    "selenium test huly language:Python",
    "selenium test outline wiki language:Python",
    "selenium test documenso language:Python",
    "selenium test formbricks language:Python",
    "selenium test typebot language:Python",
    "selenium test n8n language:Python",
    "selenium test windmill language:Python",
    "selenium test activepieces language:Python",
    "selenium test teable language:Python",
    "selenium test undb language:Python",
    "selenium test baserow language:Python",
    "selenium test grist language:Python",
    "selenium test mattermost language:Python",
    "selenium test rocket.chat language:Python",
    "selenium test element matrix language:Python",
    "selenium test chatwoot language:Python",
    "selenium test zammad language:Python",
    "selenium test freescout language:Python",
    "selenium test rallly language:Python",
    "selenium test cal booking language:Python",
    "selenium test gitlab language:Python",
    "selenium test gitea language:Python",
    "selenium test redmine language:Python",
    "selenium test taiga language:Python",
    "selenium test wekan language:Python",
    "selenium test discourse language:Python",
    "selenium test flarum language:Python",
    "selenium test lemmy language:Python",
    "selenium test postiz language:Python",
    "selenium test wordpress admin language:Python",
    "selenium test magento language:Python",
    "selenium test woocommerce language:Python",
    "selenium test shopify language:Python",
    "selenium test prestashop language:Python",
    "selenium test nextcloud language:Python",
    "selenium test immich language:Python",
    "selenium test paperless language:Python",

    # ---- Complex UI pattern queries ----
    "selenium test data grid inline edit language:Python",
    "selenium test rich text editor tiptap language:Python",
    "selenium test rich text editor prosemirror language:Python",
    "selenium test kanban board language:Python",
    "selenium test drag drop sortable language:Python",
    "selenium test multi-step wizard form language:Python",
    "selenium test nested navigation sidebar language:Python",
    "selenium test tree view language:Python",
    "selenium test file manager upload language:Python",
    "selenium test calendar scheduling language:Python",
    "selenium test table pagination filter sort language:Python",
    "selenium test dashboard charts widgets language:Python",
    "selenium test spreadsheet cells language:Python",
    "selenium test code editor monaco language:Python",
    "selenium test markdown editor preview language:Python",
    "selenium test image cropper language:Python",
    "selenium test color picker language:Python",
    "selenium test autocomplete combobox language:Python",
    "selenium test date range picker language:Python",
    "selenium test modal dialog nested language:Python",
    "selenium test tabs accordion language:Python",
    "selenium test virtual scroll infinite language:Python",
    "selenium test permission role management language:Python",
    "selenium test workflow builder language:Python",
    "selenium test form builder language:Python",
    "selenium test email template editor language:Python",
    "selenium test notification center language:Python",
    "selenium test command palette language:Python",
    "selenium test keyboard shortcuts language:Python",
    "selenium test context menu right click language:Python",
    "selenium test undo redo language:Python",

    # ---- Domain-specific (real business flows) ----
    "selenium ecommerce checkout test language:Python",
    "selenium shopping cart test language:Python",
    "selenium login authentication test language:Python",
    "selenium form submission test language:Python",
    "selenium cms admin test language:Python",
    "selenium dashboard test language:Python",
    "selenium crud test language:Python",
    "selenium search filter test language:Python",
    "selenium booking reservation test language:Python",
    "selenium inventory management test language:Python",
    "selenium invoice billing test language:Python",
    "selenium user management test language:Python",
    "selenium settings configuration test language:Python",
    "selenium onboarding wizard test language:Python",
    "selenium import export csv test language:Python",
    "selenium multi-tenant test language:Python",
    "selenium webhook integration test language:Python",
    "selenium api key management test language:Python",
    "selenium audit log test language:Python",

    # ---- Dependency-based (code search — finds real projects) ----
    "selenium filename:requirements.txt language:Python",
    "pytest-selenium filename:requirements.txt",
    "selenium webdriver filename:requirements.txt language:Python",
    '"from selenium import webdriver" language:Python',

    # ---- Topic-based ----
    "topic:selenium topic:e2e-testing",
    "topic:selenium topic:test-automation",
    "topic:selenium-webdriver",
    "topic:browser-testing selenium",

    # ---- Self-hosted apps with Selenium CI ----
    "selenium ci self-hosted language:Python stars:>50",
    "selenium e2e open source app language:Python stars:>50",
    "selenium integration test SaaS language:Python",
    '"pytest" "selenium" filename:.github/workflows',
]

SEARCH_QUERIES = SEARCH_QUERIES + SELENIUM_SEARCH_QUERIES


def wait_for_rate_limit(github_client):
    rate_limit = github_client.get_rate_limit()
    resources = rate_limit.resources

    search_remaining = resources.search.remaining
    if search_remaining < 2:
        reset_time = resources.search.reset.timestamp()
        sleep_seconds = max(reset_time - time.time() + 1, 0)
        logger.info(f"Search rate limit hit, sleeping {sleep_seconds:.0f}s")
        time.sleep(sleep_seconds)

    core_remaining = resources.core.remaining
    if core_remaining < 10:
        reset_time = resources.core.reset.timestamp()
        sleep_seconds = max(reset_time - time.time() + 1, 0)
        logger.info(f"Core rate limit hit, sleeping {sleep_seconds:.0f}s")
        time.sleep(sleep_seconds)


def extract_repo_data(repo, source_query):
    try:
        pushed_at = repo.pushed_at.isoformat() if repo.pushed_at else None
    except Exception:
        pushed_at = None

    parent_name = None
    if repo.fork:
        try:
            parent = repo.parent
            if parent:
                parent_name = parent.full_name
        except Exception:
            pass

    return {
        "full_name": repo.full_name,
        "html_url": repo.html_url,
        "stars": repo.stargazers_count,
        "last_pushed": pushed_at,
        "size_kb": repo.size,
        "language": repo.language,
        "archived": repo.archived,
        "fork": repo.fork,
        "parent_repo": parent_name,
        "description": repo.description or "",
        "topics": getattr(repo, "topics", None) or [],
        "default_branch": repo.default_branch,
        "source_query": source_query,
    }


def _search_total(github_client, query: str) -> int:
    """Match count for a query (one search request). NOTE: PyGithub's `.totalCount` is
    clamped to the 1000-result ceiling — a query with 1,415 true matches reports 1000.
    That's exactly what we need: a reported count of `PER_QUERY_CAP` means "at least the
    cap, so this window must be split further", while anything below it is the true,
    fully-retrievable count. Callers therefore split when `>= PER_QUERY_CAP`."""
    wait_for_rate_limit(github_client)
    return github_client.search_repositories(query=query).totalCount


def _star_bucket_fragments(star_floor: int) -> list[str]:
    """Descending `stars:` range fragments down to star_floor, e.g. floor 5 ->
    ['stars:>=1000', 'stars:500..999', ..., 'stars:5..9']. High-star buckets first so the
    most valuable repos are discovered earliest."""
    thresholds = sorted({t for t in STAR_THRESHOLDS if t > star_floor} | {star_floor})
    fragments = []
    for idx, low in enumerate(thresholds):
        if idx + 1 < len(thresholds):
            fragments.append(f"stars:{low}..{thresholds[idx + 1] - 1}")
        else:
            fragments.append(f"stars:>={low}")
    fragments.reverse()
    return fragments


def _created_windows(github_client, query: str, low: date, high: date):
    """Yield `query` augmented with `created:` date sub-windows that each match <=cap
    repos, recursively bisecting [low, high]. created_at is immutable, so windows are
    stable across re-runs and any interval can be split until it fits under the cap."""
    windowed = f"{query} created:{low.isoformat()}..{high.isoformat()}"
    total = _search_total(github_client, windowed)
    if total < PER_QUERY_CAP:
        if total > 0:
            yield windowed
        return
    if low >= high:
        # A single day over the cap (not expected at sane star floors): take the top 1000.
        logger.warning(f"Window exceeds cap on a single day ({total}): {windowed}")
        yield windowed
        return
    mid = low + (high - low) // 2
    yield from _created_windows(github_client, query, low, mid)
    yield from _created_windows(github_client, query, mid + timedelta(days=1), high)


def iter_windows(github_client, base_query: str, star_floor: int = DEFAULT_STAR_FLOOR):
    """Yield bounded sub-queries of base_query that each match <=cap repos, so paginating
    all of them covers every match (down to star_floor when base_query doesn't already
    constrain stars). Queries already under the cap are yielded unchanged after one probe."""
    today = date.today()

    if "stars:" in base_query:
        # Query carries its own star constraint (e.g. dependency/topic tiers) — only split
        # by date if it still reaches the cap.
        if _search_total(github_client, base_query) < PER_QUERY_CAP:
            yield base_query
        else:
            yield from _created_windows(github_client, base_query, GITHUB_EPOCH, today)
        return

    for fragment in _star_bucket_fragments(star_floor):
        sub = f"{base_query} {fragment}"
        total = _search_total(github_client, sub)
        if total == 0:
            continue
        if total < PER_QUERY_CAP:
            yield sub
        else:
            yield from _created_windows(github_client, sub, GITHUB_EPOCH, today)


def _paginate_window(github_client, registry, window_query: str, source_query: str) -> tuple[int, int]:
    """Page through one bounded window, recording every repo. Returns (found, new).
    source_query is the original base query (not the windowed form) so the registry's
    query-priority heuristic and framework attribution stay meaningful."""
    found = 0
    new = 0
    wait_for_rate_limit(github_client)
    try:
        results = github_client.search_repositories(query=window_query, sort="stars", order="desc")
        page_count = 0
        for repo in results:
            if page_count >= PER_QUERY_CAP:
                break
            found += 1
            if registry.add_candidate(extract_repo_data(repo, source_query)):
                new += 1
            page_count += 1
            if page_count % 100 == 0:
                wait_for_rate_limit(github_client)
    except RateLimitExceededException:
        logger.warning(f"Rate limited on window: {window_query}, waiting...")
        wait_for_rate_limit(github_client)
    except GithubException as e:
        logger.warning(f"GitHub API error on window '{window_query}': {e}")
        if e.status != 422:
            time.sleep(5)
    except Exception as e:
        logger.error(f"Unexpected error on window '{window_query}': {e}")
    return found, new


def run_discovery(
    registry: RepoRegistry,
    github_token: str,
    extra_queries: list[str] = None,
    star_floor: int = DEFAULT_STAR_FLOOR,
):
    github_client = Github(github_token, per_page=100)
    queries = SEARCH_QUERIES + (extra_queries or [])

    total_found = 0
    total_new = 0

    for i, base_query in enumerate(queries):
        logger.info(f"[{i + 1}/{len(queries)}] Base query: {base_query}")

        try:
            windows = list(iter_windows(github_client, base_query, star_floor))
        except RateLimitExceededException:
            wait_for_rate_limit(github_client)
            try:
                windows = list(iter_windows(github_client, base_query, star_floor))
            except Exception as e:
                logger.warning(f"Windowing failed for '{base_query}': {e}")
                continue
        except Exception as e:
            logger.warning(f"Windowing error for '{base_query}': {e}")
            continue

        logger.info(f"  partitioned into {len(windows)} window(s) (cap {PER_QUERY_CAP}/query, star floor {star_floor})")

        for window_query in windows:
            found, new = _paginate_window(github_client, registry, window_query, base_query)
            total_found += found
            total_new += new

        if (i + 1) % 5 == 0:
            stats = registry.get_stats()
            discovered = stats.get("discovered", 0)
            logger.info(f"  Progress: {total_found} found, {total_new} new, {discovered} total discovered")

    github_client.close()

    stats = registry.get_stats()
    logger.info(f"Discovery complete: {total_found} found, {total_new} new")
    logger.info(f"Registry stats: {stats}")
    return {"total_found": total_found, "total_new": total_new, "stats": stats}
