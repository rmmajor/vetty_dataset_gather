import logging
import time

from github import Github, GithubException, RateLimitExceededException

from repo_pipeline.registry import RepoRegistry

logger = logging.getLogger(__name__)

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


def run_discovery(registry: RepoRegistry, github_token: str, extra_queries: list[str] = None):
    github_client = Github(github_token, per_page=100)
    queries = SEARCH_QUERIES + (extra_queries or [])

    total_found = 0
    total_new = 0

    for i, query_str in enumerate(queries):
        logger.info(f"[{i + 1}/{len(queries)}] Searching: {query_str}")
        wait_for_rate_limit(github_client)

        try:
            results = github_client.search_repositories(query=query_str, sort="stars", order="desc")
            page_count = 0

            for repo in results:
                if page_count >= 1000:
                    break

                total_found += 1
                repo_data = extract_repo_data(repo, query_str)
                was_new = registry.add_candidate(repo_data)
                if was_new:
                    total_new += 1
                    logger.debug(f"  NEW: {repo.full_name} ({repo.stargazers_count} stars)")

                page_count += 1

                if page_count % 100 == 0:
                    wait_for_rate_limit(github_client)

        except RateLimitExceededException:
            logger.warning(f"Rate limited on query: {query_str}, waiting...")
            wait_for_rate_limit(github_client)
        except GithubException as e:
            logger.warning(f"GitHub API error on query '{query_str}': {e}")
            if e.status == 422:
                continue
            time.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error on query '{query_str}': {e}")
            continue

        if (i + 1) % 5 == 0:
            stats = registry.get_stats()
            discovered = stats.get("discovered", 0)
            logger.info(f"  Progress: {total_found} found, {total_new} new, {discovered} total discovered")

    github_client.close()

    stats = registry.get_stats()
    logger.info(f"Discovery complete: {total_found} found, {total_new} new")
    logger.info(f"Registry stats: {stats}")
    return {"total_found": total_found, "total_new": total_new, "stats": stats}
