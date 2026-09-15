import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

from github import Github, GithubException, RateLimitExceededException

from cua.pipeline.registry import RepoRegistry

logger = logging.getLogger(__name__)

# The Search API caps a query at 1000 results, so a broad query is partitioned by `stars:` buckets and then, if still over the cap, recursively by immutable `created:` dates — see iter_windows().
PER_QUERY_CAP = 1000
STAR_THRESHOLDS = [5, 10, 20, 50, 100, 200, 500, 1000]
GITHUB_EPOCH = date(2008, 1, 1)
DEFAULT_STAR_FLOOR = 5

QUERIES_PATH = Path(__file__).with_name("github_queries.json")
_QUERY_SETS = json.loads(QUERIES_PATH.read_text())
# Selenium queries mirror the Playwright set category-for-category but are pinned to language:Python, the only Selenium instrumentation that exists.
SELENIUM_SEARCH_QUERIES = _QUERY_SETS["selenium"]
SEARCH_QUERIES = _QUERY_SETS["playwright"] + SELENIUM_SEARCH_QUERIES


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
    """Match count for a query; PyGithub clamps `.totalCount` to the 1000 ceiling, so exactly PER_QUERY_CAP means "split further"."""
    wait_for_rate_limit(github_client)
    return github_client.search_repositories(query=query).totalCount


def _star_bucket_fragments(star_floor: int) -> list[str]:
    """Descending `stars:` fragments down to star_floor (['stars:>=1000', 'stars:500..999', ..., 'stars:5..9']), highest first."""
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
    """Yield `query` plus `created:` sub-windows under the cap, bisecting [low, high]; created_at is immutable, so windows are stable across re-runs."""
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
    """Yield sub-queries of base_query each under the cap, so paginating all of them covers every match down to star_floor."""
    today = date.today()

    if "stars:" in base_query:
        # Query carries its own star constraint (dependency/topic tiers), so only a date split can still be needed.
        if _search_total(github_client, base_query) < PER_QUERY_CAP:
            yield base_query
        else:
            yield from _created_windows(github_client, base_query, GITHUB_EPOCH, today)
        return

    # `{base} stars:>=floor` is exactly the union of the per-bucket queries, so one probe of it replaces a probe per bucket at identical coverage.
    floor_query = f"{base_query} stars:>={star_floor}"
    floor_total = _search_total(github_client, floor_query)
    if floor_total == 0:
        return
    if floor_total < PER_QUERY_CAP:
        yield floor_query
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
    """Page through one bounded window recording every repo, returning (found, new); source_query is the base query, not the windowed form."""
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
