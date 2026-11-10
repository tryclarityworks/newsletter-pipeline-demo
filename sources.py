import asyncio
import calendar
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser
import httpx
from dotenv import load_dotenv

from scoring import score_articles

if os.getenv("NEWSLETTER_LOAD_DOTENV") == "1":
    load_dotenv()

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HTTPX_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)
SOURCE_WINDOW_HOURS = 30
MAX_ARTICLES = 60
SOURCE_CAPS = {
    "tavily": 24,
    "reddit": 16,
    "hackernews": 16,
    "medium": 16,
}

TAVILY_QUERIES = [
    "AI tools automation workflow small business productivity last 30 hours",
    "AI automation agency founder operator news last 30 hours",
    "SMB small business AI implementation case study deployment last 30 hours",
    "no-code AI workflow automation tools SMB last 30 hours",
    "AI agents business operations practical tools last 30 hours",
]

MEDIUM_FEEDS = [
    "https://medium.com/feed/tag/artificial-intelligence",
    "https://medium.com/feed/tag/automation",
    "https://medium.com/feed/tag/small-business",
    "https://medium.com/feed/tag/ai-agents",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _window_start(now: datetime | None = None) -> datetime:
    return (now or _utc_now()) - timedelta(hours=SOURCE_WINDOW_HOURS)


def _parse_datetime(value) -> datetime | None:
    if not value:
        return None

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, time.struct_time):
        dt = datetime.fromtimestamp(calendar.timegm(value), timezone.utc)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        dt = None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = parsedate_to_datetime(raw)
            except (TypeError, ValueError, IndexError, OverflowError):
                return None
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_recent(dt: datetime | None, now: datetime | None = None) -> bool:
    if dt is None:
        return False
    current = now or _utc_now()
    return _window_start(current) <= dt <= current


def _normalize(
    title: str,
    url: str,
    snippet: str,
    published,
    source: str,
    now: datetime | None = None,
) -> dict | None:
    published_at = _parse_datetime(published)
    if not title or not url or not _is_recent(published_at, now):
        return None

    return {
        "title": title.strip(),
        "url": url.strip(),
        "snippet": snippet.strip(),
        "published_at": published_at.isoformat(),
        "_source": source,
    }


def _source_result(source: str, articles: list[dict], errors: list[str]) -> dict:
    capped = articles[: SOURCE_CAPS[source]]
    logger.info(f"[{source}] fetched {len(capped)} recent articles")
    result = {"source": source, "articles": capped}
    if not capped and errors:
        result["error"] = "; ".join(errors)
    return result


async def fetch_tavily() -> dict:
    """Fetch recent articles from Tavily Search API."""
    if not TAVILY_API_KEY:
        return {"source": "tavily", "articles": [], "error": "TAVILY_API_KEY is not set"}

    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            tasks = [
                client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": TAVILY_API_KEY,
                        "query": q,
                        "topic": "news",
                        "days": 2,
                        "search_depth": "basic",
                        "max_results": 8,
                        "include_answer": False,
                    },
                )
                for q in TAVILY_QUERIES
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        articles = []
        errors = []
        for resp in responses:
            if isinstance(resp, Exception):
                logger.warning(f"[tavily] query failed: {resp}")
                errors.append(str(resp))
                continue
            if resp.status_code != 200:
                logger.warning(f"[tavily] non-200 response: {resp.status_code} - {resp.text[:200]}")
                errors.append(f"HTTP {resp.status_code}: {resp.text[:100]}")
                continue

            for r in resp.json().get("results", []):
                article = _normalize(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    published=r.get("published_date", ""),
                    source="tavily",
                )
                if article:
                    articles.append(article)

        return _source_result("tavily", articles, errors)

    except Exception as e:
        logger.error(f"[tavily] fetch failed: {e}")
        return {"source": "tavily", "articles": [], "error": str(e)}


async def fetch_reddit() -> dict:
    """Fetch recent top posts from Reddit RSS feeds."""
    subreddits = ["entrepreneur", "automation", "artificial", "smallbusiness"]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; newsletter-bot/1.0; +https://tryclarityworks.com)"}

    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT, headers=headers, follow_redirects=True) as client:
            tasks = [client.get(f"https://www.reddit.com/r/{sub}/top.rss?t=day&limit=15") for sub in subreddits]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        articles = []
        errors = []
        for resp in responses:
            if isinstance(resp, Exception):
                logger.warning(f"[reddit] subreddit request failed: {resp}")
                errors.append(str(resp))
                continue
            if resp.status_code != 200:
                logger.warning(f"[reddit] non-200 response: {resp.status_code} - {resp.text[:200]}")
                errors.append(f"HTTP {resp.status_code}: {resp.text[:100]}")
                continue

            feed = feedparser.parse(resp.text)
            for entry in feed.entries:
                article = _normalize(
                    title=entry.get("title", ""),
                    url=entry.get("link", ""),
                    snippet=entry.get("summary", "")[:300],
                    published=entry.get("published_parsed") or entry.get("updated_parsed"),
                    source="reddit",
                )
                if article:
                    articles.append(article)

        return _source_result("reddit", articles, errors)

    except Exception as e:
        logger.error(f"[reddit] fetch failed: {e}")
        return {"source": "reddit", "articles": [], "error": str(e)}


async def fetch_hackernews() -> dict:
    """Fetch recent HN stories via the Algolia HN Search API."""
    cutoff = int(time.time()) - SOURCE_WINDOW_HOURS * 3600
    queries = ["AI", "automation", "startup", "small business"]

    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            tasks = [
                client.get(
                    "https://hn.algolia.com/api/v1/search_by_date",
                    params={
                        "query": q,
                        "tags": "story",
                        "numericFilters": f"created_at_i>{cutoff},points>5",
                        "hitsPerPage": 12,
                    },
                )
                for q in queries
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        articles = []
        errors = []
        for resp in responses:
            if isinstance(resp, Exception):
                logger.warning(f"[hn] query failed: {resp}")
                errors.append(str(resp))
                continue
            if resp.status_code != 200:
                logger.warning(f"[hn] non-200 response: {resp.status_code} - {resp.text[:200]}")
                errors.append(f"HTTP {resp.status_code}: {resp.text[:100]}")
                continue

            for hit in resp.json().get("hits", []):
                title = hit.get("title", "")
                url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
                article = _normalize(
                    title=title,
                    url=url,
                    snippet=(hit.get("story_text") or title)[:300],
                    published=hit.get("created_at", ""),
                    source="hackernews",
                )
                if article:
                    articles.append(article)

        return _source_result("hackernews", articles, errors)

    except Exception as e:
        logger.error(f"[hn] fetch failed: {e}")
        return {"source": "hackernews", "articles": [], "error": str(e)}


async def fetch_medium() -> dict:
    """Fetch recent Medium articles from curated tag RSS feeds."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; newsletter-bot/1.0; +https://tryclarityworks.com)"}

    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT, headers=headers, follow_redirects=True) as client:
            responses = await asyncio.gather(
                *(client.get(feed_url) for feed_url in MEDIUM_FEEDS),
                return_exceptions=True,
            )

        articles = []
        errors = []
        for resp in responses:
            if isinstance(resp, Exception):
                logger.warning(f"[medium] feed request failed: {resp}")
                errors.append(str(resp))
                continue
            if resp.status_code != 200:
                logger.warning(f"[medium] non-200 response: {resp.status_code} - {resp.text[:200]}")
                errors.append(f"HTTP {resp.status_code}: {resp.text[:100]}")
                continue

            feed = feedparser.parse(resp.text)
            for entry in feed.entries:
                article = _normalize(
                    title=entry.get("title", ""),
                    url=entry.get("link", ""),
                    snippet=entry.get("summary", "")[:300],
                    published=entry.get("published_parsed") or entry.get("updated_parsed"),
                    source="medium",
                )
                if article:
                    articles.append(article)

        return _source_result("medium", articles, errors)

    except Exception as e:
        logger.error(f"[medium] fetch failed: {e}")
        return {"source": "medium", "articles": [], "error": str(e)}


async def fetch_all_sources() -> dict:
    """Run all fetchers in parallel and return merged, deduplicated content."""
    results = await asyncio.gather(
        fetch_tavily(),
        fetch_reddit(),
        fetch_hackernews(),
        fetch_medium(),
    )

    merged = []
    errors = []
    for r in results:
        merged.extend(r.get("articles", []))
        if "error" in r:
            errors.append(f"{r['source']}: {r['error']}")

    if not merged:
        raise NoSourceDataError("All data sources returned empty recent results. " + "; ".join(errors))

    seen_urls: set[str] = set()
    deduped = []
    for article in merged:
        url = article["url"]
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(article)

    deduped = await score_articles(deduped)

    if len(deduped) > MAX_ARTICLES:
        logger.info(f"[sources] capping articles from {len(deduped)} to {MAX_ARTICLES}")
        deduped = deduped[:MAX_ARTICLES]

    logger.info(f"[sources] total recent articles collected: {len(deduped)} (before dedup: {len(merged)})")
    return {
        "articles": deduped,
        "source_errors": errors,
        "source_window_hours": SOURCE_WINDOW_HOURS,
    }


class NoSourceDataError(Exception):
    pass
