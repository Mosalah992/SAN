"""
feed_scanner — aggregate global + per-submolt posts for engage/search.

Used by agent_loop (scan_full_feed) and reply_handler (get_hot_topics_for_search).
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
from collections import Counter
from typing import Any, Callable, Awaitable

import aiohttp

log = logging.getLogger("soul")

# Minimal stopwords for topic extraction
_STOP = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may",
    "new", "now", "old", "see", "two", "way", "who", "boy", "did", "let", "put",
    "say", "she", "too", "use", "that", "this", "with", "from", "have", "been",
    "will", "your", "what", "when", "they", "them", "than", "then", "some", "into",
    "just", "like", "over", "also", "back", "only", "know", "take", "year", "good",
    "any", "agent", "agents", "post", "posts", "molt", "moltbook",
})


def _posts_from_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    return list(data.get("posts", data.get("data", [])) or [])


def _extract_hot_topics(posts: list[dict[str, Any]], top_n: int = 24) -> list[tuple[str, float]]:
    """Return [(word, weight), ...] from titles + content snippets."""
    words: Counter[str] = Counter()
    for p in posts:
        title = p.get("title") or ""
        body = (p.get("content") or p.get("content_preview") or "")[:400]
        text = f"{title} {body}"
        for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()):
            if w in _STOP or len(w) > 32:
                continue
            words[w] += 1
    if not words:
        return []
    max_c = max(words.values())
    out: list[tuple[str, float]] = []
    for word, c in words.most_common(top_n):
        out.append((word, round(c / max_c, 4)))
    return out


def merge_into_state(state: dict[str, Any], posts: list[dict[str, Any]], hot_topics: list[tuple[str, float]]) -> None:
    state["scanned_feed"] = posts
    state["hot_topics"] = hot_topics


def get_hot_topics_for_search(hot_topics: list[Any]) -> list[str]:
    """Turn hot_topics into search query strings."""
    if not hot_topics:
        return []
    queries: list[str] = []
    for item in hot_topics[:20]:
        if isinstance(item, (list, tuple)) and item:
            w = str(item[0]).strip()
            if len(w) >= 3:
                queries.append(w)
        elif isinstance(item, str) and len(item.strip()) >= 3:
            queries.append(item.strip())
    return queries


async def scan_full_feed(
    session: aiohttp.ClientSession,
    api_get: Callable[[aiohttp.ClientSession, str], Awaitable[dict[str, Any]]],
    target_submolts: list[str],
    *,
    global_hot_limit: int = 80,
    global_new_limit: int = 40,
    submolt_limit: int = 15,
    submolt_count: int = 12,
) -> tuple[list[dict[str, Any]], list[tuple[str, float]]]:
    """
    Merge global hot/new with hot posts from the first ``submolt_count`` target submolts.
    Returns (deduped_posts, hot_topics).
    """
    posts_by_id: dict[str, dict[str, Any]] = {}

    async def _ingest(path: str) -> None:
        try:
            data = await api_get(session, path)
            for p in _posts_from_response(data):
                pid = p.get("id")
                if pid:
                    posts_by_id[str(pid)] = p
        except Exception as e:
            log.debug("feed_scanner ingest failed %s: %s", path, e)

    await _ingest(f"/posts?sort=hot&limit={max(1, global_hot_limit)}")
    await asyncio.sleep(random.uniform(0.3, 0.8))
    await _ingest(f"/posts?sort=new&limit={max(1, global_new_limit)}")
    await asyncio.sleep(random.uniform(0.3, 0.8))

    subs = list(target_submolts)[: max(0, submolt_count)]
    for sm in subs:
        await _ingest(f"/posts?submolt={sm}&sort=hot&limit={max(1, submolt_limit)}")
        await asyncio.sleep(random.uniform(0.4, 1.0))

    posts = list(posts_by_id.values())
    hot_topics = _extract_hot_topics(posts)
    log.debug("feed_scanner: merged %d unique posts, %d hot topic terms", len(posts), len(hot_topics))
    return posts, hot_topics
