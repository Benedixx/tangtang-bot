from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import RatelimitException

LOGGER = logging.getLogger("tang.tools.search")

MAX_RESULTS = 4
CACHE_TTL_S = 1800.0
MAX_BACKOFF_S = 30.0


class WebSearchTool:
    """ddgs search with 30-min cache and jittered backoff on rate limits."""

    def __init__(self, max_results: int = MAX_RESULTS) -> None:
        self._max_results = max_results
        self._cache: dict[str, tuple[float, str]] = {}

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.lower().split())

    async def execute(self, query: str, request_id: str | None = None) -> str:
        key = self._normalize(query)
        if not key:
            return ""
        now = time.time()
        hit = self._cache.get(key)
        if hit and now - hit[0] < CACHE_TTL_S:
            LOGGER.info("[%s] search_cache_hit query=%s", request_id or "-", key)
            return hit[1]

        results = await asyncio.to_thread(self._search_sync, key)
        if not results:
            return "no results"
        text = self._format(results)
        self._cache[key] = (time.time(), text)
        LOGGER.info("[%s] search_done query=%s results=%s", request_id or "-", key, len(results))
        return text

    def _search_sync(self, query: str) -> list[dict[str, Any]]:
        attempts = 0
        while True:
            try:
                ddgs = DDGS()
                return list(ddgs.text(
                    query,
                    max_results=self._max_results,
                    region="id-ID",
                    engine=random.choice(["google", "bing", "brave", "grokipedia"]),
                ))
            except RatelimitException:
                attempts += 1
                if attempts > 3:
                    LOGGER.warning("search_ratelimit_exhausted query=%s", query)
                    return []
                wait = min(MAX_BACKOFF_S, (2**attempts) + random.uniform(0, 1))
                LOGGER.warning("search_ratelimited query=%s attempt=%s waiting=%.1fs", query, attempts, wait)
                time.sleep(wait)
                continue
            except Exception as exc:
                LOGGER.warning("search_failed query=%s err=%s", query, exc)
                return []

    @staticmethod
    def _format(results: list[dict[str, Any]]) -> str:
        lines = ["web search results — summarize this info in your reply:"]
        for i, r in enumerate(results, 1):
            title = str(r.get("title", "")).replace("\n", " ").strip()
            body = str(r.get("body", "")).replace("\n", " ").strip()
            url = str(r.get("href") or r.get("url") or "").strip()
            if len(body) > 220:
                body = body[:217].rstrip() + "..."
            lines.append(f"{i}. {title}")
            if url:
                lines.append(f"   url: {url}")
            if body:
                lines.append(f"   {body}")
        return "\n".join(lines)
