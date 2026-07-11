from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from ddgs import DDGS

LOGGER = logging.getLogger("tang.tools.search")
_MAX_RESULTS = 4


TOOL_NAME = "web_search"
TOOL_DESCRIPTION = (
    "Search the web for current information: news, prices, facts, articles, docs. "
    "Use when the user asks about recent events, factual questions, or needs up-to-date info."
)
TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific search query — concise and precise",
                }
            },
            "required": ["query"],
        },
    },
}


class WebSearch:
    """DuckDuckGo web search with rate limiting and caching."""

    def __init__(self, enabled: bool = True, max_results: int = 4, min_interval: int = 20) -> None:
        self._enabled = enabled
        self._max_results = max_results or _MAX_RESULTS
        self._min_interval = min_interval
        self._last_search: datetime | None = None
        self._cache: dict[str, tuple[str, datetime]] = {}

    async def execute(self, query: str, request_id: str | None = None) -> str:
        if not self._enabled:
            return "Web search is disabled."

        query = query.strip()
        if not query:
            return "Empty query."

        # Cache check (15 min TTL)
        hit = self._cache.get(query)
        if hit:
            text, cached_at = hit
            if (datetime.now(timezone.utc) - cached_at).total_seconds() <= 900:
                LOGGER.info("[%s] search_cache_hit query=%s", request_id or "-", query)
                return text

        if self._rate_limited():
            return "Search rate limited — try again in a moment."

        LOGGER.info("[%s] search_start query=%s", request_id or "-", query)
        results = await asyncio.to_thread(self._search_sync, query)
        self._last_search = datetime.now(timezone.utc)

        if not results:
            return "No results found."

        text = self._format(results)
        self._cache[query] = (text, datetime.now(timezone.utc))
        LOGGER.info("[%s] search_done query=%s results=%s", request_id or "-", query, len(results))
        return text

    def _search_sync(self, query: str) -> list[dict[str, Any]]:
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=self._max_results))
        except Exception:
            LOGGER.exception("DDGS search failed")
            return []

    def _format(self, results: list[dict[str, Any]]) -> str:
        lines = ["Web search results:"]
        for i, r in enumerate(results, 1):
            title = str(r.get("title", "")).replace("\n", " ").strip()
            body = str(r.get("body", "")).replace("\n", " ").strip()
            href = str(r.get("href", "")).strip()
            if len(body) > 220:
                body = body[:217].rstrip() + "..."
            lines.append(f"{i}. {title}")
            if body:
                lines.append(f"   {body}")
            if href:
                lines.append(f"   {href}")
        return "\n".join(lines)

    def _rate_limited(self) -> bool:
        if self._last_search is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self._last_search).total_seconds()
        return elapsed < self._min_interval
