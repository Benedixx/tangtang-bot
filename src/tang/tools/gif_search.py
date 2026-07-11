from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import requests

LOGGER = logging.getLogger("tang.tools.gif")
_API_BASE = "https://api.klipy.com/api/v1"
_CACHE_TTL = 300
_MAX_RESULTS = 6


TOOL_NAME = "search_gif"
TOOL_DESCRIPTION = (
    "Search for a GIF to send as a reaction. Use when the moment calls for it — "
    "strong reactions, funny situations, dramatic moments. Not for every message."
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
                    "description": "Search query describing the reaction (e.g. 'excited dance', 'facepalm', 'celebrate')",
                }
            },
            "required": ["query"],
        },
    },
}


class GifSearch:
    """Klipy GIF search with caching."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._cache: dict[str, tuple[str, datetime]] = {}

    async def execute(self, query: str, request_id: str | None = None) -> str:
        query = query.strip()
        if not query:
            return "Empty query."
        if not self._api_key:
            return "GIF search not configured."

        hit = self._cache.get(query)
        if hit:
            text, cached_at = hit
            if (datetime.now(timezone.utc) - cached_at).total_seconds() <= _CACHE_TTL:
                LOGGER.info("[%s] gif_cache_hit query=%s", request_id or "-", query)
                return text

        LOGGER.info("[%s] gif_search_start query=%s", request_id or "-", query)
        try:
            result = await asyncio.to_thread(self._search_sync, query)
        except Exception:
            LOGGER.exception("[%s] gif_search_failed", request_id or "-")
            return "GIF search failed."

        if result:
            self._cache[query] = (result, datetime.now(timezone.utc))
        return result or "No GIFs found."

    def _search_sync(self, query: str) -> str:
        url = f"{_API_BASE}/{self._api_key}/gifs/search"
        params = {"q": query, "page": 1, "per_page": _MAX_RESULTS}

        try:
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            LOGGER.error("gif_api_error query=%s error=%s", query, exc)
            return ""

        items = data.get("data", {})
        if isinstance(items, dict):
            items = items.get("data", [])
        if not isinstance(items, list) or not items:
            return ""

        lines = ["GIF options — pick the best match:"]
        for item in items[:_MAX_RESULTS]:
            title = item.get("title", "").strip()
            gif_url = (
                item.get("file", {})
                .get("md", {})
                .get("gif", {})
                .get("url", "")
            )
            if gif_url:
                lines.append(f"- {title}\n  URL: {gif_url}")

        return "\n".join(lines)
