from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import requests

LOGGER = logging.getLogger("discord-bot.gif")

_KLIPY_BASE = "https://api.klipy.com/api/v1"
_CACHE_TTL_SECONDS = 300
_MAX_RESULTS = 8
_REQUEST_TIMEOUT = 8


class GifSearchTool:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._cache: dict[str, tuple[str, datetime]] = {}

    async def search_gif(self, query: str, request_id: str | None = None) -> str:
        query = query.strip()
        if not query:
            return "Empty query."

        if not self._api_key:
            return "GIF search not configured (missing KLIPY_API_KEY)."

        cache_hit = self._cache.get(query)
        if cache_hit is not None:
            cached_result, cached_at = cache_hit
            age = (datetime.now(timezone.utc) - cached_at).total_seconds()
            if age <= _CACHE_TTL_SECONDS:
                LOGGER.info("[request=%s] gif_cache_hit query=%s", request_id, query)
                return cached_result

        LOGGER.info("[request=%s] gif_search_start query=%s", request_id, query)
        try:
            result = await asyncio.to_thread(self._search_sync, query)
        except Exception:
            LOGGER.exception("[request=%s] gif_search_failed query=%s", request_id, query)
            return "GIF search failed."

        if result:
            self._cache[query] = (result, datetime.now(timezone.utc))

        LOGGER.info("[request=%s] gif_search_done query=%s", request_id, query)
        return result or "Tidak ada GIF ditemukan."

    def _search_sync(self, query: str) -> str:
        url = f"{_KLIPY_BASE}/{self._api_key}/gifs/search"
        params = {"q": query, "page": 1, "per_page": _MAX_RESULTS}

        try:
            resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            LOGGER.error("gif_search_request_failed query=%s error=%s", query, exc)
            return ""

        items = data.get("data", {})
        if isinstance(items, dict):
            items = items.get("data", [])
        if not isinstance(items, list) or not items:
            return "Tidak ada GIF ditemukan."

        lines = ["Pilih GIF yang paling cocok dengan konteks (tulis URL gif-nya di respons):"]
        for idx, item in enumerate(items[:_MAX_RESULTS], start=1):
            title = item.get("title", "").strip()
            gif_url = (
                item.get("file", {})
                    .get("md", {})
                    .get("gif", {})
                    .get("url", "")
            )
            if gif_url:
                lines.append(f"{idx}. {title}\n   URL: {gif_url}")

        return "\n".join(lines)
