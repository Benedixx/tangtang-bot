from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import re
from typing import Any

from ddgs import DDGS

LOGGER = logging.getLogger("discord-bot.webtool")


class AdaptiveWebSearchTool:
    def __init__(
        self,
        enabled: bool = True,
        max_results: int = 4,
        min_interval_seconds: int = 20,
    ) -> None:
        self._enabled = enabled
        self._max_results = max_results
        self._min_interval_seconds = min_interval_seconds
        self._cache: dict[str, tuple[str, datetime]] = {}
        self._last_search_at: datetime | None = None

    async def maybe_search(
        self,
        *,
        latest_message: str,
        recent_history: str,
        request_id: str | None = None,
    ) -> tuple[str, str]:
        if not self._enabled:
            return "", "disabled"

        if not self._should_search(latest_message, recent_history):
            return "", "not_needed"

        query = self._build_query(latest_message)
        cache_hit = self._cache.get(query)
        if cache_hit is not None:
            cached_context, cached_at = cache_hit
            age_seconds = int((self._now() - cached_at).total_seconds())
            if age_seconds <= 900:
                LOGGER.info("[request=%s] webtool_cache_hit query=%s age_seconds=%s", request_id, query, age_seconds)
                return cached_context, "cache_hit"

        if self._is_rate_limited():
            return "", "cooldown"

        LOGGER.info("[request=%s] webtool_search_start query=%s", request_id, query)
        results = await asyncio.to_thread(self._search_sync, query)
        self._last_search_at = self._now()

        if not results:
            return "", "no_results"

        context = self._format_results(results)
        self._cache[query] = (context, self._now())
        LOGGER.info("[request=%s] webtool_search_done query=%s result_count=%s", request_id, query, len(results))
        return context, "searched"

    def _should_search(self, latest_message: str, recent_history: str) -> bool:
        message = latest_message.lower().strip()
        if len(message) < 10:
            return False

        # Avoid over-searching for purely relational/persona chatter.
        personal_markers = {"suami", "istri", "pacar", "sayang", "jodoh", "kamu siapa"}
        if any(marker in message for marker in personal_markers):
            return False

        web_intent_keywords = {
            "berita",
            "news",
            "terbaru",
            "update",
            "rilis",
            "harga",
            "tanggal",
            "kapan",
            "siapa",
            "apa itu",
            "github",
            "repo",
            "dokumentasi",
            "docs",
            "referensi",
            "sumber",
            "tutorial",
            "cara",
            "trend",
        }

        if any(keyword in message for keyword in web_intent_keywords):
            return True

        # If conversation repeatedly asks factual follow-ups, allow search.
        joined = f"{recent_history.lower()}\n{message}"
        factual_question_count = len(re.findall(r"\b(apa|siapa|kapan|berapa|dimana|gimana|bagaimana)\b", joined))
        return factual_question_count >= 2

    def _build_query(self, message: str) -> str:
        compact = re.sub(r"\s+", " ", message).strip()
        return compact[:220]

    def _search_sync(self, query: str) -> list[dict[str, Any]]:
        try:
            with DDGS() as ddgs:
                rows = list(ddgs.text(query, max_results=self._max_results))
        except Exception:
            LOGGER.exception("DDGS search failed for query=%s", query)
            return []

        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title", "")).strip()
            body = str(row.get("body", "")).strip()
            href = str(row.get("href", "")).strip()
            if not (title or body):
                continue
            normalized.append({"title": title, "body": body, "href": href})
        return normalized

    @staticmethod
    def _format_results(results: list[dict[str, Any]]) -> str:
        lines = ["Hasil web search relevan:"]
        for idx, row in enumerate(results, start=1):
            title = str(row.get("title", "")).replace("\n", " ").strip()
            body = str(row.get("body", "")).replace("\n", " ").strip()
            href = str(row.get("href", "")).strip()

            if len(body) > 220:
                body = body[:217].rstrip() + "..."

            lines.append(f"{idx}. {title}")
            if body:
                lines.append(f"   Ringkasan: {body}")
            if href:
                lines.append(f"   Link: {href}")

        return "\n".join(lines)

    def _is_rate_limited(self) -> bool:
        if self._last_search_at is None:
            return False
        elapsed = (self._now() - self._last_search_at).total_seconds()
        return elapsed < self._min_interval_seconds

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
