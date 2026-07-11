from __future__ import annotations

import logging
from typing import Any

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger("tang.tools.scraper")
_TIMEOUT = 10


TOOL_NAME = "scrape_url"
TOOL_DESCRIPTION = "Read the content of a URL or link shared by the user."
TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to scrape"},
            },
            "required": ["url"],
        },
    },
}


class WebScraper:
    """Simple URL scraper that extracts readable text content."""

    async def execute(self, url: str, request_id: str | None = None) -> str:
        url = url.strip()
        if not url:
            return "Empty URL."

        LOGGER.info("[%s] scrape_start url=%s", request_id or "-", url[:120])

        try:
            resp = await self._fetch(url)
        except Exception:
            LOGGER.exception("[%s] scrape_failed url=%s", request_id or "-", url[:120])
            return "Failed to fetch the page."

        text = self._extract(resp)
        if len(text) > 2000:
            text = text[:1997] + "..."

        LOGGER.info("[%s] scrape_done url=%s chars=%s", request_id or "-", url[:120], len(text))
        return text

    async def _fetch(self, url: str) -> str:
        """Fetch URL content in a thread to avoid blocking."""
        import asyncio
        return await asyncio.to_thread(self._fetch_sync, url)

    def _fetch_sync(self, url: str) -> str:
        resp = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.text

    def _extract(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
