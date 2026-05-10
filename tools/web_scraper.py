from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger("discord-bot.scraper")

_MAX_OUTPUT_CHARS = 1500
_REQUEST_TIMEOUT = 10
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class WebScraperTool:
    async def scrape_url(self, url: str, request_id: str | None = None) -> str:
        url = url.strip()
        if not url:
            return "Empty URL."

        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return f"Unsupported URL scheme: {parsed.scheme}"
        except Exception:
            return "Invalid URL."

        LOGGER.info("[request=%s] scraper_start url=%s", request_id, url[:100])
        try:
            result = await asyncio.to_thread(self._scrape_sync, url)
        except Exception:
            LOGGER.exception("[request=%s] scraper_failed url=%s", request_id, url[:100])
            return "Failed to fetch URL."

        LOGGER.info("[request=%s] scraper_done url=%s chars=%s", request_id, url[:100], len(result))
        return result

    @staticmethod
    def _scrape_sync(url: str) -> str:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            return f"Request failed: {exc}"

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and isinstance(meta_tag, object):
            meta_desc = (meta_tag.get("content") or "").strip()  # type: ignore[union-attr]

        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        body_text = " ".join(p for p in paragraphs if len(p) > 40)

        parts: list[str] = []
        if title:
            parts.append(f"Judul: {title}")
        if meta_desc:
            parts.append(f"Deskripsi: {meta_desc}")
        if body_text:
            parts.append(f"Isi:\n{body_text}")

        full = "\n\n".join(parts)
        if len(full) > _MAX_OUTPUT_CHARS:
            full = full[:_MAX_OUTPUT_CHARS - 3].rstrip() + "..."
        return full or "Tidak ada konten yang bisa diekstrak."
