from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import discord
from rapidfuzz import fuzz

LOGGER = logging.getLogger("tang.tools.gif")

# Discord's free-server attachment limit is ~10 MB; files right at the
# boundary 413 unpredictably, and big GIFs are pointless as reactions anyway.
MAX_FILE_BYTES = 10 * 1024 * 1024

GIF_TAGS = [
    "laugh",
    "confused",
    "facepalm",
    "hype",
    "shrug",
    "cry",
    "love",
    "angry",
    "awkward",
    "happy",
    "sad",
    "surprised",
    "tired",
    "thinking",
    "celebrate",
    "sip",
    "nope",
    "deal",
]

# Explicit tag mapping for files whose names don't contain GIF_TAGS tokens.
_TAG_MAP: dict[str, list[str]] = {
    "dance": ["happy", "celebrate", "hype"],
    "smile": ["happy", "love"],
    "mad": ["angry"],
    "pat-pat": ["love"],
    "peace": ["sip", "deal"],
    "show-up": ["hype"],
    "useless-info": ["awkward", "confused"],
    "here-take-this": ["nope", "shrug"],
    "who-is-this-kid": ["confused", "surprised", "thinking"],
}

class GifStore:
    """Local GIF store. Uploads files once to a staging channel, keeps a manifest."""

    def __init__(self, gif_dir: str, manifest_path: str) -> None:
        self._dir = Path(gif_dir)
        self._manifest_path = Path(manifest_path)
        self._entries: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self._manifest_path.exists():
            return
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.exception("gif_manifest_load_failed path=%s", self._manifest_path)
            return
        if isinstance(data, list):
            changed = False
            self._entries = []
            for e in data:
                if not isinstance(e, dict) or not e.get("cdn_url"):
                    continue
                if not e.get("tags"):
                    e["tags"] = self._tags_for(str(e.get("id", "")))
                    changed = True
                self._entries.append(e)
            if changed:
                self._save()

    def _save(self) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(json.dumps(self._entries, indent=2), encoding="utf-8")

    async def upload_all(self, client: discord.Client, staging_channel_id: int) -> None:
        if not self._dir.exists():
            LOGGER.info("gif_dir_missing dir=%s", self._dir)
            return

        channel = None
        if staging_channel_id:
            channel = client.get_channel(staging_channel_id)
            if channel is None:
                try:
                    channel = await client.fetch_channel(staging_channel_id)
                except discord.HTTPException:
                    channel = None
        if channel is None:
            LOGGER.warning(
                "gif_staging_channel_not_found id=%s — serving cached manifest only",
                staging_channel_id,
            )
            return

        known = {e.get("id") for e in self._entries}
        for path in sorted(self._dir.glob("*.gif")):
            if path.stem in known:
                continue
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                LOGGER.warning(
                    "gif_skipped_too_large file=%s size_mb=%.1f limit_mb=%.0f — compress it (e.g. gifsicle -O3)",
                    path.name, size / (1024 * 1024), MAX_FILE_BYTES / (1024 * 1024),
                )
                continue
            try:
                sent = await channel.send(file=discord.File(path))
            except discord.HTTPException:
                LOGGER.exception(
                    "gif_upload_failed file=%s size_mb=%.1f",
                    path.name, size / (1024 * 1024),
                )
                continue
            if not sent.attachments:
                continue
            cdn_url = sent.attachments[0].url
            self._entries.append({
                "id": path.stem,
                "tags": self._tags_for(path.stem),
                "cdn_url": cdn_url,
            })
            LOGGER.info("gif_uploaded id=%s", path.stem)
        self._save()

    @staticmethod
    def _tags_for(stem: str) -> list[str]:
        tags = list(_TAG_MAP.get(stem, ()))
        tokens = [t for t in stem.replace("_", "-").split("-") if t]
        for tok in tokens:
            if tok in GIF_TAGS and tok not in tags:
                tags.append(tok)
        return tags

    def contains(self, url: str) -> bool:
        return any(e["cdn_url"] == url for e in self._entries)

    async def send(self, tag: str, recent: list[str], request_id: str | None = None) -> str:
        """Pick a GIF URL: exact tag, fuzzy, shrug, then nothing."""
        if not self._entries:
            return ""

        pool = [e for e in self._entries if tag in e.get("tags", ())]
        if not pool:
            best = self._fuzzy_entry(tag)
            if best is not None:
                pool = [best]
            else:
                pool = [e for e in self._entries if "shrug" in e.get("tags", ())]
        if not pool:
            return ""

        candidates = [e for e in pool if e["cdn_url"] not in recent]
        if not candidates:
            candidates = pool
        entry = random.choice(candidates)
        LOGGER.info("[%s] gif_send tag=%s id=%s", request_id or "-", tag, entry["id"])
        return entry["cdn_url"]

    def _fuzzy_entry(self, tag: str) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_score = 0.0
        for entry in self._entries:
            for t in entry.get("tags", ()):
                score = fuzz.ratio(tag.lower(), t.lower())
                if score > best_score:
                    best_score = score
                    best = entry
        return best if best_score >= 60 else None
