from __future__ import annotations

import logging

from pydantic import ValidationError

from ..groq import GroqClient

LOGGER = logging.getLogger("tang.memory.fact_extractor")

_EXTRACT_SYSTEM = """\
You extract durable long-term memories from a Discord conversation for a \
casual chat bot. Extract ONLY information worth remembering next week:
stable preferences, ongoing projects, important decisions, recurring routines.
If a user explicitly asks the bot to remember something ("inget ya", \
"catet", "jangan lupa"), that content MUST be extracted with importance \
and confidence of at least 0.9.
Do NOT extract small talk, jokes, typos, one-off statements, or anything \
transient. If nothing qualifies, return an empty list.
Output JSON only:
{"facts": [{"text": "one short sentence", "source_channel_id": "123", \
"guild_id": "456"}]}
Be conservative — an empty list is the correct answer most of the time."""


class FactExtractor:
    """LLM-based fact extraction from conversation turns."""

    def __init__(self, llm: GroqClient) -> None:
        self._llm = llm

    async def extract(
        self,
        turns: list[dict],
        channel_id: int,
        guild_id: int | None = None,
    ) -> list[dict]:
        """Extract durable facts from conversation turns.

        Returns list of fact dicts with text, source_channel_id, guild_id.
        """
        if not turns:
            return []

        lines = "\n".join(
            f"{t.get('display_name', 'user')}: {t.get('content', '')}"
            for t in turns
        )

        try:
            data = await self._llm.complete_json(
                [
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user", "content": lines},
                ],
                temperature=0.0,
                max_tokens=250,
                label="fact_extract",
            )
        except Exception:
            LOGGER.exception("fact_extract_call_failed")
            return []

        if not data:
            return []

        facts_raw = data.get("facts", [])
        if not isinstance(facts_raw, list):
            return []

        result = []
        for f in facts_raw:
            if not isinstance(f, dict) or not f.get("text"):
                continue
            result.append({
                "text": f["text"],
                "source_channel_id": str(channel_id),
                "guild_id": str(guild_id) if guild_id else None,
            })

        return result
