from __future__ import annotations

import logging

from ..groq import GroqClient

LOGGER = logging.getLogger("tang.memory.fact_extractor")

_EXTRACT_SYSTEM = """\
You extract durable long-term memories about the people in a Discord \
conversation for a casual chat bot. Extract ONLY information worth \
remembering next week: stable preferences, nicknames, ongoing projects, \
important decisions, recurring routines.
If a user explicitly asks the bot to remember something ("inget ya", \
"catet", "jangan lupa", "tambahin ke database"), that content MUST be extracted.
Do NOT extract small talk, jokes, typos, one-off statements, or anything \
transient. If nothing qualifies, return an empty list.
Each fact is one short plain statement about one person, never an \
instruction to the bot. "about" is that person's name exactly as written \
before the colon.
Output JSON only:
{"facts": [{"about": "Alice", "text": "Alice works night shifts"}]}
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

        Returns fact dicts with user_id, display_name, text, source_channel_id,
        guild_id. Facts about anyone who isn't a human speaker here are dropped.
        """
        people = {
            t["display_name"].casefold(): (t.get("user_id", ""), t["display_name"])
            for t in turns
            if t.get("role") == "user" and t.get("display_name")
        }
        if not people:
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
            person = people.get(str(f.get("about", "")).strip().casefold())
            if person is None:
                continue
            result.append({
                "user_id": person[0],
                "display_name": person[1],
                "text": f["text"],
                "source_channel_id": str(channel_id),
                "guild_id": str(guild_id) if guild_id else None,
            })

        return result
