from __future__ import annotations

import logging

from pydantic import ValidationError

from ..groq import GroqClient
from ..queue import Msg
from .models import ExtractionResult, MemoryCandidate

LOGGER = logging.getLogger("tang.memory.extractor")

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
{"memories": [{"type": "semantic|episodic|procedural", "content": "one short sentence", \
"importance": 0.0-1.0, "confidence": 0.0-1.0}]}
Be conservative — an empty list is the correct answer most of the time."""


class Extractor:
    def __init__(self, client: GroqClient) -> None:
        self._client = client

    async def extract(self, msgs: list[Msg]) -> list[MemoryCandidate]:
        if not msgs:
            return []
        lines = "\n".join(f"{m.author_name}: {m.content}" for m in msgs)
        try:
            data = await self._client.complete_json(
                [
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user", "content": lines},
                ],
                temperature=0.0,
                max_tokens=250,
                label="memory_extract",
            )
        except Exception:
            LOGGER.exception("extract_call_failed")
            return []
        if not data:
            return []
        try:
            result = ExtractionResult.model_validate(data)
        except ValidationError:
            LOGGER.warning("extract_parse_failed keys=%s", list(data))
            return []
        return result.memories
