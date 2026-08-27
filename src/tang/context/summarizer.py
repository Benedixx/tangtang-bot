from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..groq import GroqClient
from ..memory.tokens import count_tokens
from ..storage.json_store import JsonStore

LOGGER = logging.getLogger("tang.context.summarizer")

_COMPACT_SYSTEM = """\
You maintain a rolling summary of a casual Discord conversation so the bot \
can remember earlier context. You get older messages that are about to be \
dropped. Merge them into the existing summary if one is provided.
Keep only durable information: topic, established facts, decisions, open questions.
Drop small talk, greetings, jokes and repetition.
Output JSON only:
{"summary_text": "merged summary paragraph"}
The summary_text should be a single concise paragraph. Omit empty fields."""


class Summarizer:
    """Rolling summary per channel, persisted as JSON.

    File: data/summaries/{channel_id}.json
    Schema:
    {
        "channel_id": "123",
        "updated_at": "ISO",
        "summary_text": "Alice and Bob are planning..."
    }
    """

    def __init__(self, store: JsonStore, llm: GroqClient) -> None:
        self._store = store
        self._llm = llm
        self._dir = Path(store._root) / "summaries"

    def _path(self, channel_id: int) -> Path:
        return self._dir / f"{channel_id}.json"

    def read(self, channel_id: int) -> str | None:
        data = self._store.read_json(self._path(channel_id), default={})
        return data.get("summary_text") or None

    def write(self, channel_id: int, summary_text: str) -> None:
        self._store.write_json(self._path(channel_id), {
            "channel_id": str(channel_id),
            "updated_at": datetime.now(UTC).isoformat(),
            "summary_text": summary_text,
        })

    def clear(self, channel_id: int) -> None:
        self._store.write_json(self._path(channel_id), {
            "channel_id": str(channel_id),
            "updated_at": datetime.now(UTC).isoformat(),
            "summary_text": "",
        })

    async def maybe_compact(
        self,
        channel_id: int,
        turns: list[dict[str, Any]],
        token_threshold: int = 4000,
        keep_recent: int = 5,
    ) -> bool:
        """Summarize oldest turns if total tokens exceed threshold.

        Returns True if compaction happened.
        """
        if not turns:
            return False

        total_tokens = sum(count_tokens(t.get("content", "")) + 1 for t in turns)
        if total_tokens <= token_threshold:
            return False

        old_turns = turns[:-keep_recent] if len(turns) > keep_recent else []
        if len(old_turns) < 3:
            return False

        old_text = "\n".join(
            f"{t.get('display_name', 'user')}: {t.get('content', '')}"
            for t in old_turns
        )
        existing = self.read(channel_id) or ""

        messages = [
            {"role": "system", "content": _COMPACT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"existing summary: {existing or '(none)'}\n\n"
                    f"older messages to fold in:\n{old_text}"
                ),
            },
        ]

        try:
            data = await self._llm.complete_json(
                messages, temperature=0.0, max_tokens=300, label="summarizer_compact",
            )
        except Exception:
            LOGGER.exception("summarizer_compact_failed channel=%s", channel_id)
            return False

        if not data or "summary_text" not in data:
            LOGGER.warning("summarizer_compact_empty channel=%s", channel_id)
            return False

        new_summary = data["summary_text"]
        if existing:
            new_summary = f"{existing} {new_summary}"

        self.write(channel_id, new_summary)
        LOGGER.info(
            "summarizer_compact_done channel=%s folded=%s",
            channel_id, len(old_turns),
        )
        return True
