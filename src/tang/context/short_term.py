from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..storage.json_store import JsonStore

LOGGER = logging.getLogger("tang.context.short_term")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ShortTermBuffer:
    """Per-channel message buffer persisted as JSON.

    File: data/buffers/{channel_id}.json
    Schema:
    {
        "channel_id": "123",
        "guild_id": "456",
        "updated_at": "ISO",
        "turns": [
            {
                "user_id": "111",
                "display_name": "Alice",
                "role": "user|assistant",
                "content": "hey",
                "timestamp": "ISO"
            }
        ]
    }
    """

    def __init__(self, store: JsonStore) -> None:
        self._store = store
        self._dir = Path(store._root) / "buffers"

    def _path(self, channel_id: int) -> Path:
        return self._dir / f"{channel_id}.json"

    def read(self, channel_id: int) -> list[dict[str, Any]]:
        data = self._store.read_json(self._path(channel_id), default={"turns": []})
        return data.get("turns", [])

    def append(
        self,
        channel_id: int,
        guild_id: int | None,
        user_id: str,
        display_name: str,
        role: str,
        content: str,
    ) -> list[dict[str, Any]]:
        """Append a turn and return the updated buffer."""
        path = self._path(channel_id)
        data = self._store.read_json(path, default={
            "channel_id": str(channel_id),
            "guild_id": str(guild_id) if guild_id else None,
            "updated_at": "",
            "turns": [],
        })

        turn = {
            "user_id": user_id,
            "display_name": display_name,
            "role": role,
            "content": content,
            "timestamp": _now_iso(),
        }
        data["turns"].append(turn)
        data["updated_at"] = _now_iso()
        data["channel_id"] = str(channel_id)
        if guild_id is not None:
            data["guild_id"] = str(guild_id)

        self._store.write_json(path, data)
        return data["turns"]

    def trim(self, channel_id: int, keep_last: int = 5) -> list[dict[str, Any]]:
        """Trim buffer to keep only the last N turns. Returns remaining turns."""
        path = self._path(channel_id)
        data = self._store.read_json(path, default={"turns": []})
        turns = data.get("turns", [])
        if len(turns) > keep_last:
            data["turns"] = turns[-keep_last:]
            data["updated_at"] = _now_iso()
            self._store.write_json(path, data)
            return data["turns"]
        return turns

    def clear(self, channel_id: int) -> None:
        path = self._path(channel_id)
        self._store.write_json(path, {
            "channel_id": str(channel_id),
            "guild_id": None,
            "updated_at": _now_iso(),
            "turns": [],
        })

    def count_turns(self, channel_id: int) -> int:
        turns = self.read(channel_id)
        return len(turns)
