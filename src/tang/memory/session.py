from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .models import SessionSummary


@dataclass(slots=True)
class ChannelSession:
    """Mid-term state for one channel: rolling summary + compaction bookkeeping."""

    summary: SessionSummary | None = None
    summarized_ids: set[int] = field(default_factory=set)
    replies_since_extract: int = 0
    idle_handle: asyncio.TimerHandle | None = None

    def reset_summary(self) -> None:
        self.summary = None
        self.summarized_ids.clear()


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[int, ChannelSession] = {}

    def get(self, channel_id: int) -> ChannelSession:
        session = self._sessions.get(channel_id)
        if session is None:
            session = self._sessions[channel_id] = ChannelSession()
        return session

    def touch(self, channel_id: int, delay_s: float, on_idle) -> None:
        """(Re)start the idle timer that triggers long-term memory extraction."""
        session = self.get(channel_id)
        if session.idle_handle is not None:
            session.idle_handle.cancel()
        loop = asyncio.get_running_loop()
        session.idle_handle = loop.call_later(delay_s, self._fire, channel_id, on_idle)

    def cancel(self, channel_id: int) -> None:
        session = self._sessions.get(channel_id)
        if session is not None and session.idle_handle is not None:
            session.idle_handle.cancel()
            session.idle_handle = None

    def pending_extraction(self) -> list[int]:
        return [
            cid for cid, s in self._sessions.items() if s.replies_since_extract > 0
        ]

    def _fire(self, channel_id: int, on_idle) -> None:
        session = self._sessions.get(channel_id)
        if session is not None:
            session.idle_handle = None
        asyncio.get_running_loop().create_task(on_idle(channel_id))
