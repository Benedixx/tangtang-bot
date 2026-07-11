from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TriggerType(Enum):
    """What triggered the bot to consider responding."""

    DIRECT_MENTION = "direct_mention"
    REPLY_TO_BOT = "reply_to_bot"
    NAME_MENTION = "name_mention"
    BOT_PREFIX = "bot_prefix"  # inter-bot !k protocol
    CONTEXTUAL = "contextual"  # any channel message


@dataclass(slots=True)
class ChatMessage:
    message_id: int
    author_id: int
    author_name: str
    content: str
    created_at: datetime
    is_bot: bool = False


@dataclass(slots=True)
class ChannelState:
    messages: deque[ChatMessage] = field(
        default_factory=lambda: deque(maxlen=40)
    )
    total_count: int = 0
    summary: str = ""
    summary_trigger: int = 0  # message count at last summary
    bot_message_ids: set[int] = field(default_factory=set)
    last_response_at: datetime | None = None
    participants: dict[int, str] = field(default_factory=dict)  # id → name
