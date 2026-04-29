from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Deque


class TriggerType(str, Enum):
    DIRECT_MENTION = "direct_mention"
    REPLY_TO_BOT = "reply_to_bot"
    NAME_MENTION = "name_mention"
    CONTEXTUAL = "contextual"


class ConversationStrategy(str, Enum):
    DIRECT_CHAT = "direct_chat"
    INITIATIVE_SUMMARY = "initiative_summary"
    IN_CONTEXT_CHIME_IN = "in_context_chime_in"
    KEEP_SILENT = "keep_silent"


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
    messages: Deque[ChatMessage] = field(default_factory=deque)
    summary: str = "No summary available yet."
    total_message_count: int = 0
    message_count_since_summary: int = 0
    participant_message_count: dict[int, int] = field(default_factory=dict)
    participant_names: dict[int, str] = field(default_factory=dict)
    bot_message_ids: set[int] = field(default_factory=set)
    last_response_at: datetime | None = None
