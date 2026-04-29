from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

from models import ChannelState, ChatMessage


class DialogAnalyzer:
    def __init__(self, max_context_messages: int = 20) -> None:
        if max_context_messages <= 0:
            raise ValueError("max_context_messages must be greater than zero")

        self._max_context_messages = max_context_messages
        self._states: dict[int, ChannelState] = {}

    def get_state(self, channel_id: int) -> ChannelState:
        state = self._states.get(channel_id)
        if state is None:
            state = ChannelState(messages=deque(maxlen=self._max_context_messages))
            self._states[channel_id] = state
        return state

    def add_message(self, channel_id: int, chat_message: ChatMessage) -> ChannelState:
        state = self.get_state(channel_id)
        state.messages.append(chat_message)
        state.total_message_count += 1

        if not chat_message.is_bot:
            state.message_count_since_summary += 1
            current = state.participant_message_count.get(chat_message.author_id, 0)
            state.participant_message_count[chat_message.author_id] = current + 1
            state.participant_names[chat_message.author_id] = chat_message.author_name

        return state

    def mark_bot_message(self, channel_id: int, message_id: int) -> None:
        state = self.get_state(channel_id)
        state.bot_message_ids.add(message_id)
        state.last_response_at = datetime.now(timezone.utc)

    def set_summary(self, channel_id: int, summary: str) -> None:
        state = self.get_state(channel_id)
        cleaned = summary.strip()
        if cleaned:
            state.summary = cleaned
        state.message_count_since_summary = 0

    @staticmethod
    def format_recent_history(state: ChannelState, limit: int = 20) -> str:
        lines: list[str] = []
        for msg in list(state.messages)[-limit:]:
            speaker = "BOT" if msg.is_bot else msg.author_name
            content = msg.content.replace("\n", " ").strip()
            if len(content) > 220:
                content = content[:217].rstrip() + "..."
            lines.append(f"{speaker}: {content}")

        return "\n".join(lines) if lines else "(no recent messages)"

    @staticmethod
    def format_participant_snapshot(state: ChannelState) -> str:
        if not state.participant_message_count:
            return "No participant activity yet."

        sorted_counts = sorted(
            state.participant_message_count.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        chunks: list[str] = []
        for user_id, count in sorted_counts:
            name = state.participant_names.get(user_id, str(user_id))
            chunks.append(f"{name}={count}")

        return ", ".join(chunks)
