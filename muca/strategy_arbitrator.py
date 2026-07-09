from __future__ import annotations

from models import ChannelState, ConversationStrategy, TriggerType


class StrategyArbitrator:
    def __init__(self, summary_interval_messages: int = 30, min_messages_for_summary: int = 8) -> None:
        self._summary_interval_messages = summary_interval_messages
        self._min_messages_for_summary = min_messages_for_summary

    def select_strategy(
        self,
        trigger_type: TriggerType,
        state: ChannelState,
        latest_message: str,
    ) -> ConversationStrategy:
        if trigger_type in {
            TriggerType.DIRECT_MENTION,
            TriggerType.REPLY_TO_BOT,
            TriggerType.NAME_MENTION,
        }:
            return ConversationStrategy.DIRECT_CHAT

        if state.total_message_count < 3 and not self._looks_like_question(latest_message):
            if state.last_response_at is not None:
                return ConversationStrategy.IN_CONTEXT_CHIME_IN
            return ConversationStrategy.KEEP_SILENT

        if (
            state.message_count_since_summary >= self._summary_interval_messages
            and state.total_message_count >= self._min_messages_for_summary
        ):
            return ConversationStrategy.INITIATIVE_SUMMARY

        return ConversationStrategy.IN_CONTEXT_CHIME_IN

    @staticmethod
    def _looks_like_question(message: str) -> bool:
        text = message.strip().lower()
        if not text:
            return False

        if "?" in text:
            return True

        question_markers = (
            "apa",
            "gimana",
            "bagaimana",
            "kenapa",
            "kapan",
            "siapa",
            "menurut",
            "boleh",
            "bisa",
            "setuju",
            "kah",
        )
        return any(text.startswith(marker + " ") for marker in question_markers)
