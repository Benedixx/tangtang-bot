from __future__ import annotations

from datetime import datetime, timezone
import logging

from llm import OpenRouterClient
from models import ChannelState, ConversationStrategy, TriggerType


_HARD_TRIGGERS = {
    TriggerType.DIRECT_MENTION,
    TriggerType.REPLY_TO_BOT,
    TriggerType.NAME_MENTION,
}

LOGGER = logging.getLogger("discord-bot.scheduler")


class SauceScheduler:
    def __init__(self, llm: OpenRouterClient, cooldown_seconds: int = 12) -> None:
        self._llm = llm
        self._cooldown_seconds = cooldown_seconds

    async def should_engage(
        self,
        *,
        trigger_type: TriggerType,
        strategy: ConversationStrategy,
        state: ChannelState,
        latest_message: str,
        recent_history: str,
        request_id: str | None = None,
    ) -> tuple[bool, str]:
        if strategy == ConversationStrategy.KEEP_SILENT:
            LOGGER.info("[request=%s] scheduler_decision=false reason=strategy_keep_silent", request_id)
            return False, "strategy_keep_silent"

        if trigger_type in _HARD_TRIGGERS:
            LOGGER.info("[request=%s] scheduler_decision=true reason=hard_trigger", request_id)
            return True, "hard_trigger"

        if strategy == ConversationStrategy.INITIATIVE_SUMMARY:
            LOGGER.info("[request=%s] scheduler_decision=true reason=scheduled_summary", request_id)
            return True, "scheduled_summary"

        if self._is_cooldown_active(state):
            LOGGER.info("[request=%s] scheduler_decision=false reason=cooldown", request_id)
            return False, "cooldown"

        if trigger_type == TriggerType.CONTEXTUAL and self._looks_like_question(latest_message):
            LOGGER.info("[request=%s] scheduler_decision=true reason=contextual_question_signal", request_id)
            return True, "contextual_question_signal"

        if len(latest_message.strip()) < 2:
            LOGGER.info("[request=%s] scheduler_decision=false reason=message_too_short", request_id)
            return False, "message_too_short"

        system_prompt = (
            "Kamu adalah scheduler percakapan untuk asisten Discord. "
            "Tentukan apakah bot perlu berbicara sekarang. "
            "Utamakan diam kecuali respons bot benar-benar memberi nilai tambah yang jelas. "
            "Output harus tepat satu baris: YES | alasan singkat ATAU NO | alasan singkat."
        )

        user_prompt = (
            "Percakapan terbaru:\n"
            f"{recent_history}\n\n"
            "Pesan user terbaru:\n"
            f"{latest_message}\n\n"
            "Apakah asisten perlu berbicara sekarang?"
        )

        raw_decision = await self._llm.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=60,
            request_id=request_id,
            request_label="scheduler",
        )

        decision = raw_decision.strip()
        upper = decision.upper()

        if upper.startswith("YES"):
            LOGGER.info("[request=%s] scheduler_decision=true reason=%s", request_id, decision)
            return True, decision
        if upper.startswith("NO"):
            LOGGER.info("[request=%s] scheduler_decision=false reason=%s", request_id, decision)
            return False, decision

        if '"ENGAGE":"YES"' in upper or '"ENGAGE": "YES"' in upper:
            LOGGER.info("[request=%s] scheduler_decision=true reason=%s", request_id, decision)
            return True, decision
        if '"ENGAGE":"NO"' in upper or '"ENGAGE": "NO"' in upper:
            LOGGER.info("[request=%s] scheduler_decision=false reason=%s", request_id, decision)
            return False, decision

        LOGGER.info("[request=%s] scheduler_decision=false reason=unclear_scheduler_output", request_id)
        return False, "unclear_scheduler_output"

    def _is_cooldown_active(self, state: ChannelState) -> bool:
        if state.last_response_at is None:
            return False

        elapsed = (datetime.now(timezone.utc) - state.last_response_at).total_seconds()
        return elapsed < self._cooldown_seconds

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
