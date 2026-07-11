from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .models import ChannelState, TriggerType

if TYPE_CHECKING:
    from llm.groq import GroqClient

LOGGER = logging.getLogger("tang.engage")

_ENGAGE_PROMPT = """\
You are a conversation scheduler for a Discord chat AI.

Decide if the AI should chime in based on the conversation context.
The AI should speak when:
- Someone asks a question (even indirectly)
- The AI can contribute useful information
- The conversation is heading toward something the AI can help with
- Someone seems to need assistance

The AI should stay silent when:
- It's casual banter between users
- Inside jokes or personal conversation
- The message is too vague or unclear

Reply with exactly one word: YES or NO.
Output nothing else."""


async def should_engage(
    trigger: TriggerType,
    state: ChannelState,
    message: str,
    groq: GroqClient | None,
    cooldown_seconds: int = 12,
    request_id: str | None = None,
) -> tuple[bool, str]:
    """Decide whether the bot should respond. Returns (engage, reason)."""

    # -- Hard triggers: always engage --
    if trigger in (
        TriggerType.DIRECT_MENTION,
        TriggerType.REPLY_TO_BOT,
        TriggerType.NAME_MENTION,
        TriggerType.BOT_PREFIX,
    ):
        return True, trigger.value

    # -- Cooldown --
    if state.last_response_at is not None:
        elapsed = (datetime.now(timezone.utc) - state.last_response_at).total_seconds()
        if elapsed < cooldown_seconds:
            return False, "cooldown"

    # -- Trivial message --
    if len(message.strip()) < 3:
        return False, "too_short"

    # -- Need summary check --
    if state.summary_trigger > 0 and state.total_count - state.summary_trigger >= 30:
        if state.total_count >= 10:
            return True, "summary_due"

    # -- LLM-based engagement decision (contextual) --
    if groq is None:
        return False, "no_groq"

    try:
        recent = "\n".join(
            f"{'BOT' if m.is_bot else m.author_name}: {m.content}"
            for m in list(state.messages)[-10:]
        )
        user_msg = f"Recent messages:\n{recent}\n\nLatest: {message}\n\nShould the AI chime in?"

        result = await groq.complete(
            messages=[
                {"role": "system", "content": _ENGAGE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=5,
            request_id=request_id,
            request_label="engage",
        )
    except Exception:
        LOGGER.exception("[%s] engage_error", request_id or "-")
        return False, "error"

    decision = result.strip().upper()
    if decision.startswith("YES"):
        LOGGER.info("[%s] engage=yes reason=llm", request_id or "-")
        return True, "llm_yes"
    if decision.startswith("NO"):
        LOGGER.info("[%s] engage=no reason=llm", request_id or "-")
        return False, "llm_no"

    LOGGER.info("[%s] engage=no reason=unclear", request_id or "-")
    return False, "unclear"
