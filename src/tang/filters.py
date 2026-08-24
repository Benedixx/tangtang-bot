from __future__ import annotations

from .config import ChatConfig


def tier0_reason(
    message,
    config: ChatConfig,
) -> str | None:
    """Return a drop reason, or None to keep the message as a trigger."""
    if message.author.bot or message.webhook_id is not None:
        return "bot"
    content = (message.content or "").strip()
    if len(content) < config.min_length:
        return "too_short"
    return None
