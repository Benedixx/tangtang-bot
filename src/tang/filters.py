from __future__ import annotations

from .config import ChatConfig
from .queue import ChannelState


def tier0_reason(
    message,
    config: ChatConfig,
    state: ChannelState,
    forced: bool,
    now: float,
) -> str | None:
    """Return a drop reason, or None to keep the message as a trigger."""
    if message.author.bot or message.webhook_id is not None:
        return "bot"
    content = (message.content or "").strip()
    if len(content) < config.min_length:
        return "too_short"
    if not forced and state.last_reply_ts and now - state.last_reply_ts < config.cooldown_s:
        return "cooldown"
    return None
