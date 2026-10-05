from __future__ import annotations

import re

from .config import ChatConfig

# Prefix-bot commands seen in the data: Mudae `$wg`, Waguri `!k`.
_COMMAND = re.compile(r"^[$!]\S")
# Custom emoji, mentions, links: a message made only of these has nothing to reply to.
_NOISE = re.compile(r"<a?:\w+:\d+>|<[@#][!&]?\d+>|@(everyone|here)|https?://\S+")
_LETTER = re.compile(r"[^\W\d_]")


def is_command(content: str) -> bool:
    return bool(_COMMAND.match(content))


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
    if not _LETTER.search(_NOISE.sub("", content)):
        return "no_text"
    return None
