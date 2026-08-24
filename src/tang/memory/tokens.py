from __future__ import annotations

import logging

LOGGER = logging.getLogger("tang.tokens")

_encoding = None
try:
    import tiktoken

    _encoding = tiktoken.get_encoding("o200k_base")
except Exception:
    LOGGER.warning("tiktoken_unavailable using_fallback_estimator")


def count_tokens(text: str) -> int:
    """Token count for budgeting. Exact via o200k_base (gpt-oss family),
    char/word estimator if tiktoken is unavailable."""
    if not text:
        return 0
    if _encoding is not None:
        return len(_encoding.encode(text))
    return max(len(text) // 4, len(text.split()) + len(text) // 12)
