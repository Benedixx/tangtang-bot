from __future__ import annotations

import re

# Casual Indonesian + English phrases that explicitly ask the bot to remember.
# False positives only cost one cheap extraction call, and the extractor is
# conservative anyway — better to over-trigger here than miss a request.
_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bing[ae]t(in|in ya|ya|aja|ajah)?\b",
    r"\bcatet(in)?\b",
    r"\bnyatet\b",
    r"\bsimpen\b",
    r"\bsimpan\b",
    r"\btaro di\b",
    r"\btaruh di\b",
    r"\bdi pikiran\b",
    r"\bdi kepala\b",
    r"\bdalem otak\b",
    r"\bdi otak\b",
    r"\bjangan lupa\b",
    r"\bjangan hilangin\b",
    r"\bremember\b",
    r"\bkeep in mind\b",
    r"\bdon'?t forget\b",
))


def wants_memory(text: str) -> bool:
    """True if the message explicitly asks the bot to remember something."""
    if not text:
        return False
    return any(p.search(text) for p in _PATTERNS)
