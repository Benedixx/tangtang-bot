from __future__ import annotations

import re
from collections.abc import AsyncGenerator

_GIF_URL_RE = re.compile(r"^https?://\S+\.gif$", re.IGNORECASE)
# Sentence-ending punctuation followed by whitespace, or double newline
_SENTENCE = re.compile(r"([.!?]+\s+|\n\n+)")
# Abbreviations that shouldn't trigger a sentence split
_ABBR = re.compile(
    r"\b(mr|mrs|ms|dr|prof|sr|jr|vs|etc|dept|approx|st|ave|inc|ltd|co)\.$",
    re.IGNORECASE,
)


async def bubbles(
    stream: AsyncGenerator[str, None],
    bot_prefix: str = "!k",
) -> AsyncGenerator[str, None]:
    """Split an LLM text stream into natural conversational bubbles.

    Each bubble is 1-2 sentences — conversational, not wall-of-text.
    GIF URLs get their own dedicated bubble.
    When ``bot_prefix`` is set, yields ``"!k\\n<text>"`` (prefix on its own line).
    """
    buf = ""
    async for chunk in stream:
        buf += chunk
        while True:
            buf = buf.lstrip()
            if not buf:
                break

            # Standalone GIF at the start of buffer
            gif = _GIF_URL_RE.match(buf)
            if gif:
                before = buf[: gif.start()].strip()
                if before and len(before) >= 3:
                    yield f"{bot_prefix}\n{before}"
                yield f"{bot_prefix}\n{gif.group(0)}"
                buf = buf[gif.end():].lstrip()
                continue

            # Find sentence boundary
            m = _SENTENCE.search(buf)
            if not m:
                break

            # Abbreviation check — "Dr." or "Mr." etc.
            pre = buf[: m.start()].strip()
            last_word = pre.split()[-1] if pre.split() else ""
            if last_word and _ABBR.match(last_word + "."):
                for i in range(m.start(), m.end()):
                    if buf[i] in ".!?":
                        buf = buf[:i] + buf[i + 1:]
                        break
                continue

            sentence = buf[: m.end()].strip()
            buf = buf[m.end():].lstrip()

            if len(sentence) >= 5:
                yield f"{bot_prefix}\n{sentence}"
                continue

            # Very short fragment — merge back and wait for more content
            buf = (sentence + " " + buf).strip()
            break

    remainder = buf.strip()
    if remainder and (len(remainder) >= 3 or _GIF_URL_RE.match(remainder)):
        yield f"{bot_prefix}\n{remainder}"
