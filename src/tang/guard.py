from __future__ import annotations

import logging
import random
import re

LOGGER = logging.getLogger("tang.guard")

_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"abaikan (semua )?(instruksi|perintah)( sebelumnya)?",
    r"lupakan (semua )?(instruksi|aturan)",
    r"you are now|kamu sekarang (adalah|jadi)",
    r"(system|developer) prompt",
    r"(tampilkan|show|reveal|print|repeat) (your |the )?(prompt|system|instruksi)",
    r"<\|im_(start|end)\|>|\[INST\]|\[/INST\]",
    r"^\s*(system|assistant)\s*:",
    r"jailbreak|DAN mode|developer mode",
]

_REGEXES = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

_NEWLINES = re.compile(r"\n{7,}")
_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{200,}")

INJECTION_LINES = [
    "nice try bang",
    "wkwkwk nggak semudah itu ferguso",
    "halah ketauan",
    "coba lagi tahun depan",
    "nice try, tapi engga",
]


class Guard:
    """Deterministic prompt-injection scan. No model call, ever."""

    def scan(self, text: str) -> bool:
        if not text:
            return False
        if _NEWLINES.search(text):
            return True
        if _BASE64_BLOB.search(text):
            return True
        return any(rx.search(text) for rx in _REGEXES)

    def canned(self) -> str:
        return random.choice(INJECTION_LINES)
