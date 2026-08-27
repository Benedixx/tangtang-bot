from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import partial_ratio, token_sort_ratio

from ..storage.json_store import JsonStore

LOGGER = logging.getLogger("tang.memory.dedup")

_STOPWORDS = frozenset(
    """
    yang itu apa sih gue gw lu kamu aku dia nya kita mereka dan atau tapi kalau kalo
    untuk dengan dari ke di pada nggak gak ga tidak bukan sudah belum bisa akan ada
    ini the a an of to and or is are was were be been i you he she it we they my
    your his her its our their this that these those in on at for with about as by
    from do does did have has had not no yes so if but just very too also can will
    """.split()
)

_WORD = re.compile(r"\w+", re.UNICODE)


def extract_keywords(text: str) -> list[str]:
    out: list[str] = []
    for raw in _WORD.findall(text.lower()):
        if len(raw) < 3 or raw in _STOPWORDS:
            continue
        if raw not in out:
            out.append(raw)
    return out[:10]


def _jaccard(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def find_duplicate(
    facts: list[dict[str, Any]], content: str, threshold: float = 0.85
) -> tuple[dict[str, Any] | None, float]:
    """Find the most similar existing fact. Returns (fact, similarity_ratio)."""
    best: tuple[dict[str, Any] | None, float] = (None, 0.0)
    content_lower = content.lower()
    content_keywords = extract_keywords(content)

    for fact in facts:
        fact_text = fact.get("text", "")
        ratio = token_sort_ratio(fact_text.lower(), content_lower)

        if ratio >= 70:
            fact_keywords = fact.get("keywords", extract_keywords(fact_text))
            keyword_sim = _jaccard(content_keywords, fact_keywords)
            combined = 0.7 * (ratio / 100.0) + 0.3 * keyword_sim
            if combined > best[1]:
                best = (fact, combined)

    if best[1] >= threshold:
        return best
    return None, best[1]


def merge_fact(existing: dict[str, Any], new_content: str) -> dict[str, Any]:
    """Merge new content into existing fact (overwrite if more specific)."""
    existing["text"] = new_content
    existing["keywords"] = extract_keywords(new_content)
    existing["updated_at"] = datetime.now(UTC).isoformat()
    return existing
