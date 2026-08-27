from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import partial_ratio

from ..config import MemoryConfig
from ..storage.json_store import JsonStore
from ..memory.dedup import extract_keywords, find_duplicate, merge_fact

LOGGER = logging.getLogger("tang.context.long_term")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _recency_score(updated_at: str, now: datetime) -> float:
    try:
        dt = datetime.fromisoformat(updated_at)
        age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
        return max(0.0, 1.0 - age_days / 56.0)
    except (ValueError, TypeError):
        return 0.5


class LongTermMemory:
    """Per-user fact store, persisted as JSON.

    File: data/facts/{user_id}.json
    Schema:
    {
        "user_id": "111",
        "display_name": "Alice",
        "facts": [
            {
                "fact_id": "f_001",
                "text": "Alice prefers concise answers",
                "keywords": ["alice", "prefers", "concise"],
                "source_channel_id": "123",
                "guild_id": "456",
                "created_at": "ISO",
                "updated_at": "ISO",
                "expires_at": "ISO"
            }
        ]
    }
    """

    def __init__(self, store: JsonStore, config: MemoryConfig) -> None:
        self._store = store
        self._config = config
        self._dir = Path(store._root) / "facts"

    def _path(self, user_id: str) -> Path:
        return self._dir / f"{user_id}.json"

    def read(self, user_id: str) -> list[dict[str, Any]]:
        data = self._store.read_json(self._path(user_id), default={"facts": []})
        return data.get("facts", [])

    def write(self, user_id: str, display_name: str, facts: list[dict[str, Any]]) -> None:
        self._store.write_json(self._path(user_id), {
            "user_id": user_id,
            "display_name": display_name,
            "facts": facts,
        })

    def remember(
        self,
        user_id: str,
        display_name: str,
        new_facts: list[dict[str, Any]],
        guild_id: int | None = None,
    ) -> tuple[int, int, int]:
        """Store facts with dedup. Returns (stored, updated, rejected)."""
        stored = updated = rejected = 0
        if not new_facts:
            return 0, 0, 0

        existing = self.read(user_id)
        now = _now_iso()
        ttl_days = self._config.fact_ttl_days

        for fact in new_facts:
            text = " ".join(fact.get("text", "").split())
            if not 8 <= len(text) <= 200:
                rejected += 1
                continue

            dup, ratio = find_duplicate(
                existing, text, self._config.fact_similarity_threshold
            )

            if dup is not None:
                merge_fact(dup, text)
                dup["updated_at"] = now
                updated += 1
                continue

            if ratio >= 0.7:
                rejected += 1
                continue

            keywords = extract_keywords(text)
            expires_at = ""
            if ttl_days > 0:
                from datetime import timedelta
                expires_at = (datetime.now(UTC) + timedelta(days=ttl_days)).isoformat()

            existing.append({
                "fact_id": f"f_{now.replace(':', '').replace('-', '').replace('.', '')[:16]}",
                "text": text,
                "keywords": keywords,
                "source_channel_id": fact.get("source_channel_id", ""),
                "guild_id": str(guild_id) if guild_id else fact.get("guild_id"),
                "created_at": now,
                "updated_at": now,
                "expires_at": expires_at,
            })
            stored += 1

        if stored or updated:
            self._evict_overflow(existing)
            self.write(user_id, display_name, existing)

        LOGGER.info(
            "long_term_remember user=%s stored=%s updated=%s rejected=%s total=%s",
            user_id, stored, updated, rejected, len(existing),
        )
        return stored, updated, rejected

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Keyword + fuzzy search for relevant facts."""
        k = top_k or self._config.fact_top_k
        facts = self.read(user_id)
        if not facts:
            return []

        query_lower = query.lower()
        query_keywords = extract_keywords(query)
        now = datetime.now(UTC)
        results: list[tuple[dict[str, Any], float]] = []

        for fact in facts:
            fact_text = fact.get("text", "")
            fact_keywords = fact.get("keywords", extract_keywords(fact_text))

            keyword_overlap = 0.0
            if query_keywords and fact_keywords:
                keyword_overlap = len(set(query_keywords) & set(fact_keywords)) / len(set(query_keywords) | set(fact_keywords))

            fuzzy = partial_ratio(query_lower, fact_text.lower()) / 100.0

            if keyword_overlap < 0.1 and fuzzy < 0.5:
                continue

            recency = _recency_score(fact.get("updated_at", ""), now)
            score = (
                1.0 * keyword_overlap
                + 0.5 * fuzzy
                + 0.2 * recency
            )

            if score < self._config.fact_min_score:
                continue

            results.append((fact, round(score, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return [{"fact": f, "score": s} for f, s in results[:k]]

    def sweep(self) -> int:
        """Remove expired facts. Returns count removed."""
        removed = 0
        now = datetime.now(UTC)

        for path in list(self._dir.glob("*.json")) if self._dir.exists() else []:
            user_id = path.stem
            data = self._store.read_json(path, default={"facts": []})
            facts = data.get("facts", [])
            keep = []
            for fact in facts:
                expires = fact.get("expires_at", "")
                if expires:
                    try:
                        exp_dt = datetime.fromisoformat(expires)
                        if exp_dt <= now:
                            removed += 1
                            continue
                    except (ValueError, TypeError):
                        pass
                keep.append(fact)

            if len(keep) != len(facts):
                data["facts"] = keep
                self._store.write_json(path, data)

        if removed:
            LOGGER.info("long_term_sweep_removed count=%s", removed)
        return removed

    def _evict_overflow(self, facts: list[dict[str, Any]], max_facts: int = 100) -> None:
        if len(facts) <= max_facts:
            return
        now = datetime.now(UTC)

        def _sort_key(f: dict) -> tuple[float, str]:
            updated = f.get("updated_at", "")
            return (_recency_score(updated, now), updated)

        facts.sort(key=_sort_key, reverse=True)
        dropped = facts[max_facts:]
        facts[:] = facts[:max_facts]
        LOGGER.info("long_term_evicted count=%s", len(dropped))
