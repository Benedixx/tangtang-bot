from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path

from rapidfuzz.fuzz import partial_ratio, token_sort_ratio

from ..config import MemoryConfig
from .models import Memory, MemoryCandidate, MemoryScope, RetrievedMemory, utcnow

LOGGER = logging.getLogger("tang.memory.store")

_MAX_MEMORIES_PER_SCOPE = 100
_DUP_UPDATE_RATIO = 85.0
_DUP_SKIP_RATIO = 70.0
_RECENCY_SPAN_DAYS = 56.0

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


def scope_key(guild_id: int | None, user_id: int) -> str:
    """User-in-guild scope; DMs collapse to the user alone."""
    return f"dm-{user_id}" if guild_id is None else f"{guild_id}-{user_id}"


def _jaccard(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def _recency(dt: datetime, now: datetime) -> float:
    age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return max(0.0, 1.0 - age_days / _RECENCY_SPAN_DAYS)


class MemoryStore:
    """File-based long-term memory, one JSON file per scope.

    All methods are synchronous on purpose: they touch tiny files with no
    awaits inside, so the single asyncio loop serializes access naturally —
    no locking needed.
    """

    def __init__(self, cfg: MemoryConfig) -> None:
        self._cfg = cfg
        self._root = Path(cfg.dir) / "scopes"
        self._cache: dict[str, MemoryScope] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        return self._root / f"{key}.json"

    def load(self, key: str) -> MemoryScope:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        path = self._path(key)
        scope = MemoryScope(scope=key)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                loaded = MemoryScope.model_validate(data)
                if loaded.version == 1:
                    scope = loaded
            except Exception:
                quarantine = path.with_suffix(f".corrupt-{int(utcnow().timestamp())}")
                LOGGER.exception("memory_scope_corrupt scope=%s moved=%s", key, quarantine.name)
                try:
                    path.rename(quarantine)
                except OSError:
                    pass
        self._cache[key] = scope
        return scope

    def _save(self, key: str, scope: MemoryScope) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(scope.model_dump_json(indent=1), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            LOGGER.exception("memory_save_failed scope=%s", key)
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Writing (extraction pipeline: validate → dedup → merge → persist)
    # ------------------------------------------------------------------

    def remember(self, key: str, candidates: list[MemoryCandidate]) -> tuple[int, int, int]:
        """Returns (stored, updated, rejected)."""
        stored = updated = rejected = 0
        if not candidates:
            return 0, 0, 0
        scope = self.load(key)
        now = utcnow()
        for cand in candidates:
            content = " ".join(cand.content.split())
            if not 8 <= len(content) <= 200:
                rejected += 1
                continue
            if cand.confidence < self._cfg.min_confidence or cand.importance < self._cfg.min_importance:
                rejected += 1
                continue

            dup, ratio = self._find_duplicate(scope, content)
            if dup is not None:
                dup.importance = min(1.0, max(dup.importance, cand.importance))
                dup.confidence = min(1.0, max(dup.confidence, cand.confidence))
                dup.updated_at = now
                updated += 1
                continue
            if ratio >= _DUP_SKIP_RATIO:
                rejected += 1
                continue

            scope.memories.append(Memory(
                id=uuid.uuid4().hex,
                type=cand.type,
                content=content,
                keywords=extract_keywords(content),
                importance=cand.importance,
                confidence=cand.confidence,
                created_at=now,
                updated_at=now,
            ))
            stored += 1

        if stored or updated:
            self._evict_overflow(scope)
            self._save(key, scope)
        LOGGER.info(
            "memory_remember scope=%s stored=%s updated=%s rejected=%s total=%s",
            key, stored, updated, rejected, len(scope.memories),
        )
        return stored, updated, rejected

    @staticmethod
    def _find_duplicate(scope: MemoryScope, content: str) -> tuple[Memory | None, float]:
        best: tuple[Memory | None, float] = (None, 0.0)
        for mem in scope.memories:
            ratio = token_sort_ratio(mem.content.lower(), content.lower())
            if ratio > best[1]:
                best = (mem, ratio)
        mem, ratio = best
        if ratio >= _DUP_SKIP_RATIO:
            return (mem, ratio) if ratio >= _DUP_UPDATE_RATIO else (None, ratio)
        return None, ratio

    def _evict_overflow(self, scope: MemoryScope) -> None:
        if len(scope.memories) <= _MAX_MEMORIES_PER_SCOPE:
            return
        scope.memories.sort(
            key=lambda m: (m.importance * m.confidence, m.last_used_at or m.updated_at),
            reverse=True,
        )
        dropped = scope.memories[_MAX_MEMORIES_PER_SCOPE:]
        scope.memories = scope.memories[:_MAX_MEMORIES_PER_SCOPE]
        LOGGER.info("memory_evicted count=%s", len(dropped))

    # ------------------------------------------------------------------
    # Retrieval (deterministic keyword + fuzzy scoring)
    # ------------------------------------------------------------------

    def search(self, key: str, query: str) -> list[RetrievedMemory]:
        if not self._cfg.enabled:
            return []
        terms = extract_keywords(query)
        if not terms:
            return []
        scope = self.load(key)
        now = utcnow()
        qtext = " ".join(terms)
        results: list[RetrievedMemory] = []
        for mem in scope.memories:
            overlap_terms = sorted(set(terms) & set(mem.keywords))
            overlap = _jaccard(terms, mem.keywords)
            fuzzy = partial_ratio(qtext, mem.content.lower()) / 100.0
            if not overlap_terms and fuzzy < 0.7:
                continue
            score = (
                1.0 * overlap
                + 0.5 * fuzzy
                + 0.3 * mem.importance
                + 0.2 * _recency(mem.updated_at, now)
            )
            if score < self._cfg.min_score:
                continue
            reason = f"terms={overlap_terms} fuzzy={fuzzy:.2f}"
            results.append(RetrievedMemory(memory=mem, score=round(score, 4), reason=reason))

        results.sort(key=lambda r: r.score, reverse=True)
        results = results[: self._cfg.top_k]
        if results:
            self.mark_used(key, [r.memory.id for r in results])
        return results

    def mark_used(self, key: str, ids: list[str]) -> None:
        scope = self.load(key)
        now = utcnow()
        changed = False
        idset = set(ids)
        for mem in scope.memories:
            if mem.id in idset:
                mem.last_used_at = now
                mem.use_count += 1
                changed = True
        if changed:
            self._save(key, scope)

    # ------------------------------------------------------------------
    # Forgetting
    # ------------------------------------------------------------------

    def sweep(self, now: datetime | None = None) -> int:
        now = now or utcnow()
        removed_total = 0
        for path in list(self._root.glob("*.json")) if self._root.exists() else []:
            key = path.stem
            scope = self.load(key)
            keep: list[Memory] = []
            for mem in scope.memories:
                if self._expired(mem, now):
                    removed_total += 1
                    continue
                keep.append(mem)
            if len(keep) != len(scope.memories):
                scope.memories = keep
                self._save(key, scope)
        if removed_total:
            LOGGER.info("memory_sweep_removed count=%s", removed_total)
        return removed_total

    def _expired(self, mem: Memory, now: datetime) -> bool:
        if mem.expires_at is not None and mem.expires_at <= now:
            return True
        unused_days = (
            (now - (mem.last_used_at or mem.created_at)).days
            if mem.last_used_at or mem.created_at
            else 0
        )
        never_used = mem.use_count == 0 and (now - mem.created_at).days > 30
        low_value_unused = (
            mem.importance < 0.4 and mem.last_used_at is not None and unused_days > self._cfg.forget_unused_days
        )
        return never_used or low_value_unused
