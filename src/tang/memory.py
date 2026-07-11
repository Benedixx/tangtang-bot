from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm import OpenRouterClient

LOGGER = logging.getLogger("tang.memory")

_MAX_FACTS = 80
_PROTECTED = [
    "Tito (Benedixx) adalah suami Tang.",
]


class MemoryStore:
    """Long-term fact storage with relevance-based retrieval."""

    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        self._facts: list[str] = []
        self._load()

    # -- Public API --

    def build_context(self, message: str = "", max_facts: int = 8) -> str:
        """Return relevant facts formatted for the LLM prompt."""
        if not self._facts:
            return ""
        relevant = self._rank(message, max_facts) if message else self._facts[:max_facts]
        return "\n".join(f"- {f}" for f in relevant)

    def save_fact(self, fact: str) -> str:
        fact = fact.strip()
        if not fact:
            return "ignored: empty fact"

        # Deduplicate by checking existing
        norm = fact.lower()
        for existing in self._facts:
            if existing.lower() == norm:
                return "already saved"

        self._facts.append(fact)
        self._trim()
        self._save()
        return f"saved: {fact[:80]}"

    def delete_fact(self, keyword: str) -> str:
        keyword = keyword.strip().lower()
        if not keyword:
            return "ignored: empty keyword"

        remaining: list[str] = []
        removed = 0
        for fact in self._facts:
            if keyword in fact.lower():
                removed += 1
            else:
                remaining.append(fact)

        if removed:
            self._facts = remaining
            self._save()
        return f"deleted {removed} fact(s)"

    # -- Internal --

    def _rank(self, message: str, count: int) -> list[str]:
        tokens = {t for t in re.findall(r"[a-zA-Z0-9]+", message.lower()) if len(t) >= 3}
        scored = [(sum(1 for t in tokens if t in f.lower()), f) for f in self._facts]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:count]]

    def _trim(self) -> None:
        if len(self._facts) <= _MAX_FACTS:
            return
        protected = [f for f in self._facts if self._is_protected(f)]
        others = [f for f in self._facts if not self._is_protected(f)]
        self._facts = protected + others[:_MAX_FACTS - len(protected)]

    def _load(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._facts = list(_PROTECTED)
            self._save()
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self._facts = [str(f).strip() for f in data.get("facts", []) if f]
            self._ensure_protected()
        except Exception:
            LOGGER.exception("Failed to load memory, starting fresh")
            self._facts = list(_PROTECTED)
            self._save()

    def _ensure_protected(self) -> None:
        changed = False
        for p in _PROTECTED:
            if p not in self._facts:
                self._facts.insert(0, p)
                changed = True
        if changed:
            self._save()

    def _save(self) -> None:
        payload = json.dumps({"facts": self._facts, "updated": self._now()}, indent=2)
        try:
            self._path.write_text(payload, encoding="utf-8")
        except Exception:
            LOGGER.exception("Failed to save memory")

    @staticmethod
    def _is_protected(fact: str) -> bool:
        return any(alias in fact.lower() for alias in ("benedixx", "benedihh", "tito"))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
