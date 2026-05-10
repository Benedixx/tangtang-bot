from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm import OpenRouterClient

LOGGER = logging.getLogger("discord-bot.memory")

_PROTECTED_FACTS: list[str] = [
    "Tito (juga dikenal sebagai Benedixx, Benedixxlee, Benedihh) adalah suami Tangtang.",
]

_MAX_GLOBAL_FACTS = 80


class TangtangMemoryStore:
    def __init__(self, file_path: str, llm: OpenRouterClient | None = None) -> None:
        self._path = Path(file_path)
        self._llm = llm
        self._data: dict[str, object] = {}
        self._ensure_loaded()

    async def remember_message(
        self,
        author_name: str,
        content: str,
        request_id: str | None = None,
    ) -> int:
        if not self._llm:
            return 0

        new_facts = await self._extract_facts_via_llm(author_name, content, request_id)
        if not new_facts:
            return 0

        facts = self._get_global_facts()
        existing_lower = {f.strip().lower() for f in facts}
        added = 0
        for fact in new_facts:
            normalized = fact.strip().lower()
            if normalized and normalized not in existing_lower:
                facts.append(fact.strip())
                existing_lower.add(normalized)
                added += 1

        if len(facts) > _MAX_GLOBAL_FACTS:
            protected = [f for f in facts if _is_protected(f)]
            others = [f for f in facts if not _is_protected(f)]
            facts = protected + others[: _MAX_GLOBAL_FACTS - len(protected)]

        self._data["global_facts"] = facts
        _update_meta_timestamp(self._data)

        if added:
            self._save()
            LOGGER.info("[request=%s] memory_facts_added count=%s", request_id, added)

        return added

    def upsert_fact(self, new_fact: str) -> str:
        new_fact = new_fact.strip()
        if not new_fact:
            return "ignored: empty fact"

        facts = self._get_global_facts()
        new_tokens = {t for t in re.findall(r"[a-zA-Z0-9À-ž]+", new_fact.lower()) if len(t) >= 3}

        best_idx: int | None = None
        best_overlap = 0.0
        for i, existing in enumerate(facts):
            if _is_protected(existing):
                continue
            existing_tokens = {t for t in re.findall(r"[a-zA-Z0-9À-ž]+", existing.lower()) if len(t) >= 3}
            if not existing_tokens:
                continue
            intersection = len(new_tokens & existing_tokens)
            overlap = intersection / len(existing_tokens | new_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        if best_idx is not None and best_overlap >= 0.5:
            old_fact = facts[best_idx]
            facts[best_idx] = new_fact
            self._data["global_facts"] = facts
            _update_meta_timestamp(self._data)
            self._save()
            LOGGER.info("memory_upsert action=updated old=%s new=%s", old_fact[:60], new_fact[:60])
            return f"updated: {old_fact}"

        facts.append(new_fact)
        if len(facts) > _MAX_GLOBAL_FACTS:
            protected = [f for f in facts if _is_protected(f)]
            others = [f for f in facts if not _is_protected(f)]
            facts = protected + others[: _MAX_GLOBAL_FACTS - len(protected)]

        self._data["global_facts"] = facts
        _update_meta_timestamp(self._data)
        self._save()
        LOGGER.info("memory_upsert action=inserted fact=%s", new_fact[:60])
        return "inserted"

    def delete_facts(self, pattern: str) -> int:
        pattern = pattern.strip()
        if not pattern:
            return 0

        pattern_tokens = {t.lower() for t in re.findall(r"[a-zA-Z0-9À-ž]+", pattern) if len(t) >= 3}
        facts = self._get_global_facts()
        remaining: list[str] = []
        deleted = 0

        for fact in facts:
            if _is_protected(fact):
                remaining.append(fact)
                continue
            fact_lower = fact.lower()
            if any(token in fact_lower for token in pattern_tokens):
                LOGGER.info("memory_delete removed fact=%s", fact[:60])
                deleted += 1
            else:
                remaining.append(fact)

        if deleted:
            self._data["global_facts"] = remaining
            _update_meta_timestamp(self._data)
            self._save()

        return deleted

    def build_context(self, latest_message: str = "", max_facts: int = 10) -> str:
        facts = self._get_global_facts()
        if not facts:
            return "Belum ada memori relevan."

        if latest_message:
            facts = _rank_by_relevance(facts, latest_message, max_facts)
        else:
            facts = facts[:max_facts]

        return "\n".join(f"- {f}" for f in facts)

    def _get_global_facts(self) -> list[str]:
        raw = self._data.get("global_facts", [])
        if not isinstance(raw, list):
            return []
        return [str(f).strip() for f in raw if f]

    async def _extract_facts_via_llm(
        self,
        author_name: str,
        content: str,
        request_id: str | None,
    ) -> list[str]:
        assert self._llm is not None

        existing = self._get_global_facts()
        existing_block = "\n".join(f"- {f}" for f in existing[:20]) or "(kosong)"

        system = (
            "Kamu adalah sistem ekstraksi memori. "
            "Dari pesan user, ekstrak fakta global yang penting dan layak disimpan permanen. "
            "Fakta layak simpan: nama panggilan seseorang, hubungan antar orang, preferensi kuat, "
            "profil diri, atau info penting yang akan berguna di percakapan mendatang. "
            "JANGAN simpan: percakapan biasa, pertanyaan, keluhan sementara, perintah ke bot. "
            "Output: JSON array of strings. Jika tidak ada fakta baru, output [].\n"
            "Contoh output: [\"Budi suka kopi hitam\"] atau []"
        )

        user_prompt = (
            f"Pesan dari {author_name}: {content}\n\n"
            f"Fakta yang sudah tersimpan:\n{existing_block}\n\n"
            "Output JSON array saja, tanpa penjelasan lain."
        )

        try:
            raw = await self._llm.complete(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=200,
                request_id=request_id,
                request_label="memory:extract",
            )
        except Exception:
            LOGGER.exception("[request=%s] memory_extract_failed", request_id)
            return []

        return _parse_facts_json(raw)

    def _ensure_loaded(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if not self._path.exists():
            self._data = _default_payload()
            self._save()
            return

        try:
            raw = self._path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Memory file root must be an object")
            self._data = parsed
            self._ensure_protected_facts()
        except Exception:
            LOGGER.exception("Failed to load memory file, recreating default payload")
            self._data = _default_payload()
            self._save()

    def _ensure_protected_facts(self) -> None:
        facts = self._get_global_facts()
        changed = False
        for protected in _PROTECTED_FACTS:
            if protected not in facts:
                facts.insert(0, protected)
                changed = True
        if changed:
            self._data["global_facts"] = facts
            self._save()

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, ensure_ascii=True, indent=2), encoding="utf-8")


def _default_payload() -> dict[str, object]:
    return {
        "meta": {
            "schema_version": 2,
            "persona": "Tangtang Supreme Chief",
            "updated_at": _now_iso(),
        },
        "global_facts": list(_PROTECTED_FACTS),
    }


def _update_meta_timestamp(data: dict[str, object]) -> None:
    meta = data.get("meta")
    if isinstance(meta, dict):
        meta["updated_at"] = _now_iso()


def _is_protected(fact: str) -> bool:
    lowered = fact.lower()
    return any(alias in lowered for alias in ("benedixx", "benedihh", "tito"))


def _rank_by_relevance(facts: list[str], message: str, max_count: int) -> list[str]:
    tokens = {t for t in re.findall(r"[a-zA-Z0-9]+", message.lower()) if len(t) >= 3}
    scored = [(sum(1 for t in tokens if t in f.lower()), f) for f in facts]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:max_count]]


def _parse_facts_json(raw: str) -> list[str]:
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, list):
            return []
        return [str(f).strip() for f in parsed if isinstance(f, str) and f.strip()]
    except (json.JSONDecodeError, ValueError):
        return []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
