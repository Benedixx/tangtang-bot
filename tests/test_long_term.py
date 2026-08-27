from __future__ import annotations

import json
from datetime import timedelta

from tang.config import MemoryConfig
from tang.context.long_term import LongTermMemory
from tang.memory.dedup import extract_keywords
from tang.storage.json_store import JsonStore


def test_durable_fact_stored(json_store, mem_cfg):
    store = LongTermMemory(json_store, mem_cfg)
    stored, updated, rejected = store.remember(
        "100", "Alice", [{"text": "User prefers concise technical explanations"}]
    )
    assert (stored, updated, rejected) == (1, 0, 0)


def test_short_content_rejected(json_store, mem_cfg):
    store = LongTermMemory(json_store, mem_cfg)
    s, u, r = store.remember("100", "Alice", [{"text": "short"}])
    assert (s, u, r) == (0, 0, 1)


def test_duplicate_consolidated_into_update(json_store, mem_cfg):
    store = LongTermMemory(json_store, mem_cfg)
    store.remember("100", "Alice", [{"text": "User prefers python over javascript"}])
    s, u, r = store.remember("100", "Alice", [{"text": "User prefers python over javascript"}])
    assert (s, u) == (0, 1)
    assert len(store.read("100")) == 1


def test_distinct_facts_not_merged(json_store, mem_cfg):
    store = LongTermMemory(json_store, mem_cfg)
    store.remember("100", "Alice", [{"text": "User prefers python over javascript"}])
    store.remember("100", "Alice", [{"text": "User works at a bank in Jakarta"}])
    facts = store.read("100")
    assert len(facts) == 2


def test_missing_file_ok(json_store, mem_cfg):
    store = LongTermMemory(json_store, mem_cfg)
    assert store.read("nobody") == []
    assert store.search("nobody", "apa saja") == []


def test_corrupt_file_quarantined(json_store, mem_cfg, tmp_path):
    root = tmp_path / "data" / "facts"
    (root / "100.json").write_text("{not valid json", encoding="utf-8")
    store = LongTermMemory(json_store, mem_cfg)
    facts = store.read("100")
    assert facts == []


def test_atomic_write_leaves_no_tmp(mem_cfg, tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    ltm = LongTermMemory(store, mem_cfg)
    ltm.remember("100", "Alice", [{"text": "User runs ubuntu server"}])
    leftovers = [p for p in tmp_path.rglob("*") if p.suffix == ".tmp"]
    assert leftovers == []


def test_search_top_k_limit(json_store, mem_cfg):
    mem_cfg.fact_top_k = 2
    store = LongTermMemory(json_store, mem_cfg)
    store.remember("100", "Alice", [
        {"text": "User plays valorant every night"},
        {"text": "User plays dota on weekends"},
        {"text": "User plays minecraft with friends"},
    ])
    hits = store.search("100", "game apa yang user mainkan plays")
    assert len(hits) <= 2


def test_search_irrelevant_returns_nothing(json_store, mem_cfg):
    store = LongTermMemory(json_store, mem_cfg)
    store.remember("100", "Alice", [{"text": "User prefers concise explanations"}])
    assert store.search("100", "resep masak rendang santan kelapa") == []


def test_search_stopword_only_query(json_store, mem_cfg):
    store = LongTermMemory(json_store, mem_cfg)
    store.remember("100", "Alice", [{"text": "User prefers concise explanations"}])
    assert store.search("100", "yang itu apa sih gw lu kamu") == []


def test_extract_keywords_filters_stopwords():
    kw = extract_keywords("Yang gw mau itu bikin bot pake Python di server rumah")
    assert "yang" not in kw and "gw" not in kw and "itu" not in kw
    assert "python" in kw and "bot" in kw


def test_sweep_removes_expired(json_store, mem_cfg, tmp_path):
    mem_cfg.fact_ttl_days = 1
    store = LongTermMemory(json_store, mem_cfg)
    store.remember("100", "Alice", [{"text": "User runs old server setup"}])

    facts_path = tmp_path / "data" / "facts" / "100.json"
    data = json.loads(facts_path.read_text(encoding="utf-8"))
    from datetime import datetime, timedelta, UTC
    data["facts"][0]["expires_at"] = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    facts_path.write_text(json.dumps(data), encoding="utf-8")

    removed = store.sweep()
    assert removed == 1
    assert store.read("100") == []
