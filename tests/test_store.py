from __future__ import annotations

import json
from datetime import timedelta

from tang.memory.models import Memory, MemoryCandidate, MemoryScope, MemoryType, utcnow
from tang.memory.store import MemoryStore, extract_keywords, scope_key

from conftest import make_msg


def cand(content: str, importance: float = 0.8, confidence: float = 0.9,
         type_: MemoryType = MemoryType.semantic) -> MemoryCandidate:
    return MemoryCandidate(type=type_, content=content, importance=importance, confidence=confidence)


# ----------------------------------------------------------------------
# Writing / validation
# ----------------------------------------------------------------------

def test_durable_fact_stored(mem_cfg):
    store = MemoryStore(mem_cfg)
    stored, updated, rejected = store.remember("g1-u1", [cand("User prefers concise technical explanations")])
    assert (stored, updated, rejected) == (1, 0, 0)


def test_low_confidence_rejected(mem_cfg):
    store = MemoryStore(mem_cfg)
    s, u, r = store.remember("g1-u1", [cand("User likes durian", confidence=0.3)])
    assert (s, u, r) == (0, 0, 1)


def test_low_importance_rejected(mem_cfg):
    store = MemoryStore(mem_cfg)
    s, u, r = store.remember("g1-u1", [cand("User likes durian", importance=0.2)])
    assert (s, u, r) == (0, 0, 1)


def test_typo_noise_rejected_by_length(mem_cfg):
    store = MemoryStore(mem_cfg)
    s, u, r = store.remember("g1-u1", [cand("wkwkwk")])
    assert (s, u, r) == (0, 0, 1)


def test_duplicate_consolidated_into_update(mem_cfg):
    store = MemoryStore(mem_cfg)
    store.remember("g1-u1", [cand("User prefers python over javascript")])
    s, u, r = store.remember("g1-u1", [cand("User prefers python over javascript")])
    assert (s, u) == (0, 1)
    assert len(store.load("g1-u1").memories) == 1


def test_near_duplicate_skipped(mem_cfg):
    store = MemoryStore(mem_cfg)
    store.remember("g1-u1", [cand("User is building a discord ai agent with pydanticai library")])
    s, u, _ = store.remember(
        "g1-u1",
        [cand("User is building a discord ai agent using the pydanticai framework now")],
    )
    assert (s, u) == (0, 0)
    assert len(store.load("g1-u1").memories) == 1


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def test_memory_survives_restart(mem_cfg, tmp_path):
    store = MemoryStore(mem_cfg)
    store.remember("g1-u1", [cand("User is deploying on a home server")])

    restarted = MemoryStore(MemoryConfig_like(mem_cfg))
    hits = restarted.search("g1-u1", "deploying home server setup")
    assert any("home server" in h.memory.content for h in hits)


def MemoryConfig_like(cfg):
    from tang.config import MemoryConfig

    return MemoryConfig(dir=cfg.dir)


def test_missing_file_ok(mem_cfg):
    store = MemoryStore(mem_cfg)
    assert store.load("nobody").memories == []
    assert store.search("nobody", "apa saja") == []


def test_corrupt_file_quarantined(mem_cfg, tmp_path):
    root = tmp_path / "memory" / "scopes"
    root.mkdir(parents=True)
    (root / "g1-u1.json").write_text("{not valid json", encoding="utf-8")
    store = MemoryStore(mem_cfg)
    scope = store.load("g1-u1")
    assert scope.memories == []
    files = list(root.iterdir())
    assert len(files) == 1 and "corrupt" in files[0].name


def test_atomic_write_leaves_no_tmp(mem_cfg):
    store = MemoryStore(mem_cfg)
    store.remember("g1-u1", [cand("User runs ubuntu server")])
    leftovers = [p for p in (mem_cfg.dir and list(__import__("pathlib").Path(mem_cfg.dir).rglob("*"))) if p.suffix == ".tmp"]
    assert leftovers == []


# ----------------------------------------------------------------------
# Scoping
# ----------------------------------------------------------------------

def test_scoping_isolation(mem_cfg):
    store = MemoryStore(mem_cfg)
    store.remember(scope_key(1, 100), [cand("User A works at bank mega")])
    store.remember(scope_key(2, 100), [cand("User A loves street racing")])
    store.remember(scope_key(None, 100), [cand("User A chats privately about health")])

    assert store.search(scope_key(1, 100), "bank mega kerja")[0].memory.content.startswith("User A works")
    assert store.search(scope_key(2, 100), "street racing balap") != []
    assert store.search(scope_key(2, 100), "bank mega kerja") == []  # guild 1 memory invisible to guild 2
    assert store.search(scope_key(None, 100), "health kesehatan") != []


def test_scope_key_format():
    assert scope_key(123, 456) == "123-456"
    assert scope_key(None, 456) == "dm-456"


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------

def test_search_top_k_limit(mem_cfg):
    mem_cfg.top_k = 2
    store = MemoryStore(mem_cfg)
    store.remember("g1-u1", [
        cand("User plays valorant every night"),
        cand("User plays dota on weekends"),
        cand("User plays minecraft with friends"),
    ])
    hits = store.search("g1-u1", "game apa yang user mainkan plays")
    assert len(hits) <= 2
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_search_irrelevant_returns_nothing(mem_cfg):
    store = MemoryStore(mem_cfg)
    store.remember("g1-u1", [cand("User prefers concise explanations")])
    assert store.search("g1-u1", "resep masak rendang santan kelapa") == []


def test_search_stopword_only_query(mem_cfg):
    store = MemoryStore(mem_cfg)
    store.remember("g1-u1", [cand("User prefers concise explanations")])
    assert store.search("g1-u1", "yang itu apa sih gw lu kamu") == []


def test_mark_used_updates_stats(mem_cfg):
    store = MemoryStore(mem_cfg)
    store.remember("g1-u1", [cand("User prefers concise explanations")])
    before = store.load("g1-u1").memories[0]
    assert before.use_count == 0
    store.search("g1-u1", "concise explanation preferences")
    after = store.load("g1-u1").memories[0]
    assert after.use_count == 1
    assert after.last_used_at is not None


# ----------------------------------------------------------------------
# Forgetting
# ----------------------------------------------------------------------

def test_sweep_removes_never_used_old(mem_cfg, tmp_path):
    store = MemoryStore(mem_cfg)
    old = utcnow() - timedelta(days=40)
    scope = MemoryScope(scope="g1-u1", memories=[
        Memory(id="a", type=MemoryType.semantic, content="stale never used fact here",
               keywords=["stale"], importance=0.9, confidence=0.9,
               created_at=old, updated_at=old),
    ])
    root = __import__("pathlib").Path(mem_cfg.dir) / "scopes"
    root.mkdir(parents=True, exist_ok=True)
    (root / "g1-u1.json").write_text(scope.model_dump_json(), encoding="utf-8")

    removed = store.sweep()
    assert removed == 1
    assert store.load("g1-u1").memories == []


def test_sweep_keeps_recent_and_used(mem_cfg, tmp_path):
    store = MemoryStore(mem_cfg)
    now = utcnow()
    scope = MemoryScope(scope="g1-u1", memories=[
        Memory(id="b", type=MemoryType.semantic, content="recent important preference",
               keywords=["recent"], importance=0.9, confidence=0.9,
               created_at=now - timedelta(days=40), updated_at=now - timedelta(days=1),
               last_used_at=now - timedelta(days=1), use_count=3),
    ])
    root = __import__("pathlib").Path(mem_cfg.dir) / "scopes"
    root.mkdir(parents=True, exist_ok=True)
    (root / "g1-u1.json").write_text(json.dumps(scope.model_dump(mode="json")), encoding="utf-8")

    assert store.sweep() == 0
    assert len(store.load("g1-u1").memories) == 1


# ----------------------------------------------------------------------
# Keywords helper
# ----------------------------------------------------------------------

def test_extract_keywords_filters_stopwords():
    kw = extract_keywords("Yang gw mau itu bikin bot pake Python di server rumah")
    assert "yang" not in kw and "gw" not in kw and "itu" not in kw
    assert "python" in kw and "bot" in kw


def test_store_roundtrip_preserves_fields(mem_cfg):
    store = MemoryStore(mem_cfg)
    store.remember("g1-u1", [cand("User streams on twitch weekly", type_=MemoryType.procedural)])
    mem = store.load("g1-u1").memories[0]
    assert mem.type == MemoryType.procedural
    assert mem.importance == 0.8 and mem.confidence == 0.9
