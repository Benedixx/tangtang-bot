from __future__ import annotations

from tang.storage.json_store import JsonStore


def test_read_missing_returns_default(tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    result = store.read_json(tmp_path / "nonexistent.json", default=[])
    assert result == []


def test_read_missing_returns_empty_dict(tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    result = store.read_json(tmp_path / "nonexistent.json")
    assert result == {}


def test_write_and_read(tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    path = tmp_path / "data" / "test.json"
    assert store.write_json(path, {"key": "value"}) is True
    assert store.read_json(path) == {"key": "value"}


def test_atomic_write_leaves_no_tmp(tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    path = tmp_path / "data" / "test.json"
    store.write_json(path, {"key": "value"})
    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert tmp_files == []


def test_corrupt_file_quarantined(tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    path = tmp_path / "data" / "test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    result = store.read_json(path, default=[])
    assert result == []
    quarantined = list(tmp_path.rglob("*.corrupt-*"))
    assert len(quarantined) == 1


def test_append_jsonl(tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    path = tmp_path / "data" / "test.jsonl"
    store.append_jsonl(path, {"line": 1})
    store.append_jsonl(path, {"line": 2})
    records = store.read_jsonl(path)
    assert len(records) == 2
    assert records[0]["line"] == 1
    assert records[1]["line"] == 2


def test_read_jsonl_missing(tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    records = store.read_jsonl(tmp_path / "nonexistent.jsonl")
    assert records == []


def test_ensure_dirs(tmp_path):
    data_dir = tmp_path / "new_data"
    store = JsonStore(str(data_dir))
    assert (data_dir / "buffers").is_dir()
    assert (data_dir / "summaries").is_dir()
    assert (data_dir / "facts").is_dir()
    assert (data_dir / "archive").is_dir()
