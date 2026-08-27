from __future__ import annotations

import pytest

from tang.context.summarizer import Summarizer
from tang.storage.json_store import JsonStore


def test_read_write_summary(tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    summarizer = Summarizer(store, None)
    assert summarizer.read(123) is None
    summarizer.write(123, "Alice and Bob are planning a launch")
    assert summarizer.read(123) == "Alice and Bob are planning a launch"


def test_clear_summary(tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    summarizer = Summarizer(store, None)
    summarizer.write(123, "some summary")
    summarizer.clear(123)
    assert summarizer.read(123) is None


async def test_compact_below_threshold(tmp_path, stub_client):
    store = JsonStore(str(tmp_path / "data"))
    summarizer = Summarizer(store, stub_client())
    turns = [{"content": "short", "display_name": "user"}]
    compacted = await summarizer.maybe_compact(123, turns, token_threshold=10000)
    assert compacted is False
