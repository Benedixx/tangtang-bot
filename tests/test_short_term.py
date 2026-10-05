from __future__ import annotations

from tang.context.short_term import ShortTermBuffer
from tang.storage.json_store import JsonStore


def test_append_and_read(tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    buf = ShortTermBuffer(store)
    buf.append(123, 456, "100", "Alice", "user", "hello")
    buf.append(123, 456, "bot", "Bot", "assistant", "hi there")
    turns = buf.read(123)
    assert len(turns) == 2
    assert turns[0]["content"] == "hello"
    assert turns[1]["role"] == "assistant"


def test_append_caps_turns(tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    buf = ShortTermBuffer(store, max_turns=3)
    for i in range(10):
        buf.append(123, 456, str(i), f"user{i}", "user", f"msg{i}")
    turns = buf.read(123)
    assert [t["content"] for t in turns] == ["msg7", "msg8", "msg9"]


def test_multiple_channels(tmp_path):
    store = JsonStore(str(tmp_path / "data"))
    buf = ShortTermBuffer(store)
    buf.append(111, 456, "100", "Alice", "user", "hello ch1")
    buf.append(222, 456, "200", "Bob", "user", "hello ch2")
    assert len(buf.read(111)) == 1
    assert len(buf.read(222)) == 1
    assert buf.read(111)[0]["content"] == "hello ch1"
    assert buf.read(222)[0]["content"] == "hello ch2"
