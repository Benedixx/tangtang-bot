from __future__ import annotations

from tang.memory.fact_extractor import FactExtractor


async def test_extract_valid_payload(stub_client):
    client = stub_client(payload={
        "facts": [
            {"text": "User prefers concise answers", "source_channel_id": "123", "guild_id": "456"},
        ]
    })
    result = await FactExtractor(client).extract(
        [{"display_name": "user", "content": "gw suka jawaban singkat ya"}],
        channel_id=123,
        guild_id=456,
    )
    assert len(result) == 1
    assert result[0]["text"] == "User prefers concise answers"


async def test_extract_empty_is_normal(stub_client):
    client = stub_client(payload={"facts": []})
    result = await FactExtractor(client).extract(
        [{"display_name": "user", "content": "wkwkwk lucu banget"}],
        channel_id=123,
    )
    assert result == []


async def test_extract_garbage_json_returns_empty(stub_client):
    client = stub_client(payload={})
    result = await FactExtractor(client).extract(
        [{"display_name": "user", "content": "hello"}],
        channel_id=123,
    )
    assert result == []


async def test_extract_call_failure_returns_empty(stub_client):
    client = stub_client(payload=RuntimeError("down"))
    result = await FactExtractor(client).extract(
        [{"display_name": "user", "content": "hello"}],
        channel_id=123,
    )
    assert result == []


async def test_extract_no_messages_short_circuits(stub_client):
    client = stub_client()
    assert await FactExtractor(client).extract([], channel_id=123) == []
    assert client.calls == []
