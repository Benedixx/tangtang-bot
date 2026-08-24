from __future__ import annotations

from tang.memory.extractor import Extractor

from conftest import make_msg


async def test_extract_valid_payload(mem_cfg, stub_client):
    client = stub_client(payload={
        "memories": [
            {"type": "semantic", "content": "User prefers concise answers",
             "importance": 0.8, "confidence": 0.9},
        ]
    })
    result = await Extractor(client).extract([make_msg(1, content="gw suka jawaban singkat ya")])
    assert len(result) == 1
    assert result[0].type == "semantic"
    assert result[0].importance == 0.8


async def test_extract_empty_is_normal(mem_cfg, stub_client):
    client = stub_client(payload={"memories": []})
    result = await Extractor(client).extract([make_msg(1, content="wkwkwk lucu banget")])
    assert result == []


async def test_extract_garbage_json_returns_empty(mem_cfg, stub_client):
    client = stub_client(payload={})
    result = await Extractor(client).extract([make_msg(1)])
    assert result == []


async def test_extract_call_failure_returns_empty(mem_cfg, stub_client):
    client = stub_client(payload=RuntimeError("down"))
    result = await Extractor(client).extract([make_msg(1)])
    assert result == []


async def test_extract_no_messages_short_circuits(mem_cfg, stub_client):
    client = stub_client()
    assert await Extractor(client).extract([]) == []
    assert client.calls == []
