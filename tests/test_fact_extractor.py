from __future__ import annotations

from tang.memory.fact_extractor import FactExtractor


_HUMAN = {"user_id": "100", "display_name": "user", "role": "user", "content": "hello"}


async def test_extract_attributes_fact_to_named_person(stub_client):
    client = stub_client(payload={
        "facts": [
            {"about": "dofii(agung)", "text": "Dofii mains Kagura"},
            {"about": "tangtang", "text": "tangtang is a bot"},
            {"about": "nobody", "text": "Someone likes pizza"},
        ]
    })
    turns = [
        {"user_id": "100", "display_name": "Ben", "role": "user", "content": "dof main kagura mulu"},
        {"user_id": "200", "display_name": "Dofii(Agung)", "role": "user", "content": "kagura is life"},
        {"user_id": "1", "display_name": "tangtang", "role": "assistant", "content": "wkwk"},
    ]
    result = await FactExtractor(client).extract(turns, channel_id=123, guild_id=456)
    assert [(f["user_id"], f["display_name"], f["text"]) for f in result] == [
        ("200", "Dofii(Agung)", "Dofii mains Kagura"),
    ]


async def test_extract_empty_is_normal(stub_client):
    client = stub_client(payload={"facts": []})
    result = await FactExtractor(client).extract(
        [{**_HUMAN, "content": "wkwkwk lucu banget"}],
        channel_id=123,
    )
    assert result == []


async def test_extract_garbage_json_returns_empty(stub_client):
    client = stub_client(payload={})
    result = await FactExtractor(client).extract(
        [_HUMAN],
        channel_id=123,
    )
    assert result == []


async def test_extract_call_failure_returns_empty(stub_client):
    client = stub_client(payload=RuntimeError("down"))
    result = await FactExtractor(client).extract(
        [_HUMAN],
        channel_id=123,
    )
    assert result == []


async def test_extract_no_messages_short_circuits(stub_client):
    client = stub_client()
    assert await FactExtractor(client).extract([], channel_id=123) == []
    assert client.calls == []
