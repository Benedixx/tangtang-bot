from __future__ import annotations

import pytest

from tang.config import MemoryConfig
from tang.queue import Msg


@pytest.fixture
def mem_cfg(tmp_path) -> MemoryConfig:
    return MemoryConfig(dir=str(tmp_path / "memory"))


@pytest.fixture
def stub_client():
    class StubGroqClient:
        def __init__(self, payload=None):
            self.payload = payload
            self.calls: list = []

        async def complete_json(self, messages, **kwargs):
            self.calls.append(messages)
            if isinstance(self.payload, Exception):
                raise self.payload
            return self.payload

    return StubGroqClient


def make_msg(mid: int, name: str = "budi", content: str = "halo dunia", is_bot: bool = False,
             guild_id: int | None = 1, author_id: int = 100) -> Msg:
    return Msg(
        message_id=mid,
        author_id=author_id,
        author_name=name,
        content=content,
        is_bot=is_bot,
        guild_id=guild_id,
    )
