from __future__ import annotations

import asyncio

from tang.memory.session import SessionManager


async def test_touch_fires_callback_after_delay():
    mgr = SessionManager()
    fired: list[int] = []

    async def on_idle(channel_id: int) -> None:
        fired.append(channel_id)

    loop = asyncio.get_running_loop()
    mgr.touch(1, 0.02, on_idle)
    assert not fired
    await asyncio.sleep(0.1)
    assert fired == [1]


async def test_touch_resets_timer():
    mgr = SessionManager()
    fired: list[int] = []

    async def on_idle(channel_id: int) -> None:
        fired.append(channel_id)

    mgr.touch(1, 0.5, on_idle)
    await asyncio.sleep(0.1)
    mgr.touch(1, 0.5, on_idle)  # reset — first timer must never fire
    await asyncio.sleep(0.2)
    assert fired == []
    await asyncio.sleep(0.5)
    assert fired == [1]


async def test_cancel_prevents_fire():
    mgr = SessionManager()
    fired: list[int] = []

    async def on_idle(channel_id: int) -> None:
        fired.append(channel_id)

    mgr.touch(1, 0.02, on_idle)
    mgr.cancel(1)
    await asyncio.sleep(0.06)
    assert fired == []


def test_pending_extraction_tracks_replies():
    mgr = SessionManager()
    session = mgr.get(7)
    assert mgr.pending_extraction() == []
    session.replies_since_extract = 2
    assert mgr.pending_extraction() == [7]
