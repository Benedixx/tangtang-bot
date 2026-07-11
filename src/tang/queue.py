from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

LOGGER = logging.getLogger("tang.queue")


class GeneratorQueue:
    """Single-worker queue serializing all generation requests across channels."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Coroutine[Any, Any, Any]] = asyncio.Queue()
        self._worker: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())
            LOGGER.info("queue_started")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
            LOGGER.info("queue_stopped")

    async def enqueue(self, coro: Coroutine[Any, Any, Any]) -> None:
        size = self._queue.qsize()
        await self._queue.put(coro)
        LOGGER.debug("queue_enqueued size=%s", size + 1)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        while True:
            coro = await self._queue.get()
            try:
                LOGGER.debug("queue_dequeue size=%s", self._queue.qsize())
                await coro
            except asyncio.CancelledError:
                self._queue.task_done()
                raise
            except Exception:
                LOGGER.exception("queue_worker_error")
            finally:
                self._queue.task_done()
