from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Dict, Tuple

LOGGER = logging.getLogger("discord-bot.response-queue")


class AsyncResponseQueue:
    """A dedicated per-channel FIFO queue with a shared concurrency limit.

    Each channel gets its own queue and worker task to preserve ordering.
    A shared semaphore limits the number of concurrent LLM jobs across all
    channels.
    """

    def __init__(
        self,
        concurrency: int = 3,
        timeout_seconds: int = 60,
    ) -> None:
        self._queues: Dict[int, Deque[Tuple[Callable[[], Awaitable[Any]], asyncio.Future]]] = {}
        self._workers: Dict[int, asyncio.Task] = {}
        self._concurrency = concurrency
        self._timeout = timeout_seconds
        self._sem = asyncio.Semaphore(concurrency)
        self._started = False

    def start(self) -> None:
        self._started = True
        LOGGER.info(
            "response-queue ready concurrency=%s", self._concurrency,
        )

    def enqueue(self, channel_id: int, coro_fn: Callable[[], Awaitable[Any]]) -> asyncio.Future:
        """Enqueue a zero-arg coroutine function for the given channel."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        queue = self._queues.setdefault(channel_id, deque())
        queue.append((coro_fn, fut))
        LOGGER.debug(
            "enqueued job channel=%s queue_size=%s", channel_id, len(queue)
        )

        if channel_id not in self._workers or self._workers[channel_id].done():
            self._workers[channel_id] = loop.create_task(self._channel_worker(channel_id))
            LOGGER.debug("started channel worker channel=%s", channel_id)

        if not self._started:
            self.start()

        return fut

    async def _channel_worker(self, channel_id: int) -> None:
        queue = self._queues[channel_id]
        while queue:
            coro_fn, fut = queue.popleft()
            try:
                async with self._sem:
                    try:
                        result = await asyncio.wait_for(coro_fn(), timeout=self._timeout)
                        if not fut.done():
                            fut.set_result(result)
                    except asyncio.TimeoutError as exc:
                        LOGGER.warning(
                            "channel=%s job timed out after %s seconds",
                            channel_id,
                            self._timeout,
                        )
                        if not fut.done():
                            fut.set_exception(exc)
                    except Exception as exc:
                        LOGGER.exception("channel=%s job raised exception", channel_id)
                        if not fut.done():
                            fut.set_exception(exc)
            finally:
                if not queue:
                    break
        self._workers.pop(channel_id, None)
        self._queues.pop(channel_id, None)
        LOGGER.debug("stopped channel worker channel=%s", channel_id)
