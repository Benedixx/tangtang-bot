from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Tuple

LOGGER = logging.getLogger("discord-bot.response-queue")


class AsyncResponseQueue:
    """A simple bounded worker pool that runs coroutine jobs.

    Jobs are enqueued as zero-arg coroutine functions (callables that return an
    awaitable). The queue enforces a global concurrency semaphore for LLM calls
    and starts a fixed number of workers.
    """

    def __init__(
        self,
        worker_count: int = 3,
        concurrency: int = 3,
        timeout_seconds: int = 60,
    ) -> None:
        self._queue: asyncio.Queue[Tuple[Callable[[], Awaitable[Any]], asyncio.Future]] = (
            asyncio.Queue()
        )
        self._worker_count = worker_count
        self._concurrency = concurrency
        self._timeout = timeout_seconds
        self._workers: list[asyncio.Task] = []
        self._sem = asyncio.Semaphore(concurrency)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        loop = asyncio.get_event_loop()
        for _ in range(self._worker_count):
            task = loop.create_task(self._worker())
            self._workers.append(task)
        self._started = True
        LOGGER.info("response-queue started workers=%s concurrency=%s", self._worker_count, self._concurrency)

    def enqueue(self, coro_fn: Callable[[], Awaitable[Any]]) -> asyncio.Future:
        """Enqueue a zero-arg coroutine function. Returns a Future for the result.

        The returned Future will be resolved with the coroutine's return value
        or set with an exception if the job fails or times out.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._queue.put_nowait((coro_fn, fut))
        LOGGER.debug("enqueued job queue_size=%s", self._queue.qsize())
        # Ensure workers are started even if caller forgot to call start()
        if not self._started:
            self.start()
        return fut

    async def _worker(self) -> None:
        while True:
            coro_fn, fut = await self._queue.get()
            try:
                async with self._sem:
                    try:
                        result = await asyncio.wait_for(coro_fn(), timeout=self._timeout)
                        if not fut.done():
                            fut.set_result(result)
                    except asyncio.TimeoutError as exc:
                        LOGGER.warning("job timed out after %s seconds", self._timeout)
                        if not fut.done():
                            fut.set_exception(exc)
                    except Exception as exc:  # capture job exceptions
                        LOGGER.exception("job raised exception")
                        if not fut.done():
                            fut.set_exception(exc)
            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass
