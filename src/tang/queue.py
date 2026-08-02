from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

LOGGER = logging.getLogger("tang.queue")

BUFFER_SIZE = 20
STALE_AFTER = 4
MIN_TYPING_S = 1.2
TYPING_RATE = 25.0


@dataclass(slots=True)
class Msg:
    message_id: int
    author_id: int
    author_name: str
    content: str
    is_bot: bool = False


@dataclass(slots=True)
class Job:
    channel_id: int
    snapshot: list[Msg]
    counter_at_enqueue: int
    forced: bool
    trigger_message_id: int


@dataclass(slots=True)
class Reply:
    text: str
    anchored: bool = False


@dataclass(slots=True)
class ChannelState:
    buffer: deque[Msg] = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    counter: int = 0
    last_reply_ts: float = 0.0
    interjections: list[float] = field(default_factory=list)
    threshold: float = 0.65
    recent_gifs: deque[str] = field(default_factory=lambda: deque(maxlen=5))
    watch_until: float = 0.0
    watch_engaged: bool = False
    debounce: asyncio.TimerHandle | None = None
    queue: asyncio.Queue[Job] = field(default_factory=lambda: asyncio.Queue(maxsize=1))
    worker: asyncio.Task | None = None


def is_stale(state: ChannelState, job: Job) -> bool:
    return state.counter - job.counter_at_enqueue > STALE_AFTER


class ChannelRegistry:
    """Per-channel state, debounce, and drop-old job queue."""

    def __init__(self, manager, base_threshold: float, debounce_s: float) -> None:
        self._manager = manager
        self._base_threshold = base_threshold
        self._debounce_s = debounce_s
        self._channels: dict[int, ChannelState] = {}

    def get(self, channel_id: int) -> ChannelState:
        state = self._channels.get(channel_id)
        if state is None:
            state = ChannelState(threshold=self._base_threshold)
            self._channels[channel_id] = state
        return state

    def append(self, channel_id: int, msg: Msg) -> ChannelState:
        state = self.get(channel_id)
        state.buffer.append(msg)
        state.counter += 1
        return state

    def schedule(self, channel_id: int, forced: bool, trigger_message_id: int) -> None:
        state = self.get(channel_id)
        if forced:
            self._cancel_debounce(state)
            self._enqueue(state, self._build_job(channel_id, state, forced, trigger_message_id))
            self._ensure_worker(channel_id, state)
            return

        if state.debounce is not None:
            state.debounce.cancel()
        loop = asyncio.get_running_loop()
        state.debounce = loop.call_later(self._debounce_s, self._fire, channel_id, trigger_message_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _cancel_debounce(state: ChannelState) -> None:
        if state.debounce is not None:
            state.debounce.cancel()
            state.debounce = None

    def _fire(self, channel_id: int, trigger_message_id: int) -> None:
        state = self._channels.get(channel_id)
        if state is None:
            return
        self._enqueue(state, self._build_job(channel_id, state, False, trigger_message_id))
        self._ensure_worker(channel_id, state)

    @staticmethod
    def _build_job(
        channel_id: int,
        state: ChannelState,
        forced: bool,
        trigger_message_id: int,
    ) -> Job:
        return Job(
            channel_id=channel_id,
            snapshot=list(state.buffer),
            counter_at_enqueue=state.counter,
            forced=forced,
            trigger_message_id=trigger_message_id,
        )

    @staticmethod
    def _enqueue(state: ChannelState, job: Job) -> None:
        if state.queue.full():
            try:
                state.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        state.queue.put_nowait(job)

    def _ensure_worker(self, channel_id: int, state: ChannelState) -> None:
        if state.worker is None or state.worker.done():
            state.worker = asyncio.get_running_loop().create_task(
                worker(channel_id, state, self._manager)
            )


async def worker(channel_id: int, state: ChannelState, manager) -> None:
    while True:
        job = await state.queue.get()
        try:
            channel = manager.get_channel(channel_id)
            reply: Reply | None = None
            if channel is not None:
                async with channel.typing():
                    started = time.perf_counter()
                    reply = await manager.process(job)
                    if reply:
                        want = max(MIN_TYPING_S, len(reply.text) / TYPING_RATE)
                        elapsed = time.perf_counter() - started
                        if elapsed < want:
                            await asyncio.sleep(want - elapsed)

            if reply:
                if is_stale(state, job):
                    if not job.forced:
                        reply = None
                    else:
                        reply.anchored = True

            if reply:
                await manager.deliver(job, reply)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("worker_failed channel=%s", channel_id)
        finally:
            state.queue.task_done()
