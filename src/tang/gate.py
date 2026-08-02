from __future__ import annotations

import asyncio
import logging
from typing import Any

from .config import ChatConfig
from .groq import GroqClient
from .queue import ChannelState, Job

LOGGER = logging.getLogger("tang.gate")

_WATCH_S = 60.0
_THRESHOLD_RAISE = 0.05
_THRESHOLD_DECAY = 0.03
_THRESHOLD_MAX = 0.95

_GATE_SYSTEM = """\
You are the engagement scheduler for a casual Discord chat bot.

You receive recent conversation lines as "name: text". Decide whether the bot should send a message now.
- respond true only if the latest message is directed at the bot or the bot can naturally add something of value.
- otherwise respond false.
Output JSON only:
{"respond": true, "confidence": 0.72}
confidence is how sure you are (0.0 to 1.0)."""


class Gate:
    """Binary engagement gate with an adaptive per-channel threshold."""

    def __init__(self, client: GroqClient, config: ChatConfig) -> None:
        self._client = client
        self._config = config

    async def decide(self, job: Job, state: ChannelState, now: float, request_id: str) -> bool:
        if job.forced:
            return True
        if not self._within_budget(state, now):
            LOGGER.info("[%s] gate_budget_exhausted", request_id)
            return False

        context = "\n".join(f"{m.author_name}: {m.content}" for m in job.snapshot)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _GATE_SYSTEM},
            {"role": "user", "content": context or "(no conversation)"},
        ]
        for attempt in range(2):
            try:
                data = await self._client.complete_json(
                    messages,
                    temperature=0.0,
                    max_tokens=80 if attempt == 0 else 256,
                    request_id=request_id,
                    label="gate",
                )
            except Exception:
                LOGGER.exception("[%s] gate_error", request_id)
                return False
            if isinstance(data, dict) and "respond" in data and "confidence" in data:
                try:
                    respond = bool(data["respond"])
                    confidence = float(data["confidence"])
                except (TypeError, ValueError):
                    continue
                LOGGER.info(
                    "[%s] gate respond=%s conf=%.2f threshold=%.2f",
                    request_id, respond, confidence, state.threshold,
                )
                return respond and confidence >= state.threshold
        return False

    def start_watch(self, state: ChannelState, now: float) -> None:
        state.watch_until = now + _WATCH_S
        state.watch_engaged = False
        asyncio.get_running_loop().call_later(_WATCH_S, self._settle, state)

    def _settle(self, state: ChannelState) -> None:
        if state.watch_engaged:
            state.threshold = max(self._config.base_threshold, state.threshold - _THRESHOLD_DECAY)
            LOGGER.info("gate_threshold_decay threshold=%.2f", state.threshold)
        else:
            state.threshold = min(_THRESHOLD_MAX, state.threshold + _THRESHOLD_RAISE)
            LOGGER.info("gate_threshold_raise threshold=%.2f", state.threshold)

    def _within_budget(self, state: ChannelState, now: float) -> bool:
        state.interjections = [t for t in state.interjections if now - t < 3600.0]
        return len(state.interjections) < self._config.budget_per_hour
