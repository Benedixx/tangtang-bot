from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

LOGGER = logging.getLogger("tang.groq")


class GroqClient:
    """Thin Groq chat completions wrapper: plain, JSON, and tool calling."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            max_retries=0,
        )
        self._model = model

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 100,
        request_id: str | None = None,
        label: str = "complete",
    ) -> str:
        started = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            LOGGER.exception("[%s] groq_error label=%s", request_id or "-", label)
            raise

        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        LOGGER.info(
            "[%s] groq_done label=%s ms=%s chars=%s",
            request_id or "-", label,
            int((time.perf_counter() - started) * 1000), len(text),
        )
        return text

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 80,
        request_id: str | None = None,
        label: str = "json",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            LOGGER.exception("[%s] groq_json_error label=%s", request_id or "-", label)
            raise

        text = (resp.choices[0].message.content or "") if resp.choices else ""
        LOGGER.info(
            "[%s] groq_json_done label=%s ms=%s",
            request_id or "-", label,
            int((time.perf_counter() - started) * 1000),
        )
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            LOGGER.warning("[%s] groq_json_parse_failed label=%s text=%s", request_id or "-", label, text[:120])
            return {}
        return data if isinstance(data, dict) else {}

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 200,
        request_id: str | None = None,
        label: str = "tools",
    ) -> tuple[str | None, list[ChatCompletionMessageToolCall] | None]:
        started = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
            )
        except Exception:
            LOGGER.exception("[%s] groq_tools_error label=%s", request_id or "-", label)
            raise

        if not resp.choices:
            return "", None
        msg = resp.choices[0].message
        if msg.tool_calls:
            LOGGER.info(
                "[%s] groq_tool_calls label=%s count=%s",
                request_id or "-", label, len(msg.tool_calls),
            )
            return None, list(msg.tool_calls)
        text = (msg.content or "").strip()
        LOGGER.info(
            "[%s] groq_tools_done label=%s chars=%s ms=%s",
            request_id or "-", label,
            len(text), int((time.perf_counter() - started) * 1000),
        )
        return text, None
