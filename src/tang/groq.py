from __future__ import annotations

import json
import logging
import re
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
        """Completion parsed as a dict. No Groq JSON mode — the strict schema
        validation 400s on this model class, so we request JSON in the prompt
        and parse leniently (fences, prose, nested braces)."""
        started = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
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
        data = _extract_json(text)
        if data is None:
            LOGGER.warning(
                "[%s] groq_json_parse_failed label=%s text=%s",
                request_id or "-", label, text[:160],
            )
            return {}
        return data

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


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Lenient JSON-object extraction from a model reply."""
    if not text:
        return None
    cleaned = _FENCE.sub("", text.strip())

    def _parse(raw: str) -> dict[str, Any] | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    data = _parse(cleaned)
    if data is not None:
        return data
    m = _JSON_OBJECT.search(cleaned)
    if m:
        data = _parse(m.group(0))
        if data is not None:
            return data
    return None
