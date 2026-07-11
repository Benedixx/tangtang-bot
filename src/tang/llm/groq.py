from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletionMessageToolCall

LOGGER = logging.getLogger("tang.groq")


class GroqClient:
    """Fast LLM client for classification, guardrail, and tool decisions."""

    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._model = model

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 30,
        request_id: str | None = None,
        request_label: str = "groq",
    ) -> str:
        started = time.perf_counter()
        LOGGER.info("[%s] groq_start label=%s", request_id or "-", request_label)

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            LOGGER.exception("[%s] groq_error label=%s", request_id or "-", request_label)
            raise

        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        LOGGER.info(
            "[%s] groq_done label=%s duration_ms=%s chars=%s",
            request_id or "-", request_label,
            int((time.perf_counter() - started) * 1000), len(text),
        )
        return text

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 200,
        tools: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        request_label: str = "groq",
    ) -> tuple[str | None, list[ChatCompletionMessageToolCall] | None]:
        started = time.perf_counter()
        LOGGER.info("[%s] groq_tools_start label=%s", request_id or "-", request_label)

        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if getattr(exc, "code", None) == "tool_use_failed":
                LOGGER.info("[%s] groq_tool_use_failed — plain text fallback", request_id or "-")
                return "", None
            LOGGER.exception("[%s] groq_tools_error", request_id or "-")
            raise
        except Exception:
            LOGGER.exception("[%s] groq_tools_error", request_id or "-")
            raise

        if not resp.choices:
            return "", None

        msg = resp.choices[0].message
        if msg.tool_calls:
            LOGGER.info(
                "[%s] groq_tool_calls label=%s count=%s",
                request_id or "-", request_label, len(msg.tool_calls),
            )
            return None, list(msg.tool_calls)

        text = (msg.content or "").strip()
        LOGGER.info(
            "[%s] groq_tools_done label=%s chars=%s",
            request_id or "-", request_label, len(text),
        )
        return text, None

    async def stream_complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.5,
        max_tokens: int = 500,
        request_id: str | None = None,
        request_label: str = "generic",
    ) -> AsyncGenerator[str, None]:
        """Groq streaming — used infrequently, mainly for debugging."""
        started = time.perf_counter()
        LOGGER.info("[%s] groq_stream_start label=%s", request_id or "-", request_label)

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        total = 0
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                total += len(delta)
                yield delta

        LOGGER.info(
            "[%s] groq_stream_done label=%s duration_ms=%s chars=%s",
            request_id or "-", request_label,
            int((time.perf_counter() - started) * 1000), total,
        )
