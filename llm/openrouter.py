from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, Sequence

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

LOGGER = logging.getLogger("discord-bot.openrouter")


class OpenRouterClient:
    def __init__(
        self,
        api_keys: Sequence[str] | None,
        model_name: str,
        site_name: str | None = None,
        site_url: str | None = None,
    ) -> None:
        if not api_keys:
            raise ValueError("OpenRouterClient requires at least one API key.")

        self._clients = [
            AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
            for key in api_keys
            if key
        ]
        if not self._clients:
            raise ValueError("OpenRouterClient requires at least one non-empty API key.")

        self._model_name = model_name
        self._site_name = site_name
        self._site_url = site_url
        self._next_client_index = 0

    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 350,
        request_id: str | None = None,
        request_label: str = "generic",
    ) -> str:
        headers: dict[str, str] = {}
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._site_name:
            headers["X-Title"] = self._site_name

        started = time.perf_counter()

        LOGGER.info(
            "[request=%s] openrouter_start label=%s model=%s temperature=%.2f max_tokens=%s prompt_messages=%s key_pool=%s",
            request_id,
            request_label,
            self._model_name,
            temperature,
            max_tokens,
            len(messages),
            len(self._clients),
        )

        attempt_order = self._next_attempt_order()
        last_error: Exception | None = None

        for attempt_number, client_index in enumerate(attempt_order, start=1):
            client = self._clients[client_index]
            try:
                response = await client.chat.completions.create(
                    model=self._model_name,
                    messages=list(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=headers or None,
                )
            except Exception as exc:
                last_error = exc
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                LOGGER.exception(
                    "[request=%s] openrouter_error label=%s duration_ms=%s key_slot=%s attempt=%s",
                    request_id,
                    request_label,
                    elapsed_ms,
                    client_index + 1,
                    attempt_number,
                )
                continue

            if not response.choices:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                LOGGER.info(
                    "[request=%s] openrouter_complete label=%s duration_ms=%s response_chars=0 key_slot=%s attempt=%s",
                    request_id,
                    request_label,
                    elapsed_ms,
                    client_index + 1,
                    attempt_number,
                )
                return ""

            text = self._extract_text(response.choices[0].message.content).strip()
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            LOGGER.info(
                "[request=%s] openrouter_complete label=%s duration_ms=%s response_chars=%s key_slot=%s attempt=%s",
                request_id,
                request_label,
                elapsed_ms,
                len(text),
                client_index + 1,
                attempt_number,
            )

            return text

        if last_error is not None:
            raise last_error

        raise RuntimeError("OpenRouter request failed without a specific exception.")

    async def complete_with_tools(
        self,
        messages: list[dict],
        temperature: float = 0.5,
        max_tokens: int = 300,
        tools: list[dict] | None = None,
        request_id: str | None = None,
        request_label: str = "generic",
    ) -> tuple[str | None, list[ChatCompletionMessageToolCall] | None]:
        """Returns (text, None) for final answer or (None, tool_calls) when LLM invokes tools."""
        headers: dict[str, str] = {}
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._site_name:
            headers["X-Title"] = self._site_name

        started = time.perf_counter()
        LOGGER.info(
            "[request=%s] openrouter_start label=%s model=%s temperature=%.2f max_tokens=%s prompt_messages=%s tools=%s",
            request_id,
            request_label,
            self._model_name,
            temperature,
            max_tokens,
            len(messages),
            len(tools) if tools else 0,
        )

        attempt_order = self._next_attempt_order()
        last_error: Exception | None = None

        for attempt_number, client_index in enumerate(attempt_order, start=1):
            client = self._clients[client_index]
            try:
                kwargs: dict[str, Any] = dict(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=headers or None,
                )
                if tools:
                    kwargs["tools"] = tools
                response = await client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                LOGGER.exception(
                    "[request=%s] openrouter_error label=%s duration_ms=%s key_slot=%s attempt=%s",
                    request_id,
                    request_label,
                    elapsed_ms,
                    client_index + 1,
                    attempt_number,
                )
                continue

            if not response.choices:
                return "", None

            message = response.choices[0].message
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            if message.tool_calls:
                LOGGER.info(
                    "[request=%s] openrouter_tool_calls label=%s duration_ms=%s tool_count=%s key_slot=%s",
                    request_id,
                    request_label,
                    elapsed_ms,
                    len(message.tool_calls),
                    client_index + 1,
                )
                return None, list(message.tool_calls)

            text = self._extract_text(message.content).strip()
            LOGGER.info(
                "[request=%s] openrouter_complete label=%s duration_ms=%s response_chars=%s key_slot=%s attempt=%s",
                request_id,
                request_label,
                elapsed_ms,
                len(text),
                client_index + 1,
                attempt_number,
            )
            return text, None

        if last_error is not None:
            raise last_error

        raise RuntimeError("OpenRouter request failed without a specific exception.")

    async def stream_complete(
        self,
        messages: Sequence[dict],
        temperature: float = 0.5,
        max_tokens: int = 500,
        request_id: str | None = None,
        request_label: str = "generic",
    ) -> AsyncGenerator[str, None]:
        headers: dict[str, str] = {}
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._site_name:
            headers["X-Title"] = self._site_name

        client_index = self._next_client_index % len(self._clients)
        self._next_client_index = (self._next_client_index + 1) % len(self._clients)
        client = self._clients[client_index]

        started = time.perf_counter()
        LOGGER.info(
            "[request=%s] openrouter_stream_start label=%s model=%s temperature=%.2f max_tokens=%s key_slot=%s",
            request_id,
            request_label,
            self._model_name,
            temperature,
            max_tokens,
            client_index + 1,
        )

        stream = await client.chat.completions.create(
            model=self._model_name,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_headers=headers or None,
        )

        total_chars = 0
        deadline = started + 8.0  # bail out if no content arrives within 8 s
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta is not None and total_chars == 0:
                LOGGER.debug(
                    "[request=%s] stream_first_chunk repr=%r",
                    request_id,
                    delta[:80] if delta else delta,
                )
            if delta:
                total_chars += len(delta)
                yield delta
            elif total_chars == 0 and time.perf_counter() > deadline:
                LOGGER.warning(
                    "[request=%s] stream_no_content_timeout label=%s elapsed_ms=%s",
                    request_id,
                    request_label,
                    int((time.perf_counter() - started) * 1000),
                )
                break

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        LOGGER.info(
            "[request=%s] openrouter_stream_complete label=%s duration_ms=%s total_chars=%s",
            request_id,
            request_label,
            elapsed_ms,
            total_chars,
        )

    def _next_attempt_order(self) -> list[int]:
        count = len(self._clients)
        start = self._next_client_index
        self._next_client_index = (self._next_client_index + 1) % count
        return [((start + offset) % count) for offset in range(count)]

    @staticmethod
    def _extract_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    continue

                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts)
        return str(content)
