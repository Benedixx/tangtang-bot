from __future__ import annotations

import logging
import time
from typing import Any

from openai import AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletionMessageToolCall

LOGGER = logging.getLogger("discord-bot.groq")


class GroqClient:
    def __init__(self, api_key: str, model: str = "compound-beta") -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 20,
        request_id: str | None = None,
        request_label: str = "groq",
    ) -> str:
        started = time.perf_counter()
        LOGGER.info(
            "[request=%s] groq_start label=%s model=%s",
            request_id,
            request_label,
            self.model,
        )

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            LOGGER.exception("[request=%s] groq_error label=%s", request_id, request_label)
            raise

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if not response.choices:
            LOGGER.info("[request=%s] groq_complete label=%s duration_ms=%s response_chars=0", request_id, request_label, elapsed_ms)
            return ""

        content = response.choices[0].message.content or ""
        LOGGER.info(
            "[request=%s] groq_complete label=%s duration_ms=%s response_chars=%s",
            request_id,
            request_label,
            elapsed_ms,
            len(content),
        )
        return content.strip()

    async def complete_with_tools(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 100,
        tools: list[dict] | None = None,
        request_id: str | None = None,
        request_label: str = "groq",
    ) -> tuple[str | None, list[ChatCompletionMessageToolCall] | None]:
        """Returns (None, tool_calls) or (text, None). Same contract as OpenRouterClient."""
        started = time.perf_counter()
        LOGGER.info(
            "[request=%s] groq_tools_start label=%s model=%s tools=%s",
            request_id,
            request_label,
            self.model,
            len(tools) if tools else 0,
        )

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if getattr(exc, "code", None) == "tool_use_failed":
                LOGGER.warning(
                    "[request=%s] groq_tool_use_failed label=%s — model generated plain text; breaking tool loop",
                    request_id,
                    request_label,
                )
                return "", None
            LOGGER.exception("[request=%s] groq_tools_error label=%s", request_id, request_label)
            raise
        except Exception:
            LOGGER.exception("[request=%s] groq_tools_error label=%s", request_id, request_label)
            raise

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if not response.choices:
            return "", None

        message = response.choices[0].message
        if message.tool_calls:
            LOGGER.info(
                "[request=%s] groq_tool_calls label=%s duration_ms=%s tool_count=%s",
                request_id,
                request_label,
                elapsed_ms,
                len(message.tool_calls),
            )
            return None, list(message.tool_calls)

        content = (message.content or "").strip()
        LOGGER.info(
            "[request=%s] groq_tools_complete label=%s duration_ms=%s response_chars=%s",
            request_id,
            request_label,
            elapsed_ms,
            len(content),
        )
        return content, None
