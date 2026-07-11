from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol

from openai.types.chat import ChatCompletionMessageToolCall


class LLMClient(Protocol):
    """Shared interface that both OpenRouter and Groq implement."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 350,
        request_id: str | None = None,
        request_label: str = "generic",
    ) -> str: ...

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 200,
        tools: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        request_label: str = "generic",
    ) -> tuple[str | None, list[ChatCompletionMessageToolCall] | None]: ...

    async def stream_complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.5,
        max_tokens: int = 500,
        request_id: str | None = None,
        request_label: str = "generic",
    ) -> AsyncGenerator[str, None]: ...
