from __future__ import annotations

import logging
import time

from openai import AsyncOpenAI

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
