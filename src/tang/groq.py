from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

LOGGER = logging.getLogger("tang.groq")

# gpt-oss models produce reasoning tokens that count against max_tokens.
# At the default effort the gate/responder budgets get eaten and content
# comes back empty, so pin it low.
REASONING_EFFORT = "low"


class GroqClient:
    """Thin Groq chat completions wrapper: plain, JSON, and tool calling."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            max_retries=0,
        )
        self._model = model

    @staticmethod
    def _kwargs(
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": None,  # set by caller
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        return kwargs

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 80,
        request_id: str | None = None,
        label: str = "json",
        reasoning_effort: str | None = REASONING_EFFORT,
    ) -> dict[str, Any]:
        """Completion parsed as a dict. No Groq JSON mode — the strict schema
        validation 400s on this model class, so we request JSON in the prompt
        and parse leniently (fences, prose, nested braces)."""
        started = time.perf_counter()
        kwargs = self._kwargs(messages, temperature, max_tokens, reasoning_effort)
        kwargs["model"] = self._model
        try:
            resp = await self._client.chat.completions.create(**kwargs)
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
