from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, Sequence

from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageToolCall

LOGGER = logging.getLogger("tang.openrouter")


class OpenRouterClient:
    """OpenRouter LLM client with key rotation and streaming."""

    def __init__(
        self,
        api_keys: Sequence[str],
        model_name: str,
        site_name: str | None = None,
        site_url: str | None = None,
    ) -> None:
        if not api_keys:
            raise ValueError("OpenRouterClient requires at least one API key.")

        self._clients = [
            AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0)
            for key in api_keys
            if key
        ]
        if not self._clients:
            raise ValueError("OpenRouterClient requires at least one non-empty API key.")

        self._model = model_name
        self._site_name = site_name
        self._site_url = site_url
        self._next = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rate_limit_delay(exc: RateLimitError) -> float | None:
        try:
            body = exc.body
            if isinstance(body, dict):
                raw = (
                    body.get("error", {})
                    .get("metadata", {})
                    .get("retry_after_seconds")
                    or body.get("error", {})
                    .get("metadata", {})
                    .get("retry_after_seconds_raw")
                )
                if raw is not None:
                    return float(raw)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    MAX_RETRIES_PER_KEY = 2

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 350,
        request_id: str | None = None,
        request_label: str = "generic",
    ) -> str:
        headers = self._headers()
        started = time.perf_counter()
        self._log_start(request_id, request_label, temperature, max_tokens, len(messages))

        for key_idx, (client, slot) in enumerate(self._attempts(), start=1):
            for retry in range(self.MAX_RETRIES_PER_KEY):
                try:
                    resp = await client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_headers=headers or None,
                    )
                except RateLimitError as exc:
                    delay = self._rate_limit_delay(exc)
                    if retry < self.MAX_RETRIES_PER_KEY - 1 and delay is not None:
                        LOGGER.warning(
                            "[%s] or_rate_limited label=%s key_slot=%s retry=%s waiting=%.1fs",
                            request_id or "-", request_label, slot, retry + 1, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    self._log_error(request_id, request_label, slot, key_idx, exc)
                    break
                except Exception as exc:
                    self._log_error(request_id, request_label, slot, key_idx, exc)
                    break

                text = self._extract(resp.choices[0].message.content)
                self._log_done(request_id, request_label, started, len(text), slot, key_idx)
                return text

        raise RuntimeError("OpenRouter request failed without a specific exception.")

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 200,
        tools: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        request_label: str = "generic",
    ) -> tuple[str | None, list[ChatCompletionMessageToolCall] | None]:
        headers = self._headers()
        LOGGER.info(
            "[%s] or_start label=%s tools=%s",
            request_id or "-", request_label, len(tools) if tools else 0,
        )

        for key_idx, (client, slot) in enumerate(self._attempts(), start=1):
            for retry in range(self.MAX_RETRIES_PER_KEY):
                kwargs: dict[str, Any] = dict(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=headers or None,
                )
                if tools:
                    kwargs["tools"] = tools

                try:
                    resp = await client.chat.completions.create(**kwargs)
                except RateLimitError as exc:
                    delay = self._rate_limit_delay(exc)
                    if retry < self.MAX_RETRIES_PER_KEY - 1 and delay is not None:
                        LOGGER.warning(
                            "[%s] or_rate_limited label=%s key_slot=%s retry=%s waiting=%.1fs",
                            request_id or "-", request_label, slot, retry + 1, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    LOGGER.error("[%s] or_tools_error label=%s key_slot=%s: %s", request_id or "-", request_label, slot, exc)
                    break
                except Exception:
                    LOGGER.exception("[%s] or_tools_error label=%s key_slot=%s", request_id or "-", request_label, slot)
                    break

                if not resp.choices:
                    return "", None

                msg = resp.choices[0].message
                if msg.tool_calls:
                    LOGGER.info("[%s] or_tool_calls label=%s count=%s", request_id or "-", request_label, len(msg.tool_calls))
                    return None, list(msg.tool_calls)

                text = self._extract(msg.content)
                LOGGER.info("[%s] or_tools_done label=%s chars=%s", request_id or "-", request_label, len(text))
                return text, None

        raise RuntimeError("OpenRouter tools request failed.")

    async def stream_complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.5,
        max_tokens: int = 500,
        request_id: str | None = None,
        request_label: str = "generic",
    ) -> AsyncGenerator[str, None]:
        headers = self._headers()
        key_attempts = self._attempts()
        last_exc: Exception | None = None

        for key_idx, (client, slot) in enumerate(key_attempts, start=1):
            for retry in range(self.MAX_RETRIES_PER_KEY):
                started = time.perf_counter()
                try:
                    LOGGER.info(
                        "[%s] or_stream_start label=%s key_slot=%s key_attempt=%s retry=%s",
                        request_id or "-", request_label, slot, key_idx, retry + 1,
                    )

                    stream = await client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        extra_headers=headers or None,
                    )

                    total = 0
                    deadline = started + 8.0
                    async for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta.content
                        if delta:
                            total += len(delta)
                            yield delta
                        elif total == 0 and time.perf_counter() > deadline:
                            LOGGER.warning("[%s] stream_no_content_timeout", request_id or "-")
                            break

                    LOGGER.info(
                        "[%s] or_stream_done label=%s duration_ms=%s chars=%s key_slot=%s key_attempt=%s",
                        request_id or "-", request_label,
                        int((time.perf_counter() - started) * 1000), total, slot, key_idx,
                    )
                    return

                except RateLimitError as exc:
                    last_exc = exc
                    delay = self._rate_limit_delay(exc)
                    if retry < self.MAX_RETRIES_PER_KEY - 1 and delay is not None:
                        LOGGER.warning(
                            "[%s] or_rate_limited label=%s key_slot=%s retry=%s waiting=%.1fs",
                            request_id or "-", request_label, slot, retry + 1, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    self._log_error(request_id, request_label, slot, key_idx, exc)
                    break
                except Exception as exc:
                    last_exc = exc
                    self._log_error(request_id, request_label, slot, key_idx, exc)
                    break

        LOGGER.error("[%s] or_all_keys_exhausted label=%s", request_id or "-", request_label)
        if last_exc is not None:
            raise last_exc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self._site_url:
            h["HTTP-Referer"] = self._site_url
        if self._site_name:
            h["X-Title"] = self._site_name
        return h

    def _attempts(self):
        """Round-robin over clients with retry order on each call."""
        count = len(self._clients)
        start = self._next
        self._next = (self._next + 1) % count
        return [(self._clients[(start + i) % count], (start + i) % count + 1) for i in range(count)]

    def _log_start(self, rid, label, temp, mt, n_msg):
        LOGGER.info(
            "[%s] or_start label=%s temp=%.2f max_tokens=%s messages=%s keys=%s",
            rid or "-", label, temp, mt, n_msg, len(self._clients),
        )

    def _log_error(self, rid, label, idx, attempt, exc):
        LOGGER.error(
            "[%s] or_error label=%s key_slot=%s attempt=%s: %s",
            rid or "-", label, idx, attempt, exc,
        )

    def _log_done(self, rid, label, started, chars, idx, attempt):
        ms = int((time.perf_counter() - started) * 1000)
        LOGGER.info(
            "[%s] or_done label=%s duration_ms=%s chars=%s key_slot=%s attempt=%s",
            rid or "-", label, ms, chars, idx, attempt,
        )

    @staticmethod
    def _extract(content: Any) -> str:
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
                elif hasattr(item, "text"):
                    t = item.text
                    if isinstance(t, str):
                        parts.append(t)
            return "".join(parts)
        return str(content)
