from __future__ import annotations

import json
import logging
from typing import Any

from .groq import GroqClient
from .persona import PersonaBuilder
from .queue import ChannelState, Job

LOGGER = logging.getLogger("tang.responder")

MAX_TOOL_ROUNDS = 2
HISTORY_OPEN = "[untrusted conversation]"
HISTORY_CLOSE = "[/untrusted]"


class Responder:
    """Tool loop + final reply via the responder model."""

    def __init__(
        self,
        client: GroqClient,
        persona: PersonaBuilder,
        schemas: list[dict[str, Any]],
        executors: dict[str, Any],
    ) -> None:
        self._client = client
        self._persona = persona
        self._schemas = schemas
        self._executors = executors

    async def run(self, job: Job, state: ChannelState, request_id: str) -> str | None:
        lines = [f"{m.author_name}: {m.content}" for m in job.snapshot]
        history = f"{HISTORY_OPEN}\n" + "\n".join(lines) + f"\n{HISTORY_CLOSE}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._persona.system_prompt()},
            {"role": "user", "content": history},
        ]

        try:
            for rnd in range(MAX_TOOL_ROUNDS):
                text, tool_calls = await self._client.complete_with_tools(
                    messages,
                    self._schemas,
                    temperature=0.9,
                    max_tokens=100,
                    request_id=request_id,
                    label=f"respond:r{rnd}",
                )
                if not tool_calls:
                    return text

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                })
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    executor = self._executors.get(name)
                    if executor is None:
                        result = f"unknown tool: {name}"
                    else:
                        try:
                            result = await executor(state, request_id=request_id, **args)
                        except Exception as exc:
                            result = f"tool error: {exc}"
                            LOGGER.exception("[%s] tool_error name=%s", request_id, name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result)[:1500],
                    })

            text, _ = await self._client.complete_with_tools(
                messages,
                self._schemas,
                temperature=0.9,
                max_tokens=100,
                request_id=request_id,
                label="respond:final",
            )
            return text
        except Exception:
            LOGGER.exception("[%s] responder_failed", request_id)
            return None
