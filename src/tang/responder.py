from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.settings import ModelSettings

from .persona import PersonaBuilder
from .queue import ChannelState, Job
from .tools.gifs import GIF_TAGS, GifStore
from .tools.search import WebSearchTool

LOGGER = logging.getLogger("tang.responder")

HISTORY_OPEN = "[untrusted conversation]"
HISTORY_CLOSE = "[/untrusted]"

GifTag = Enum("GifTag", {t: t for t in GIF_TAGS}, type=str)


@dataclass(slots=True)
class ResponderReply:
    text: str
    gif: str = ""


@dataclass(slots=True)
class ToolDeps:
    state: ChannelState
    request_id: str
    gif_url: str = ""


class Responder:
    """Pydantic-AI agent: web search + gif tools.

    Output is plain text; the gif URL is captured as a tool side-effect on
    ToolDeps so the model never has to echo it (gpt-oss + Groq's native JSON
    output parsing 400s when it tries).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        persona: PersonaBuilder,
        gif_store: GifStore,
        search: WebSearchTool,
    ) -> None:
        groq_model = GroqModel(model, settings=ModelSettings(
            temperature=0.9,
            max_tokens=300,
            thinking="minimal",
            timeout=30,
        ))
        self._agent = Agent(
            groq_model,
            system_prompt=persona.system_prompt(),
            deps_type=ToolDeps,
            retries=1,
        )

        @self._agent.tool(retries=0)
        async def send_gif(ctx: RunContext[ToolDeps], tag: GifTag) -> str:
            """Send a reaction GIF. Use when a GIF fits the moment better than
            words. The gif goes out on its own — no need to describe it."""
            tag_value = tag.value if isinstance(tag, GifTag) else str(tag)
            url = await gif_store.send(
                tag_value,
                list(ctx.deps.state.recent_gifs),
                ctx.deps.request_id,
            )
            if not url:
                return "no gif available for that tag"
            ctx.deps.gif_url = url
            return "udah kekirim"

        @self._agent.tool(retries=0)
        async def web_search(ctx: RunContext[ToolDeps], query: str) -> str:
            """Search the web for current information (news, prices, facts).
            Results include URLs — when naming a source, cite only a URL from
            the results, never invent one. Summarize the snippets in your reply."""
            return await search.execute(query, ctx.deps.request_id)

    async def run(self, job: Job, state: ChannelState, request_id: str) -> ResponderReply | None:
        lines = [f"{m.author_name}: {m.content}" for m in job.snapshot]
        LOGGER.info(
            "[%s] respond_context msgs=%s chars=%s",
            request_id, len(job.snapshot), sum(len(l) for l in lines),
        )
        history = f"{HISTORY_OPEN}\n" + "\n".join(lines) + f"\n{HISTORY_CLOSE}"
        deps = ToolDeps(state, request_id)
        try:
            result = await self._agent.run(history, deps=deps)
        except Exception:
            LOGGER.exception("[%s] responder_failed", request_id)
            return None
        return ResponderReply(result.output, deps.gif_url)
