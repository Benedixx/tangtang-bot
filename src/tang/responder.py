from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.settings import ModelSettings

from .config import MemoryConfig
from .memory.compaction import (
    ContextBudget,
    build_context,
    render_lines,
    render_memory_block,
    render_summary_block,
    trim_lines,
)
from .memory.models import RetrievedMemory, SessionSummary
from .memory.tokens import count_tokens
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
        memory_cfg: MemoryConfig | None = None,
    ) -> None:
        self._memory_cfg = memory_cfg or MemoryConfig()
        groq_model = GroqModel(model, settings=ModelSettings(
            temperature=0.9,
            max_tokens=700,
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

    async def run(
        self,
        job: Job,
        state: ChannelState,
        request_id: str,
        session=None,
        memories: list[RetrievedMemory] | None = None,
    ) -> ResponderReply | None:
        cfg = self._memory_cfg
        summarized = session.summarized_ids if session is not None else set()
        lines = render_lines(job.snapshot, summarized)
        lines = trim_lines(lines, cfg.recent_max_tokens)
        summary: SessionSummary | None = session.summary if session is not None else None
        memory_block = render_memory_block(memories or [])
        if memory_block:
            # Hard cap on injected memory tokens: drop lowest-scored first.
            while count_tokens(memory_block) > cfg.memory_max_tokens and len(memories) > 1:
                memories = sorted(memories, key=lambda r: r.score)[:-1] or None
                memory_block = render_memory_block(memories or [])
        context, budget = build_context(
            lines,
            render_summary_block(summary, cfg.summary_max_tokens),
            memory_block,
            HISTORY_OPEN,
            HISTORY_CLOSE,
        )
        LOGGER.info(
            "[%s] respond_context msgs=%s mem=%s tok=%s (mem=%s sum=%s raw=%s)",
            request_id, len(lines), len(memories or []),
            budget.total_tokens, budget.memory_tokens,
            budget.summary_tokens, budget.raw_tokens,
        )
        deps = ToolDeps(state, request_id)
        try:
            result = await self._agent.run(context, deps=deps)
        except Exception:
            LOGGER.exception("[%s] responder_failed", request_id)
            return None
        return ResponderReply(result.output, deps.gif_url)
