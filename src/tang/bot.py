from __future__ import annotations

import logging

import discord

from .cogs.chat import ChatHandler
from .cogs.moderation import Moderation
from .config import Config
from .guardrail import Guardrail
from .llm import GroqClient, OpenRouterClient
from .memory import MemoryStore
from .queue import GeneratorQueue
from .tools import GifSearch, WebScraper, WebSearch, build_registry

LOGGER = logging.getLogger("tang")


class TangBot(discord.Client):
    """Tang — a Discord chat AI. Clean, concurrent, conversational."""

    def __init__(self, config: Config) -> None:
        intents = discord.Intents.default()
        intents.guilds = True
        intents.messages = True
        intents.message_content = True

        super().__init__(intents=intents)

        self.config = config

        # LLM clients
        self.llm = OpenRouterClient(
            api_keys=config.openrouter_api_keys,
            model_name=config.model_name,
        )
        self.groq = (
            GroqClient(api_key=config.groq_api_key, model=config.groq_model)
            if config.groq_api_key
            else None
        )

        # Tools
        self.web_search = WebSearch(
            enabled=config.web_search_enabled,
            max_results=config.web_search_max_results,
            min_interval=config.web_search_min_interval_seconds,
        )
        self.web_scraper = WebScraper()
        self.gif_search = GifSearch(api_key=config.klipy_api_key)
        self.memory = MemoryStore(file_path=config.memory_file_path)

        self.tool_schemas, self.tool_executors = build_registry(
            web_search=self.web_search,
            web_scraper=self.web_scraper,
            gif_search=self.gif_search,
            memory_store=self.memory,
        )
        self.tools = (self.tool_schemas, self.tool_executors)

        # Guardrail
        self.guardrail = Guardrail(groq=self.groq)

        # Generation queue (single worker, no per-channel locks)
        self.gen_queue = GeneratorQueue()

        # Handlers
        self.chat = ChatHandler(self)
        self.mod = Moderation(
            trap_channel_ids=frozenset(config.trap_channel_ids)
        )

        # Build name patterns
        names = [config.bot_name, *config.bot_name_aliases]
        self.chat.build_name_patterns(names)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_ready(self) -> None:
        if self.user is not None:
            self.gen_queue.start()
            LOGGER.info("Tang is ready! (%s / %s)", self.user, self.user.id)

    async def on_message(self, message: discord.Message) -> None:
        """Main message dispatch — delegates to handlers."""

        # Ignore self
        if self.user is None or message.author.id == self.user.id:
            return

        # Trap channel check (before any other processing)
        if message.guild and self.mod.is_trap(message.channel.id):
            await self.mod.handle_trap(message)
            return

        # Chat handling
        await self.chat.on_message(message)
