from __future__ import annotations

import logging
import re
import time

import discord

from .config import Config
from .filters import tier0_reason
from .gate import Gate
from .groq import GroqClient
from .guard import Guard
from .persona import PersonaBuilder, sanitize
from .queue import ChannelRegistry, Msg, Reply
from .responder import Responder
from .tools.gifs import GifStore
from .tools.search import WebSearchTool
from .trap import TrapGuard

LOGGER = logging.getLogger("tang.gateway")


class Gateway(discord.Client):
    def __init__(self, config: Config) -> None:
        intents = discord.Intents.default()
        intents.guilds = True
        intents.members = True
        intents.messages = True
        intents.message_content = True
        super().__init__(intents=intents)

        self.config = config

        self.trap = TrapGuard(config.trap)
        self.registry = ChannelRegistry(
            self,
            base_threshold=config.chat.base_threshold,
            debounce_s=config.chat.debounce_s,
        )
        self.gate_groq = GroqClient(config.groq_api_key, config.models.gate)
        self.gate = Gate(self.gate_groq, config.chat)
        self.guard = Guard()
        self.persona = PersonaBuilder(config)
        self.gif_store = GifStore(config.gif.dir, config.gif.manifest)
        self.search = WebSearchTool()
        self.responder = Responder(
            config.groq_api_key,
            config.models.responder,
            self.persona,
            self.gif_store,
            self.search,
        )

        self._allowed = frozenset(config.chat.allowed_channels)
        names = [config.bot_name, *config.bot_name_aliases]
        self._name_patterns = [
            re.compile(rf"(^|\W){re.escape(n)}(\W|$)", re.IGNORECASE)
            for n in names
            if n.strip()
        ]
        self._trap_names: frozenset[str] = frozenset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_ready(self) -> None:
        LOGGER.info(
            "ready user=%s id=%s",
            self.user, self.user.id if self.user else None,
        )
        self._trap_names = self._collect_trap_names()
        await self.gif_store.upload_all(self, self.config.gif.staging_channel)

    def _collect_trap_names(self) -> frozenset[str]:
        ids = set(self.config.trap.channels)
        names: set[str] = set()
        for guild in self.guilds:
            for ch in guild.channels:
                if ch.id in ids and isinstance(ch, discord.TextChannel):
                    names.add(ch.name.lower())
        return frozenset(names)

    # ------------------------------------------------------------------
    # Message pipeline
    # ------------------------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        if self.user is None or message.author.id == self.user.id:
            return

        # Trap guard — terminal.
        if message.guild is not None and self.trap.is_trap(message.channel.id):
            await self.trap.handle(message, self.user.id)
            return

        if not self._tracked(message):
            return

        msg = Msg(
            message_id=message.id,
            author_id=message.author.id,
            author_name=message.author.display_name,
            content=(message.content or "").strip(),
            is_bot=message.author.bot or message.webhook_id is not None,
        )
        if not msg.content:
            return

        channel_id = message.channel.id
        request_id = f"msg-{message.id}"
        state = self.registry.append(channel_id, msg)
        self._note_engagement(message, state)

        # Injection scan — canned reply, terminal.
        if self.guard.scan(msg.content):
            LOGGER.warning(
                "[%s] injection_attempt channel=%s author=%s content=%s",
                request_id, channel_id, message.author.id, msg.content[:120],
            )
            if self._is_forced(message):
                await self._send_canned(message)
            return

        forced = self._is_forced(message)

        # Tier 0 filter — drop the trigger, keep the buffer.
        reason = tier0_reason(message, self.config.chat, state, forced, time.time())
        if reason:
            LOGGER.info("[%s] tier0_drop channel=%s reason=%s", request_id, channel_id, reason)
            return

        self.registry.schedule(channel_id, forced, message.id)

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        if self.user is not None and payload.user_id == self.user.id:
            return
        state = self.registry.get(payload.channel_id)
        if state.watch_until <= time.time():
            return
        if any(m.message_id == payload.message_id and m.is_bot for m in state.buffer):
            state.watch_engaged = True

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    async def process(self, job) -> Reply | None:
        state = self.registry.get(job.channel_id)
        request_id = f"job-{job.trigger_message_id}"

        if not await self.gate.decide(job, state, time.time(), request_id):
            return None

        result = await self.responder.run(job, state, request_id)
        if result is None:
            return None

        reply_text = sanitize(result.text, self._trap_names)
        gif = (result.gif or "").strip()
        if gif and gif in reply_text:
            reply_text = reply_text.replace(gif, "").strip()
        if not reply_text and not gif:
            return None

        return Reply(text=reply_text, gif=gif)

    async def deliver(self, job, reply: Reply) -> None:
        channel = self.get_channel(job.channel_id)
        if channel is None or self.user is None:
            return
        state = self.registry.get(job.channel_id)

        content = reply.text
        if reply.gif:
            content = f"{reply.text}\n{reply.gif}" if reply.text else reply.gif

        try:
            if reply.anchored:
                ref = channel.get_partial_message(job.trigger_message_id)
                sent = await channel.send(content, reference=ref)
            else:
                sent = await channel.send(content)
        except discord.HTTPException:
            LOGGER.exception("send_failed channel=%s", job.channel_id)
            return

        state.last_reply_ts = time.time()
        if reply.gif and self.gif_store.contains(reply.gif):
            state.recent_gifs.append(reply.gif)

        self.registry.append(job.channel_id, Msg(
            message_id=sent.id,
            author_id=self.user.id,
            author_name=self.user.display_name,
            content=reply.text,
            is_bot=True,
        ))

        if not job.forced:
            state.interjections.append(time.time())
            self.gate.start_watch(state, time.time())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tracked(self, message: discord.Message) -> bool:
        if isinstance(message.channel, discord.DMChannel):
            return self.config.chat.dm_allowed
        if message.guild is None:
            return False
        # Empty allowed_channels means "all channels".
        return not self._allowed or message.channel.id in self._allowed

    def _is_forced(self, message: discord.Message) -> bool:
        if isinstance(message.channel, discord.DMChannel):
            return True
        if self.user is not None and self.user in message.mentions:
            return True
        ref = message.reference
        if ref is not None:
            if ref.message_id is not None:
                state = self.registry.get(message.channel.id)
                if any(m.message_id == ref.message_id and m.is_bot for m in state.buffer):
                    return True
            resolved = ref.resolved
            if isinstance(resolved, discord.Message) and self.user is not None:
                if resolved.author.id == self.user.id:
                    return True
        content = message.content or ""
        return any(p.search(content) for p in self._name_patterns)

    def _note_engagement(self, message: discord.Message, state) -> None:
        if state.watch_until <= time.time():
            return
        ref = message.reference
        if ref is None or ref.message_id is None:
            return
        if any(m.message_id == ref.message_id and m.is_bot for m in state.buffer):
            state.watch_engaged = True

    async def _send_canned(self, message: discord.Message) -> None:
        try:
            await message.reply(self.guard.canned(), mention_author=False)
        except discord.HTTPException:
            pass
