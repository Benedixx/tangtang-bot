from __future__ import annotations

import logging
import random

import discord

from .config import TrapConfig

LOGGER = logging.getLogger("tang.trap")

TRAP_DM_LINES = [
    "get a life",
    "touch grass bang",
    "wkwk kena trap, touch some grass",
    "nice try, sekarang keluar",
]


class TrapGuard:
    """Honeypot channel guard. Any message by a non-exempt user → ban."""

    def __init__(self, config: TrapConfig) -> None:
        self._config = config
        self._channels = frozenset(config.channels)
        self._exempt_roles = frozenset(config.exempt_roles)
        self._exempt_bots = frozenset(config.exempt_bots)

    def is_trap(self, channel_id: int) -> bool:
        return self._config.enabled and channel_id in self._channels

    def is_exempt(self, message: discord.Message, bot_user_id: int) -> bool:
        guild = message.guild
        if guild is None:
            return False
        author = message.author
        if author.id in self._exempt_bots:
            return True
        if guild.owner is not None and author.id == guild.owner.id:
            return True
        roles = getattr(author, "roles", ())
        if any(r.id in self._exempt_roles for r in roles):
            return True
        bot_member = guild.get_member(bot_user_id)
        if bot_member is not None and roles:
            if author.top_role >= bot_member.top_role:
                return True
        return False

    async def handle(self, message: discord.Message, bot_user_id: int) -> None:
        guild = message.guild
        if guild is None:
            return
        if self.is_exempt(message, bot_user_id):
            LOGGER.info("trap_skip_exempt channel=%s user=%s", message.channel.name, message.author.id)
            return

        user = message.author
        channel_name = getattr(message.channel, "name", str(message.channel.id))
        LOGGER.warning(
            "trap guild=%s channel=%s user=%s(%s)",
            guild.id, channel_name, user, user.id,
        )

        # 1. DM the snark line first — a closed DM must not block the ban.
        try:
            await user.send(random.choice(TRAP_DM_LINES))
        except (discord.Forbidden, discord.HTTPException):
            pass

        # 2. Ban with guild-wide message purge.
        try:
            await guild.ban(
                user,
                reason=f"trap: #{channel_name}",
                delete_message_seconds=self._config.delete_message_seconds,
            )
        except discord.Forbidden:
            LOGGER.error("trap_ban_forbidden guild=%s user=%s", guild.id, user.id)
            return
        except discord.HTTPException:
            LOGGER.exception("trap_ban_failed guild=%s", guild.id)
            return

        # 3. One-line notice to the mod channel.
        await self._notify_mod(guild, user, channel_name)

    async def _notify_mod(self, guild: discord.Guild, user, channel_name: str) -> None:
        channel_id = self._config.mod_log_channel
        if not channel_id:
            return
        channel = guild.get_channel(channel_id)
        if channel is None:
            try:
                channel = await guild.fetch_channel(channel_id)
            except discord.HTTPException:
                return
        try:
            await channel.send(f"banned {user} (id {user.id}) for trap: #{channel_name}")
        except discord.HTTPException:
            pass
