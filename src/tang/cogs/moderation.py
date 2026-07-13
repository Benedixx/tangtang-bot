from __future__ import annotations

import logging

import discord

LOGGER = logging.getLogger("tang.moderation")


class Moderation:
    """Trap channel handler — auto-ban users who post in designated channels."""

    def __init__(self, trap_channel_ids: frozenset[int]) -> None:
        self._trap_channel_ids = trap_channel_ids

    def is_trap(self, channel_id: int) -> bool:
        return channel_id in self._trap_channel_ids

    async def handle_trap(self, message: discord.Message) -> None:
        """Ban the user and delete their message."""
        guild = message.guild
        if guild is None:
            return

        user = message.author
        channel_name = getattr(message.channel, "name", str(message.channel.id))

        LOGGER.warning(
            "trap guild=%s channel=%s(%s) user=%s(%s)",
            guild.id, channel_name, message.channel.id, user.display_name, user.id,
        )

        # Delete trigger message
        try:
            await message.delete()
        except discord.Forbidden:
            LOGGER.warning("trap: missing MANAGE_MESSAGES to delete")
        except discord.HTTPException:
            LOGGER.exception("trap: failed to delete message")

        # Ban + purge 7 days
        try:
            await guild.ban(
                user,
                reason="Get a fucking life dude",
                delete_message_seconds=604800,
            )
            LOGGER.warning("trap_ban guild=%s user=%s(%s)", guild.id, user.display_name, user.id)
        except discord.Forbidden:
            LOGGER.error(
                "trap_ban_forbidden: bot lacks BAN_MEMBERS or target outranks bot "
                "guild=%s user=%s(%s)",
                guild.id, user.display_name, user.id,
            )
        except discord.HTTPException:
            LOGGER.exception("trap_ban_failed guild=%s", guild.id)
