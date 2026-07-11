from __future__ import annotations

import asyncio
import logging

from .bot import TangBot
from .config import load


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = load()
    bot = TangBot(config)

    try:
        await bot.start(config.discord_token)
    except KeyboardInterrupt:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
