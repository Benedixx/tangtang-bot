from __future__ import annotations

import asyncio
import logging

from .config import load
from .gateway import Gateway


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = load()
    gateway = Gateway(config)

    try:
        await gateway.start(config.discord_token)
    except KeyboardInterrupt:
        await gateway.close()


if __name__ == "__main__":
    asyncio.run(main())
