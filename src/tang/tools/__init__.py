from __future__ import annotations

from typing import Any, Callable

from .gifs import GifStore, TOOL_NAME as GIF_NAME, TOOL_SCHEMA as GIF_SCHEMA
from .search import TOOL_NAME as SEARCH_NAME, TOOL_SCHEMA as SEARCH_SCHEMA, WebSearchTool

ExecFunc = Callable[..., Any]


def build_registry(
    gif_store: GifStore,
    search: WebSearchTool,
) -> tuple[list[dict[str, Any]], dict[str, ExecFunc]]:
    """Build (schemas, executors). Executors take the channel state first."""

    async def _send_gif(state: Any, tag: str, request_id: str | None = None) -> str:
        return await gif_store.send(tag, list(state.recent_gifs), request_id)

    async def _web_search(state: Any, query: str, request_id: str | None = None) -> str:
        return await search.execute(query, request_id)

    schemas = [GIF_SCHEMA, SEARCH_SCHEMA]
    executors: dict[str, ExecFunc] = {
        GIF_NAME: _send_gif,
        SEARCH_NAME: _web_search,
    }
    return schemas, executors
