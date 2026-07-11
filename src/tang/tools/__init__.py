from __future__ import annotations

from typing import Any, Callable

from .gif_search import GifSearch, TOOL_NAME as GIF_NAME, TOOL_SCHEMA as GIF_SCHEMA
from .web_scraper import WebScraper, TOOL_NAME as SCRAPE_NAME, TOOL_SCHEMA as SCRAPE_SCHEMA
from .web_search import WebSearch, TOOL_NAME as SEARCH_NAME, TOOL_SCHEMA as SEARCH_SCHEMA

ExecuteFunc = Callable[..., Any]

# Memory tools are defined inline since they execute directly on the memory store.
MEMORY_UPSEART_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_save",
        "description": "Save or update a fact about the user in long-term memory. Use when the user shares personal info: name, preferences, habits, profile.",
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "Fact to remember, e.g. 'Alice likes black coffee'",
                }
            },
            "required": ["fact"],
        },
    },
}

MEMORY_DELETE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_delete",
        "description": "Delete a fact from memory that is outdated or incorrect.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Keyword to match facts for deletion",
                }
            },
            "required": ["keyword"],
        },
    },
}


def build_registry(
    web_search: WebSearch,
    web_scraper: WebScraper,
    gif_search: GifSearch,
    memory_store: Any,  # MemoryStore duck-typed
) -> tuple[list[dict[str, Any]], dict[str, ExecuteFunc]]:
    """Build (schemas, executors) for the LLM tool-calling loop."""

    schemas: list[dict[str, Any]] = [
        SEARCH_SCHEMA,
        SCRAPE_SCHEMA,
        GIF_SCHEMA,
        MEMORY_UPSEART_SCHEMA,
        MEMORY_DELETE_SCHEMA,
    ]

    async def _search(query: str, **kw: Any) -> str:
        return await web_search.execute(query, request_id=kw.pop("request_id", None))

    async def _scrape(url: str, **kw: Any) -> str:
        return await web_scraper.execute(url, request_id=kw.pop("request_id", None))

    async def _gif(query: str, **kw: Any) -> str:
        return await gif_search.execute(query, request_id=kw.pop("request_id", None))

    async def _mem_save(fact: str, **kw: Any) -> str:
        return memory_store.save_fact(fact)

    async def _mem_delete(keyword: str, **kw: Any) -> str:
        return memory_store.delete_fact(keyword)

    executors: dict[str, ExecuteFunc] = {
        SEARCH_NAME: _search,
        SCRAPE_NAME: _scrape,
        GIF_NAME: _gif,
        "memory_save": _mem_save,
        "memory_delete": _mem_delete,
    }

    return schemas, executors
