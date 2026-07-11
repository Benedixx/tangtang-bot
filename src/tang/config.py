from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"
DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_COOLDOWN = 12
DEFAULT_MAX_CONTEXT = 40


@dataclass(slots=True)
class Config:
    discord_token: str
    openrouter_api_keys: tuple[str, ...]
    groq_api_key: str
    model_name: str = DEFAULT_MODEL
    groq_model: str = DEFAULT_GROQ_MODEL
    bot_name: str = "Tang"
    bot_name_aliases: tuple[str, ...] = ("tang", "koyuki")
    max_context_messages: int = DEFAULT_MAX_CONTEXT
    response_cooldown_seconds: int = DEFAULT_COOLDOWN
    max_response_chars: int = 900
    memory_file_path: str = "data/tang_memory.json"
    web_search_enabled: bool = True
    web_search_max_results: int = 4
    web_search_min_interval_seconds: int = 20
    klipy_api_key: str = ""
    trap_channel_ids: tuple[int, ...] = ()
    bot_prefix: str = "!k"


def _require(name: str, value: str | None) -> str:
    if value:
        return value
    raise ValueError(f"Missing required environment variable: {name}")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _openrouter_keys() -> tuple[str, ...]:
    keys: list[str] = []

    csv = os.getenv("OPENROUTER_API_KEYS", "")
    if csv:
        keys.extend(k.strip() for k in csv.split(",") if k.strip())

    for env in ("OPENROUTER_API_KEY", "OPENROUTER_KEY", "OPENROUTER_API_KEY_2", "OPENROUTER_KEY_2", "OPENROUTER_API_KEY_3", "OPENROUTER_KEY_3"):
        val = os.getenv(env)
        if val:
            keys.append(val.strip())

    seen: set[str] = set()
    deduped = [k for k in keys if k and k not in seen and not seen.add(k)]

    if not deduped:
        raise ValueError(
            "Missing OpenRouter key. Set OPENROUTER_API_KEYS or OPENROUTER_API_KEY."
        )
    return tuple(deduped)


def _int_csv(name: str) -> tuple[int, ...]:
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return ()
    result: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            result.append(int(chunk))
        except ValueError:
            raise ValueError(f"{name} must be comma-separated integers") from None
    return tuple(result)


def _csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return default
    items = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not items:
        return default
    seen: set[str] = set()
    return tuple(item for item in items if item.lower() not in seen and not seen.add(item.lower()))


def load() -> Config:
    load_dotenv()

    return Config(
        discord_token=_require("DISCORD_TOKEN", os.getenv("DISCORD_TOKEN")),
        openrouter_api_keys=_openrouter_keys(),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        model_name=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        groq_model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        bot_name=os.getenv("BOT_NAME", "Tang"),
        bot_name_aliases=_csv("BOT_NAME_ALIASES", ("tang", "koyuki")),
        max_context_messages=_int("MAX_CONTEXT_MESSAGES", DEFAULT_MAX_CONTEXT),
        response_cooldown_seconds=_int("RESPONSE_COOLDOWN_SECONDS", DEFAULT_COOLDOWN),
        max_response_chars=_int("MAX_RESPONSE_CHARS", 900),
        memory_file_path=os.getenv("MEMORY_FILE_PATH", "data/tang_memory.json"),
        web_search_enabled=_bool("WEB_SEARCH_ENABLED", True),
        web_search_max_results=_int("WEB_SEARCH_MAX_RESULTS", 4),
        web_search_min_interval_seconds=_int("WEB_SEARCH_MIN_INTERVAL_SECONDS", 20),
        klipy_api_key=os.getenv("KLIPY_API_KEY", ""),
        trap_channel_ids=_int_csv("TRAP_CHANNEL_IDS"),
    )
