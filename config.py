from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv

DEFAULT_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"


@dataclass(slots=True)
class AppConfig:
    discord_token: str
    openrouter_api_keys: tuple[str, ...]
    model_name: str = DEFAULT_MODEL
    bot_name: str = "MucaSauce"
    bot_name_aliases: tuple[str, ...] = ("tangtang",)
    max_context_messages: int = 20
    summary_interval_messages: int = 30
    response_cooldown_seconds: int = 12
    max_response_chars: int = 900
    openrouter_site_name: str = "MUCA-SAUCEDiscordBot"
    openrouter_site_url: str | None = None
    memory_file_path: str = "data/tangtang_memory.json"
    web_search_enabled: bool = True
    web_search_max_results: int = 4
    web_search_min_interval_seconds: int = 20


def _require(name: str, value: str | None) -> str:
    if value:
        return value
    raise ValueError(f"Missing required environment variable: {name}")


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default

    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"Environment variable {name} must be a boolean value.")


def _collect_openrouter_keys() -> tuple[str, ...]:
    keys: list[str] = []

    csv_keys = os.getenv("OPENROUTER_API_KEYS", "")
    if csv_keys:
        for chunk in csv_keys.split(","):
            key = chunk.strip()
            if key:
                keys.append(key)

    single_env_candidates = [
        os.getenv("OPENROUTER_API_KEY"),
        os.getenv("OPENROUTER_KEY"),
        os.getenv("OPENROUTER_API_KEY_2"),
        os.getenv("OPENROUTER_KEY_2"),
    ]

    for value in single_env_candidates:
        if value:
            keys.append(value.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)

    if not deduped:
        raise ValueError(
            "Missing OpenRouter key. Set OPENROUTER_API_KEYS or OPENROUTER_API_KEY/OPENROUTER_KEY."
        )

    return tuple(deduped)


def _read_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    items = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not items:
        return default

    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(item)
    return tuple(deduped)


def load_config() -> AppConfig:
    load_dotenv()

    discord_token = _require("DISCORD_TOKEN", os.getenv("DISCORD_TOKEN"))
    openrouter_api_keys = _collect_openrouter_keys()

    return AppConfig(
        discord_token=discord_token,
        openrouter_api_keys=openrouter_api_keys,
        model_name=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        bot_name=os.getenv("BOT_NAME", "MucaSauce"),
        bot_name_aliases=_read_csv("BOT_NAME_ALIASES", ("tangtang",)),
        max_context_messages=_read_int("MAX_CONTEXT_MESSAGES", 20),
        summary_interval_messages=_read_int("SUMMARY_INTERVAL_MESSAGES", 30),
        response_cooldown_seconds=_read_int("RESPONSE_COOLDOWN_SECONDS", 12),
        max_response_chars=_read_int("MAX_RESPONSE_CHARS", 900),
        openrouter_site_name=os.getenv("OPENROUTER_SITE_NAME", "MUCA-SAUCEDiscordBot"),
        openrouter_site_url=os.getenv("OPENROUTER_SITE_URL"),
        memory_file_path=os.getenv("MEMORY_FILE_PATH", "data/tangtang_memory.json"),
        web_search_enabled=_read_bool("WEB_SEARCH_ENABLED", True),
        web_search_max_results=_read_int("WEB_SEARCH_MAX_RESULTS", 4),
        web_search_min_interval_seconds=_read_int("WEB_SEARCH_MIN_INTERVAL_SECONDS", 20),
    )
