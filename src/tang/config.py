from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class TrapConfig:
    enabled: bool = True
    channels: tuple[int, ...] = ()
    exempt_roles: tuple[int, ...] = ()
    exempt_bots: tuple[int, ...] = ()
    delete_message_seconds: int = 3600
    mod_log_channel: int = 0


@dataclass(slots=True)
class ChatConfig:
    allowed_channels: tuple[int, ...] = ()
    dm_allowed: bool = True
    min_length: int = 8
    cooldown_s: float = 120.0
    debounce_s: float = 3.0
    budget_per_hour: int = 4
    base_threshold: float = 0.65
    register: str = "gw_lu"


@dataclass(slots=True)
class ModelsConfig:
    gate: str = "openai/gpt-oss-20b"
    responder: str = "openai/gpt-oss-120b"


@dataclass(slots=True)
class GifConfig:
    staging_channel: int = 0
    dir: str = "data/gifs"
    manifest: str = "data/gif_manifest.json"


@dataclass(slots=True)
class Config:
    discord_token: str
    groq_api_key: str
    bot_name: str = "koyuki"
    bot_name_aliases: tuple[str, ...] = ()
    trap: TrapConfig = field(default_factory=TrapConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    gif: GifConfig = field(default_factory=GifConfig)
    persona_examples: str = "data/persona_examples.yaml"


def _require(name: str) -> str:
    val = os.getenv(name)
    if val:
        return val
    raise ValueError(f"Missing required environment variable: {name}")


def _csv_env(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def _int_list(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (int, str)):
        try:
            return (int(raw),)
        except (TypeError, ValueError):
            return ()
    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def _f(key: str, mapping: dict[str, Any], default: float) -> float:
    raw = mapping.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _i(key: str, mapping: dict[str, Any], default: int) -> int:
    raw = mapping.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _b(key: str, mapping: dict[str, Any], default: bool) -> bool:
    raw = mapping.get(key, default)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def load() -> Config:
    load_dotenv()

    env_path = os.getenv("CONFIG_PATH", "")
    config_path = Path(env_path) if env_path else ROOT / "config.yaml"
    data: dict[str, Any] = {}
    if config_path.exists():
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            raise ValueError(f"Failed to parse config file: {config_path}") from None
    if not isinstance(data, dict):
        data = {}

    trap = data.get("trap") if isinstance(data.get("trap"), dict) else {}
    chat = data.get("chat") if isinstance(data.get("chat"), dict) else {}
    models = data.get("models") if isinstance(data.get("models"), dict) else {}
    gif = data.get("gif") if isinstance(data.get("gif"), dict) else {}

    return Config(
        discord_token=_require("DISCORD_TOKEN"),
        groq_api_key=_require("GROQ_API_KEY"),
        bot_name=os.getenv("BOT_NAME", "koyuki"),
        bot_name_aliases=_csv_env("BOT_NAME_ALIASES"),
        trap=TrapConfig(
            enabled=_b("enabled", trap, True),
            channels=_int_list(trap.get("channels")),
            exempt_roles=_int_list(trap.get("exempt_roles")),
            exempt_bots=_int_list(trap.get("exempt_bots")),
            delete_message_seconds=_i("delete_message_seconds", trap, 3600),
            mod_log_channel=_i("mod_log_channel", trap, 0),
        ),
        chat=ChatConfig(
            allowed_channels=_int_list(chat.get("allowed_channels")),
            dm_allowed=_b("dm_allowed", chat, True),
            min_length=_i("min_length", chat, 8),
            cooldown_s=_f("cooldown_s", chat, 120.0),
            debounce_s=_f("debounce_s", chat, 3.0),
            budget_per_hour=_i("budget_per_hour", chat, 4),
            base_threshold=_f("base_threshold", chat, 0.65),
            register=str(chat.get("register", "gw_lu")),
        ),
        models=ModelsConfig(
            gate=str(models.get("gate", "openai/gpt-oss-20b")),
            responder=str(models.get("responder", "openai/gpt-oss-120b")),
        ),
        gif=GifConfig(
            staging_channel=_i("staging_channel", gif, 0),
            dir=str(gif.get("dir", "data/gifs")),
            manifest=str(gif.get("manifest", "data/gif_manifest.json")),
        ),
        persona_examples=str(data.get("persona_examples", "data/persona_examples.yaml")),
    )
