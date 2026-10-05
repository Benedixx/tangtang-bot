from __future__ import annotations

from types import SimpleNamespace

from tang.config import ChatConfig
from tang.filters import is_command, tier0_reason


def _msg(content: str):
    return SimpleNamespace(content=content, author=SimpleNamespace(bot=False), webhook_id=None)


def test_commands_and_textless_messages_dropped():
    assert is_command("$wr linnea") and is_command("!k tes")
    assert not is_command("$ 20 itu mahal") and not is_command("halo $wg")
    cfg = ChatConfig()
    assert tier0_reason(_msg("<:NilouShock:1233332881649172490>"), cfg) == "no_text"
    assert tier0_reason(_msg("<@737533245138141234> @here"), cfg) == "no_text"
    assert tier0_reason(_msg("https://klipy.com/gifs/robin-sad-2"), cfg) == "no_text"
    assert tier0_reason(_msg("<@1339848001316720660> coba ketik"), cfg) is None
