from __future__ import annotations

import pytest

from tang.memory.triggers import wants_memory


@pytest.mark.parametrize("text", [
    "inget ya gw keren",
    "ingat aja semua yang gw bilang",
    "ingetin gw besok bangun pagi",
    "coba taro di pikiran lu, gw suka pizza",
    "catetin ya, ulang tahun gw bulan depan",
    "jangan lupa gw kerja di bank",
    "simpen di kepala lu nama kucing gw",
    "remember that i hate mondays",
    "keep in mind gw gak suka telat",
    "don't forget the meeting",
    "INGET INI BANG",
])
def test_memory_requests_detected(text):
    assert wants_memory(text) is True


@pytest.mark.parametrize("text", [
    "halo bang gimana kabarnya",
    "itu ingatan orang gak jelas",
    "wkwkwk lucu banget",
    "",
])
def test_non_requests_not_detected(text):
    assert wants_memory(text) is False
