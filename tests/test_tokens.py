from tang.memory.tokens import count_tokens


def test_empty_string():
    assert count_tokens("") == 0


def test_longer_text_more_tokens():
    short = "halo"
    long = "gw lagi bikin discord bot pake pydantic ai dan memory system"
    assert count_tokens(short) < count_tokens(long)


def test_monotonic_under_concatenation():
    a = "satu dua tiga empat lima"
    b = "enam tujuh delapan sembilan sepuluh"
    assert count_tokens(a + " " + b) >= max(count_tokens(a), count_tokens(b))
