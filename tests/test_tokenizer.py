import pytest

from tokenizer import ByteTokenizer


def test_utf8_round_trip():
    tokenizer = ByteTokenizer()
    text = "NeuroMindAI — intelligent systems 🧠"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_rejects_out_of_range_ids():
    with pytest.raises(ValueError):
        ByteTokenizer().decode([256])
