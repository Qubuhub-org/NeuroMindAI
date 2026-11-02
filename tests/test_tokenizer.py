import pytest

from tokenizer import BpeTokenizer, ByteTokenizer


def test_utf8_round_trip():
    tokenizer = ByteTokenizer()
    text = "NeuroMindAI — intelligent systems 🧠"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_rejects_out_of_range_ids():
    with pytest.raises(ValueError):
        ByteTokenizer().decode([256])


def test_bpe_round_trip_with_unicode():
    text = "NeuroMindAI learns patterns. NeuroMindAI 🧠"
    tokenizer = BpeTokenizer.train(text * 4, vocab_size=300, min_frequency=2)
    assert tokenizer.vocab_size > 256
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_serialization_preserves_encoding():
    text = "abababab multilingual: café 🧠"
    tokenizer = BpeTokenizer.train(text * 5, vocab_size=280)
    restored = BpeTokenizer.from_dict(tokenizer.to_dict())
    assert restored.encode(text) == tokenizer.encode(text)
    assert restored.decode(restored.encode(text)) == text


def test_bpe_unknown_text_uses_byte_fallback():
    tokenizer = BpeTokenizer.train("aaaaabbbbb" * 4, vocab_size=270)
    text = "unseen 東京"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_rejects_invalid_vocabulary_size():
    with pytest.raises(ValueError):
        BpeTokenizer.train("text", vocab_size=128)
