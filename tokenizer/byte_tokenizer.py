"""Deterministic UTF-8 byte tokenizer.

Each UTF-8 byte maps directly to an integer in [0, 255]. This baseline has no
learned vocabulary and is fully reversible for valid UTF-8 input.
"""


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        if any(not isinstance(token, int) or token < 0 or token >= self.vocab_size for token in token_ids):
            raise ValueError("token IDs must be integers in [0, 255]")
        return bytes(token_ids).decode("utf-8", errors="replace")
