"""Deterministic byte-level BPE tokenizer with a complete UTF-8 fallback.

The initial 256 token IDs represent individual bytes. Learned merge tokens are
appended in training order, making the vocabulary deterministic and reversible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Merge:
    left: int
    right: int
    token_id: int


class BpeTokenizer:
    """A small, dependency-free byte-pair encoding tokenizer."""

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[Merge] = []
        self._merge_ranks: dict[tuple[int, int], tuple[int, int]] = {}
        self._children: dict[int, tuple[int, int]] = {}
        for left, right in merges or []:
            self._add_merge(left, right)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def _add_merge(self, left: int, right: int) -> None:
        token_id = 256 + len(self.merges)
        pair = (int(left), int(right))
        if pair in self._merge_ranks:
            raise ValueError(f"duplicate BPE merge pair: {pair}")
        if pair[0] >= token_id or pair[1] >= token_id or min(pair) < 0:
            raise ValueError("merge inputs must refer to previously defined tokens")
        self._merge_ranks[pair] = (len(self.merges), token_id)
        self._children[token_id] = pair
        self.merges.append(Merge(pair[0], pair[1], token_id))

    @classmethod
    def train(
        cls, text: str, vocab_size: int = 512, min_frequency: int = 2
    ) -> "BpeTokenizer":
        """Learn merges from UTF-8 bytes; ties are broken lexicographically."""
        if not isinstance(text, str):
            raise TypeError("training text must be a string")
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte fallback")
        if min_frequency < 2:
            raise ValueError("min_frequency must be at least 2")
        sequence = list(text.encode("utf-8"))
        tokenizer = cls()
        while tokenizer.vocab_size < vocab_size and len(sequence) > 1:
            counts = Counter(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, frequency = min(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            if frequency < min_frequency:
                break
            tokenizer._add_merge(*pair)
            new_id = tokenizer.vocab_size - 1
            merged: list[int] = []
            index = 0
            while index < len(sequence):
                if (
                    index + 1 < len(sequence)
                    and sequence[index] == pair[0]
                    and sequence[index + 1] == pair[1]
                ):
                    merged.append(new_id)
                    index += 2
                else:
                    merged.append(sequence[index])
                    index += 1
            sequence = merged
        return tokenizer

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            best_rank = None
            best_pair = None
            for pair in zip(tokens, tokens[1:]):
                ranked = self._merge_ranks.get(pair)
                if ranked is not None and (best_rank is None or ranked[0] < best_rank):
                    best_rank = ranked[0]
                    best_pair = pair
            if best_pair is None:
                break
            token_id = self._merge_ranks[best_pair][1]
            merged: list[int] = []
            index = 0
            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and tokens[index] == best_pair[0]
                    and tokens[index + 1] == best_pair[1]
                ):
                    merged.append(token_id)
                    index += 2
                else:
                    merged.append(tokens[index])
                    index += 1
            tokens = merged
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        output = bytearray()
        for token_id in token_ids:
            if not isinstance(token_id, int) or not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token ID must be an integer in [0, {self.vocab_size - 1}]")
            stack = [token_id]
            while stack:
                current = stack.pop()
                children = self._children.get(current)
                if children is None:
                    output.append(current)
                else:
                    stack.append(children[1])
                    stack.append(children[0])
        return output.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {
            "type": "bpe",
            "version": 1,
            "base_vocab_size": 256,
            "merges": [[merge.left, merge.right] for merge in self.merges],
        }

    @classmethod
    def from_dict(cls, state: dict) -> "BpeTokenizer":
        if state.get("type") != "bpe" or state.get("version") != 1:
            raise ValueError("unsupported tokenizer format or version")
        if state.get("base_vocab_size") != 256:
            raise ValueError("BPE tokenizer requires a 256-byte base vocabulary")
        return cls([tuple(pair) for pair in state.get("merges", [])])
