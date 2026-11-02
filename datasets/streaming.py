"""Map-style and streaming datasets for next-token language modeling."""

from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset

from tokenizer import BpeTokenizer, ByteTokenizer


class TextWindowDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        context_length: int,
        tokenizer: ByteTokenizer | BpeTokenizer | None = None,
    ):
        if context_length < 1:
            raise ValueError("context_length must be positive")
        text = Path(path).read_text(encoding="utf-8")
        self.tokenizer = tokenizer or ByteTokenizer()
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= context_length:
            raise ValueError(
                f"corpus needs at least {context_length + 1} tokens; found {len(tokens)}"
            )
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return self.tokens.numel() - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.tokens[index : index + self.context_length + 1]
        return window[:-1], window[1:]


class StreamingTextWindowDataset(IterableDataset):
    """Read UTF-8 text line-by-line and emit overlapping token windows.

    The tokenizer is supplied by the caller and should already be trained.
    Distributed ranks are assigned disjoint input lines; windows never cross
    rank boundaries, avoiding duplicated samples at the cost of line boundaries.
    """

    def __init__(
        self,
        path: str | Path,
        context_length: int,
        tokenizer: ByteTokenizer | BpeTokenizer,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be within [0, world_size)")
        self.path = Path(path)
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        carry: list[int] = []
        with self.path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source):
                if line_number % self.world_size != self.rank:
                    continue
                tokens = carry + self.tokenizer.encode(line)
                limit = len(tokens) - self.context_length
                for start in range(max(0, limit)):
                    window = torch.tensor(
                        tokens[start : start + self.context_length + 1],
                        dtype=torch.long,
                    )
                    yield window[:-1], window[1:]
                carry = tokens[-self.context_length :]
