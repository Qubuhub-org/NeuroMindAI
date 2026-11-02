"""Sliding-window dataset for next-token language modeling."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer import ByteTokenizer


class TextWindowDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        context_length: int,
        tokenizer: ByteTokenizer | None = None,
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
