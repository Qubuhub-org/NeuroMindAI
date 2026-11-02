"""Compact causal Transformer language model."""

import torch
from torch import nn
from torch.nn import functional as F

from neuromind.config import ModelConfig


class NeuroMindLM(nn.Module):
    """Decoder-only Transformer predicting the next byte token."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        block = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=config.n_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, sequence_length = input_ids.shape
        if sequence_length == 0 or sequence_length > self.config.context_length:
            raise ValueError("sequence length must be within the configured context window")
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=causal_mask)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != (batch_size, sequence_length):
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
