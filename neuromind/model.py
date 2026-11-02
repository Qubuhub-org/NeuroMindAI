"""NeuroMindAI decoder architectures.

neuromind_v1 is a native pre-norm decoder using RMSNorm, rotary position
embeddings, fused causal self-attention, and a SwiGLU feed-forward network.
The legacy path loads checkpoints created by the initial Transformer baseline.
"""

import torch
from torch import nn
from torch.nn import functional as F

from neuromind.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden * torch.rsqrt(variance + self.eps).to(hidden.dtype)
        return normalized * self.weight


def apply_rotary(
    query: torch.Tensor, key: torch.Tensor, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to [batch, heads, time, head_dim] tensors."""
    sequence_length = query.size(-2)
    head_dim = query.size(-1)
    device = query.device
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
            / head_dim
        )
    )
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    cosine = angles.cos().to(query.dtype)[None, None, :, :]
    sine = angles.sin().to(query.dtype)[None, None, :, :]

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        )
        return rotated.flatten(-2)

    return rotate(query), rotate(key)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.rope_theta = config.rope_theta
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = hidden.shape
        qkv = self.qkv(hidden).view(
            batch, sequence, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_rotary(query, key, self.rope_theta)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.residual_dropout(self.output(attended))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(hidden)) * self.up(hidden)))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.mlp(self.mlp_norm(hidden))


class NeuroMindLM(nn.Module):
    """Decoder-only language model with a checkpoint-compatible legacy path."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        if config.architecture == "legacy_transformer":
            self.position_embedding = nn.Embedding(
                config.context_length, config.d_model
            )
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
        else:
            self.blocks = nn.ModuleList(
                [DecoderBlock(config) for _ in range(config.n_layers)]
            )
            self.final_norm = RMSNorm(config.d_model)

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

        hidden = self.token_embedding(input_ids)
        if self.config.architecture == "legacy_transformer":
            positions = torch.arange(sequence_length, device=input_ids.device)
            hidden = hidden + self.position_embedding(positions)[None, :, :]
            causal_mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    device=input_ids.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            hidden = self.blocks(hidden, mask=causal_mask)
        else:
            for block in self.blocks:
                hidden = block(hidden)

        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != (batch_size, sequence_length):
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss
