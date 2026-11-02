"""NeuroMindAI decoder architectures with optional incremental KV caching."""

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
    query: torch.Tensor,
    key: torch.Tensor,
    theta: float,
    position_offset: int = 0,
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
    positions = torch.arange(
        position_offset,
        position_offset + sequence_length,
        device=device,
        dtype=torch.float32,
    )
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

    def _project(self, hidden: torch.Tensor):
        batch, sequence, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, sequence, 3, width)
        query, key, value = qkv.unbind(dim=2)
        shape = (batch, sequence, self.n_heads, self.head_dim)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        return query, key, value

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        query, key, value = self._project(hidden)
        query, key = apply_rotary(query, key, self.rope_theta)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        batch, _, sequence, _ = attended.shape
        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, self.n_heads * self.head_dim
        )
        return self.residual_dropout(self.output(attended))

    def forward_with_cache(
        self,
        hidden: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query, key, value = self._project(hidden)
        past_length = 0 if past_key_value is None else past_key_value[0].size(-2)
        query, key = apply_rotary(
            query, key, self.rope_theta, position_offset=past_length
        )
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)

        query_length = query.size(-2)
        key_length = key.size(-2)
        query_positions = torch.arange(
            past_length, past_length + query_length, device=hidden.device
        )
        key_positions = torch.arange(key_length, device=hidden.device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask[None, None, :, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        batch, _, sequence, _ = attended.shape
        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, self.n_heads * self.head_dim
        )
        output = self.residual_dropout(self.output(attended))
        return output, (key, value)


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

    def forward_with_cache(
        self,
        hidden: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attention_output, present = self.attention.forward_with_cache(
            self.attention_norm(hidden), past_key_value
        )
        hidden = hidden + attention_output
        return hidden + self.mlp(self.mlp_norm(hidden)), present


class NeuroMindLM(nn.Module):
    """Decoder-only language model with legacy checkpoint compatibility."""

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

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[
            tuple[torch.Tensor, torch.Tensor], ...
        ] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Incrementally process tokens and return per-layer key/value caches."""
        if self.config.architecture != "neuromind_v1":
            raise ValueError("KV caching is only supported by neuromind_v1")
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must be a non-empty [batch, sequence] tensor")
        past_length = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        if past_length + input_ids.size(1) > self.config.context_length:
            raise ValueError("cached sequence exceeds the configured context window")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("cache layer count does not match model depth")

        hidden = self.token_embedding(input_ids)
        present_key_values = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            hidden, present = block.forward_with_cache(hidden, past)
            present_key_values.append(present)
        logits = self.lm_head(self.final_norm(hidden))
        return logits, tuple(present_key_values)
