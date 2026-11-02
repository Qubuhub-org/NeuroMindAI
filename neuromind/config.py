"""Validated configuration for the NeuroMindAI decoder model."""

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.context_length < 1:
            raise ValueError("vocab_size must be >= 2 and context_length must be >= 1")
        if self.d_model < 1 or self.n_layers < 1 or self.n_heads < 1 or self.d_ff < 1:
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})
