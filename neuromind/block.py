from torch import nn
from .attention import NeuroMindAttention
from .mlp import NeuroMindMLP
from .normalization import RMSNorm

class NeuroMindBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = NeuroMindAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = NeuroMindMLP(config.hidden_size, config.intermediate_size)

    def forward(self, x):
        x = x + self.attention(self.input_norm(x))
        return x + self.mlp(self.post_attention_norm(x))
